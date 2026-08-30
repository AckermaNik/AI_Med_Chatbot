"""Build the canonical tables and write them to Postgres.

    python etl/load.py

Idempotent: wipes and rebuilds the data tables on every run. Conversations are
never touched. Hand-curated specialty mappings are keyed by slug and re-applied
afterwards by the M3 seed script, so wiping disease rows here is safe.

Quality gates run BEFORE the write. A failure aborts with a non-zero exit and
nothing is inserted.
"""

from __future__ import annotations

import asyncio
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sqlalchemy import delete, insert, select, text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import RAW_DIR  # noqa: E402
from app.db import SessionLocal, engine  # noqa: E402
from app.models import (  # noqa: E402
    Disease,
    DiseaseSymptom,
    Precaution,
    Symptom,
    SymptomAlias,
)
from etl.normalize import (  # noqa: E402
    classify,
    disease_vocabulary,
    normalize,
    reconcile,
    slugify,
    symptom_vocabulary,
)

CORE = RAW_DIR / "core" # Database A 
BROAD = RAW_DIR / "broad" / "Final_Augmented_dataset_Diseases_and_Symptoms.csv" # Database B


class QualityGateError(RuntimeError):
    """Raised when the assembled data fails a sanity check. Nothing gets written."""


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #

"""
    Pours both datasets into it, one after the other.
"""

@dataclass
class Assembled:
    symptoms: dict[str, int | None] = field(default_factory=dict)  # name -> severity
    aliases: dict[str, str] = field(default_factory=dict)  # alias -> canonical
    diseases: dict[str, dict] = field(default_factory=dict)  # name -> {source, description}
    edges: dict[tuple[str, str], float] = field(default_factory=dict) # (disease, symptom) -> support
    precautions: list[tuple[str, int, str]] = field(default_factory=list) # (disease , column number at the OG dataset, description)


def build_alias_map(kind: str) -> dict[str, str]:
    """alias -> canonical, for every pair in the database the rules said to merge.""" 
    vocab = reconcile(symptom_vocabulary() if kind == "symptom" else disease_vocabulary()) # makes a proposal for pairs
    out: dict[str, str] = {}
    for p in vocab.proposals:
        verdict, _ = classify(p.canonical, p.alias) # decide if they are identical in meaning
        if verdict == "merge":
            out[p.alias] = p.canonical
    return out


def load_severity() -> dict[str, int]:
    """Symptom -> weight. fluid_overload appears twice (6 and 4); keep the higher.

    Under-reporting urgency is the worse error, so max() rather than first-wins.
    See docs/DATA.md A3.
    """
    sev = pd.read_csv(CORE / "Symptom-severity.csv")
    out: dict[str, int] = {}
    for raw, weight in zip(sev["Symptom"], sev["weight"]):
        name = normalize(raw)
        out[name] = max(out.get(name, 0), int(weight))
    return out


def assemble_core(a: Assembled, alias_sym: dict[str, str], alias_dis: dict[str, str]) -> None:
    """Dataset A: 41 diseases, severity weights, descriptions, precautions."""
    resolve_s = lambda n: alias_sym.get(n, n)  # noqa: E731
    resolve_d = lambda n: alias_dis.get(n, n)  # noqa: E731

    ds = pd.read_csv(CORE / "dataset.csv").drop_duplicates()
    sym_cols = [c for c in ds.columns if c.startswith("Symptom_")]


    hits: dict[tuple[str, str], int] = defaultdict(int)
    totals: dict[str, int] = defaultdict(int)

    for _, row in ds.iterrows():
        disease = resolve_d(normalize(row["Disease"]))
        totals[disease] += 1
        seen = {
            resolve_s(normalize(row[c]))
            for c in sym_cols
            if isinstance(row[c], str) and row[c].strip()
        }
        for symptom in seen:
            hits[(disease, symptom)] += 1

    for (disease, symptom), n in hits.items():
        a.edges[(disease, symptom)] = n / totals[disease] # ("fungal_infection", "itching") = support

    for disease in totals:
        a.diseases.setdefault(disease, {"source": set(), "description": None})
        a.diseases[disease]["source"].add("itachi9604")

    desc = pd.read_csv(CORE / "symptom_Description.csv")
    for raw, text_ in zip(desc["Disease"], desc["Description"]):
        name = resolve_d(normalize(raw))
        if name in a.diseases: # WE ONLY CARE ABOUT THE DISEASE WE ALREADY HAVE
            a.diseases[name]["description"] = str(text_).strip()

    prec = pd.read_csv(CORE / "symptom_precaution.csv")
    prec_cols = [c for c in prec.columns if c.startswith("Precaution_")]
    for _, row in prec.iterrows():
        name = resolve_d(normalize(row["Disease"]))
        if name not in a.diseases: # we only care about the diseases we already have
            continue
        for i, col in enumerate(prec_cols, start=1):
            value = row[col]
            if isinstance(value, str) and value.strip():
                a.precautions.append((name, i, value.strip()))

    return totals


def assemble_broad(
    a: Assembled,
    alias_sym: dict[str, str],
    alias_dis: dict[str, str],
    core_totals: dict[str, int],
) -> None:
    """Dataset B: 773 diseases as a 377-column one-hot matrix.

    Read with int8 dtypes: at int64 the frame is ~750MB, at int8 it is ~95MB.
    """
    header = pd.read_csv(BROAD, nrows=0)
    label_col = header.columns[0]
    sym_cols = list(header.columns[1:])
    dtypes = dict.fromkeys(sym_cols, "int8") # save a massive amount of computer memory (RAM)

    df = pd.read_csv(BROAD, dtype=dtypes)

    counts = df.groupby(label_col)[sym_cols].sum() # groups by disease and adds up vertically all the 1s in every symptom column.
    sizes = df.groupby(label_col).size() # groups the rows by disease and counts how many rows exist for each.
    support = counts.div(sizes, axis=0) # massive 2D matrix

    resolve_s = lambda n: alias_sym.get(n, n)  # noqa: E731
    resolve_d = lambda n: alias_dis.get(n, n)  # noqa: E731

    long = support.stack() # smash that 2D matrix into a 1D dictionary and throw away all the zeroes
    long = long[long > 0]

    for (raw_disease, raw_symptom), value in long.items():
        disease = resolve_d(normalize(raw_disease))
        symptom = resolve_s(normalize(raw_symptom))
        key = (disease, symptom)
        n_b = int(sizes[raw_disease])

        if key in a.edges:
            # Present in both datasets: weighted mean by evidence count, so A's
            # ~7 unique rows do not outvote B's hundreds.
            n_a = core_totals.get(disease, 0)
            a.edges[key] = (n_a * a.edges[key] + n_b * float(value)) / (n_a + n_b)
        else:
            a.edges[key] = float(value)

    for raw_disease in sizes.index:
        disease = resolve_d(normalize(raw_disease))
        a.diseases.setdefault(disease, {"source": set(), "description": None})
        a.diseases[disease]["source"].add("dhivyeshrk")


def assemble() -> Assembled:
    a = Assembled()

    alias_sym = build_alias_map("symptom")
    alias_dis = build_alias_map("disease")
    a.aliases = alias_sym

    core_totals = assemble_core(a, alias_sym, alias_dis)
    assemble_broad(a, alias_sym, alias_dis, core_totals)

    severity = load_severity()
    for _, symptom in a.edges:
        if symptom not in a.symptoms:
            a.symptoms[symptom] = severity.get(symptom)

    return a


# --------------------------------------------------------------------------- #
# quality gates
# --------------------------------------------------------------------------- #


def check(a: Assembled) -> None:
    problems: list[str] = []

    if not 400 <= len(a.symptoms) <= 600:
        problems.append(f"symptom count {len(a.symptoms)} outside [400, 600]")
    if not 700 <= len(a.diseases) <= 900:
        problems.append(f"disease count {len(a.diseases)} outside [700, 900]")

    bad_support = [k for k, v in a.edges.items() if not 0 < v <= 1]
    if bad_support:
        problems.append(f"{len(bad_support)} edges with support outside (0, 1]")

    orphan_s = {s for _, s in a.edges} - set(a.symptoms)
    orphan_d = {d for d, _ in a.edges} - set(a.diseases)
    if orphan_s:
        problems.append(f"{len(orphan_s)} edges reference unknown symptoms")
    if orphan_d:
        problems.append(f"{len(orphan_d)} edges reference unknown diseases")

    no_symptoms = set(a.diseases) - {d for d, _ in a.edges}
    if no_symptoms:
        problems.append(f"{len(no_symptoms)} diseases have no symptoms: {sorted(no_symptoms)[:5]}")

    slugs = defaultdict(list)
    for name in a.diseases:
        slugs[slugify(name)].append(name)
    collisions = {k: v for k, v in slugs.items() if len(v) > 1}
    if collisions:
        problems.append(f"{len(collisions)} slug collisions: {list(collisions.items())[:3]}")

    if problems:
        raise QualityGateError("; ".join(problems))


def summarize(a: Assembled) -> None:
    with_sev = sum(1 for v in a.symptoms.values() if v is not None)
    per_disease = defaultdict(int)
    for d, _ in a.edges:
        per_disease[d] += 1
    thin = sum(1 for n in per_disease.values() if n < 3)

    print("\n--- assembled ---")
    print(f"  symptoms            {len(a.symptoms):>6}")
    print(f"    with severity     {with_sev:>6}  ({len(a.symptoms) - with_sev} unknown)")
    print(f"  aliases             {len(a.aliases):>6}")
    print(f"  diseases            {len(a.diseases):>6}")
    print(f"  disease_symptom     {len(a.edges):>6}")
    print(f"  precautions         {len(a.precautions):>6}")
    print(f"  symptoms/disease    {len(a.edges) / len(a.diseases):>6.1f} mean")
    print(f"  low-evidence (<3)   {thin:>6}")


# --------------------------------------------------------------------------- #
# writes to the database
# --------------------------------------------------------------------------- #


async def write(a: Assembled) -> None:
    async with SessionLocal() as session:
        # Order matters: children before parents. disease_specialty is wiped too
        # (it cascades from disease)
        await session.execute(delete(Precaution))
        await session.execute(delete(DiseaseSymptom))
        await session.execute(delete(SymptomAlias))
        await session.execute(delete(Symptom))
        await session.execute(delete(Disease))
        await session.execute(text("ALTER SEQUENCE symptom_id_seq RESTART WITH 1"))
        await session.execute(text("ALTER SEQUENCE disease_id_seq RESTART WITH 1"))
        await session.commit()

        # how many diff symptoms each disease has
        per_disease: dict[str, int] = defaultdict(int)
        for d, _ in a.edges:
            per_disease[d] += 1

        await session.execute(
            insert(Symptom),
            [
                {
                    "canonical_name": name,
                    "slug": slugify(name),
                    "severity_weight": severity,
                    "body_system": None,  # assigned by hand at M3
                    "idf": 0.0,  # computed by etl/stats.py
                }
                for name, severity in sorted(a.symptoms.items())
            ],
        )
        await session.execute(
            insert(Disease),
            [
                {
                    "name": name,
                    "slug": slugify(name),
                    "description": meta["description"],
                    "source": (
                        "both" if len(meta["source"]) > 1 else next(iter(meta["source"]))
                    ),
                    "low_evidence": per_disease[name] < 3,
                }
                for name, meta in sorted(a.diseases.items())
            ],
        )
        await session.commit()

        sym_ids = dict(
            (await session.execute(select(Symptom.canonical_name, Symptom.id))).all()
        )
        dis_ids = dict((await session.execute(select(Disease.name, Disease.id))).all())

        if a.aliases:
            await session.execute(
                insert(SymptomAlias),
                [
                    {
                        "symptom_id": sym_ids[canonical],
                        "alias": alias,
                        "source": "dataset_b",
                    }
                    for alias, canonical in a.aliases.items()
                    if canonical in sym_ids
                ],
            )

        rows = [
            {
                "disease_id": dis_ids[d],
                "symptom_id": sym_ids[s],
                "support": round(v, 6),
            }
            for (d, s), v in a.edges.items()
        ]
        for i in range(0, len(rows), 10_000):
            await session.execute(insert(DiseaseSymptom), rows[i : i + 10_000])

        if a.precautions:
            await session.execute(
                insert(Precaution),
                [
                    {"disease_id": dis_ids[d], "ordinal": o, "text": t}
                    for d, o, t in a.precautions
                    if d in dis_ids
                ],
            )

        await session.commit()

    await engine.dispose()


async def main() -> int:
    print("assembling from raw files (dataset B is 190MB, this takes a moment)...")
    a = assemble()
    summarize(a)

    try:
        check(a)
    except QualityGateError as exc:
        print(f"\nQUALITY GATE FAILED: {exc}")
        print("Nothing was written.")
        return 1
    print("\n  quality gates passed")

    print("\nwriting to postgres...")
    await write(a)
    print("done. next: etl/stats.py (idf) and etl/embed.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
