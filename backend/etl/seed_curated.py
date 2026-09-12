"""Load the hand-authored data from data/curated/ into Postgres.

    python etl/seed_curated.py

Reads five files, all committed to the repo:

    specialties.csv            ~27 specialties
    disease_specialty.csv      the curated core-41 mapping
    body_system_specialty.csv  fallback routing, 15 rows
    body_systems.csv           body_system per symptom, 433 rows
    red_flags.csv              deterministic escalation rules

Everything is keyed by SLUG, never by database id, so this can be re-run after any
etl/load.py without the mappings breaking when ids are reassigned.

Diseases outside the curated set get General Practice with
mapping_source='fallback', which the UI states honestly rather than implying a
referral nobody verified.
"""

from __future__ import annotations

import asyncio
import csv
import sys
from pathlib import Path

from sqlalchemy import delete, insert, select, text, update

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import CURATED_DIR  # noqa: E402
from app.db import SessionLocal, engine  # noqa: E402
from app.models import (  # noqa: E402
    BodySystemSpecialty,
    Disease,
    DiseaseSpecialty,
    RedFlag,
    Specialty,
    Symptom,
)

FALLBACK_SPECIALTY = "general-practice"


def read(name: str) -> list[dict]:
    path = CURATED_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"missing curated file: {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


async def main() -> int:
    specialties = read("specialties.csv")
    mappings = read("disease_specialty.csv")
    fallbacks = read("body_system_specialty.csv")
    body_systems = read("body_systems.csv")
    red_flags = read("red_flags.csv")

    async with SessionLocal() as session:
        # --- specialties -----------------------------------------------------
        await session.execute(delete(DiseaseSpecialty))
        await session.execute(delete(BodySystemSpecialty))
        await session.execute(delete(Specialty))
        await session.execute(text("ALTER SEQUENCE specialty_id_seq RESTART WITH 1"))
        await session.execute(
            insert(Specialty),
            [
                {"slug": r["slug"], "name": r["name"], "description": r["description"]}
                for r in specialties
            ],
        )
        await session.commit()

        # Specialty is an SQLAlchemy Model—which is just a Python class that acts as a direct blueprint for the specialty table in your Postgres database
        spec_ids = dict(
            (await session.execute(select(Specialty.slug, Specialty.id))).all()
        )
        disease_ids = dict(
            (await session.execute(select(Disease.slug, Disease.id))).all()
        )
        if not disease_ids:
            print("no diseases loaded - run etl/load.py first")
            return 1

        # --- curated disease -> specialty ------------------------------------
        curated_rows, unknown_disease, unknown_specialty = [], [], []
        for r in mappings:
            d, s = r["disease_slug"], r["specialty_slug"]
            if d not in disease_ids:
                unknown_disease.append(d)
                continue
            if s not in spec_ids:
                unknown_specialty.append(s)
                continue
            curated_rows.append(
                {
                    "disease_id": disease_ids[d],
                    "specialty_id": spec_ids[s],
                    "mapping_source": "curated",
                }
            )

        # --- fallback for everything else ------------------------------------
        mapped = {row["disease_id"] for row in curated_rows}
        fallback_rows = [
            {
                "disease_id": did,
                "specialty_id": spec_ids[FALLBACK_SPECIALTY],
                "mapping_source": "fallback",
            }
            for did in disease_ids.values()
            if did not in mapped
        ]

        await session.execute(insert(DiseaseSpecialty), curated_rows + fallback_rows)

        # --- body system -> specialty ----------------------------------------
        await session.execute(
            insert(BodySystemSpecialty),
            [
                {
                    "body_system": r["body_system"],
                    "specialty_id": spec_ids[r["specialty_slug"]],
                }
                for r in fallbacks
                if r["specialty_slug"] in spec_ids
            ],
        )

        # --- symptom.body_system ---------------------------------------------
        known_systems = {r["body_system"] for r in fallbacks}
        tagged, bad_system = 0, set()
        for r in body_systems:
            system = r["body_system"]
            if not system:
                continue
            if system not in known_systems:
                bad_system.add(system)
                continue
            result = await session.execute(
                update(Symptom)
                .where(Symptom.slug == r["symptom_slug"])
                .values(body_system=system)
            )
            tagged += result.rowcount or 0

        # --- red flags --------------------------------------------------------
        await session.execute(delete(RedFlag))
        await session.execute(text("ALTER SEQUENCE red_flag_id_seq RESTART WITH 1"))

        symptom_ids = dict(
            (await session.execute(select(Symptom.slug, Symptom.id))).all()
        )
        flag_rows, skipped_flags = [], []
        for r in red_flags:
            slugs = [s.strip() for s in r["require_all"].split("|") if s.strip()]
            missing = [s for s in slugs if s not in symptom_ids]
            if missing:
                skipped_flags.append(f"{r['name']} (missing: {', '.join(missing)})")
                continue
            flag_rows.append(
                {
                    "name": r["name"],
                    "level": r["level"],
                    "require_all": [symptom_ids[s] for s in slugs],
                    "message": r["message"],
                }
            )
        if flag_rows:
            await session.execute(insert(RedFlag), flag_rows)

        await session.commit()

        # --- report -----------------------------------------------------------
        print(f"  specialties           {len(specialties)}")
        print(f"  curated mappings      {len(curated_rows)}")
        print(f"  fallback mappings     {len(fallback_rows)}")
        print(f"  body_system routes    {len(fallbacks)}")
        print(f"  symptoms tagged       {tagged} / {len(body_systems)}")
        print(f"  red flags             {len(flag_rows)}")

        coverage = len(curated_rows) / len(disease_ids) * 100
        print(f"\n  specialty coverage    {coverage:.1f}% curated, "
              f"{100 - coverage:.1f}% fallback")

        for label, items in [
            ("unknown disease slug", unknown_disease),
            ("unknown specialty slug", unknown_specialty),
            ("unknown body_system", sorted(bad_system)),
            ("skipped red flag", skipped_flags),
        ]:
            if items:
                print(f"\n  [warn] {label}: {', '.join(items)}")

    await engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
