"""Propose a body_system tag for every symptom, for human review.

    python etl/curate_body_systems.py

Writes data/curated/body_systems.csv with one row per symptom. Rows the keyword
rules cannot decide are written with an empty body_system and must be filled in by
hand — the file is the source of truth, not this script.

Re-running preserves everything already in the file, so hand edits are never lost.
Only genuinely new symptoms get a fresh proposal.

body_system drives ONLY the specialty fallback, used when no disease matches
confidently (docs/PLAN.md section 6). It never affects ranking.
"""

from __future__ import annotations

import asyncio
import csv
import sys
from pathlib import Path

from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import CURATED_DIR  # noqa: E402
from app.db import SessionLocal, engine  # noqa: E402

OUT = CURATED_DIR / "body_systems.csv"

# Ordered: the first matching group wins, so put the specific before the general.
# 'skin rash' must hit skin before 'rash' can be caught by anything broader.
RULES: list[tuple[str, tuple[str, ...]]] = [
    ("ophthalmic", ("eye", "vision", "visual", "pupil", "blurr", "eyelid", "retina",
                    "conjunctiv", "sight", "blind")),
    ("ent", ("ear", "hearing", "nose", "nasal", "sinus", "throat", "tonsil", "voice",
             "hoarse", "smell", "sneez", "snor", "tinnitus", "vertigo", "swallow",
             "larynx", "deaf")),
    ("skin", ("skin", "rash", "itch", "acne", "blister", "pimple", "mole", "nail",
              "hair", "scar", "bruis", "pigment", "patches", "eruption", "peeling",
              "dandruff", "wart", "boil", "ulcer on", "callus", "wrinkle", "hive",
              "lesion", "abscess", "scaliness", "dermat")),
    ("cardiac", ("heart", "chest pain", "palpitation", "cardiac", "angina",
                 "blood pressure", "pulse", "circulat")),
    ("respiratory", ("breath", "cough", "lung", "wheez", "sputum", "phlegm",
                     "respirat", "asthma", "chok", "hyperventil", "apnea")),
    ("gastrointestinal", ("stomach", "abdom", "belly", "nausea", "vomit", "diarrhea",
                          "constipat", "bowel", "stool", "digest", "appetite",
                          "hunger", "swallowing", "gas", "bloat", "heartburn",
                          "acidity", "anus", "rectal", "rectum", "flatulence",
                          "indigestion", "colic", "hemorrhoid")),
    ("hepatic", ("liver", "jaundice", "yellow", "bile", "hepat", "gallbladder")),
    ("urinary", ("urin", "bladder", "kidney", "micturition", "polyuria", "renal")),
    ("reproductive", ("menstrua", "period", "vagin", "penis", "testic", "pregnan",
                      "breast", "genital", "sexual", "libido", "erectile", "sperm",
                      "uter", "ovar", "prostate", "nipple", "scrotum", "groin")),
    ("neurological", ("headache", "migraine", "seizure", "dizz", "faint", "numb",
                      "tingl", "paraly", "tremor", "coma", "conscious", "memory",
                      "coordination", "balance", "speech", "slurr", "nerve",
                      "spasm", "convuls", "confus", "concentrat", "neck stiff")),
    ("psychological", ("anxiety", "depress", "mood", "panic", "stress", "irritab",
                       "hallucinat", "psychotic", "insomnia", "sleep", "suicid",
                       "restless", "nervous", "obsessiv", "delusion", "addict")),
    ("endocrine", ("thyroid", "sugar", "diabet", "hormone", "sweat", "obesity",
                   "weight gain", "weight loss", "thirst", "goitre", "glucose",
                   "metabol")),
    ("musculoskeletal", ("joint", "muscle", "bone", "back pain", "knee", "elbow",
                         "shoulder", "hip", "ankle", "wrist", "spine", "arthrit",
                         "stiff", "cramp", "limp", "posture", "sprain", "fracture",
                         "swelling", "weakness", "arm", "leg", "foot", "toe",
                         "finger", "hand", "neck", "jaw")),
    ("systemic", ("fever", "fatigue", "chills", "malaise", "lethargy", "weight",
                  "sweating", "dehydrat", "lymph", "bleeding", "anemia", "pain",
                  "swollen", "infection", "toxic", "shock", "allerg")),
]


# Symptoms no keyword rule can place, assigned by hand. Keyed on canonical name.
# Where a symptom straddles two systems the safer routing wins: 'chest tightness'
# goes to cardiac rather than respiratory because missing a cardiac cause is the
# more costly error.
HAND_ASSIGNED: dict[str, str] = {
    # neurological
    "abnormal involuntary movements": "neurological",
    "altered sensorium": "neurological",
    "difficulty speaking": "neurological",
    "loss of sensation": "neurological",
    "paresthesia": "neurological",
    "problems with movement": "neurological",
    "unsteadiness": "neurological",
    # psychological
    "abusing alcohol": "psychological",
    "antisocial behavior": "psychological",
    "drug abuse": "psychological",
    "emotional symptoms": "psychological",
    "excessive anger": "psychological",
    "hostile behavior": "psychological",
    "hysterical behavior": "psychological",
    "low self esteem": "psychological",
    "nightmares": "psychological",
    "obsessions and compulsions": "psychological",
    "temper problems": "psychological",
    # skin
    "blackheads": "skin",
    "dry or flaky scalp": "skin",
    "flushing": "skin",
    "red spots over body": "skin",
    "scurring": "skin",
    "silver like dusting": "skin",
    # cardiac (incl. vascular)
    "chest tightness": "cardiac",
    "fluid overload": "cardiac",
    "fluid retention": "cardiac",
    "peripheral edema": "cardiac",
    "prominent veins on calf": "cardiac",
    # respiratory
    "congestion in chest": "respiratory",
    "hemoptysis": "respiratory",
    "smoking problems": "respiratory",
    # ent (incl. oral mucosa)
    "congestion": "ent",
    "coryza": "ent",
    "dry lips": "ent",
    "lip sore": "ent",
    "mouth dryness": "ent",
    "mouth ulcer": "ent",
    "ulcers on tongue": "ent",
    "spinning movements": "ent",
    # dental
    "toothache": "dental",
    # ophthalmic
    "lacrimation": "ophthalmic",
    # gastrointestinal
    "difficulty eating": "gastrointestinal",
    "infant feeding problem": "gastrointestinal",
    "infant spitting up": "gastrointestinal",
    "melena": "gastrointestinal",
    "regurgitation": "gastrointestinal",
    "regurgitation 1": "gastrointestinal",
    # endocrine
    "excessive growth": "endocrine",
    "hot flashes": "endocrine",
    "lack of growth": "endocrine",
    # urinary
    "bedwetting": "urinary",
    "hesitancy": "urinary",
    # reproductive
    "extra marital contacts": "reproductive",
    "impotence": "reproductive",
    "infertility": "reproductive",
    "loss of sex drive": "reproductive",
    "pelvic pressure": "reproductive",
    "penile discharge": "reproductive",
    "premature ejaculation": "reproductive",
    "vulvar irritation": "reproductive",
    "vulvar sore": "reproductive",
    # systemic - constitutional, history items and anything genuinely non-localising
    "ache all over": "systemic",
    "back mass or lump": "systemic",
    "family history": "systemic",
    "feeling cold": "systemic",
    "feeling hot": "systemic",
    "feeling ill": "systemic",
    "flu like syndrome": "systemic",
    "history of alcohol consumption": "systemic",
    "receiving blood transfusion": "systemic",
    "receiving unsterile injections": "systemic",
    "symptoms of infants": "systemic",
    "symptoms of the face": "systemic",
}


def propose(name: str) -> str:
    if name in HAND_ASSIGNED:
        return HAND_ASSIGNED[name]
    lowered = name.lower()
    for system, keywords in RULES:
        if any(k in lowered for k in keywords):
            return system
    return ""


async def main() -> int:
    existing: dict[str, str] = {}
    if OUT.exists():
        with OUT.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("body_system"):
                    existing[row["symptom_slug"]] = row["body_system"]

    async with SessionLocal() as session:
        rows = (
            await session.execute(
                text("SELECT slug, canonical_name FROM symptom ORDER BY canonical_name")
            )
        ).all()

    if not rows:
        print("no symptoms loaded - run etl/load.py first")
        return 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    unassigned: list[str] = []
    counts: dict[str, int] = {}

    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["symptom_slug", "symptom_name", "body_system"])
        for slug, name in rows:
            system = existing.get(slug) or propose(name)
            if not system:
                unassigned.append(name)
            else:
                counts[system] = counts.get(system, 0) + 1
            writer.writerow([slug, name, system])

    print(f"  symptoms          {len(rows)}")
    print(f"  preserved by hand {len(existing)}")
    print(f"  assigned          {len(rows) - len(unassigned)}")
    print(f"  UNASSIGNED        {len(unassigned)}")

    print("\n  distribution:")
    for system, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"    {system:<20} {n:>4}")

    if unassigned:
        print(f"\n  needs hand assignment in {OUT.name}:")
        for name in unassigned:
            print(f"    {name}")

    await engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
