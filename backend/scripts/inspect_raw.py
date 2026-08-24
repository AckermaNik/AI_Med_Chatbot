"""Look at what the Kaggle files actually contain, before trusting any of it.

    python scripts/inspect_raw.py

Read-only. Prints shapes, dtypes, sample values and the specific quirks the ETL
will have to handle (stray whitespace, duplicate rows, non-binary one-hot values).
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import RAW_DIR  # noqa: E402

pd.set_option("display.width", 120)


def rule(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def inspect_core() -> None:
    rule("DATASET A (core) - itachi9604")

    ds = pd.read_csv(RAW_DIR / "core" / "dataset.csv")
    print(f"dataset.csv           shape={ds.shape}")
    print(f"  columns: {list(ds.columns)}")
    print(f"  unique diseases: {ds['Disease'].nunique()}")
    print(f"  exact duplicate rows: {ds.duplicated().sum()}")

    sym_cols = [c for c in ds.columns if c.startswith("Symptom_")]
    values = pd.unique(ds[sym_cols].values.ravel())
    values = [v for v in values if isinstance(v, str)]
    leading_ws = [v for v in values if v != v.strip()]
    print(f"  distinct symptom values: {len(values)}")
    print(f"  values with stray whitespace: {len(leading_ws)}")
    if leading_ws:
        print(f"    e.g. {[repr(v) for v in leading_ws[:5]]}")
    underscore_ws = [v for v in values if "_ " in v or " _" in v]
    if underscore_ws:
        print(f"  values with ' _' or '_ ': {[repr(v) for v in underscore_ws[:5]]}")

    sev = pd.read_csv(RAW_DIR / "core" / "Symptom-severity.csv")
    print(f"\nSymptom-severity.csv  shape={sev.shape}")
    print(f"  columns: {list(sev.columns)}")
    print(f"  weight range: {sev['weight'].min()}..{sev['weight'].max()}")
    dupes = sev[sev.duplicated("Symptom", keep=False)]
    print(f"  duplicated Symptom rows: {len(dupes)}")
    if len(dupes):
        print(dupes.to_string(index=False))

    # Which symptoms in dataset.csv have no severity row?
    norm = lambda s: s.strip().lower().replace("_", " ").replace("  ", " ")  # noqa: E731
    sev_names = {norm(s) for s in sev["Symptom"]}
    ds_names = {norm(v) for v in values}
    missing = sorted(ds_names - sev_names)
    print(f"\n  symptoms in dataset.csv with NO severity row: {len(missing)}")
    for m in missing[:10]:
        print(f"    {m!r}")

    for name in ("symptom_Description.csv", "symptom_precaution.csv"):
        df = pd.read_csv(RAW_DIR / "core" / name)
        print(f"\n{name:22} shape={df.shape}  columns={list(df.columns)}")


def inspect_broad() -> None:
    rule("DATASET B (broad) - dhivyeshrk")

    path = RAW_DIR / "broad" / "Final_Augmented_dataset_Diseases_and_Symptoms.csv"
    print(f"file size: {path.stat().st_size / 1e6:.1f} MB")

    df = pd.read_csv(path)
    print(f"shape={df.shape}")

    label = df.columns[0]
    print(f"  label column: {label!r}")
    print(f"  unique diseases: {df[label].nunique()}")
    print(f"  first 5 diseases in file order: {df[label].head().tolist()}")
    print(f"  5 random diseases: {df[label].drop_duplicates().sample(5, random_state=0).tolist()}")

    sym_cols = list(df.columns[1:])
    print(f"  symptom columns: {len(sym_cols)}")
    print(f"  e.g. {sym_cols[:6]}")

    block = df[sym_cols]
    uniq = pd.unique(block.values.ravel())
    print(f"  distinct cell values across one-hot block: {sorted(uniq)[:10]}")

    all_zero = (block.sum(axis=1) == 0).sum()
    print(f"  rows with zero symptoms set: {all_zero}")
    print(f"  mean symptoms per row: {block.sum(axis=1).mean():.2f}")

    per_disease = df.groupby(label).size()
    print(f"  rows per disease: min={per_disease.min()} "
          f"median={int(per_disease.median())} max={per_disease.max()}")


def inspect_eval() -> None:
    rule("EVAL SET - Symptom2Disease (never loaded into the DB)")

    df = pd.read_csv(RAW_DIR / "eval" / "Symptom2Disease.csv")
    print(f"shape={df.shape}  columns={list(df.columns)}")
    label = "label" if "label" in df.columns else df.columns[-2]
    print(f"  unique labels: {df[label].nunique()}")
    print(f"  labels: {sorted(df[label].unique())}")
    text_col = "text" if "text" in df.columns else df.columns[-1]
    print("\n  sample complaints:")
    for t in df[text_col].head(3):
        print(f"    - {t[:110]}...")


if __name__ == "__main__":
    inspect_core()
    inspect_broad()
    inspect_eval()
