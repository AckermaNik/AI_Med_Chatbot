"""Fetch the Kaggle source datasets into data/raw/.

    python etl/download.py

Needs Kaggle API credentials at ~/.kaggle/kaggle.json (Kaggle -> Settings -> API ->
Create New Token). If they are missing the script prints manual download instructions
and exits non-zero rather than failing obscurely inside the Kaggle client.

Raw data is gitignored and never redistributed; only the hand-authored files in
data/curated/ are committed.
"""

import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import RAW_DIR  # noqa: E402


@dataclass(frozen=True)
class Source:
    slug: str
    subdir: str
    purpose: str
    expect: tuple[str, ...] # make sure the downloaded zip file actually contains the files we specify here.


SOURCES = (
    Source(
        slug="itachi9604/disease-symptom-description-dataset",
        subdir="core",
        purpose="curated core: 41 diseases, severity weights, descriptions, precautions",
        expect=(
            "dataset.csv",
            "Symptom-severity.csv",
            "symptom_Description.csv",
            "symptom_precaution.csv",
        ),
    ),
    Source(
        slug="dhivyeshrk/diseases-and-symptoms-dataset",
        subdir="broad",
        purpose="breadth: 773 diseases, 377 one-hot symptom columns",
        expect=(),
    ),
    Source(
        slug="niyarrbarman/symptom2disease",
        subdir="eval",
        purpose="EVAL ONLY - natural-language complaints, never loaded into the DB",
        expect=("Symptom2Disease.csv",),
    ),
)


def have_credentials() -> bool:
    return (Path.home() / ".kaggle" / "kaggle.json").exists()


def manual_instructions() -> None:
    print("No Kaggle credentials found at ~/.kaggle/kaggle.json\n")
    print("Either create a token (Kaggle -> Settings -> API -> Create New Token),")
    print("or download these by hand and unzip into the paths shown:\n")
    for src in SOURCES:
        print(f"  https://www.kaggle.com/datasets/{src.slug}")
        print(f"    -> {RAW_DIR / src.subdir}")
        print(f"       ({src.purpose})\n")


def unzip_in_place(target: Path) -> None:
    for archive in target.glob("*.zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target)
        archive.unlink()


def main() -> int:
    if not have_credentials():
        manual_instructions()
        return 1

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    for src in SOURCES:
        target = RAW_DIR / src.subdir
        target.mkdir(parents=True, exist_ok=True)
        print(f"downloading {src.slug} -> {target}")
        api.dataset_download_files(src.slug, path=str(target), unzip=True, quiet=False)
        unzip_in_place(target)

        missing = [name for name in src.expect if not (target / name).exists()]
        if missing:
            print(f"  [warn] expected files not found: {', '.join(missing)}")
        found = sorted(p.name for p in target.glob("*.csv"))
        print(f"  got: {', '.join(found) if found else '(no csv files!)'}")

    print("\nDownload complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())