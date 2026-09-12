"""Vocabulary reconciliation between the two datasets.

    python etl/normalize.py

Reads nothing from the database and writes nothing to it. Produces:

  * a canonical symptom vocabulary and a canonical disease vocabulary
  * fuzzy-match *proposals* for human review, written to data/curated/

Merging is a three-stage process (docs/PLAN.md section 3):

  1. exact match on the normalized string  -> merged automatically
  2. fuzzy match above THRESHOLD           -> written as a proposal, NOT merged
  3. no match                              -> new canonical entry

Nothing in stage 2 takes effect until a human sets accept=yes in the proposals file.
The importer reads the reviewed file; unreviewed rows are ignored.
"""

from __future__ import annotations

import csv
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import CURATED_DIR, RAW_DIR  # noqa: E402

# Above this, two names are proposed as the same thing. Deliberately high: a false
# merge silently corrupts the knowledge base, while a missed one just leaves two
# separate symptoms, which is recoverable.
FUZZY_THRESHOLD = 88

_PUNCT = re.compile(r"[^a-z0-9 ]")
_WS = re.compile(r"\s+")

# --- rule 1: polarity -------------------------------------------------------
# Terms on the same axis but opposite ends. Two names differing ONLY by a swap
# across an axis are opposites, not synonyms. Embeddings cannot catch these:
# antonyms occur in near-identical contexts, so distributional models place them
# very close together — often closer than true synonyms.
POLARITY: dict[str, str] = {}
for _axis, _high, _low in [
    ("quantity", "increased elevated excessive high hyper raised more",
                 "decreased reduced diminished low hypo lowered less"),
    ("speed", "rapid fast tachy quick", "slow brady sluggish"),
    ("temperature", "hot burning warm", "cold cool chills"),
    ("size", "swollen enlarged distended", "shrunken atrophied wasted"),
]:
    for _t in _high.split():
        POLARITY[_t] = f"{_axis}:high"
    for _t in _low.split():
        POLARITY[_t] = f"{_axis}:low"

# --- rule 3: known corrections ---------------------------------------------
# Typos and spelling variants in the source data. Applied during normalization so
# both datasets land on one canonical form. Each entry is justified in docs/DATA.md.
CORRECTIONS: dict[str, str] = {
    "osteoarthristis": "osteoarthritis",   # dataset A misspelling
    "diarrhoea": "diarrhea",               # British -> American, matches dataset B
    "chicken pox": "chickenpox",           # dataset A spacing
    "swelled lymph nodes": "swollen lymph nodes",
    "hemmorhoids": "hemorrhoids",
    "dischromic patches": "discoloured patches",
    # Dataset A mangles this one; it denotes plain haemorrhoids. Handled as a
    # rename rather than a fuzzy merge so the leading-modifier rule stays strict.
    "dimorphic hemorrhoids piles": "hemorrhoids",
}

# --- rule 2: modifier position ---------------------------------------------
# A trailing bare noun is usually a semantic no-op ('typhoid' -> 'typhoid fever').
# A trailing PREPOSITIONAL phrase names a site or cause and marks a distinct
# condition ('fungal infection OF THE hair', 'hepatitis DUE TO a toxin').
PREPOSITIONS = {"of", "due", "from", "with", "in", "on", "by", "after", "during"}

# --- human decisions --------------------------------------------------------
# Pairs no rule can settle, resolved by a person. Keyed on (canonical, alias).
# Each carries its reasoning so the call is auditable rather than folklore.
MANUAL_VERDICTS: dict[tuple[str, str], tuple[str, str]] = {
    ("hepatitis a", "viral hepatitis"): (
        "reject",
        "hepatitis A is one specific virus; 'viral hepatitis' is the parent class " # implicit string concatenation
        "covering A/B/C/D/E - a specific diagnosis must not absorb its category",
    ),
    ("skin peeling", "skin dryness peeling scaliness or roughness"): (
        "reject",
        "B's column bundles four distinct findings; collapsing it into 'peeling' " # implicit string concatenation
        "would silently discard dryness, scaliness and roughness",
    ),
}


def normalize(raw: str) -> str:
    """lowercase -> underscores to spaces -> strip punctuation -> collapse whitespace.

    Handles dataset A's corruption automatically:
        ' dischromic _patches'  -> 'dischromic patches'
        ' spotting_ urination'  -> 'spotting urination'
    """
    s = str(raw).strip().lower().replace("_", " ")
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()

    # Rule 3: known typos and spelling variants. Per-word FIRST, so that a
    # whole-string rule keyed on the corrected spelling still matches:
    #   'dimorphic hemmorhoids piles' -> 'dimorphic hemorrhoids piles' -> 'hemorrhoids'
    s = " ".join(CORRECTIONS.get(w, w) for w in s.split())
    return CORRECTIONS.get(s, s)


def slugify(name: str) -> str:
    return _WS.sub("-", normalize(name))


# --------------------------------------------------------------------------- #
# classification rules
# --------------------------------------------------------------------------- #


def classify(a: str, b: str) -> tuple[str, str]:
    """Decide whether two names denote the same thing.

    Returns (verdict, reason) where verdict is 'merge', 'reject' or 'review'.
    Only 'review' reaches a human.
    """
    if (a, b) in MANUAL_VERDICTS:
        verdict, reason = MANUAL_VERDICTS[(a, b)]
        return verdict, f"human decision: {reason}"

    wa, wb = a.split(), b.split()
    sa, sb = set(wa), set(wb)

    # --- rule 1a: polarity swap --------------------------------------------
    # Same length, differing in exactly one position, and the two differing
    # words sit on opposite ends of the same axis.
    if len(wa) == len(wb): # it checks if the phrases have the exact same number of words
        diffs = [(x, y) for x, y in zip(wa, wb) if x != y] # It pairs the words up side-by-side and only keeps the pairs that do not match
        if len(diffs) == 1: # checks if there is exactly one difference between the two phrases.
            x, y = diffs[0]
            px, py = POLARITY.get(x), POLARITY.get(y)
            if px and py and px != py and px.split(":")[0] == py.split(":")[0]:
                return "reject", f"polarity opposite ({x} vs {y})"

    # --- rule 1b: head noun must match -------------------------------------
    # 'excessive hunger' vs 'excessive anger': same shape, different subject.
    if wa[-1] != wb[-1] and wa[-1] not in sb and wb[-1] not in sa:
        return "reject", f"different head noun ({wa[-1]} vs {wb[-1]})"

    # --- rule 1c: same words, different order ------------------------------
    # 'swelling joints' vs 'joint swelling'. Word order carries no meaning here.
    # Turn prular words to singular ('joints' vs 'joint')
    stem = lambda ws: {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in ws}  # noqa: E731
    if sa == sb or stem(sa) == stem(sb):
        return "merge", "same tokens, different order"

    # --- rule 2: modifier position -----------------------------------------
    shorter, longer = (wa, wb) if len(wa) < len(wb) else (wb, wa)
    if set(shorter) < set(longer): # Are all the words in the shorter list perfectly contained inside the longer list
        head = shorter[0] # It grabs the very first word of the short phrase
        first_shared = longer.index(head) if head in longer else 0 # What position is the word head in longer phrase
        extra_before = longer[:first_shared] # creates a list of any words that appeared before that anchor point
        extra_after = [w for w in longer[first_shared:] if w not in set(shorter)] # slices off anything that came before the shared anchor word and is not in the shorter phrase

        # body site: 'lower abdominal pain', 'septic arthritis', 'arm cramps'.
        if extra_before:
            return (
                "reject",
                f"leading modifier ({' '.join(extra_before)}) - distinct condition",
            )

        if PREPOSITIONS & set(extra_after):
            return (
                "reject",
                f"prepositional qualifier ({' '.join(extra_after)}) - names a site or cause",
            )

        # example: muscle pain symptom
        if extra_after:
            return "merge", f"trailing qualifier ({' '.join(extra_after)}) - semantic no-op"

    return "review", "no rule applied"


# --------------------------------------------------------------------------- #
# vocabulary extraction
# --------------------------------------------------------------------------- #


@dataclass
class Proposal:
    """A candidate merge, with enough context for a human to judge it.

    `relation` is the important field. SUBSET means one name's words are wholly
    contained in the other ('hypertension' inside 'malignant hypertension'), which
    token_set and ratio always scores 100 regardless of whether the pair is a synonym
    or a distinct subtype. Those two cases are indistinguishable by string metrics
    and require medical judgement, so they are surfaced separately rather than
    buried among genuine spelling variants.
    """

    canonical: str
    alias: str
    token_set: float #  Fuzzy match score (0-100) that ignores word order ("blood pressure" vs "pressure blood" = 100)
    ratio: float # Strict fuzzy match score (0-100) based on exact character order and typos
    relation: str  # 'subset' if all words in the shorter string exist in the longer one, otherwise 'variant'
    verdict: str = "review"  # 'merge' | 'reject' | 'review'
    reason: str = ""


@dataclass
class Vocabulary:
    """A reconciled name space: canonical names plus the aliases folded into them.
    
                        WE ONLY CARE ABOUT THE DISTINCT NAMES
    """

    kind: str # symptom or disease
    from_a: set[str] = field(default_factory=set) # Set containing every unique name found in Dataset A (the core Kaggle dataset)
    from_b: set[str] = field(default_factory=set) # Set containing every unique name found in Dataset B (the broad Kaggle dataset)

    # Names that were spelled exactly the same in both datasets (no rules needed to merge these)
    exact_overlap: set[str] = field(default_factory=set)
    
    # A list holding all the fuzzy `Proposal` objects that need to be run through the classification rules
    proposals: list[Proposal] = field(default_factory=list)

    @property
    def a_only(self) -> set[str]:
        return self.from_a - self.from_b

    @property
    def b_only(self) -> set[str]:
        return self.from_b - self.from_a

    @property
    def canonical(self) -> set[str]:
        return self.from_a | self.from_b


def symptom_vocabulary() -> Vocabulary:
    vocab = Vocabulary(kind="symptom")

    ds = pd.read_csv(RAW_DIR / "core" / "dataset.csv")
    sym_cols = [c for c in ds.columns if c.startswith("Symptom_")]
    for value in pd.unique(ds[sym_cols].values.ravel()):
        if isinstance(value, str) and value.strip():
            vocab.from_a.add(normalize(value))

    sev = pd.read_csv(RAW_DIR / "core" / "Symptom-severity.csv")
    for value in sev["Symptom"]:
        vocab.from_a.add(normalize(value))

    broad = pd.read_csv(
        RAW_DIR / "broad" / "Final_Augmented_dataset_Diseases_and_Symptoms.csv",
        nrows=0,  # header only - the 190MB body is irrelevant here
    )
    for column in broad.columns[1:]:
        vocab.from_b.add(normalize(column))

    return vocab


def disease_vocabulary() -> Vocabulary:
    vocab = Vocabulary(kind="disease")

    ds = pd.read_csv(RAW_DIR / "core" / "dataset.csv")
    vocab.from_a = {normalize(d) for d in ds["Disease"].unique()}

    broad = pd.read_csv(
        RAW_DIR / "broad" / "Final_Augmented_dataset_Diseases_and_Symptoms.csv",
        usecols=[0],
    )
    vocab.from_b = {normalize(d) for d in broad.iloc[:, 0].unique()}

    return vocab


# --------------------------------------------------------------------------- #
# reconciliation -> figure out which symptoms in Dataset B are actually just typos 
# or alternate spellings of the symptoms in Dataset A.
# --------------------------------------------------------------------------- #


def reconcile(vocab: Vocabulary) -> Vocabulary:
    """Stage 1 (exact) then stage 2 (fuzzy proposals) over the non-overlapping names."""
    vocab.exact_overlap = vocab.from_a & vocab.from_b # Set Intersection

    candidates = sorted(vocab.b_only)
    targets = sorted(vocab.a_only) # these are the official names we want to keep
    if not candidates or not targets:
        return vocab

    # For every candidate in Dataset B, search Dataset A for the closest match. 
    for candidate in candidates:
        hit = process.extractOne(
            candidate, targets, scorer=fuzz.token_set_ratio, score_cutoff=FUZZY_THRESHOLD
        )
        if not hit:
            continue
        match, token_set, _ = hit # match -> the actual string from the targets list (Dataset A)

        a_words, b_words = set(match.split()), set(candidate.split())
        relation = "subset" if (a_words <= b_words or b_words <= a_words) else "variant"
        
        verdict, reason = classify(match, candidate) # decide if the two words MEAN the same thing

        vocab.proposals.append(
            Proposal(
                canonical=match,
                alias=candidate,
                token_set=token_set,
                ratio=fuzz.ratio(match, candidate),
                relation=relation,
                verdict=verdict,
                reason=reason,
            )
        )

    # Anything still needing a human first, then merges, then rejects.
    order = {"review": 0, "merge": 1, "reject": 2}
    vocab.proposals.sort(key=lambda p: (order[p.verdict], -p.ratio)) # sorts the entire list of proposals so that everything flagged for "review" goes to the very top
    return vocab


def write_proposals(vocab: Vocabulary, path: Path) -> None:
    """Write review file. Never overwrites decisions already made."""
    existing: dict[tuple[str, str], str] = {}
    if path.exists():
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                existing[(row["canonical"], row["alias"])] = (
                    row.get("override") or row.get("accept") or ""
                )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["canonical", "alias", "verdict", "reason", "ratio", "override"]
        )
        for p in vocab.proposals:
            writer.writerow(
                [
                    p.canonical,
                    p.alias,
                    p.verdict,
                    p.reason,
                    f"{p.ratio:.0f}",
                    existing.get((p.canonical, p.alias), ""),
                ]
            )

    kept = sum(1 for v in existing.values() if v)
    if kept:
        print(f"  (preserved {kept} existing decision(s))")


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #


def report(vocab: Vocabulary) -> None:
    print(f"\n{'=' * 70}\n{vocab.kind.upper()} VOCABULARY\n{'=' * 70}")
    print(f"  dataset A            {len(vocab.from_a):>5}")
    print(f"  dataset B            {len(vocab.from_b):>5}")
    print(f"  exact overlap        {len(vocab.exact_overlap):>5}  (auto-merged)")
    print(f"  A only               {len(vocab.a_only):>5}")
    print(f"  B only               {len(vocab.b_only):>5}")
    print(f"  canonical total      {len(vocab.canonical):>5}")
    print(f"  fuzzy proposals      {len(vocab.proposals):>5}  (need review)")

    if vocab.exact_overlap:
        sample = sorted(vocab.exact_overlap)[:6]
        print(f"\n  exact-match examples: {sample}")

    by = lambda v: [p for p in vocab.proposals if p.verdict == v]  # noqa: E731
    print(f"    auto-merge           {len(by('merge')):>5}")
    print(f"    auto-reject          {len(by('reject')):>5}")
    print(f"    NEEDS REVIEW         {len(by('review')):>5}")

    for verdict, label in [
        ("review", "NEEDS REVIEW - no rule applied"),
        ("merge", "AUTO-MERGE"),
        ("reject", "AUTO-REJECT (showing first 8)"),
    ]:
        rows = by(verdict)
        if not rows:
            continue
        print(f"\n  {label}:")
        for p in rows[:8]:
            print(f"    {p.canonical!r:<34} <- {p.alias!r:<38} {p.reason}")


def main() -> int:
    symptoms = reconcile(symptom_vocabulary())
    report(symptoms)
    write_proposals(symptoms, CURATED_DIR / "symptom_alias_proposals.csv")

    diseases = reconcile(disease_vocabulary())
    report(diseases)
    write_proposals(diseases, CURATED_DIR / "disease_alias_proposals.csv")

    print(f"\n{'=' * 70}")
    print("Proposals written to data/curated/. Nothing is merged until you set")
    print("accept=yes on a row. Re-running preserves decisions already made.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
