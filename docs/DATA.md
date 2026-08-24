# Data provenance and cleaning decisions

Every judgement call made about the source data, with the measurements that motivated
it. Reproduce any figure here with:

```bash
backend/.venv/Scripts/python.exe backend/scripts/inspect_raw.py
```

Raw files live in `data/raw/` and are **gitignored** — they are not redistributed.
`data/curated/` is hand-authored and **is** committed.

---

## Sources

| Dir | Dataset | Role | License |
|---|---|---|---|
| `raw/core/` | [itachi9604/disease-symptom-description-dataset](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset) | curated core | GPL-3.0 (per the [author's mirror](https://github.com/itachi9604/Disease-Symptom-dataset)) |
| `raw/broad/` | [dhivyeshrk/diseases-and-symptoms-dataset](https://www.kaggle.com/datasets/dhivyeshrk/diseases-and-symptoms-dataset) | breadth | **TODO: confirm before publishing** |
| `raw/eval/` | [niyarrbarman/Symptom2Disease](https://www.kaggle.com/datasets/niyarrbarman/symptom2disease) | eval only, never loaded | **TODO: confirm before publishing** |

---

## Dataset A — measured

```
dataset.csv               4920 x 18      41 diseases
Symptom-severity.csv       133 x 2       weights 1..7
symptom_Description.csv     41 x 2
symptom_precaution.csv      41 x 5
```

### A1. 94% of rows are duplicates

`4616 / 4920` rows are exact duplicates, leaving **304 unique rows** — roughly
**7 distinct symptom combinations per disease**, padded ~16x.

**Decision:** deduplicate before computing anything. `support` is calculated over the
304 unique rows, not the 4920.

**Consequence:** support values from dataset A are coarse (multiples of ~1/7). This is
acceptable — support is a weighting term, not a probability we report — but treating
the raw 4920 rows as independent observations would have been quietly wrong, inflating
apparent confidence 16-fold.

### A2. Whitespace corruption in 130 of 131 symptom values

Values arrive as `' skin_rash'`, `' nodal_skin_eruptions'` — a leading space on nearly
every one. Two have broken *internal* spacing:

```
' dischromic _patches'      ->  dischromic patches
' spotting_ urination'      ->  spotting urination
```

**Decision:** normalize with `strip -> lowercase -> '_' to space -> collapse whitespace`.
The internal cases fall out of that automatically once `_` becomes a space.

### A3. `fluid_overload` appears twice with different weights

```
fluid_overload  6
fluid_overload  4
```

**Decision:** keep the **higher** weight (6). Severity drives urgency escalation, and
under-reporting urgency is the worse error. Recorded here because it is an arbitrary
call, not a derived one.

### A4. One symptom has no severity row

`foul smell of urine` appears in `dataset.csv` but not `Symptom-severity.csv`.

**Decision:** load it with `severity_weight = NULL`. NULL means *unknown*, never zero —
urgency is computed only over symptoms whose severity we actually have.

---

## Dataset B — measured

```
Final_Augmented_dataset_Diseases_and_Symptoms.csv    190.8 MB
shape = (246945, 378)        label column: 'diseases'
773 unique diseases          377 one-hot symptom columns
```

### B1. The file is grouped by disease, not shuffled

The first ~1200 rows are all `panic disorder`, which makes the Kaggle preview look like
a single-disease file. The column header's "773 unique values" is the accurate figure.
Confirmed by sampling: `sarcoidosis`, `lactose intolerance`, `floaters`,
`metabolic disorder`, `schizophrenia`.

### B2. One-hot block is clean

- Cell values are strictly `{0, 1}` — no NaN, no stray strings.
- Zero rows have an all-zero symptom vector.
- Mean symptoms per row: **5.33**.

No cleaning required. The quality gates still assert all three, so a future re-download
that breaks them fails loudly.

### B3. Severe class imbalance

```
rows per disease:  min = 1     median = 168     max = 1219
```

At least one disease is backed by a **single row**. This is exactly the failure case the
μ prior exists for (see [PLAN.md](PLAN.md) §5): a disease with one recorded symptom
scores a perfect 1.00 when that symptom is reported.

**Decision:** flag any disease with fewer than 3 distinct symptoms as
`disease.low_evidence = true`, and damp all diseases via μ in scoring. Do **not** drop
them — a rare condition being rare is not a reason to make it undiagnosable.

---

## Eval set — measured

```
Symptom2Disease.csv     1200 x 3     columns: ['Unnamed: 0', 'label', 'text']
24 unique labels
```

- `Unnamed: 0` is a pandas index artefact — dropped on read.
- **Label casing is inconsistent**: `Acne`, `Arthritis`, `Chicken pox` alongside
  `allergy`, `diabetes`, `drug reaction`.
- Some labels will not match our disease names directly and need a hand-written
  crosswalk, e.g. `gastroesophageal reflux disease` -> `GERD`,
  `Dimorphic Hemorrhoids` -> `Dimorphic hemmorhoids(piles)`.

**Decision:** the crosswalk lives in `data/curated/eval_label_map.csv`, hand-authored at
M7. Labels with no confident equivalent are excluded from the accuracy metric and the
exclusion count is reported alongside it — silently dropping them would inflate the
score.

**This file is never loaded into the database.** It is held out so that accuracy numbers
mean something.

---

## Vocabulary reconciliation

Run with `python etl/normalize.py`. Reads no database, writes no database — it
produces the canonical vocabularies and a decision record in `data/curated/`.

```
symptoms   A: 133   B: 377   exact overlap:  24   canonical: 486
diseases   A:  41   B: 773   exact overlap:  18   canonical: 796
```

**The two datasets barely overlap** — 24 of 486 symptoms and 18 of 796 diseases match
exactly. That is not a defect: the merge is mostly a *union*, which means less
reconciliation risk, and it is precisely why both are needed. A carries severity
weights, descriptions and precautions that B lacks entirely; B carries 755 diseases A
has never heard of.

Note 796 diseases, not 773 — dataset A contributes 23 that B does not have.

### Why fuzzy matching alone is not safe here

`token_set_ratio` scores **100** whenever one name's tokens are a subset of the other's.
That is the single most common pattern in medical naming, so the raw matcher proposed:

```
100  'hypertension'  <- 'malignant hypertension'      distinct emergency condition
100  'hepatitis a'   <- 'hepatitis due to a toxin'    different aetiology entirely
100  'arthritis'     <- 'septic arthritis'            distinct condition
```

Worse, plain string similarity has no concept of negation:

```
 89  'increased appetite' <- 'decreased appetite'     clinically OPPOSITE
 90  'excessive hunger'   <- 'excessive anger'        different symptom entirely
```

**Embeddings do not fix this — they make it worse.** Distributional models place
antonyms very close together, because antonyms occur in near-identical contexts.
Embeddings are used at query time (matcher stage 3), where the output is a ranked list
a human sees; they are deliberately not used for the irreversible merge decision.

### The five rules

Applied in order by `classify()` in `etl/normalize.py`:

| # | Rule | Verdict | Example |
|---|---|---|---|
| 0 | Recorded human decision | as decided | `hepatitis a` / `viral hepatitis` |
| 1a | Polarity swap on a shared axis | reject | `increased` vs `decreased appetite` |
| 1b | Head noun differs | reject | `excessive hunger` vs `excessive anger` |
| 1c | Same tokens (singular-stemmed), different order | merge | `swelling joints` / `joint swelling` |
| 2a | Leading modifier | reject | `septic arthritis`, `lower abdominal pain` |
| 2b | Trailing prepositional phrase | reject | `fungal infection of the hair` |
| 2c | Trailing bare qualifier | merge | `typhoid` / `typhoid fever` |

Rule 2a is the load-bearing one. Any word qualifying a term **from the left** narrows it
to a subtype or a body site, so the pair denotes different things. This correctly keeps
dataset B's six site-specific cramp columns (`arm`, `leg`, `knee`, `neck`, `back`,
`elbow`) as distinct symptoms rather than collapsing them into a single `cramps`, which
would have destroyed six findings and inflated its support.

**Result: 61 fuzzy proposals reduced to 0 requiring review** — 6 merges, 52 rejects,
3 human decisions.

### Recorded human decisions

| Pair | Decision | Reason |
|---|---|---|
| `swelling joints` / `joint swelling` | **merge** | Same symptom; plural-vs-singular defeated the reorder rule, now handled by stemming |
| `hepatitis a` / `viral hepatitis` | **keep separate** | Hepatitis A is one specific virus; "viral hepatitis" is the parent class over A/B/C/D/E. A specific diagnosis must not absorb its own category |
| `skin peeling` / `skin dryness peeling scaliness or roughness` | **keep separate** | B's column bundles four distinct findings; collapsing to "peeling" silently discards dryness, scaliness and roughness |

The latter two live in `MANUAL_VERDICTS` in `etl/normalize.py`, each with its reasoning
attached, so the calls are auditable rather than folklore.

### Corrections applied during normalization

| Raw | Corrected | Why |
|---|---|---|
| `osteoarthristis` | `osteoarthritis` | dataset A misspelling |
| `diarrhoea` | `diarrhea` | British → American, matches dataset B |
| `chicken pox` | `chickenpox` | dataset A spacing |
| `hemmorhoids` | `hemorrhoids` | dataset A misspelling |
| `dimorphic hemorrhoids piles` | `hemorrhoids` | dataset A mangles the name; handled as a rename so rule 2a stays strict |
| `dischromic patches` | `discoloured patches` | obscure term, no match in B |

---

## Loaded state (M1 complete)

```
diseases          793      symptoms          433
disease_symptom  5687      aliases             4
precautions       162      low_evidence       46
idf   min 1.771  /  mean 4.969  /  max 6.676
```

Symptom count is 433, not the 486 canonical: 53 vocabulary entries appear in no
disease at all (severity-table rows with no edges). A symptom attached to nothing can
never be diagnostic, so they are dropped rather than loaded as dead rows.

Disease sources: 752 from B only, 21 in both, 20 from A only.

IDF behaves as intended — `headache` (123 diseases) scores 1.86, `wrinkles on skin`
(1 disease) scores 6.68.

### Matcher threshold calibration — OPEN

The two matcher stages demonstrably cover different failures:

| Probe | Trigram | Embedding |
|---|---|---|
| `tummy hurts` | 0.333 → *hurts to breath* (wrong) | **0.682 → stomach pain** |
| `stomache ake` | **0.368 → stomach pain** | 0.471 → stomach pain |
| `cant catch my breath` | no match | **0.660 → hurts to breath** |

But the configured thresholds (`MATCH_TRIGRAM_THRESHOLD=0.45`,
`MATCH_EMBEDDING_THRESHOLD=0.60`) reject **both** correct answers for `stomache ake`.
A plain typo currently resolves to nothing.

Do not tune these on anecdotes. They are swept against the held-out eval set at M7
along with `mu` and `mmr_lambda`.

### The mu prior is under-powered — OPEN

`SCORING_MU=2.0` damps thinly-documented diseases in the right direction but not
nearly hard enough. Ranking on the single symptom `headache` (idf 1.86, present in 123
diseases):

| Disease | mu = 0 | mu = 2 |
|---|---|---|
| high blood pressure *(low evidence)* | 0.692 | 0.562 |
| open wound of the head *(low evidence)* | 0.606 | 0.513 |
| rocky mountain spotted fever *(low evidence)* | 0.594 | 0.506 |
| intracranial abscess | 0.541 | 0.472 |
| ependymoma | 0.468 | 0.421 |

The mechanism works — the top entry loses 19% while the bottom loses 10%, so thin
diseases are penalised harder, and the unit tests confirm the ratio exceeds 3x. But the
constant is far too small to demote them below well-documented conditions. The top three
are all `low_evidence` either way.

**Second, related gap:** a single common symptom should probably not produce any
confident candidate at all. Every result above clears `min_score = 0.15` comfortably.
Both constants need fitting against the eval set at M7.

---

## Open items

- [ ] Confirm licenses for datasets B and the eval set before publishing the repo.
- [x] Cross-dataset disease-name reconciliation — done, see above.
- [x] Symptom vocabulary reconciliation — done, see above.
- [ ] Quality-gate bounds need updating: expect **486** symptoms and **796** diseases,
      not the 400–500 / 773 originally assumed.
