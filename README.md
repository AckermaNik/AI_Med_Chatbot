# MedAi Clinic

A symptom-triage chatbot. Describe your symptoms in plain language; it returns ranked
candidate conditions with an explanation of *why* each ranked where it did, an urgency
signal, and the medical specialty to consult.

**The LLM never diagnoses.** It extracts symptoms from prose, calls tools, and writes
prose. Every ranking, score and specialty comes from Python and SQL over curated data,
and a server-side guardrail rejects any response naming a condition no tool returned.

Full design: [docs/PLAN.md](docs/PLAN.md).

---

## Stack

| Layer | Choice |
|---|---|
| Frontend | React 19, Tailwind 4, Vite |
| Backend | FastAPI, Pydantic v2, SQLAlchemy 2.0 (async) |
| Database | PostgreSQL 16 + `pg_trgm` + `pgvector` |
| LLM | Gemini `gemini-3.5-flash-lite` via `google-genai`, manual tool-calling loop |
| Embeddings | `fastembed` / all-MiniLM-L6-v2 (ONNX, no PyTorch) |
| Data | Two Kaggle disease-symptom datasets, ~773 diseases |

---

## Setup

### Prerequisites

- Python 3.11 (3.14 lacks wheels for `asyncpg` / `onnxruntime`)
- Docker Desktop, running
- A free Gemini API key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

### 1. Database

```bash
docker compose up -d
```

Starts PostgreSQL 16 on host port **5433** (not 5432, to avoid clashing with any local
install). The image is `pgvector/pgvector:pg16` — plain `postgres:16` does not ship the
`vector` extension. `backend/db/init/01_extensions.sql` enables `pg_trgm` and `vector`
on first start.

### 2. Backend environment

```bash
py -3.11 -m venv backend/.venv
backend/.venv/Scripts/python.exe -m pip install -r backend/requirements.txt
```

Then copy `backend/.env.example` to `backend/.env` and fill in `GEMINI_API_KEY`.

### 3. Verify

```bash
backend/.venv/Scripts/python.exe backend/scripts/check_env.py
```

Checks imports, connects to the database, and confirms both extensions respond to real
queries. Should end with `M0 complete`.

### 4. Load the data

Needs Kaggle API credentials at `~/.kaggle/kaggle.json`
([Kaggle → Settings → API → Create New Token](https://www.kaggle.com/settings)).

```bash
backend/.venv/Scripts/python.exe backend/etl/download.py
```

Apply the schema (run from `backend/`):

```bash
alembic upgrade head
```

Then, in order — each step is idempotent and safe to re-run:

```bash
backend/.venv/Scripts/python.exe backend/etl/normalize.py
```

```bash
backend/.venv/Scripts/python.exe backend/etl/load.py
```

```bash
backend/.venv/Scripts/python.exe backend/etl/stats.py
```

```bash
backend/.venv/Scripts/python.exe backend/etl/embed.py
```

```bash
backend/.venv/Scripts/python.exe backend/etl/seed_curated.py
```

`normalize.py` reconciles the two datasets' vocabularies and writes decision records to
`data/curated/`. `load.py` runs quality gates before writing anything — a failure
aborts with nothing inserted. See [docs/DATA.md](docs/DATA.md) for every cleaning
decision and its justification.

---

## Progress

- [x] **M0** — environment, Docker, extensions verified
- [x] **M1** — schema, Alembic, ETL, symptom reconciliation, embeddings
      — 793 diseases, 433 symptoms, 5687 edges, IDF + 384-dim embeddings loaded
- [x] **M2** — scoring engine + tests (no LLM) — 23 unit tests, ranks 793 diseases in ~60ms
- [x] **M3** — curated specialties, body-system tags, red flags
      — 27 specialties, 41 curated mappings, 433 symptoms tagged, 13 red flags
- [ ] **M4** — tools + Gemini loop (CLI)
- [ ] **M5** — FastAPI + SSE + session persistence
- [ ] **M6** — frontend wiring, tool trace, banners
- [ ] **M7** — grounding guardrail + eval harness
- [ ] **M8** — full docker-compose, deploy

---

## Trying the ranking engine

No LLM involved — this is the scoring engine straight against the database:

```bash
backend/.venv/Scripts/python.exe backend/scripts/try_diagnose.py itching skin-rash
```

Hyperparameters can be overridden per run to see their effect:

```bash
backend/.venv/Scripts/python.exe backend/scripts/try_diagnose.py headache --mu 0 --lambda 1.0
```

Run the unit tests (no database needed — the scoring functions are pure):

```bash
backend/.venv/Scripts/python.exe -m pytest tests/ -q
```

---

## Disclaimer

This is a portfolio project demonstrating LLM tool-calling and information retrieval.
It is **not** a medical device and gives no medical advice. Always consult a qualified
healthcare professional.
