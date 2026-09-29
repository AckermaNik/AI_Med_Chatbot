# MedAi Clinic

MedAi Clinic is a symptom-triage chatbot built as a portfolio project. A user describes symptoms in plain language; the application matches those phrases to canonical symptoms, ranks possible conditions, explains the evidence used for the ranking, suggests a medical specialty, and surfaces deterministic urgency alerts.

The language model is deliberately not the medical ranking engine:

- Gemini extracts symptom phrases, chooses tools, and turns tool results into a reply.
- Python and PostgreSQL perform matching, scoring, specialty routing, and red-flag checks.
- Pydantic validates tool arguments before they reach SQL.
- Emergency and urgent alerts are emitted independently of the model's final prose.
- A medical-assessment disclaimer is enforced on assistant responses.

This is a research and portfolio demonstration, not a medical device, diagnostic service, or substitute for professional medical assessment.

## What is included

- React chat UI with streaming responses and collapsible engine activity.
- FastAPI backend with a Server-Sent Events (SSE) chat endpoint.
- PostgreSQL 16 with pg_trgm and pgvector.
- Manually controlled Gemini tool-calling loop with a ten-step safety limit.
- Three-stage symptom matching: exact canonical/alias match, semantic embedding match with all-MiniLM-L6-v2, then PostgreSQL trigram fallback.
- Deterministic disease ranking using IDF-weighted symptom evidence and MMR diversity.
- Curated specialty routing, body-system fallbacks, precautions, and red flags.
- Repeatable ETL for downloading, normalizing, validating, and loading data.
- Backend tests for scoring, matching, tool schemas, loop dependencies, API events, and curated CSV files.

## Architecture

~~~text
Browser
  │ POST /api/chat
  ▼
FastAPI ── Server-Sent Events ──► React UI
  │
  ├── deterministic raw-text red-flag check
  ├── Gemini manual tool-calling loop
  │     ├── search_symptoms
  │     ├── diagnose
  │     ├── get_disease_info
  │     └── recommend_specialty
  │
  └── PostgreSQL
        ├── pg_trgm symptom matching
        ├── pgvector embedding search
        ├── curated disease/symptom data
        └── scoring, specialty, precaution, and red-flag inputs
~~~

The frontend is served by Nginx in the production-style container. Nginx serves the compiled React application and proxies /api/* to the backend with buffering disabled for SSE.

## Technology stack

| Area | Technology |
| --- | --- |
| UI | React 19, Vite 8, Tailwind CSS 4, Lucide React |
| API | FastAPI, Uvicorn, Pydantic v2 |
| Database | PostgreSQL 16, SQLAlchemy 2 async, Alembic |
| PostgreSQL extensions | pg_trgm, pgvector |
| LLM | Gemini through google-genai |
| Embeddings | FastEmbed / sentence-transformers/all-MiniLM-L6-v2 via ONNX Runtime |
| ETL | pandas, RapidFuzz, Kaggle API |
| Deployment files | Docker Compose, Render blueprint, Vercel configuration |

## Quick start with Docker Compose

### Prerequisites

- Docker Desktop with Docker Compose.
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey).
- Python 3.11 only if you also want to run backend tools or tests locally. Python 3.14 currently lacks compatible wheels for every dependency, especially asyncpg and ONNX Runtime.

### Start the complete application

From the repository root:

~~~powershell
Copy-Item backend\.env.example backend\.env
~~~

Edit backend/.env and set GEMINI_API_KEY. Then start the stack:

~~~powershell
docker compose up --build
~~~

Open the application at [http://localhost:5173](http://localhost:5173).

The backend is also available at [http://localhost:8000](http://localhost:8000):

- Health/readiness check: [http://localhost:8000/api/health](http://localhost:8000/api/health)
- OpenAPI documentation: [http://localhost:8000/docs](http://localhost:8000/docs)

On first startup, PostgreSQL enables both extensions and the backend applies the latest Alembic migrations. The Compose services are:

| Service | Host address | Purpose |
| --- | --- | --- |
| db | localhost:5433 | PostgreSQL 16 + vector support |
| backend | localhost:8000 | FastAPI API |
| frontend | localhost:5173 | Nginx-served React app |

Stop the stack while preserving database data:

~~~powershell
docker compose down
~~~

To deliberately remove the named PostgreSQL volume and all local database data:

~~~powershell
docker compose down -v
~~~

The backend image includes the committed curated data. Downloading original Kaggle files is only needed when rebuilding the database from raw sources or changing the ETL.

## Local development

### Backend

Start PostgreSQL only:

~~~powershell
docker compose up -d db
~~~

Create a Python 3.11 environment, install dependencies, and create the environment file:

~~~powershell
py -3.11 -m venv backend\.venv
backend\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
Copy-Item backend\.env.example backend\.env
~~~

Set GEMINI_API_KEY in backend/.env, then run the environment smoke test:

~~~powershell
backend\.venv\Scripts\python.exe backend\scripts\check_env.py
~~~

Run the API from the repository root:

~~~powershell
backend\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --reload --port 8000
~~~

Docker applies migrations automatically. For a local process, apply them explicitly from backend/:

~~~powershell
Set-Location backend
..\\backend\.venv\Scripts\alembic.exe upgrade head
Set-Location ..
~~~

### Frontend

~~~powershell
Set-Location frontend
npm install
npm run dev
~~~

With the default empty VITE_API_URL, Vite proxies /api to http://127.0.0.1:8000. For a separately deployed API:

~~~text
VITE_API_URL=https://your-api.example.com
~~~

Useful commands:

~~~powershell
npm run lint
npm run build
npm run preview
~~~

## Data pipeline

The database combines two Kaggle sources and committed curation files:

| Source | Role |
| --- | --- |
| itachi9604/disease-symptom-description-dataset | Core descriptions, symptom severity, and precautions |
| dhivyeshrk/diseases-and-symptoms-dataset | Broad disease/symptom coverage |
| niyarrbarman/Symptom2Disease | Evaluation data only; never loaded into the application database |
| data/curated/*.csv | Hand-reviewed specialties, body systems, aliases, and red flags |

The loaded dataset is designed around approximately 793 diseases, 433 canonical symptoms, and 5,687 disease-symptom relationships. Exact totals are determined by the source files and normalization decisions made by the ETL.

Raw Kaggle files are not committed. To rebuild from scratch, configure credentials at ~/.kaggle/kaggle.json and run:

~~~powershell
backend\.venv\Scripts\python.exe backend\etl\download.py
backend\.venv\Scripts\python.exe backend\etl\normalize.py
backend\.venv\Scripts\python.exe backend\etl\load.py
backend\.venv\Scripts\python.exe backend\etl\stats.py
backend\.venv\Scripts\python.exe backend\etl\embed.py
backend\.venv\Scripts\python.exe backend\etl\seed_curated.py
~~~

load.py performs quality gates before inserting data. The pipeline is intended to be idempotent and safe to rerun. See [docs/DATA.md](docs/DATA.md) for provenance and cleaning decisions, and [docs/PLAN.md](docs/PLAN.md) for the data model and scoring design.

## How a chat request works

1. The backend receives the message and browser session ID.
2. Red-flag phrases are checked against the database before Gemini is called.
3. Gemini receives the conversation and four tool declarations.
4. search_symptoms extracts individual symptom phrases and resolves canonical symptom names.
5. diagnose ranks candidate conditions using the deterministic scorer.
6. get_disease_info can provide a condition description and precautions.
7. recommend_specialty uses curated disease mappings, body-system routing, then General Practice fallback.
8. Tool calls and results are streamed to the browser.
9. Gemini writes a plain-text response from the tool results; the disclaimer is enforced.

Independent tool calls run concurrently. Same-batch dependent calls are deferred until their inputs exist, and the loop stops after ten tool rounds.

## API

### GET /api/health

Checks that the API can connect to PostgreSQL.

~~~json
{"status":"ok"}
~~~

### POST /api/chat

Request:

~~~json
{
  "message": "I have a sore throat and fever",
  "session_id": "optional-previous-session-id"
}
~~~

message must contain 1–4,000 characters. The response is an SSE stream with these named events:

| Event | Payload purpose |
| --- | --- |
| tool_call | Tool name and model-generated arguments |
| tool_result | Validated tool output |
| alert | Deterministic urgent or emergency escalation |
| message | Final assistant text |
| error | User-facing model or processing error |
| done | Returns the session ID for later turns |

The browser stores the returned ID in local storage. Conversation history is currently kept in backend process memory; it is not a durable multi-user conversation store.

## Configuration

Copy backend/.env.example to backend/.env.

| Variable | Default | Purpose |
| --- | --- | --- |
| DATABASE_URL | local Docker URL on port 5433 | Async PostgreSQL connection |
| GEMINI_API_KEY | empty | Gemini credential; required for model-backed chat |
| GEMINI_MODEL | gemini-3.5-flash-lite | Gemini model |
| CORS_ORIGINS | local frontend origins | Comma-separated allowed browser origins |
| EMBEDDING_MODEL | sentence-transformers/all-MiniLM-L6-v2 | Symptom embedding model |
| SCORING_MU | 2.0 | Prior damping thinly documented diseases |
| SCORING_MMR_LAMBDA | 0.7 | Relevance/diversity balance |
| SCORING_MMR_POOL | 20 | Candidates considered before MMR |
| SCORING_TOP_K | 5 | Candidates returned |
| SCORING_MIN_SCORE | 0.15 | Minimum useful score |
| SCORING_CONFIDENT_MARGIN | 0.15 | Top-result confidence margin |
| SCORING_MAX_FOLLOWUPS | 3 | Follow-up question limit |
| MATCH_TRIGRAM_THRESHOLD | 0.35 | Trigram acceptance threshold |
| MATCH_EMBEDDING_THRESHOLD | 0.60 | Semantic-match acceptance threshold |

## Testing

Run backend tests:

~~~powershell
backend\.venv\Scripts\python.exe -m pytest backend\tests -q
~~~

The suite covers scoring, IDF weighting, evidence priors, urgency, determinism, MMR, matcher SQL behavior, aliases, Gemini tool schemas, argument validation, loop ordering, reply normalization, SSE payloads, and curated data.

Run the environment smoke test:

~~~powershell
backend\.venv\Scripts\python.exe backend\scripts\check_env.py
~~~

Exercise the deterministic ranking engine without Gemini:

~~~powershell
backend\.venv\Scripts\python.exe backend\scripts\try_diagnose.py itching skin-rash
backend\.venv\Scripts\python.exe backend\scripts\try_diagnose.py headache --mu 0 --lambda 1.0
~~~

Run the terminal client using the same Gemini loop as the API:

~~~powershell
backend\.venv\Scripts\python.exe backend\scripts\chat.py
backend\.venv\Scripts\python.exe backend\scripts\chat.py --quiet
~~~

## Deployment shape

- render.yaml describes the Dockerized FastAPI service on Render and its environment variables.
- frontend/vercel.json and frontend/.env.example support deploying the React build separately with VITE_API_URL pointing at the backend.
- Docker Compose remains the recommended local and self-hosted all-in-one deployment.

For a hosted database, set DATABASE_URL to an async PostgreSQL connection string. The Render example is written for a Supabase session-pooler connection and documents the postgresql+asyncpg:// scheme and password URL encoding.

## Repository layout

~~~text
.
├── backend/
│   ├── app/
│   │   ├── engine/       # matching, deterministic scoring, safety, repository queries
│   │   ├── llm/          # Gemini prompts, schemas, tools, and manual loop
│   │   ├── main.py       # FastAPI app and SSE adapter
│   │   └── models.py     # SQLAlchemy models
│   ├── alembic/          # migrations
│   ├── db/init/          # PostgreSQL extension setup
│   ├── etl/              # download, normalize, load, stats, embeddings, curation
│   ├── scripts/          # smoke test, terminal chat, ranking inspection
│   └── tests/            # backend tests
├── data/
│   ├── curated/          # committed routing and safety data
│   └── raw/              # downloaded source files; not committed
├── docs/
│   ├── DATA.md           # provenance and cleaning decisions
│   └── PLAN.md           # design and scoring details
├── frontend/
│   ├── src/              # React UI
│   ├── Dockerfile        # Vite build + Nginx runtime image
│   └── nginx.conf        # static serving and /api SSE proxy
├── docker-compose.yml
└── render.yaml
~~~

## Known limitations and next steps

- This is a triage and information-retrieval demonstration; it must not be used to diagnose or rule out a medical condition.
- Results are bounded by the source datasets and hand-curated mappings.
- Conversation history is process memory. Restarting the backend or using multiple replicas does not provide durable shared sessions.
- Authentication, rate limiting, user accounts, audit logging, and persistent chat history are not implemented.
- Model-backed turns depend on Gemini availability and quota.
- Deployment manifests are a starting point, not a complete production hardening or compliance setup.

## License and data provenance

This repository contains code and committed curation files for a portfolio project. Upstream Kaggle datasets have their own licenses and terms; review those terms before redistributing downloaded raw data. Raw source files are excluded from version control. Dataset details and cleaning decisions are recorded in [docs/DATA.md](docs/DATA.md).

## Disclaimer

MedAi Clinic is a portfolio demonstration. It is not a medical device and does not provide medical advice, diagnosis, treatment, or emergency services. If symptoms may be life-threatening, seek emergency care immediately. Always consult a qualified healthcare professional for medical assessment.
