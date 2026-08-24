-- Runs automatically on first container start, before Alembic ever connects.
-- pg_trgm  -> fuzzy symptom matching (stage 2 of the matcher)
-- vector   -> symptom embeddings (stage 3 of the matcher)
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS vector;
