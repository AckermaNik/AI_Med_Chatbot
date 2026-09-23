"""Search phrases typed by a user and try to find the closest matching symptoms per phrase
    in the database, even if the user misspelled it or used an alternative name.
    (canonical symptoms).

Each stage runs only if the previous one missed, so the common case costs a single indexed query.

    1. exact / alias      known vocabulary            indexed lookup
    2. embedding cosine   paraphrase                  pgvector
    3. trigram fuzzy      typos, morphology           pg_trgm
    4. miss               nothing cleared threshold   ask the user

Stages 2 and 3 fail in different places, which is why both exist. Measured against
the loaded data:

'tummy hurts'   trigram 0.333 -> 'hurts to breath'  (below threshold)
                    embedding 0.682 -> 'stomach pain'   (right)
'stomache ake'  embedding 0.471 -> below threshold
                    trigram 0.368 -> below threshold
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from etl.normalize import normalize


@dataclass(frozen=True)
class Match:
    phrase: str
    symptom_id: int
    slug: str
    name: str
    confidence: float
    stage: str  # 'exact' | 'alias' | 'trigram' | 'embedding'


@dataclass(frozen=True)
class Miss:
    phrase: str
    near: list[str]  # best guesses, for the model to ask about


@lru_cache(maxsize=1)
def _embedder():
    """Loaded once per process — construction costs ~1s, inference ~5ms."""
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=get_settings().embedding_model)


def embed(phrase: str) -> list[float]:
    return list(_embedder().embed([phrase]))[0].tolist()


async def _exact(session: AsyncSession, norm: str) -> tuple[int, str, str, str] | None:
    row = (
        await session.execute(
            text(
                """
                SELECT s.id, s.slug, s.canonical_name,
                       CASE WHEN s.canonical_name = :q THEN 'exact' ELSE 'alias' END AS stage
                FROM symptom s
                LEFT JOIN symptom_alias a ON a.symptom_id = s.id
                WHERE s.canonical_name = :q OR a.alias = :q
                LIMIT 1
                """
            ),
            {"q": norm},
        )
    ).first()
    return (row.id, row.slug, row.canonical_name, row.stage) if row else None


# If a column is listed in the SELECT clause, 
# it MUST either be in the GROUP BY clause (safe from the blender)
# OR wrapped in an aggregator (instructions for the blender like MAX()).

# GROUP BY acts as a blender that crushes multiple rows into one,
# SQL literally refuses to run unless you use an aggregator 
# to tell it how to process the data that are getting crushed

# GROUP BY doesn't need an aggregator when you only SELECT the exact things you GROUP BY.

# If you are not grouping data, you do not need GROUP BY at all
async def _trigram(
    session: AsyncSession, norm: str, threshold: float
) -> tuple[int, str, str, float] | None:
    row = (
        await session.execute(
            text(
                """
                SELECT s.id, s.slug, s.canonical_name,
                       GREATEST(
                           similarity(s.canonical_name, :q),
                           COALESCE(MAX(similarity(a.alias, :q)), 0)
                       ) AS score
                FROM symptom s
                LEFT JOIN symptom_alias a ON a.symptom_id = s.id
                -- The % operator is what uses the GIN index. Calling similarity()
                -- in the WHERE clause instead would scan every row to compute each score and find the best one.
                WHERE s.canonical_name % :q OR a.alias % :q
                GROUP BY s.id, s.slug, s.canonical_name
                ORDER BY score DESC
                LIMIT 1
                """
            ),
            {"q": norm},
        )
    ).first()
    if row and row.score >= threshold:
        return (row.id, row.slug, row.canonical_name, float(row.score))
    return None

# Cosine Similarity = 1 - Cosine Distance (Cosine Distance = 0 they are identical)
# CAST( data AS target_type ) -> "Take this text string and parse it into your custom vector data type."
# Sorting by the raw <=> operator instead of the cosine score allows Postgres to use a special Vector Index (like HNSW or IVFFlat)

async def _semantic(
    session: AsyncSession, phrase: str, threshold: float, limit: int = 3
) -> list[tuple[int, str, str, float]]:
    vector = str(embed(phrase))
    rows = (
        await session.execute(
            text(
                """
                SELECT id, slug, canonical_name,
                       1 - (embedding <=> CAST(:v AS vector)) AS cosine
                FROM symptom
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> CAST(:v AS vector)
                LIMIT :k
                """
            ),
            {"v": vector, "k": limit},
        )
    ).all()
    return [(r.id, r.slug, r.canonical_name, float(r.cosine)) for r in rows]


async def match_phrase(session: AsyncSession, phrase: str) -> Match | Miss:
    settings = get_settings()
    norm = normalize(phrase)
    if not norm:
        return Miss(phrase=phrase, near=[])

    # Await the result, assign it to 'hit', and check if it's truthy in one line
    if hit := await _exact(session, norm):
        sid, slug, name, stage = hit
        return Match(phrase, sid, slug, name, 1.0, stage)

    # Try semantic similarity before lexical fuzziness. The two scores are not
    # directly comparable, so this is an intentional priority order: embeddings
    # handle paraphrases, while trigram is the fallback for spelling/morphology.
    semantic = await _semantic(session, phrase, settings.match.embedding_threshold)
    if semantic and semantic[0][3] >= settings.match.embedding_threshold:
        sid, slug, name, cosine = semantic[0]
        return Match(phrase, sid, slug, name, round(cosine, 3), "embedding")

    if hit := await _trigram(session, norm, settings.match.trigram_threshold):
        sid, slug, name, score = hit
        return Match(phrase, sid, slug, name, round(score, 3), "trigram")

    # Nothing cleared threshold. Hand back the misses so the model can ask
    # rather than silently dropping what the user said.
    return Miss(phrase=phrase, near=[name for _, _, name, _ in semantic])


async def match_phrases(
    session: AsyncSession, phrases: list[str]
) -> tuple[list[Match], list[Miss]]:
    matches: list[Match] = []
    misses: list[Miss] = []
    seen: set[int] = set()

    for phrase in phrases:
        result = await match_phrase(session, phrase)
        if isinstance(result, Miss):
            misses.append(result)
        elif result.symptom_id not in seen:
            seen.add(result.symptom_id)
            matches.append(result)

    return matches, misses
