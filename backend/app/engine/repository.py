"""Database access for the ranking engine.

    Support(d, s) = (Number of times Disease appears with Symptom s) / (Total number of times Disease appears in general)

Kept deliberately separate from scoring.py: everything here touches Postgres,
everything there is pure. That split is what lets the maths be unit-tested with
no database running.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import ScoringConfig, get_settings
from app.engine.scoring import DiseaseProfile, Ranking, SymptomInfo, rank


async def load_symptoms(session: AsyncSession) -> dict[int, SymptomInfo]:
    """All 433 symptoms. Small enough to load whole — and we need the full set
    anyway, since D(d) sums over every symptom a candidate disease has."""
    rows = await session.execute(
        text(
            """
            SELECT id, slug, canonical_name, idf, severity_weight, body_system
            FROM symptom
            """
        )
    )
    return {
        r.id: SymptomInfo(
            id=r.id,
            slug=r.slug,
            name=r.canonical_name,
            idf=float(r.idf),
            severity=r.severity_weight,
            body_system=r.body_system,
        )
        for r in rows
    }


async def load_candidates(
    session: AsyncSession, symptom_ids: set[int]
) -> list[DiseaseProfile]:

    """Builds the relationship between diseases and symptoms"""
    if not symptom_ids:
        return []

    rows = await session.execute(
        text(
            """
            SELECT ds.disease_id, ds.symptom_id, ds.support,
                   d.slug, d.name, d.low_evidence
            FROM disease_symptom ds
            JOIN disease d ON d.id = ds.disease_id
            WHERE ds.disease_id IN (
                SELECT DISTINCT disease_id
                FROM disease_symptom
                WHERE symptom_id = ANY(:ids)
            )
            """
        ),
        {"ids": list(symptom_ids)},
    )

    build: dict[int, dict] = {}
    for r in rows:
        entry = build.setdefault(
            r.disease_id,
            {"slug": r.slug, "name": r.name, "low": r.low_evidence, "support": {}},
        )
        entry["support"][r.symptom_id] = float(r.support)

    return [
        DiseaseProfile(
            id=did,
            slug=e["slug"],
            name=e["name"],
            support=e["support"],
            low_evidence=e["low"],
        )
        for did, e in build.items()
    ]


async def resolve_slugs(session: AsyncSession, slugs: list[str]) -> dict[str, int]:
    """slug -> symptom id. Unknown slugs are simply absent from the result."""
    if not slugs:
        return {}
    rows = await session.execute(
        text("SELECT slug, id FROM symptom WHERE slug = ANY(:slugs)"),
        {"slugs": slugs},
    )
    return {r.slug: r.id for r in rows}


@dataclass(frozen=True)
class SpecialtyAdvice:
    name: str
    description: str
    mapping_source: str  # 'curated' | 'body-system' | 'fallback'

    @property
    def is_verified(self) -> bool:
        """False means no human checked this referral — say so in the UI."""
        return self.mapping_source == "curated"


async def recommend_specialty(
    session: AsyncSession,
    disease_slug: str | None,
    symptom_ids: set[int] | None = None,
) -> list[SpecialtyAdvice]:
    """Three-tier resolution (docs/PLAN.md section 6).

    1. curated disease -> specialty
    2. body-system fallback, from the modal system across reported symptoms
    3. General Practice, stated honestly as unverified
    """
    if disease_slug:
        rows = (
            await session.execute(
                text(
                    """
                    SELECT sp.name, sp.description, dsp.mapping_source
                    FROM disease d
                    JOIN disease_specialty dsp ON dsp.disease_id = d.id
                    JOIN specialty sp ON sp.id = dsp.specialty_id
                    WHERE d.slug = :slug AND dsp.mapping_source = 'curated'
                    ORDER BY sp.name
                    """
                ),
                {"slug": disease_slug},
            )
        ).all()
        if rows:
            return [
                SpecialtyAdvice(r.name, r.description, r.mapping_source) for r in rows
            ]

    if symptom_ids:
        row = (
            await session.execute(
                text(
                    """
                    SELECT sp.name, sp.description
                    FROM symptom s
                    JOIN body_system_specialty bss ON bss.body_system = s.body_system
                    JOIN specialty sp ON sp.id = bss.specialty_id
                    WHERE s.id = ANY(:ids) AND s.body_system IS NOT NULL
                    GROUP BY sp.id, sp.name, sp.description
                    ORDER BY count(*) DESC, sp.name
                    LIMIT 1
                    """
                ),
                {"ids": sorted(symptom_ids)},
            )
        ).first()
        if row:
            return [SpecialtyAdvice(row.name, row.description, "body-system")]

    row = (
        await session.execute(
            text(
                "SELECT name, description FROM specialty WHERE slug = 'general-practice'"
            )
        )
    ).first()
    return [SpecialtyAdvice(row.name, row.description, "fallback")] if row else []


async def diagnose(
    session: AsyncSession,
    symptom_ids: set[int],
    config: ScoringConfig | None = None,
) -> Ranking:
    config = config or get_settings().scoring
    symptoms = await load_symptoms(session)
    profiles = await load_candidates(session, symptom_ids)
    return rank(symptom_ids, profiles, symptoms, config)
