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


async def resolve_names(session: AsyncSession, names: list[str]) -> dict[str, int]:
    """canonical symptom name -> symptom id.

    Unknown names are simply absent from the result. The LLM-facing tools use
    canonical names; database slugs remain internal stable identifiers.
    """
    if not names:
        return {}
    rows = await session.execute(
        text("SELECT canonical_name, id FROM symptom WHERE canonical_name = ANY(:names)"),
        {"names": names},
    )
    return {r.canonical_name: r.id for r in rows}


@dataclass(frozen=True)
class SpecialtyAdvice:
    name: str
    description: str
    mapping_source: str  # 'curated' | 'body-system' | 'fallback'

    @property
    def is_verified(self) -> bool:
        """False means no human checked this referral — say so in the UI."""
        return self.mapping_source == "curated"


async def _general_practice(session: AsyncSession) -> SpecialtyAdvice | None:
    """Load the General Practice fallback recommendation."""
    row = (
        await session.execute(
            text(
                "SELECT name, description FROM specialty "
                "WHERE slug = 'general-practice'"
            )
        )
    ).first()
    return SpecialtyAdvice(row.name, row.description, "fallback") if row else None


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
                    LIMIT 2
                    """
                ),
                {"slug": disease_slug},
            )
        ).all()
        if rows:
            advice = [
                SpecialtyAdvice(r.name, r.description, r.mapping_source)
                for r in rows
            ]
            if advice[0].name.strip().casefold() == "general practice":
                return advice[:1]
            return advice

    if symptom_ids:
        rows = (
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
                    LIMIT 2
                    """
                ),
                {"ids": sorted(symptom_ids)},
            )
        ).all()
        if rows:
            advice = [
                SpecialtyAdvice(row.name, row.description, "body-system")
                for row in rows
            ]
            if advice[0].name.strip().casefold() == "general practice":
                return advice[:1]
            return advice

    gp = await _general_practice(session)
    return [gp] if gp else []


async def diagnose(
    session: AsyncSession,
    symptom_ids: set[int],
    config: ScoringConfig | None = None,
) -> Ranking:
    config = config or get_settings().scoring
    symptoms = await load_symptoms(session) #load ALL symptoms of the dataset
    profiles = await load_candidates(session, symptom_ids) # load disease candidates based on user's symptoms
    return rank(symptom_ids, profiles, symptoms, config)
