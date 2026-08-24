"""The four tools the model may call, and the dispatcher that runs them.

Each tool has a Pydantic argument model which serves two purposes: it generates the
schema Gemini receives, and it validates whatever Gemini actually sends before any
of it reaches SQL. Model output is never trusted into a query.
"""

from __future__ import annotations

from typing import Any

from google.genai import types
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.config import get_settings
from app.engine.matching import match_phrases
from app.engine.repository import diagnose as run_diagnose
from app.engine.repository import recommend_specialty, resolve_slugs
from app.engine.safety import check_red_flags
from app.llm.schema import flatten_schema
from app.llm.prompts import SYSTEM_PROMPT


# --------------------------------------------------------------------------- #
# argument models
# --------------------------------------------------------------------------- #


class SearchSymptomsArgs(BaseModel):
    phrases: list[str] = Field(
        description=(
            "Short symptom phrases taken from what the user said, e.g. "
            "['itchy rash', 'been throwing up']. One phrase per distinct symptom."
        )
    )


class DiagnoseArgs(BaseModel):
    symptom_slugs: list[str] = Field(
        description="Canonical symptom slugs returned by search_symptoms."
    )


class DiseaseArgs(BaseModel):
    disease_slug: str = Field(description="Disease slug returned by diagnose.")


class SpecialtyArgs(BaseModel):
    disease_slug: str | None = Field(
        default=None, description="Disease slug returned by diagnose, if known."
    )
    symptom_slugs: list[str] = Field(
        default_factory=list,
        description="Reported symptom slugs, used if no disease is confident.",
    )


# --------------------------------------------------------------------------- #
# implementations
# --------------------------------------------------------------------------- #


async def search_symptoms(session: AsyncSession, args: SearchSymptomsArgs) -> dict:
    matches, misses = await match_phrases(session, args.phrases)

    # Red flags are evaluated here, the moment symptoms become known, rather than
    # on the raw sentence — a whole sentence never matches a canonical symptom, so
    # checking earlier would mean the check effectively never fires. The result is
    # returned to the model AND surfaced by the loop as its own event, so the
    # escalation reaches the user even if the model ignores it.
    alerts = await check_red_flags(session, {m.symptom_id for m in matches})

    return {
        "matched": [
            {
                "phrase": m.phrase,
                "slug": m.slug,
                "name": m.name,
                "confidence": m.confidence,
                "matched_by": m.stage,
            }
            for m in matches
        ],
        "unmatched": [
            {"phrase": m.phrase, "did_you_mean": m.near} for m in misses
        ],
        "alerts": [
            {"level": a.level, "name": a.name, "message": a.message} for a in alerts
        ],
    }


async def diagnose(session: AsyncSession, args: DiagnoseArgs) -> dict:
    found = await resolve_slugs(session, args.symptom_slugs)
    unknown = [s for s in args.symptom_slugs if s not in found]
    if not found:
        return {
            "candidates": [],
            "note": "None of those symptom slugs exist. Call search_symptoms first.",
            "unknown_slugs": unknown,
        }

    result = await run_diagnose(session, set(found.values()))
    return {
        "candidates": [
            {
                "slug": c.slug,
                "name": c.name,
                "score": c.score,
                "matched_symptoms": [m.name for m in c.matched],
                "urgency": c.urgency,
                "evidence_is_thin": c.low_evidence,
            }
            for c in result.candidates
        ],
        "is_confident": result.is_confident,
        "discriminating_symptom": result.discriminating_symptom,
        "unknown_slugs": unknown,
    }


async def get_disease_info(session: AsyncSession, args: DiseaseArgs) -> dict:

    row = (
        await session.execute(
            text(
                "SELECT name, description, low_evidence FROM disease WHERE slug = :slug"
            ),
            {"slug": args.disease_slug},
        )
    ).first()
    if not row:
        return {"error": f"No disease with slug {args.disease_slug!r}."}

    precautions = (
        await session.execute(
            text(
                """
                SELECT p.text FROM precaution p
                JOIN disease d ON d.id = p.disease_id
                WHERE d.slug = :slug ORDER BY p.ordinal
                """
            ),
            {"slug": args.disease_slug},
        )
    ).scalars().all()

    return {
        "name": row.name,
        "description": row.description or "No description available for this condition.",
        "precautions": list(precautions),
        "evidence_is_thin": row.low_evidence,
    }


async def recommend_specialty_tool(
    session: AsyncSession, args: SpecialtyArgs
) -> dict:
    found = await resolve_slugs(session, args.symptom_slugs)
    advice = await recommend_specialty(
        session, args.disease_slug, set(found.values()) or None
    )
    return {
        "specialties": [
            {
                "name": a.name,
                "description": a.description,
                "mapping_source": a.mapping_source,
                "individually_verified": a.is_verified,
            }
            for a in advice
        ]
    }


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #

TOOLS: dict[str, tuple[type[BaseModel], Any, str]] = {
    "search_symptoms": (
        SearchSymptomsArgs,
        search_symptoms,
        "Resolve the user's own words into canonical symptoms from the database. "
        "Always call this before diagnose.",
    ),
    "diagnose": (
        DiagnoseArgs,
        diagnose,
        "Rank possible conditions for a set of canonical symptom slugs. Returns "
        "scores, the symptoms that matched, and the single most useful follow-up "
        "question to ask next.",
    ),
    "get_disease_info": (
        DiseaseArgs,
        get_disease_info,
        "Get the description and precautions for one condition.",
    ),
    "recommend_specialty": (
        SpecialtyArgs,
        recommend_specialty_tool,
        "Get the medical specialty to consult for a condition, or for the reported "
        "symptoms if no condition is confident.",
    ),
}


def declarations() -> list[types.FunctionDeclaration]:
    """Gemini-facing schemas, generated from the same Pydantic models used to
    validate incoming arguments — so the two can never drift apart."""
    return [
        types.FunctionDeclaration(
            name=name,
            description=description,
            parameters_json_schema=flatten_schema(model.model_json_schema()),
        )
        for name, (model, _, description) in TOOLS.items()
    ]


async def dispatch(session: AsyncSession, name: str, args: dict) -> dict:
    """Validate then run. Invalid arguments come back as an error the model can
    read and retry from, rather than raising and killing the conversation."""
    entry = TOOLS.get(name)
    if entry is None:
        return {"error": f"Unknown tool {name!r}. Available: {list(TOOLS)}"}

    model, fn, _ = entry
    try:
        validated = model.model_validate(args or {})
    except ValidationError as exc:
        return {"error": "Invalid arguments.", "details": exc.errors(include_url=False)}

    return await fn(session, validated)


def tool_config() -> types.GenerateContentConfig:
    

    settings = get_settings()
    return types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[types.Tool(function_declarations=declarations())],
        # We run the loop ourselves. Automatic mode would execute tools invisibly,
        # which hides the trace we want to show and removes our validation step.
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        temperature=0.2,
    )
