"""Deterministic safety checks.

Red-flag evaluation runs BEFORE the LLM is invoked and never depends on it. If a
combination of reported symptoms matches a rule, the escalation fires regardless
of what any model would have said.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

LEVEL_ORDER = {"emergency": 2, "urgent": 1}

# These phrases are used only as an early safety net before the LLM extracts
# canonical symptoms. The normal database matcher still remains the source of
# truth for ordinary symptom matching.
TEXT_SYMPTOM_PATTERNS = {
    "chest-pain": (r"\bchest pain\b",),
    "breathlessness": (
        r"\bshortness of breath\b",
        r"\bshort of breath\b",
        r"\bbreathless\b",
        r"\bcan(?:not|'t) breathe\b",
    ),
    "sweating": (r"\bsweating\b", r"\bclammy\b"),
    "weakness-on-one-body-side": (
        r"\bweakness on (?:one|the) (?:side|half)\b",
        r"\bone side of (?:my|the) body is weak\b",
    ),
    "focal-weakness": (r"\bface droop(?:ing)?\b", r"\bfacial weakness\b"),
    "slurred-speech": (r"\bslurred speech\b", r"\bcan(?:not|'t) speak clearly\b"),
    "altered-sensorium": (
        r"\bconfused\b",
        r"\bdisoriented\b",
        r"\bunconscious\b",
        r"\bunresponsive\b",
        r"\blost consciousness\b",
        r"\bpassed out\b",
    ),
    "coma": (r"\bcoma\b",),
    "seizures": (r"\bseizure\b", r"\bseizures\b", r"\bconvulsion(?:s)?\b"),
    "stiff-neck": (r"\bstiff neck\b",),
    "high-fever": (r"\bhigh fever\b", r"\bvery high fever\b"),
    "vomiting-blood": (r"\bvomit(?:ing)? blood\b", r"\bthrowing up blood\b"),
}


@dataclass(frozen=True)
class Alert:
    name: str
    level: str
    message: str

    @property
    def is_emergency(self) -> bool:
        return self.level == "emergency"


async def check_red_flags(
    session: AsyncSession, symptom_ids: set[int]
) -> list[Alert]:
    """Every rule whose required symptoms are ALL present.

    `require_all` is a property of red_flag table. `@>` is 'contains', so the test is
    'the reported set contains every symptom this rule requires'.
    """
    if not symptom_ids:
        return []

    rows = await session.execute(
        text(
            """
            SELECT name, level, message
            FROM red_flag
            WHERE CAST(:ids AS int[]) @> require_all
            """
        ),
        {"ids": sorted(symptom_ids)},
    )
    
    # Sorts the alerts list in-place.
    # Primary sort: Severity level descending (-LEVEL_ORDER.get(a.level, default=0)).
    # Secondary sort: Alphabetical by alert name.
        
    alerts = [Alert(name=r.name, level=r.level, message=r.message) for r in rows]
    alerts.sort(key=lambda a: (-LEVEL_ORDER.get(a.level, 0), a.name))
    return alerts


async def check_red_flags_in_text(session: AsyncSession, message: str) -> list[Alert]:
    """Run a conservative emergency check before the LLM extracts symptoms."""
    normalized = message.casefold()
    slugs = {
        slug
        for slug, patterns in TEXT_SYMPTOM_PATTERNS.items()
        if any(re.search(pattern, normalized) for pattern in patterns)
    }
    if not slugs:
        return []

    rows = await session.execute(
        text("SELECT id FROM symptom WHERE slug = ANY(CAST(:slugs AS text[]))"),
        {"slugs": sorted(slugs)},
    )
    return await check_red_flags(session, {row.id for row in rows})
