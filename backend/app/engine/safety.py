"""Deterministic safety checks.

Red-flag evaluation runs BEFORE the LLM is invoked and never depends on it. If a
combination of reported symptoms matches a rule, the escalation fires regardless
of what any model would have said.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

LEVEL_ORDER = {"emergency": 2, "urgent": 1}


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
    """Every rule whose required symptoms are ALL present, worst level first.

    `require_all` is an int[] property of symptom ids; `@>` is 'contains', so the test is
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
