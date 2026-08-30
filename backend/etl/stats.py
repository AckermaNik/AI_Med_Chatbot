"""Compute derived statistics over the loaded data.

    python etl/stats.py

Sets `symptom.idf` and refreshes `disease.low_evidence`. Idempotent — recomputes
from scratch every run, so it is safe after any reload.

IDF is the weight that makes ranking work:

    idf(s) = ln(N_diseases / n_diseases_containing(s))

A symptom in 3 of 793 diseases is highly diagnostic; one in 400 is nearly noise.
The dataset's severity weight is a DIFFERENT quantity and does not appear here —
severity drives urgency, IDF drives ranking (docs/PLAN.md section 1).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import SessionLocal, engine  # noqa: E402

LOW_EVIDENCE_THRESHOLD = 3


async def main() -> int:
    async with SessionLocal() as session:
        n_diseases = (await session.execute(text("SELECT count(*) FROM disease"))).scalar_one()
        if not n_diseases:
            print("no diseases loaded - run etl/load.py first")
            return 1

        # calculates the IDF (Inverse Document Frequency)
        await session.execute(
            text(
                """
                UPDATE symptom s
                -- CAST(), not :n::float - text() would read ':n:' as a variable name.
                SET idf = ln(CAST(:n AS float) / sub.cnt)
                FROM (
                    SELECT symptom_id, count(DISTINCT disease_id) AS cnt
                    FROM disease_symptom
                    GROUP BY symptom_id
                ) sub
                WHERE s.id = sub.symptom_id
                """
            ),
            {"n": n_diseases},
        )


        # It counts exactly how many symptoms each disease has and flags it as low_evidence if needed to.
        await session.execute(
            text(
                """
                UPDATE disease d
                SET low_evidence = COALESCE(sub.cnt, 0) < :threshold
                FROM (
                    SELECT d2.id, count(ds.symptom_id) AS cnt
                    FROM disease d2
                    LEFT JOIN disease_symptom ds ON ds.disease_id = d2.id
                    GROUP BY d2.id
                ) sub
                WHERE d.id = sub.id
                """
            ),
            {"threshold": LOW_EVIDENCE_THRESHOLD},
        )
        await session.commit()

        stats = (
            await session.execute(
                text(
                    """
                    SELECT count(*)                       AS n,
                           count(*) FILTER (WHERE idf <= 0) AS zero_idf,
                           round(min(idf)::numeric, 3)    AS lo,
                           round(avg(idf)::numeric, 3)    AS mean,
                           round(max(idf)::numeric, 3)    AS hi
                    FROM symptom
                    """
                )
            )
        ).one()

        print(f"  diseases                {n_diseases}")
        print(f"  symptoms                {stats.n}")
        print(f"  idf  min/mean/max       {stats.lo} / {stats.mean} / {stats.hi}")

        low_ev = (
            await session.execute(
                text("SELECT count(*) FROM disease WHERE low_evidence")
            )
        ).scalar_one()
        print(f"  low_evidence diseases   {low_ev}")

        if stats.zero_idf:
            print(
                f"  [warn] {stats.zero_idf} symptom(s) have idf <= 0 - they appear in "
                "every disease and carry no diagnostic signal"
            )

        print("\n  most diagnostic (rare):")
        for row in (
            await session.execute(
                text(
                    """
                    SELECT s.canonical_name, round(s.idf::numeric,2) AS idf,
                           count(ds.disease_id) AS n_diseases
                    FROM symptom s JOIN disease_symptom ds ON ds.symptom_id = s.id
                    GROUP BY s.id ORDER BY s.idf DESC LIMIT 5
                    """
                )
            )
        ).all():
            print(f"    {row.idf:5.2f}  {row.canonical_name:<38} in {row.n_diseases} disease(s)")

        print("\n  least diagnostic (common):")
        for row in (
            await session.execute(
                text(
                    """
                    SELECT s.canonical_name, round(s.idf::numeric,2) AS idf,
                           count(ds.disease_id) AS n_diseases
                    FROM symptom s JOIN disease_symptom ds ON ds.symptom_id = s.id
                    GROUP BY s.id ORDER BY s.idf ASC LIMIT 5
                    """
                )
            )
        ).all():
            print(f"    {row.idf:5.2f}  {row.canonical_name:<38} in {row.n_diseases} diseases")

    await engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
