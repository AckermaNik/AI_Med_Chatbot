"""Drive the scoring engine against the real database from the command line.

    python scripts/try_diagnose.py itching skin-rash
    python scripts/try_diagnose.py headache vomiting --mu 0.5 --lambda 1.0

Slugs, not free text — symptom matching (stages 1-3) is a separate concern and
does not exist yet. This is here to sanity-check ranking against 793 real
diseases before any LLM is involved.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.db import SessionLocal, engine  # noqa: E402
from app.engine.repository import (  # noqa: E402
    diagnose,
    recommend_specialty,
    resolve_slugs,
)
from app.engine.safety import check_red_flags  # noqa: E402


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("slugs", nargs="+")
    parser.add_argument("--mu", type=float)
    parser.add_argument("--lambda", dest="lam", type=float)
    parser.add_argument("--top-k", type=int)
    args = parser.parse_args()

    config = get_settings().scoring.model_copy()
    if args.mu is not None:
        config.mu = args.mu
    if args.lam is not None:
        config.mmr_lambda = args.lam
    if args.top_k is not None:
        config.top_k = args.top_k

    async with SessionLocal() as session:
        found = await resolve_slugs(session, args.slugs)
        missing = [s for s in args.slugs if s not in found]
        if missing:
            print(f"unknown symptom slug(s): {', '.join(missing)}")
            if not found:
                return 1

        print(f"reported: {', '.join(sorted(found))}")
        print(f"config:   mu={config.mu}  lambda={config.mmr_lambda}  top_k={config.top_k}")

        ids = set(found.values())

        # Runs before anything else and never consults the model.
        for alert in await check_red_flags(session, ids):
            marker = "!!!" if alert.is_emergency else " ! "
            print(f"\n{marker} {alert.level.upper()}: {alert.name}")
            print(f"    {alert.message}")

        started = time.perf_counter()
        result = await diagnose(session, ids, config)
        elapsed = (time.perf_counter() - started) * 1000

        print(f"\n{len(result.candidates)} candidate(s) in {elapsed:.0f}ms\n")
        for i, c in enumerate(result.candidates, start=1):
            flags = " [low evidence]" if c.low_evidence else ""
            urgency = f"  urgency={c.urgency}" if c.urgency is not None else ""
            print(f"  {i}. {c.score:.3f}  {c.name}{flags}{urgency}")
            why = ", ".join(f"{m.name} ({m.contribution:.2f})" for m in c.matched)
            print(f"       because: {why}")
            if c.missing_key:
                print(f"       missing: {', '.join(c.missing_key)}")

        if result.candidates:
            top = result.candidates[0]
            for advice in await recommend_specialty(session, top.slug, ids):
                verified = "" if advice.is_verified else f"  [{advice.mapping_source}]"
                print(f"\n  specialty: {advice.name}{verified}")
                if not advice.is_verified:
                    print("             not individually verified - a GP can refer onward")

        print(f"\n  confident: {result.is_confident}")
        if result.discriminating_symptom:
            print(f"  ask next:  {result.discriminating_symptom}")
        if result.unmatched_reported:
            print(f"  unexplained: {', '.join(result.unmatched_reported)}")

    await engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
