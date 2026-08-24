"""Talk to the assistant from the terminal. No web server, no React.

    python scripts/chat.py
    python scripts/chat.py --quiet          # hide the tool trace

If the conversation works here, the loop is correct. M5 wraps this in FastAPI and
M6 connects the React UI — one layer at a time, so a bug has one place to hide.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db import SessionLocal, engine  # noqa: E402
from app.llm.loop import build_client, run_turn  # noqa: E402

DIM, BOLD, RED, YELLOW, CYAN, RESET = (
    "\033[2m", "\033[1m", "\033[31m", "\033[33m", "\033[36m", "\033[0m",
)


def summarize(result: dict) -> str:
    """One line per tool result — the full payload is noise in a terminal."""
    if "matched" in result:
        names = [m["name"] for m in result["matched"]]
        out = f"{len(names)} matched: {', '.join(names)}" if names else "no matches"
        if result.get("unmatched"):
            out += f" | unmatched: {[u['phrase'] for u in result['unmatched']]}"
        return out
    if "candidates" in result:
        top = result["candidates"][:3]
        listed = ", ".join(f"{c['name']} ({c['score']})" for c in top)
        return f"{len(result['candidates'])} candidates: {listed or 'none'}"
    if "specialties" in result:
        return ", ".join(s["name"] for s in result["specialties"])
    if "name" in result:
        return result["name"]
    if "error" in result:
        return f"ERROR {result['error']}"
    return json.dumps(result)[:100]


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quiet", action="store_true", help="hide the tool trace")
    args = parser.parse_args()

    try:
        client = build_client()
    except RuntimeError as exc:
        print(exc)
        return 1

    history: list = []

    print(f"{BOLD}MedAi Clinic{RESET} {DIM}- terminal client. Ctrl-C to quit.{RESET}")
    print(f"{DIM}Not a substitute for real medical assessment.{RESET}\n")

    async with SessionLocal() as session:
        while True:
            try:
                message = input(f"{BOLD}you >{RESET} ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not message:
                continue
            if message in {"exit", "quit"}:
                break

            async for event in run_turn(session, client, history, message):
                if event.kind == "tool_call" and not args.quiet:
                    call = event.payload
                    print(f"{DIM}  -> {call.name}({json.dumps(call.args)}){RESET}")
                elif event.kind == "tool_result" and not args.quiet:
                    print(f"{DIM}     {summarize(event.payload.result)}{RESET}")
                elif event.kind == "alert":
                    alert = event.payload
                    colour = RED if alert["level"] == "emergency" else YELLOW
                    print(f"\n{colour}{BOLD}[{alert['level'].upper()}]{RESET} "
                          f"{colour}{alert['message']}{RESET}")
                elif event.kind == "message":
                    turn = event.payload
                    print(f"\n{CYAN}bot >{RESET} {turn.text}\n")
                elif event.kind == "error":
                    print(f"\n{RED}!{RESET} {event.payload}\n")

    await engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
