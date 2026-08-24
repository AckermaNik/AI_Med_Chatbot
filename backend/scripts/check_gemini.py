"""Verify the Gemini API key works, before building anything on top of it.

    python scripts/check_gemini.py

Never prints the key. Reports its shape, then makes one real call and one
function-calling call to confirm the account, model and tool support are all live.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402


def inspect_key(key: str) -> None:
    print("--- key shape (value never printed) ---")
    print(f"  length            {len(key)}")
    print(f"  leading/trailing quote  {key[:1] in {chr(34), chr(39)}} / "
          f"{key[-1:] in {chr(34), chr(39)}}")
    print(f"  contains whitespace     {any(c.isspace() for c in key)}")
    print(f"  starts with 'AIza'      {key.startswith('AIza')}")


async def main() -> int:
    settings = get_settings()
    key = settings.gemini_api_key

    if not key:
        print("GEMINI_API_KEY is empty in backend/.env")
        return 1

    inspect_key(key)

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=key)

    print(f"\n--- plain call ({settings.gemini_model}) ---")
    try:
        response = await client.aio.models.generate_content(
            model=settings.gemini_model,
            contents="Reply with exactly the word: ready",
        )
        print(f"  model replied: {response.text.strip()!r}")
    except Exception as exc:  # noqa: BLE001 - diagnostic script
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        return 1

    print("\n--- function calling (automatic mode disabled) ---")
    declaration = types.FunctionDeclaration(
        name="lookup_symptom",
        description="Look up a symptom in the medical database.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "phrase": {"type": "string", "description": "The symptom to look up."}
            },
            "required": ["phrase"],
        },
    )

    try:
        response = await client.aio.models.generate_content(
            model=settings.gemini_model,
            contents="My skin is really itchy. Look it up.",
            config=types.GenerateContentConfig(
                tools=[types.Tool(function_declarations=[declaration])],
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
            ),
        )
        calls = response.function_calls
        if calls:
            for call in calls:
                print(f"  model requested: {call.name}({dict(call.args)})")
            print("\n  tool calling works - ready for M4")
        else:
            print(f"  model answered without calling a tool: {response.text!r}")
            print("  (not fatal, but check the tool description)")
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
