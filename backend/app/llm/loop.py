"""The manual tool-calling loop.

Automatic function calling is deliberately switched off. Running the loop by hand
costs about forty lines and buys three things worth more than that:

  * arguments are validated before they reach SQL
  * every tool call is observable, so the UI can show the trace
  * the number of round trips is bounded, so a confused model cannot spin

Gemini can return SEVERAL function calls in one response. They are dispatched
concurrently; taking only function_calls[0] would silently drop work.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from google import genai
from google.genai import types
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.llm.tools import dispatch, tool_config

MAX_TOOL_STEPS = 5


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    result: dict[str, Any] | None = None


@dataclass
class Turn:
    """One user message and everything that happened because of it."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    steps: int = 0
    truncated: bool = False
    error: str | None = None


@dataclass
class Event:
    kind: str  # 'tool_call' | 'tool_result' | 'message' | 'error'
    payload: Any


def build_client() -> genai.Client:
    settings = get_settings()
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in backend/.env")
    return genai.Client(api_key=settings.gemini_api_key)


def user_turn(text: str) -> types.Content:
    return types.Content(role="user", parts=[types.Part(text=text)])


async def run_turn(
    session: AsyncSession,
    client: genai.Client,
    history: list[types.Content],
    message: str,
) -> AsyncIterator[Event]:
    """Drive one user message to completion, yielding events as they happen.

    `history` is mutated in place so the caller keeps full conversation context
    across turns, including the tool calls and their results.
    """
    settings = get_settings()
    config = tool_config()
    history.append(user_turn(message))

    turn = Turn()

    for step in range(MAX_TOOL_STEPS):
        turn.steps = step + 1
        try:
            response = await client.aio.models.generate_content(
                model=settings.gemini_model, contents=history, config=config
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
            message_text = str(exc)
            if "429" in message_text or "RESOURCE_EXHAUSTED" in message_text:
                friendly = (
                    "The free Gemini quota is used up for now. "
                    "Try again in a minute."
                )
            else:
                friendly = f"The model call failed: {type(exc).__name__}"
            turn.error = friendly
            yield Event("error", friendly)
            return

        calls = response.function_calls
        if not calls:
            turn.text = (response.text or "").strip()
            history.append(
                types.Content(role="model", parts=[types.Part(text=turn.text)])
            )
            yield Event("message", turn)
            return

        history.append(response.candidates[0].content)

        for call in calls:
            yield Event("tool_call", ToolCall(name=call.name, args=dict(call.args or {})))

        results = await asyncio.gather(
            *(dispatch(session, c.name, dict(c.args or {})) for c in calls)
        )

        parts: list[types.Part] = []
        for call, result in zip(calls, results):
            record = ToolCall(name=call.name, args=dict(call.args or {}), result=result)
            turn.tool_calls.append(record)
            yield Event("tool_result", record)

            # Escalations are emitted as their own event, not left to the model to
            # relay. If the model ignores or softens the warning, the user still
            # sees it.
            for alert in result.get("alerts") or []:
                yield Event("alert", alert)

            parts.append(
                types.Part.from_function_response(name=call.name, response=result)
            )

        # Function responses go back under role="user". It reads oddly; it is what
        # the API expects.
        history.append(types.Content(role="user", parts=parts))

    turn.truncated = True
    turn.text = (
        "I wasn't able to work that out in a reasonable number of steps. "
        "Could you describe your symptoms again, a little more simply?"
    )
    history.append(types.Content(role="model", parts=[types.Part(text=turn.text)]))
    yield Event("message", turn)
