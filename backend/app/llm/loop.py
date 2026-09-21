"""The manual tool-calling loop.

Automatic function calling is deliberately switched off. Running the loop by hand
costs about forty lines and buys three things worth more than that:

  * arguments are validated before they reach SQL
  * every tool call is observable, so the UI can show the trace
  * the number of round trips is bounded, so a confused model cannot spin

Gemini can return SEVERAL function calls in one response. They are dispatched
concurrently; taking only function_calls[0] would silently drop work.

    It sends the user message plus chat history to Gemini, 
    runs whatever tool calls Gemini requests, returns results to Gemini, and emits events.
    
    Gemini requests tool:
        → Python executes tool
        → result becomes a function_response part
        → appended to history
        → next generate_content call sends it to Gemini
    
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from google import genai
from google.genai import types
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.llm.tools import dispatch, tool_config

# Safety limit: one user message can trigger at most this many model/tool cycles.
# It prevents an accidental infinite loop if the model keeps asking for tools.
MAX_TOOL_STEPS = 10
DISCLAIMER = "This is not a substitute for a real medical assessment."


def normalize_reply(text: str) -> str:
    """Apply the plain-text and closing-disclaimer rules after model generation."""
    # The UI is intentionally plain text. Remove Markdown emphasis/code markers
    # instead of trusting the model to remember that formatting constraint.
    cleaned = re.sub(r"[*_`~#]", "", text)
    # Do not allow a comma immediately before a coordinating conjunction.
    cleaned = re.sub(r",\s*(?=(?:and|or)\b)", " ", cleaned, flags=re.IGNORECASE)
    # Move any model-generated disclaimer to the end and normalize its wording.
    cleaned = re.sub(
        r"(?:please note that\s+)?(?:i am|this is) not a substitute for a real medical assessment\.?",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return f"{cleaned}".strip()


@dataclass
class ToolCall:
    """One tool request made by the model during a conversation turn."""

    name: str  # Registered tool name, for example "search_symptoms".
    args: dict[str, Any]  # Arguments supplied by the model for that tool.
    result: dict[str, Any] | None = None  # Filled in after the local tool finishes.


@dataclass
class Turn:
    """One user message and everything that happened because of it."""

    text: str = ""  # The final natural-language reply returned by the model.
    tool_calls: list[ToolCall] = field(default_factory=list)  # Completed local tool calls.
    steps: int = 0  # Number of model/tool cycles used for this user message.
    truncated: bool = False  # True when MAX_TOOL_STEPS stopped the loop early.
    error: str | None = None  # Friendly error text, when a model request failed.


@dataclass
class Event:
    """A single update for the terminal or future web UI to display."""

    kind: str  # Event category: tool_call, tool_result, message, error, or alert.
    payload: Any  # Data for the event; its type depends on ``kind``.


def build_client() -> genai.Client:
    """Create a configured Gemini API client, or fail clearly if its key is absent."""
    settings = get_settings()
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in backend/.env")
    return genai.Client(api_key=settings.gemini_api_key)


def user_turn(text: str) -> types.Content:
    """Wrap plain user text in Gemini's required conversation-message format."""
    return types.Content(role="user", parts=[types.Part(text=text)])


def _defer_until_next_round(name: str, batch_names: set[str]) -> bool:
    """Return whether a tool needs an earlier tool result from this batch.

    Args:
        name: The tool currently being considered, such as ``"diagnose"``.
        batch_names: Names of every tool Gemini requested in its current response.

    Gemini may request several tools in one response. Those requests normally run
    concurrently, but the triage tools form a workflow:

    * ``diagnose`` needs canonical slugs from ``search_symptoms``.
    * ``get_disease_info`` needs a disease returned by ``diagnose``.
    * ``recommend_specialty`` needs either search slugs for its fallback or a
      disease returned by ``diagnose``.

    A result does not exist until the current batch has finished, so a dependent
    tool must wait for Gemini's next response.
    """
    dependencies = {
        "diagnose": {"search_symptoms"},
        "get_disease_info": {"diagnose"},
        "recommend_specialty": {"search_symptoms", "diagnose"},
    }
    return bool(dependencies.get(name, set()) & batch_names)


async def _dispatch_in_batch(
    session: AsyncSession, name: str, args: dict[str, Any], batch_names: set[str]
) -> dict[str, Any]:
    """Run one tool, deferring tools that need a same-batch result."""
    if _defer_until_next_round(name, batch_names):
        return {
            "deferred": True,
            "note": (
                f"{name} was not run because it depends on a tool requested in "
                "the same batch. Read that tool's result, then call this tool "
                "again in the next tool round."
            ),
        }
    return await dispatch(session, name, args)


# runs in a WHILE TRUE loop
async def run_turn( 
    session: AsyncSession,
    client: genai.Client,
    history: list[types.Content],
    message: str,
) -> AsyncIterator[Event]:
    """Run one user message through the model-and-tools loop.

    The function yields progress events so callers can show a live tool trace.
    ``history`` is updated in place, preserving the full conversation context for
    the next user message, including tool requests and their returned data.
    """
    settings = get_settings()  # API model name and other application settings.
    config = tool_config()  # Gemini tool declarations and system prompt.
    history.append(user_turn(message))  # Add this new message before asking Gemini.

    turn = Turn()  # Collects all activity and the final reply for this message.

    for step in range(MAX_TOOL_STEPS):
        turn.steps = step + 1  # Human-friendly count: 1 through MAX_TOOL_STEPS.
        try:
            # Ask Gemini whether it wants to call tools or has a final answer.
            response = await client.aio.models.generate_content(
                model=settings.gemini_model, contents=history, config=config
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
            message_text = str(exc)  # Inspect text because quota errors vary by SDK.
            if "429" in message_text or "RESOURCE_EXHAUSTED" in message_text:
                friendly = (
                    "The free Gemini quota is used up for now. "
                    "Try again in a minute."
                )
            else:
                friendly = f"The model call failed: {type(exc).__name__}"
            turn.error = friendly  # Keep the error on the turn for other callers.
            yield Event("error", friendly)  # Let the UI display it immediately.
            return  # A failed model call cannot safely continue this turn.

        calls = response.function_calls
        if not calls:
            # No tool call so Gemini considers this its final user-facing reply.
            turn.text = normalize_reply(response.text or "")
            history.append(
                types.Content(role="model", parts=[types.Part(text=turn.text)])
            )  # Save the reply so Gemini can refer to it on the next turn.
            yield Event("message", turn)  # Give the caller the completed response.
            return

        # Save Gemini's tool-request message exactly as returned by the API.
        history.append(response.candidates[0].content)

        batch_names = {call.name for call in calls}  # Names requested in this response.
        for call in calls:
            # Emit each request before running it so a UI can show activity early.
            yield Event("tool_call", ToolCall(name=call.name, args=dict(call.args or {})))

        # Calls with no data dependency run together. A tool that needs a result
        # from this batch is deferred until Gemini has received that result.
        results = await asyncio.gather(
            *(
                _dispatch_in_batch(session, c.name, dict(c.args or {}), batch_names)
                for c in calls
            )
        )

        parts: list[types.Part] = []  # Function-response parts sent back to Gemini.
        for call, result in zip(calls, results):
            # Pair each result with the corresponding request in the same order.
            record = ToolCall(name=call.name, args=dict(call.args or {}), result=result)
            turn.tool_calls.append(record)  # Retain it in the completed-turn record.
            yield Event("tool_result", record)  # Allow the caller to render the trace.

            # Escalations are emitted as their own event, not left to the model to
            # relay. If the model ignores or softens the warning, the user still
            # sees it.
            for alert in result.get("alerts") or []:
                yield Event("alert", alert)

            parts.append(
                types.Part.from_function_response(name=call.name, response=result)
            )  # Convert ordinary Python data to Gemini's tool-response format.

        # Function responses go back under role="user". It reads oddly; it is what
        # the API expects.
        history.append(types.Content(role="user", parts=parts))

        # Go around the loop again so Gemini can read the tool results and respond.
    # The model requested tools too many times without reaching a final answer.
    turn.truncated = True
    turn.text = (
        "I wasn't able to work that out in a reasonable number of steps. "
        "Could you describe your symptoms again, a little more simply?"
    )
    history.append(types.Content(role="model", parts=[types.Part(text=turn.text)]))
    yield Event("message", turn)  # Return the safe fallback response to the caller.
