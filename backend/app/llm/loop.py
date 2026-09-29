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
from app.engine.safety import check_red_flags_in_text
from app.llm.tools import dispatch, tool_config

# Safety limit: one user message can trigger at most this many model/tool cycles.
# It prevents an accidental infinite loop if the model keeps asking for tools.
MAX_TOOL_STEPS = 10
DISCLAIMER = "This is not a substitute for a real medical assessment."
NO_SYMPTOM_MATCH_REPLY = (
    "I could not match a symptom in that message. Please describe what you are "
    "experiencing, including where it is and when it started."
)
NO_NEW_SYMPTOM_REPLY = (
    "Understood. I will keep the previous possible matches in mind. Tell me if "
    "you notice another symptom or have more information to add."
)

# Short conversational replies do not contain symptoms and should not trigger
# the expensive LLM -> matcher -> embedding pipeline.
CONFIRMATION_ONLY = re.compile(
    r"^(?:u\s+sure\??|are\s+you\s+sure\??|ok(?:ay)?\.?|what\s+do\s+you\s+mean\??)$",
    re.IGNORECASE,
)

SOCIAL_ONLY = re.compile(
    r"^(?:thanks?\.?|thank\s+you\.?|hello\.?|hi\.?|hey\.?|"
    r"you(?:'re|\s+are)\s+(?:awesome|great|amazing|helpful|the\s+best)\.?|"
    r"you(?:'re|\s+are)\s+(?:bad|terrible|useless|awful)\.?|you\s+suck\.?|"
    r"good\s+job\.?|nice\s+job\.?)$",
    re.IGNORECASE,
)

NON_MEDICAL_ONLY = re.compile(
    r"^(?:how\s+are\s+you\??|what(?:'s| is)\s+the\s+weather(?:\s+like)?\??|"
    r"who\s+is\s+.+\??|tell\s+me\s+about\s+(?:a|an|the)\s+.+|"
    r"what\s+is\s+(?:this|that)\s+object\??|what\s+does\s+.+\s+do\??|"
    r"what\s+jobs?\s+(?:are|is)\s+there\??|how\s+do\s+i\s+become\s+.+\??)$",
    re.IGNORECASE,
)

 # The UI is intentionally plain text. Remove Markdown emphasis/code markers
def normalize_reply(text: str) -> str:
    """Apply the plain-text and closing-disclaimer rules after model generation."""
   
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


def alert_payload(alert: Any) -> dict[str, str]:
    """Make every emergency event actionable even when the LLM is bypassed."""
    message = alert.message
    if alert.level == "emergency" and "112" not in message and "166" not in message:
        message = f"{message} Call 112 or 166 immediately."
    return {"level": alert.level, "name": alert.name, "message": message}


def is_conversational_only(message: str) -> bool:
    """Identify short confirmation messages that need no tool or model call."""
    return bool(CONFIRMATION_ONLY.fullmatch(message.strip()))


def is_social_only(message: str) -> bool:
    """Identify greetings and feedback that need no tool or model call."""
    return bool(SOCIAL_ONLY.fullmatch(message.strip()))


def is_non_medical_only(message: str) -> bool:
    """Identify clearly unrelated questions without blocking mixed health messages."""
    return bool(NON_MEDICAL_ONLY.fullmatch(message.strip()))


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

    * ``diagnose`` needs canonical names from ``search_symptoms``.
    * ``get_disease_info`` needs a disease returned by ``diagnose``.
    * ``recommend_specialty`` needs either search names for its fallback or a
      disease returned by ``diagnose``.

    A result does not exist until the current batch has finished, so a dependent
    tool must wait for Gemini's next response.
    """
    # A symptom search establishes the current-turn input. Hold every other
    # tool until its result is known so an empty search can end the turn without
    # executing or displaying downstream tools.
    if name != "search_symptoms" and "search_symptoms" in batch_names:
        return True

    dependencies = {
        "diagnose": {"search_symptoms"},
        "get_disease_info": {"diagnose"},
        "recommend_specialty": {"search_symptoms", "diagnose"},
    }
    return bool(dependencies.get(name, set()) & batch_names)


def _merge_matched_symptoms(
    active_symptoms: list[str], calls: list[Any], results: list[dict[str, Any]]
) -> bool:
    """Keep positive symptoms across follow-up turns and report new matches."""
    matched = [
        item.get("name")
        for call, result in zip(calls, results)
        if call.name == "search_symptoms"
        for item in result.get("matched") or []
        if item.get("name")
    ]
    for name in matched:
        if name not in active_symptoms:
            active_symptoms.append(name)
    return bool(matched)


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
    active_symptoms: list[str] | None = None,
) -> AsyncIterator[Event]:
    """Run one user message through the model-and-tools loop.

    The function yields progress events so callers can show a live tool trace.
    ``history`` is updated in place, preserving the full conversation context for
    the next user message, including tool requests and their returned data.
    """
    settings = get_settings()  # API model name and other application settings.
    config = tool_config()  # Gemini tool declarations and system prompt.
    active_symptoms = active_symptoms if active_symptoms is not None else []
    history.append(user_turn(message))  # Add this new message before asking Gemini.

    turn = Turn()  # Temporary Object to collect all activity and the final reply for this message.

    # Check the raw user message before asking the LLM to extract symptoms. This
    # prevents an emergency from being missed when the model omits one symptom,
    # such as "confused" in "I am having a seizure and I am confused".
    early_alerts = await check_red_flags_in_text(session, message)
    if early_alerts:
        for alert in early_alerts:
            yield Event("alert", alert_payload(alert))
        return

    if is_conversational_only(message):
        quick_reply = (
            "I am an AI chatbot, not a real doctor, so for a proper diagnosis you "
            "should consult a qualified doctor. If you think your condition is an "
            "emergency, call 112 or 166 immediately. Otherwise, I am happy to help "
            "you as far as I can."
        )
        history.append(types.Content(role="model", parts=[types.Part(text=quick_reply)]))
        yield Event("message", Turn(text=quick_reply))
        return

    if is_social_only(message):
        quick_reply = (
            "Hello! I am here to help with health symptoms and medical triage. "
            "Describe what you are experiencing and when it started."
        )
        history.append(types.Content(role="model", parts=[types.Part(text=quick_reply)]))
        yield Event("message", Turn(text=quick_reply))
        return

    if is_non_medical_only(message):
        quick_reply = (
            "I am designed to help with health symptoms and medical triage, so I "
            "cannot answer general questions about that topic. If you need help "
            "with symptoms, describe what you are experiencing and when it started."
        )
        history.append(types.Content(role="model", parts=[types.Part(text=quick_reply)]))
        yield Event("message", Turn(text=quick_reply))
        return

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
        effective_args: list[dict[str, Any]] = []
        for call in calls:
            args = dict(call.args or {})
            # The model may repeat or omit old symptom names while answering a
            # follow-up. The server-owned positive symptom state is authoritative.
            if call.name in {"diagnose", "recommend_specialty"} and active_symptoms:
                args["symptom_names"] = list(active_symptoms)
            effective_args.append(args)
            # Emit each request before running it so a UI can show activity early.
            if not _defer_until_next_round(call.name, batch_names):
                yield Event("tool_call", ToolCall(name=call.name, args=args))

        # Calls with no data dependency run together. 
        # A tool that needs a result from this batch is deferred until Gemini has received that result.
        # asyncio.gather guarantees the order of results to be according the calls order
        results = await asyncio.gather(
            *(
                _dispatch_in_batch(session, c.name, args, batch_names)
                for c, args in zip(calls, effective_args)
            )
        )

        parts: list[types.Part] = []  # Function-response parts sent back to Gemini.
        for index, (call, result) in enumerate(zip(calls, results)):
            # Pair each result with the corresponding request in the same order.
            record = ToolCall(name=call.name, args=effective_args[index], result=result)
            if not _defer_until_next_round(call.name, batch_names):
                turn.tool_calls.append(record)  # Retain it in the completed-turn record.
                yield Event("tool_result", record)  # Allow the caller to render the trace.

            # Escalations are emitted as their own event, not left to the model to
            # relay. If the model ignores or softens the warning, the user still
            # sees it.
            for alert in result.get("alerts") or []:
                message_text = alert["message"]
                if (
                    alert.get("level") == "emergency"
                    and "112" not in message_text
                    and "166" not in message_text
                ):
                    message_text = f"{message_text} Call 112 or 166 immediately."
                yield Event("alert", {**alert, "message": message_text})

            parts.append(
                types.Part.from_function_response(name=call.name, response=result)
            )  # Convert ordinary Python data to Gemini's tool-response format.

        # A negative answer to a follow-up question is not a new symptom, but it
        # also does not invalidate the positive symptoms already reported. Keep
        # those symptoms available so prior possible conditions remain possible.
        has_search = any(call.name == "search_symptoms" for call in calls)
        new_symptom_found = _merge_matched_symptoms(active_symptoms, calls, results)
        if has_search and not new_symptom_found:
            reply = NO_NEW_SYMPTOM_REPLY if active_symptoms else NO_SYMPTOM_MATCH_REPLY
            history.append(types.Content(role="user", parts=parts))
            history.append(
                types.Content(
                    role="model",
                    parts=[types.Part(text=reply)],
                )
            )
            yield Event("message", Turn(text=reply))
            return

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
