"""Serialization tests for the browser-facing SSE adapter."""

from __future__ import annotations

from app.llm.loop import Event, ToolCall, Turn
from app.main import event_payload, sse_event


def test_tool_result_is_json_ready_for_the_browser():
    event = Event(
        "tool_result",
        ToolCall("search_symptoms", {"phrases": ["head hurts"]}, {"matched": []}),
    )

    assert event_payload(event) == {
        "name": "search_symptoms",
        "args": {"phrases": ["head hurts"]},
        "result": {"matched": []},
    }


def test_message_and_sse_frame_have_the_expected_shape():
    payload = event_payload(Event("message", Turn(text="Hello")))

    assert payload == {"text": "Hello", "truncated": False}
    assert sse_event("message", payload) == 'event: message\ndata: {"text": "Hello", "truncated": false}\n\n'
