"""HTTP/SSE adapter for the manual Gemini tool-calling loop.

The CLI in ``scripts/chat.py`` remains useful for debugging. This module exposes
the exact same ``run_turn`` events to the React interface over Server-Sent Events.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from google.genai import types
from pydantic import BaseModel, Field
from sqlalchemy import text

from app.config import get_settings
from app.db import SessionLocal, engine
from app.llm.loop import Event, ToolCall, Turn, build_client, run_turn


@dataclass
class Conversation:
    """Server-side state for one browser chat session."""

    history: list[types.Content] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class ChatRequest(BaseModel):
    """The browser sends a message and, after its first request, a session ID."""

    message: str = Field(min_length=1, max_length=4_000)
    session_id: str | None = None


conversations: dict[str, Conversation] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create shared resources once at startup and close database connections safely."""
    app.state.gemini_client = build_client()
    yield
    await engine.dispose()


app = FastAPI(title="MedAi Clinic API", version="0.1.0", lifespan=lifespan)

settings = get_settings()
allowed_origins = [
    origin.strip()
    for origin in settings.cors_origins.split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


def sse_event(kind: str, payload: dict) -> str:
    """Encode one named Server-Sent Event frame."""
    return f"event: {kind}\ndata: {json.dumps(payload)}\n\n"


def event_payload(event: Event) -> dict:
    """Convert internal loop events into JSON-safe values for the browser."""
    if event.kind in {"tool_call", "tool_result"}:
        call: ToolCall = event.payload
        payload = {"name": call.name, "args": call.args}
        if event.kind == "tool_result":
            payload["result"] = call.result
        return payload
    if event.kind == "message":
        turn: Turn = event.payload
        return {"text": turn.text, "truncated": turn.truncated}
    if event.kind == "error":
        return {"message": str(event.payload)}
    return event.payload

@app.head("/")
async def health_check():
    return {"status": "Server is awake and running!"}

@app.get("/api/health")
async def health() -> dict[str, str]:
    """Small readiness check for local development and deployment platforms."""
    async with engine.connect() as connection:
        await connection.execute(text("SELECT 1"))
    return {"status": "ok"}


@app.post("/api/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    """Stream tool activity and the final assistant reply for one chat message."""
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="Message cannot be blank.")

    session_id = request.session_id or str(uuid4())
    conversation = conversations.setdefault(session_id, Conversation())

    async def stream() -> AsyncIterator[str]:
        # Serialize requests for one chat so history remains in user-message order.
        async with conversation.lock:
            async with SessionLocal() as session:
                async for event in run_turn(
                    session,
                    app.state.gemini_client,
                    conversation.history,
                    message,
                ):
                    yield sse_event(event.kind, event_payload(event))
        yield sse_event("done", {"session_id": session_id})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
