"""
GodsEye HTTP server — bridges the NetLogo extension to the Python SDK.

Exposes a minimal REST API so that the Java NetLogo extension can:
  - Start a conversation session
  - Ask questions (with conversation history preserved across turns)
  - Record per-tick simulation snapshots for history/trend analysis
  - End a session

Intended to be launched as a subprocess by the NetLogo extension:

    python -m abm_gods_eye.server --provider anthropic --model claude-sonnet-4-6 --port 8765

Environment variables (can also be supplied via a .env file):
    GODS_EYE_PROVIDER   anthropic | openai | google
    GODS_EYE_MODEL      model identifier (provider-specific)
    GODS_EYE_PORT       port to listen on (default 8765)
    ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from typing import Any

# Load .env before anything else so env vars are available to LangChain
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

from abm_gods_eye.llm import make_llm
from abm_gods_eye.tools import make_netlogo_tools

app = FastAPI(title="abm-gods-eye server")

# ---------------------------------------------------------------------------
# In-memory session store
# ---------------------------------------------------------------------------

class _Session:
    """Holds per-session state: agent, conversation history, simulation snapshots."""

    def __init__(self, provider: str, model: str, system_prompt: str | None = None) -> None:
        from abm_gods_eye.observer import SYSTEM_PROMPT
        self.llm = make_llm(provider, model)
        self.snapshots: list[dict[str, Any]] = []
        self.history: list[BaseMessage] = []
        prompt = system_prompt or SYSTEM_PROMPT
        tools = make_netlogo_tools(self.snapshots)
        self.agent = create_react_agent(model=self.llm, tools=tools, prompt=prompt)


_sessions: dict[str, _Session] = {}

# Provider/model set at server startup
_provider: str = "anthropic"
_model: str | None = None


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class StartRequest(BaseModel):
    system_prompt: str | None = None


class StartResponse(BaseModel):
    session_id: str


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    response: str


class SnapshotRequest(BaseModel):
    """A single tick's simulation state pushed from NetLogo."""
    state: dict[str, Any]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _last_ai_text(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content:
            return msg.content
    return ""


def _get_session(session_id: str) -> _Session:
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return session


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "provider": _provider, "model": _model})


@app.post("/session/start", response_model=StartResponse)
def start_session(req: StartRequest) -> StartResponse:
    session_id = str(uuid.uuid4())
    _sessions[session_id] = _Session(_provider, _model, req.system_prompt)
    return StartResponse(session_id=session_id)


@app.post("/session/{session_id}/ask", response_model=AskResponse)
def ask(session_id: str, req: AskRequest) -> AskResponse:
    session = _get_session(session_id)
    session.history.append(HumanMessage(content=req.question))
    result = session.agent.invoke({"messages": session.history})
    session.history = result.get("messages", session.history)
    return AskResponse(response=_last_ai_text(session.history))


@app.post("/session/{session_id}/snapshot")
def record_snapshot(session_id: str, req: SnapshotRequest) -> JSONResponse:
    """Push a simulation state snapshot (called each tick from NetLogo)."""
    session = _get_session(session_id)
    session.snapshots.append(req.state)
    # Keep only the last 100 snapshots to bound memory
    if len(session.snapshots) > 100:
        session.snapshots = session.snapshots[-100:]
    return JSONResponse({"snapshot_count": len(session.snapshots)})


@app.delete("/session/{session_id}")
def end_session(session_id: str) -> JSONResponse:
    _sessions.pop(session_id, None)
    return JSONResponse({"ended": session_id})


@app.get("/session/{session_id}/history")
def get_history(session_id: str) -> JSONResponse:
    session = _get_session(session_id)
    serialised = [
        {"role": "human" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
        for m in session.history
        if isinstance(m, (HumanMessage, AIMessage))
    ]
    return JSONResponse({"history": serialised})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="abm-gods-eye HTTP bridge server")
    p.add_argument("--provider", default=None, help="LLM provider: anthropic | openai | google")
    p.add_argument("--model", default=None, help="Model identifier")
    p.add_argument("--port", type=int, default=None, help="Port to listen on (default 8765)")
    p.add_argument("--env-file", default=".env", help="Path to .env file (default: .env)")
    return p.parse_args()


def main() -> None:
    import uvicorn

    global _provider, _model

    args = _parse_args()

    # Load .env file first so env vars are available below
    load_dotenv(dotenv_path=args.env_file, override=False)

    # Resolve provider / model / port: CLI args > env vars > defaults
    _provider = args.provider or os.environ.get("GODS_EYE_PROVIDER", "anthropic")
    _model = args.model or os.environ.get("GODS_EYE_MODEL") or None  # None → factory picks default
    port = args.port or int(os.environ.get("GODS_EYE_PORT", "8765"))

    # Validate provider early so users get a clear error before any request
    from abm_gods_eye.llm import SUPPORTED_PROVIDERS
    if _provider not in SUPPORTED_PROVIDERS:
        print(
            f"ERROR: Unknown provider '{_provider}'. "
            f"Supported: {', '.join(SUPPORTED_PROVIDERS)}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[gods-eye] Starting server: provider={_provider} model={_model or '(default)'} port={port}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
