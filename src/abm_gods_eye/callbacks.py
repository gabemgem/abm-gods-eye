"""
Callback handler that surfaces the agent's thought process as log messages.

Uses Python's standard logging module so callers can control verbosity
via their own logging config. The default logger name is "abm_gods_eye".
"""

import json
import logging
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

logger = logging.getLogger("abm_gods_eye")


def _preview(value: Any, max_chars: int = 120) -> str:
    """Truncate a value to a readable one-liner for log output."""
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    text = text.replace("\n", " ")
    return text[:max_chars] + "..." if len(text) > max_chars else text


class ThoughtLogger(BaseCallbackHandler):
    """
    Logs the agent's reasoning steps at DEBUG/INFO level.

    Events emitted:

    - INFO  "Calling tool: <name>  args: <preview>"
    - DEBUG "Tool result: <name>  → <preview>"
    - DEBUG "LLM thinking..." / "LLM response ready"
    - INFO  "Agent error: <error>"

    Usage::

        import logging
        logging.basicConfig(level=logging.INFO)

        eye = GodsEye(adapter, verbose=True)
        eye.ask("What is happening?")
    """

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        name = serialized.get("name", "unknown_tool")
        logger.info("Calling tool: %-20s  args: %s", name, _preview(input_str))

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        # kwargs may carry 'name' depending on LangChain version
        name = kwargs.get("name", "")
        label = f"Tool result: {name}" if name else "Tool result"
        logger.debug("%-30s → %s", label, _preview(output))

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        name = kwargs.get("name", "unknown_tool")
        logger.warning("Tool error (%s): %s", name, error)

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        logger.debug("LLM thinking...")

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        logger.debug("LLM response ready")

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        logger.error("LLM error: %s", error)
