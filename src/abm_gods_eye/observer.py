"""
GodsEye: the main entry point for the SDK.

Wraps a SimulationAdapter with a LangGraph ReAct agent that can answer
natural-language questions about the simulation, step it forward, and
identify emergent patterns.
"""

from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from langgraph.prebuilt import create_react_agent

from abm_gods_eye.adapter import SimulationAdapter
from abm_gods_eye.tools import make_tools

SYSTEM_PROMPT = """You are an all-seeing observer of a running agent-based simulation — a "god's eye" view.

You have access to tools that let you inspect the simulation state, query individual agents,
retrieve aggregate metrics, step the simulation forward, and compare states across time.

When answering questions:
- Use the tools to ground your answers in actual simulation data.
- Highlight emergent patterns, trends, and anomalies when you notice them.
- Be concise but precise. Prefer numbers and specifics over vague descriptions.
- If asked to step the simulation, do so and describe what changed.
- If you notice something interesting that the user didn't ask about, mention it briefly.

The user may be a researcher, student, or curious observer. Calibrate your language accordingly.
"""


class GodsEye:
    """
    Drop-in god's-eye observer for any Python ABM.

    Usage::

        from abm_gods_eye import GodsEye
        from my_sim import MyMesaAdapter  # your SimulationAdapter implementation

        adapter = MyMesaAdapter(model)
        eye = GodsEye(adapter)

        response = eye.ask("What patterns are emerging among the agents?")
        print(response)

    Parameters
    ----------
    adapter:
        Any object satisfying the SimulationAdapter protocol.
    llm:
        A LangChain chat model. Defaults to Claude claude-sonnet-4-6 via langchain-anthropic.
        Pass any BaseChatModel to swap backends (OpenAI, Ollama, etc.).
    """

    def __init__(
        self,
        adapter: SimulationAdapter,
        llm: BaseChatModel | None = None,
    ) -> None:
        if not isinstance(adapter, SimulationAdapter):
            raise TypeError(
                "adapter must implement the SimulationAdapter protocol. "
                "See abm_gods_eye.adapter.SimulationAdapter for required methods."
            )
        self._adapter = adapter
        self._llm = llm or ChatAnthropic(model="claude-sonnet-4-6")  # type: ignore[call-arg]
        self._tools = make_tools(adapter)
        self._agent = create_react_agent(
            model=self._llm,
            tools=self._tools,
            prompt=SYSTEM_PROMPT,
        )

    def ask(self, question: str) -> str:
        """
        Ask a natural-language question about the simulation.

        The agent will use its tools to inspect the simulation and return
        a grounded, human-readable answer.

        Parameters
        ----------
        question:
            Any question about the simulation state, agents, metrics, or trends.

        Returns
        -------
        str
            The LLM's response.
        """
        result = self._agent.invoke({"messages": [HumanMessage(content=question)]})
        messages = result.get("messages", [])
        # Last message is the final AI response
        for msg in reversed(messages):
            if hasattr(msg, "content") and isinstance(msg.content, str) and msg.content:
                return msg.content
        return ""

    def stream(self, question: str):
        """
        Stream the agent's response token by token.

        Yields string chunks suitable for progressive display in a UI.
        """
        for chunk in self._agent.stream({"messages": [HumanMessage(content=question)]}):
            if "agent" in chunk:
                for msg in chunk["agent"].get("messages", []):
                    if hasattr(msg, "content") and isinstance(msg.content, str):
                        yield msg.content
