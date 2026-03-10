"""
GodsEye: the main entry point for the SDK.

Wraps a SimulationAdapter with a LangGraph ReAct agent that can answer
natural-language questions about the simulation, step it forward, and
identify emergent patterns.
"""

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.language_models import BaseChatModel
from langgraph.prebuilt import create_react_agent

from abm_gods_eye.adapter import SimulationAdapter
from abm_gods_eye.callbacks import ThoughtLogger
from abm_gods_eye.llm import make_llm
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

_CHAT_BANNER = """
╔══════════════════════════════════════════════╗
║           abm-gods-eye  ·  Observer          ║
╚══════════════════════════════════════════════╝
Ask anything about the simulation.
Type  exit  or press Ctrl+C to quit.
"""

_DIVIDER = "─" * 48


def _last_ai_text(messages: list[BaseMessage]) -> str:
    """Return the text content of the last AIMessage in a message list."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and isinstance(msg.content, str) and msg.content:
            return msg.content
    return ""


class GodsEye:
    """
    Drop-in god's-eye observer for any Python ABM.

    Usage::

        from abm_gods_eye import GodsEye
        from my_sim import MyMesaAdapter  # your SimulationAdapter implementation

        adapter = MyMesaAdapter(model)
        eye = GodsEye(adapter, verbose=True)

        # Single question
        print(eye.ask("What patterns are emerging?"))

        # Interactive session (blocks until the user exits)
        eye.chat()

    Parameters
    ----------
    adapter:
        Any object satisfying the SimulationAdapter protocol.
    llm:
        A LangChain chat model. Defaults to Claude claude-sonnet-4-6 via langchain-anthropic.
        Pass any BaseChatModel to swap backends (OpenAI, Ollama, etc.).
    verbose:
        When True, attaches a ThoughtLogger callback that emits INFO-level log messages
        for each tool call and DEBUG-level messages for LLM steps. Configure the
        "abm_gods_eye" logger to control output destination and format.
    """

    def __init__(
        self,
        adapter: SimulationAdapter,
        llm: BaseChatModel | None = None,
        verbose: bool = False,
    ) -> None:
        if not isinstance(adapter, SimulationAdapter):
            raise TypeError(
                "adapter must implement the SimulationAdapter protocol. "
                "See abm_gods_eye.adapter.SimulationAdapter for required methods."
            )
        self._adapter = adapter
        self._llm = llm or make_llm("anthropic")
        self._tools = make_tools(adapter)
        self._agent = create_react_agent(
            model=self._llm,
            tools=self._tools,
            prompt=SYSTEM_PROMPT,
        )
        self._callbacks = [ThoughtLogger()] if verbose else []

    def ask(self, question: str) -> str:
        """
        Ask a stateless question about the simulation.

        Each call is independent — no conversation history is retained.
        For a multi-turn session with memory, use chat() instead.

        Parameters
        ----------
        question:
            Any question about the simulation state, agents, metrics, or trends.
        """
        config = {"callbacks": self._callbacks} if self._callbacks else {}
        result = self._agent.invoke(
            {"messages": [HumanMessage(content=question)]},
            config=config,
        )
        return _last_ai_text(result.get("messages", []))

    def stream(self, question: str):
        """
        Stream the agent's response token by token (stateless, like ask()).

        Yields string chunks suitable for progressive display in a UI.
        Thought-process logging (if verbose=True) is emitted as a side-effect
        via the logging module, not interleaved in the yielded chunks.
        """
        config = {"callbacks": self._callbacks} if self._callbacks else {}
        for chunk in self._agent.stream(
            {"messages": [HumanMessage(content=question)]},
            config=config,
        ):
            if "agent" in chunk:
                for msg in chunk["agent"].get("messages", []):
                    if isinstance(msg, AIMessage) and isinstance(msg.content, str):
                        yield msg.content

    def chat(self) -> None:
        """
        Start an interactive terminal chat session about the simulation.

        Conversation history is preserved across turns so the observer
        remembers what has already been discussed. The session blocks until
        the user types 'exit' or presses Ctrl+C / Ctrl+D.

        Example::

            eye = GodsEye(adapter, verbose=True)
            eye.chat()
        """
        print(_CHAT_BANNER)

        # Accumulated message history — passed into every agent call so the
        # LLM has full context of the conversation so far.
        history: list[BaseMessage] = []
        config = {"callbacks": self._callbacks} if self._callbacks else {}

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting. Goodbye.")
                break

            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", "q"}:
                print("Exiting. Goodbye.")
                break

            history.append(HumanMessage(content=user_input))

            result = self._agent.invoke({"messages": history}, config=config)
            history = result.get("messages", history)

            response = _last_ai_text(history)
            print(f"\nObserver: {response}\n{_DIVIDER}")
