"""
LangChain tools auto-generated from a SimulationAdapter.

Each tool wraps one or more adapter methods and formats results
for LLM consumption. The GodsEye class assembles these into a
LangGraph workflow.
"""

import json
from typing import Any

from langchain_core.tools import tool

from abm_gods_eye.adapter import SimulationAdapter


def make_tools(adapter: SimulationAdapter) -> list:
    """Build LangChain tools bound to a specific adapter instance."""

    @tool
    def get_state() -> str:
        """
        Get the current overall state of the simulation.
        Returns step number, global parameters, and environment info.
        """
        state = adapter.get_state()
        return json.dumps(state, indent=2, default=str)

    @tool
    def get_metrics() -> str:
        """
        Get aggregate statistics about the current simulation state.
        Returns numeric metrics like population counts, averages, and distributions.
        """
        metrics = adapter.get_metrics()
        return json.dumps(metrics, indent=2, default=str)

    @tool
    def query_agents(filter_key: str = "", filter_value: str = "") -> str:
        """
        Get the list of agents in the simulation.
        Optionally filter by an agent attribute: provide filter_key and filter_value
        to return only agents where agent[filter_key] == filter_value.
        Leave both empty to return all agents (may be sampled for large populations).
        """
        agents = adapter.get_agents()
        if filter_key and filter_value:
            agents = [a for a in agents if str(a.get(filter_key)) == filter_value]
        # Cap at 50 agents to stay within context limits
        sample = agents[:50]
        result: dict[str, Any] = {"total": len(agents), "returned": len(sample), "agents": sample}
        if len(agents) > 50:
            result["note"] = f"Showing first 50 of {len(agents)} agents."
        return json.dumps(result, indent=2, default=str)

    @tool
    def step_simulation(n: int = 1) -> str:
        """
        Advance the simulation by n steps (default 1) and return the new state.
        Use this to observe how the simulation evolves over time.
        """
        adapter.step(n)
        state = adapter.get_state()
        metrics = adapter.get_metrics()
        return json.dumps({"new_state": state, "metrics": metrics}, indent=2, default=str)

    @tool
    def get_history() -> str:
        """
        Get the history of past simulation states, oldest first.
        Use this to identify trends, cycles, or turning points over time.
        """
        history = adapter.get_history()
        # Summarise to avoid blowing the context window
        summary = [{"step": i, **s} for i, s in enumerate(history)]
        summary = summary[-20:]  # last 20 snapshots
        return json.dumps(summary, indent=2, default=str)

    @tool
    def compare_states(step_a: int = -2, step_b: int = -1) -> str:
        """
        Compare two historical states by their index in the history list.
        step_a and step_b are indices into get_history() (negative indices supported).
        Returns a delta showing what changed between the two steps.
        """
        history = adapter.get_history()
        if not history:
            return "No history available yet."
        try:
            a = history[step_a]
            b = history[step_b]
        except IndexError:
            return f"History only has {len(history)} entries."

        delta: dict[str, Any] = {}
        all_keys = set(a) | set(b)
        for k in all_keys:
            va, vb = a.get(k), b.get(k)
            if va != vb:
                delta[k] = {"before": va, "after": vb}
        return json.dumps({"delta": delta}, indent=2, default=str)

    return [get_state, get_metrics, query_agents, step_simulation, get_history, compare_states]
