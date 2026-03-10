"""
LangChain tools for abm-gods-eye.

Two factories:
- make_tools(adapter)         — for Python ABMs via SimulationAdapter
- make_netlogo_tools(snapshots) — for NetLogo bridge server (reads from
                                  a list of push-recorded snapshots)

Each tool wraps simulation data and formats results for LLM consumption.
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


def make_netlogo_tools(snapshots: list[dict[str, Any]]) -> list:
    """
    Build LangChain tools backed by a mutable list of NetLogo snapshots.

    The ``snapshots`` list is owned by the session and mutated in-place by
    the ``/session/{id}/snapshot`` endpoint each tick. Tools read from it
    at call time, so they always reflect the latest pushed state.

    Note: NetLogo controls its own tick loop, so there is no step_simulation
    tool here — stepping is done from NetLogo's own ``go`` procedure.
    """

    @tool
    def get_state() -> str:
        """
        Get the most recent simulation state snapshot pushed from NetLogo.
        Returns tick number, global variables, and any other state the model
        records via gods-eye:record-state.
        """
        if not snapshots:
            return json.dumps({"error": "No snapshots recorded yet. Call gods-eye:record-state in your NetLogo model."})
        return json.dumps(snapshots[-1], indent=2, default=str)

    @tool
    def get_metrics() -> str:
        """
        Get aggregate statistics from the most recent NetLogo snapshot.
        Returns all numeric values from the latest recorded state.
        """
        if not snapshots:
            return json.dumps({"error": "No snapshots recorded yet."})
        latest = snapshots[-1]
        metrics = {k: v for k, v in latest.items() if isinstance(v, (int, float))}
        return json.dumps(metrics, indent=2, default=str)

    @tool
    def query_agents(filter_key: str = "", filter_value: str = "") -> str:
        """
        Get the list of agents (turtles) from the most recent NetLogo snapshot.
        Optionally filter by an agent attribute: provide filter_key and filter_value
        to return only agents where agent[filter_key] == filter_value.
        Leave both empty to return all agents (capped at 50 for large populations).
        """
        if not snapshots:
            return json.dumps({"error": "No snapshots recorded yet."})
        agents: list[dict[str, Any]] = snapshots[-1].get("agents", [])
        if filter_key and filter_value:
            agents = [a for a in agents if str(a.get(filter_key)) == filter_value]
        sample = agents[:50]
        result: dict[str, Any] = {"total": len(agents), "returned": len(sample), "agents": sample}
        if len(agents) > 50:
            result["note"] = f"Showing first 50 of {len(agents)} agents."
        return json.dumps(result, indent=2, default=str)

    @tool
    def get_history() -> str:
        """
        Get the history of past simulation states, oldest first (up to last 20).
        Use this to identify trends, cycles, or turning points over time.
        Each entry corresponds to one call to gods-eye:record-state in NetLogo.
        """
        if not snapshots:
            return json.dumps({"error": "No history recorded yet."})
        summary = [{"snapshot_index": i, **s} for i, s in enumerate(snapshots)]
        summary = summary[-20:]
        return json.dumps(summary, indent=2, default=str)

    @tool
    def compare_states(step_a: int = -2, step_b: int = -1) -> str:
        """
        Compare two historical snapshots by index (negative indices supported).
        step_a and step_b index into the snapshot history list.
        Returns a delta showing what changed between the two snapshots.
        """
        if not snapshots:
            return "No history available yet."
        try:
            a = snapshots[step_a]
            b = snapshots[step_b]
        except IndexError:
            return f"History only has {len(snapshots)} snapshots."

        delta: dict[str, Any] = {}
        all_keys = set(a) | set(b)
        for k in all_keys:
            va, vb = a.get(k), b.get(k)
            if va != vb:
                delta[k] = {"before": va, "after": vb}
        return json.dumps({"delta": delta}, indent=2, default=str)

    return [get_state, get_metrics, query_agents, get_history, compare_states]
