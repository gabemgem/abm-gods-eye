"""
Adapter protocol that any ABM must implement to work with abm-gods-eye.

Users write a thin adapter (~30 lines) wrapping their simulation,
and the SDK handles the rest.
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SimulationAdapter(Protocol):
    """
    Protocol that any simulation must satisfy to plug into GodsEye.

    Implement these five methods in a wrapper around your ABM
    (Mesa, Agentpy, custom, etc.) and pass the wrapper to GodsEye.
    """

    def get_state(self) -> dict[str, Any]:
        """
        Return a full snapshot of the current simulation state.

        Should include high-level metadata: step number, global params,
        environment state, aggregate counts, etc.
        """
        ...

    def get_agents(self) -> list[dict[str, Any]]:
        """
        Return a list of agent snapshots.

        Each dict should contain at minimum: {"id": ..., "type": ...}.
        Include whatever agent attributes are meaningful for the LLM to reason about.
        For large populations consider sampling or aggregating here.
        """
        ...

    def get_metrics(self) -> dict[str, float | int]:
        """
        Return aggregate statistics about the current simulation state.

        Examples: {"population": 500, "avg_wealth": 42.3, "gini": 0.4}
        These are surfaced as a dedicated LangChain tool for quick queries.
        """
        ...

    def step(self, n: int = 1) -> None:
        """Advance the simulation by n steps."""
        ...

    def get_history(self) -> list[dict[str, Any]]:
        """
        Return a list of past state snapshots, oldest first.

        Each entry should have the same shape as get_state().
        The SDK uses this for trend analysis and change detection.
        """
        ...
