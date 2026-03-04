"""
Example Mesa adapter for abm-gods-eye.

Shows how to wrap a Mesa model in ~30 lines so GodsEye can observe it.
The Boltzmann Wealth Model is Mesa's canonical hello-world example.

Install extras:
    uv add mesa

Run:
    uv run python examples/mesa_adapter.py
"""

from __future__ import annotations

import random
from typing import Any

# Mesa imports — install with: uv add mesa
try:
    import mesa
except ImportError as e:
    raise ImportError("Install mesa: uv add mesa") from e

from abm_gods_eye import GodsEye, SimulationAdapter


# ---------------------------------------------------------------------------
# A minimal Mesa model (Boltzmann Wealth)
# ---------------------------------------------------------------------------

class WealthAgent(mesa.Agent):
    def __init__(self, model: mesa.Model) -> None:
        super().__init__(model)
        self.wealth: int = 1

    def step(self) -> None:
        if self.wealth == 0:
            return
        other = self.random.choice(self.model.agents)
        other.wealth += 1
        self.wealth -= 1


class WealthModel(mesa.Model):
    def __init__(self, n_agents: int = 100, seed: int = 42) -> None:
        super().__init__(seed=seed)
        self.current_step = 0
        WealthAgent.create_agents(self, n_agents)

    def step(self) -> None:
        self.agents.shuffle_do("step")
        self.current_step += 1


# ---------------------------------------------------------------------------
# The adapter — this is the only thing the user needs to write
# ---------------------------------------------------------------------------

class WealthModelAdapter:
    """SimulationAdapter wrapping Mesa's WealthModel."""

    def __init__(self, model: WealthModel) -> None:
        self._model = model
        self._history: list[dict[str, Any]] = []

    def get_state(self) -> dict[str, Any]:
        return {
            "step": self._model.current_step,
            "n_agents": len(self._model.agents),
        }

    def get_agents(self) -> list[dict[str, Any]]:
        return [
            {"id": a.unique_id, "wealth": a.wealth}
            for a in self._model.agents
        ]

    def get_metrics(self) -> dict[str, float | int]:
        wealths = [a.wealth for a in self._model.agents]
        n = len(wealths)
        avg = sum(wealths) / n if n else 0
        sorted_w = sorted(wealths)
        # Gini coefficient
        cumulative = sum((i + 1) * w for i, w in enumerate(sorted_w))
        gini = (2 * cumulative) / (n * sum(sorted_w)) - (n + 1) / n if sum(sorted_w) else 0
        return {
            "step": self._model.current_step,
            "total_wealth": sum(wealths),
            "avg_wealth": round(avg, 2),
            "min_wealth": min(wealths),
            "max_wealth": max(wealths),
            "gini": round(gini, 4),
            "pct_with_zero_wealth": round(wealths.count(0) / n * 100, 1),
        }

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            self._history.append(self.get_state())
            self._model.step()

    def get_history(self) -> list[dict[str, Any]]:
        return list(self._history)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging
    # Show INFO logs so tool calls are visible; use DEBUG to also see LLM steps.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")

    model = WealthModel(n_agents=100)
    adapter = WealthModelAdapter(model)

    # Run a few steps before asking questions
    adapter.step(10)

    eye = GodsEye(adapter, verbose=True)

    questions = [
        "What does the current wealth distribution look like? Is inequality high or low?",
        "Step the simulation forward 20 more steps and tell me what changed.",
        "Which agents have the most wealth right now?",
    ]

    for q in questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print(f"{'='*60}")
        print(eye.ask(q))
