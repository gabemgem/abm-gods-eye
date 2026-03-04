"""
abm-gods-eye adapter for the Schelling Segregation model.

════════════════════════════════════════════════════════════════
  THIS FILE is the only thing you need to add to an existing
  Mesa project to plug it into abm-gods-eye.
  model.py and agents.py are completely unchanged.
════════════════════════════════════════════════════════════════

The adapter implements the five-method SimulationAdapter protocol.
For a 20×20 grid with ~320 agents this comes to roughly 30 lines
of domain-specific code — everything else is handled by the SDK.
"""

from __future__ import annotations

from typing import Any

# ── abm-gods-eye import (the only new dependency) ────────────────────────────
from abm_gods_eye import SimulationAdapter  # noqa: F401 – imported for type-check

# ── existing project imports (unchanged) ─────────────────────────────────────
from examples.schelling_segregation.model import Schelling


# ── NEW: adapter class ────────────────────────────────────────────────────────
class SchellingAdapter:
    """
    SimulationAdapter wrapping the Schelling model.

    Pass an instance to GodsEye::

        from abm_gods_eye import GodsEye
        model = Schelling()
        eye = GodsEye(SchellingAdapter(model))
        eye.chat()
    """

    def __init__(self, model: Schelling) -> None:
        self._model = model
        self._history: list[dict[str, Any]] = []

    # -- SimulationAdapter protocol (5 required methods) ----------------------

    def get_state(self) -> dict[str, Any]:
        m = self._model
        return {
            "step": m.steps if hasattr(m, "steps") else None,
            "grid_width": m.grid.width,
            "grid_height": m.grid.height,
            "population": len(m.agents),
            "happy": m.happy,
            "running": getattr(m, "running", True),
            "homophily_threshold": next(iter(m.agents)).homophily if m.agents else None,
        }

    def get_agents(self) -> list[dict[str, Any]]:
        return [
            {
                "id": a.unique_id,
                "type": "minority" if a.type == 1 else "majority",
                "happy": a.happy,
                "x": a.cell.coordinate[0],
                "y": a.cell.coordinate[1],
            }
            for a in self._model.agents
        ]

    def get_metrics(self) -> dict[str, float | int]:
        m = self._model
        agents = list(m.agents)
        n = len(agents)
        if n == 0:
            return {}

        n_minority = sum(1 for a in agents if a.type == 1)
        n_happy = sum(1 for a in agents if a.happy)

        # Average fraction of same-type neighbors per agent (segregation index)
        seg_fractions: list[float] = []
        for a in agents:
            neighbors = list(a.cell.get_neighborhood(radius=a.radius).agents)
            if neighbors:
                seg_fractions.append(
                    sum(1 for nb in neighbors if nb.type == a.type) / len(neighbors)
                )
        avg_segregation = round(sum(seg_fractions) / len(seg_fractions), 4) if seg_fractions else 0.0

        return {
            "step": m.steps if hasattr(m, "steps") else None,
            "population": n,
            "happy": n_happy,
            "pct_happy": round(n_happy / n * 100, 1),
            "n_minority": n_minority,
            "n_majority": n - n_minority,
            "minority_pct": round(n_minority / n * 100, 1),
            "avg_segregation_index": avg_segregation,
        }

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            self._history.append(self.get_state())
            self._model.step()

    def get_history(self) -> list[dict[str, Any]]:
        return list(self._history)
# ── END of abm-gods-eye addition ─────────────────────────────────────────────
