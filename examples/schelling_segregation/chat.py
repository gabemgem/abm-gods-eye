"""
Launch an interactive god's-eye chat session on the Schelling model.

════════════════════════════════════════════════════════════════
  This script is the only NEW entry point added to the project.
  Everything below the "abm-gods-eye wiring" comment is new;
  the model construction above it is identical to what you
  would already have in any existing run script.
════════════════════════════════════════════════════════════════

Run:
    uv run --env-file=.env python examples/schelling_segregation/chat.py
"""

import logging
import sys

# ── existing project code (unchanged) ────────────────────────────────────────
sys.path.insert(0, ".")  # make sure the repo root is on the path

from examples.schelling_segregation.model import Schelling  # noqa: E402

model = Schelling(width=20, height=20, density=0.8, minority_pc=0.4, homophily=0.4)

# Run a few steps so there's something meaningful to observe from the start.
for _ in range(5):
    model.step()

# ── abm-gods-eye wiring (the only new code) ───────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")

from abm_gods_eye import GodsEye  # noqa: E402
from examples.schelling_segregation.gods_eye_adapter import SchellingAdapter  # noqa: E402

eye = GodsEye(SchellingAdapter(model), verbose=True)
eye.chat()
# ── END of abm-gods-eye wiring ────────────────────────────────────────────────
