"""
abm-gods-eye: A simulation observer SDK that drops into any Python ABM.

Provides a LangGraph-powered "god's eye" view over any simulation
that implements the SimulationAdapter protocol.

For the NetLogo bridge server, run:
    python -m abm_gods_eye.server --provider anthropic --model claude-sonnet-4-6
"""

from abm_gods_eye.adapter import SimulationAdapter
from abm_gods_eye.callbacks import ThoughtLogger
from abm_gods_eye.llm import make_llm
from abm_gods_eye.observer import GodsEye

__all__ = ["SimulationAdapter", "GodsEye", "ThoughtLogger", "make_llm"]
