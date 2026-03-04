"""
abm-gods-eye: A simulation observer SDK that drops into any Python ABM.

Provides a LangGraph-powered "god's eye" view over any simulation
that implements the SimulationAdapter protocol.
"""

from abm_gods_eye.adapter import SimulationAdapter
from abm_gods_eye.observer import GodsEye

__all__ = ["SimulationAdapter", "GodsEye"]
