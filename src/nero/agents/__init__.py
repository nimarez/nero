"""Nero agents package.

Available agents:
- goto_agent: Navigate to a detected object
- mapping_agent: Map a space and create Gaussian splat
"""

from nero.agents.goto_agent import main as goto_agent_main
from nero.agents.mapping_agent import main as mapping_agent_main

__all__ = ["goto_agent_main", "mapping_agent_main"]