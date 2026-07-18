"""Nero simulation package.

Provides mock robot and simulated camera for testing agents without physical hardware.
"""

from nero.simulation.mock_robot import MockRobot
from nero.simulation.sim_camera import SimCamera
from nero.simulation.environment import SimEnvironment

__all__ = ["MockRobot", "SimCamera", "SimEnvironment"]
