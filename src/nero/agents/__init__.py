"""Nero agents package.

Available agents:
- orb_slam_agent: Navigate to a detected object using ORB-SLAM based navigation
- sim_agent: Test navigation in simulation mode
- booster_studio_agent: Run the same policy on a Booster Studio virtual K1
- mapping_agent: Map a space and create Gaussian splat
- map_nav_agent: Navigate using a pre-built map
"""

__all__ = [
    "orb_slam_agent_main",
    "sim_agent_main",
    "booster_studio_agent_main",
    "mapping_agent_main",
    "map_nav_agent_main",
]


def __getattr__(name: str):
    """Lazy imports to avoid requiring boosteros for sim-only usage."""
    if name == "orb_slam_agent_main":
        from nero.agents.orb_slam_agent import main as orb_slam_agent_main

        return orb_slam_agent_main
    elif name == "sim_agent_main":
        from nero.agents.sim_agent import main as sim_agent_main

        return sim_agent_main
    elif name == "booster_studio_agent_main":
        from nero.agents.booster_studio_agent import main as booster_studio_agent_main

        return booster_studio_agent_main
    elif name == "mapping_agent_main":
        from nero.agents.mapping_agent import main as mapping_agent_main

        return mapping_agent_main
    elif name == "map_nav_agent_main":
        from nero.agents.map_nav_agent import main as map_nav_agent_main

        return map_nav_agent_main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
