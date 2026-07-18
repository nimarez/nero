"""M4 planner - straight-line path from pose to goal (clear floor, no obstacles).
Pure function over contracts. Swap in Dubins / A* later if arrival heading or obstacles matter."""
from __future__ import annotations
import math
from ..contracts import Trajectory, TrajPoint


def plan(pose, goal, traj_id=1, spacing=0.05):
    """Straight line pose -> goal, points ~spacing apart, heading = path tangent."""
    dx, dy = goal.x - pose.x, goal.y - pose.y
    dist = math.hypot(dx, dy)
    heading = math.atan2(dy, dx)
    n = max(2, int(dist / spacing) + 1)
    pts = [TrajPoint(x=pose.x + (i / (n - 1)) * dx,
                     y=pose.y + (i / (n - 1)) * dy,
                     heading=heading) for i in range(n)]
    return Trajectory(points=pts, traj_id=traj_id, t=goal.t)


if __name__ == "__main__":
    from ..contracts import RobotPose, Goal
    tr = plan(RobotPose(0, 0, 0, 0.0), Goal(2.5, 1.5, "demo", 0.0))
    print(f"plan: {len(tr.points)} pts, first={tr.points[0]}, last={tr.points[-1]}, traj_id={tr.traj_id}")
