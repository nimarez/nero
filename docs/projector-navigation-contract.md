# Projector navigation handoff

The projector stack owns visualization and room-frame tracking only. It never sends velocity or motor commands.

## Frame

`room_floor` is a right-handed planar metric frame derived from the saved five-point floor calibration:

- origin: the captured floor center
- `+X`: projected grid-right
- `+Y`: projected grid-up
- units: metres from Lighthouse tracking
- vertical controller position: discarded

After the controller is taped to the robot, physically point the robot along projected `+X` and capture the mount yaw offset once:

```bash
curl -X POST http://10.2.4.14:8765/api/navigation/calibrate-forward
```

## Read state

One-shot JSON:

```bash
curl http://10.2.4.14:8765/api/navigation/state
```

Continuous JSON at 30 Hz:

```text
ws://10.2.4.14:8765/ws/navigation
```

The payload is versioned and contains the shared semantics Nima needs:

```json
{
  "version": 1,
  "frame_id": "room_floor",
  "robot_pose": {
    "x": 0.4,
    "y": -0.2,
    "yaw": 1.1,
    "t": 1784433616.8,
    "valid": true,
    "source": "vive:WW0",
    "frame_id": "room_floor",
    "heading_calibrated": true
  },
  "goal_pose": {
    "x": 1.8,
    "y": 0.7,
    "yaw": 0.0,
    "frame_id": "room_floor",
    "source": "operator-ui"
  },
  "trajectory": {
    "frame_id": "room_floor",
    "waypoints": [[0.4, -0.2], [1.1, 0.1], [1.8, 0.7]],
    "source": "nima-a-star"
  },
  "control_authority": "none"
}
```

Invalid or stale Vive tracking fails closed with `robot_pose.valid: false` or no robot pose.

## Write goal and plan

Goals include the grid-aligned direction the box faces:

```bash
curl -X POST http://10.2.4.14:8765/api/navigation/goal \
  -H 'content-type: application/json' \
  -d '{"x":1.8,"y":0.7,"yaw":0.0,"source":"operator"}'
```

Until Nima supplies a plan, the projector previews a straight vector from robot to goal. Nima replaces it with A* waypoints:

```bash
curl -X POST http://10.2.4.14:8765/api/navigation/trajectory \
  -H 'content-type: application/json' \
  -d '{"waypoints":[[0.4,-0.2],[1.1,0.1],[1.8,0.7]],"source":"nima-a-star"}'
```

ROS mapping is direct: `robot_pose` maps to `/nero/reference/pose`, `goal_pose` to `/nero/navigation/goal_pose`, and `trajectory.waypoints` to `/nero/navigation/plan`. The existing `ViveRosBridge` remains the ROS adapter; robot control stays on Nima's side.
