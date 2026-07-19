# M2 Vive localization -> ROS 2 handoff

Owner: Jon (`hi@jonny.sh`)  
Consumer: Nima / navigation and K1 integration  
Implementation PR: <https://github.com/nimarez/nero/pull/6>

## What is live now

```text
Vive Controller 1.0 (WW0, wired USB, powered on)
  -> Raspberry Pi 400 + libsurvive
  -> versioned JSON/UDP over NERA-WIFI
  -> jscore/POS 10.77.0.1:43100
  -> /run/nero/vive_pose.json (atomic latest state)
  -> nero-vive-ros (this handoff boundary)
```

Two Base Station 1.0 units are configured on optical channels B+C. The Pi and
POS services are enabled across reboot. Real validation produced 147/147 valid
samples in the continuity window with no sequence loss. The full Python suite
passes.

The base POS host does **not** currently contain ROS 2 or `rclpy`. Run the bridge
inside Nima's sourced ROS 2 environment/container and mount the POS state file
read-only. Do not add ROS imports to the proven UDP receiver service.

## Input contract on POS

`/run/nero/vive_pose.json` contains:

```json
{
  "version": 1,
  "sequence": 123,
  "timestamp": 1750000000.25,
  "controller_id": "WW0",
  "position": [0.1, 0.2, 0.3],
  "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
  "linear_velocity": [0.0, 0.0, 0.0],
  "angular_velocity": [0.0, 0.0, 0.0],
  "tracking_valid": true,
  "transport": {
    "received_at": 1750000000.26,
    "dropped_since_previous": 0,
    "out_of_order": false
  }
}
```

Units are metres, seconds, m/s, and rad/s. Quaternion order is ROS-compatible
`x,y,z,w`. `timestamp` is the Pi source clock; `transport.received_at` is the
POS arrival clock and is the authoritative value for local staleness checks.

## ROS topics Nima receives

Run from the ROS-equipped environment:

```bash
source /opt/ros/<distro>/setup.bash
uv run nero-vive-ros --latest-file /run/nero/vive_pose.json
```

If that runtime is a container, mount the endpoint without write access:

```bash
-v /run/nero/vive_pose.json:/run/nero/vive_pose.json:ro
```

| Topic | Type | Frame | Meaning |
|---|---|---|---|
| `/nero/localization/vive/controller_pose` | `geometry_msgs/PoseStamped` | `lighthouse_world` | Fresh raw controller 6-DoF pose |
| `/nero/localization/vive/controller_odometry` | `nav_msgs/Odometry` | parent `lighthouse_world`, child `vive_controller` | Pose plus linear/angular velocity |
| `/nero/localization/vive/tracking` | `std_msgs/Bool` | n/a | Fail-closed validity gate |
| `/nero/localization/vive/diagnostics` | `std_msgs/String` JSON | n/a | Controller ID, sequence, age, source time, error |

Pose/odometry use ROS sensor-data QoS. Validity and diagnostics use reliable
depth-10 QoS. The bridge polls at 100 Hz and publishes each sequence once.

Verify before connecting control:

```bash
ros2 topic hz /nero/localization/vive/controller_pose
ros2 topic echo /nero/localization/vive/tracking
ros2 topic echo --once /nero/localization/vive/diagnostics
```

## Safety and freshness contract

A sample is usable only when all of the following are true:

1. `tracking_valid == true`.
2. `now - transport.received_at <= 0.150 s`.
3. Position/quaternion values are finite and the quaternion is nonzero.
4. Sequence is newer than the last consumed sequence.

The POS receiver now rewrites the endpoint with `tracking_valid=false` after
150 ms without UDP. The ROS bridge independently applies the same deadline.
When validity becomes false, do not extrapolate the last pose into locomotion:
publish/command zero velocity and enter Nima's existing fail-closed stop path.

## Frames and the missing calibration

The raw controller pose is **not yet robot pose**. ROS transform convention here
is `T_parent_child`, mapping coordinates from child into parent:

- `T_lighthouse_controller`: live value from Vive.
- `T_base_controller`: fixed transform measured after rigid mounting.
- `T_map_lighthouse`: fixed alignment from Lighthouse world to the team's floor/map.

The robot body pose Nima needs is:

```text
T_map_base = T_map_lighthouse
             * T_lighthouse_controller
             * inverse(T_base_controller)
```

Until both fixed transforms are calibrated, consume only the raw
`lighthouse_world -> vive_controller` data for visualization and logging. Do not
publish it as `/nero/slam/pose` or `map -> base_link`; that would silently mix
coordinate frames.

After calibration, publish a separate candidate such as
`/nero/localization/vive/base_pose` in frame `map`, then let Nima explicitly
select/fuse it into the navigation pose. Use a distinct TF child such as
`base_link_vive` during validation to avoid fighting the K1 odometry TF.

## Physical robot checklist

1. Rigidly attach the controller to the K1; record mount orientation and offset.
2. Do not cover the controller sensor ring; add USB strain relief.
3. Keep the controller powered on (green LED). Orange means charging only.
4. Move the unpowered/non-walking robot through the work area and verify
   `tracking=true` everywhere in Rerun and ROS.
5. Calibrate `T_base_controller` and `T_map_lighthouse`.
6. Validate stationary noise, motion direction, yaw sign, and loss recovery.
7. Only then connect the calibrated pose to navigation/locomotion.

## Explicitly not provided

- No wireless dongle or wireless controller path.
- No second-controller stream.
- No controller-to-K1 extrinsic yet.
- No Lighthouse-to-map/projector alignment yet.
- No automatic pose fusion or K1 actuation from Vive.
- No promise that raw Lighthouse axes match Nero's `map` axes until calibrated.
