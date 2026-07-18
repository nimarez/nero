# Nero

Nero is a command-driven object-navigation stack for the Booster K1 Geek. A
person says or types `go to the chair`; the robot announces `Going to the chair`,
detects only the requested class in its live RGB-D stream, builds a
visual-inertial world pose with ORB-SLAM3, and drives toward a dynamic `(x, y, yaw)`
approach pose. Object names, target distances, cameras, depth devices, and IMUs
are deliberately not CLI arguments: they are commands or intrinsic K1 hardware.

## What is supported

The physical runtime uses interfaces verified on K1 Geek firmware
`v1.5.0.9-release-0387-2026-01-23`:

| Function | Production interface |
|---|---|
| RGB | `/boostercamera/head/rgb`, NV12, 544×448 |
| Depth | `/boostercamera/head/depth`, `mono16`, 544×448 |
| Intrinsics | `/boostercamera/head/rgb/camera_info` |
| IMU | Official `B1LowStateSubscriber`, approximately 500 Hz |
| Planar odometry | `/odometer_state` |
| Locomotion | Official `B1LocoClient` on loopback |
| Speech | Booster LUI ASR/TTS with `flite`/ALSA playback fallback |
| Detection | Target-conditioned YOLO-World v2 with arbitrary text prompts |
| SLAM | Native ORB-SLAM3 `Sensor.IMU_RGBD` |

Only the interfaces in this table are part of the physical runtime. RGB and depth
must have the same timestamp. IMU samples are synchronized to each pair before
native SLAM.

Startup is fail-closed. Nero requires live RGB, depth, CameraInfo, IMU, odometry,
the open-vocabulary model, ORB vocabulary, robot calibration, walking mode, and the native
IMU-RGBD backend before it enables velocity output. It never changes the real
robot's mode automatically. Keep the area clear and the hardware stop reachable.

## Local development

The repository uses Python 3.10 and `uv`:

```bash
uv sync --all-groups --locked
uv run pytest -q
```

The Booster and ORB-SLAM3 wheels are Linux-only. macOS runs the deterministic
simulator, tests, tooling, and the development RGB-D odometry fallback. The real
K1 and clean Linux Docker test are strict visual-inertial paths.

## One-time real K1 setup

Clone the repository on the robot, then run:

```bash
source /opt/ros/humble/setup.bash
source /opt/booster/BoosterAgent/install/setup.bash
./scripts/setup_k1_runtime.sh
./scripts/setup_object_detector.sh
uv run nero-setup-orbslam
uv run nero-k1-calibration --iface lo --duration 60
```

The prefix commands expose ROS messages and native libraries to the current
shell. `setup_k1_runtime.sh` creates a uv environment that can see those
preinstalled packages. The detector and vocabulary installers
download fixed artifacts and verify their checksums. Calibration reads the live
production ROS intrinsics and RGB-D rate, measures stationary low-state IMU noise,
and combines those measurements with the nominal K1 Geek camera mount. It writes:

- `config/k1_calibration.json`
- `config/k1_orbslam3_imu_rgbd.yaml`

These robot-specific generated files are ignored by Git.

In every new robot shell, source the installed ROS prefixes before using Nero:

```bash
source /opt/ros/humble/setup.bash
source /opt/booster/BoosterAgent/install/setup.bash
```

If the second prefix is absent on a different firmware image, locate the prefix
that provides `booster_interface/msg/Odometer` and source its `setup.bash`.

## Run on the real robot

Put the K1 in walking mode through the supported Booster UI, keep it stationary
for startup, and run:

```bash
export BOOSTER_NET_IF=lo
uv run nero-orb-slam --no-display
```

From this repository on the Mac, open the command terminal and live Rerun viewer
with one command:

```bash
uv run nero-command
```

Enter an open-vocabulary object name such as `chair` at the `object>` prompt. The
command uses your normal SSH authentication, starts a live-only telemetry bridge
on the robot, and removes that bridge when the terminal exits. It does not create
a Rerun recording. Use `quit` to close both the prompt and the viewer, or pass
`--no-rerun` to leave visualization off.

YOLO-World runs asynchronously at a 384-pixel performance default while the
camera and depth topics remain at their native 544x448 resolution. Override the
detector without changing code using `NERO_YOLO_IMGSZ=448` (higher recall, lower
rate), `NERO_YOLO_THREADS`, or `NERO_YOLO_MAX_DETECTIONS` when launching the
robot policy.

The robot announces an accepted command before movement. Nero stops when the
target track expires, a safety check fails, SLAM loses the required state, or the
process receives an interrupt.

The policy does not drive to a scalar distance. Each current 3D observation is
transformed into the SLAM world frame, filtered into an object track, and converted
to a full approach pose facing the object. An internal class-aware safety radius
is used only to construct that pose.

## Commands

All commands run through uv:

| Command | Purpose |
|---|---|
| `uv run nero-orb-slam` | Spoken/typed object navigation on a real K1 |
| `uv run nero-command` | Mac object-command prompt plus live K1 Rerun viewer |
| `uv run nero-sim --demo` | Fast deterministic in-process policy test |
| `uv run nero-booster-studio` | Full policy on a Booster Studio virtual K1 |
| `uv run nero-sim-benchmark` | Compare native SLAM with simulator truth |
| `uv run nero-mapping` | Collect RGB-D/pose frames and invoke COLMAP + gsplat |
| `uv run nero-map-nav --map MAP --initial-pose X Y YAW` | Occupancy-grid navigation on shared IMU-RGBD ORB-SLAM |
| `uv run nero-pc2map CLOUD -o MAP` | Convert a point cloud to an occupancy map |
| `uv run nero-k1-calibration` | Capture real K1 IMU-RGBD calibration |
| `uv run --extra viz nero-rerun` | Bridge normalized Nero ROS topics into Rerun |

`nero-mapping` is a separate reconstruction pipeline. COLMAP and the configured
gsplat training command must already be installed; they are intentionally not
robot runtime dependencies. Object navigation and `nero-map-nav` now share the
same synchronized K1 sensor reader, IMU-RGBD ORB-SLAM localization, pose fusion,
safety monitor, depth obstacle processing, and goal-pose velocity controller.
Only goal selection differs: a live semantic object pose versus A* waypoints in
an occupancy map.

The bundled real main-room splat can be converted without Open3D. The alias
`assets/main_room.ply` resolves to its Git-LFS location:

```bash
git lfs install --local
git lfs pull --include='src/nero/simulation/scenes/main_room/assets/main_room.ply'
uv run nero-pc2map assets/main_room.ply -o output/main_room_map \
  --name main_room --resolution 0.05 --grid-size 40 \
  --up-axis y --height-thresh 0.15 --max-height 2.0
uv run nero-map-nav \
  --map output/main_room_map/main_room.png \
  --yaml output/main_room_map/main_room.yaml \
  --initial-pose X Y YAW --goal X Y YAW
```

`--initial-pose` is the robot's measured pose in the map frame at startup. It is
required for meaningful navigation because a new ORB-SLAM session has an
arbitrary origin; the policy performs rigid SE(2) alignment and never assumes
the captured splat and live SLAM frames already coincide. Inspect the generated
grid before commanding the real robot—the splat projection is a planning input,
not collision-certified geometry.

## Booster Studio

Use `nero-sim` for quick policy tests. Use Booster Studio when physics, vendor
transport, native ORB-SLAM3, or sim-to-real sensor boundaries matter.

Inside the virtual K1 terminal:

```bash
./scripts/setup_k1_runtime.sh
uv run nero-setup-orbslam
./scripts/run_booster_studio.sh
```

The Studio adapter changes only environment and sensors. It consumes live
simulated RGB, 16-bit depth, CameraInfo, IMU, odometry, clock, and detection
topics; velocity still goes through `B1LocoClient`. Simulator truth is isolated
under `/nero/reference` and is never policy input.

Nero enforces the K1 Geek profile: 544×448 RGB and depth at 20 fps, global
shutter, 105°×94° RGB/depth field of view, 0.5–6 m depth range, and 3% depth
accuracy at 1 m. Studio's renderer may run faster, so the adapter decimates exact
RGB-D pairs to 20 fps and rejects a source that is too slow. If
`config/k1_calibration.json` exists, Studio uses it as the expected real sensor
profile; otherwise it uses `config/k1_geek_nominal_calibration.json`.

Multi-robot/topic overrides are listed by:

```bash
uv run nero-booster-studio --help
```

Object names and stand-off distances remain absent from that CLI.

### Furnished living room

The repository contains a small CC0, collision-enabled living room for the
disposable Linux simulator. From the virtual K1 terminal:

```bash
uv run nero-setup-booster-room --activate
```

This backs up the empty K1 scene and simulator transport setting, installs a
calibrated K1 model and furnished room, and enables ROS sensor transport. Restart
the virtual robot or switch away from and back to the empty K1 scene. Restore the
original files with:

```bash
uv run nero-setup-booster-room --restore
```

Pass `--sensor-calibration config/k1_calibration.json` to stage the measured real
profile explicitly. Use `--sim-root` or `BOOSTER_STUDIO_SIM_ROOT` if auto-detection
fails. The installer refuses to modify the signed macOS application bundle.

The large `industrial_storage_room` and `main_room` splat/collider pairs are
reference scene assets stored in Git LFS. They are not automatically activated by
`nero-setup-booster-room`; see their adjacent READMEs before integrating them.

### Native reference benchmark

With a visible scene and the virtual K1 in walking mode:

```bash
./scripts/run_booster_benchmark.sh
```

The default trajectory is conservative and always sends zero velocity in cleanup.
Reports in `output/sim_benchmark/` include tracking coverage, rigid-SE(2)-aligned
ATE/RPE, yaw error, metric scale drift, and symmetric point-cloud error. Alignment
does not rescale the estimate, so RGB-D scale errors remain visible.

## ROS 2 and Rerun

Hardware, Studio, mapping, and benchmark runtimes publish normalized telemetry:

| Namespace | Contents |
|---|---|
| `/nero/sensors` | RGB, metric depth, CameraInfo, IMU, odometry, joint states |
| `/nero/slam` | Pose, path, tracking state, map points |
| `/nero/navigation` | Requested detections, object/goal poses, status, velocity |
| `/nero/reference` | Simulator-only truth for visualization and benchmarks |

Control never consumes `/nero/reference`. Pass `--no-ros-observability` to a
supported agent to disable publication.

On a ROS-equipped machine with a desktop, one command subscribes to every Nero
topic and opens Rerun:

```bash
uv run --extra viz nero-rerun
```

The bridge plots RGB, metric depth, camera calibration, IMU orientation/rates,
odometry, every available joint position/velocity/effort, SLAM pose/path/map,
detection boxes/labels/confidence/depth centroids, goals, commands, and simulator
reference data. Print the exact
subscription contract without requiring ROS or Rerun to be installed:

```bash
uv run nero-rerun --print-topics
```

On the macOS host, start the viewer:

```bash
./scripts/run_rerun_viewer.sh
```

In a Studio container, start the bridge with its default host address:

```bash
./scripts/run_rerun_bridge.sh
```

On a physical K1, point the bridge at the Mac's reachable IP instead:

```bash
NERO_RERUN_URL=<mac-ip>:9876 ./scripts/run_rerun_bridge.sh
```

To record without a viewer:

```bash
uv run --extra viz nero-rerun --save output/nero.rrd
```

Live viewing and `--connect` do not write a bag to disk. With no sink option,
`nero-rerun` spawns a local viewer. Use `--connect` for the
split robot-to-workstation setup or `--save` for a recording. Rerun is an optional
uv extra and is not installed in the headless robot runtime.

## Why not `booster_deploy`?

Nero's navigation policies depend on a transport-neutral `RobotAdapter`. The real
adapter uses Booster's high-level walking controller; the Studio adapter provides
the same contract from simulated sensors. Booster's
[`booster_deploy`](https://github.com/BoosterRobotics/booster_deploy) framework is
a joint-space policy loop built around low-state input, low-command output, and a
custom-mode prepare state. Running it beside the walking controller would create
competing locomotion owners. It is an alternative future adapter for custom gait
policies, not a requirement for RGB-D navigation or sim-to-real sensor parity.

## Testing

Run the complete local suite:

```bash
uv sync --all-groups --locked
uv run ruff check .
uv run pytest -q
uv build --no-sources
```

Exercise the Linux vendor wheels and native inertial binding in Docker:

```bash
./scripts/test_docker.sh
```

The Docker test installs the official Booster and ORB-SLAM3 wheels, verifies the
ORB vocabulary checksum, runs the suite, initializes `Sensor.IMU_RGBD`, submits a
synchronized RGB-D/IMU frame, and shuts the system down. Set
`NERO_DOCKER_PLATFORM` to override the target architecture.
