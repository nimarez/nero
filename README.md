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
| Detection | QNN HTP target-conditioned YOLO-World v2 with arbitrary text prompts |
| SLAM | Native ORB-SLAM3 `Sensor.IMU_RGBD` |

Only the interfaces in this table are part of the physical runtime. RGB and depth
must arrive within the 20 ms synchronization tolerance. IMU samples are
synchronized to each pair before native SLAM.

Startup is fail-closed. Nero requires live RGB, depth, CameraInfo, IMU, odometry,
the configured detector, ORB vocabulary, robot calibration, walking mode, and
the native IMU-RGBD backend before it enables velocity output. It never changes
the real robot's mode automatically. Keep the area clear and the hardware stop
reachable.

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
./scripts/setup_qnn_runtime.sh
```

From the Mac checkout, install the private generated artifact with the configured
AI Hub client and transfer it:

```bash
uv run --extra ai-hub nero-install-qnn-model
uv run nero-deploy-qnn-model
```

Back on the robot, finish the sensor/runtime setup:

```bash
./scripts/setup_object_detector.sh
uv run nero-setup-orbslam
uv run nero-k1-calibration --iface lo --duration 60
```

The prefix commands expose ROS messages and native libraries to the current
shell. `setup_k1_runtime.sh` creates a uv environment that can see those
preinstalled packages. The QNN setup creates an isolated Python 3.11
`.venv-qnn` with the official Linux ARM64 `onnxruntime-qnn` plugin. Nero's main
Python 3.10 process keeps the Booster and ORB-SLAM wheels and talks to one
persistent QNN worker, so the graph is compiled only once. Detector setup
verifies the pinned AI Hub graph and resolves the text encoder before the live
loop. Calibration reads the live production
ROS intrinsics and RGB-D rate, measures stationary low-state IMU noise, and
combines those measurements with the nominal K1 Geek camera mount. It writes:

- `config/k1_calibration.json`
- `config/k1_orbslam3_imu_rgbd.yaml`

These robot-specific generated files are ignored by Git.

The K1 setup also makes the vendor RGB-D bridge persistent by setting
`EnableCameraBridge: true` and enabling `booster-daemon-perception.service`.
It preserves the original vendor configuration as
`/opt/booster/perception_info.yaml.nero-backup` and asks for a reboot; it does
not restart perception inside the active SSH session because Booster resets its
USB controller during that operation.

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

To use the same object-command policy with the fixed main-room map, add the map
instead of launching a different policy:

```bash
uv run nero-orb-slam --no-display \
  --map output/main_room_map/main_room.png \
  --map-yaml output/main_room_map/main_room.yaml
```

From this repository on the Mac, open the command terminal and live Rerun viewer
with one command:

```bash
uv run nero-command
```

Enter any target description, such as `green can`, at the `object>` prompt. The
command uses your normal SSH authentication. If the robot policy is absent, it
starts `nero-orb-slam --no-display`, waits for its command socket, and prints the
robot-side policy log if startup fails. It also starts a live-only telemetry
bridge and removes that bridge when the terminal exits; the navigation policy is
left running. It does not create a Rerun recording. Use `quit` to close both the
prompt and the viewer, or pass `--no-rerun` to leave visualization off. SSH
connection and keepalive failures are bounded, and a command that the robot
policy does not acknowledge returns after five seconds; tune that with
`--ack-timeout` if needed.

Safety enforcement is enabled by default. For controlled testing only, pass
`--disable-safety`; Nero continues computing and publishing tilt, depth,
obstacle, and battery diagnostics, but those conditions no longer veto motion:

```bash
uv run nero-command --policy pure-pursuit --disable-safety
```

The flag is applied when the robot policy starts. To prevent accidental reuse
in the wrong mode, `nero-command` refuses to attach to an existing policy whose
safety mode differs from the requested mode; stop that policy and run the
command again. Sensor, localization, locomotion-command, and shutdown failures
still stop the robot.

Before starting a missing policy, `nero-command` now waits for actual 544×448
RGB, depth, and CameraInfo messages and a synchronized RGB-D pair. DDS publisher
discovery alone does not pass this gate. The camera and policy startup windows
default to 120 and 240 seconds respectively because Booster's first-launch
perception sequence can take more than a minute and the isolated QNN worker has
its own 180-second startup ceiling. Override them with
`--camera-start-timeout` and `--policy-start-timeout`.

The vendor bridge's maximum registered RGB-D mode is 544×448. Nero consumes
those frames at their native published resolution and performs no additional
downsampling before ArUco, depth processing, SLAM, or Rerun. The physical robot
adapter independently verifies RGB, depth, and CameraInfo dimensions during every
policy startup, so launching a policy directly cannot bypass the maximum-resolution
preflight. A lower or mismatched mode fails before walking is armed.

To keep the complete runtime on the physical robot and view it from any browser
on the robot network, SSH into the K1 and run:

```bash
cd /home/booster/Workspace/nero
./scripts/run_robot_web.sh \
  --policy pure-pursuit \
  --disable-safety \
  --object-backend aruco \
  --aruco-map config/aruco_markers.json \
  --aruco-dictionary DICT_4X4_50
```

The command performs the live RGB-D preflight, starts the policy and local
ROS-to-Rerun bridge, and leaves the object prompt in the SSH terminal. Open
`http://10.2.1.130:8080/rerun` in a browser to see the live dashboard. Rendering
happens in the browser; the robot serves the viewer and buffers up to 256 MB of
recent telemetry. Ports 8080 and 9877 must be reachable from the browser; 8081
is used only between the robot-local gateway and Rerun. No Mac-side Nero or
Rerun process is required.
The checked-in marker map assigns `DICT_4X4_50` ID 45 to the command
`marker 45`. Its physical 90 mm size needs no runtime flag because Nero obtains
the marker's metric position from registered K1 depth.

To diagnose the camera without launching navigation or issuing locomotion
commands, run this on the robot after sourcing ROS:

```bash
uv run nero-k1-preflight --timeout 120
```

If raw RGB is live but aligned RGB/depth are silent, use Booster's supported
recovery command and expect the SSH connection to drop temporarily:

```bash
booster-cli launch -c restart -m perception
```

For deterministic tagged-object navigation instead of YOLO, copy the example
marker map and start the same command interface with ArUco options:

```bash
cp config/aruco_markers.example.json config/aruco_markers.json
uv run nero-command --object-backend aruco \
  --aruco-map config/aruco_markers.json
```

The map path is robot-side. The marker detector uses the same K1 RGB-D stream,
goal-pose controller, ROS detection topic, and Rerun overlay. See
[`docs/aruco_navigation.md`](docs/aruco_navigation.md) for marker setup details.

The real Linux ARM64 K1 defaults to `yolo-world-qnn`: a 256-pixel visual graph on
the QCS8550 HTP/NPU and an exact CLIP text embedding computed only when the
command changes. Session creation sets `session.disable_cpu_ep_fallback=1` and
also disables ONNX Runtime fallback, so a missing provider or partially
unsupported graph prevents detector initialization and therefore prevents
motion. It never silently becomes a CPU workload.

The older CPU YOLO-World backend remains an explicit diagnostic fallback:

```bash
export NERO_OBJECT_BACKEND=yolo-world
./scripts/setup_object_detector.sh
```

That CPU backend runs in a separate process on K1 so ORB-SLAM3 cannot starve
inference. It reserves the two fastest available cores for detection and leaves
the remaining allowed cores to camera, SLAM, ROS, and control. Disable this only
for diagnostics with `NERO_CPU_PARTITION=0`; disable process isolation with
`NERO_DETECTOR_PROCESS=0`.

### Modal GPU perception path

Modal is an optional remote perception backend. It sends one JPEG at a time to
an authenticated L4-backed YOLO-World endpoint and returns only 2D boxes. Depth
projection, object/world tracking, SLAM, safety, and control remain on the K1.
Keeping IMU-RGBD SLAM local avoids putting the 20 Hz localization critical path
behind network jitter or a serverless cold start.

Deploy the endpoint from the Mac and create a proxy token:

```bash
uvx modal setup
uvx modal deploy deploy/modal_perception.py
uvx modal workspace proxy-tokens create
```

The deploy command prints the `detect` URL. Configure the robot with that URL
and the proxy token pair, then validate and warm the endpoint before starting
navigation:

```bash
export NERO_OBJECT_BACKEND=yolo-world-modal
export NERO_MODAL_URL=https://YOUR-WORKSPACE--nero-perception-yoloworldendpoint-detect.modal.run
export NERO_MODAL_KEY=wk-...
export NERO_MODAL_SECRET=ws-...
./scripts/setup_object_detector.sh
uv run nero-orb-slam --no-display
```

The endpoint is proxy-authenticated; do not commit the token values. Tune the
request with `NERO_MODAL_TIMEOUT` (default `120` seconds),
`NERO_MODAL_JPEG_QUALITY` (default `85`), and the usual
`NERO_OBJECT_IMGSZ`/`NERO_YOLO_MAX_DETECTIONS` settings. Set
`NERO_MODAL_WARMUP=0` only for diagnostics; the default startup warmup makes an
unreachable endpoint fail before the policy enables motion.

YOLOE-26n is an optional open-vocabulary alternative. Install and run it with:

```bash
NERO_OBJECT_BACKEND=yoloe ./scripts/setup_object_detector.sh
export NERO_OBJECT_BACKEND=yoloe
uv run nero-orb-slam --no-display
```

The measured K1 result should decide the backend: YOLOE-26n was approximately
0.5 seconds per standalone frame in the initial robot test, while YOLO-World at
256 pixels was approximately 0.3 seconds. YOLOE therefore remains opt-in rather
than becoming the default. A fixed-vocabulary OpenCV fallback exists for
diagnostics, but it is not used by the object-command policy.

### Qualcomm AI Hub NPU path

Nero can export YOLO-World with two inputs: a 256-pixel RGB tensor and one
512-dimensional text feature. The class count stays fixed at one because the
policy pursues one requested target, while the text feature remains a runtime
input. This preserves arbitrary spoken targets; unlike a normal YOLO export, it
does not bake a fixed vocabulary into the graph.

Configure the free Qualcomm AI Hub client outside the repository, then run:

```bash
uv sync --extra ai-hub
uv run nero-export-yolo-world
uv run nero-ai-hub-profile
uv run nero-install-qnn-model
```

Credentials belong in `~/.qai_hub/client.ini`. A repo-local `.env.aihub` is
ignored as a convenience, but must never be committed. Generated models and
profiles are ignored under `config/` and `output/`.

The first QCS8550 proxy profile of the runtime-prompt graph completed with every
reported operation on the NPU: 1.495 ms model inference, 83,132,416 bytes peak
inference memory, 1.171 seconds first load, and 357 ms warm load. AI Hub proxy
timings are a model microbenchmark, not an end-to-end K1 guarantee; preprocessing,
CLIP encoding when a command changes, postprocessing, camera transfer, SLAM, and
robot scheduling remain outside that figure. The corresponding jobs were
`jgjwyj175` (compile) and `jp1v1r3kp` (profile).

The runtime artifact is pinned to AI Hub target model `mn09g883n`; both ONNX
files are size- and SHA-256-verified before installation. Generated model data
is intentionally ignored by Git. Transfer it to the robot over one non-TTY SSH
stream after the robot has pulled the current source:

```bash
uv run nero-deploy-qnn-model
```

Then verify the real provider, full-graph HTP placement contract, prompt encoder,
input/output shapes, and warm latency on the K1 without moving it:

```bash
./scripts/setup_qnn_runtime.sh
uv run nero-qnn-smoke --target "green can" --runs 20
```

The setup command is idempotent and pins ONNX Runtime 1.26.0 with the official
QNN plugin 2.4.0. The worker disables CPU fallback for the session and binds the
registered Qualcomm device explicitly to the packaged HTP backend. The K1's
older `/dev/adsprpc-smd` device name is handled automatically. Override the
worker interpreter only for diagnostics with `NERO_QNN_WORKER_PYTHON`.

Override CPU open-vocabulary inference using `NERO_OBJECT_IMGSZ` (higher values
trade rate for recall), `NERO_YOLO_THREADS`, or `NERO_YOLO_MAX_DETECTIONS`.
The pinned QNN graph is fixed at 256 pixels and rejects any other size.
`NERO_OBJECT_MODEL` can point at a non-default CPU checkpoint. The older
`NERO_YOLO_IMGSZ` name remains supported. Measure the exact live workload
without commanding robot motion using:

```bash
uv run nero-perception-benchmark --live --with-slam --target "green can"
```

This benchmark reports both completed detector round-trip latency and native
SLAM frame latency, so the slower path can be identified on the actual robot.

The robot announces an accepted command before movement. With the default
safety enforcement, Nero stops when the target track expires, a safety check
fails, SLAM loses the required state, or the process receives an interrupt.

The policy does not drive to a scalar distance. Each current 3D observation is
transformed into the SLAM world frame, filtered into an object track, and converted
to a full approach pose facing the object. An internal class-aware safety radius
is used only to construct that pose.

## Commands

All commands run through uv:

| Command | Purpose |
|---|---|
| `uv run nero-orb-slam` | Spoken/typed object navigation on a real K1 |
| `uv run nero-pure-pursuit` | Same command/arrival loop using direct RGB-D pursuit, without SLAM or a map |
| `uv run nero-command` | Mac object-command prompt plus live K1 Rerun viewer |
| `uv run nero-sim --demo` | Fast deterministic in-process policy test |
| `uv run nero-booster-studio` | Full policy on a Booster Studio virtual K1 |
| `uv run nero-sim-benchmark` | Compare native SLAM with simulator truth |
| `uv run nero-perception-benchmark` | Non-moving detector benchmark, optionally under live SLAM load |
| `uv run nero-export-yolo-world` | Export one-target YOLO-World with a runtime text embedding |
| `uv run nero-ai-hub-profile` | Compile/profile the runtime-prompt graph on the QCS8550 proxy |
| `uv run nero-install-qnn-model` | Download and checksum-verify the pinned AI Hub QNN graph |
| `uv run nero-deploy-qnn-model` | Transfer and re-verify the generated QNN graph on the K1 |
| `uv run nero-qnn-smoke` | Fail-closed real-K1 QNN provider and latency smoke test |
| `uv run nero-mapping` | Collect RGB-D/pose frames and invoke COLMAP + gsplat |
| `uv run nero-orb-slam --map MAP` | Same object policy with map alignment and A* routing |
| `uv run nero-map-nav --map MAP` | Explicit pose-goal CLI using the unified policy |
| `uv run nero-pc2map CLOUD -o MAP` | Convert a point cloud to an occupancy map |
| `uv run nero-k1-calibration` | Capture real K1 IMU-RGBD calibration |
| `uv run --extra viz nero-rerun` | Bridge normalized Nero ROS topics into Rerun |
| `./scripts/run_robot_web.sh` | Run a physical policy plus browser-hosted Rerun on the robot |
| `uv run nero-vive-ros` | Publish jscore's fail-closed Vive pose into ROS 2 |

`nero-mapping` is a separate reconstruction pipeline. COLMAP and the configured
gsplat training command must already be installed; they are intentionally not
robot runtime dependencies. There is one `NavigationPolicy`. Without a map, its
world frame is the live SLAM session and it drives directly toward the
object-derived goal pose. With a map,
an optional `MapNavigator` aligns that same SLAM session to the fixed map and
routes the same goal through A*. `nero-map-nav` is only an explicit pose-goal
front end; it does not own a second sensor, safety, localization, or control loop.

`nero-pure-pursuit` is the intentionally smaller alternative for visible-object
goals. It uses the target's live camera-frame RGB-D position as the pursuit point,
keeps the tilt/depth/battery safety gate, and stops at the same class-aware
stand-off distance. While acquiring or reacquiring a target, the walking base stays
stopped and the K1 head performs a 15-pose pan/tilt raster. A side-looking detection
must be confirmed, the head is then centered, and the body only rotates in place
until a fresh centered detection permits pursuit. Because it has no persistent world
pose, it cannot route around occlusions or map obstacles.
It publishes the same sensor, detection, status, and command topics as the SLAM
policy, so Rerun still shows the live RGB/depth images, labeled detection boxes,
3D camera-frame centroids, and commanded velocities. A world-frame route is
intentionally absent because this controller does not claim to know one.

To have the Mac command interface start or reuse this policy on the robot:

```bash
uv run nero-command --policy pure-pursuit
```

Add `--object-backend aruco --aruco-map config/aruco_markers.json` to the same
command for marker-based pursuit. `nero-command` refuses to silently switch if
the other navigation policy already owns the command socket; stop that process
first so two controllers can never compete for walking control.

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
  --goal X Y YAW
```

Map and SLAM information are fused without feeding a possibly imperfect mesh
back into visual-inertial odometry. The policy estimates a rigid
`T_map_slam`, evaluates the live fused SLAM pose in the map frame, uses static
occupancy for global A* planning, and applies current depth obstacles as the
local safety layer. Object detections are transformed through the same
`T_map_slam`, so object goals, robot poses, SLAM points, and Rerun telemetry stay
in one coordinate frame. This keeps short-term motion smooth if map matching is
temporarily ambiguous. Continuous drift correction should later add only
confidence-gated, low-rate SE(2) constraints to `T_map_slam`; it should not
replace or hard-reset ORB-SLAM poses while walking.

A new ORB-SLAM session has an arbitrary origin, so the policy performs rigid
SE(2) alignment and never assumes the captured splat and live SLAM frames
already coincide. By default the startup pose in the map frame is localized
automatically: depth frames are back-projected into gravity-aligned obstacle
scans and matched against the fixed map with a coarse-to-fine correlative
search that penalizes scan rays crossing mapped structure (see
`nero.navigation.global_localization`). Because a single camera view is often
ambiguous, the policy stays in `LOCALIZING` and remains stopped until it has a
goal. After a goal is accepted it may slowly spin in place
(`--localization-spin-speed`, 0 disables this) and accumulate scans across
viewpoints in the SLAM session frame until the composite match is both strong
and unambiguous. Pass `--initial-pose X Y YAW` to skip
auto-localization and use a measured map-frame pose, and `--camera-height` if
the depth camera is not ~1.1 m above the floor. Inspect the generated grid
before commanding the real robot—the splat projection is a planning input,
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
| `/nero/navigation` | Detections, object/goal poses, controller plan, status, velocity |
| `/nero/reference` | Simulator or calibrated external truth for visualization and benchmarks |

Control never consumes `/nero/reference`. Pass `--no-ros-observability` to a
supported agent to disable publication.

On a ROS-equipped machine with a desktop, one command subscribes to every Nero
topic and opens Rerun:

```bash
uv run --extra viz nero-rerun
```

The bridge plots RGB, metric depth, camera calibration, IMU orientation/rates,
odometry, every available joint position/velocity/effort, SLAM pose/path/map,
detection boxes/labels/confidence/depth centroids, goals, the green controller
plan, commands, and simulator or calibrated Vive reference data. Navigation
geometry includes the class-aware stand-off circle, target-bearing ray, labeled
stand-off waypoint, and a two-second linear/yaw command preview. The radius is
cyan while approaching, green while holding, and red when inside the protected
distance. A separate robot-centered clearance ring and Safety and State panel
show SAFE/WARNING/E-STOP, obstacle distance, depth blindness, left/center/right
corridor availability, roll/pitch, battery percentage, and the reason for any
stop. Print the exact
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
