# Nero

Nero obeys object-navigation directions such as "go to the chair," announces
"Going to the chair" over the Booster K1 speaker, and then uses the built-in
RGB-D camera to detect, track, and approach only that requested target. Camera,
depth, and IMU inputs are internal K1 capabilities; agents do not accept sensor
handles, object names, or target distances as CLI arguments.

## Environment

The project uses `uv` and Python 3.10:

```bash
uv sync --all-groups
```

The official Booster and ORB-SLAM3 Python wheels are Linux-only, so `uv` installs
them only on Linux. macOS can run tests and the visual RGB-D fallback, but native
SLAM on a K1 is always initialized as `Sensor.IMU_RGBD`.

On a minimal Debian/Ubuntu image, install the shared libraries linked by the
native ORB wheel (verified in a clean Linux ARM64 Docker container):

```bash
sudo apt-get install libopengl0 libglx0 libglu1-mesa libglib2.0-0 libsm6 libice6 \
  libx11-6 libxext6 libegl1 libgl1
```

## One-time K1 setup

Run these commands on the robot, with the robot stationary during calibration:

```bash
uv run nero-setup-orbslam
uv run nero-k1-calibration --iface lo --duration 60
```

The first command downloads and verifies the vocabulary from the official
ORB-SLAM3 repository. The second reads the live camera resolution, intrinsics,
delivered frame rate, depth scale, and factory frame geometry from the K1, then
measures IMU frequency and estimates noise from the stationary sample.
It produces robot-specific files under `config/`; these are intentionally ignored
by git. Set `BOOSTER_NET_IF` if DDS uses an interface other than `lo`.

Then start an agent normally, for example:

```bash
uv run nero-orb-slam
```

All Nero commands run through the uv-managed environment:

```bash
uv run nero-orb-slam
uv run nero-sim
uv run nero-booster-studio
uv run nero-mapping
uv run nero-map-nav --map maps/office.npy --goal 3.5 2.0
uv run nero-pc2map scan.ply -o maps/office
uv run nero-k1-calibration --iface lo --duration 60
uv run nero-setup-orbslam
```

The SLAM wrapper subscribes to the K1 IMU itself and synchronizes samples to each
RGB-D frame. A native inertial frame without IMU samples is marked lost rather
than silently processed as visual-only SLAM. Linux/K1 initialization is strict:
missing native libraries, vocabulary, or robot calibration is an error. The
RGB-D odometry fallback is automatic only on non-Linux development machines.

## Booster Studio virtual K1

`nero-sim` is the fast, in-process deterministic test environment. For a
physics and sensor integration test, start a **K1 virtual robot** in Booster
Studio, switch it to WALK mode, and use its Linux robot terminal:

```bash
./scripts/setup_booster_studio.sh
uv run nero-setup-orbslam
./scripts/run_booster_studio.sh
```

The setup script creates uv's `.venv` with system-site-package access because
ROS 2 and Booster's SDK are part of the virtual K1 image rather than packages in
PyPI. Nero's own dependency graph remains locked and installed by `uv sync`.
The run wrapper sources Studio's ROS 2 environment, then executes
`uv run nero-booster-studio`; pass normal Nero flags to the wrapper.

The Booster Studio command runs the same command-driven object targeting,
`IMU_RGBD` SLAM, obstacle processing, navigation, and safety policy used by
`nero-orb-slam`. Only the environment and command adapters change. It consumes
the simulator's live RGB, 16-bit depth, CameraInfo, IMU, and odometry topics,
synchronizes RGB-D and IMU on a shared receipt-time clock, derives simulator
calibration from live intrinsics plus the K1 MJCF camera mount, and sends velocity
through the official Booster locomotion client on `127.0.0.1`.
The simulator localization topic is isolated as benchmark-only ground truth; it
is never exposed to the navigation policy as odometry.
It verifies that RGB, depth, and CameraInfo dimensions agree, requires the K1
simulator's 16-bit millimetre depth format, and measures rendered camera and IMU
rates before generating the ORB-SLAM settings.

The staged Geek sensor profile is 544x448 RGB and depth at 20 fps, global shutter,
105x94 degree RGB/depth field of view, and a 0.5-6 m depth range. Scene activation
creates a vendor-model copy with that resolution and vertical FOV. Studio's
container relay has no host-renderer control channel, so Nero deterministically
decimates exact synchronized RGB-D pairs to the delivered 20 fps contract and
rejects a source that is too slow. A calibration captured from the real robot
takes precedence over the nominal profile.

The default single-robot topics match Booster Studio's installed K1 simulator.
For a named/multi-robot scene, override them with `--rgb-topic`, `--depth-topic`,
`--camera-info-topic`, `--imu-topic`, `--pose-topic`, and `--detections-topic`.
Object names and target distances remain intentionally absent from the CLI:
enter a direction such as `go to the chair` at the runtime prompt, and the
simulated speaker acknowledges it before motion begins. The physical K1 uses the
official Booster LUI ASR service for the same command contract and falls back to
terminal input if ASR initialization fails. Nero ignores unrelated speech and
does not announce every object it sees. Each fresh matching 3D observation is
transformed into the SLAM world frame and filtered into an object track. Nero
derives a dynamic `(x, y, yaw)` approach pose that faces the object from an
internal safety radius; both position and heading must converge.
Brief occlusions use the last world-frame goal, never stale camera-relative
coordinates, and an expired track stops the robot.

### Sim-reference benchmark

With the furnished scene visible and the virtual K1 in WALK mode, run the native
localization and mapping benchmark from its terminal:

```bash
./scripts/run_booster_benchmark.sh
```

The conservative built-in trajectory always stops in a `finally` block. Reports
land in `output/sim_benchmark/` and include tracking coverage, rigid-SE(2)-aligned
ATE and RPE, yaw error, metric scale drift, and symmetric point-cloud geometry
scores. Alignment never rescales the trajectory, so RGB-D scale errors remain
visible. The simulator reference pose is used only by this evaluator.

### Policy and ROS boundary

Navigation and mapping policies depend on Nero's transport-neutral `RobotAdapter`.
The physical K1 implementation uses the high-level vendor walking controller; the
Studio implementation supplies the same contract from ROS 2 sensor/state topics
and sends body-velocity commands through `B1LocoClient`. Nero intentionally does
not run `booster_deploy` beside that controller: its joint-space policy loop enters
custom control mode and publishes motor targets, so it is an alternative gait
backend rather than a sensor/navigation framework. It can be added later behind
the same adapter if a custom locomotion policy is required.

### ROS 2 and Rerun observability

The hardware, Studio, mapping, and benchmark runtimes publish a common ROS 2
observability contract by default. Sensors live under `/nero/sensors`, fused
SLAM pose/path/map data under `/nero/slam`, detections and commands under
`/nero/navigation`, and simulator-only truth under `/nero/reference`. The
reference namespace is visualization and benchmark data only; policies never
consume it. Pass `--no-ros-observability` to an agent to disable this layer.

Start the viewer on the macOS host:

```bash
./scripts/run_rerun_viewer.sh
```

Then start the bridge in a Booster Studio terminal:

```bash
./scripts/run_rerun_bridge.sh
```

Run `nero-booster-studio`, `nero-mapping`, or `nero-sim-benchmark` in another
Studio terminal. Rerun will show synchronized RGB and metric depth, intrinsics,
IMU plots, estimated and reference transforms/paths/map points, detections,
the tracked world-space object, dynamic goal pose, tracking and navigation
status, and velocity commands. To record without a live viewer, run
`uv run --extra viz nero-rerun --save output/nero.rrd` inside the ROS environment.
The Rerun dependency is an optional uv extra so headless robot runtimes do not
install visualization packages.

### Furnished-room scene

Nero includes a small, CC0, collision-enabled living room for testing RGB-D SLAM
and obstacle handling somewhere more representative than an empty soccer pitch.
From the **virtual robot terminal**, install it into Booster Studio's disposable
simulator container and activate it as the empty K1 scene:

```bash
uv run nero-setup-booster-room --activate
```

Activation also changes the disposable simulator from its default shared-memory
motion transport to its supported ROS transport. Nero automatically consumes the
K1 IMU from `/imu/data` or the legacy robot-specific topic, depending on which
one the installed Studio version publishes. The original container setting is
backed up and restored together with the scene.

Then restart the virtual robot, or switch away from and back to the empty K1
scene. The room contains walls, a couch, chairs, a coffee table, cabinets,
shelves, and a red ball. The ball deliberately retains Booster Studio's special
`ball` body name so its built-in simulated detector can exercise Nero's spoken
command-driven flow. Restore the original scene with:

```bash
uv run nero-setup-booster-room --restore
```

Use `--sim-root` or `BOOSTER_STUDIO_SIM_ROOT` if the simulator source directory
is not auto-detected. The installer refuses to modify the signed macOS app; scene
activation is intentionally limited to the disposable Linux virtual robot.

## Testing

Run the complete local suite through uv:

```bash
uv sync --all-groups --locked
uv run ruff check .
uv run pytest -q
uv build --no-sources
```

Run the native Linux ARM64 path in Docker:

```bash
./scripts/test_docker.sh
```

The Docker test installs the actual Linux Booster and ORB-SLAM3 wheels, verifies
the official vocabulary checksum, runs the full test suite, initializes
`Sensor.IMU_RGBD`, submits synchronized RGB-D and IMU data, and shuts the native
system down. Override `NERO_DOCKER_PLATFORM` when testing another Linux target.
