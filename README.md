# Nero

Nero detects objects through the Booster K1's built-in RGB-D camera, announces
them over the robot speaker, waits for human confirmation, and then follows the
confirmed object. Camera, depth, and IMU inputs are internal K1 capabilities;
agents do not accept sensor handles, object names, or target distances as CLI
arguments.

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
uv run nero-booster-studio
```

The setup script creates uv's `.venv` with system-site-package access because
ROS 2 and Booster's SDK are part of the virtual K1 image rather than packages in
PyPI. Nero's own dependency graph remains locked and installed by `uv sync`.

The Booster Studio command runs the same object detection, spoken announcement,
human confirmation, `IMU_RGBD` SLAM, obstacle processing, navigation, and safety
policy used by `nero-orb-slam`. Only the environment adapter changes. It consumes
the simulator's live RGB, 16-bit depth, CameraInfo, IMU, and localization topics,
synchronizes RGB-D and IMU on a shared receipt-time clock, derives simulator
calibration from live intrinsics plus the K1 MJCF camera mount, and sends velocity
through the official Booster locomotion client on `127.0.0.1`.
It verifies that RGB, depth, and CameraInfo dimensions agree, requires the K1
simulator's 16-bit millimetre depth format, and measures rendered camera and IMU
rates before generating the ORB-SLAM settings.

The default single-robot topics match Booster Studio's installed K1 simulator.
For a named/multi-robot scene, override them with `--rgb-topic`, `--depth-topic`,
`--camera-info-topic`, `--imu-topic`, `--pose-topic`, and `--detections-topic`.
Object names and target
distances remain intentionally absent from the CLI: detections are live, the
simulated speaker announces each candidate in the terminal, and a human confirms
before motion begins.

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
confirmation flow. Restore the original scene with:

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
