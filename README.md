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
sudo apt-get install libopengl0 libglx0 libglu1-mesa libsm6 libice6 \
  libx11-6 libxext6 libegl1 libgl1
```

## One-time K1 setup

Run these commands on the robot, with the robot stationary during calibration:

```bash
uv run nero-setup-orbslam
uv run nero-k1-calibration --iface lo --duration 60
```

The first command downloads and verifies the vocabulary from the official
ORB-SLAM3 repository. The second reads the live camera intrinsics and factory
frame geometry from the K1, then estimates IMU noise from the stationary sample.
It produces robot-specific files under `config/`; these are intentionally ignored
by git. Set `BOOSTER_NET_IF` if DDS uses an interface other than `lo`.

Then start an agent normally, for example:

```bash
uv run nero-orb-slam
```

The SLAM wrapper subscribes to the K1 IMU itself and synchronizes samples to each
RGB-D frame. A native inertial frame without IMU samples is marked lost rather
than silently processed as visual-only SLAM. Linux/K1 initialization is strict:
missing native libraries, vocabulary, or robot calibration is an error. The
RGB-D odometry fallback is automatic only on non-Linux development machines.
