# Vive / Lighthouse pose tracking → Rerun

External 6-DoF pose tracking at ~280 Hz, sub-centimetre, **independent of
ORB-SLAM3**. Useful as a ground-truth reference for SLAM evaluation, and as a
live 3D view in Rerun.

Distinct from `nero.observability.rerun_bridge`, which visualises the robot's own
ROS 2 sensor topics. This module adds an *external* position reference the robot
cannot produce itself.

---

## Quick start

Assumes libsurvive is built and a tracker is plugged in (see setup below).

**1 — Install the viewer extra:**

```bash
uv sync --extra viz          # installs rerun-sdk 0.22.1
```

**2 — On your laptop, start the Rerun viewer:**

```bash
rerun --port 9876
```

> The viewer must be **the same version as the SDK (0.22.1)**. A newer viewer
> will not talk to a 0.22.1 stream.

**3 — Open a reverse tunnel to the host the tracker is wired to:**

```bash
ssh -R 9876:localhost:9876 user@host
```

**4 — In that shell on the host, stream poses:**

```bash
export PYTHONPATH=$HOME/src/libsurvive/bindings/python
export LD_LIBRARY_PATH=$HOME/src/libsurvive/bin
uv run nero-vive-rerun --lighthouse-count 1
```

`pysurvive` ships inside the libsurvive source tree rather than on PyPI, so those
two env vars are required even though `rerun` now comes from the project venv.

You should see, and the same poses appear in the viewer:

```
INFO nero.vive.rerun_stream: logged 100 poses | WW0 at x=-0.430 y=+0.664 z=+0.077 m
```

**In Rerun:** `world/WW0` is the tracker (axes + origin dot), `world/LH0` the
base station, `world/trails/WW0` its path. The scene is only ~1 m across — use
the viewport's recenter control if it looks empty.

## Raspberry Pi → jscore UDP pose transport

For the robot-mounted controller, libsurvive runs on the Raspberry Pi and sends
poses over the private `NERA-WIFI` link. This avoids USB/IP and keeps the optical
tracking loop beside the controller.

On `jscore` (`10.77.0.1`), start the receiver:

```bash
uv run nero-vive-udp-receive --bind 10.77.0.1 --port 43100 --json
```

On the Pi, with this repo and libsurvive installed:

```bash
export PYTHONPATH=$PWD/src:$HOME/src/libsurvive/bindings/python
export LD_LIBRARY_PATH=$HOME/src/libsurvive/bin
uv run nero-vive-udp-send --host 10.77.0.1 --port 43100 --device WW0 --lighthouse-count 1
```

Each datagram is versioned JSON and contains:

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
  "tracking_valid": true
}
```

Units are metres, m/s, and rad/s. `timestamp` is Unix time on the Pi at receipt.
The receiver reports sequence gaps and end-to-end latency, and projects each
sample into the shared planar `RobotPose {x, y, yaw, t}` contract. A pose older
than 150 ms is treated as invalid even if its last packet said it was valid.

For the hackathon deployment, install the included services on their respective
hosts:

```bash
# Raspberry Pi
sudo cp deploy/systemd/nero-vive-publisher.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now nero-vive-publisher

# jscore
sudo cp deploy/systemd/nero-vive-receiver.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now nero-vive-receiver
```

The receiver atomically maintains `/run/nero/vive_pose.json`. Local planner,
controller, and telemetry processes can read that file without owning the UDP
socket. It includes the full packet, a planar `robot_pose`, and transport latency
and sequence-loss diagnostics.

---

## Hardware

| Required | Note |
|---|---|
| ≥1 Lighthouse base station (gen 1 or 2) | **Power only** — it has no data link to the host |
| A Vive controller or Tracker | This is what *senses*; nothing works without it |
| A USB **data** cable (or a dongle) | Connects the device **to the host**, never to the base station |

**Not required:** a headset, SteamVR, or — when wired — the wireless dongle.

> A base station alone streams nothing. It only emits IR; every pose is computed
> from the tracked device's photodiodes.

Device names follow libsurvive's convention: `WW0` wired watchman (controller on
a cable), `WM0` wireless watchman, `TR0` Vive Tracker, `LH0`/`LH1` base stations.

## One-time setup

```bash
sudo apt install -y build-essential cmake zlib1g-dev libx11-dev \
  libusb-1.0-0-dev freeglut3-dev liblapacke-dev libopenblas-dev
git clone https://github.com/cntools/libsurvive.git ~/src/libsurvive
cd ~/src/libsurvive && make

# USB permissions, else the device is invisible to a non-root user
sudo cp ./useful_files/81-vive.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

## Verify tracking before involving Rerun

```bash
lsusb | grep -i 28de                     # expect: Valve Software ... Controller
cd ~/src/libsurvive && ./bin/survive-cli --lighthouse-count 1
grep -oE '"(OOTXSet|PositionSet)":"[01]"' ~/.config/libsurvive/config.json
```

Both flags must read `1`:

- `OOTXSet=1` — the station's factory calibration was decoded. It is broadcast
  optically, one bit per sync flash, so it takes seconds to arrive.
- `PositionSet=1` — libsurvive has solved where the station sits in the world.

If `PositionSet=0`, hold the device still and visible and re-run with
`--force-calibrate` for 30–60 s.

## Gotchas

- **Nothing in Rerun, no error** — the streamer probably died on its first pose.
  Run in the foreground; data only flows once `logged N poses` prints.
- **Viewer version mismatch** — the viewer must be 0.22.1, matching the pinned
  SDK. Rerun's Python API also churns between releases: `set_time_sequence` here
  is spelled `set_time(..., sequence=)` in >=0.23, and `Transform3D(axis_length=)`
  exists in 0.22 but was removed later.
- **`Transform3D` draws nothing on its own** — it is only a transform; geometry
  must be logged *under* that entity to be visible.
- **Trails smear** — log them in world space, not under the moving entity.
- **`ModuleNotFoundError: pysurvive`** — set both env vars from the quick start.
- **`remote port forwarding failed`** — port 9876 is already taken on the host by
  a previous run; kill it and retry.
- **Only `LH0`, no `WW0`** — the device isn't reaching the host: check
  `lsusb | grep -i 28de`, the udev rules, and that it's a data (not charge-only)
  cable.
- **Coverage** — one base station covers roughly a 4–5 m line-of-sight cone. This
  is lab-scale ground truth; a second station helps materially.
