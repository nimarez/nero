# Vive / Lighthouse tracking

External 6-DoF pose tracking at ~280 Hz, sub-centimetre, **independent of
ORB-SLAM3**. Intended as a ground-truth reference for SLAM evaluation and as a
live 3D view in Rerun.

```
 base station ──IR sweeps──▶  controller/tracker  ──USB──▶  host ──▶ libsurvive ──▶ Rerun
  (power only,                 (photodiodes =                        (pysurvive)
   no data cable)               the actual sensor)
```

## Hardware

| Required | Note |
|---|---|
| ≥1 Lighthouse base station (gen 1 or 2) | **Power only** — it has no data link to the host |
| A Vive controller or Tracker | This is what *senses*; it computes nothing without it |
| A USB **data** cable (or a dongle) | Connects the device **to the host**, never to the base station |

**Not required:** a headset, SteamVR, or — when wired — the wireless dongle.

> A base station alone streams nothing. It only emits light; every pose comes
> from the tracked device's photodiodes.

Device names follow libsurvive's convention: `WW0` wired watchman (controller on
a cable), `WM0` wireless watchman, `TR0` Vive Tracker, `LH0`/`LH1` base stations.

## Setup

`pysurvive` lives inside the libsurvive source tree, not on PyPI:

```bash
sudo apt install -y build-essential cmake zlib1g-dev libx11-dev \
  libusb-1.0-0-dev freeglut3-dev liblapacke-dev libopenblas-dev
git clone https://github.com/cntools/libsurvive.git ~/src/libsurvive
cd ~/src/libsurvive && make

# USB permissions, else the device is invisible to a non-root user
sudo cp ./useful_files/81-vive.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

Both paths must be exported for `import pysurvive` to work:

```bash
export PYTHONPATH=$HOME/src/libsurvive/bindings/python
export LD_LIBRARY_PATH=$HOME/src/libsurvive/bin
```

Rerun comes from the optional dependency group:

```bash
uv sync --group eval
```

## Verify tracking before involving Rerun

```bash
lsusb | grep -i 28de                     # expect: Valve Software ... Controller
cd ~/src/libsurvive && ./bin/survive-cli --lighthouse-count 1
grep -oE '"(OOTXSet|PositionSet)":"[01]"' ~/.config/libsurvive/config.json
```

Both flags must read `1`:

- `OOTXSet=1` — the station's factory calibration was decoded (broadcast
  optically, one bit per sync flash, so it takes seconds to receive).
- `PositionSet=1` — libsurvive has solved where the station sits in the world.

If `PositionSet=0`, hold the device still and visible and run with
`--force-calibrate` for 30–60 s.

## Run

```bash
# laptop
rerun --port 9876
# host (over a reverse tunnel: ssh -R 9876:localhost:9876 user@host)
uv run nero-vive-rerun --lighthouse-count 1
```

Entities: `world/WW0` (device frame + axes), `world/LH0` (base station),
`world/trails/WW0` (path). The scene is only ~1 m across — use the viewport's
recenter control if it looks empty.

## Gotchas

- **Nothing appears in Rerun, no error** — the streamer likely died on its first
  pose. Run in the foreground; data only flows once `logged N poses` prints.
- **`Transform3D` draws nothing on its own** — it is only a transform; geometry
  must be logged *under* that entity to be visible.
- **Trails smear** — log them in world space, not under the moving entity.
- **`ModuleNotFoundError: pysurvive`** — set both env vars above.
- **Only `LH0`, no `WW0`** — the device isn't reaching the host: check
  `lsusb | grep -i 28de`, udev rules, and that it's a data (not charge-only) cable.
- **Coverage** — one base station covers roughly a 4–5 m line-of-sight cone. This
  is lab-scale ground truth; a second station helps materially.
