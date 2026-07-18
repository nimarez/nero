# Vive / Lighthouse pose tracking → Rerun

External 6-DoF pose tracking at ~280 Hz, sub-centimetre, **independent of
ORB-SLAM3**. Useful as a ground-truth reference for SLAM evaluation, and as a
live 3D view in Rerun.

---

## Quick start

Assumes the one-time setup below is already done (libsurvive built, tracker
plugged in, Rerun venv created).

**1 — On your laptop, start the Rerun viewer:**

```bash
rerun --port 9876
```

**2 — Open a reverse tunnel to the host the tracker is wired to:**

```bash
ssh -R 9876:localhost:9876 user@host
```

**3 — In that shell on the host, stream poses:**

```bash
export PYTHONPATH=/path/to/nero/src:$HOME/src/libsurvive/bindings/python
export LD_LIBRARY_PATH=$HOME/src/libsurvive/bin
~/venvs/vive/bin/python -m nero.vive.rerun_stream --lighthouse-count 1
```

You should see, and the same poses appear in the viewer:

```
INFO nero.vive.rerun_stream: logged 100 poses | WW0 at x=-0.430 y=+0.664 z=+0.077 m
```

**In Rerun:** `world/WW0` is the tracker (axes + origin dot), `world/LH0` the
base station, `world/trails/WW0` its path. The scene is only ~1 m across — use
the viewport's recenter control if it looks empty.

> **Why not `uv run nero-vive-rerun`?** The console script is registered, but
> `rerun-sdk` is *not* a nero dependency: it requires numpy>=2, which conflicts
> with this project's numpy<2 pin. Rerun therefore lives in its own virtualenv
> (`~/venvs/vive`) and is reached via `PYTHONPATH`, leaving nero's environment
> untouched. See "Rerun / numpy" below.

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

`pysurvive` ships inside the libsurvive source tree, not on PyPI:

```bash
sudo apt install -y build-essential cmake zlib1g-dev libx11-dev \
  libusb-1.0-0-dev freeglut3-dev liblapacke-dev libopenblas-dev
git clone https://github.com/cntools/libsurvive.git ~/src/libsurvive
cd ~/src/libsurvive && make

# USB permissions, else the device is invisible to a non-root user
sudo cp ./useful_files/81-vive.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

A separate venv for the Rerun tooling (kept out of nero's environment):

```bash
python3 -m venv ~/venvs/vive
~/venvs/vive/bin/pip install "rerun-sdk>=0.34,<0.35" numpy
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

## Rerun / numpy

`rerun-sdk` >= 0.24 requires `numpy>=2`; nero pins `numpy<2.0.0` for its compiled
`orbslam3-python` / `booster-robotics-sdk-python` wheels. The newest rerun-sdk
that resolves against numpy 1.x is **0.23.1**.

Rather than downgrade Rerun or loosen a pin protecting the SLAM stack, the viewer
tooling runs from its own virtualenv. `rerun_stream.py` imports `rerun` lazily, so
this package stays importable in nero's environment either way.

The pin's origin is undocumented and may be stale — both compiled wheels were
observed to install *and* import cleanly under numpy 2.5.1 on Python 3.12/x86_64.
Confirming that on the K1 itself (Python 3.10, ARM64, running ORB-SLAM3 rather
than merely importing it) would allow dropping the pin and depending on Rerun
directly.

## Gotchas

- **Nothing in Rerun, no error** — the streamer probably died on its first pose.
  Run in the foreground; data only flows once `logged N poses` prints.
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
