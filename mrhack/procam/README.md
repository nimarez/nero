# procam — projector↔floor calibration + Vive-following safety circle

The **external** layer of the K1 demo: an outside RealSense + a projector paint a live safety circle
(and trajectory) on the floor that **follows the K1**, tracked by a Vive tracker on its back. Nima's
onboard `nero` agent walks the robot (YOLO-World → nav → `B1LocoClient` walk); this projects *where it
is* + an ANSI keep-out zone, all in one room-fixed frame.

## Pieces
| File | Where | What |
|---|---|---|
| `context_snippets.py` | box | RealSense **IR** capture (emitter off — immune to the projection), ArUco `4x4_50` detect (tuned), sway projection (`swaymsg output HDMI-A-1 bg`), handle load |
| `mapper.py` | phone/laptop → :8091 | drag-GUI: place the projector↔floor mapping by eye (drag 4 handles onto the 4 tags). Saves `/tmp/procam_calib.json` |
| `procam_true.py` | box | *(Sol)* detect 4 tags → **metric floor rectangle** from each tag's square → **true circle** centred in the tags |
| `follow_circle.py` | box | **THE CLOSED LOOP** — Vive pose → floor → projector; the safety circle follows the K1 (+ heading) |
| `vive_bridge.py` | **Pi** | forwards the Vive pose (libsurvive) as UDP JSON `{x,y,yaw}` to jscore |
| `vive_floor_cal.py` | box | touch the 4 tags with the controller → SE(2) `vive→floor` (reuses `mrhack…frame_align.umeyama_se2`) |

## Frames
```
Vive (libsurvive world) --SE2 [vive_floor_cal]--> floor metres --H [procam_true: tags+handles]--> projector px
```

## Runbook (rig up in the new room)
0. Projector on `HDMI-A-1` via sway; RealSense sees the 4 tags (IR); **Pi on the tailnet** with the tracker.
1. **Mapping** — open the mapper (`:8091`), drag the 4 handles onto the 4 tags → `/tmp/procam_calib.json`.
2. **Static check** — `~/Prismos-x/venv/bin/python procam_true.py` → a true circle centred in the tags.
3. **Vive bridge (on the Pi)** — `python vive_bridge.py --host <jscore-ip> --device TR0`.
4. **Vive→floor** — `python vive_floor_cal.py` (touch each tag with the controller).
5. **CLOSED LOOP** — `PYTHONPATH=<nero>/src python follow_circle.py` → circle follows the K1.

## No-hardware checks (done — all green)
- `follow_circle.py --selftest` — transform-chain math (5/5 PASS).
- `follow_circle.py --mock` — synthetic motion, real projector (run on the box).

## Notes / TODO
- **`TAG_SIZE_M`** — measure the physical ArUco side; set it in `follow_circle.py` / `vive_floor_cal.py` (metric scale).
- Keep the Vive base station roughly level so tracker `x,y` ≈ the floor plane.
- ANSI radius = `mrhack.safety.safety_circle.safety_radius` (reach + stopping distance, inflates when pose confidence drops).
- **Detection**: use **IR** (emitter off) for the tags — the visible projection can't wash them out; the earlier RGB fragility is gone.
- **Still to stand up**: the Pi on the tailnet + `pose_source` streaming. Everything else is code-complete and verified.
