# procam — projector↔floor calibration + Vive-following safety circle

The **external** layer of the K1 demo: an outside RealSense + a projector paint a live safety circle
(and trajectory) on the floor that **follows the K1**, tracked by a Vive tracker on its back. Nima's
onboard `nero` agent walks the robot (YOLO-World/QNN → ArUco nav → pure-pursuit → `B1LocoClient` walk);
this projects *where it is* + an ANSI keep-out zone, all in one room-fixed frame.

## Pieces
| File | Where | What |
|---|---|---|
| `context_snippets.py` | box | RealSense **IR** capture (emitter off — immune to the projection), ArUco `4x4_50` detect (tuned), sway projection (`swaymsg output HDMI-A-1 bg`), handle load |
| `mapper.py` | phone/laptop → :8091 | drag-GUI: place the projector↔floor mapping by eye (drag 4 handles onto the 4 tags). Saves `/tmp/procam_calib.json` |
| `procam_true.py` | box | *(Sol)* detect 4 tags → **metric floor rectangle** from each tag's square → **true circle** centred in the tags |
| `follow_circle.py` | box | **THE CLOSED LOOP** — reads the Vive pose from `/run/nero/vive_pose.json` → floor → projector; the ANSI safety circle follows the K1 (+ heading) |
| `vive_floor_cal.py` | box | touch the 4 tags with the controller → SE(2) `vive→floor` (reuses `mrhack…frame_align.umeyama_se2`) |
| `vive_bridge.py` | Pi | minimal tailnet UDP fallback. **Canonical Vive stream is PR #6** (`nero-vive-udp-send`/`-receive`). |

## Frames
```
Vive (libsurvive) --SE2 [vive_floor_cal]--> floor metres --H [procam_true: tags+handles]--> projector px
```

## Vive stream — use PR #6 (M2, Jonny)
`nero-vive-udp-send` on the **Pi** → UDP `10.77.0.1:43100` → `nero-vive-udp-receive` on **jscore** writes
`/run/nero/vive_pose.json` (atomic latest pose: `position[xyz]`, `quaternion_xyzw`, `tracking_valid`,
`transport.received_at`; 150 ms freshness). `follow_circle.py` just polls that file.

## Runbook (rig up in the new room)
0. Projector on `HDMI-A-1` via sway; RealSense sees the 4 tags (IR); **Pi on NERA-WIFI** with the tracker.
1. **Mapping** — open the mapper (`:8091`), drag the 4 handles onto the 4 tags → `/tmp/procam_calib.json`.
2. **Static check** — `~/Prismos-x/venv/bin/python procam_true.py` → a true circle centred in the tags.
3. **Vive stream** — start the PR #6 services (`nero-vive-udp-send` on the Pi, `nero-vive-udp-receive` on jscore) → `/run/nero/vive_pose.json`.
4. **Vive→floor** — `python vive_floor_cal.py` (touch each tag with the controller).
5. **CLOSED LOOP** — `PYTHONPATH=<nero>/src python follow_circle.py` → circle follows the K1.

## No-hardware checks (done — all green)
- `follow_circle.py --selftest` — transform chain + `vive_pose.json` parse/freshness/validity (7/7 PASS).
- `follow_circle.py --mock` — synthetic motion, real projector (run on the box).

## Notes / TODO
- **`TAG_SIZE_M`** — measure the physical ArUco side; set it in `follow_circle.py` / `vive_floor_cal.py`.
- Keep the Vive base station roughly level so tracker `x,y` ≈ the floor plane.
- ANSI radius = `mrhack.safety.safety_circle.safety_radius` (reach + stopping distance, inflates when pose confidence drops).
- **Detection**: use **IR** (emitter off) for the tags — the visible projection can't wash them out.
- `vive_floor_cal.py` reads the same `/run/nero/vive_pose.json` while you touch the tags.
