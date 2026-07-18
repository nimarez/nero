"""#3 KEYSTONE - align the onboard SLAM frame to the Vive/floor frame, then fuse.

The robot is FLOOR-CONSTRAINED and both sensors are gravity-aligned, so the map between the onboard
ORB-SLAM3 world frame and the HTC Vive/floor frame is a 3-DOF SE(2) transform (x, y, yaw), fit by a
closed-form 2D Kabsch over time-synced XY pairs. Once SLAM is in the floor frame we fuse it with the
Vive (the drift-free anchor) into one confident pose, and flag disagreement.

Why SE(2) and not SE(3): a real "walk to the object" is nearly a straight line. In SE(3) that is
DEGENERATE - rotation about the walked line is unobservable, so the fit returns an arbitrary-looking
rotation with a deceptively low RMS. In SE(2) there is no out-of-plane rotation DOF, so even a
straight walk determines yaw + translation. (So this module is SE(2) by design; genuine 3D
excitation would need a separate SE(3) fit.)

Hardened after a cross-model review (Claude Fable 5 + Sol / GPT-5.6 independently converged):
  - SE(2) fit with a degeneracy guard (reject a calibration with no real motion).
  - RANSAC + inlier refit so one Vive glitch / SLAM relocalization jump can't poison the transform.
  - Time-synced pairing: interpolate one stream onto the other's stamps; drop pairs beyond max_dt.
  - Health-arbitrated fusion: "disagree -> trust Vive" ONLY while the Vive is fresh + valid; a stale
    or dropped Vive never silently drives the pose (that was a confidently-wrong-safety-circle bug).
  - Confidence from BOTH position and yaw innovation, decayed by sample age; feeds safety_circle.zd.
  - ValueError on bad input (not assert, which vanishes under `python -O`).

Verify the math (no hardware):
    uv run --with numpy python -m mrhack.localization.frame_align --selftest

Loop usage:
    ts, a_xyyaw, b_xyyaw = pair_by_time(slam_stamped, vive_stamped, max_dt=0.03)   # sync the streams
    R, t2, yaw, rms, inl = umeyama_se2_ransac(a_xyyaw[:, :2], b_xyyaw[:, :2])       # calibrate once
    slam_floor = transform_pose(R, t2, slam_xyyaw)                                  # every SLAM frame
    fused, conf, info = fuse(slam_floor, vive_xyyaw, vive_age=..., slam_age=...)    # the trusted pose
"""
from __future__ import annotations
import argparse
import math
import sys

import numpy as np


# --------------------------------------------------------------------------- geometry
def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _fit_se2_pair(src2, dst2):
    """Exact SE(2) from a 2-point correspondence (the RANSAC minimal sample)."""
    ds, dd = src2[1] - src2[0], dst2[1] - dst2[0]
    if math.hypot(*ds) < 1e-9 or math.hypot(*dd) < 1e-9:
        raise ValueError("coincident pair")
    ang = math.atan2(dd[1], dd[0]) - math.atan2(ds[1], ds[0])
    c, s = math.cos(ang), math.sin(ang)
    R = np.array([[c, -s], [s, c]])
    return R, dst2.mean(0) - R @ src2.mean(0)


def umeyama_se2(src_xy, dst_xy, min_spread=0.10):
    """Rigid SE(2): (R 2x2, t 2, yaw, rms) with dst ~= R @ src + t over Nx2 XY points.

    Closed-form 2D Kabsch (robust for a straight/collinear walk - a 2D rotation is fully determined
    by mapping one motion direction to another). Raises ValueError on bad shape, non-finite data,
    < 2 points, or XY spread below min_spread (i.e. no real motion to calibrate on)."""
    src = np.asarray(src_xy, float)
    dst = np.asarray(dst_xy, float)
    if src.ndim != 2 or src.shape[1] != 2 or src.shape != dst.shape:
        raise ValueError("src/dst must be matching Nx2 arrays")
    if len(src) < 2:
        raise ValueError("need >= 2 matched points")
    if not (np.isfinite(src).all() and np.isfinite(dst).all()):
        raise ValueError("non-finite input")
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    spread = math.sqrt(max(float(np.mean(np.sum(S * S, 1))), float(np.mean(np.sum(D * D, 1)))))
    if spread < min_spread:
        raise ValueError(f"degenerate: XY spread {spread:.3f} m < {min_spread} m (walk further / add a turn)")
    a = float(np.sum(S[:, 0] * D[:, 0] + S[:, 1] * D[:, 1]))   # sum cos-part
    b = float(np.sum(S[:, 0] * D[:, 1] - S[:, 1] * D[:, 0]))   # sum sin-part
    yaw = math.atan2(b, a)
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    t = mu_d - R @ mu_s
    rms = float(np.sqrt(np.mean(np.sum((dst - (src @ R.T + t)) ** 2, 1))))
    return R, t, yaw, rms


def umeyama_se2_ransac(src_xy, dst_xy, thresh=0.05, iters=200, min_inlier_frac=0.5,
                       seed=0, min_spread=0.10):
    """Robust SE(2): RANSAC over 2-point samples -> inlier-only refit. One bad Vive sample or SLAM
    relocalization jump can't move the transform. Returns (R, t, yaw, inlier_rms, inlier_mask).
    Falls back to the guarded all-point fit if no consensus is found (which raises if degenerate)."""
    src = np.asarray(src_xy, float)
    dst = np.asarray(dst_xy, float)
    n = len(src)
    if n < 2:
        raise ValueError("need >= 2 points")
    if n == 2:
        R, t, yaw, rms = umeyama_se2(src, dst, min_spread=min_spread)
        return R, t, yaw, rms, np.ones(2, bool)
    rng = np.random.default_rng(seed)
    best_mask, best_count = None, -1
    for _ in range(iters):
        i, j = (int(x) for x in rng.choice(n, size=2, replace=False))
        try:
            R, t = _fit_se2_pair(src[[i, j]], dst[[i, j]])
        except ValueError:
            continue
        res = np.sqrt(np.sum((dst - (src @ R.T + t)) ** 2, 1))
        mask = res < thresh
        if int(mask.sum()) > best_count:
            best_count, best_mask = int(mask.sum()), mask
    if best_mask is None or best_count < max(2, int(min_inlier_frac * n)):
        R, t, yaw, rms = umeyama_se2(src, dst, min_spread=min_spread)   # no consensus -> guarded fit
        return R, t, yaw, rms, np.ones(n, bool)
    R, t, yaw, rms = umeyama_se2(src[best_mask], dst[best_mask], min_spread=min_spread)
    return R, t, yaw, rms, best_mask


def transform_pose(R, t, xyyaw):
    """Map a SLAM (x, y, yaw) into the floor frame with an SE(2) (R 2x2, t 2)."""
    R = np.asarray(R, float)
    x, y, yaw = xyyaw
    p = R @ np.array([x, y], float) + np.asarray(t, float)
    return float(p[0]), float(p[1]), _wrap(yaw + math.atan2(R[1, 0], R[0, 0]))


# --------------------------------------------------------------------------- time sync
def pair_by_time(a_stamped, b_stamped, max_dt=0.03):
    """Interpolate stream B onto stream A's timestamps. Rows are (t, x, y, yaw).

    Returns (ts, a_xyyaw, b_xyyaw) for the A-stamps that B brackets within max_dt (linear XY,
    shortest-arc yaw). Protects the calibration from rate/latency skew between the two clocks -
    without this, motion turns a 50 ms offset straight into alignment error."""
    a = np.asarray(a_stamped, float)
    b = np.asarray(b_stamped, float)
    if a.ndim != 2 or a.shape[1] != 4 or b.ndim != 2 or b.shape[1] != 4:
        raise ValueError("streams must be Nx4 (t, x, y, yaw)")
    b = b[np.argsort(b[:, 0])]
    tb = b[:, 0]
    ts, aa, bb = [], [], []
    for row in a:
        t = float(row[0])
        k = int(np.searchsorted(tb, t))
        if k == 0 or k == len(tb):
            continue                                   # outside B's coverage
        t0, t1 = tb[k - 1], tb[k]
        if min(t - t0, t1 - t) > max_dt:
            continue                                   # nearest bracket too far in time
        w = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        xy = (1 - w) * b[k - 1, 1:3] + w * b[k, 1:3]
        yaw = _wrap(b[k - 1, 3] + w * _wrap(b[k, 3] - b[k - 1, 3]))
        ts.append(t)
        aa.append(row[1:4])
        bb.append([xy[0], xy[1], yaw])
    return np.array(ts), np.array(aa).reshape(-1, 3), np.array(bb).reshape(-1, 3)


# --------------------------------------------------------------------------- fusion
def fuse(slam_floor, vive, *, slam_age=0.0, vive_age=0.0, vive_valid=True, slam_valid=True,
         w_vive=0.75, disagree_m=0.15, disagree_yaw=math.radians(15), max_age=0.20):
    """Health-arbitrated fusion of SLAM-in-floor + Vive, each (x, y, yaw).

    Vive is the drift-free anchor, trusted ONLY while fresh (age <= max_age) and valid; a stale or
    dropped Vive never silently drives the pose. On disagreement between two HEALTHY sources we lean
    on the drift-free Vive but drop confidence. Confidence folds position AND yaw innovation and
    sample age. Returns (fused_xyyaw, confidence 0..1, info)."""
    for name, p in (("slam", slam_floor), ("vive", vive)):
        if len(p) != 3 or not all(map(math.isfinite, p)):
            raise ValueError(f"{name} pose must be 3 finite values (x, y, yaw)")
    if not 0.0 <= w_vive <= 1.0:
        raise ValueError("w_vive must be in [0, 1]")
    sx, sy, syaw = slam_floor
    vx, vy, vyaw = vive
    vive_ok = bool(vive_valid) and vive_age <= max_age
    slam_ok = bool(slam_valid) and slam_age <= max_age
    dpos = math.hypot(sx - vx, sy - vy)
    dyaw = abs(_wrap(syaw - vyaw))
    innov = max(dpos / disagree_m, dyaw / disagree_yaw)        # BOTH channels
    agree = innov <= 1.0

    if vive_ok and slam_ok:
        w, src = (w_vive, "blend") if agree else (1.0, "vive")
    elif vive_ok:
        w, src = 1.0, "vive_only"          # SLAM stale/lost -> Vive
    elif slam_ok:
        w, src = 0.0, "slam_only"          # Vive stale/dropped -> aligned SLAM (never the stale Vive)
    else:
        w, src = w_vive, "degraded"        # neither fresh -> best-effort, confidence -> ~0

    fx = (1 - w) * sx + w * vx
    fy = (1 - w) * sy + w * vy
    fyaw = _wrap(syaw + w * _wrap(vyaw - syaw))

    agree_conf = max(0.0, 1.0 - innov / 3.0)
    used_age = max(slam_age, vive_age) if src in ("blend", "degraded") \
        else (vive_age if "vive" in src else slam_age)
    age_conf = max(0.0, 1.0 - used_age / max_age)
    health = 0.4 if src == "degraded" else 1.0
    conf = min(0.98, agree_conf) * age_conf * health           # never the "unjustified 1.0"

    return (fx, fy, fyaw), float(conf), {
        "pos_m": dpos, "yaw_rad": dyaw, "innov": innov, "agree": agree,
        "src": src, "vive_ok": vive_ok, "slam_ok": slam_ok,
    }


# --------------------------------------------------------------------------- self-test
def _selftest():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    th, tt = 0.6, np.array([1.2, -0.4])
    c, s = math.cos(th), math.sin(th)
    Rt = np.array([[c, -s], [s, c]])
    N = 40

    # 1. SE(2) recovers a known transform on a CURVED path
    vive = np.column_stack([np.linspace(0, 3, N), 0.6 * np.sin(np.linspace(0, 3, N))])
    slam = (vive - tt) @ Rt                              # slam = Rt^T (vive - tt)
    R, t, yaw, rms = umeyama_se2(slam, vive)
    check("SE2 curved recovers R,t (rms<1e-9)",
          rms < 1e-9 and np.linalg.norm(R - Rt) < 1e-9 and np.linalg.norm(t - tt) < 1e-9)

    # 2. SE(2) recovers on a STRAIGHT walk - the SE(3)-degenerate case (the whole reason for SE(2))
    xs = np.linspace(0, 4, N)
    vive_line = np.column_stack([xs, np.full(N, 0.5)])
    slam_line = (vive_line - tt) @ Rt
    Rl, tl, yl, rl = umeyama_se2(slam_line, vive_line)
    check("SE2 straight walk still well-determined (rms<1e-9)",
          rl < 1e-9 and np.linalg.norm(Rl - Rt) < 1e-9)

    # 3. degeneracy guard: no motion (coincident points) -> ValueError
    try:
        umeyama_se2(np.zeros((5, 2)), np.ones((5, 2)))
        check("no-motion calibration rejected", False)
    except ValueError:
        check("no-motion calibration rejected", True)

    # 4. transform_pose maps a fresh SLAM pose onto the true floor pose (position + yaw)
    sp = (np.array([1.0, 0.5]) - tt) @ Rt
    fx, fy, fyaw = transform_pose(R, t, (sp[0], sp[1], 0.3))
    check("transform_pose position exact", math.hypot(fx - 1.0, fy - 0.5) < 1e-9)
    check("transform_pose yaw carried (=0.3+th)", abs(_wrap(fyaw - (0.3 + th))) < 1e-9)

    # 5. RANSAC rejects a gross outlier and still recovers the transform
    slam_o, vive_o = slam.copy(), vive.copy()
    vive_o[7] += np.array([1.3, -0.9])                   # one bad Vive sample
    Rr, tr, yr, rr, mask = umeyama_se2_ransac(slam_o, vive_o, thresh=0.05)
    check("RANSAC flags the outlier only", (not mask[7]) and int(mask.sum()) == N - 1)
    check("RANSAC transform still correct", np.linalg.norm(Rr - Rt) < 1e-6)

    # 6. time-sync pairing interpolates + drops out-of-coverage / large-dt stamps
    tb = np.arange(0, 1.0, 0.02)
    b = np.column_stack([tb, tb, 2 * tb, np.zeros(len(tb))])   # x=t, y=2t
    ta = np.array([0.101, 0.505, 5.0])                        # last is outside B's coverage
    a = np.column_stack([ta, ta, 2 * ta, np.zeros(3)])
    tsync, aa, bb = pair_by_time(a, b, max_dt=0.03)
    check("pair_by_time drops out-of-coverage stamp", len(tsync) == 2)
    check("pair_by_time interpolates B onto A (x~0.101)", abs(bb[0, 0] - 0.101) < 1e-6)

    # 7. fuse - agreement -> high confidence, blended
    _, cf, i0 = fuse((1.00, 2.00, 0.10), (1.02, 1.99, 0.11))
    check("fuse agree: src=blend + conf>0.9", i0["src"] == "blend" and cf > 0.9)

    # 8. fuse - POSITION disagreement -> low conf, leans on drift-free Vive
    _, cd, id_ = fuse((1.00, 2.00, 0.10), (1.40, 2.00, 0.10))
    check("fuse pos-disagree: src=vive + conf<0.6", id_["src"] == "vive" and cd < 0.6)

    # 9. fuse - YAW-ONLY disagreement must NOT return conf 1.0 (Sol #5 / Fable #3)
    _, cy, iy = fuse((1.00, 2.00, 0.00), (1.00, 2.00, math.radians(90)))
    check("fuse yaw-only disagree: conf<0.6 (not 1.0)", cy < 0.6 and not iy["agree"])

    # 10. fuse - STALE Vive must NOT take over; use aligned SLAM instead (Sol #1 / Fable #3)
    fpose, cs2, is_ = fuse((1.00, 2.00, 0.10), (5.00, 9.00, 3.00), vive_age=1.0)
    check("stale Vive ignored: src=slam_only + pose=SLAM",
          is_["src"] == "slam_only" and math.hypot(fpose[0] - 1.0, fpose[1] - 2.0) < 1e-9)

    # 11. fuse - both stale -> degraded, ~zero confidence
    _, cb, ib = fuse((1.0, 2.0, 0.1), (1.0, 2.0, 0.1), vive_age=1.0, slam_age=1.0)
    check("both stale -> degraded + low conf", ib["src"] == "degraded" and cb < 0.5)

    # 12. input validation raises (not a silent assert)
    try:
        fuse((1.0, float("nan"), 0.0), (0.0, 0.0, 0.0))
        check("NaN pose rejected", False)
    except ValueError:
        check("NaN pose rejected", True)

    print("SELFTEST:", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    print("import umeyama_se2 / umeyama_se2_ransac / pair_by_time / transform_pose / fuse, or --selftest")


if __name__ == "__main__":
    main()
