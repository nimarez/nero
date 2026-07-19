#!/usr/bin/env python3
"""
frame_bridge.py -- the ONE explicit transform between our metric floor frame
and Jonny's normalized tag-quad frame (PR #7, src/nero/projector/).

WHY this file exists (Sol's convergence call, option C):
  We keep two projector renderers for now -- ours (metric meters -> H_floor2proj
  -> pixels, swaybg fallback) and Jonny's (normalized [0,1]^2 tag-quad ->
  ProjectorCalibration.transform -> pixels, low-latency in-memory). Our unique
  value is the NAV OVERLAY (bearing ray / multi-arrow path / goal / barrier),
  computed in metric floor metres. To port that overlay onto Jonny's renderer
  after PR #7 merges, every metric floor point must convert to his normalized
  frame with ZERO ambiguity. This is that seam -- axis orientation, scale,
  origin, and clipping all pinned down and tested, so the port is mechanical.

Jonny's convention (src/nero/projector/motion.py::map_floor_position):
    u = 0.5 + (x - origin_x) / span_x_m
    v = 0.5 - (y - origin_y) / span_y_m          # v FLIPS: world +y is up, image v is down
  origin = tag-quad centre; defaults span_x_m = 3.0, span_y_m = 2.2 (anisotropic).
  A metric circle stays circular on the floor because he samples the PERIMETER in
  physical space and converts each point here BEFORE the keystone homography.

Usage (post-#7-merge port sketch, do NOT wire until his branch is in):
    from nero.projector.calibration import ProjectorCalibration
    norm = metric_to_norm(goal_xy_metres, origin, span)     # our overlay geometry
    px   = calibration.transform(norm)                      # his keystone homography
"""

import math

SPAN_X_M = 3.0        # metres spanned by u in [0,1]  (Jonny's default)
SPAN_Y_M = 2.2        # metres spanned by v in [0,1]  (Jonny's default)
ORIGIN = (0.0, 0.0)   # tag-quad centre in metric floor metres (our frame's origin)


def _pairs(pts):
    """Accept a single (x,y) or an iterable of them; yield (x,y) floats."""
    if pts is None:
        return []
    if len(pts) == 2 and not hasattr(pts[0], "__len__"):
        return [(float(pts[0]), float(pts[1]))]
    return [(float(p[0]), float(p[1])) for p in pts]


def metric_to_norm(pts, origin=ORIGIN, span=(SPAN_X_M, SPAN_Y_M), clip=False):
    """
    Metric floor (metres, y-up, origin at tag centre) -> normalized tag-quad [0,1]^2.
    `pts` is a single (x,y) or a list of them. Returns the same shape as a list.
    clip=True clamps to [0,1] (Jonny's canvas edge); default keeps out-of-quad
    coordinates so the caller can decide to draw or drop them.
    """
    ox, oy = origin
    sx, sy = span
    out = []
    for x, y in _pairs(pts):
        u = 0.5 + (x - ox) / sx
        v = 0.5 - (y - oy) / sy          # axis flip: +y (forward/up) -> smaller v
        if clip:
            u = min(1.0, max(0.0, u))
            v = min(1.0, max(0.0, v))
        out.append((u, v))
    return out[0] if len(out) == 1 else out


def norm_to_metric(pts, origin=ORIGIN, span=(SPAN_X_M, SPAN_Y_M)):
    """Inverse of metric_to_norm (no clipping -- exact round-trip)."""
    ox, oy = origin
    sx, sy = span
    out = []
    for u, v in _pairs(pts):
        x = ox + (u - 0.5) * sx
        y = oy - (v - 0.5) * sy
        out.append((x, y))
    return out[0] if len(out) == 1 else out


def _selftest():
    ok = True

    def chk(name, cond):
        nonlocal ok
        print("  [%s] %s" % ("PASS" if cond else "FAIL", name))
        ok = ok and bool(cond)

    u, v = metric_to_norm((0.0, 0.0))
    chk("origin -> (0.5, 0.5)", abs(u - 0.5) < 1e-12 and abs(v - 0.5) < 1e-12)

    u, v = metric_to_norm((SPAN_X_M / 2.0, 0.0))
    chk("+x half-span -> u=1.0 (right edge)", abs(u - 1.0) < 1e-12 and abs(v - 0.5) < 1e-12)

    u, v = metric_to_norm((0.0, SPAN_Y_M / 2.0))
    chk("+y half-span -> v=0.0 (top, axis flipped)", abs(u - 0.5) < 1e-12 and abs(v - 0.0) < 1e-12)

    # matches Jonny's map_floor_position formula exactly (same numbers, origin form)
    x, y = 0.6, -0.3
    u, v = metric_to_norm((x, y))
    chk("matches map_floor_position formula",
        abs(u - (0.5 + x / SPAN_X_M)) < 1e-12 and abs(v - (0.5 - y / SPAN_Y_M)) < 1e-12)

    # round-trip metric -> norm -> metric is identity
    pts = [(0.0, 0.0), (1.2, -0.7), (-0.9, 0.55), (1.49, 1.09)]
    rt = norm_to_metric(metric_to_norm(pts))
    chk("round-trip identity", all(abs(a[0] - b[0]) < 1e-9 and abs(a[1] - b[1]) < 1e-9 for a, b in zip(pts, rt)))

    # clip keeps out-of-quad points inside [0,1]
    u, v = metric_to_norm((10.0, -10.0), clip=True)
    chk("clip clamps to [0,1]", 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0)
    u, v = metric_to_norm((10.0, -10.0), clip=False)
    chk("no-clip preserves out-of-quad", u > 1.0 and v > 1.0)

    # a metric circle samples to a closed loop that stays centred at (0.5,0.5)
    circ = [(0.4 * math.cos(a), 0.4 * math.sin(a)) for a in [i * math.pi / 8 for i in range(16)]]
    nc = metric_to_norm(circ)
    cu = sum(p[0] for p in nc) / len(nc)
    cv = sum(p[1] for p in nc) / len(nc)
    chk("circle centroid stays at (0.5,0.5)", abs(cu - 0.5) < 1e-9 and abs(cv - 0.5) < 1e-9)

    print("SELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
