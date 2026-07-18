"""#7 - ANSI/RIA-style DYNAMIC safety circle for the projected keep-out zone.

The circle the projector paints around the robot is a KEEP-OUT zone sized on the speed-and-
separation-monitoring principle (ISO/TS 15066, ANSI/RIA R15.06): a static hazard envelope (the
arm/body reach) PLUS the distance the robot needs to stop from its current speed, PLUS sensing
uncertainty and a fixed margin. It grows with speed and collapses to the reach envelope at rest.

    r(v) = reach + [ v*t_react + v^2/(2*a_decel) ] + margin + Zd + Zr
           \____/   \_______________________________/   \_____________/
           static        robot stopping distance          uncertainty

The `Zd` term is the sensing uncertainty - drive it from the fused pose confidence
(mrhack.localization.frame_align.fuse): when SLAM and Vive disagree, confidence drops, Zd rises,
and the painted circle INFLATES automatically. Safe degradation, not a silent wrong circle.

Verify (no hardware):
    uv run python -m mrhack.safety.safety_circle --selftest
Calibrate reach / t_react / a_decel from the velocity-ID stop test on the real K1.
"""
from __future__ import annotations
import argparse
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class SafetyParams:
    reach: float = 0.55       # K1 arm/body hazard envelope from center (m)   [MEASURE on robot]
    t_react: float = 0.20     # comms + deadman + brake-onset latency (s)      [MEASURE: loop + DEADMAN_S]
    a_decel: float = 0.80     # effective deceleration to a halt (m/s^2)       [MEASURE: velocity-ID stop test]
    margin: float = 0.15      # fixed operational margin (m)
    zd: float = 0.05          # sensing position uncertainty (m)              [from fused-pose confidence]
    zr: float = 0.05          # robot position uncertainty (m)
    r_min: float = 0.60       # never paint a circle smaller than this (m)
    r_max: float = 2.50       # clamp to projector FOV / floor bounds (m)


def stopping_distance(v, p: SafetyParams = SafetyParams()):
    """Distance the robot covers before halting: reaction-time travel + decel-to-zero (m)."""
    v = max(0.0, float(v))
    return v * p.t_react + (v * v) / (2 * p.a_decel) if p.a_decel > 0 else v * p.t_react


def zd_from_confidence(conf, zd_floor=0.05, zd_ceiling=0.30):
    """Map fused-pose confidence (0..1, from frame_align.fuse) -> sensing uncertainty Zd (m).
    conf=1 -> zd_floor; conf=0 -> zd_ceiling. Low confidence => bigger safety circle."""
    conf = min(1.0, max(0.0, float(conf)))
    return zd_ceiling + (zd_floor - zd_ceiling) * conf


def safety_radius(v, p: SafetyParams = SafetyParams()):
    """Keep-out radius (m) at speed v (m/s). Monotonic in v; >= reach at rest; clamped to [r_min, r_max]."""
    r = p.reach + stopping_distance(v, p) + p.margin + p.zd + p.zr
    return float(min(p.r_max, max(p.r_min, r)))


def _selftest():
    ok = True
    p = SafetyParams()
    r0, rw, rf = safety_radius(0.0, p), safety_radius(0.3, p), safety_radius(0.6, p)
    print(f"r(0)={r0:.3f}  r(0.3 walk)={rw:.3f}  r(0.6)={rf:.3f} m")
    ok &= r0 >= p.reach                                  # at rest, covers the reach envelope
    ok &= rw > r0 and rf > rw                             # grows monotonically with speed
    want = p.reach + (0.3 * p.t_react + 0.09 / (2 * p.a_decel)) + p.margin + p.zd + p.zr
    print(f"closed-form @0.3: got {rw:.4f}  want {want:.4f}")
    ok &= abs(rw - want) < 1e-9
    # confidence tie-in: low pose confidence -> larger Zd -> larger circle
    lc = SafetyParams(zd=zd_from_confidence(0.11))        # the 0.11 conf from a frame_align disagreement
    r_lc = safety_radius(0.3, lc)
    print(f"low-confidence(conf=0.11) circle @0.3: {r_lc:.3f} m  (> {rw:.3f})")
    ok &= r_lc > rw and zd_from_confidence(1.0) < zd_from_confidence(0.0)
    # clamp floor: a tiny reach still yields >= r_min
    ok &= safety_radius(0.0, SafetyParams(reach=0.1)) == p.r_min
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--speed", type=float, help="print the safety radius at this speed (m/s)")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    if a.speed is not None:
        print(f"safety_radius({a.speed} m/s) = {safety_radius(a.speed):.3f} m")
    else:
        print("import safety_radius / stopping_distance / zd_from_confidence, or run --selftest")


if __name__ == "__main__":
    main()
