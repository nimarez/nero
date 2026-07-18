"""K1Gate - the ONLY code that touches the robot client. Guaranteed safe stop on any exit.
Error regime: any unhandled exception -> shutdown() then re-raise (fail-safe).
Importing this module is side-effect-free on Windows (SDK import is lazy in make_client)."""
from __future__ import annotations
import atexit, logging, signal, time
from .. import config
from ..contracts import VelCmd

log = logging.getLogger("K1Gate")


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def _snap(v, db):
    return 0.0 if abs(v) < db else v


class K1Gate:
    def __init__(self, client, limits=None, prep_wait_s=config.PREP_WAIT_S):
        self.client = client
        self.limits = limits or config.LIMITS
        self.prep_wait_s = prep_wait_s
        self.mode = "DAMP"
        self._shutdown_done = False
        self._guards = False

    def start(self):
        log.info("start: DAMP -> PREP")
        self._mode("PREP")
        time.sleep(self.prep_wait_s)
        log.info("PREP settled (%.1fs) -> WALK", self.prep_wait_s)
        self._mode("WALK")

    def command(self, cmd):
        vx = _snap(_clamp(cmd.vx, -self.limits.vx_max, self.limits.vx_max), config.V_DEADBAND)
        vy = _snap(_clamp(cmd.vy, -self.limits.vy_max, self.limits.vy_max), config.V_DEADBAND)
        wz = _snap(_clamp(cmd.wz, -self.limits.wz_max, self.limits.wz_max), config.WZ_DEADBAND)
        self.client.walk(vx, vy, wz)

    def zero(self):
        self.client.walk(0.0, 0.0, 0.0)

    def safe_rest(self):
        self.zero()
        time.sleep(0.3)
        self._mode("PREP")

    def shutdown(self):
        if self._shutdown_done:
            return
        self._shutdown_done = True
        try:
            self.zero()
            time.sleep(0.2)
            self._mode("DAMP")
            log.info("shutdown: robot DAMP")
        except Exception:
            log.exception("shutdown FAILED - robot may still be active!")

    def _mode(self, mode):
        if hasattr(self.client, "change_mode"):
            self.client.change_mode(mode)
        else:
            getattr(self.client, mode.lower(), lambda: None)()
        self.mode = mode

    def install_guards(self):
        if self._guards:
            return
        self._guards = True
        atexit.register(self.shutdown)
        for sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
            if sig is None:
                continue
            try:
                signal.signal(sig, self._on_signal)
            except (ValueError, OSError):
                pass

    def _on_signal(self, signum, frame):
        log.warning("signal %s -> shutdown", signum)
        self.shutdown()
        raise SystemExit(0)

    def __enter__(self):
        self.install_guards()
        return self

    def __exit__(self, et, e, tb):
        self.shutdown()
        return False


def make_client(ip):
    """Real SDK client factory (lazy import). VERIFY-ON-ROBOT: bind to boosteros or
    booster_robotics_sdk once the inspect pass confirms which is on jscore."""
    import booster_robotics_sdk_python as sdk  # noqa: F401
    for name in ("B1LocoClient", "BoosterClient", "LocoClient"):
        if hasattr(sdk, name):
            return getattr(sdk, name)(ip)
    raise RuntimeError("No known Booster client class; run the inspect pass and bind make_client().")
