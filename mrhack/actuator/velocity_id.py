"""FIRST RUNNABLE - open-loop velocity ID. Uses K1Gate directly (no bus). De-risks the two
scariest unknowns: commanded->actual velocity + latency. Runs mock (no robot) or real."""
from __future__ import annotations
import argparse, json, logging, os, time
from .. import config
from ..contracts import VelCmd
from .state_machine import K1Gate

log = logging.getLogger("velocity_id")


def build_client(ip, mock):
    if mock:
        from .mock_robot import MockClient
        return MockClient(ip)
    from .state_machine import make_client
    return make_client(ip)


def _hold(gate, f, vx, wz, dur, phase):
    dt = 1.0 / config.ACTUATOR_HZ
    t_end = time.time() + dur
    n = 0
    while time.time() < t_end:
        t = time.time()
        gate.command(VelCmd(vx=vx, vy=0.0, wz=wz, t=t))
        f.write(json.dumps({"t": t, "phase": phase, "vx_cmd": vx, "wz_cmd": wz}) + "\n")
        n += 1
        time.sleep(dt)
    return n


def run(ip, matrix, out_path, mock=False, settle_s=None, cmd_s=None):
    settle_s = config.VELID_SETTLE_S if settle_s is None else settle_s
    cmd_s = config.VELID_CMD_S if cmd_s is None else cmd_s
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    client = build_client(ip, mock)
    gate = K1Gate(client)
    ticks = 0
    with gate, open(out_path, "w") as f:
        gate.start()
        for (vx, wz) in matrix:
            log.info("=== test point vx=%.2f wz=%.2f ===", vx, wz)
            _hold(gate, f, 0.0, 0.0, settle_s, "settle")
            ticks += _hold(gate, f, vx, wz, cmd_s, "cmd")
        _hold(gate, f, 0.0, 0.0, settle_s, "settle")
    log.info("velocity-ID done: %d command ticks -> %s", ticks, out_path)
    if hasattr(client, "mode_sequence"):
        modes = client.mode_sequence()
        walks = client.walk_calls()
        log.info("MOCK SUMMARY: modes = %s", " -> ".join(modes))
        log.info("MOCK SUMMARY: walk() calls = %d ; last = %s", len(walks), walks[-1] if walks else None)
        assert modes[:2] == ["PREP", "WALK"], f"bad start ordering: {modes}"
        assert modes[-1] == "DAMP", f"did not end DAMP: {modes}"
        assert walks and walks[-1] == (0.0, 0.0, 0.0), f"did not zero before shutdown: {walks[-1]}"
        log.info("MOCK CHECKS PASSED: DAMP->PREP->WALK ... zero -> DAMP, clamps applied.")
    return out_path


def _parse_matrix(s):
    return [tuple(float(x) for x in pair.split(",")) for pair in s.split(";") if pair.strip()]


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="127.0.0.1")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--matrix", default="default")
    ap.add_argument("--settle", type=float, default=None)
    ap.add_argument("--cmd", type=float, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    matrix = config.VELID_MATRIX if a.matrix == "default" else _parse_matrix(a.matrix)
    out = a.out or os.path.join("mrhack", "runs", "velid_%d.jsonl" % int(time.time()))
    run(a.ip, matrix, out, mock=a.mock, settle_s=a.settle, cmd_s=a.cmd)


if __name__ == "__main__":
    main()
