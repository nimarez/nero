"""ZeroMQ transport - one XPUB/XSUB proxy, JSON-on-the-wire. drain_latest = latest-wins
so a control loop never acts on a stale queued message. Mocks are drop-in publishers on the
same topic, which is what lets the team build all 8 modules in parallel.

  python -m mrhack.bus proxy       # run the forwarder (once, on jscore)
  python -m mrhack.bus selftest    # prove pub -> proxy -> sub round-trips (needs pyzmq)
"""
from __future__ import annotations
import logging, sys, time
from . import config
from .contracts import from_wire, to_wire

log = logging.getLogger("bus")


def _zmq():
    import zmq
    return zmq


def pub_socket(host=None):
    zmq = _zmq()
    s = zmq.Context.instance().socket(zmq.PUB)
    s.connect(f"tcp://{host or config.MRHACK_HOST}:{config.PUB_PORT}")
    return s


def sub_socket(topics, host=None):
    zmq = _zmq()
    s = zmq.Context.instance().socket(zmq.SUB)
    s.connect(f"tcp://{host or config.MRHACK_HOST}:{config.SUB_PORT}")
    for t in topics:
        s.setsockopt_string(zmq.SUBSCRIBE, t)
    return s


def publish(sock, msg):
    topic, payload = to_wire(msg)
    sock.send_multipart([topic, payload])


def drain_latest(sock):
    """Non-blocking drain; keep only the newest message per topic (freshness > completeness)."""
    zmq = _zmq()
    latest = {}
    while True:
        try:
            topic, payload = sock.recv_multipart(flags=zmq.NOBLOCK)
        except zmq.Again:
            break
        latest[topic.decode()] = from_wire(topic, payload)
    return latest


def run_proxy():
    zmq = _zmq()
    ctx = zmq.Context.instance()
    xsub = ctx.socket(zmq.XSUB)
    xsub.bind(f"tcp://*:{config.PUB_PORT}")
    xpub = ctx.socket(zmq.XPUB)
    xpub.bind(f"tcp://*:{config.SUB_PORT}")
    log.info("proxy up: publishers -> :%d  |  subscribers <- :%d", config.PUB_PORT, config.SUB_PORT)
    try:
        zmq.proxy(xsub, xpub)
    except KeyboardInterrupt:
        pass


def _selftest():
    import threading
    from .contracts import RobotPose
    threading.Thread(target=run_proxy, daemon=True).start()
    time.sleep(0.4)
    sub = sub_socket("127.0.0.1", ) if False else sub_socket(["pose"], "127.0.0.1")
    pub = pub_socket("127.0.0.1")
    time.sleep(0.4)  # let the SUB subscription reach the proxy (PUB/SUB slow-joiner)
    got = None
    for _ in range(60):
        publish(pub, RobotPose(x=1.0, y=2.0, yaw=0.5, t=time.time()))
        time.sleep(0.03)
        latest = drain_latest(sub)
        if "pose" in latest:
            got = latest["pose"]
            break
    ok = got is not None and abs(got.x - 1.0) < 1e-9 and abs(got.y - 2.0) < 1e-9 and got.__class__.__name__ == "RobotPose"
    print(f"bus selftest: received {got}")
    print("SELFTEST:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s bus: %(message)s")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "proxy"
    if cmd == "proxy":
        run_proxy()
    elif cmd == "selftest":
        sys.exit(0 if _selftest() else 1)
    else:
        print("usage: python -m mrhack.bus [proxy|selftest]")
