"""Run the POS camera display, projector output, and browser calibrator."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from .calibration import CalibrationState, ProjectorCalibration
from .camera import RealSenseArucoCamera
from .render import render_projector_grid
from .server import CalibrationWebServer

logger = logging.getLogger(__name__)
DEFAULT_CALIBRATION = "~/.config/nero/projector-calibration.json"


def _sway(command: str) -> None:
    environment = {
        **os.environ,
        "SWAYSOCK": os.getenv("SWAYSOCK", "/run/user/1000/sway-ipc.1000.1292.sock"),
        "XDG_RUNTIME_DIR": os.getenv("XDG_RUNTIME_DIR", "/run/user/1000"),
    }
    subprocess.run(
        ["swaymsg", command],
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def _configure_sway_rules() -> None:
    _sway('for_window [title="^Nero Projector$"] move container to workspace 1, fullscreen enable')
    _sway('for_window [title="^Nero Camera$"] move container to workspace 2, fullscreen enable')


def _open_windows() -> None:
    _configure_sway_rules()
    for name in ("Nero Projector", "Nero Camera"):
        # Qt's default "expanded" OpenCV window adds an image toolbar and
        # status bar.  GUI_NORMAL keeps the physical outputs presentation-only.
        cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.imshow(name, np.zeros((1080, 1920, 3), dtype=np.uint8))
    cv2.waitKey(1)
    time.sleep(0.4)
    _sway('[title="^Nero Projector$"] move container to workspace 1, fullscreen enable')
    _sway('[title="^Nero Camera$"] move container to workspace 2, fullscreen enable')


def _load_or_default(path: Path) -> ProjectorCalibration:
    if not path.exists():
        return ProjectorCalibration()
    try:
        return ProjectorCalibration.load(path)
    except Exception:
        logger.exception("ignoring invalid projector calibration at %s", path)
        return ProjectorCalibration()


def run(args: argparse.Namespace) -> None:
    calibration_path = Path(args.calibration).expanduser()
    state = CalibrationState(_load_or_default(calibration_path))
    camera = RealSenseArucoCamera(marker_ids=(1, 2, 3, 4), marker_size_m=0.130).start()
    server = CalibrationWebServer(
        state=state,
        camera=camera,
        calibration_path=calibration_path,
        host=args.host,
        port=args.port,
    ).start()
    del server

    if args.no_display:
        logger.info("web calibration ready at http://%s:%d", args.host, args.port)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            camera.stop()
        return

    os.environ.setdefault("DISPLAY", ":0")
    os.environ.setdefault("XDG_RUNTIME_DIR", "/run/user/1000")
    _open_windows()
    last_calibration_version = -1
    last_camera_sequence = -1
    logger.info("calibration UI ready on port %d", args.port)

    try:
        while True:
            calibration, version, _ = state.snapshot()
            if version != last_calibration_version:
                started = time.perf_counter()
                cv2.imshow("Nero Projector", render_projector_grid(calibration))
                last_calibration_version = version
                logger.debug("projector update %.2f ms", (time.perf_counter() - started) * 1000)
            frame = camera.latest()
            if frame is not None and frame.sequence != last_camera_sequence:
                cv2.imshow("Nero Camera", frame.image)
                last_camera_sequence = frame.sequence
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            time.sleep(0.001)
    finally:
        camera.stop()
        cv2.destroyAllWindows()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--calibration", default=DEFAULT_CALIBRATION)
    parser.add_argument("--no-display", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
