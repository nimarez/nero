"""Install the official ORB-SLAM3 vocabulary used by Nero."""

from __future__ import annotations

import argparse
import hashlib
import io
import tarfile
import urllib.request
import sys
from pathlib import Path

VOCAB_URL = (
    "https://raw.githubusercontent.com/UZ-SLAMLab/ORB_SLAM3/master/"
    "Vocabulary/ORBvoc.txt.tar.gz"
)
VOCAB_ARCHIVE_SHA256 = (
    "ff2d0e82a69a8f4c5c002e3a0dff82a00e5872e1659fac1b996f41166b92693b"
)
VOCAB_SHA256 = "f8dd027f7a6cb88129821341194d7f2c75b77b3394257ddd0d2229863d1a3570"
DEBIAN_RUNTIME_PACKAGES = (
    "libopengl0 libglx0 libglu1-mesa libsm6 libice6 " "libx11-6 libxext6 libegl1 libgl1"
)


def check_native_runtime() -> None:
    """Verify the Linux wheels and their non-Python shared libraries."""
    if sys.platform != "linux":
        return
    try:
        import booster_robotics_sdk_python  # noqa: F401
        import orbslam3

        if not hasattr(orbslam3.Sensor, "IMU_RGBD"):
            raise RuntimeError("orbslam3.Sensor.IMU_RGBD is missing")
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "ORB-SLAM3 Linux runtime is incomplete. Run `uv sync`, and on "
            f"Debian/Ubuntu install: sudo apt-get install {DEBIAN_RUNTIME_PACKAGES}"
        ) from exc


def install_vocabulary(destination: Path, *, force: bool = False) -> Path:
    """Download, verify, and extract the official vocabulary atomically."""
    destination = destination.resolve()
    if destination.exists() and not force:
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if digest == VOCAB_SHA256:
            return destination
        raise RuntimeError(
            f"existing ORB vocabulary checksum mismatch: {digest}; "
            "rerun with --force to replace it"
        )

    with urllib.request.urlopen(VOCAB_URL, timeout=120) as response:
        archive = response.read()
    digest = hashlib.sha256(archive).hexdigest()
    if digest != VOCAB_ARCHIVE_SHA256:
        raise RuntimeError(f"ORB vocabulary checksum mismatch: {digest}")

    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
        member = bundle.getmember("ORBvoc.txt")
        source = bundle.extractfile(member)
        if source is None:
            raise RuntimeError("ORBvoc.txt is missing from the official archive")
        payload = source.read()
    payload_digest = hashlib.sha256(payload).hexdigest()
    if payload_digest != VOCAB_SHA256:
        raise RuntimeError(f"ORB vocabulary payload checksum mismatch: {payload_digest}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install the official ORB-SLAM3 vocabulary"
    )
    parser.add_argument("--output", type=Path, default=Path("config/ORBvoc.txt"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    check_native_runtime()
    path = install_vocabulary(args.output, force=args.force)
    print(f"ORB-SLAM3 vocabulary ready: {path}")


if __name__ == "__main__":
    main()
