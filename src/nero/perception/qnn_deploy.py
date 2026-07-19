"""Transfer the verified QNN detector artifact from a Mac checkout to the K1."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

from nero.perception.qnn_artifact import DEFAULT_OUTPUT_DIR, verify_qnn_artifact


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="10.2.1.130")
    parser.add_argument("--user", default="booster")
    parser.add_argument("--repo", default="/home/booster/Workspace/nero")
    parser.add_argument("--uv", help="Absolute uv path on the robot")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model_path = verify_qnn_artifact(args.model_dir)
    artifact_dir = model_path.parent.resolve()
    relative_dir = Path("config/yolov8s-worldv2-open-vocab-256-qnn")
    remote_dir = Path(args.repo) / relative_dir
    remote_uv = args.uv or f"/home/{args.user}/.local/bin/uv"
    remote_script = " && ".join(
        (
            f"mkdir -p {shlex.quote(str(remote_dir))}",
            f"tar -C {shlex.quote(str(remote_dir))} -xf -",
            f"cd {shlex.quote(args.repo)}",
            f"{shlex.quote(remote_uv)} run nero-install-qnn-model --output-dir "
            f"{shlex.quote(str(relative_dir))} --verify-only",
        )
    )
    archive = subprocess.Popen(
        [
            "tar",
            "-C",
            str(artifact_dir),
            "-cf",
            "-",
            "model.onnx",
            "model.data",
            "manifest.json",
        ],
        stdout=subprocess.PIPE,
    )
    try:
        completed = subprocess.run(
            ["ssh", "-T", f"{args.user}@{args.host}", remote_script],
            stdin=archive.stdout,
            check=False,
        )
    finally:
        if archive.stdout is not None:
            archive.stdout.close()
    archive_status = archive.wait()
    if archive_status:
        raise SystemExit(archive_status)
    if completed.returncode:
        raise SystemExit(completed.returncode)
    print(f"QNN model transferred and verified at {args.user}@{args.host}:{remote_dir}")


if __name__ == "__main__":
    main()
