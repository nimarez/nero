"""Install and verify Nero's AI Hub optimized QCS8550 detector artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path


DEFAULT_TARGET_MODEL = "mn09g883n"
DEFAULT_OUTPUT_DIR = Path("config/yolov8s-worldv2-open-vocab-256-qnn")
EXPECTED_FILES = {
    "model.onnx": {
        "sha256": "e97a5ad507e8cffb5ab99f59d2aa2217cd8b6a9d479da5bc1003b61c0110cf23",
        "size": 117381,
    },
    "model.data": {
        "sha256": "b23b2b791571864d5eaeb2e042052ca49a976a4f344182d2c205552745a7670a",
        "size": 50983040,
    },
}
MANIFEST_NAME = "manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_files(directory: Path, expected: dict | None = None) -> None:
    expected = EXPECTED_FILES if expected is None else expected
    for name, metadata in expected.items():
        path = directory / name
        if not path.is_file():
            raise RuntimeError(f"QNN artifact is missing {path}")
        size = path.stat().st_size
        if size != metadata["size"]:
            raise RuntimeError(
                f"QNN artifact size mismatch for {name}: {size} != {metadata['size']}"
            )
        actual = _sha256(path)
        if actual != metadata["sha256"]:
            raise RuntimeError(
                f"QNN artifact checksum mismatch for {name}: {actual}"
            )


def verify_qnn_artifact(directory: str | Path) -> Path:
    """Verify manifest, file sizes, and SHA-256 digests; return the ONNX path."""
    root = Path(directory)
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise RuntimeError(f"QNN artifact manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"QNN artifact manifest is invalid: {exc}") from exc
    if manifest.get("schema_version") != 1:
        raise RuntimeError("unsupported QNN artifact manifest schema")
    if manifest.get("target_model") != DEFAULT_TARGET_MODEL:
        raise RuntimeError(
            f"unexpected QNN target model {manifest.get('target_model')!r}"
        )
    if manifest.get("files") != EXPECTED_FILES:
        raise RuntimeError("QNN artifact manifest does not match Nero's pinned artifact")
    _check_files(root)
    return root / "model.onnx"


def _extract_archive(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as bundle:
        members = {}
        for info in bundle.infolist():
            name = Path(info.filename).name
            if info.is_dir() or name not in EXPECTED_FILES:
                continue
            if name in members:
                raise RuntimeError(f"AI Hub archive contains duplicate {name}")
            if info.file_size != EXPECTED_FILES[name]["size"]:
                raise RuntimeError(
                    f"AI Hub archive size mismatch for {name}: {info.file_size}"
                )
            members[name] = info
        if set(members) != set(EXPECTED_FILES):
            raise RuntimeError(
                f"AI Hub archive must contain {sorted(EXPECTED_FILES)}, got {sorted(members)}"
            )
        for name, info in members.items():
            target = destination / name
            with bundle.open(info) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def install_qnn_artifact(archive: str | Path, output_dir: str | Path) -> Path:
    """Install a pinned AI Hub archive atomically after full verification."""
    archive_path = Path(archive)
    if not archive_path.is_file():
        raise RuntimeError(f"AI Hub model archive is missing: {archive_path}")
    destination = Path(output_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}-", dir=destination.parent
    ) as temporary:
        staging = Path(temporary)
        _extract_archive(archive_path, staging)
        _check_files(staging)
        manifest = {
            "schema_version": 1,
            "target_model": DEFAULT_TARGET_MODEL,
            "files": EXPECTED_FILES,
        }
        (staging / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        destination.mkdir(parents=True, exist_ok=True)
        for name in (*EXPECTED_FILES, MANIFEST_NAME):
            os.replace(staging / name, destination / name)
    return verify_qnn_artifact(destination)


def _download_target_model(target_model: str, output: Path) -> Path:
    if target_model != DEFAULT_TARGET_MODEL:
        raise RuntimeError(
            f"only pinned target model {DEFAULT_TARGET_MODEL} is accepted"
        )
    try:
        import qai_hub as hub
    except ImportError as exc:
        raise RuntimeError("install the AI Hub extra: uv sync --extra ai-hub") from exc
    return Path(hub.get_model(target_model).download(str(output)))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--archive", type=Path, help="Use an existing AI Hub zip")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--verify-only", action="store_true", help="Verify the installed artifact"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.verify_only:
        model = verify_qnn_artifact(args.output_dir)
    elif args.archive:
        model = install_qnn_artifact(args.archive, args.output_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="nero-ai-hub-") as temporary:
            archive = _download_target_model(
                args.target_model, Path(temporary) / "model.zip"
            )
            model = install_qnn_artifact(archive, args.output_dir)
    print(f"Verified QCS8550 QNN detector: {model}")


if __name__ == "__main__":
    main()
