"""Compile and profile Nero's runtime-prompt detector on Qualcomm AI Hub."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def summarize_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Return stable headline metrics without the thousands of per-op entries."""
    summary = profile["execution_summary"]
    units = sorted({entry["compute_unit"] for entry in profile.get("execution_detail", [])})
    return {
        "inference_time_us": int(summary["estimated_inference_time"]),
        "inference_peak_memory_bytes": int(summary["estimated_inference_peak_memory"]),
        "first_load_time_us": int(summary["first_load_time"]),
        "warm_load_time_us": int(summary["warm_load_time"]),
        "compute_units": units,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="config/yolov8s-worldv2-open-vocab-256.onnx",
        help="Source ONNX model",
    )
    parser.add_argument("--device", default="QCS8550 (Proxy)")
    parser.add_argument("--device-os", default="12")
    parser.add_argument("--name", default="nero-yolov8s-worldv2-open-vocab-256")
    parser.add_argument("--compile-job", help="Reuse a successful compile job ID")
    parser.add_argument("--profile-job", help="Reuse a successful profile job ID")
    parser.add_argument("--output-dir", default="output/ai_hub")
    parser.add_argument("--download-model", action="store_true")
    return parser.parse_args()


def _require_success(job: Any, stage: str) -> None:
    status = job.get_status()
    if status.code != "SUCCESS":
        raise RuntimeError(f"AI Hub {stage} job failed: {status}")


def main() -> None:
    args = _parse_args()
    try:
        import qai_hub as hub
    except ImportError as exc:
        raise SystemExit("Install the AI Hub extra: uv sync --extra ai-hub") from exc

    client = hub.Client()
    device = hub.Device(args.device, os=args.device_os)
    if args.compile_job:
        compile_job = client.get_job(args.compile_job)
    else:
        compile_job = client.submit_compile_job(
            model=args.model,
            device=device,
            name=args.name,
            options="--target_runtime onnx",
        )
        compile_job.wait()
    _require_success(compile_job, "compile")
    target_model = compile_job.get_target_model()

    if args.profile_job:
        profile_job = client.get_job(args.profile_job)
    else:
        profile_job = client.submit_profile_job(
            model=target_model,
            device=device,
            name=f"{args.name}-profile",
        )
        profile_job.wait()
    _require_success(profile_job, "profile")

    profile = profile_job.download_profile()
    headline = summarize_profile(profile)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "profile.json").write_text(json.dumps(profile, indent=2, sort_keys=True))
    manifest = {
        "device": args.device,
        "device_os": args.device_os,
        "compile_job": compile_job.job_id,
        "profile_job": profile_job.job_id,
        "target_model": target_model.model_id,
        **headline,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    if args.download_model:
        target_model.download(str(output_dir / "optimized_model"))

    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
