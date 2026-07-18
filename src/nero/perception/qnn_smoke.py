"""Fail-closed K1 smoke test for the QNN YOLO-World runtime."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from nero.perception.qnn_artifact import DEFAULT_OUTPUT_DIR, verify_qnn_artifact
from nero.perception.qnn_yolo_world import (
    QNNYoloWorldRuntime,
    YoloWorldPromptEncoder,
    decode_yolo_world,
    preprocess_yolo_world,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target", default="green can")
    parser.add_argument("--runs", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.runs < 1:
        raise SystemExit("--runs must be positive")
    model_path = verify_qnn_artifact(args.model_dir)

    started = time.perf_counter()
    runtime = QNNYoloWorldRuntime(model_path)
    load_seconds = time.perf_counter() - started
    encoder = YoloWorldPromptEncoder()
    text_features = encoder.encode(args.target)
    image, geometry = preprocess_yolo_world(
        np.zeros((448, 544, 3), dtype=np.uint8), 256
    )

    latencies = []
    output = None
    for _ in range(args.runs):
        output, elapsed = runtime.infer(image, text_features)
        latencies.append(elapsed * 1000.0)
    detections = decode_yolo_world(output, geometry, 0.5, 10)
    result = {
        "providers": runtime.providers,
        "load_ms": round(load_seconds * 1000.0, 3),
        "runs": args.runs,
        "inference_ms_median": round(statistics.median(latencies), 3),
        "inference_ms_max": round(max(latencies), 3),
        "output_shape": list(output.shape),
        "blank_image_detections": len(detections),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
