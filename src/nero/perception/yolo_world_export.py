"""Export YOLO-World with a runtime text embedding input.

Ultralytics normally bakes the selected vocabulary into exported models. Nero
follows one target at a time, so this exporter fixes the class count at one but
keeps its 512-dimensional CLIP feature as an input. The resulting graph remains
open-vocabulary without running the visual backbone in PyTorch.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="config/yolov8s-worldv2.pt", help="YOLO-World v2 checkpoint"
    )
    parser.add_argument(
        "--output",
        default="config/yolov8s-worldv2-open-vocab-256.onnx",
        help="Output ONNX model",
    )
    parser.add_argument("--imgsz", type=int, default=256, help="Square inference resolution")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset")
    parser.add_argument(
        "--no-verify", action="store_true", help="Skip ONNX checker and numerical comparison"
    )
    return parser.parse_args()


def _validate_imgsz(imgsz: int) -> None:
    if imgsz < 256 or imgsz % 32:
        raise ValueError("--imgsz must be at least 256 and divisible by 32")


def export_model(model_path: Path, output_path: Path, imgsz: int, opset: int) -> None:
    """Export the one-target visual graph while retaining text features as input."""
    import torch
    from ultralytics import YOLOWorld

    class OneTargetYOLOWorld(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, images: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
            result = self.inner(images, txt_feats=text_features)
            return result[0] if isinstance(result, (tuple, list)) else result

    world = YOLOWorld(str(model_path))
    inner = world.model.eval()
    head = inner.model[-1]
    head.nc = 1
    head.no = head.nc + head.reg_max * 4
    inner.names = {0: "target"}

    feature_width = int(inner.txt_feats.shape[-1])
    images = torch.zeros(1, 3, imgsz, imgsz)
    text_features = torch.randn(1, 1, feature_width)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    wrapper = OneTargetYOLOWorld(inner).eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (images, text_features),
            output_path,
            opset_version=opset,
            input_names=["images", "text_features"],
            output_names=["detections"],
            dynamo=False,
        )


def verify_model(model_path: Path, output_path: Path, imgsz: int) -> tuple[float, float]:
    """Check the graph and compare it with the PyTorch source on deterministic data."""
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from ultralytics import YOLOWorld

    onnx.checker.check_model(onnx.load(output_path))
    torch.manual_seed(7)
    images = torch.rand(1, 3, imgsz, imgsz)

    world = YOLOWorld(str(model_path))
    inner = world.model.eval()
    head = inner.model[-1]
    head.nc = 1
    head.no = head.nc + head.reg_max * 4
    inner.names = {0: "target"}
    text_features = torch.randn(1, 1, int(inner.txt_feats.shape[-1]))
    text_features /= text_features.norm(dim=-1, keepdim=True)

    with torch.no_grad():
        expected = inner(images, txt_feats=text_features)[0].numpy()
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    actual = session.run(
        None,
        {"images": images.numpy(), "text_features": text_features.numpy()},
    )[0]
    difference = np.abs(actual - expected)
    return float(difference.max()), float(difference.mean())


def main() -> None:
    args = _parse_args()
    _validate_imgsz(args.imgsz)
    model_path = Path(args.model)
    output_path = Path(args.output)
    export_model(model_path, output_path, args.imgsz, args.opset)
    print(f"Exported runtime-prompt YOLO-World: {output_path}")
    if not args.no_verify:
        maximum, mean = verify_model(model_path, output_path, args.imgsz)
        print(f"ONNX parity: max_abs={maximum:.6f} mean_abs={mean:.6f}")


if __name__ == "__main__":
    main()
