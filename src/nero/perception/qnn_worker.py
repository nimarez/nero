"""Isolated Python 3.11 worker for the ONNX Runtime QNN plugin."""

from __future__ import annotations

import argparse
import json
import os
import socket
import struct
import time
from pathlib import Path

import numpy as np


QNN_PROVIDER = "QNNExecutionProvider"
_IMAGE_SHAPE = (1, 3, 256, 256)
_TEXT_SHAPE = (1, 1, 512)
_OUTPUT_SHAPE = (1, 5, 1344)


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = connection.recv(size - len(chunks))
        if not chunk:
            raise EOFError("QNN worker connection closed")
        chunks.extend(chunk)
    return bytes(chunks)


def _send_message(connection: socket.socket, kind: bytes, payload: bytes) -> None:
    connection.sendall(kind + struct.pack("<I", len(payload)) + payload)


def _create_session(model_path: Path):
    # QNN 2.4 discovers only /dev/fastrpc-cdsp*. The K1 image exposes the same
    # CDSP through the older /dev/adsprpc-smd name, so retain QNN's documented
    # compatibility registration device and explicitly bind it to HTP below.
    if Path("/dev/adsprpc-smd").exists():
        os.environ["ORT_QNN_ENABLE_CPU_BACKEND"] = "1"

    import onnxruntime as ort
    import onnxruntime_qnn as qnn_ep

    package_dir = Path(qnn_ep.get_library_path()).parent
    if not os.environ.get("ADSP_LIBRARY_PATH"):
        os.environ["ADSP_LIBRARY_PATH"] = f"{package_dir};/usr/lib/rfsa/adsp;/dsp"
    ort.register_execution_provider_library(QNN_PROVIDER, qnn_ep.get_library_path())
    devices = [device for device in ort.get_ep_devices() if device.ep_name == QNN_PROVIDER]
    if not devices:
        raise RuntimeError(
            "QNN plugin registered but no Qualcomm device was exposed; "
            "expected /dev/fastrpc-cdsp* or the K1 /dev/adsprpc-smd device"
        )

    options = ort.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    options.add_provider_for_devices(
        devices,
        {
            "backend_path": qnn_ep.get_qnn_htp_path(),
            "htp_graph_finalization_optimization_mode": os.getenv(
                "NERO_QNN_GRAPH_OPTIMIZATION", "3"
            ),
            "offload_graph_io_quantization": "0",
        },
    )
    started = time.perf_counter()
    session = ort.InferenceSession(str(model_path), sess_options=options)
    load_seconds = time.perf_counter() - started
    inputs = {item.name: tuple(item.shape) for item in session.get_inputs()}
    expected = {"images": _IMAGE_SHAPE, "text_features": _TEXT_SHAPE}
    if inputs != expected:
        raise RuntimeError(f"unexpected QNN model inputs {inputs}; expected {expected}")
    outputs = session.get_outputs()
    if len(outputs) != 1 or tuple(outputs[0].shape) != _OUTPUT_SHAPE:
        raise RuntimeError(
            f"unexpected QNN model output {[(item.name, tuple(item.shape)) for item in outputs]}"
        )
    run_options = ort.RunOptions()
    run_options.add_run_config_entry(
        "qnn.perf_mode", os.getenv("NERO_QNN_PERFORMANCE_MODE", "burst")
    )
    return session, run_options, ort.__version__, qnn_ep.__version__, load_seconds


def serve(model_path: Path, ipc_fd: int) -> None:
    connection = socket.socket(fileno=ipc_fd)
    try:
        session, run_options, ort_version, qnn_version, load_seconds = _create_session(model_path)
        _send_message(
            connection,
            b"R",
            json.dumps(
                {
                    "provider": QNN_PROVIDER,
                    "ort_version": ort_version,
                    "qnn_version": qnn_version,
                    "load_seconds": load_seconds,
                }
            ).encode(),
        )
        image_bytes = int(np.prod(_IMAGE_SHAPE)) * np.dtype(np.float32).itemsize
        text_bytes = int(np.prod(_TEXT_SHAPE)) * np.dtype(np.float32).itemsize
        while True:
            command = _recv_exact(connection, 1)
            if command == b"Q":
                return
            if command != b"I":
                raise RuntimeError(f"unknown QNN worker command {command!r}")
            images = np.frombuffer(_recv_exact(connection, image_bytes), dtype=np.float32).reshape(
                _IMAGE_SHAPE
            )
            text_features = np.frombuffer(
                _recv_exact(connection, text_bytes), dtype=np.float32
            ).reshape(_TEXT_SHAPE)
            started = time.perf_counter()
            output = session.run(
                None,
                {"images": images, "text_features": text_features},
                run_options,
            )[0]
            elapsed = time.perf_counter() - started
            output = np.ascontiguousarray(output, dtype=np.float32)
            if output.shape != _OUTPUT_SHAPE:
                raise RuntimeError(f"unexpected QNN output shape {output.shape}")
            connection.sendall(b"O" + struct.pack("<d", elapsed) + output.tobytes())
    except Exception as exc:
        try:
            _send_message(connection, b"E", f"{type(exc).__name__}: {exc}".encode())
        except OSError:
            pass
        raise
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--ipc-fd", type=int, required=True)
    args = parser.parse_args()
    serve(args.model.resolve(), args.ipc_fd)


if __name__ == "__main__":
    main()
