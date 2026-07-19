#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

if [[ "$(uname -s)" != "Linux" || ! "$(uname -m)" =~ ^(aarch64|arm64)$ ]]; then
  echo "The QNN runtime must be installed on Linux ARM64." >&2
  exit 1
fi

uv_bin="${NERO_UV_BIN:-$(command -v uv || true)}"
if [[ -z "${uv_bin}" && -x /home/booster/.local/bin/uv ]]; then
  uv_bin=/home/booster/.local/bin/uv
fi
if [[ -z "${uv_bin}" ]]; then
  echo "uv is required to install the isolated QNN runtime." >&2
  exit 1
fi

"${uv_bin}" python install 3.11
"${uv_bin}" venv --python 3.11 --allow-existing .venv-qnn
"${uv_bin}" pip install --python .venv-qnn/bin/python \
  'numpy==1.26.4' \
  'onnxruntime==1.26.0' \
  'onnxruntime-qnn==2.4.0'

ORT_QNN_ENABLE_CPU_BACKEND=1 .venv-qnn/bin/python - <<'PY'
import onnxruntime as ort
import onnxruntime_qnn as qnn_ep

ort.register_execution_provider_library(
    "QNNExecutionProvider", qnn_ep.get_library_path()
)
devices = [d for d in ort.get_ep_devices() if d.ep_name == "QNNExecutionProvider"]
if not devices:
    raise SystemExit("QNN plugin registered but no Qualcomm device was exposed")
print(
    "Isolated QNN plugin ready:",
    f"onnxruntime={ort.__version__}",
    f"onnxruntime-qnn={qnn_ep.__version__}",
)
PY
