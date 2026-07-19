#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

check_provider() {
  uv run python - <<'PY'
import onnxruntime as ort

providers = ort.get_available_providers()
if "QNNExecutionProvider" not in providers:
    raise SystemExit(
        "QNNExecutionProvider is absent; available providers: " + ", ".join(providers)
    )
print("ONNX Runtime QNN provider ready:", ort.__version__, providers)
PY
}

if check_provider 2>/dev/null; then
  exit 0
fi

if [[ "${1:-}" != "--build" ]]; then
  echo "The installed ONNX Runtime does not contain QNNExecutionProvider." >&2
  echo "Run scripts/setup_qnn_runtime.sh --build on the K1, or set" >&2
  echo "NERO_QNN_ORT_WHEEL to a compatible Linux ARM64 QNN wheel." >&2
  exit 1
fi

if [[ "$(uname -s)" != "Linux" || ! "$(uname -m)" =~ ^(aarch64|arm64)$ ]]; then
  echo "The QNN runtime must be built or installed on Linux ARM64." >&2
  exit 1
fi

if [[ -n "${NERO_QNN_ORT_WHEEL:-}" ]]; then
  if [[ ! -f "${NERO_QNN_ORT_WHEEL}" ]]; then
    echo "QNN wheel does not exist: ${NERO_QNN_ORT_WHEEL}" >&2
    exit 1
  fi
  uv pip install --reinstall "${NERO_QNN_ORT_WHEEL}"
  check_provider
  exit 0
fi

qnn_search_root="${QNN_SDK_ROOT:-${QNN_SDK_HOME:-/opt/qcom/qirp-sdk}}"
if [[ ! -d "${qnn_search_root}" ]]; then
  echo "QNN SDK root not found: ${qnn_search_root}" >&2
  echo "Set QNN_SDK_ROOT to the QAIRT/QNN SDK root on the K1." >&2
  exit 1
fi
qnn_header="$(find "${qnn_search_root}" -type f -path '*/QNN/QnnInterface.h' -print -quit)"
if [[ -n "${qnn_header}" ]]; then
  qnn_home="$(cd "$(dirname "${qnn_header}")/../.." && pwd)"
else
  qnn_home="${qnn_search_root}"
fi
qnn_backend="$(find "${qnn_home}" -type f -path '*aarch64*' -name 'libQnnHtp.so' -print -quit)"
if [[ -z "${qnn_backend}" ]]; then
  qnn_backend="$(find "${qnn_home}" -type f -name 'libQnnHtp.so' -print -quit)"
fi
if [[ -z "${qnn_header}" || -z "${qnn_backend}" ]]; then
  echo "The SDK root must contain QNN/QnnInterface.h and libQnnHtp.so: ${qnn_home}" >&2
  exit 1
fi

for command in cmake git ninja; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    echo "Missing build tool: ${command}" >&2
    exit 1
  fi
done

cache_base="${XDG_CACHE_HOME:-${HOME}/.cache}/nero"
source_dir="${cache_base}/onnxruntime-1.23.2"
build_dir="${cache_base}/onnxruntime-qnn-build-1.23.2"
mkdir -p "${cache_base}"
if [[ ! -d "${source_dir}/.git" ]]; then
  git clone --branch v1.23.2 --depth 1 --recursive --shallow-submodules \
    https://github.com/microsoft/onnxruntime.git "${source_dir}"
fi

jobs="${NERO_QNN_BUILD_JOBS:-2}"
"${source_dir}/build.sh" \
  --config Release \
  --build_shared_lib \
  --build_wheel \
  --skip_tests \
  --parallel "${jobs}" \
  --use_qnn shared_lib \
  --qnn_home "${qnn_home}" \
  --cmake_generator Ninja \
  --build_dir "${build_dir}"

wheel="$(find "${build_dir}" -type f -name 'onnxruntime-*.whl' -print -quit)"
if [[ -z "${wheel}" ]]; then
  echo "ONNX Runtime build completed without producing a wheel." >&2
  exit 1
fi
uv pip install --reinstall "${wheel}"
export NERO_QNN_BACKEND_PATH="${NERO_QNN_BACKEND_PATH:-${qnn_backend}}"
check_provider
echo "Use this runtime backend path in K1 shells:"
echo "export NERO_QNN_BACKEND_PATH=${NERO_QNN_BACKEND_PATH}"
