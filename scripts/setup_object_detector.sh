#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
if [[ "$(uname -s)" == "Linux" && "$(uname -m)" =~ ^(aarch64|arm64)$ ]]; then
  default_backend="yolo-world-qnn"
else
  default_backend="yolo-world"
fi
backend="${NERO_OBJECT_BACKEND:-${default_backend}}"

case "${backend}" in
  world|yolo-world|yoloworld)
    backend="yolo-world"
    default_model_path="${repo_root}/config/yolov8s-worldv2.pt"
    default_model_url="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-worldv2.pt"
    default_model_sha256="9b2c17ab6124a913e9b3a5c170617920d91b0f01111a8479da69f00e2cf27792"
    ;;
  qnn|yolo-world-qnn|yoloworld-qnn)
    backend="yolo-world-qnn"
    default_model_path="${repo_root}/config/yolov8s-worldv2-open-vocab-256-qnn/model.onnx"
    artifact_mode=1
    ;;
  modal|yolo-world-modal|yoloworld-modal)
    backend="yolo-world-modal"
    default_model_path="remote:yolov8s-worldv2.pt"
    remote_mode=1
    ;;
  yoloe)
    default_model_path="${repo_root}/config/yoloe-26n-seg.pt"
    default_model_url="https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-26n-seg.pt"
    default_model_sha256="1741c1f8da3cea47e2c01829c334a50dc0b9bbd05e685b90a3ce84fae32c8c1b"
    ;;
  *)
    echo "Unsupported object detector backend: ${backend}" >&2
    echo "Choose yolo-world-qnn, yolo-world-modal, yolo-world, or yoloe." >&2
    exit 2
    ;;
esac

model_path="${NERO_OBJECT_MODEL:-${default_model_path}}"
model_url="${NERO_OBJECT_MODEL_URL:-${default_model_url:-}}"
model_sha256="${NERO_OBJECT_MODEL_SHA256:-${default_model_sha256:-}}"

checksum() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

if [[ "${remote_mode:-0}" == 1 ]]; then
  : "${NERO_MODAL_URL:?Set NERO_MODAL_URL to the deployed detect endpoint}"
  : "${NERO_MODAL_KEY:?Set NERO_MODAL_KEY to the Modal proxy token ID}"
  : "${NERO_MODAL_SECRET:?Set NERO_MODAL_SECRET to the Modal proxy token secret}"
elif [[ "${artifact_mode:-0}" == 1 ]]; then
  uv run nero-install-qnn-model \
    --output-dir "$(dirname "${model_path}")" \
    --verify-only
else
  mkdir -p "$(dirname "${model_path}")"
  if [[ -s "${model_path}" ]]; then
    actual_sha256="$(checksum "${model_path}")"
    if [[ "${actual_sha256}" == "${model_sha256}" ]]; then
      echo "Object model ready (checksum verified): ${model_path}"
      model_ready=1
    else
      echo "Existing object model checksum mismatch: ${actual_sha256}" >&2
      echo "Downloading a verified replacement." >&2
    fi
  fi

  if [[ "${model_ready:-0}" != 1 ]]; then
    temporary="${model_path}.download"
    trap 'rm -f "${temporary}"' EXIT
    curl --fail --location --retry 3 --output "${temporary}" "${model_url}"
    actual_sha256="$(checksum "${temporary}")"
    if [[ "${actual_sha256}" != "${model_sha256}" ]]; then
      echo "Object model checksum mismatch: ${actual_sha256}" >&2
      exit 1
    fi
    mv "${temporary}" "${model_path}"
    echo "Object model installed: ${model_path}"
  fi
fi

if [[ "${backend}" == "yoloe" ]]; then
  text_model_path="${repo_root}/config/mobileclip2_b.ts"
  text_model_url="https://github.com/ultralytics/assets/releases/download/v8.4.0/mobileclip2_b.ts"
  text_model_sha256="35d7f213e4d75f38514e4656ad3cb91158bd33e3805d8ac349f23b186f66982f"
  if [[ -s "${text_model_path}" ]]; then
    actual_sha256="$(checksum "${text_model_path}")"
    if [[ "${actual_sha256}" == "${text_model_sha256}" ]]; then
      echo "YOLOE text model ready (checksum verified): ${text_model_path}"
      text_model_ready=1
    else
      echo "Existing YOLOE text model checksum mismatch: ${actual_sha256}" >&2
      echo "Downloading a verified replacement." >&2
    fi
  fi

  if [[ "${text_model_ready:-0}" != 1 ]]; then
    temporary="${text_model_path}.download"
    trap 'rm -f "${temporary}"' EXIT
    curl --fail --location --retry 3 --output "${temporary}" "${text_model_url}"
    actual_sha256="$(checksum "${temporary}")"
    if [[ "${actual_sha256}" != "${text_model_sha256}" ]]; then
      echo "YOLOE text model checksum mismatch: ${actual_sha256}" >&2
      exit 1
    fi
    mv "${temporary}" "${text_model_path}"
    echo "YOLOE text model installed: ${text_model_path}"
  fi
fi

if [[ "${remote_mode:-0}" != 1 ]] && \
  ! uv run python -c 'import torch' >/dev/null 2>&1
then
  echo "Open-vocabulary detection requires the K1 image's torch runtime." >&2
  exit 1
fi

if [[ "${remote_mode:-0}" == 1 ]]; then
  :
elif [[ "${backend}" == "yoloe" ]]; then
  if ! uv run python -c \
    'import re, ultralytics; from ultralytics import YOLOE; version = tuple(map(int, re.match(r"^(\d+)\.(\d+)\.(\d+)", ultralytics.__version__).groups())); assert version >= (8, 4, 0)' \
    >/dev/null 2>&1
  then
    # Reuse the K1 image's optimized torch build instead of resolving a second one.
    uv pip install --no-deps 'ultralytics==8.4.0'
  fi
elif [[ "${backend}" == "yolo-world-qnn" ]] && \
  ! uv run python -c 'from ultralytics.nn.text_model import build_text_model' \
    >/dev/null 2>&1
then
  echo "QNN YOLO-World requires the K1 image's ultralytics text runtime." >&2
  exit 1
elif ! uv run python -c 'from ultralytics import YOLOWorld' >/dev/null 2>&1; then
  echo "YOLO-World requires the K1 image's ultralytics runtime." >&2
  exit 1
fi

if [[ "${remote_mode:-0}" != 1 ]]; then
  uv pip install --no-deps ftfy regex tqdm
  uv pip install --no-deps \
    'git+https://github.com/ultralytics/CLIP.git@a57ec09a1668b6a5905ff323d734701f8d11d0e2'
fi

# Resolve the backend's text encoder now so the policy loop never downloads it.
NERO_OBJECT_BACKEND="${backend}" \
NERO_OBJECT_MODEL="${model_path}" \
NERO_YOLOE_TEXT_MODEL="${repo_root}/config/mobileclip2_b.ts" \
NERO_DETECTOR_PROCESS=0 \
NERO_YOLO_WARMUP=0 \
uv run python -c \
  'from nero.perception.object_detector import ObjectDetector; detector = ObjectDetector(); assert detector.initialize(); detector.close(); print("Open-vocabulary backend ready:", detector.backend, detector.model_path)'
