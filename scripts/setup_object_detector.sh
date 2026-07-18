#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_path="${NERO_OBJECT_MODEL:-${repo_root}/config/yolov8s-worldv2.pt}"
model_url="${NERO_OBJECT_MODEL_URL:-https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-worldv2.pt}"
model_sha256="${NERO_OBJECT_MODEL_SHA256:-9b2c17ab6124a913e9b3a5c170617920d91b0f01111a8479da69f00e2cf27792}"

checksum() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

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

if uv run python -c 'import torch, ultralytics' >/dev/null 2>&1; then
  uv pip install --no-deps ftfy regex tqdm
  uv pip install --no-deps \
    'git+https://github.com/ultralytics/CLIP.git@a57ec09a1668b6a5905ff323d734701f8d11d0e2'
  uv run python -c 'import clip; clip.load("ViT-B/32", device="cpu"); print("Open-vocabulary text encoder ready:", clip.__file__)'
else
  echo "YOLO-World requires the K1 image's torch and ultralytics runtime." >&2
  exit 1
fi
