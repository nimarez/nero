#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_path="${NERO_OBJECT_MODEL:-${repo_root}/config/yolov8n.onnx}"
model_url="${NERO_OBJECT_MODEL_URL:-https://huggingface.co/webml/yolov8n/resolve/main/onnx/yolov8n.onnx}"
model_sha256="${NERO_OBJECT_MODEL_SHA256:-190ba5f1e61411a001683e349d6b2cdb0804c0dc67a5e34cd8ff6fd00ee54b4d}"

mkdir -p "$(dirname "${model_path}")"
if [[ -s "${model_path}" ]]; then
  echo "Object model already present: ${model_path}"
  exit 0
fi

temporary="${model_path}.download"
trap 'rm -f "${temporary}"' EXIT
curl --fail --location --retry 3 --output "${temporary}" "${model_url}"
actual_sha256="$(shasum -a 256 "${temporary}" | awk '{print $1}')"
if [[ "${actual_sha256}" != "${model_sha256}" ]]; then
  echo "Object model checksum mismatch: ${actual_sha256}" >&2
  exit 1
fi
mv "${temporary}" "${model_path}"
echo "Object model installed: ${model_path}"
