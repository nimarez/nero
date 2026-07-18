#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_path="${NERO_OBJECT_MODEL:-${repo_root}/config/yolov8n.onnx}"
model_url="${NERO_OBJECT_MODEL_URL:-https://huggingface.co/webml/yolov8n/resolve/main/onnx/yolov8n.onnx}"
model_sha256="${NERO_OBJECT_MODEL_SHA256:-190ba5f1e61411a001683e349d6b2cdb0804c0dc67a5e34cd8ff6fd00ee54b4d}"

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
    exit 0
  fi
  echo "Existing object model checksum mismatch: ${actual_sha256}" >&2
  echo "Downloading a verified replacement." >&2
fi

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
