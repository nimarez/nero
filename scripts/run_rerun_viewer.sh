#!/bin/bash
set -euo pipefail

exec uv run --extra viz rerun --memory-limit "${NERO_RERUN_MEMORY_LIMIT:-4GB}" "$@"
