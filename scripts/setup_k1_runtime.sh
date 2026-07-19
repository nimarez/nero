#!/bin/bash
set -eo pipefail

# ROS 2 and Booster message packages are supplied by the real/virtual K1 image,
# not PyPI. Source any installed prefixes before exposing them to uv's venv.
for setup in \
    /opt/ros/humble/setup.bash \
    /opt/booster/BoosterAgent/install/setup.bash
do
    if [[ -f "$setup" ]]; then
        source "$setup"
    fi
done

if ! command -v ros2 >/dev/null 2>&1; then
    echo "ROS 2 was not found; run this inside a real or virtual K1 terminal." >&2
    exit 1
fi

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
if [[ "${NERO_RECREATE_VENV:-0}" == "1" ]] || \
  [[ ! -x .venv/bin/python ]] || \
  ! grep -Eqi '^include-system-site-packages = true$' .venv/pyvenv.cfg
then
  uv venv --clear --system-site-packages
fi
# Inexact sync preserves vendor and locally built QNN wheels that are
# intentionally outside the cross-platform lock file.
uv sync --all-groups --locked --inexact

if [[ -f /opt/booster/perception_info.yaml ]]; then
    ./scripts/configure_k1_perception.sh
fi
