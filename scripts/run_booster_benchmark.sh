#!/bin/bash
set -eo pipefail

ros_setup="${NERO_ROS_SETUP:-/opt/ros/humble/setup.bash}"
if [[ ! -f "$ros_setup" ]]; then
    echo "ROS 2 setup not found: $ros_setup" >&2
    exit 1
fi

source "$ros_setup"
set -u
exec uv run nero-sim-benchmark "$@"
