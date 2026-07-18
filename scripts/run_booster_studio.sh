#!/bin/bash
set -euo pipefail

ros_setup="${NERO_ROS_SETUP:-/opt/ros/humble/setup.bash}"
if [[ ! -f "$ros_setup" ]]; then
    echo "ROS 2 setup not found: $ros_setup" >&2
    exit 1
fi

# Booster Studio installs rclpy and its native libraries in the ROS prefix.
# Sourcing this is required even when uv can see ordinary system packages.
source "$ros_setup"
exec uv run nero-booster-studio "$@"
