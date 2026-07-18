#!/bin/bash
set -eo pipefail

for ros_setup in \
    /opt/ros/humble/setup.bash \
    /opt/booster/BoosterRos2/install/setup.bash \
    /opt/booster/BoosterRos2Interface/install/setup.bash
do
    if [[ -f "$ros_setup" ]]; then
        source "$ros_setup"
    fi
done

set -u
rerun_url="${NERO_RERUN_URL:-host.docker.internal:9876}"
exec uv run --extra viz nero-rerun --connect "$rerun_url" "$@"
