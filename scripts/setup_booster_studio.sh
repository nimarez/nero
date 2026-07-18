#!/bin/sh
set -eu

# ROS 2 and Booster's SDK are supplied by the virtual K1 image, not PyPI. Keep
# Nero's locked dependencies in uv while allowing those vendor packages through.
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
uv venv --clear --system-site-packages
uv sync --all-groups --locked
