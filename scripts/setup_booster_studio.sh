#!/bin/sh
set -eu

# ROS 2 and Booster's SDK are supplied by the virtual K1 image, not PyPI. Keep
# Nero's locked dependencies in uv while allowing those vendor packages through.
uv venv --clear --system-site-packages
uv sync --all-groups --locked
