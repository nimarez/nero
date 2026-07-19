FROM ros:humble-ros-base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH=/app/src

RUN apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends python3-numpy python3-scipy \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY src/nero ./src/nero

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["python3", "-m", "nero.vive.ros_bridge", "--latest-file", "/run/nero/vive_pose.json"]
