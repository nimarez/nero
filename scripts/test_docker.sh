#!/bin/sh
set -eu

platform="${NERO_DOCKER_PLATFORM:-linux/arm64}"
image="${NERO_DOCKER_IMAGE:-nero-test:local}"

docker build --platform "$platform" --file Dockerfile.test --tag "$image" .
docker run --rm --platform "$platform" "$image"
