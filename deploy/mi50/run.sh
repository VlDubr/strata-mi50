#!/bin/sh
# Start (or recreate) the strata-mi50 server container.
#
#   deploy/mi50/run.sh                         start with the saved config in $DATA/config
#   SETUP_ARGS="--family unsloth ..." deploy/mi50/run.sh
#                                              first start: setup.py downloads and prepares the model
#   STRATA_SETUP=1 SETUP_ARGS="..." deploy/mi50/run.sh
#                                              run setup again (another model, other options)
#
# API: http://<host>:$PORT/v1 (OpenAI), /v1/messages (Anthropic), web chat at http://<host>:$PORT/
# Logs: docker logs -f $NAME, and the engine log in $DATA/logs.
set -e
NAME="${NAME:-strata-mi50}"
IMAGE="${IMAGE:-strata-mi50}"
DATA="${DATA:-$HOME/strata-data}"
PORT="${PORT:-8095}"
RENDER_GID="$(getent group render | cut -d: -f3)"

mkdir -p "$DATA"
docker rm -f "$NAME" >/dev/null 2>&1 || true   # only this container
i=0; while docker inspect "$NAME" >/dev/null 2>&1 && [ $i -lt 60 ]; do sleep 1; i=$((i+1)); done   # removal can lag
# SETUP_ARGS, STRATA_SETUP, SETUP_ONLY and START_ARGS pass through only when they are set.
# shellcheck disable=SC2086
docker run -d --name "$NAME" --restart unless-stopped \
  --device=/dev/kfd --device=/dev/dri --group-add video ${RENDER_GID:+--group-add $RENDER_GID} \
  --ipc=host --ulimit memlock=-1 --security-opt seccomp=unconfined \
  -p "$PORT:8080" \
  -v "$DATA:/data" \
  -e HSA_ENABLE_SDMA=0 -e HSA_FORCE_FINE_GRAIN_PCIE=1 -e GPU_MAX_HW_QUEUES=8 \
  -e SETUP_ARGS -e STRATA_SETUP -e SETUP_ONLY -e START_ARGS \
  ${STRATA_EXTRA_ENV:-} \
  "$IMAGE"
echo "started $NAME ($IMAGE) on port $PORT; logs: docker logs -f $NAME"
