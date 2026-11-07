#!/bin/sh
# Start (or recreate) Qwen on the all-cards split engine, 4x MI50. It takes every GPU and port 8095, so it does not start beside another server.
# Run deploy/mi50/run.sh once first: setup downloads the model and writes the config this one is made from.
# Back to the layer split: docker stop strata-split && deploy/mi50/run.sh
set -e
NAME=strata-split
IMAGE="${IMAGE:-strata-mi50-split}"
DATA="${DATA:-$HOME/strata-data}"
PORT="${PORT:-8095}"
RENDER_GID="$(getent group render | cut -d: -f3)"
for other in strata-mi50 strata-glm5; do
  if [ "$(docker inspect -f '{{.State.Running}}' "$other" 2>/dev/null)" = "true" ]; then
    echo "$other is running: stop it first (docker stop $other)" >&2
    exit 1
  fi
done
docker rm -f "$NAME" >/dev/null 2>&1 || true   # only this container
i=0; while docker inspect "$NAME" >/dev/null 2>&1 && [ $i -lt 60 ]; do sleep 1; i=$((i+1)); done   # removal can lag
# shellcheck disable=SC2086
docker run -d --name "$NAME" --restart unless-stopped \
  --device=/dev/kfd --device=/dev/dri --group-add video ${RENDER_GID:+--group-add $RENDER_GID} \
  --ipc=host --ulimit memlock=-1 --security-opt seccomp=unconfined \
  -p "$PORT:8080" \
  -v "$DATA:/data" \
  -e HSA_ENABLE_SDMA=0 -e HSA_FORCE_FINE_GRAIN_PCIE=1 -e GPU_MAX_HW_QUEUES=8 \
  ${STRATA_EXTRA_ENV:-} \
  "$IMAGE"
echo "started $NAME ($IMAGE) on port $PORT; logs: docker logs -f $NAME"
