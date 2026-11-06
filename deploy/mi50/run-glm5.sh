#!/bin/sh
# Start (or recreate) the GLM-5.3-Flash container on 4x MI50.  It takes every GPU, so it does not start beside the Qwen container.
# API: http://<server>:8095/v1 (OpenAI) and /v1/messages (Anthropic); the key comes from the config (docs/GLM5.md).
# Back to Qwen: docker stop strata-glm5 && ~/strata-deploy/run.sh
set -e
NAME=strata-glm5
IMAGE="${IMAGE:-strata-gfx906:0.2.10-mi50-glm5}"
if [ "$(docker inspect -f '{{.State.Running}}' strata-mi50 2>/dev/null)" = "true" ]; then
  echo "strata-mi50 is running: stop it first (docker stop strata-mi50)" >&2
  exit 1
fi
docker rm -f "$NAME" >/dev/null 2>&1 || true   # only this container
i=0; while docker inspect "$NAME" >/dev/null 2>&1 && [ $i -lt 60 ]; do sleep 1; i=$((i+1)); done   # removal can lag
docker run -d --name "$NAME" --restart unless-stopped \
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add 991 \
  --ipc=host --ulimit memlock=-1 --security-opt seccomp=unconfined \
  -p 8095:8080 \
  -v "$HOME/strata-data:/data" \
  -e HSA_ENABLE_SDMA=0 -e HSA_FORCE_FINE_GRAIN_PCIE=1 -e GPU_MAX_HW_QUEUES=8 \
  ${STRATA_EXTRA_ENV:-} \
  "$IMAGE"
echo "started $NAME ($IMAGE); logs: docker logs -f $NAME"
