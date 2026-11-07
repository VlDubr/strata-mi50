#!/bin/sh
# Start (or recreate) the GLM-5.3-Flash container on 4x MI50.  It takes every GPU, so it does not start beside the Qwen container.
# API: http://<server>:8095/v1 (OpenAI) and /v1/messages (Anthropic); the key comes from the config (docs/GLM5.md).
# Back to Qwen: docker stop strata-glm5 && ~/strata-deploy/run.sh
# SYS_ADMIN: --ram-stripe reads the frames of its pages from /proc/self/pagemap (docs/GLM5.md).
# HSA_ENABLE_SDMA=1: --prompt-ring's copies go on the copy engines, beside the kernels.
set -e
NAME=strata-glm5
IMAGE="${IMAGE:-strata-gfx906:0.2.10-mi50-glm5}"
for other in strata-mi50 strata-split; do
  if [ "$(docker inspect -f '{{.State.Running}}' "$other" 2>/dev/null)" = "true" ]; then
    echo "$other is running: stop it first (docker stop $other)" >&2
    exit 1
  fi
done
docker rm -f "$NAME" >/dev/null 2>&1 || true   # only this container
i=0; while docker inspect "$NAME" >/dev/null 2>&1 && [ $i -lt 60 ]; do sleep 1; i=$((i+1)); done   # removal can lag
docker run -d --name "$NAME" --restart unless-stopped \
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add 991 \
  --ipc=host --ulimit memlock=-1 --security-opt seccomp=unconfined --cap-add SYS_ADMIN \
  -p 8095:8080 \
  -v "$HOME/strata-data:/data" \
  -e HSA_ENABLE_SDMA=1 -e HSA_FORCE_FINE_GRAIN_PCIE=1 -e GPU_MAX_HW_QUEUES=8 \
  ${STRATA_EXTRA_ENV:-} \
  "$IMAGE"
echo "started $NAME ($IMAGE); logs: docker logs -f $NAME"
