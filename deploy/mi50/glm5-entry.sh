#!/bin/sh
# GLM-5.3-Flash container entry point.  The pack and the config live on the /data volume (docs/GLM5.md); the config is not named
# strata-*.json, so the Qwen image's setup never takes it.
set -e
cd /opt/strata
D="${STRATA_DATA:-/data}"
CFG="${GLM5_CONFIG:-$D/config/glm5.json}"
[ -f "$CFG" ] || { echo "no $CFG: see docs/GLM5.md" >&2; exit 2; }
exec .venv/bin/python serve/server.py --engine strata --config "$CFG" --host 0.0.0.0 --port "${PORT:-8080}"
