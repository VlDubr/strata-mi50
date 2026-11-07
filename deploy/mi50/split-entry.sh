#!/bin/sh
# All-cards split engine container entry point. The server starts the engine directly: the Qwen image's setup would put its own engine back.
# The config lives on the /data volume; it is not named strata-*.json, so the Qwen image's setup never takes it.
set -e
cd /opt/strata
D="${STRATA_DATA:-/data}"
CFG="${SPLIT_CONFIG:-$D/config/split.json}"
[ -f "$CFG" ] || .venv/bin/python split-config.py "$D/config" "$CFG"
exec .venv/bin/python serve/server.py --engine strata --config "$CFG" --host 0.0.0.0 --port "${PORT:-8080}"
