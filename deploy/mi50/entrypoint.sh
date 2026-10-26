#!/bin/sh
# Strata container entry point (MI50 / gfx906).
# Configs live on the /data volume (data/config) and are linked into /opt/strata, so a recreated
# container skips setup. STRATA_SETUP=1 forces a setup pass with SETUP_ARGS.
set -e
cd /opt/strata
D="${STRATA_DATA:-/data}"
mkdir -p "$D/config" "$D/logs"
unset HSA_OVERRIDE_GFX_VERSION

link_configs() {
  for f in "$D"/config/strata-*.json; do
    [ -f "$f" ] && ln -sfn "$f" "/opt/strata/$(basename "$f")"
  done
  return 0
}
link_configs

if [ "${STRATA_SETUP:-0}" = "1" ] || ! ls "$D"/config/strata-*.json >/dev/null 2>&1; then
  echo "Setup: $SETUP_ARGS"
  # shellcheck disable=SC2086
  .venv/bin/python setup.py --setup --yes --no-start --backend hip --experimental-gfx906 \
    --data-dir "$D" $SETUP_ARGS
  for f in /opt/strata/strata-*.json; do
    [ -f "$f" ] || continue
    [ -L "$f" ] && continue
    cp -f "$f" "$D/config/" && ln -sfn "$D/config/$(basename "$f")" "$f"
  done
  [ "${SETUP_ONLY:-0}" = "1" ] && { echo "Setup done (SETUP_ONLY=1)."; exit 0; }
fi

# shellcheck disable=SC2086
exec .venv/bin/python setup.py --port "${PORT:-8080}" --no-browser $START_ARGS
