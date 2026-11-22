#!/bin/sh
# Strata container entry point (MI50 / gfx906).
# Configs live on the /data volume (data/config) and are linked into /opt/strata, so a recreated
# container skips setup. STRATA_SETUP=1 forces a setup pass with SETUP_ARGS.
set -e
cd /opt/strata
D="${STRATA_DATA:-/data}"
mkdir -p "$D/config" "$D/logs"
unset HSA_OVERRIDE_GFX_VERSION

# 0.2.19: setup.py and its config folder moved into terminal/; configs are linked into both
# /opt/strata/terminal (setup's ROOT) and /opt/strata (the legacy path).
link_configs() {
  for f in "$D"/config/strata-*.json; do
    [ -f "$f" ] || continue
    ln -sfn "$f" "/opt/strata/terminal/$(basename "$f")"
    ln -sfn "$f" "/opt/strata/$(basename "$f")"
  done
  return 0
}
link_configs

if [ "${STRATA_SETUP:-0}" = "1" ] || ! ls "$D"/config/strata-*.json >/dev/null 2>&1; then
  echo "Setup: $SETUP_ARGS"
  # shellcheck disable=SC2086
  .venv/bin/python terminal/setup.py --setup --yes --no-start --backend hip --experimental-gfx906 \
    --data-dir "$D" $SETUP_ARGS
  for d in /opt/strata/terminal /opt/strata; do
    for f in "$d"/strata-*.json; do
      [ -f "$f" ] || continue
      [ -L "$f" ] && continue
      cp -f "$f" "$D/config/" && ln -sfn "$D/config/$(basename "$f")" "$f"
    done
  done
  [ "${SETUP_ONLY:-0}" = "1" ] && { echo "Setup done (SETUP_ONLY=1)."; exit 0; }
fi

# shellcheck disable=SC2086
exec .venv/bin/python terminal/setup.py --port "${PORT:-8080}" --no-browser $START_ARGS
