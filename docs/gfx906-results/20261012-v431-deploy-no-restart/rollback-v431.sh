#!/usr/bin/env bash
# t5810 - put the v0.1.43 conversation-copy-fix engine (dd7eaf97...) back as the 8082 text engine.
#
#   rollback-v431.sh --dry-run
#   rollback-v431.sh --stop-current
#
# Restores build-text-rocm10/strata.v0143-dd7eaf97.bak (v0.1.43 / merge 24eb7211 + e450d67a) over
# build-text-rocm10/strata (currently v0.1.43.1 / merge 9c26f21f -> 11b847dc...). Vision untouched.
# WARNING: with --stop-current it restarts 8082; without it, it only swaps the file and leaves the
# running process alone (the same no-restart install the v0.1.43.1 deploy used on 2026-10-12).
set -uo pipefail
ROOT=/home/chris/dev/strata-gfx906
CUR=$ROOT/build-text-rocm10/strata
BAK=$ROOT/build-text-rocm10/strata.v0143-dd7eaf97.bak
EXPECT_ROLLBACK=dd7eaf975c612a505c4ab2c734bd87273c548c7ee2c016ebec94eb47bd70e442
STOP_CURRENT=0; DRY=0
for a in "$@"; do case "$a" in
  --stop-current) STOP_CURRENT=1 ;; --dry-run) DRY=1 ;;
  *) echo "unknown argument: $a" >&2; exit 2 ;;
esac; done
TS=$(date -u +%Y%m%dT%H%M%SZ); LOGD=$ROOT/logs/rollback-v431-$TS; mkdir -p "$LOGD"
say() { printf "[%s] %s\n" "$(date -u +%H:%M:%S)" "$*" | tee -a "$LOGD/run.log"; }
fail() { say "FAIL: $*"; exit 1; }
say "log dir: $LOGD"
[ -f "$BAK" ] || fail "missing backup $BAK"
[ "$(sha256sum "$BAK" | cut -d" " -f1)" = "$EXPECT_ROLLBACK" ] || fail "backup hash != $EXPECT_ROLLBACK"
SRV=$(pgrep -f "serve/server.py --engine strata" | head -1 || true)
say "8082 server: ${SRV:-not running}"
if [ "$DRY" = 1 ]; then say "dry run: nothing changed"; exit 0; fi
if [ -n "$SRV" ] && [ "$STOP_CURRENT" != 1 ]; then
  say "8082 is running (PID $SRV): swapping the file only, NO stop and NO restart"
  say "the running engine keeps the old inode until it is restarted"
fi
install -m 0755 "$BAK" "$CUR"
say "restored: $CUR = $(sha256sum "$CUR" | cut -c1-16)..."
if [ -n "$SRV" ] && [ "$STOP_CURRENT" = 1 ]; then
  kill -TERM "$SRV"
  for i in $(seq 1 60); do pgrep -f "serve/server.py --engine strata" >/dev/null || break; sleep 1; done
  pgrep -f "serve/server.py --engine strata" >/dev/null && fail "server still running"
  cd "$ROOT" || fail "cd $ROOT"
  nohup ./run-iq3-s.sh > "$LOGD/launcher.out" 2>&1 &
  for i in $(seq 1 180); do curl -s --max-time 10 http://127.0.0.1:8082/health | grep -q "\"loaded\": true" && break; sleep 5; done
  say "health: $(curl -s --max-time 10 http://127.0.0.1:8082/health | head -c 200)"
else
  say "no restart performed"
fi
