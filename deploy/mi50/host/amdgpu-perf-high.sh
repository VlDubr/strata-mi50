#!/bin/sh
# Pin the AMD GPU DPM performance level (Strata decode latency on MI50).
# At boot the amdgpu cards can appear after this unit starts (the old one-line unit failed with
# "Directory nonexistent" and left the cards at "auto", about 13% slower decode), so wait up to
# WAIT seconds for EXPECT cards before writing, and never fail on a card that is not there.
LEVEL="${1:-high}"
EXPECT="${EXPECT:-4}"
WAIT="${WAIT:-120}"
i=0
while [ "$i" -lt "$WAIT" ]; do
  n=$(ls /sys/class/drm/card*/device/power_dpm_force_performance_level 2>/dev/null | wc -l)
  [ "$n" -ge "$EXPECT" ] && break
  i=$((i + 1))
  sleep 1
done
rc=0
found=0
for f in /sys/class/drm/card*/device/power_dpm_force_performance_level; do
  [ -e "$f" ] || continue
  found=$((found + 1))
  echo "$LEVEL" > "$f" || rc=1
  echo "$f: $(cat "$f")"
done
[ "$found" -ge "$EXPECT" ] || echo "only $found of $EXPECT cards after ${i}s" >&2
[ "$found" -gt 0 ] || rc=1
exit $rc
