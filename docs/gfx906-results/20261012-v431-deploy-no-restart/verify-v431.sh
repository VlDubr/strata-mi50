#!/usr/bin/env bash
# t5810 - read-only check that the 8082 engine really runs the v0.1.43.1 artifact (11b847dc...).
# Run this after the owner restarts 8082. It starts nothing and stops nothing.
set -uo pipefail
ROOT=/home/chris/dev/strata-gfx906
EXPECT=11b847dcbce3fbf88d9a3d8673e6fa28545e0ee3448cb7d8b653083f79841bd0
say() { printf "%s\n" "$*"; }
cd "$ROOT" || exit 1
say "deployed file : $(sha256sum build-text-rocm10/strata | cut -c1-16)... (expect 11b847dc...)"
say "health        : $(curl -s -m 10 http://127.0.0.1:8082/health | head -c 200)"
say "engine version: $(curl -s -m 10 http://127.0.0.1:8082/metrics | python3 -c "import sys,json;print(json.load(sys.stdin).get(\"engine\",{}).get(\"version\"))" 2>/dev/null) (expect 0.1.43.1)"
for p in $(pgrep -f "strata --serve" || true); do
  say "engine PID $p exe: $(readlink /proc/$p/exe 2>/dev/null)"
  say "             runtime: $(grep -o "/opt/rocm[^ ]*libamdhip64[^ ]*" /proc/$p/maps 2>/dev/null | sort -u | head -1)"
done
say "one real greedy generation:"
curl -s -m 120 http://127.0.0.1:8082/v1/chat/completions -H "Content-Type: application/json" \
  -d "{\"model\":\"strata\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: STRATA OK\"}],\"temperature\":0,\"max_tokens\":8,\"chat_template_kwargs\":{\"enable_thinking\":false}}" | head -c 400; echo
say "strata-current.sh:"; bash /home/chris/mi50-t5810/scripts/strata-current.sh | tail -14
