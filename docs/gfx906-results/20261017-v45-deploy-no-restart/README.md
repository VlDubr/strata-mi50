# v0.1.45 installed on 8082's text path without a restart (2026-10-17)

Raw evidence for the [v0.1.45 deployment](../../GFX906.md#v0145-installed-on-8082s-text-path-without-a-restart-2026-10-17-2315-utc).

Owner instruction: **deploy, I restart myself**. Nothing was stopped, restarted or reconfigured.

- `preflight.txt` — the three PIDs and their exe links, `/health`, `/metrics` engine `0.1.44.1`, both
  GPUs 0 % use, `/` 20 GiB free, `run-iq3-s.json` md5, and the two deployed hashes before the swap.
  The `tailscaled be-child ssh` lines are this round's own SSH command line matched by `pgrep -f`; they
  are not service processes.
- `deploy.txt` — the backup (`strata.v01441-050b83e6.bak`, hash re-checked against the deployed file
  before the swap), the staged hash, `install -m 0755`, the deployed hash, `/health` and `/metrics`
  after the swap, and the three PIDs' exe links showing the engine's `(deleted)` inode.
- `postflight.txt` — the same three PIDs with their elapsed times, `/health` `ok loaded:true`,
  `/metrics` still engine `0.1.44.1`, one real greedy generation answered `STRATA OK`
  (`finish_reason: stop`) through the still-running old engine, both GPUs 0 % use, `/` 20 GiB free,
  `run-iq3-s.json` md5 unchanged, and the deployed/backup hashes.
- `artifact-hashes.txt` — staged, deployed and backup hashes, the running processes' exe links, and the
  `ldd` counts for the staged engine (20 libraries under `/opt/rocm-10.0`, none from `/opt/rocm-7.2.4`).

The vision encoder was **not** replaced: upstream changed nothing under `tools/vision` in
`296b59d2..477ac13c` and the llama.cpp pin is unchanged, so the deployed `f8a183ff…` is the current
artifact for v0.1.45 as it was for v0.1.44.1. The v0.1.45 tree's vision rebuild
(`build-vision-rocm10-v45/bin/strata-vision`, `d11f35e7…`) is a compile check only; its hash differs
from the deployed one because the build directory path is embedded in the binary.

Machine side: `~/dev/strata-gfx906/logs/deploy-v45-no-restart-20261017T231458Z/`.
Machine scripts written this round: `~/mi50-t5810/scripts/verify-v45.sh` (read-only post-restart check)
and `~/mi50-t5810/scripts/rollback-v45.sh` (`--dry-run` verified); `strata-current.sh` updated with its
previous copy saved as `strata-current.sh.bak-20261017T2315Z`.
