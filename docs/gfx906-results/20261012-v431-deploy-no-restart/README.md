# v0.1.43.1 installed on T5810 without a restart (2026-10-12 20:37-20:40 UTC)

Machine: Dell Precision T5810, 2x AMD Instinct MI50 32 GB (gfx906), Ubuntu 26.04.1 LTS,
kernel 7.0.0-31-generic, ROCm 10.0.0-gfx906. Artifact installed:
`build-text-rocm10-v431/strata` (upstream v0.1.43.1 `a0116b27`, fork merge `9c26f21f`, fork
conversation-copy fix `e450d67a`), SHA256 `11b847dcbce3fbf88d9a3d8673e6fa28545e0ee3448cb7d8b653083f79841bd0`.
Build evidence: [`../20261012-v431-build-check/`](../20261012-v431-build-check/).

**Owner instruction: install, do not restart.** The restart is the owner's own action. Nothing in this
round stopped, restarted or reconfigured 8082: server 409978, vision 409984 and text engine 410015 kept
running the whole round on the old binary.

## What was done

1. Preflight 20:37 UTC: `/health` `ok loaded:true`, engine `0.1.43` (PID 410015, up 1 d), conversation
   cache `parked 2`, `bytes 6,766,865,728`, `requests 500`, `reused_tokens 53,133,597` of
   `prompt_tokens 54,388,451`; both GPUs 0 % use, VRAM 32,471,547,904 / 33,596,694,528 B; `/` 21 GiB
   free; 32 GiB RAM available. Staged binary re-verified by hash, `0.1.43.1` string (2 hits) and
   linkage (20 ROCm 10.0 libraries, 0 from 7.2.4).
2. Backup of the deployed binary: `build-text-rocm10/strata.v0143-dd7eaf97.bak`
   (`dd7eaf97…`, `cp -a`, hash re-checked).
3. Install with `install -m 0755` (unlink + create, so no `ETXTBSY` against the running process):
   `build-text-rocm10/strata` is now `11b847dc…`. `run-iq3-s.json` was **not** touched; it still points
   at that path.
4. Post-install verification (20:37-20:39 UTC): the deployed file hashes to `11b847dc…`; the running
   engine is still PID 410015 with `/proc/410015/exe` = `…/build-text-rocm10/strata (deleted)`, i.e. the
   old `dd7eaf97…` inode; `/health` `ok loaded:true`; `/metrics` engine `0.1.43`; both GPUs 0 % use;
   one real greedy request answered `STRATA OK` with `finish_reason: stop` (`cached_tokens 0`).
5. `strata-current.sh` expectations moved to the new artifact (`EXPECT_TEXT=11b847dc…`, `dd7eaf97…`
   added to the superseded list, with a comment that the running process is still the old one until the
   owner restarts). The old script is kept as `strata-current.sh.before` in the machine log directory.
6. `rollback-v431.sh` and `verify-v431.sh` staged in `/home/chris/mi50-t5810/scripts/` (copies in this
   directory). `rollback-v431.sh --dry-run` was run and changed nothing; `verify-v431.sh` is read-only.

## State after this round

| Item | Value |
|---|---|
| Deployed file `build-text-rocm10/strata` | `11b847dcbce3fbf8…` (v0.1.43.1) — **not yet executed** |
| Running engine PID 410015 | still `dd7eaf97…` (v0.1.43), `/proc/…/exe` shows `(deleted)` |
| Backup of the previous artifact | `build-text-rocm10/strata.v0143-dd7eaf97.bak` |
| Vision encoder | unchanged `709fc4d2…` |
| `strata-current.sh` | exits **1** with `违反规则: 进程 410015 跑的不是配置所指文件: …/strata (deleted)` — correct for this transitional state; it exits 0 once the owner restarts |
| `run-iq3-s.json`, launcher, drivers, ROCm, network, thermal/tuning | unchanged |

## Owner restart and what to check afterwards

Restart is the owner's action (`./run-iq3-s.sh` in `/home/chris/dev/strata-gfx906`, or the usual
launcher). Then run `bash /home/chris/mi50-t5810/scripts/verify-v431.sh`, which reads only: the deployed
hash, `/health`, `/metrics` engine version (must read `0.1.43.1`), the engine PID's `exe` and mapped
`libamdhip64`, one real greedy `STRATA OK` generation, and `strata-current.sh` (expected exit 0).

The restart drops the RAM-only conversation cache (at preflight: 2 parked conversations,
6,766,865,728 bytes), so clients re-prefill their conversations.

## Known risk of an install-without-restart, stated plainly

While the old engine runs, the file at its path is already the new binary. If the engine dies and the
server fatal-recovers it before the owner restarts, the **untested v0.1.43.1 binary is what gets
executed**, with no acceptance run on it. This is the reason `deploy-v43.sh` refuses to install while
8082 runs; the owner asked for the install anyway. Mitigation kept on the machine:
`rollback-v431.sh` restores `dd7eaf97…` (with `--stop-current` it also restarts), and the previous
artifact stays as `strata.v0143-dd7eaf97.bak`.

## Files

- `run.log` - the install transcript, verbatim.
- `hashes.txt` - SHA256 of the deployed file, the backup and the staged artifact.
- `generation-after-install.json` - the real request the still-running engine answered after the swap.
- `strata-current.after-install.out` - the current-deployment check in the transitional state (exit 1).
- `rollback-v431.sh`, `verify-v431.sh` - copies of the staged machine scripts.

Machine log directory: `/home/chris/dev/strata-gfx906/logs/deploy-v431-no-restart-20261012T203749Z/`.

## Limits of this evidence

- The v0.1.43.1 binary has **still never been executed**: no engine start, no model request against it,
  no GPU test, no speed number.
- The `STRATA OK` generation proves only that the **old** engine survived the file swap.
- No workload rate was measured in this round; the newest workload numbers in this repository remain
  the v0.1.42 observations.
