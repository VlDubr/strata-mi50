# v0.1.43.1 build check on T5810 (2026-10-12 20:03-20:10 UTC)

Machine: Dell Precision T5810, 2x AMD Instinct MI50 32 GB (gfx906), Ubuntu 26.04.1 LTS,
kernel 7.0.0-31-generic, ROCm 10.0.0-gfx906. Source: `gfx906` at `9c26f21f` (upstream v0.1.43.1
`a0116b27` merged as `9c26f21f`, fork conversation-copy fix `e450d67a` included). Pinned
llama.cpp/ggml `3cf03257f219afbe7334045ff7c6a06ac68c627d`, re-checked in place:
`third_party/ggml/VERSION.txt` and the shared source directory
`build-text-rocm10-cachefix/_deps/strata_llamacpp-src` both report that commit, and configure
printed `ggml commit: 3cf03257f`.

What this round did: merged upstream v0.1.43.1 into `gfx906` (merge commit `9c26f21f`, pushed to
`origin/gfx906`), fast-forwarded the T5810 checkout `b5c65877` -> `9c26f21f`, built the engine in a
**new** directory `build-text-rocm10-v431`, ran the tests that do not need a GPU, and backfilled the
missing build-identity record of the engine that has run 8082 since 2026-10-11 19:39 UTC. The 8082
service (server 409978, vision 409984, text engine 410015, started 2026-10-11 19:39 UTC by the
`deploy-v43.sh --stop-current` run) kept running the whole round and was not stopped, restarted or
reconfigured. **Nothing was deployed and no binary from this round was executed.**

Preflight at 19:55-20:02 UTC: 8082 listening, `/health` `ok` with `loaded: true`, both GPUs 0 % use
with the model resident (GPU0 32,471,547,904 B, GPU1 33,596,694,528 B VRAM used), load average 1.36,
32 GiB RAM available, `/` 22 GiB free, `/data` 107 GiB free. No Docker workload.

## Files

- `build-identity.json` - source commits, configure switches, artifact hash and the test summary for
  the staged v0.1.43.1 binary (also written on the machine at
  `logs/v431-build-20261012T200319Z/build-identity.json`).
- `v43-running-build-identity.json` - the backfilled identity record for the **running** v0.1.43
  engine `dd7eaf97…`; on the machine at `logs/v43-identity-20261012T200921Z/build-identity.json`.
- `host-tests.txt` - the verbatim `ctest` run of the 21 host-side tests.
- `artifact-hashes.txt` - SHA256 of the staged v0.1.43.1 binary, the deployed v0.1.43 binary and the
  unchanged vision encoder.

The 1.9 MB compiler log is not committed; it stays on the machine at
`/home/chris/dev/strata-gfx906/logs/v431-build-20261012T200319Z/build.log` (with `configure.log`,
`status`, `launcher.log`).

## Key numbers

- Configure exit 0 at 20:03:23 UTC: `Strata: HIP (gfx906 compat) enabled, arch gfx906`,
  `CMAKE_PROJECT_VERSION 0.1.43.1`, `ggml version 0.24.0`, `ggml commit: 3cf03257f`.
- `cmake --build build-text-rocm10-v431 -j 6` exit 0 at 20:06:25 UTC (3 m 2 s): 300 targets, last
  `[299/300] Linking HIP executable strata`, **0 errors**, **2129 warnings** — per class 1250
  `-Wunused-result`, 492 `-Wsign-compare`, 262 `-Wunused-value`, 52 `-Wunused-parameter`, 22
  `-Wunused-const-variable`, 20 `-Wunused-function`, 10 `-Wunused-private-field`, 8
  `-Wunused-variable`, 3 `-Wmisleading-indentation`, 2 each `-Wunused-but-set-variable`,
  `-Winconsistent-missing-override` and `-Wcuda-compat`, 1 each `-Wtype-limits`, `-Wparentheses` and
  `-Wmissing-field-initializers` — up from 2103 at v0.1.43 with the same counting method (+26, all of
  it `-Wunused-result` on `hipError_t` returns). `STRATA_WERROR` stays OFF.
- Staged artifact, **never executed**: `build-text-rocm10-v431/strata`, 22,743,160 B, SHA256
  `11b847dcbce3fbf88d9a3d8673e6fa28545e0ee3448cb7d8b653083f79841bd0`. `strings` finds `0.1.43.1`
  twice; `ldd` resolves 20 libraries under `/opt/rocm-10.0` and none from `/opt/rocm-7.2.4`.
- Host-side tests (`CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=`, ctest exit 0): **21 of 21 ran and
  passed** — `conversation_cache_test`, `conversation_split_failure_test`, `conversation_memory_test`,
  `conversation_file_test`, `session_save_reclaim_test`, `file_expert_source_test`,
  `expert_profile_save_test`, `expert_profile_version_test`, `conversation_validation_test`,
  `pinned_shared_test`, `ple_reader_selftest`, `message_boundary_test`, `split_ring_test`,
  `conv_cache_test`, `draft_kv_plan_test`, `coupled_draft_test`, `spec_prob_test`, `bf16_bits_test`,
  `batch_rows_test`, `batch_groups_test`. `conversation_copy_test` reported `***Skipped` (its
  recorded exit-77 code with no device) and is **not** a pass.
- `batch_groups_test` passing matters here: upstream `a0116b27` (#2038) changed its expectations, and
  the merged fork tree satisfies them.
- The vision encoder was not rebuilt: the merge touches no file under `tools/vision/` and none under
  `third_party/`, and `tools/vision/CMakeLists.txt` compiles no `src/` source, so the deployed
  `709fc4d2…` encoder stays valid.

## Deployment record fixed in this round

`/home/chris/mi50-t5810/scripts/strata-current.sh` exited **1** before this round: the running text
engine `dd7eaf97…` had no `logs/*/build-identity.json` record, because the 2026-10-11 18:55-19:04 UTC
build round wrote its logs but not that file (`NORECORD text: 运行哈希 dd7eaf975c612a50 无构建身份记录`).
The backfilled record (`v43-running-build-identity.json`) closes it: the script now exits **0** with
`结论: 符合现役规则(上游文本 + fork HIP 视觉,均 ROCm 10)`. No binary, config, launcher or service
state was changed by that fix.

## Postflight (20:10 UTC)

`/health` `ok` with `loaded: true` and 262,144 context; the same three PIDs (409978 / 409984 /
410015) up 1 day; both GPUs 0 % use with VRAM unchanged (32,471,547,904 / 33,596,694,528 B);
`build-text-rocm10/strata` still hashes to `dd7eaf97…`; `journalctl -k -b 0 --since 20:00` shows 0
page-fault lines; `/` 21 GiB free, 31 GiB RAM available. No driver, ROCm, network, thermal/tuning or
Docker change.

## Limits of this evidence

- **No runtime acceptance of the v0.1.43.1 binary**: it was never executed. Both cards were held by
  the live service, so no engine start, model request, image request or speed measurement was
  possible. Build success and host-side unit tests are not model parity, speed or stability.
- **No GPU test was run at all.** `conversation_copy_test` (165 checks, single device) and the
  `*_parity` group, `conversation_snapshot_test` and `vmm_test` are deferred to a window with idle
  GPUs; they were not run here because the deployed engine's VRAM is nearly full and a failed
  allocation could disturb the running service.
- The three behaviour changes this merge brings to the gfx906 path — the pinned prompt-upload ring
  (#1996), the PLE `madvise(MADV_WILLNEED)` window (#1984 #1971) and the copy progress beats
  (#2049 #2065) — have **no MI50 measurement**.
- Deploying `11b847dc…` needs an owner-approved restart of 8082; the fork's conversation-copy fix is
  already in this tree.
