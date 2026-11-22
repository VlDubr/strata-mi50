# 4x MI50 32 GB, Strata v0.2.19 (engine 0.1.65), 2026-11-23

The v0.2.19 merge check and the tuning A/B on the deployed server (pc-ii, 192.168.0.198): AMD EPYC 7302,
107 GB RAM, Ubuntu (kernel 7.0.0-44), ROCm 10.0 in the container (`mixa3607/rocm-gfx906:10.0-complete-20261018172427`),
GPU DPM level `high` (`amdgpu-perf-high.service`). Qwen3.8-Flash-Next, Unsloth UD-Q5_K_XL, int8 KV cache,
MTP speculative decoding (`--spec 2 --spec-min-p 0.8 --mtp-q4 head`, Cyrillic draft vocab), layer split 12 / 24 / 36,
`--kv-resident 32768 --trim-stage-weights --resident-budget-gib 24 --max-context 262144`, 8 batch slots, vision on.

Images: `strata-gfx906:0.2.19-mi50` and `strata-gfx906:0.2.19-mi50-vision`, built from the merge commit on the server.
The old deployment was `strata-gfx906:0.2.12-mi50-vision` (engine 0.1.58).

## What was measured

- `session_bench.py <tag> [reps]`: four fixed greedy prompts (chat / code / story 500 tokens, a 17.7K-token document
  200 tokens), each with a per-run nonce prefix so the conversation cache cannot reuse K/V (`cached_tokens` 0).
  Decode and prefill from the engine's own `/metrics` row recorded after the request was sent. `mean3_warm` is the
  mean of the second rep of chat/code/story.
- `ab2.py ab2-variants.json --reps 2`: the same workload per variant, container recreated from `tuned.json` with the
  variant's env/args/parallel. One JSON line per variant in `ab2-solo.jsonl`.
- `par_ab.py`: 10 concurrent clients (`par_bench.py` prompts, greedy, streaming, 300 tokens each), stock vs best
  config, two interleaved rounds. Aggregate = all output tokens / wall time. `par-ab.jsonl`.

The first three `session-benches.jsonl` rows (`baseline-v0212*`, `base0212`) ran on the OLD image while the server
was under live external load (other users' 40-100K-token requests); they are not a quiet baseline and are kept only
as the record of that load. The quiet measurements are `new0219-stock` and `final-best`, and the A/B rows.

## Solo decode (mean of chat/code/story, tok/s)

| Variant | mean3 | chat | code | story | long decode | long prefill (real) |
|---|---:|---:|---:|---:|---:|---:|
| v0.2.19 stock (8 slots, no PW/PIPED) | 62.85 | 64.6 | 61.4 | 62.55 | 59.8 | 1,215 |
| + `--batch-mtp` `STRATA_PIPED_MTP=1` | 63.38 | 63.95 | 63.1 | 63.1 | 61.2 | 1,174 |
| + `STRATA_PW_WITH_BATCH=1` | 83.48 | 87.4 | 83.3 | 79.75 | 83.0 | 618 |
| + `STRATA_PW_WITH_BATCH=1` + piped MTP | **84.28** | 87.0 | 85.7 | 80.15 | 84.7 | 1,187 |
| parallel 12 (stock) | 62.72 | 65.05 | 61.45 | 61.65 | 60.4 | 1,175 |
| parallel 12 + piped MTP | 62.98 | 64.85 | 61.75 | 62.35 | 58.4 | 1,175 |

`STRATA_PW_WITH_BATCH=1` keeps the solo path's `--pipeline-windows 2` overlap when batch slots are on: +33% solo
decode (62.85 → 83.48). Piped MTP on top adds ~1% solo and restores the long-prompt prefill rate (618 → 1,187 tok/s;
without piped MTP the second verifier halves the prompt read rate). The deployed config is the piped-MTP + PW row.

## 10 concurrent clients (aggregate tok/s, two interleaved rounds)

| Config | round 1 | round 2 |
|---|---:|---:|
| stock | 113.9 | 113.5 |
| best (PW + piped MTP + batch-mtp) | 117.1 | 120.5 |

+3–6% aggregate; per-request rates and TTFT ranges overlap the stock ones. The 10-client aggregate is below the
2026-11-08 README number (135 tok/s on the deployed 8-slot config) because these prompts are short (300 tokens)
and the nonce prefix defeats K/V reuse; the earlier round used `par.py` without nonces.

## Exactness

Greedy output hashes differ between reps of the same prompt even on the stock config (the nonce prefix changes the
prompt; batch scheduling also varies), so hash parity across variants is not asserted here. The v0.2.12 merge's
`eqcheck.py` 8/8 parity was measured without nonces; that check was not repeated in this round.

## Not measured

Sampled decoding, more than two rounds per point, long soak, MI60, other models, the all-cards split engine and
GLM-5.3 on v0.2.19 (their images were not rebuilt this round). GPU junction temperatures reached 103-104 C on
cards 1 and 3 under the earlier live load; no throttle flag was read, and no cooling or power change was made.

## Files

- `ab2-solo.jsonl` — the six solo variants, full per-request rows.
- `par-ab.jsonl` — the four 10-client rounds.
- `session-benches.jsonl` — the session runs, including the loaded old-image rows.
- `ab2.py`, `ab2-variants.json`, `par_ab.py`, `par_bench.py`, `session_bench.py` — the harnesses as run.