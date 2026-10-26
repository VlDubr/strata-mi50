# Several requests at once on 4x MI50, 262144 context: raw evidence (2026-10-26)

The summary is in `docs/GFX906.md`, section "Several requests at once on 4x MI50, 256K context (2026-10-26)", and the
newest-result block of `README.md`.

Machine: AMD EPYC 7302, 92 GB RAM, 4x Instinct MI50 32 GB, Ubuntu 26.04 (kernel 7.0.0-38), ROCm 10.0 in the container
(`deploy/mi50/Dockerfile`), GPU DPM level `high`. Image `strata-gfx906:0.2.6-mi50` (the v0.2.6 merge `45005f55` with the
`parallel_args` fix of `03ad6638`), engine 0.1.51. Model Qwen3.8-Flash-Next, Unsloth UD-Q5_K_XL, int8 KV, MTP drafts
(`--spec 2 --spec-min-p 0.8 --mtp-q4 head --mtp-draft-vocab draft_vocab_cyrillic.bin`).

| File | What it is |
|---|---|
| `p1.json` ... `p4.json` | The variant files: top-level config keys (`parallel`, `layer_split`), engine `args` and `env` changed against the deployed config. |
| `p1.log` ... `p4.log` | `pexp.py` output for each variant file: startup time, the engine's slot / memory / expert lines, one `par.py` line per concurrency (1, 4, 10) and, for `p3` / `p4`, one `longctx.py 380000` line. |
| `pexp.py` | The variant runner: writes the variant config, restarts the server (`run.sh`), runs `par.py` and `longctx.py`, restores the deployed config at the end. Imports `ab.py` of `../20261023-mi50x4-sync/`. |
| `par.py` | The load test: N different Russian prompts at once, greedy, streaming, 300 tokens each. |
| `longctx.py` | One long prompt (N characters of the repository's docs and sources), 300 greedy tokens; prints the engine's own `/metrics` record. |
| `eqcheck.py` | N greedy prompts one after another, then the same N at once; compares the SHA-1 of the output texts. |
| `det.py` | The four fixed prompts of the decode rounds; prints their output hashes. |
| `mkpar.py` | Writes the chosen config (`parallel` 12) from the previous one and keeps that as `tuned-solo.json`. |
| `verify_par.sh`, `verify_par.log` | The checks on the deployed config: `eqcheck.py 8 200`, `det.py`, `longctx.py 820000`, `free -g`. |

## How the numbers are computed

- **Aggregate rate** (`aggregate_tok_s`): all output tokens of the round (the server's `usage.completion_tokens`) divided by
  the wall time from the first send to the last streamed token.
- **Per-request rate** (`per_req_dec_min_med_max`): a request's tokens after its first, divided by the time from its first to
  its last streamed token.
- **First token** (`ttft_min_med_max`): time from the send to the first streamed token, queueing included.
- **Prompt rate** (`longctx.py`): `prompt_tokens / prompt_ms` from the engine's `/metrics` record of that request. The server
  restarts before each variant and no earlier request in the variant shares the long prompt, so no K/V is reused: this is the
  real rate, and the effective rate equals it. The record does not carry the `reused` counter itself.

## Which runs are valid

| Run | Valid | Note |
|---|---|---|
| `p1` `P0_solo`, `P1_par10_128k` | yes | |
| `p1` `P2_*` | no | `--kv-streaming on` is a setup option, not an engine flag: the engine stopped with its usage text (exit 2), "server did not come up". |
| `p1` `P3_*`, `p2` `P2a_*`, `P2b_*` | no | Two runners overlapped and rewrote each other's config, so the server did not run the named variant (`P2b` shows the solo config's rates). `P3a` and `P3c` of `p3` repeat them. |
| `p1` `P4_*`, `p3` `P3b_*`, `P3d_*` | no | `run.sh` exit 125: `docker run` started while `docker rm -f` was still removing the old container. `run.sh` now waits for the removal; `p4` repeats `P3b` and `P3d`. |
| `p2` `P2c_*`, `P2d_*` | yes | |
| `p3` `P3a_*`, `P3c_*` | yes | `P3a` is the deployed config. |
| `p4` `P3b_*`, `P3d_*` | yes | |

Not measured: sampled decoding, more than one round per point, prompts arriving while others decode at long context,
a long soak, MI60.
