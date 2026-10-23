# Upstream v0.2.3.1 merge on 4x MI50: raw evidence (2026-10-23)

The summary is in `docs/GFX906.md`, section "Upstream v0.2.3.1 synchronization (2026-10-23)".

| File | What it is |
|---|---|
| `ab-results.jsonl` | One record per variant of rounds `s1` and `s2` (9 records), written by the decode harness of the first round (`docs/gfx906-results/20261020-mi50x4-decode/ab.py`, extended with an `"image"` key that runs the variant on another Docker image). Every request's `decode_tok_s`, output tokens, prompt tokens and time, drafts accepted / offered, and the SHA-1 prefix of the output text; `same_as_first` compares the output hashes with the round's first variant. |
| `longab.log` | The 115,982-token prompt check (`docs/gfx906-results/20261021-mi50x4-cycle2/longab.py` with `longctx.py 380000`): prompt time, decode rate and drafts, one line per run. |
| `variants.txt` | The variant files of both rounds and of the long-prompt runs. |
| `ab.py` | The harness as run: the first round's `ab.py` plus the `"image"` key (`start(..., image)` sets `IMAGE` for `run.sh`). |

Images: `strata-gfx906:0.2.0-perf2` is `c3d0ca37` (the decode round of 2026-10-21); `strata-gfx906:0.2.3-sync` is the merge
with its follow-up commit (All-Off HIP mask, the early PLE flag on the gfx906 build). Both run the same tuned config
(`--spec 2 --spec-min-p 0.8 --mtp-q4 head --mtp-draft-vocab draft_vocab_cyrillic.bin`, `--pipeline-windows 2`, 4-GPU layer split,
131072 context, int8 KV).

Round `s1` ran with the cards' DPM level at `auto` (a boot-time race had left the service that pins it to `high` failed),
so its rates are about 13% below round `s2`, which ran at `high`; both rounds compare the two images under the same
conditions and give the same output hashes. `s1_perf2`'s long-prompt rate (610 tok/s) is the halved rate another GPU
program on this machine causes now and then.
