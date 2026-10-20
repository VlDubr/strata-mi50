# gfx906 decode, second round: raw evidence (4x MI50, 2026-10-20 / 21)

The summary is in `docs/GFX906.md`, section "Draft subset, window size and exact kernels (2026-10-21)".

| File | What it is |
|---|---|
| `ab-results.jsonl` | One JSON record per engine variant, rounds `b1` to `b9` (69 records), written by the harness of the first round (`docs/gfx906-results/20261020-mi50x4-decode/ab.py`): the variant (engine binary, environment, arguments), every request's `decode_tok_s`, output tokens, prompt tokens and time, drafts accepted / offered, and the SHA-1 prefix of the output text. `same_as_first` compares a variant's output hashes with its round's first variant. |
| `winms.py` | Tokens per verify window (output tokens minus accepted drafts) and milliseconds per window, from those records. |
| `longctx.py`, `longab.py` | The long-prompt check: about N characters of this repository's docs and sources, then 300 greedy tokens, per variant. |
| `route_test.cpp` | Parity and timing of `route_wave64` against `route` / `route_multi` (65,536 rows with exact ties, NaN, -inf, equal rows). |
| `kfloor.cpp`, `kfloor2.cpp`, `glaunch.cpp` | Launch cost on MI50: a small kernel in a HIP graph costs ~1.3-1.6 us (rocprofv3 reports ~3.5 us for the same kernel, so its per-kernel times of small kernels are inflated); a graph launch costs ~12 us of host time; memcpy / memset / event nodes in the graph do not change either. |

Engines in the records (all from `57fa6d85` plus the changes named; the binaries lived in `~/strata-engines/fN`):

| Name | Changes |
|---|---|
| (none) | the deployed `strata-gfx906:0.2.0-perf` image's engine (`57fa6d85`) |
| f1 | `route_wave64` (`STRATA_ROUTE_WAVE64=0` restores the old router) |
| f2 | f1 + HC up exact-T instantiations + the long mapped copy on gfx906 + a register HC down kernel (`STRATA_GR_DOWN_REG`, 0%, removed) |
| f3 | f2 + the unroll hints (`gdn_ab`, `argmax_rows`, `row_top_prob_split`, `mtp_select`, `norm`) |
| f4 | f3 without the register HC down kernel |
| f5 | f4 + `#pragma unroll 3` in `row_dot` (-5.6%, removed) |
| f6 | f4 + `STRATA_VERIFY_QDEDUP` and `STRATA_GR_NORM_FAST` in the gfx906 default table: the committed set |

Another program on this machine (an embedding server) holds 1.25 GB of VRAM and now and then works on the GPUs. Some
records show it: a halved prompt rate (`pf` below ~1000 tok/s) or one prompt far slower than its repeat (`b8_spec2_b`
story, `b9_after_b` long). Those records are not used for a conclusion; each conclusion rests on a repeat that agrees.
