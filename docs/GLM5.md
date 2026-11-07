# GLM-5.3-Flash on MI50

GLM-5.3-Flash (GGUF architecture `glm5-next`, 320B total parameters, 288 experts) runs on the DeepSeek engine path
(`strata-dsv41`) on 4x MI50 32 GB. This page records the design, the deployment and the measurements.

## Status

Validated on 4x MI50 32 GB, 91 GB RAM, ROCm 10, with `unsloth/GLM-5.3-Flash` UD-Q4_K_XL (199.7 GB).

The deployed configuration: context 131072, four conversations, the multi-token head with two drafts, the vision tower, the experts
striped over the RAM's regions. The speeds are API rates over six chat answers of 256 tokens (`glm53/ref/bench.py`).

| Measure | llama.cpp b11786 | Strata, deployed | Strata, without head, tower and seats |
|---|---|---|---|
| Decode, one conversation, sampled at the model card's settings | 10.8 tok/s | 36 tok/s | 20.5-21.5 tok/s |
| Decode, two answers at once | not measured | 31 tok/s in all | - |
| Decode, four answers at once | not measured | 37 tok/s in all | - |
| Prefill, 14K tokens of C++ | not measured | 77 tok/s | not measured |
| Prefill, 2647 tokens of documentation | 74 tok/s | not measured | 145 tok/s |
| Prefill, 16.7K and 27.9K tokens of C++ | not measured | not measured | 99 and 108 tok/s |
| Load | about 14 min | about 2 min | about 2 min |

The four conversations, the head and the tower take VRAM from the experts: 4950 experts on the CPU lane instead of about 4100.
That is what slows prefill. With two conversations the 14K prompt reads at 84 tok/s, and chat decode gains 2%.
The 145 tok/s was measured with a profile of documents and code alone. A fact at the start of a 27.9K-token prompt was found
correctly. A prompt that ends differently goes back to a restore point: a 14K-token prompt took 10 s instead of 179 s.

Numerics against llama.cpp on the same weights, greedy:

- The short prompts give the same tokens until a near-tie of the top two tokens.
- With no expert profile, the 2647-token prompt in windows of 16 gave the same 96 of 96 tokens.
- llama.cpp itself differs between its CPU and GPU backends on the same prompts, at step 32 of the long prompt.
- A greedy comparison on long prompts must therefore use the top-k logprobs, not the exact text.

The server's token ids and rendered chat template match llama.cpp on all reference prompts.

## What is not implemented

- **Drafts beside other answers:** the multi-token head drafts only while one conversation is answering. With two or more,
  each takes a plain step in a shared window.
- **Conversation slots:** the model holds four conversations. A fifth takes the seat used longest ago, and the conversation that
  leaves is read again from its first token. Restore points inside a conversation work, and they include the KDA state.
- **Video and audio:** the template has their markers, the engine does not read them.
- **Tensor parallel, the DeepSeek chunk path:** GLM uses its own path, below.

## Design

The engine reads the geometry from the pack, so GLM is a family inside `src/dsv41/ds4_model.cu`, with `g.glm5` as the switch.
It reuses the mHC kernels, top-k, GGUF GEMV, the expert tiers, the CPU lane, the serve protocol and sampling.

- `tools/dsv41/pack_glm5.py` writes the pack: `ds4.txt` with `family glm5-next`, `tensors.txt`, `tokenizer/`.
- `src/dsv41/glm5_kernels.cu` holds KDA (conv, scan, output norm), the absorbed MLA, the k-pool indexer and the sigmoid router.
- `Impl::glm5_attn` is the attention sub-block. `glm5_dense_ffn` is the FFN of layers 0-2. The MoE of a decode step is the
  device-routed path of `layer()`, as DeepSeek's: the host queues a whole step, the router publishes its picks to the CPU lane's
  service thread, and the device waits for the answer only before the experts' sum. `DS4_GLM_SYNC=1` goes back to the host-routed path.
- `Impl::glm5_chunks` reads prompts: a thread a GPU, chunks of 256 rows, attention in windows of 16, experts once a chunk.
- `serve/glm5.py` translates the model's tool calls to the Qwen form that `OutputParser` reads.
- `src/dsv41/glm5_vision.cu` is the picture tower. `serve/ds_pictures.py` (`Glm5Pictures`) sizes pictures as llama.cpp does for
  `glm5v` and sends them through the `GENI` line, as DeepSeek's.

Multi-token head (`--dspark on`, block 45 as `L[g.nl]`, on the head's GPU):

- Row p of its caches is the trunk's final hidden row of p - 1 with token p. A prompt's rows write only those caches, because
  nothing later reads their attention or experts. The token just picked goes through the whole block and the shared head.
- A verify window keeps the KDA state and the conv state after each row but the last (`k_bak`, `k_cbak`). A rejected draft
  copies them back. The MLA caches are by position, so the next window overwrites a rejected row, and a pool it completed.
- A draft is verified when the head's probability is at least `--dspark-conf` (0.5). `--dspark-block N` chains up to 3 drafts:
  33.6 tok/s with one, 35.8 with two, 36.1 with three. The deployed setting is two.
- `--dspark off` keeps the block loaded and unused. Use it with the same `--dspark-block` to compare tokens: the greedy tokens were the same
  160 of 160 with two drafts on. `--dspark none` or another `--dspark-block` changes the VRAM left, which moves experts between the GPU and the
  CPU lane, and their rounding differs, so their tokens are not comparable: a near-tie of the top two tokens can fall the other way.
- 84-93% of the drafts were kept on code, Russian and English prose at temperature 1.0.

Parallel conversations (`--parallel N`): each conversation has its own KDA state, MLA caches and hidden rows, about 2.7 GB at
131072 tokens. A multi step runs dense weights and experts once for its rows and the per-conversation kernels a run at a time.
Three conversations together gave the same logits bit for bit as each alone. A long prompt reads 1024 rows at a time, then every
answer writes 64 tokens, so an answer stops for about 20 s at most (`STRATA_DS4_READ_PIECES=0` reads it whole, `STRATA_DS4_READ_ROUNDS`
sets the tokens). A prompt with pictures is read whole.

Thinking off (`reasoning_effort: none`, `enable_thinking: false`, Anthropic `thinking: disabled`): the template has no switch and
always opens `<think>`, so the server appends `</think>`, as the template writes the past turns, with "Reasoning Effort: Low".

Pictures: the tower's weights stay in pinned host memory as F16 and go to the GPU for each picture, so it costs the card 674 MiB of
work buffers. Its rows replace the `<|image|>` tokens; the model's attention has no positions, so nothing else changes. At most
4096 tokens a picture (the model takes 8000; the buffers are sized for this). 48 tokens take 90 ms. Against a numpy reference after
llama.cpp's graph, blocks 0-1 agree to five digits. The final rows have a mean cosine of 0.997 and a worst row of 0.96: the error
grows in a few rows with very large activations, more than F16 rounding alone gives in numpy (0.9999 and 0.998). The model runs
its tower in BF16, which is coarser. Use `mmproj-BF16.gguf`: the file is the checkpoint's format, and the pack prefers it.

Differences from DeepSeek-V4.1 that are easy to miss:

- Each sub-block uses its own `pre`. DeepSeek staggers it by one sub-block.
- The head input is a plain mean of the streams: `d.pre` is 1/4 for `hc_pre_norm`.
- The embedding stays Q8_0 on the first GPU and is read in F32, not rounded to BF16.
- Layers 0-2 have no experts: `n_exp` is 0, and `lay.fmt` still says MXFP4 because code reads `L[0].lay.fmt`.
- The caches are by position and in F16, as in llama.cpp: latent and pooled key. The indexer's key | gate rows are read once, for the
  pool they complete, so they live in a ring of 1024 rows; a restore point keeps the rows of its open pool. That gives the experts 3 GB.

## Build

```sh
cmake -S . -B build-glm5 -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
  -DSTRATA_EXPERIMENTAL_GFX906=ON -DCMAKE_HIP_ARCHITECTURES=gfx906 -DSTRATA_BUILD_TESTS=OFF -DSTRATA_BUILD_DSV41=ON \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE=ON -DSTRATA_GGML_DIR=/opt/strata/third_party/llama.cpp
ninja -C build-glm5 strata-dsv41 glm5_kernels_test ds4_gguf_test ds4_iq_test ds4_tq_test ds4_route_guard_test
```

The main `strata` engine keeps its `STRATA_HIP_GFX906` build. `strata-dsv41` uses the wave64 `STRATA_ENABLE_HIP` backend.
`Dockerfile.glm5` builds both from the same tree.
Image: `docker build -t strata-gfx906:0.2.10-mi50-glm5 -f deploy/mi50/Dockerfile.glm5 .`

## Deployment

1. Pack, with the paths the container sees. It takes the `mmproj*.gguf` in the model's folder, BF16 first:
   `python tools/dsv41/pack_glm5.py --model /data/models/GLM-5.3-Flash-UD-Q4_K_XL/UD-Q4_K_XL --out /data/packs/glm5-flash`
2. Put the expert profile at `/data/packs/glm5-flash/expert-profile.bin`. Without one the experts are placed in index order,
   and the CPU lane takes 33% of the picks instead of about 21%. The deployed profile is two runs, each layer's counts as
   shares, added with equal weight: 40 prompts of documents and code, and about 30 chat answers in several languages.
   Prompts alone are not enough: their tokens outnumber the answers', and chat then stays at 33%.
3. Write `/data/config/glm5.json`. Do not name it `strata-*.json`, or the Qwen image's setup takes it:

```json
{"exe": "/opt/strata/engine/strata-dsv41",
 "args": ["--pack", "/data/packs/glm5-flash", "--max-context", "131072", "--eos", "154820,154827,154829", "--dspark", "on",
          "--dspark-block", "2", "--parallel", "4", "--ram-stripe", "33.5,65.5", "--tp", "1", "--threads", "off", "--prefill-chunk", "0",
          "--expert-profile", "/data/packs/glm5-flash/expert-profile.bin", "--profile-readonly", "--no-expert-cache", "--ssd-cache", "0"],
 "cwd": "/opt/strata", "tokenizer": "/data/packs/glm5-flash/tokenizer", "model_name": "glm-5.3-flash",
 "arch": "deepseek41", "family": "glm5-next", "log": "/data/logs/strata-glm5.log", "lib_dirs": ["/opt/rocm/lib"], "port": 8080,
 "engine_ready_s": 3600, "backend": "hip", "experimental_gfx906": true, "gpu": [0, 1, 2, 3], "gpus_asked": true, "api_key": "..."}
```

`"arch": "deepseek41"` selects the engine plumbing. `"family": "glm5-next"` selects the template, stop ids, tool calls, effort
words and sampling.

4. Start with `deploy/mi50/run-glm5.sh`. It takes every GPU and refuses to start beside the Qwen container.

The RAM's regions: an EPYC 7002 has no six-way interleave, so six DIMMs make three regions of two channels. Each region reads at about
41 GB/s, whatever the number of threads. Read in turn, the three give 95-100 GB/s. The page cache lands where it lands: 50-65 GB/s.
`--ram-stripe 33.5,65.5` (where the regions end, in GiB) takes huge pages, sorts them by their frames and moves them, a region at a time, to
consecutive addresses; the CPU lane's experts live there, locked, and its jobs run a third of a matrix apart. It needs `--cap-add SYS_ADMIN`
for the frames. Without the capability the memory comes in the kernel's order, and the log says so. If the RAM is not free the experts stay in
the page cache, and the log says that too. To find the bounds of another machine: `glm53/ref/physbw.cpp` with `PROBE=<GiB>` reads each
GiB band beside a reference band, and the sum tells whether they share channels. Two more DIMMs, one in each empty channel, would make one
eight-channel region: that needs no striping, and the CPU lane would read at about twice this rate.

Settings measured on this machine:

- `--no-expert-cache --ssd-cache 0`: 18 tok/s after a 2647-token prompt. The defaults copy the CPU lane's experts into RAM and
  swap them: 15 tok/s.
- The model locks 76 GB of RAM for the CPU lane's experts with four conversations. This machine has 91 GB. While it loads it takes
  all the RAM it can for a moment, to choose its pages. Other large workloads have about 12 GB left: start them first, or the model
  falls back to the page cache, at about 30 tok/s.
- `--parallel 2` gains 2% in decode. `--reserve-mib 300` gains nothing measurable. Neither is deployed.
- `--dspark-conf 0.3`, more CPU threads, the expert cache: no gain. The expert cache takes decode from 26 to 11 tok/s: each move goes through a bounce buffer, and the RAM is what is short.
- `STRATA_DS4_RAM_SKIP_W` leaves out light CPU-lane picks. At 0.1 decode reaches 41 tok/s, but the top token changes in 17.5% of the rows of
  the reference texts (25% at 0.15). It is off.

P2P and tensor parallel: `hipMemcpyPeer` between these cards runs at 11-22 GB/s without enabling peer access, so the prompt path's copies need no change, and a
chunk's hand-offs take about 0.3% of its time by estimate. The model has no tensor-parallel path: its KDA, MLA and indexer kernels would need splitting by head. It was not
built or measured. A card-to-card sum takes about 46 us on MI50 (docs/SPLIT_ENGINE.md), so two a layer cost about 4 ms a round. That leaves about 7 ms of the 11 ms a
four-way split would save on dense GEMVs, an estimated +12% at best, and the CPU lane's time stays.

## Tests

- `tests/test_glm5_pack.py`: the pack's name map and checks. No model.
- `serve/test_glm5.py`: tool-call translation and parsing, the thinking switch, picture sizes. No model.
- `glm5_kernels_test`: the device kernels against double-precision references, and the KDA snapshots. Needs a GPU, no model.
- `strata-dsv41 --parallel 3 --dspark none --multi-check --ids ... --n 32`: conversations together against each alone. Needs the model.
- `ds4_route_guard_test`: GLM's router and gate against the host, with the rest of the routing. `glm5_moe_bench`: the expert kernels' rate.
- `ds4_gguf_test`, `ds4_iq_test`, `ds4_tq_test`, `ds4_route_guard_test` pass on an MI50 with the wave64 build.

## Where the decode time goes

A round of two drafts takes 55-75 ms and gives 2.1-2.8 tokens. Of it, the GPU's chain takes about 50 ms: 15 ms of dense GEMVs, 12 ms of
experts, 4.5 ms of drafts, the rest small kernels of 4-20 us that wait for each other. The CPU lane takes 20-45 ms, depending on how
many picks a text sends it (50-130 a token), and runs beside the experts' kernels. Only one GPU works at a time (layer split).

What was tried and did not help: reading last round's CPU experts into the caches during attention, guessing the picks before attention (55%
of them), a conversation at a time with drafts instead of shared steps, several rows a warp in the expert kernels, 16-byte loads for Q4_K.
Next steps, by expected gain: two more DIMMs, tensor parallel for the dense layers, drafts in shared steps.
