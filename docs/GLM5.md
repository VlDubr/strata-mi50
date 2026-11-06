# GLM-5.3-Flash on MI50

GLM-5.3-Flash (GGUF architecture `glm5-next`, 320B total parameters, 288 experts) runs on the DeepSeek engine path
(`strata-dsv41`) on 4x MI50 32 GB. This page records the design, the deployment and the measurements.

## Status

Validated on 4x MI50 32 GB, 91 GB RAM, ROCm 10, with `unsloth/GLM-5.3-Flash` UD-Q4_K_XL (199.7 GB).

| Measure | llama.cpp b11786 | Strata |
|---|---|---|
| Decode, short prompt | 10.8 tok/s | 20 tok/s |
| Decode after a 2647-token prompt | 10.7 tok/s | 18 tok/s |
| Prefill, 2647 tokens | 74 tok/s | 145 tok/s |
| Load | about 14 min | about 2 min |

Numerics against llama.cpp on the same weights, greedy:

- The short prompts give the same tokens until a near-tie of the top two tokens.
- With no expert profile, the 2647-token prompt in windows of 16 gave the same 96 of 96 tokens.
- llama.cpp itself differs between its CPU and GPU backends on the same prompts, at step 32 of the long prompt.
- A greedy comparison on long prompts must therefore use the top-k logprobs, not the exact text.

The server's token ids and rendered chat template match llama.cpp on all reference prompts.

## What is not implemented

- **MTP (block 45, `nextn.*`):** packed, not loaded. Decode reads about 9 GB of dense weights and 5 GB of experts a token.
  One draft would share the dense read, but it doubles the experts and the CPU lane, and it needs rollback of the KDA state.
  The estimated gain is below 1.3x, so it was left out.
- **Vision:** `mmproj-BF16.gguf` is not used. Pictures are refused.
- **Parallel conversations:** one at a time (`o.seqs = 1`).
- **Conversation slots:** a conversation that leaves the model is read again from its first token. Restore points inside a
  conversation work, and they include the KDA state.
- **Tensor parallel, the DeepSeek chunk path, the async router:** GLM uses its own path, below.
- **Thinking off:** the template always opens `<think>`. A request with reasoning off gets "Reasoning Effort: Low".

## Design

The engine reads the geometry from the pack, so GLM is a family inside `src/dsv41/ds4_model.cu`, with `g.glm5` as the switch.
It reuses the mHC kernels, top-k, GGUF GEMV, the expert tiers, the CPU lane, the serve protocol and sampling.

- `tools/dsv41/pack_glm5.py` writes the pack: `ds4.txt` with `family glm5-next`, `tensors.txt`, `tokenizer/`.
- `src/dsv41/glm5_kernels.cu` holds KDA (conv, scan, output norm), the absorbed MLA, the k-pool indexer and the sigmoid router.
- `Impl::glm5_attn` is the attention sub-block. `glm5_dense_ffn` is the FFN of layers 0-2. The MoE of a decode step is the
  host-routed path of `layer()`.
- `Impl::glm5_chunks` reads prompts: a thread a GPU, chunks of 256 rows, attention in windows of 16, experts once a chunk.
- `serve/glm5.py` translates the model's tool calls to the Qwen form that `OutputParser` reads.

Differences from DeepSeek-V4.1 that are easy to miss:

- Each sub-block uses its own `pre`. DeepSeek staggers it by one sub-block.
- The head input is a plain mean of the streams: `d.pre` is 1/4 for `hc_pre_norm`.
- The embedding stays Q8_0 on the first GPU and is read in F32, not rounded to BF16.
- Layers 0-2 have no experts: `n_exp` is 0, and `lay.fmt` still says MXFP4 because code reads `L[0].lay.fmt`.
- The caches are by position and in F16, as in llama.cpp: latent, key | gate, pooled key.

## Build

```sh
cmake -S . -B build-glm5 -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
  -DSTRATA_EXPERIMENTAL_GFX906=ON -DCMAKE_HIP_ARCHITECTURES=gfx906 -DSTRATA_BUILD_TESTS=OFF -DSTRATA_BUILD_DSV41=ON \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE=ON -DSTRATA_GGML_DIR=/opt/strata/third_party/llama.cpp
ninja -C build-glm5 strata-dsv41 glm5_kernels_test ds4_gguf_test ds4_iq_test ds4_tq_test ds4_route_guard_test
```

The main `strata` engine keeps its `STRATA_HIP_GFX906` build. `strata-dsv41` uses the wave64 `STRATA_ENABLE_HIP` backend.
Image: `docker build -t strata-gfx906:0.2.10-mi50-glm5 -f deploy/mi50/Dockerfile.glm5 .`

## Deployment

1. Pack, with the paths the container sees:
   `python tools/dsv41/pack_glm5.py --model /data/models/GLM-5.3-Flash-UD-Q4_K_XL/UD-Q4_K_XL --out /data/packs/glm5-flash`
2. Put the expert profile at `/data/packs/glm5-flash/expert-profile.bin`. Without one the experts are placed in index order,
   and the CPU lane takes 33% of the picks instead of about 18%.
3. Write `/data/config/glm5.json`. Do not name it `strata-*.json`, or the Qwen image's setup takes it:

```json
{"exe": "/opt/strata/engine/strata-dsv41",
 "args": ["--pack", "/data/packs/glm5-flash", "--max-context", "131072", "--eos", "154820,154827,154829", "--dspark", "none",
          "--vision", "off", "--tp", "1", "--threads", "off", "--prefill-chunk", "0",
          "--expert-profile", "/data/packs/glm5-flash/expert-profile.bin", "--profile-readonly", "--no-expert-cache", "--ssd-cache", "0"],
 "cwd": "/opt/strata", "tokenizer": "/data/packs/glm5-flash/tokenizer", "model_name": "glm-5.3-flash",
 "arch": "deepseek41", "family": "glm5-next", "log": "/data/logs/strata-glm5.log", "lib_dirs": ["/opt/rocm/lib"], "port": 8080,
 "engine_ready_s": 3600, "backend": "hip", "experimental_gfx906": true, "gpu": [0, 1, 2, 3], "gpus_asked": true, "api_key": "..."}
```

`"arch": "deepseek41"` selects the engine plumbing. `"family": "glm5-next"` selects the template, stop ids, tool calls, effort
words and sampling.

4. Start with `deploy/mi50/run-glm5.sh`. It takes every GPU and refuses to start beside the Qwen container.

Settings measured on this machine:

- `--no-expert-cache --ssd-cache 0`: 18 tok/s. The defaults copy the CPU lane's experts into RAM and swap them: 15 tok/s.
- The model needs about 65 GB of RAM for the page cache of the CPU lane's experts. Other large workloads slow decode.

## Tests

- `tests/test_glm5_pack.py`: the pack's name map and checks. No model.
- `serve/test_glm5.py`: tool-call translation and parsing. No model.
- `glm5_kernels_test`: the device kernels against double-precision references. Needs a GPU, no model.
- `ds4_gguf_test`, `ds4_iq_test`, `ds4_tq_test`, `ds4_route_guard_test` pass on an MI50 with the wave64 build.

## Where the decode time goes

A decode step takes about 54 ms: about 33 ms of waits for the GPUs, 16 ms in the CPU lane, the rest host launches.
Only one GPU works at a time on a single conversation (layer split). The dense weights alone take about 12 ms at the
MI50's memory rate. Next steps, by expected gain: tensor parallel for the dense layers, the async router.
