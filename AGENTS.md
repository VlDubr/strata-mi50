# 尽量使用 ASD-STE100（Simplified Technical English）

# strata-mi50 — Development Rules

Version 0.2.19, engine 0.1.65. The public model families are
Qwen3.8-Flash-Next and Qwen3.6-35B-A3B. Read **[docs/AI_REFERENCE.md](docs/AI_REFERENCE.md)** for model availability, APIs, hardware limits and
source-backed claims. Articles and videos should use **[docs/MEDIA_KIT.md](docs/MEDIA_KIT.md)**.

This is a dedicated experimental MI50/MI60 gfx906 fork of `Niko1221/Strata`, synchronized through
v0.2.19 / `c6f6427d9` (original port: v0.1.30 / `30ec18e`).
The working branch is `mi50` of `VlDubr/strata-mi50` (a fork of `Niko1221/Strata`); the gfx906 port and its history
come from `webzone/strata-gfx906` (branch `gfx906`). Absorb upstream updates with
reviewed merge commits, never rebasing published fork history or pushing to `upstream`.
Preserving and optimizing real gfx906 support is the fork's purpose: upstream's wave32-only GPU
support must not replace it. MI50 has archived hardware validation; MI60 still needs its own testing.
All project documentation and maintenance notes must be written in English. Preserve raw test inputs and captured output verbatim as evidence.

1. **Read the device record before operating the machine.** The T5810 authority record is
   `/Users/chris/dev/local_devices/t5810/README.md`. Use its documented `chris@t5810` SSH entry;
   do not guess accounts, scan ports, or search unrelated projects for credentials.
2. **Protect connectivity and existing workloads.** Before builds/tests, inspect GPU, RAM, disk,
   Docker, and listener state. Do not stop workloads to free resources. Do not change drivers, ROCm,
   networking, thermal/tuning services, or restore historical containers. Keep project dependencies
   in this repository's `.venv`; install no system packages.
3. **Make gfx906 compatibility opt-in.** `STRATA_EXPERIMENTAL_GFX906` defaults OFF; the installer
   requires `--experimental-gfx906`. Compile for the real gfx906 architecture and retain runtime
   architecture checks. Never impersonate a GPU or use `HSA_OVERRIDE_GFX_VERSION`. Do not substitute
   RDNA TheRock wheels; there is no gfx906 wheel index.
4. **Preserve logical wave32 on physical wave64.** Do not mechanically change every `32` to `64`.
   Select ballot halves using linear block-thread indexing. Numerically test 2D/3D blocks, signed
   INT8 dot4 behavior/overflow, and shuffles. Recalibrate the gfx906 clock conversion on every test card.
5. **Use layer/pipeline splitting, not tensor parallelism.** Each GPU owns its layers and caches.
   Obtain a separate mapped-host alias on each device and synchronize the producer before the consumer
   reads. Do not require P2P or run dual-GPU tests while either GPU is busy. Tests enabled by
   `STRATA_HIP_MULTIGPU_TESTS=ON` require two visible cards; a missing-card skip is not a pass.
6. **Pin dependencies and model identity.** Pin llama.cpp/ggml to
   `3cf03257f219afbe7334045ff7c6a06ac68c627d`. The acceptance model is GSQ-RCO IQ2_XS; use the
   Hugging Face revision and LFS SHA256 values in the tools. A filename is not the type of every
   tensor: this model mixes IQ2_S, IQ2_XXS, and IQ1_M gate/up tensors with Q2_0 down tensors. Do not
   force every expert through MMQ or write a duplicate ~35.5 GB main-model `experts.bin`; use the
   existing resident-from-GGUF path.
7. **Protect disk space and source data.** Keep at least 4 GiB free during downloads/preparation.
   Calculate peak space for GGUF, float pack, MTP source tensors, and quantized/runtime outputs first.
   Resume only with validated source identity and ranges; never overwrite a conflicting/corrupt file
   or delete unrelated models/images to make room.
8. **Report evidence accurately.** An independent CPU dequantizer/double-accumulation oracle is a
   synthetic operator check; a dual-GPU handoff probe is not model parity. Disclose skipped tests,
   missing fixtures, and sampled-only validation. Report model tok/s only from real generation; do
   not convert dot4 microbenchmarks or warm-cache operator timings into model speed.
9. **Test, document, commit, and push changes.** Commit related changes to `origin/gfx906`; never
   force-push or discard another author's work. Update `docs/GFX906.md`. After every upstream merge,
   update the current source-version line at the top of `README.md` and its current version/commit
   references to match the merged source, in the same merge commit. Preserve historical benchmark and
   deployment version labels; never relabel old evidence as the new version. Keep unrelated machine
   diagnostics out of this project. Do not commit weights, binaries, virtual environments, or full
   compiler-warning logs; small, auditable validation evidence may be committed.
10. **Keep only the newest workload result in `README.md`, and always name both rates.** The *MI50 workload
   results* section holds one observation: the latest. Move the previous one to `docs/GFX906.md` under its
   own dated heading, with its original version and ROCm labels intact, before adding the new one. State the
   decode rate and the prefill rate explicitly. Give prefill in two labelled forms: the **real** rate (new
   prompt tokens per prefill second, measured with `cached_tokens: 0`) and the **effective** rate (the whole
   prompt per prompt-phase second, reused K/V tokens counted). The effective number shows how fast a warm
   turn is accepted; it is not compute throughput, so never quote it as a prefill speed. Record how each
   number was measured, and keep the raw payload under `docs/gfx906-results/`.
11. **Resolve every upstream-merge conflict by the recorded pattern, and carry no workflow files.** Fork behavior
    wins in a conflict. Take upstream's side only where the two do not overlap. Rebuild the fork's gating on top
    of upstream's new structure instead of keeping the old shape. Record each conflict in `docs/GFX906.md` —
    both sides' intent and the resolution — in the same merge commit. Merge no `.github/workflows/` file:
    this fork carries none, and the origin token cannot push them.

## Upstream navigation

- **The app** (`app/desktop`, the Tauri shell; `app/host`, the Node host and the web UI). It is for people who do
  not want a terminal. One installer per system ([latest release](https://github.com/Niko1221/Strata/releases/latest)):
  `Strata-0.2.19-windows-x64-setup.exe`, `Strata-0.2.19-linux-x86_64.AppImage`, `Strata-0.2.19-macos-arm64.dmg`.
  The Windows and Linux installers carry no engine: the app downloads the engine for the graphics card on first
  start, or installs a file the user already has. The Mac dmg keeps its small Metal engine inside.
- **The Terminal** (`terminal/strata.py`, `terminal/strata_cli/`, `terminal/strata_control/`): the `strata` command.
  It sets up a model, serves it and watches it. Agents use this part.
- **The engine and the server** (`terminal/src/`, `terminal/include/`, `terminal/metal/`, `terminal/sycl/` for the
  C++/CUDA/HIP/SYCL/Metal engine, `engine/` for the split engine; `terminal/serve/` for the Python server with an
  OpenAI- and Anthropic-compatible API and a web page). `terminal/setup.py`, started by `START-HERE.bat` or
  `setup.sh`, is setup from a source checkout, for developers. The release paths are the installer and
  `strata launch`.

When invoking `omp`, use the `herdr` skill from a Herdr-managed pane, as requested by the owner.
The primary agent/thread owns only architecture, organization, review, and integration closeout;
concrete tasks are delegated through `herdr` to fresh omp instances, at most four concurrent, each
with a disjoint file scope (worktrees allowed as needed). When a task ends, the primary closes that
round's omp instances and removes only that round's worktrees/branches, never another round's leftovers.

Upstream's general user workflow is in `docs/AI_SETUP.md` and `docs/MCP_SERVER.md`; for this fork's
MI50/MI60 installation, use `docs/GFX906.md` and the rules above instead of upstream's RDNA/prebuilt defaults.
The owner-authorized T5810 private-LAN deployment uses `0.0.0.0:8082` without an API key; do not add
one unless the owner requests it. Other non-loopback deployments require an API key. Engine/API details are in `docs/DETAILS.md`,
AMD upstream scope in `docs/AMD_HIP.md`, and pipeline splitting in `docs/MULTI_GPU.md`.
GPU-independent setup tests run with `.venv/bin/python -m unittest discover -s terminal/tools -p 'test_setup_*.py'`.
Document measured results with their hardware, workload and validation limits, not unsupported performance claims.

Follow **[docs/AI_SETUP.md](docs/AI_SETUP.md)**. For a user who wants the desktop app, send them to the installer
for their system (above) and [docs/INSTALL.md](docs/INSTALL.md). For an agent that has a shell, use the Terminal; the
short version is:

1. Install the Terminal. It needs no administrator rights.

   ```sh
   curl -fsSL --connect-timeout 15 --max-time 120 https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh                  # Linux, Mac
   irm -TimeoutSec 120 https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex                         # Windows PowerShell
   ```

   On Windows, open a new terminal afterwards so that `strata` is on the PATH.
2. Check what would happen. Nothing is changed:

   ```sh
   strata launch --plan --family qwen --model q2_0
   ```

3. Set up and serve without questions. It downloads the engine for the card and the model (tens of GB), starts the
   server, prints the addresses and returns. The server keeps running.

   ```sh
   strata launch --family qwen --model q2_0 --yes --no-tui
   ```

   Useful flags (all of them: `strata launch --help`): `--family`, `--model-path` (files the user already has),
   `--data-dir`, `--context 32k|64k|128k`, `--vision off|on|gpu|cpu`, `--parallel N`, `--gpu 0,1`,
   `--access local|lan|cloudflare|tailscale`, `--port`, `--api-key-env NAME`, `--new-key`, `--save-preset NAME`,
   `--output json`. `strata launch --status` shows what is serving, `strata launch --stop` stops it, and
   `strata launch --show-key` prints the API key.
4. Connect the user's apps: `strata connect` lists them, `strata connect <app> --check` sends one prompt and one tool
   call the way that app does.

Keep the server on `127.0.0.1` unless the user asks for more. Beyond it an API key is recommended: `--access lan` makes one for you, and `--no-api-key` serves without one (a warning, never a refusal; on a public tunnel the warning is stronger). Source setup (`START-HERE.bat`, `setup.sh`) is the fallback for a developer or for a system with no release package; it is
documented in [docs/SOURCE_INSTALL.md](docs/SOURCE_INSTALL.md). Those wrappers exist in a source checkout, not in the
released app. Do not tell an app user to find a `.bat` file.
Source setup does not make an unsupported model or backend compatible. Strata's MCP server ([docs/MCP_SERVER.md](docs/MCP_SERVER.md)) offers the same steps as tools.

## Working on the code

- How the engine works, every measured number, the API and all settings: [docs/DETAILS.md](docs/DETAILS.md) and
  the [paper](docs/paper/Strata-Paper.pdf).
- The Terminal: [docs/TERMINAL.md](docs/TERMINAL.md) and [docs/TERMINAL_LAUNCH.md](docs/TERMINAL_LAUNCH.md). The app,
  the packages and the update feed: [packaging/README.md](tools/packaging/README.md) and [packaging/UPDATES.md](tools/packaging/UPDATES.md).
- AMD (HIP) build and validation: [docs/AMD_HIP.md](docs/AMD_HIP.md); multi-GPU: [docs/MULTI_GPU.md](docs/MULTI_GPU.md);
  Mac (Metal): [docs/MAC.md](docs/MAC.md).
- Tests: the Terminal and Control tests are in `terminal/tests/terminal` (`python -m pytest terminal/tests/terminal`). The app's UI
  tests run with `npm test` in `app/host`. Setup's own tests run without a GPU or downloads:
  `python tools/test_setup_<name>.py` (for example `terminal/tools/test_setup_amd.py`, `terminal/tools/test_setup_choices.py`).
- Keep the docs' style: plain words, short sentences, written for non-technical users first. Give measured numbers
  with what they were measured on. Make no claim without a measurement.

## Contributing a change or report

The rules are in **[CONTRIBUTING.md](.github/CONTRIBUTING.md)**. Read it before you open an issue or a pull request, and
follow it whether a person or an agent writes the text. The short version:

- Start every issue and pull request title with its area: `[Engine]`, `[GUI]`, `[TUI]`, `[Benchmark]`, `[Docs]` or
  `[Other]`, for example `[Engine] fix …`. A check on GitHub fails a pull request whose title has no area.
- Search the open and closed issues and pull requests first, and add to a thread that already covers your point.
- Agree the scope in an issue before a large feature, a refactor or a change to a default.
- One problem per pull request. Lead with the problem and the result, then the approach, then the validation
  (passed, failed and not run, kept apart) and the limits.
- A new feature is opt-in, and the default path stays byte-identical to the last release. Say how you checked.
- Build every backend a file touches (CUDA, HIP, SYCL) before asking for review.
- Change a default only where you measured it faster, and show the numbers with what they were measured on.
- A report from hardware the maintainers do not have is welcome. Follow
  [docs/COMMUNITY_BENCHMARKS.md](docs/COMMUNITY_BENCHMARKS.md), compare against a same-day run of the build you are
  testing, and say what you did not test.
- Open test requests and the hardware that is wanted are listed in [docs/TEST_REQUESTS.md](docs/TEST_REQUESTS.md).
