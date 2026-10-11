# v0.1.43 deploy-prep build on T5810 (2026-10-11 18:55-19:04 UTC)

Machine: Dell Precision T5810, 2x AMD Instinct MI50 32 GB (gfx906),
Ubuntu 26.04.1 LTS, kernel 7.0.0-31-generic, ROCm 10.0.0-gfx906. Source: `gfx906` at `7ffd929e`
(upstream v0.1.43 `7695859a` merged as `24eb7211`,
fork conversation-copy fix `e450d67a` included). Pinned llama.cpp/ggml `3cf03257f219afbe7334045ff7c6a06ac68c627d`.

What this round did: fast-forwarded the T5810 checkout `3562a346` -> `7ffd929e`, built v0.1.43 in a new
directory `build-text-rocm10-v43`, ran the tests that do not need the GPUs, and staged the artifact.
The 8082 service (server 375759, vision 375769, engine 375796) kept running the whole round and was not
stopped, restarted or reconfigured. Nothing was deployed.

## Files

- `build-identity.json` - source commits, configure switches, build counts, artifact hashes, test results,
  and the explanation of why this hash differs from the earlier build-check artifact.
- `host-tests.txt` - one line per test binary: exit code and its own last output line, verbatim.
- `artifact-hashes.txt` - SHA256 of the staged v0.1.43 binary, the deployed v0.1.42 copy-fix binary and
  the unchanged vision encoder.

The 1.8 MB compiler log is not committed; it stays on the machine at
`/home/chris/dev/strata-gfx906/logs/v43-prep-20261011T185526Z/build.log` (with `configure.log`).

## Key numbers

- Build: exit 0, 2 m 56 s, 300 targets, 0 errors, 2103 warnings (per class identical to the 16:49 UTC
  build-check round).
- Staged artifact: `build-text-rocm10-v43/strata`, 22,729,416 B,
  SHA256 `dd7eaf975c612a505c4ab2c734bd87273c548c7ee2c016ebec94eb47bd70e442`; `ldd` shows 20 ROCm 10.0
  libraries and none from `/opt/rocm-7.2.4`.
- Deployed pair unchanged: text `ff24e50b980fa555c7a014f7bd60f67a7a61c36a650a4ee03ca885316755d8dc`,
  vision `709fc4d2c34abbede8b17cb93a4a9571c931338339d3848691c5b59f156d8452`.

## Limits of this evidence

- **No runtime acceptance of the v0.1.43 binary**: it was never executed. Both cards were held by the
  running service (2,186 MiB VRAM free), so no engine start, model request, image request or speed
  measurement was possible. Build success and unit tests are not model parity or speed.
- `conversation_copy_test` (165 checks) is a single-device test and ran on GPU0 while the service held
  both cards; the service was unaffected (same PIDs, `/health ok loaded:true`, VRAM unchanged, 0 new
  kernel page faults), but it should be re-run in a maintenance window.
- Not run: `conversation_snapshot_test` (3901 checks, device), `vmm_test`, and the `*_parity` group
  (`STRATA_BUILD_TESTS=OFF`, so those binaries were not built).
- `platform_memory_test` fails on this machine for an environment reason: memlock is 8 MiB
  (`ulimit -l` = 8192 KB) and the test locks a 256 MiB region. The older `build-hip/` build's copy of the
  same test fails identically here, so it is not a v0.1.43 regression.
- `ple_reader_test` exits 2 with `nothing to do`: it is a CLI tool that needs a fixture argument, not a
  test failure.
- The staged binary's hash differs from the 16:49 UTC build-check artifact `adf2c616…`. The compiled
  source is the same (`8c84ecb0..7ffd929e` is docs-only); the binary embeds absolute source paths and the
  two builds used different directories. Artifact hashes are only comparable within one build directory.
