# v0.1.45 build check on T5810 (2026-10-17)

Raw evidence for the [v0.1.45 build check](../../GFX906.md#v0145-build-check-on-t5810-2026-10-17-2248-2305-utc).

- `build-identity.json` — the merge `d3d94a1f`, the three gfx906 build fixes (`8b62485e`, `e31ba87b`,
  `b1daaa28`), the llama.cpp pin, both staged artifacts and the exact configure command lines.
- `configure.log` — the text-engine configure (exit 0, `Strata: HIP (gfx906 compat) enabled, arch gfx906`,
  `ggml commit: 3cf03257f`).
- `host-tests.txt` — the same 21 host-side ctest tests as the v0.1.44.1 round, all passing with
  `CUDA_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=`.
- `ctest-full.txt` — the whole suite: 77 tests, 25 pass, 52 fail (every one a GPU-requiring `*_parity`,
  `cuda_device_selftest` or `platform_memory_test` while the live 8082 service holds 95 % / 98 % of both
  cards' VRAM), 2 skipped for no device (`conversation_copy_test`, `conversation_snapshot_test`).

Machine side: `~/dev/strata-gfx906/logs/v45-build-20261017T224855Z/` carries the configure logs, the
failed first build (`build.log`, the four undefined `f16kq` symbols), the incremental build after the
fix (`build2.log`), the clean rebuild (`build-clean.log`, 304 edges, 0 errors, 2808 warnings,
3 m 36.9 s), the conversation-test reconfigure and rebuild (`build3.log`), and both vision
configure/build logs (275 edges, 0 errors, 0 warnings, 4 m 14.6 s).

Nothing was stopped or restarted. The 8082 service (server 1786203, vision 1786213, text engine
1786240, up since 2026-10-14 ~20:55 UTC on the v0.1.44.1 pair) ran the whole round. The text artifact
was then **deployed to the live path** on owner instruction ("deploy, I restart myself") at 23:15 UTC:
`build-text-rocm10/strata` = `1ba389de…`, while the running engine keeps its v0.1.44.1 inode
(`(deleted)`) until the owner restarts. See
[GFX906.md's deployment section](../../GFX906.md#v0145-installed-on-8082s-text-path-without-a-restart-2026-10-17-2315-utc).
