# v0.1.44.1 build check on T5810 (2026-10-14)

Raw evidence for the [v0.1.44.1 build check](../../GFX906.md#v01441-build-check-on-t5810-2026-10-14-02080215-utc).

- `build-identity.json` — the merge, the two gfx906 build fixes, the pin, both staged artifacts.
- `artifact-hashes.txt` — full shas of the merge and fix commits, both staged artifacts, and the two
  deployed (running) binaries for contrast: text `11b847dc…` (v0.1.43.1) and vision `709fc4d2…`
  are untouched by this round.
- `host-tests.txt` — the 21 host-side ctest tests (no GPU), all passing.
- Machine side: `~/dev/strata-gfx906/logs/v441-build-20261014T021846Z/` carries the configure logs,
  the full 329-edge build log (`build.log`), the vision configure/build logs, and the complete
  `ctest` output (`v441-ctest-full.txt`: 69 tests — 23 pass, 44 GPU-requiring failures, 2 no-device
  skips).

Nothing was installed, restarted or reconfigured; the 8082 service (server 1065666, vision 1065676,
text engine 1065703) ran the whole round on the v0.1.43.1 binaries.
