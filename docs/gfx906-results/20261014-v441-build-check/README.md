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

Nothing was stopped or restarted; the 8082 service (server 1734867, vision 1734877, text engine
1734906) ran the whole round on the v0.1.43.1 binaries. Both artifacts were then **deployed to the
live paths** on owner instruction ("deploy, I restart myself") at ~19:24 UTC: `build-text-rocm10/strata`
= `050b83e6…`, `build-vision-rocm10/bin/strata-vision` = `f8a183ff…`, with the running processes
keeping the v0.1.43.1 inodes (`(deleted)`) until the owner restarts. See
[GFX906.md's deployment section](../../GFX906.md#v01441-installed-on-8082s-paths-without-a-restart-2026-10-14-1924-utc).
