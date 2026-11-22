# syntax=docker/dockerfile:1
#
# strata-mi50 for AMD Instinct MI50 / MI60 (gfx906), ROCm 10 (community gfx906 build).
# Text engine: upstream STRATA_HIP_GFX906 path (the one docs/GFX906.md marks "current").
# This is the image of the tested 4x MI50 32 GB deployment (README.md, "Quick start").
#
# Build from the repository root:
#   docker build -t strata-mi50 -f deploy/mi50/Dockerfile .

FROM mixa3607/rocm-gfx906:10.0-complete-20261018172427

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 LANG=C.UTF-8 STRATA_EXECV=1 \
    ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build git curl ca-certificates libatomic1 libgomp1 \
        python3 python3-pip python3-venv unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/strata
COPY . .

RUN python3 -m venv .venv \
    && .venv/bin/pip install --no-cache-dir --upgrade pip \
    && .venv/bin/pip install --no-cache-dir -r terminal/requirements.txt

# 0.2.19: no llama.cpp download any more; the engine builds against the vendored ggml at the fork's pin
# (engine/third_party/strata-ggml, commit 3cf03257), so no -DSTRATA_GGML_DIR is needed.

RUN cmake -S engine -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DSTRATA_HIP_GFX906=ON -DCMAKE_HIP_ARCHITECTURES=gfx906 \
        -DSTRATA_PORTABLE=ON -DSTRATA_BUILD_TESTS=OFF \
        -DCMAKE_C_COMPILER=/usr/bin/cc -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
        -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=/opt/rocm \
    && cmake --build build -j "$(nproc)" --target strata

# Stamp the engine the way setup.py's build_engine_hip() does, so setup reuses it instead of building the
# deprecated fork wave64 backend. setup.py lives in terminal/ now: its ROOT is /opt/strata/terminal and it
# stamps terminal/engine/. The build dir stays /opt/strata/build. The legacy deploy path /opt/strata/engine/
# is the engine's source tree, so the binary is copied beside it too, for configs that still name it.
RUN PYTHONPATH=terminal .venv/bin/python - <<'PYEOF'
import json, shutil, setup
from pathlib import Path
root = Path("/opt/strata")
eng = setup.ROOT / "engine"
eng.mkdir(exist_ok=True)
shutil.copy2(root / "build" / setup.EXE, eng / setup.EXE)
meta = {"source": "local-hip", "backend": "hip", "version": setup.source_version(), "archs": ["gfx906"],
        "experimental_gfx906": True, "vision": "none", "lib_dirs": ["/opt/rocm/lib"],
        "src": setup.source_hash(setup.ENGINE_SOURCES), "build": "STRATA_HIP_GFX906 (upstream gfx906 text engine)"}
floor = setup.cpu_floor(True)
if floor:
    meta["isa_floor"] = floor
(eng / "BUILD.json").write_text(json.dumps(meta, indent=1))
legacy = root / "engine"
shutil.copy2(eng / setup.EXE, legacy / setup.EXE)
shutil.copy2(eng / "BUILD.json", legacy / "BUILD.json")
print(meta)
PYEOF

# data/ moved into terminal/; keep the old /opt/strata/data path working for configs and docs
RUN ln -s terminal/data /opt/strata/data

RUN rm -rf build

COPY --chmod=755 deploy/mi50/entrypoint.sh /opt/strata/strata-entry.sh

VOLUME ["/data"]
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=1800s --retries=3 \
  CMD curl -fs "http://127.0.0.1:${PORT:-8080}/health" || exit 1
ENTRYPOINT ["/opt/strata/strata-entry.sh"]
