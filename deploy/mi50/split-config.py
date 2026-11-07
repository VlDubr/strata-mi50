#!/usr/bin/env python3
"""The split engine's run config from the Qwen config that setup wrote: split-config.py CONFIG_DIR OUT [LANES].

The model, API key, tokenizer and picture reader stay. The engine and its switches change; docs/SPLIT_ENGINE.md says why each is set."""
import glob
import json
import os
import sys


def main(config_dir: str, out: str, lanes: int = 4) -> None:
    found = sorted(glob.glob(os.path.join(config_dir, "strata-*.json")))
    if not found:
        sys.exit(f"no strata-*.json in {config_dir}: run deploy/mi50/run.sh once first")
    cfg = json.load(open(found[0], encoding="utf-8"))
    args = cfg["args"]

    def flag(name: str) -> str:
        if name not in args:
            sys.exit(f"{found[0]} has no {name}")
        return args[args.index(name) + 1]

    cfg["exe"] = "/opt/strata/engine/strata-split-4"
    cfg["args"] = ["--native", flag("--native"), "--mtp", flag("--mtp"), "--draft-vocab", flag("--mtp-draft-vocab"),
                   "--max-context", "131072", "--spec", "2", "--lookup", "15", "--conversation-cache-mib", "4096", "--batch", str(lanes)]
    if cfg.get("vision"):
        cfg["args"].append("--vision")
    cfg["env"] = {"FNX_ARENA": "0", "FNX_EXPERT_RAM_GIB": "off", "FNX_PF_RCCL": "0", "FNX_RING_GRID": "32", "FNX_PRECAP": "0"}
    cfg["parallel"] = lanes
    cfg["drafting"] = "mtp"
    cfg["split_engine_note"] = "all-cards split on 4x MI50"
    cfg["log"] = os.path.join(os.path.dirname(cfg.get("log") or "/data/logs/x"), "strata-split.log")
    cfg.pop("layer_split", None)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 4)
