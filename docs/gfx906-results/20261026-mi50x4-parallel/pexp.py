#!/usr/bin/env python3
"""Parallel-slot experiments: for each variant, rewrite the model config from tuned.json (top-level keys, args, env),
restart the container (run.sh), print the engine's slot / memory lines, then run par.py at the given concurrencies.

    pexp.py VARIANTS.json "1,4,10" [max_tokens]

A variant: {"name", "top": {"parallel": 10, "layer_split": "12,24,36"}, "args": {"--flag": "v" | "" (bare flag) | null},
"env": {...}, "image": optional}.  tuned.json is restored at the end."""
import json, os, re, subprocess, sys, time
sys.argv, rest = sys.argv[:1], sys.argv[1:]
import ab  # noqa: E402  (start(), the paths)

HOME = os.path.expanduser("~")
LOG = f"{HOME}/strata-data/logs/strata-unsloth-ud-q5_k_xl.log"
KEYS = re.compile(r"batch|slot|expert cache|free of|room for|resident|kv|stream|trim|pinned|memory guard|group|"
                  r"refus|cannot|error|fail|warn", re.I)


def configure(v):
    c = json.load(open(ab.TUNED))
    for k, val in (v.get("top") or {}).items():
        if val is None:
            c.pop(k, None)
        else:
            c[k] = val
    args = c["args"]
    for k, val in (v.get("args") or {}).items():
        if k in args:
            i = args.index(k)
            bare = i + 1 >= len(args) or args[i + 1].startswith("--")
            if val is None:
                del args[i:i + (1 if bare else 2)]
            elif val != "":
                if bare:
                    args.insert(i + 1, str(val))
                else:
                    args[i + 1] = str(val)
        elif val is not None:
            args += [k] if val == "" else [k, str(val)]
    env = c.setdefault("env", {})
    for k, val in (v.get("env") or {}).items():
        if val is None:
            env.pop(k, None)
        else:
            env[k] = str(val)
    json.dump(c, open(ab.CFG, "w"), indent=1)


def engine_lines():
    lines = open(LOG, encoding="utf-8", errors="replace").read().splitlines()
    start = max(i for i, l in enumerate(lines) if "engine started" in l)
    return [l[:220] for l in lines[start:] if KEYS.search(l) and "capturing the" not in l][:60]


def main():
    variants = json.load(open(rest[0]))
    ns = [int(x) for x in rest[1].split(",")]
    mt = rest[2] if len(rest) > 2 and not rest[2].startswith("--") else "300"
    long_chars = next((a.split("=", 1)[1] for a in rest if a.startswith("--long=")), None)
    try:
        for v in variants:
            print(f"===== {v['name']}", flush=True)
            configure(v)
            try:
                s = ab.start(v.get("engine"), v.get("docker", ""), v.get("image"))
            except Exception as e:  # noqa: BLE001
                print("START FAILED", e, flush=True)
                subprocess.run(f"docker logs --tail 60 strata-mi50 2>&1 | tail -60", shell=True)
                for l in engine_lines()[-30:]:
                    print("  |", l)
                continue
            print("startup", s, "s", flush=True)
            for l in engine_lines():
                print("  |", l, flush=True)
            ab.ask("Привет!", 16)
            for n in ns:
                r = subprocess.run(["python3", "par.py", str(n), mt], capture_output=True, text=True)
                print(r.stdout.strip() or r.stderr.strip()[-500:], flush=True)
            if long_chars:
                r = subprocess.run(["python3", "longctx.py", long_chars, "300"], capture_output=True, text=True)
                print("long", long_chars, r.stdout.strip() or r.stderr.strip()[-500:], flush=True)
            subprocess.run("free -g | head -2; rocm-smi --showmemuse 2>/dev/null | grep 'VRAM%'", shell=True)
    finally:
        if "--keep" not in rest:
            configure({})
            ab.start()


if __name__ == "__main__":
    main()
