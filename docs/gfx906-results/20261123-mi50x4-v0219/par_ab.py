#!/usr/bin/env python3
"""Parallel A/B: stock (no PW/PIPED/batch-mtp) vs best, 10 concurrent clients, same prompts.
Restarts container per variant via run.sh, restores best config at end."""
import json, os, subprocess, sys, time, urllib.request

HOME = os.path.expanduser("~")
CFG = f"{HOME}/strata-data/config/strata-unsloth-ud-q5_k_xl.json"
RUN = f"{HOME}/strata-v0219-build/strata-merge/deploy/mi50/run.sh"
IMAGE = "strata-gfx906:0.2.19-mi50-vision"
URL = "http://127.0.0.1:8095"
sys.path.insert(0, f"{HOME}/strata-deploy")
import par_bench


def set_config(best):
    c = json.load(open(CFG))
    env = c.setdefault("env", {})
    args = c["args"]
    if best:
        env["STRATA_PW_WITH_BATCH"] = "1"
        env["STRATA_PIPED_MTP"] = "1"
        if "--batch-mtp" not in args:
            args.append("--batch-mtp")
    else:
        env.pop("STRATA_PW_WITH_BATCH", None)
        env.pop("STRATA_PIPED_MTP", None)
        if "--batch-mtp" in args:
            args.remove("--batch-mtp")
    env.pop("STRATA_MMQ_KQ", None)
    c["parallel"] = 8
    json.dump(c, open(CFG, "w"), indent=1)


def start():
    e = dict(os.environ, IMAGE=IMAGE, NAME="strata-mi50", PORT="8095")
    for attempt in range(4):
        r = subprocess.run(["sh", RUN], capture_output=True, env=e, text=True)
        if r.returncode == 0:
            break
        time.sleep(20)
    else:
        raise RuntimeError(f"run.sh failed: {r.stderr[-300:]}")
    t0 = time.time()
    while time.time() - t0 < 1200:
        try:
            with urllib.request.urlopen(URL + "/health", timeout=5) as r:
                if json.loads(r.read()).get("loaded"):
                    return round(time.time() - t0)
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError("no health")


def run_par(n, tag):
    import threading
    results = {}
    par_bench.tag = tag
    threads = [threading.Thread(target=par_bench.one, args=(i, results)) for i in range(n)]
    t0 = time.time()
    for t in threads: t.start()
    for t in threads: t.join()
    wall = time.time() - t0
    total = sum(r["tok"] for r in results.values())
    return {"tag": tag, "n": n, "wall_s": round(wall, 1), "total_tok": total,
            "aggregate_tok_s": round(total / wall, 1),
            "per_req": [results[i]["per"] for i in sorted(results)],
            "ttft": [results[i]["ttft"] for i in sorted(results)]}


def main():
    out = []
    for best in (False, True, False, True):
        name = "best" if best else "stock"
        set_config(best)
        st = start()
        # warm-up
        run_par(2, f"{name}-warm")
        r = run_par(10, f"{name}-10")
        r["startup_s"] = st
        out.append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)
        with open(f"{HOME}/strata-deploy/par-ab-results.jsonl", "a") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    set_config(True)
    start()
    print("restored best config", flush=True)


if __name__ == "__main__":
    main()