#!/usr/bin/env python3
"""A/B harness for the v0.2.19 deployment: variants over ~/strata-data/config/tuned.json,
container recreated with IMAGE=strata-gfx906:0.2.19-mi50-vision via the merged run.sh.
Fixed greedy workload with a per-run nonce (cold prompt, no conversation-cache reuse).
One JSON line per variant to ~/strata-deploy/ab2-results.jsonl. Restores tuned config at end."""
import glob, hashlib, json, os, subprocess, sys, time, urllib.request

HOME = os.path.expanduser("~")
CFG = f"{HOME}/strata-data/config/strata-unsloth-ud-q5_k_xl.json"
TUNED = f"{HOME}/strata-data/config/tuned.json"
OUT = f"{HOME}/strata-deploy/ab2-results.jsonl"
RUN = f"{HOME}/strata-v0219-build/strata-merge/deploy/mi50/run.sh"
IMAGE = "strata-gfx906:0.2.19-mi50-vision"
URL = "http://127.0.0.1:8095"
KEY = open(f"{HOME}/strata-deploy/api_key").read().strip()
HDR = {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}

DOC = ""
for p in sorted(glob.glob(f"{HOME}/strata-deploy/corpus/docs/*.md")):
    DOC += open(p, encoding="utf-8", errors="replace").read()
    if len(DOC) > 60_000:
        break
DOC = DOC[:60_000]

WORK = [
    ("chat", "Объясни простыми словами, чем TCP отличается от UDP, и приведи по два примера приложений для каждого.", 500),
    ("code", "Write a Python function that parses an ISO-8601 date string and returns the weekday name. Include tests.", 500),
    ("story", "Напиши короткий рассказ (примерно 300 слов) о роботе на маяке.", 500),
    ("long", DOC + "\n\nName the three main topics of these documents.", 200),
]


def metric_rows():
    with urllib.request.urlopen(urllib.request.Request(URL + "/metrics", headers=HDR), timeout=30) as r:
        return json.loads(r.read()).get("requests", [])


def ask(prompt, max_tokens, nonce):
    t_send = time.time()
    body = {"model": "x", "max_tokens": max_tokens, "temperature": 0,
            "messages": [{"role": "user", "content": f"[ab {nonce}]\n" + prompt}]}
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), HDR)
    with urllib.request.urlopen(req, timeout=1800) as r:
        m = json.loads(r.read())["choices"][0]["message"]
    out = (m.get("reasoning_content") or "") + "\n---\n" + (m.get("content") or "")
    rows = [x for x in metric_rows() if (x.get("time") or 0) >= t_send - 2]
    last = max(rows, key=lambda x: x.get("time", 0)) if rows else {}
    pt = last.get("prompt_tokens") or 0
    pms = last.get("prompt_ms") or 0
    read = last.get("prompt_read")
    cached = last.get("reused") or 0
    newtok = read if read is not None else pt - cached
    real = round(newtok / (pms / 1000), 1) if pms and newtok else None
    return {"dec": last.get("decode_tok_s"), "out": last.get("output_tokens"),
            "prompt": pt, "cached": cached, "prompt_read": read, "prompt_ms": pms,
            "prefill_real": real,
            "drafts": f'{last.get("drafts_accepted")}/{last.get("drafts_offered")}',
            "hash": hashlib.sha256(out.encode()).hexdigest()[:10]}


def configure(v):
    c = json.load(open(TUNED))
    args = c["args"]
    for k, val in v.get("args", {}).items():
        if k in args:
            i = args.index(k)
            if val is None:
                del args[i:i + 2]
            elif val == "":
                pass                      # boolean flag already present
            else:
                args[i + 1] = str(val)
        elif val is not None:
            args += [k] if val == "" else [k, str(val)]
    env = c.setdefault("env", {})
    for k, val in v.get("env", {}).items():
        if val is None:
            env.pop(k, None)
        else:
            env[k] = str(val)
    if "parallel" in v:
        c["parallel"] = v["parallel"]
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
    raise RuntimeError("server did not come up")


def main():
    variants = json.load(open(sys.argv[1]))
    reps = int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 2
    base_hash = None
    try:
        for v in variants:
            rec = {"name": v["name"], "variant": v, "time": time.strftime("%H:%M:%S")}
            try:
                configure(v)
                rec["startup_s"] = start()
                nonce0 = f'{v["name"]}-warm-{time.time_ns()}'
                ask("Say OK.", 8, nonce0)
                res = {}
                for name, prompt, mt in WORK:
                    n = reps if name != "long" else 1
                    res[name] = [ask(prompt, mt, f'{v["name"]}-{name}-{i}-{time.time_ns()}') for i in range(n)]
                s = {}
                for name, runs in res.items():
                    decs = [r["dec"] or 0 for r in runs]
                    s[name] = {"dec": round(sum(decs) / len(decs), 2), "min": min(decs), "max": max(decs),
                               "hash": sorted({r["hash"] for r in runs}), "drafts": runs[0]["drafts"],
                               "prefill_real": runs[0].get("prefill_real")}
                hashes = {k: s[k]["hash"] for k in s}
                if base_hash is None:
                    base_hash = hashes
                s["same_as_first"] = hashes == base_hash
                tot = [s[k]["dec"] for k in ("chat", "code", "story")]
                s["mean3"] = round(sum(tot) / 3, 2)
                rec["results"] = res
                rec["summary"] = s
                print(v["name"], json.dumps(s, ensure_ascii=False), flush=True)
            except Exception as e:
                rec["error"] = repr(e)
                print(v["name"], "ERROR", e, flush=True)
                subprocess.run(f"docker logs --tail 40 strata-mi50 > {HOME}/strata-deploy/ab2-fail-{v['name']}.log 2>&1",
                               shell=True)
            with open(OUT, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    finally:
        if "--keep" not in sys.argv:
            configure({})
            start()
            print("restored tuned config", flush=True)


if __name__ == "__main__":
    main()