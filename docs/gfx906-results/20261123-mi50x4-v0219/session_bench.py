#!/usr/bin/env python3
"""Fixed greedy workload against the running server; prints per-request decode/prefill + hashes.
Run on the server host: python3 baseline_bench.py <tag> [reps]
Appends one JSON line to ~/strata-deploy/session-baseline.jsonl. No container restarts."""
import glob, hashlib, json, os, sys, time, urllib.request

HOME = os.path.expanduser("~")
OUT = f"{HOME}/strata-deploy/session-baseline.jsonl"
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

def ask(prompt, max_tokens):
    t_send = time.time()
    # session-unique prefix defeats conversation-cache prefix reuse: every run measures a cold prompt
    nonce = f"[bench {tag_global} {time.time_ns()}]\n"
    body = {"model": "x", "max_tokens": max_tokens, "temperature": 0,
            "messages": [{"role": "user", "content": nonce + prompt}]}
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), HDR)
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        m = json.loads(r.read())["choices"][0]["message"]
    wall = time.time() - t0
    out = (m.get("reasoning_content") or "") + "\n---\n" + (m.get("content") or "")
    # only rows recorded after this request was sent belong to it; pick the newest of those
    rows = [x for x in metric_rows() if (x.get("time") or 0) >= t_send - 2]
    last = max(rows, key=lambda x: x.get("time", 0)) if rows else {}
    pt = last.get("prompt_tokens") or 0
    pms = last.get("prompt_ms") or 0
    read = last.get("prompt_read")
    cached = last.get("reused") or 0
    newtok = read if read is not None else pt - cached
    real = round(newtok / (pms / 1000), 1) if pms and newtok else None
    eff = round(pt / (pms / 1000), 1) if pms and pt else None
    return {"dec": last.get("decode_tok_s"), "out": last.get("output_tokens"),
            "prompt": pt, "cached": cached, "prompt_read": read, "prompt_ms": pms,
            "prefill_real": real, "prefill_effective": eff,
            "drafts": f'{last.get("drafts_accepted")}/{last.get("drafts_offered")}',
            "wall_s": round(wall, 1), "hash": hashlib.sha256(out.encode()).hexdigest()[:10]}

tag_global = "run"

def main():
    global tag_global
    tag = sys.argv[1] if len(sys.argv) > 1 else "run"
    tag_global = tag
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    res = {}
    # warm-up (not recorded)
    ask("Say OK.", 8)
    for name, prompt, mt in WORK:
        rows = [ask(prompt, mt) for _ in range(reps)]
        res[name] = rows
        print(name, json.dumps(rows, ensure_ascii=False))
    short = [r["dec"] for n in ("chat", "code", "story") for r in res[n][1:]]
    mean3 = round(sum(short) / len(short), 2)
    rec = {"tag": tag, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "results": res, "mean3_warm": mean3}
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("mean3_warm", mean3)

if __name__ == "__main__":
    main()