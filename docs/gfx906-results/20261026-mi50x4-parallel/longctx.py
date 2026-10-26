#!/usr/bin/env python3
"""Long-context decode: a prompt of about N characters of the repository's docs/sources, then 300 greedy tokens.
usage: longctx.py <chars> [max_tokens]"""
import glob, json, os, sys, time, urllib.request
N = int(sys.argv[1]); MT = int(sys.argv[2]) if len(sys.argv) > 2 else 300
HOME = os.path.expanduser("~")
KEY = open(f"{HOME}/strata-deploy/api_key").read().strip()
HDR = {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}
text = ""
for pat in ("docs/*.md", "src/core/*.cpp", "src/program/*.cpp", "src/kernels/cuda/*.cu"):
    for p in sorted(glob.glob(f"{HOME}/strata-gfx906/{pat}")):
        text += open(p, encoding="utf-8", errors="replace").read() + "\n"
        if len(text) >= N: break
    if len(text) >= N: break
text = text[:N]
body = {"model": "x", "max_tokens": MT, "temperature": 0,
        "messages": [{"role": "user", "content": text + "\n\nList the five most important components described above, one line each."}]}
t0 = time.time()
with urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8095/v1/chat/completions", json.dumps(body).encode(), HDR), timeout=7200) as r:
    m = json.loads(r.read())
with urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8095/metrics", headers=HDR), timeout=30) as r:
    mt = json.loads(r.read())
last = max(mt.get("requests", []), key=lambda x: x.get("time", 0))
print(json.dumps({k: last.get(k) for k in ("prompt_tokens", "prompt_ms", "output_tokens", "decode_tok_s", "drafts_accepted", "drafts_offered")}),
      f"wall {time.time() - t0:.0f}s")
