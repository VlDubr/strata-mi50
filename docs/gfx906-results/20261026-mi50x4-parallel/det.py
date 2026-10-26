#!/usr/bin/env python3
"""Greedy determinism check: fixed prompts at temperature 0, prints a hash of each full output (reasoning + answer)."""
import glob, hashlib, json, os, sys, urllib.request

URL = "http://127.0.0.1:8095/v1/chat/completions"
KEY = open(os.path.expanduser("~/strata-deploy/api_key")).read().strip()
DOC = "".join(open(p, encoding="utf-8", errors="replace").read()
              for p in sorted(glob.glob(os.path.expanduser("~/strata-gfx906/docs/*.md")))[:3])[:60_000]
PROMPTS = [
    ("math", "Сколько будет 17*23? Ответь одним числом.", 200),
    ("code", "Write a Python function that merges two sorted lists into one sorted list.", 300),
    ("ru", "Объясни, что такое конвейерный параллелизм в нейросетях, в трёх предложениях.", 300),
    ("long", DOC + "\n\nName the three main topics of these documents.", 200),
]
res = {}
for name, prompt, mt in PROMPTS:
    body = {"model": "x", "max_tokens": mt, "temperature": 0, "messages": [{"role": "user", "content": prompt}]}
    req = urllib.request.Request(URL, json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        m = json.loads(r.read())["choices"][0]["message"]
    out = (m.get("reasoning_content") or "") + "\n---\n" + (m.get("content") or "")
    res[name] = hashlib.sha1(out.encode()).hexdigest()[:12]
    print(name, res[name], repr((m.get("content") or "")[:60]), flush=True)
print(json.dumps(res))
