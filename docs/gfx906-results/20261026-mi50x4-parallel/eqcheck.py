#!/usr/bin/env python3
"""Batch exactness check: N greedy prompts one after another, then the same N at once; compares the output texts.
usage: eqcheck.py [N] [max_tokens]"""
import hashlib, json, os, sys, threading, urllib.request

HOME = os.path.expanduser("~")
URL = "http://127.0.0.1:8095/v1/chat/completions"
KEY = open(f"{HOME}/strata-deploy/api_key").read().strip()
HDR = {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}
PROMPTS = ["Перечисли пять фактов о Луне.", "Write a haiku about GPUs, then explain it.", "Что такое конвейерный параллелизм?",
           "Explain binary search with a Python example.", "Назови столицы пяти стран Африки.", "What is a mixture-of-experts model?",
           "Объясни разницу между RAM и VRAM.", "Give three tips for writing clear documentation.", "Как работает HTTPS?",
           "Summarize the plot of Hamlet in five sentences.", "Что такое квантование нейросетей?", "List four sorting algorithms."]


def ask(p, mt):
    body = {"model": "x", "max_tokens": mt, "temperature": 0, "messages": [{"role": "user", "content": p}]}
    with urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), HDR), timeout=1800) as r:
        m = json.loads(r.read())["choices"][0]["message"]
    return hashlib.sha1(((m.get("reasoning_content") or "") + "\n---\n" + (m.get("content") or "")).encode()).hexdigest()[:10]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    mt = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    ps = PROMPTS[:n]
    solo = [ask(p, mt) for p in ps]
    out = {}
    th = [threading.Thread(target=lambda i=i: out.__setitem__(i, ask(ps[i], mt))) for i in range(n)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    together = [out[i] for i in range(n)]
    same = sum(a == b for a, b in zip(solo, together))
    print(json.dumps({"n": n, "same": same, "solo": solo, "together": together}))


if __name__ == "__main__":
    main()
