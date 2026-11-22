#!/usr/bin/env python3
"""Aggregate decode under N concurrent clients (greedy, streaming, 300 tokens each).
Run on server: python3 par_bench.py N [tag]. Prints total tok/s, per-request rates, TTFT."""
import hashlib, json, os, sys, threading, time, urllib.request

HOME = os.path.expanduser("~")
URL = "http://127.0.0.1:8095"
KEY = open(f"{HOME}/strata-deploy/api_key").read().strip()
HDR = {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}

PROMPTS = [
    "Объясни, как работает конвейеризация в процессоре, и почему она ускоряет выполнение команд.",
    "Write a short Python script that reads a CSV and prints the top 5 rows by a numeric column.",
    "Расскажи историю о маяке и роботе, который его обслуживает, примерно 250 слов.",
    "Explain the difference between TCP and UDP with two example applications each.",
    "Опиши, как устроен кэш процессора и зачем нужны уровни L1, L2, L3.",
    "Write a Python function that merges two sorted lists into one sorted list, with tests.",
    "Что такое виртуальная память и как страницы отображаются в физические адреса?",
    "Explain how a hash table resolves collisions, with chaining and open addressing.",
    "Напиши рассказ о коте, который научился открывать двери, примерно 250 слов.",
    "Describe how garbage collection works in Python, reference counting and cycles.",
    "Как работает протокол TLS и зачем нужны сертификаты?",
    "Write a Python decorator that retries a function up to 3 times on exception.",
]


def one(i, results):
    nonce = f"[par {tag} {i} {time.time_ns()}]\n"
    body = {"model": "x", "max_tokens": 300, "temperature": 0, "stream": True,
            "messages": [{"role": "user", "content": nonce + PROMPTS[i % len(PROMPTS)]}]}
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), HDR)
    t0 = time.time()
    first = None
    ntok = 0
    text = []
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                d = json.loads(data)
            except Exception:
                continue
            delta = d.get("choices", [{}])[0].get("delta", {})
            c = delta.get("content") or delta.get("reasoning_content") or ""
            if c:
                if first is None:
                    first = time.time() - t0
                ntok += 1
                text.append(c)
    wall = time.time() - t0
    results[i] = {"ttft": round(first or wall, 2), "tok": ntok, "wall": round(wall, 2),
                  "per": round(ntok / (wall - (first or 0)), 1) if wall > (first or 0) else 0,
                  "hash": hashlib.sha256("".join(text).encode()).hexdigest()[:10]}


def main():
    global tag
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    tag = sys.argv[2] if len(sys.argv) > 2 else "run"
    results = {}
    threads = [threading.Thread(target=one, args=(i, results)) for i in range(n)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0
    total_tok = sum(r["tok"] for r in results.values())
    agg = round(total_tok / wall, 1)
    print(json.dumps({"tag": tag, "n": n, "wall_s": round(wall, 1), "total_tok": total_tok,
                      "aggregate_tok_s": agg,
                      "per_req": [results[i]["per"] for i in sorted(results)],
                      "ttft": [results[i]["ttft"] for i in sorted(results)],
                      "hashes": [results[i]["hash"] for i in sorted(results)]}, ensure_ascii=False))
    with open(f"{HOME}/strata-deploy/par-results.jsonl", "a") as f:
        f.write(json.dumps({"tag": tag, "n": n, "agg": agg, "results": results}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()