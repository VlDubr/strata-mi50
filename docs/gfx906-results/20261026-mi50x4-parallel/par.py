#!/usr/bin/env python3
"""Concurrent-request load test for the Strata server (OpenAI streaming API).

    par.py N [max_tokens] [rounds]

Sends N different prompts at once (greedy, streaming) and prints, per round: the aggregate output rate (all output
tokens / wall time from the first send to the last token), each request's time to first token and its own decode
rate (tokens after the first / time from the first to the last token), and the server's /v1/status concurrency.
"""
import json, os, sys, threading, time, urllib.request

HOME = os.path.expanduser("~")
URL = "http://127.0.0.1:8095"
KEY = open(f"{HOME}/strata-deploy/api_key").read().strip()
HDR = {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}

TOPICS = ["TCP и UDP", "квантовые компьютеры", "фотосинтез", "историю Рима", "работу GPU", "блокчейн", "вулканы",
          "иммунную систему", "теорию относительности", "нейронные сети", "климат Арктики", "устройство ДВС",
          "биржевые индексы", "архитектуру ARM", "чёрные дыры", "сжатие данных"]


def one(i, mt, out):
    body = {"model": "x", "max_tokens": mt, "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": f"Подробно объясни {TOPICS[i % len(TOPICS)]} (запрос {i})."}]}
    t0 = time.time()
    first = last = None
    n = 0
    usage = None
    try:
        req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), HDR)
        with urllib.request.urlopen(req, timeout=3600) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                try:
                    d = json.loads(line[5:])
                except ValueError:
                    continue
                if d.get("usage"):
                    usage = d["usage"].get("completion_tokens")
                ch = (d.get("choices") or [{}])[0].get("delta") or {}
                if ch.get("content") or ch.get("reasoning_content"):
                    now = time.time()
                    first = first or now
                    last = now
                    n += 1
        n = usage or n
        out[i] = {"ttft": round(first - t0, 2) if first else None, "chunks": n,
                  "dec": round((n - 1) / (last - first), 1) if first and last and last > first else None,
                  "t0": t0, "last": last}
    except Exception as e:  # noqa: BLE001
        out[i] = {"error": repr(e)}


def status():
    try:
        with urllib.request.urlopen(urllib.request.Request(URL + "/v1/status", headers=HDR), timeout=10) as r:
            s = json.loads(r.read())
        return s.get("concurrency")
    except Exception as e:  # noqa: BLE001
        return repr(e)


def main():
    n = int(sys.argv[1]); mt = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    for rd in range(rounds):
        out = {}
        th = [threading.Thread(target=one, args=(i, mt, out)) for i in range(n)]
        t0 = time.time()
        for t in th:
            t.start()
        for t in th:
            t.join()
        ok = [v for v in out.values() if "error" not in v and v.get("last")]
        tot = sum(v["chunks"] for v in ok)
        wall = max(v["last"] for v in ok) - t0 if ok else 0
        decs = sorted(v["dec"] for v in ok if v["dec"])
        ttfts = sorted(v["ttft"] for v in ok if v["ttft"] is not None)
        print(json.dumps({"n": n, "round": rd, "ok": len(ok), "errors": [v["error"] for v in out.values() if "error" in v][:3],
                          "aggregate_tok_s": round(tot / wall, 1) if wall else None, "wall_s": round(wall, 1),
                          "per_req_dec_min_med_max": [decs[0], decs[len(decs) // 2], decs[-1]] if decs else None,
                          "ttft_min_med_max": [ttfts[0], ttfts[len(ttfts) // 2], ttfts[-1]] if ttfts else None,
                          "concurrency": status()}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
