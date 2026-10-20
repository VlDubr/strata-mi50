#!/usr/bin/env python3
"""Long-context before/after: for each variant of VARIANTS.json, restart the engine (ab.py's configure/start), read a
prompt of N characters and decode 300 tokens; prints the server's metrics.  usage: longab.py VARIANTS.json N"""
import json, subprocess, sys
sys.argv, args = [sys.argv[0]], sys.argv[1:]
import ab
variants = json.load(open(args[0])); n = args[1]
try:
    for v in variants:
        ab.configure(v)
        s = ab.start(v.get("engine"))
        ab.ask("Привет!", 16)
        out = subprocess.run(["python3", "longctx.py", n, "300"], capture_output=True, text=True).stdout.strip()
        print(v["name"], "startup", s, "s |", out, flush=True)
finally:
    ab.configure({})
    ab.start()
