import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
names = sys.argv[2:]
for r in rows:
    if r["name"] not in names or "results" not in r: continue
    out = []
    for k, runs in r["results"].items():
        x = runs[0]
        acc, off = map(int, x["drafts"].split("/"))
        wins = x["out"] - acc
        ms = x["out"] / x["dec"] * 1000 / wins if x["dec"] else 0
        out.append(f"{k}: {x['out'] / wins:.2f} tok/win {ms:.2f} ms/win")
    print(f"{r['name']:18s}", " | ".join(out))
