import json, os, shutil
H = os.path.expanduser("~/strata-data/config")
shutil.copy2(f"{H}/tuned.json", f"{H}/tuned-solo.json")
c = json.load(open(f"{H}/tuned.json"))
a = c["args"]
def setarg(k, v=None):
    if k in a:
        i = a.index(k)
        if v is not None:
            a[i + 1] = v
    else:
        a.extend([k] if v is None else [k, v])
setarg("--max-context", "262144")
setarg("--resident-budget-gib", "24")
setarg("--kv-resident", "32768")
setarg("--trim-stage-weights")
c["parallel"] = 12
c["layer_split"] = "12,24,36"
json.dump(c, open(f"{H}/tuned.json", "w"), indent=1)
shutil.copy2(f"{H}/tuned.json", f"{H}/strata-unsloth-ud-q5_k_xl.json")
print(" ".join(a)); print("parallel", c["parallel"], "layer_split", c["layer_split"])
