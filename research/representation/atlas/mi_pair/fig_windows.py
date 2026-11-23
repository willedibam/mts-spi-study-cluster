"""Copy the few raw atlas windows that fig:mi draws out of the NeuroTycho banks (on Gadi; the banks are not local).

    python fig_windows.py <banks dir> <out.npz> <window id> [<window id> ...]

Each id is looked up in neurotycho-long-stage-a then -b. Stores, per id, the window (T x M, float32, as banked) under
`<id>` and its kept electrode numbers under `<id>|chan`; the plot selects the 32 zns channels by electrode number.
"""
import json, sys
import numpy as np

banks, out, ids = sys.argv[1], sys.argv[2], sys.argv[3:]
res = {}
for stage in ("a", "b"):
    d = f"{banks}/neurotycho-long-stage-{stage}"
    man = json.load(open(f"{d}/manifest.json")); rows = {r["id"]: r for r in man["rows"]}
    kept = {u["unit"]: u["kept_channels"] for u in man["preprocessing"]["units"]}
    with np.load(f"{d}/windows.npz") as W:
        for i in ids:
            if i in rows and i not in res:
                r = rows[i]; res[i] = W[i].astype(np.float32); res[i + "|chan"] = np.array(kept[r["unit"]][:r["M"]])
missing = [i for i in ids if i not in res]
if missing: sys.exit(f"not in either bank: {missing}")
np.savez_compressed(out, **res)
