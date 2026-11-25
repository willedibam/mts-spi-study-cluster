"""Full-catalogue (pyspi's bundled benchmarked_p90) rerun of the drift and form cases (2026-11-26).

The figures use each case's own three SPIs (`drift_blocks_261029.py`, `form_ladder_261028.py`); this run puts
the same recordings through the whole p90 catalogue on Gadi, to screen every SPI pair for responses to each
mechanism. `build` writes one named-npz archive per case (exactly the recordings `make(LEVELS[c], rep)` gives
the figures) and its corpus YAML.

    python -m research.cases.case_p90_261126 build
    rsync -av results/cases/p90-261126/data/ gadi-dm.nci.org.au:/g/data/ql44/we2614/mts-spi-study/cases/p90-261126/data/
    # on Gadi, per case: bash jobs/gadi/submit_farm.sh research/cases/configs/case-p90-261126-<case>.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import yaml

from research.cases.case_run import ROOT

LOCAL = ROOT / "results/cases/p90-261126"
REMOTE = "/g/data/ql44/we2614/mts-spi-study/cases/p90-261126"
CONFIGS = ROOT / "research/cases/configs"
SEED = 20261126


def cases():
    from research.cases.ed_xed_dtw import drift_blocks_261029 as D
    from research.cases.r_rho_mi import form_ladder_261028 as F
    return {"drift-blocks": D, "form-ladder": F}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build():
    (LOCAL / "data").mkdir(parents=True, exist_ok=True)
    CONFIGS.mkdir(parents=True, exist_ok=True)
    for name, mod in cases().items():
        arrays, labels, shapes = {}, [], []
        for level, kw in mod.LEVELS.items():
            for rep in range(mod.CASE.reps):
                x, _ = mod.make(kw, rep)                         # (T, M), z-scored, as the figures use it
                key = f"{level.replace(' ', '-')}-r{rep:02d}"
                arrays[key] = np.ascontiguousarray(x.T)          # (M, T): axis_order process, observation
                labels.append(json.dumps([name, level, f"r{rep:02d}"]))
                shapes.append(x.T.shape)
        path = LOCAL / "data" / f"{name}.npz"
        np.savez_compressed(path, **arrays, __dataset_names__=np.array(list(arrays)),
                            __labels_json__=np.array(labels), __shapes__=np.array(shapes),
                            __axis_order__=np.array(["process", "observation"]))
        config = dict(name=f"case-p90-261126-{name}",
                      source=dict(format="named-npz-v1", archive=f"{REMOTE}/data/{name}.npz", sha256=sha256(path),
                                  axis_order=["process", "observation"]),
                      base_output_dir=f"{REMOTE}/mpis", pyspi_config="benchmarked_p90", normalise=False,
                      random_seed=SEED)
        out = CONFIGS / f"case-p90-261126-{name}.yaml"
        out.write_text(f"# {__doc__.splitlines()[0]}\n# Recordings: {len(arrays)} x {shapes[0]} (M, T); "
                       f"built by `python -m research.cases.case_p90_261126 build`.\n"
                       + yaml.safe_dump(config, sort_keys=False))
        print(out, len(arrays), shapes[0])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", choices=("build",))
    if ap.parse_args().action == "build":
        build()
