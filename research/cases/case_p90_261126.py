"""Full-catalogue (pyspi's bundled benchmarked_p90) rerun of the drift and form cases (2026-11-26).

The figures use each case's own three SPIs (`drift_blocks_261029.py`, `form_ladder_261028.py`); this run puts
the same recordings through the whole p90 catalogue on Gadi, to screen every SPI pair for responses to each
mechanism. `build` writes one named-npz archive per case (exactly the recordings `make(LEVELS[c], rep)` gives
the figures) and its corpus YAML.

    python -m research.cases.case_p90_261126 build
    rsync -av results/cases/p90-261126/data/ gadi-dm.nci.org.au:/g/data/ql44/we2614/mts-spi-study/cases/p90-261126/data/
    # on Gadi, per case: bash jobs/gadi/submit_farm.sh research/cases/configs/case-p90-261126-<case>.yaml
    rsync -av gadi-dm.nci.org.au:/g/data/ql44/we2614/mts-spi-study/cases/p90-261126/mpis/ results/cases/p90-261126/mpis/
    python -m research.cases.case_p90_261126 screen

Screen (rules fixed and committed before the full-farm MPIs were loaded, 26 November 2026). Reading rules:
docs/interpreting-spi-pairs.md. Question: which SPI pairs' z responds to a mechanism that their assumption contrast
does not name ("spurious positives" of a one-assumption reading)?
- Units. Levels share the seed within a rep, so contrasts are paired. Each mechanism has two arms in the 2 x 2:
  lag = (lag - no lag, all three - warping), warping = (warping - no lag, all three - lag); monotone = (monotone -
  linear, both - fold), fold = (fold - linear, both - monotone).
- Validity. Keep SPIs finite in every recording of the case (pyspi errors from meta.json are listed). Drop pairs
  whose z is undefined in any recording (a constant edge profile) and pairs with |z| < 1e-9 in every recording
  (antisymmetric x symmetric exact zeros).
- Mover. On a half of the reps: |mean dz| >= .05 over both arms, both arm means of that sign, and |t| >= 4 on the
  pooled per-rep arm differences. Selected on reps 0-9, confirmed when reps 10-19 pass with the same sign. Raw dz
  and its SD multiple are both reported (ceiling pairs have tiny SDs).
- Labels (by estimand, declared before data; `assumptions.classify` covers 225 of 21,321 rc2 pairs, too few).
  SAME_TIME: the estimand is a function of the same-time joint distribution (invariant to a common permutation of
  time). LINEAR: the estimand is fixed by second-order (covariance / cross-spectral) structure. For a timing
  mechanism (lag, warping) a pair is `contrast` when exactly one member is same-time, `spurious` when both are
  (neither differs in timing, so a response comes through strength, saturation or range restriction), and
  `ambiguous` when neither is. For a form mechanism (monotone, fold) the same with LINEAR. Report counts per label;
  unlabelled movers are not called spurious.
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


# Entries ending in "_" are prefixes; others are exact pyspi names.
SAME_TIME = ("cov_", "pcorr_", "spearmanr", "xi_tau-0", "pdist_", "dcorr", "dcorr_biased", "hsic", "anm", "cds", "reci",
             "je_", "ce_", "mi_", "gpfit_DotProduct", "ids", "tailasym_", "edge_excessvar", "edge_topshare_f-0-05")
LINEAR = ("cov_", "pcorr_", "lagcorr_", "xpdist_", "mi_gaussian", "tlmi_gaussian", "gc_gaussian_", "cce_gaussian_",
          "xme_gaussian_", "di_gaussian_", "si_gaussian_", "phase_multitaper_", "cohmag_", "icoh_", "psi_", "dcoh_",
          "sgc_", "cgc_", "cpdc_", "cgpdc_", "cddtf_", "pcohmag_", "liang_", "liang-cond_", "phasealigned_", "coint_",
          "wtc_", "phi_", "gpfit_DotProduct")
ARMS = {"drift-blocks": {"lag": (("lag", "no lag"), ("all three", "warping")),
                         "warping": (("warping", "no lag"), ("all three", "lag"))},
        "form-ladder": {"monotone": (("monotone", "linear"), ("both", "fold")),
                        "fold": (("fold", "linear"), ("both", "monotone"))}}
KIND = {"lag": SAME_TIME, "warping": SAME_TIME, "monotone": LINEAR, "fold": LINEAR}
DZ, T_MIN, SPLIT = .05, 4.0, 10


def has(name, entries):
    return name in entries or name.startswith(tuple(e for e in entries if e.endswith("_")))


def label(a, b, prefixes):
    n = has(a, prefixes) + has(b, prefixes)
    return ("ambiguous", "contrast", "spurious")[n]


def load(name):
    """Per recording: (level, rep, {spi: M x M}) and the pyspi errors per recording."""
    rows, errors = [], {}
    for d in sorted((LOCAL / "mpis" / f"case-p90-261126-{name}").iterdir()):
        meta = json.loads((d / "meta.json").read_text())
        _, level, rep = meta["labels"]
        if meta["pyspi"]["errors"]:
            errors[d.name] = meta["pyspi"]["errors"]
        with np.load(d / "spi_mpis.npz", allow_pickle=False) as z:
            rows.append((level, int(rep[1:]), {k: np.asarray(z[k], float) for k in z.files if not k.startswith("__")}))
    return rows, errors


def zmatrix(mats, spis):
    off = ~np.eye(len(next(iter(mats.values()))), dtype=bool)
    v = np.stack([mats[s][off] for s in spis])
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.corrcoef(v)


def mover(d):
    """d: (reps, 2 arms, pairs) paired dz. Returns mean dz, t and the pass mask."""
    arm = d.mean(0)
    flat = d.reshape(-1, d.shape[-1])
    m, s = flat.mean(0), flat.std(0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = m / (s / np.sqrt(len(flat)))
    ok = (np.abs(m) >= DZ) & (np.sign(arm[0]) == np.sign(m)) & (np.sign(arm[1]) == np.sign(m)) & (np.abs(t) >= T_MIN)
    return m, t, ok


def screen():
    import pandas as pd
    out = LOCAL / "screen"
    out.mkdir(parents=True, exist_ok=True)
    report = {"rules": dict(dz=DZ, t=T_MIN, select="reps 0-9", confirm="reps 10-19"), "cases": {}}
    for name, mechanisms in ARMS.items():
        rows, errors = load(name)
        spis = sorted(set.intersection(*(set(m) for _, _, m in rows)))
        finite = [s for s in spis if all(np.isfinite(m[s]).all() for _, _, m in rows)]
        Z = {(lv, rep): zmatrix(m, finite) for lv, rep, m in rows}
        iu = np.triu_indices(len(finite), 1)
        stack = np.stack([z[iu] for z in Z.values()])
        undefined = ~np.isfinite(stack).all(0)
        zero = (np.abs(np.nan_to_num(stack)) < 1e-9).all(0)
        keep = ~undefined & ~zero
        a, b = np.array(finite)[iu[0]][keep], np.array(finite)[iu[1]][keep]
        reps = sorted({rep for _, rep in Z})
        frame = pd.DataFrame({"a": a, "b": b})
        case = dict(recordings=len(rows), spis=len(spis), finite=len(finite),
                    dropped_spis=sorted(set(spis) - set(finite)), errors=errors,
                    pairs=int(len(keep)), undefined=int(undefined.sum()), exact_zero=int((zero & ~undefined).sum()),
                    analysed=int(keep.sum()), mechanisms={})
        for mech, arms in mechanisms.items():
            d = np.stack([[Z[(hi, r)][iu][keep] - Z[(lo, r)][iu][keep] for hi, lo in arms] for r in reps])
            ms, ts, sel = mover(d[:SPLIT])
            mc, tc, con = mover(d[SPLIT:])
            conf = sel & con & (np.sign(ms) == np.sign(mc))
            lab = np.array([label(x, y, KIND[mech]) for x, y in zip(a, b)])
            frame[f"{mech}_label"] = lab
            frame[f"{mech}_dz_sel"], frame[f"{mech}_t_sel"] = ms, ts
            frame[f"{mech}_dz_conf"], frame[f"{mech}_t_conf"] = mc, tc
            frame[f"{mech}_mover"] = conf
            counts = {k: dict(pairs=int((lab == k).sum()), selected=int((sel & (lab == k)).sum()),
                              confirmed=int((conf & (lab == k)).sum())) for k in ("contrast", "spurious", "ambiguous")}
            top = frame[conf & (lab == "spurious")].assign(adz=lambda f: f[f"{mech}_dz_conf"].abs()) \
                .nlargest(15, "adz")[["a", "b", f"{mech}_dz_sel", f"{mech}_dz_conf", f"{mech}_t_conf"]]
            case["mechanisms"][mech] = dict(arms=[list(x) for x in arms], counts=counts,
                                            top_spurious=top.round(4).to_dict("records"))
        frame.to_csv(out / f"{name}_pairs.csv.gz", index=False, float_format="%.5g")
        report["cases"][name] = case
        if name == "form-ladder":                                 # plumbing: the figure triplet, every recording
            ref = pd.read_csv(ROOT / "results/cases/form-ladder-261126/z_by_recording.csv")
            keys = {"r": "cov_EmpiricalCovariance", "rho": "spearmanr", "mi": "mi_kraskov_NN-4"}
            diffs = {}
            for x, y in (("r", "rho"), ("r", "mi"), ("rho", "mi")):
                i, j = finite.index(keys[x]), finite.index(keys[y])
                got = np.array([Z[(lv, rep)][i, j] for lv, rep in zip(ref.level, ref.rep)])
                diffs[f"z_{x}_{y}"] = float(np.abs(got - ref[f"z_{x}_{y}"].to_numpy()).max())
            report["plumbing_form_max_abs_diff"] = diffs
    report["pyspi"] = json.loads(next((LOCAL / "mpis").glob("*/*/meta.json")).read_text())["pyspi"]["version"]
    (out / "summary.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({c: {m: v["counts"] for m, v in r["mechanisms"].items()} for c, r in report["cases"].items()},
                     indent=1), report.get("plumbing_form_max_abs_diff"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", choices=("build", "screen"))
    {"build": build, "screen": screen}[ap.parse_args().action]()
