"""Fixed export and confirmation audit; never train on confirmation records."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.prepare_cml2d_corpus import prepare
from scripts.analyze_cml2d_spi import run, corr
from src.order_parameter_analysis import clustered_bootstrap_spearman

OLD_CONTROLS = [3.84, 3.85, 3.858, 3.86212, 3.864, 3.866, 3.87, 3.875, 3.89]
FROZEN_HASHES = {
    "model.npz": "fddda6abf3b1ddbd30f9aff84158dee5436716f1eb1c01704516f753b22995b2",
    "geometry.json": "bc1777579703c232a060f536aa3969791fdb1f07ffcde1a77a9a1e3251c02357",
    "summary.json": "78e1c9ebf83f45c2007839a562b76d2335630d22333a87583f680ac7484ad9d8",
}


def verify_frozen(frozen):
    for name, expected in FROZEN_HASHES.items():
        assert hashlib.sha256((frozen / name).read_bytes()).hexdigest() == expected, name


def confirmation_gate(frame):
    counts = frame.groupby(["view", "M", "T", "r"]).eligible.agg(["sum", "size"])
    assert (counts["size"] == 32).all(), "incomplete planned cells"
    excluded = float((~frame.eligible).mean())
    return dict(passes=bool(excluded <= .1 and (counts["sum"] >= 24).all()),
        excluded_fraction=excluded, minimum_cell_retained=int(counts["sum"].min()),
        minimum_required_per_cell=24, planned_per_cell=32)


def export(physics, output, frozen):
    verify_frozen(frozen)
    assert len(list(physics.glob("case-*.npz"))) == 544
    for arm, ms, ts, controls, excluded in [
        ("primary", [32], [1000], None, ()),
        ("sensitivity", [16, 32], [500, 1000], OLD_CONTROLS, ((32, 1000),)),
    ]:
        prepare(physics, output / arm, output / f"{arm}-corpus.yaml", ms, ts,
            ["dispersed"], [], select_L=256, select_r=controls, exclude_shapes=excluded)
    expected = {"primary": 544, "sensitivity": 864}
    for arm, count in expected.items():
        rows = json.loads((output / arm / "manifest.json").read_text())["rows"]
        assert len(rows) == count and all(row["role"] == "evaluation" for row in rows)
        assert sorted({row["seed"] for row in rows}) == list(range(260911101, 260911133))
    (output / "frozen-input-identity.json").write_text(json.dumps(FROZEN_HASHES, indent=2)+"\n")


def report(root, arm, frozen):
    verify_frozen(frozen)
    output = root / f"{arm}-analysis"
    corpus = root / arm
    run(corpus, corpus / "mpi" / arm, output, frozen=frozen, minimum_retained_per_cell=24)
    frame = pd.read_csv(output / "scores.csv")
    assert set(frame.role) == {"evaluation"}
    with np.load(frozen / "model.npz", allow_pickle=False) as original, np.load(output / "model.npz", allow_pickle=False) as copied:
        for key in original.files:
            assert np.array_equal(original[key], copied[key]), key
    gate = confirmation_gate(frame)
    # Seal the stricter prospectively specified coverage gate before endpoint reporting.
    (output / "confirmation-gate.json").write_text(json.dumps(gate, indent=2)+"\n")
    eligible = frame[frame.eligible].copy()
    sign = json.loads((frozen / "summary.json").read_text())["display_sign"]
    endpoints = []
    for (M, T), group in eligible.groupby(["M", "T"]):
        subsets = {"all": group}
        if arm == "primary":
            subsets["interleaved_only"] = group[~group.r.isin(OLD_CONTROLS)]
            subsets["original_grid"] = group[group.r.isin(OLD_CONTROLS)]
        for subset, part in subsets.items():
            boot, within = clustered_bootstrap_spearman(sign*part.q, part.Q_reference,
                part.r, part.seed, n_resamples=2000, seed=260911)
            means = part.groupby("r")[["q", "Q_reference"]].mean()
            endpoints.append(dict(M=int(M), T=int(T), subset=subset, rows=len(part),
                seed_clusters=int(part.seed.nunique()), rho=corr(sign*part.q, part.Q_reference),
                rho_ci=np.nanquantile(boot, [.025, .975]).tolist(),
                control_mean_rho=corr(sign*means.q, means.Q_reference),
                within_r_ci=np.nanquantile(within, [.025, .975]).tolist()))
    manifest = json.loads((corpus / "manifest.json").read_text())
    physics = []
    for source in manifest["source_archives"]:
        meta = source["metadata"]
        physics.append(dict(r=meta["r"], seed=meta["seed"], Q=meta["Q"],
            half_difference=abs(meta["Q_first_half"]-meta["Q_second_half"]),
            block_mean_se=float(np.std(meta["Q_blocks"], ddof=1)/np.sqrt(8))))
    pd.DataFrame(physics).to_csv(output / "physics-reference-audit.csv", index=False)
    result = dict(status="independent frozen confirmation" if gate["passes"] else "confirmation gate failed; descriptive only",
        gate=gate, frozen_hashes=FROZEN_HASHES, endpoints=endpoints,
        generic_summary_note="The reused generic scorer labels all outputs exploratory; this separate prospective contract determines confirmation status.")
    (output / "confirmation-report.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["export", "primary", "sensitivity"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--physics", type=Path)
    parser.add_argument("--frozen", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "export":
        export(args.physics, args.root, args.frozen)
    else:
        report(args.root, args.stage, args.frozen)
