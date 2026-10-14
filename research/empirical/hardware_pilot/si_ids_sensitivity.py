"""Focused bandwidth/window sensitivity of the hardware SI × IDS coordinate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from mtsspi.provenance import array_sha256, atomic_json, atomic_savez, file_sha256
from .calibrate_hardware_pilot import I, J, edge_corr, phase_properties

WIDTHS = (.25, .375, .5, .75, 1.)
RECORDS = (("ER", 23), ("ER", 24), ("SW", 31), ("SW", 34))
WINDOWS = {"full": (0, 5000), "first": (0, 2500), "second": (2500, 5000)}
CONFIG = Path(__file__).parent / "configs/si-ids-bandwidth.yaml"
SI = "si_kernel_W-0.5_k-1"


def verify_pyspi(bank):
    import pyspi
    source = Path(pyspi.__file__).resolve().parent.parent
    expected = json.loads((bank / "pyspi-source-verification.json").read_text())
    if pyspi.COMPUTATION_VERSION != "3.0.0.r9":
        raise ValueError("This follow-up requires the original r9 computation")
    for name, digest in expected["files"].items():
        if file_sha256(source / name) != digest:
            raise ValueError(f"Frozen pyspi source mismatch: {name}")
    return dict(source=str(source), revision=expected["revision"], files=expected["files"])


def prepare(bank, out):
    """Slice audited inputs only; retain full-record scaling and sampling rate."""
    from pyspi.data import Data
    from pyspi.statistics.misc import InterDependenceScore
    if (out / "manifest.json").exists():
        raise FileExistsError("Prepared follow-up exists; inspect rather than overwrite")
    provenance = verify_pyspi(bank)
    parent = json.loads((bank / "manifest.json").read_text())
    if file_sha256(bank / "observations.npz") != parent["archive_sha256"]:
        raise ValueError("Original input archive changed")
    out.mkdir(parents=True, exist_ok=True)
    rows, arrays, extras = [], {"full": {}, "halves": {}}, {}
    ids = InterDependenceScore(terms=6, pnorm="max", bandwidth=.5)
    with np.load(bank / "observations.npz") as data:
        for topology, step in RECORDS:
            row, = [r for r in parent["rows"] if (r["topology"], r["step"]) == (topology, step)]
            x = data[row["name"]]
            if x.shape != (28, 5000) or array_sha256(x) != row["member_sha256"]:
                raise ValueError("Original input member changed")
            for window, (start, stop) in WINDOWS.items():
                name = f"{topology.lower()}-{step:03d}-{window}"
                y = x[:, start:stop].copy()
                cohort = "full" if window == "full" else "halves"
                arrays[cohort][name] = y
                # A cheap secondary control: IDS with each half restandardised.
                # SI internally rescales its kernel by the window's own SD.
                standardised = (y - y.mean(axis=1, keepdims=True)) / y.std(axis=1, keepdims=True)
                extras[name] = ids.multivariate(Data(standardised, zscore=False))
                physical = phase_properties(y, (90, 140))
                rows.append(dict(name=name, parent_name=row["name"], topology=topology, step=step,
                                 window=window, cohort=cohort, start=start, stop=stop, T=stop-start,
                                 member_sha256=array_sha256(y), frequency_range=float(np.ptp(physical["frequency"])),
                                 mean_plv=float(physical["plv"][I, J].mean()),
                                 channel_sd_min=float(y.std(axis=1).min()), channel_sd_max=float(y.std(axis=1).max())))
    for cohort, members in arrays.items():
        archive = out / f"{cohort}.npz"
        names = list(members)
        members.update(__dataset_names__=np.array(names), __axis_order__=np.array(["process", "observation"]),
                       __labels_json__=np.array([json.dumps([r["topology"], r["window"]]) for r in rows if r["cohort"] == cohort]),
                       __shapes__=np.array([members[n].shape for n in names]))
        atomic_savez(archive, members)
        config = dict(name=f"hardware-si-ids-{cohort}", source=dict(format="named-npz-v1", archive=str(archive.resolve()),
                      sha256=file_sha256(archive), axis_order=["process", "observation"]),
                      base_output_dir=str((out / "mpis").resolve()), pyspi_config=str(CONFIG.resolve()),
                      normalise=False, random_seed=261012)
        (out / f"{cohort}.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    atomic_savez(out / "ids-restandardised.npz", extras)
    atomic_json(out / "manifest.json", dict(rows=rows, widths=WIDTHS, fs=500, M=28, all_dyads=True,
                parent_archive_sha256=parent["archive_sha256"], pyspi=provenance,
                config_sha256=file_sha256(CONFIG), runner_sha256=file_sha256(Path(__file__)),
                ids_control_sha256=file_sha256(out / "ids-restandardised.npz"),
                scale="Inherited full-record zscores; SI kernel uses each entropy coordinate's own sample SD",
                ids="terms=6, pnorm=max, bandwidth=0.5; refit for every window",
                secondary_control="IDS recomputed after per-window population zscore; same SI"))


def _partition(a, b, group):
    """Law of total covariance, normalised by global profile standard deviations."""
    scale = a.std() * b.std()
    within = [np.mean(g) * np.mean((a[g] - a[g].mean()) * (b[g] - b[g].mean())) / scale
              for g in (group, ~group)]
    between = sum(np.mean(g) * (a[g].mean() - a.mean()) * (b[g].mean() - b.mean()) / scale
                  for g in (group, ~group))
    return dict(within_coherent=float(within[0]), within_other=float(within[1]), between=float(between))


def analyse(bank, out):
    manifest = json.loads((out / "manifest.json").read_text())
    if manifest["widths"] != list(WIDTHS) or manifest["config_sha256"] != file_sha256(CONFIG):
        raise ValueError("Sensitivity design changed")
    if file_sha256(out / "ids-restandardised.npz") != manifest["ids_control_sha256"]:
        raise ValueError("Secondary IDS control changed")
    original, groups, paths = {}, {}, {}
    with np.load(bank / "observations.npz") as data:
        for row in manifest["rows"]:
            if row["window"] != "full":
                continue
            path, = (bank / "mpis").glob(f"*/????-{row['parent_name']}/spi_mpis.npz")
            with np.load(path) as matrices:
                original[row["parent_name"]] = {key: matrices[key] for key in (SI, "ids")}
            if row["step"] in (23, 31):
                groups[row["topology"]] = phase_properties(data[row["parent_name"]], (90, 140))["plv"][I, J] > .95
    profiles, records, parity = {}, [], []
    with np.load(out / "ids-restandardised.npz") as controls:
        for row in manifest["rows"]:
            path, = (out / "mpis").glob(f"*/????-{row['name']}/spi_mpis.npz")
            meta = json.loads(path.with_name("meta.json").read_text())
            if (meta["status"] != "complete" or meta["pyspi"]["version"]["computation"] != "3.0.0.r9"
                    or meta["pyspi"]["config_sha256"] != manifest["config_sha256"]
                    or meta["M"] != 28 or meta["T"] != row["T"] or meta["pyspi"]["errors"]
                    or meta["normalise"] or meta["source"]["member_sha256"] != row["member_sha256"]):
                raise ValueError(f"Invalid selected-SPI receipt: {row['name']}")
            with np.load(path) as matrices:
                expected = {"ids", *(f"si_kernel_W-{w}_k-1" for w in WIDTHS)}
                if set(matrices.files) != expected:
                    raise ValueError("Unexpected selected-SPI catalogue")
                for key in expected:
                    a = matrices[key]
                    if a.shape != (28, 28) or not np.isfinite(a[I, J]).all() or not np.allclose(a, a.T, equal_nan=True):
                        raise ValueError(f"Incomplete/asymmetric profile: {row['name']}/{key}")
                b = matrices["ids"][I, J]
                reference = matrices[SI][I, J]
                if row["window"] == "full":
                    for key in (SI, "ids"):
                        error = float(np.max(abs(matrices[key][I, J] - original[row["parent_name"]][key][I, J])))
                        if error > 1e-10:
                            raise ValueError("Full-record replay disagrees with archived r9")
                        parity.append(dict(name=row["name"], spi=key, max_error=error))
                for width in WIDTHS:
                    a = matrices[f"si_kernel_W-{width}_k-1"][I, J]
                    group = groups[row["topology"]]
                    partition = _partition(a, b, group)
                    z = edge_corr(a, b)
                    if not np.isclose(sum(partition.values()), z, atol=1e-12):
                        raise ValueError("Covariance partition failed")
                    deletion = [edge_corr(a[(I != k) & (J != k)], b[(I != k) & (J != k)]) for k in range(28)]
                    records.append(dict(**row, width=width, z=z, z_restandardised_ids=edge_corr(a, controls[row["name"]][I, J]),
                                        mean_si=float(a.mean()), sd_si=float(a.std()), min_si=float(a.min()), max_si=float(a.max()),
                                        mean_ids=float(b.mean()), sd_ids=float(b.std()),
                                        delete_min=min(deletion), delete_max=max(deletion),
                                        si_profile_agreement_w05=edge_corr(a, reference), z_coherent=edge_corr(a[group], b[group]),
                                        coherent_dyads=int(group.sum()), **partition))
                    profiles[(row["topology"], row["step"], row["window"], width)] = (a.copy(), b.copy())
            paths[str(path.relative_to(out))] = file_sha256(path)
    frame = pd.DataFrame(records)
    comparisons = []
    for topology, before, after in (("ER", 23, 24), ("SW", 31, 34)):
        for width in WIDTHS:
            for window in WINDOWS:
                selected = frame[(frame.topology == topology) & (frame.width == width) & (frame.window == window)].set_index("step")
                pre, post = selected.loc[before], selected.loc[after]
                comparisons.append(dict(topology=topology, width=width, window=window, before=pre.z, after=post.z,
                                        delta=post.z-pre.z, reversal=bool(pre.z > 0 > post.z)))
    halves = []
    for topology, step in RECORDS:
        for width in WIDTHS:
            a, b = profiles[(topology, step, "first", width)], profiles[(topology, step, "second", width)]
            halves.append(dict(topology=topology, step=step, width=width,
                               si_profile_agreement=edge_corr(a[0], b[0]), ids_profile_agreement=edge_corr(a[1], b[1])))
    frame.to_csv(out / "profiles.csv", index=False)
    pd.DataFrame(comparisons).to_csv(out / "contrasts.csv", index=False)
    pd.DataFrame(halves).to_csv(out / "half-profile-agreement.csv", index=False)
    summary = dict(records=4, windows=3, widths=list(WIDTHS), feature_estimates=len(frame),
                   reversals=sum(r["reversal"] for r in comparisons), contrasts=len(comparisons),
                   manifest_sha256=file_sha256(out / "manifest.json"), analysis_source_sha256=file_sha256(Path(__file__)),
                   archive_replay=parity, matrices_sha256=paths,
                   max_ids_restandardisation_z_change=float(abs(frame.z-frame.z_restandardised_ids).max()),
                   uncertainty="Paired duration/estimator sensitivities, not confidence intervals or independent replications")
    atomic_json(out / "summary.json", summary)
    return frame, pd.DataFrame(comparisons), pd.DataFrame(halves)


def plot(frame):
    from matplotlib import pyplot as plt
    from .plot_hardware_pilot import style, box
    style()
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.8), sharey=True)
    for ax, (topology, before, after) in zip(axes, (("ER", 23, 24), ("SW", 31, 34)), strict=True):
        for step, color in ((before, "#0072B2"), (after, "#D55E00")):
            for window, linestyle, marker in (("full", "-", "o"), ("first", "--", "^"), ("second", ":", "v")):
                d = frame[(frame.topology == topology) & (frame.step == step) & (frame.window == window)].sort_values("width")
                ax.plot(d.width, d.z, color=color, ls=linestyle, marker=marker, label=f"{step}: {window}", lw=1)
        ax.axhline(0, color=".6", lw=.6)
        ax.axvline(.5, color=".7", lw=.6)
        ax.set(title=f"{topology}: settings {before} → {after}", xlabel="SI kernel half-width (SD)", ylim=(-1.05, 1.05),
               xticks=[.25, .5, .75, 1.])
        ax.legend(ncol=2, handlelength=1.6, columnspacing=.6, fontsize=6)
        box(ax)
    axes[0].set_ylabel(r"$z_{\mathrm{SI,IDS}}$")
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "analyse"))
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.bank, args.out)
    else:
        frame, _, _ = analyse(args.bank, args.out)
        fig = plot(frame)
        for ext in ("png", "svg", "pdf"):
            fig.savefig(args.out / f"si-ids-bandwidth-halves.{ext}", dpi=180, bbox_inches="tight")
