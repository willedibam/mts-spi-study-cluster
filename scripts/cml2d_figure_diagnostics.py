"""Reproducible, post-hoc CML presentation diagnostics; never fit or alter q."""
from pathlib import Path
from contextlib import nullcontext
import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.corpus_visualization import plot_mts_heatmap, scale_timeseries
from src.spi_spi_contract import build_unified_schema


def export_snapshots(root: Path, output: Path, *, tracking_panel=None, seed=26091115, M=32, full=False):
    """Fixed held seed, controls and prefix; common full-input display scaling."""
    controls = (3.84, 3.86212, 3.89)
    labels = ("Range start", "Boundary neighbourhood", "Range end")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "primary/manifest.json"
    master_only = full and not manifest_path.exists()
    if master_only:
        rows = []
        for path in sorted((root / "primary-masters").glob("*.npz")):
            with np.load(path, allow_pickle=False) as master:
                meta = json.loads(str(master["metadata_json"]))
            if meta["seed"] == seed and meta["r"] in controls:
                rows.append(dict(r=meta["r"], seed=seed, M=M, N=meta["N"], view="full",
                    role="evaluation", row_id=f"raw-master-{path.stem}", master=path.name,
                    master_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        assert len(rows) == 3
        manifest = dict(rows=rows, archive_sha256=None)
    else:
        manifest = json.loads(manifest_path.read_text())
    examples = []
    values = []
    means = []
    lattices = []
    archive_context = nullcontext(None) if master_only else np.load(root / "primary/observations.npz", allow_pickle=False)
    with archive_context as archive:
        for r in controls:
            row = next(row for row in manifest["rows"] if row["seed"] == seed and row["r"] == r)
            assert row["role"] == "evaluation"
            if full:
                assert row["M"] == row["N"] and row["view"] == "full"
            master_path = root / "primary-masters" / row["master"]
            assert hashlib.sha256(master_path.read_bytes()).hexdigest() == row["master_sha256"]
            with np.load(master_path, allow_pickle=False) as master:
                x = master["observed"][:1000, 0, :M].T.copy() if master_only else archive[row["row_id"]]
                assert x.shape == (M, 1000)
                np.testing.assert_array_equal(master["observed"][:1000, 0, :M].T, x)
                if full:
                    np.testing.assert_array_equal(master["sensor_indices"][0], np.arange(M))
                    np.testing.assert_allclose(x.mean(axis=0), master["global_mean"][:1000], atol=1e-14)
                means.append(master["global_mean"][:100].copy())
                lattices.append((master["final_state"].copy(), master["sensor_indices"][0, :M].copy(),
                    len(master["global_mean"])-1))
            values.append(x)
            examples.append(dict(r=r, seed=seed, row_id=row["row_id"], master=row["master"],
                master_sha256=row["master_sha256"], input_sha256=hashlib.sha256(x.tobytes()).hexdigest()))
    limit = max(scale_timeseries(x, "robust")[1] for x in values)
    if full:
        limit = max(float(np.quantile(np.abs(scale_timeseries(x, "robust")[0][x.std(axis=1)>1e-8]), .99))
            if (x.std(axis=1)>1e-8).any() else 1.0 for x in values)
    if tracking_panel is None:
        fig, axes = plt.subplots(3, 3, figsize=(10.2, 6.5), gridspec_kw={"height_ratios": [1, 2, 3]},
            constrained_layout=True)
    else:
        fig = plt.figure(figsize=(10.2, 9.3), constrained_layout=True)
        grid = fig.add_gridspec(4, 3, height_ratios=[1, 2, 3, 2.6])
        axes = np.array([[fig.add_subplot(grid[row, col]) for col in range(3)] for row in range(3)])
    for upper, ax, x, mean, r, label in zip(axes[0], axes[1], values, means, controls, labels):
        upper.sharex(ax)
        upper.plot(np.arange(100), mean, color="#222222", lw=.8)
        upper.set(ylim=(.3, .9), ylabel=r"Global $\bar{x}(t)$", title=rf"{label}: $r={r:g}$")
        upper.spines[["top", "right"]].set_visible(False)
        upper.tick_params(direction="out", labelbottom=False)
        plot_mts_heatmap(x, method="robust", ax=ax)
        if full:
            # Display-only mask: never magnify roundoff in numerically constant sites.
            masked = np.ma.array(ax.images[0].get_array(),
                mask=np.broadcast_to((x.std(axis=1)<=1e-8)[:,None], x.shape))
            palette=ax.images[0].get_cmap().copy(); palette.set_bad("#bdbdbd")
            ax.images[0].set_data(masked); ax.images[0].set_cmap(palette)
        ax.images[0].set_clim(-limit, limit)
        ax.set(xlim=(-.5, 99.5), xlabel="Consecutive step", ylabel="Site (row-major)" if full else "Dispersed sensor")
        ax.set_yticks([0, M//2-1, M-1], [1, M//2, M])
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(direction="out")
    fig.colorbar(axes[1,-1].images[0], ax=list(axes[1]), shrink=.8, label="Robust-scaled value")
    for ax, (field, sensors, _) in zip(axes[2], lattices):
        side = field.shape[0]
        lattice_image = ax.imshow(field, origin="lower", cmap="viridis", vmin=0, vmax=1,
            interpolation="nearest", rasterized=True)
        if not full:
            ax.scatter(sensors % side, sensors // side, s=12, facecolors="none",
                edgecolors="white", linewidths=.65)
        ax.set(xlabel="Lattice column", ylabel="Lattice row", title="Late-time lattice")
        ax.set_xticks([0, side//2, side-1]); ax.set_yticks([0, side//2, side-1])
        ax.tick_params(direction="out")
    fig.colorbar(lattice_image, ax=list(axes[2]), shrink=.8, label=r"Raw $x_{ij}$")
    if tracking_panel is not None:
        tracking_panel(fig.add_subplot(grid[3, :]))
    stem = "headline-tracking-preview" if tracking_panel is not None else "headline-mts-snapshots"
    for suffix in ("png", "svg"):
        fig.savefig(output / f"{stem}.{suffix}", dpi=600, bbox_inches="tight")
    plt.close(fig)
    if tracking_panel is not None:
        if full:
            (output / f"{stem}-provenance.json").write_text(json.dumps(dict(examples=examples,
                M=M, input_T=1000, display_T=100, master_only=master_only,
                constant_site_display="SD <=1e-8 masked grey; raw data unchanged",
                lattice_recorded_step=[entry[2] for entry in lattices]), indent=2)+"\n")
        return output / f"{stem}.png"
    for x, r in zip(values, controls):
        slug = str(r).replace(".", "p")
        for steps in (100, 1000):
            fig = plt.figure(figsize=(3.6, 1.35))
            ax = fig.add_axes((0, 0, 1, 1))
            plot_mts_heatmap(x, method="robust", ax=ax)
            ax.images[0].set_clim(-limit, limit)
            ax.set_xlim(-.5, steps-.5)
            ax.set_axis_off()
            for suffix in ("png", "svg"):
                fig.savefig(output / f"mts-r-{slug}-T{steps}.{suffix}", dpi=600, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
    result = dict(examples=examples, M=M, input_T=1000, display_T=100,
        temporal_stride=1, scaling="per-process robust, estimated on full T1000 input",
        common_colour_limit=limit, selection="first held seed and first input prefix; fixed control anchors",
        global_trace="exact matching first100 steps, unscaled full-lattice mean",
        lattice_context=[dict(r=r, seed=seed, recorded_step=lattice[2],
            note="saved final field after the reference window; not simultaneous with the MTS panels")
            for r, lattice in zip(controls, lattices)],
        source_archive_sha256=manifest["archive_sha256"])
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def feature_audit(root: Path):
    """Exact frozen linear decomposition; correlated loadings are not causal."""
    analysis = root / "primary-analysis"
    with np.load(analysis / "model.npz", allow_pickle=False) as model:
        keep, w = model["keep"], model["component"]
        center, impute, scale = model["center"], model["impute"], float(model["score_scale"])
        order = model["spi_order"].tolist()
    schema = build_unified_schema(order)
    selected = [schema[i] for i in keep]
    mass = w*w
    pairs = pd.DataFrame(dict(spi_a=[p.spi_a for p in selected], spi_b=[p.spi_b for p in selected],
        loading=w, squared_loading=mass))
    features = root / "primary-replay/features.npz"
    with np.load(features, allow_pickle=False) as archive:
        z = archive["z"][:, keep]
        ids = archive["row_id"].tolist()
        assert archive["spi_order"].tolist() == order
    x = np.where(np.isfinite(z), z, impute) - center
    scores = pd.read_csv(analysis / "scores.csv").set_index("row_id").loc[ids]
    reconstructed = x @ w / scale
    error = float(np.max(np.abs(reconstructed - scores.q.to_numpy())))
    assert error < 1e-10
    held = (scores.role == "evaluation") & scores.eligible
    low, high = float(scores.r.min()), float(scores.r.max())
    change = x[(held & (scores.r == high)).to_numpy()].mean(axis=0) - x[(held & (scores.r == low)).to_numpy()].mean(axis=0)
    contribution = change*w/scale
    observed_change = float(scores.loc[held & (scores.r == high), "q"].mean() - scores.loc[held & (scores.r == low), "q"].mean())
    assert abs(contribution.sum()-observed_change) < 1e-10
    pairs["held_endpoint_delta_q_contribution"] = contribution
    pairs = pairs.iloc[np.argsort(-abs(w))].reset_index(drop=True)
    endpoint_rows = []
    for side in ("spi_a", "spi_b"):
        endpoint_rows.append(pd.DataFrame(dict(spi=pairs[side], mass=pairs.squared_loading/2,
            contribution=pairs.held_endpoint_delta_q_contribution/2)))
    spis = pd.concat(endpoint_rows).groupby("spi").agg(
        loading_mass=("mass", "sum"), retained_partners=("mass", "size"),
        mean_mass_per_partner=("mass", "mean"), endpoint_contribution=("contribution", "sum"),
    ).sort_values("loading_mass", ascending=False).reset_index()
    summary = dict(retained_pairs=len(w), represented_SPIs=len(spis),
        top_10_pair_loading_mass=float(np.sort(mass)[-10:].sum()),
        top_100_pair_loading_mass=float(np.sort(mass)[-100:].sum()),
        effective_loading_count=float(mass.sum()**2/(mass@mass)),
        held_endpoint_delta_q=observed_change,
        sum_endpoint_contributions=float(contribution.sum()), maximum_q_reconstruction_error=error)
    output = root / "feature-audit"
    output.mkdir(exist_ok=True)
    pairs.to_csv(output / "pairs.csv", index=False)
    spis.to_csv(output / "spis.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return dict(summary=summary, pairs=pairs, spis=spis)
