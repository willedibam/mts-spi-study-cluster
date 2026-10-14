"""Full r9 hardware analysis: bounded z, ER-fitted PCA and original-edge checks.

No pyspi extraction, imputation, variance scaling or mechanistic naming occurs here.
SW projection is exploratory transfer: its fixed-feature results were already seen.
Call ``analyse(root)`` after mirroring the complete run; plotting is opt-in.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mtsspi.corpus import ArchiveCorpus
from mtsspi.provenance import array_sha256, file_sha256
from mtsspi.run_corpus import completion_error
from mtsspi.spi_spi_contract import FeatureSpec, build_unified_features, schema_sha256
from research.empirical.hardware_pilot.hardware_pilot import correlation_endpoint

LAG_PAIR = ("corr_pearson_tau-1", "corr_pearson_tau-10")
COLORS = {"ER": "#26828e", "SW": "#c66a25"}


@dataclass(frozen=True)
class FullP90:
    root: Path
    rows: tuple[dict, ...]
    z: np.ndarray
    schema: tuple[FeatureSpec, ...]
    paths: tuple[Path, ...]
    sources: tuple[dict, ...]
    identity: dict

    def mpis(self, index, names):
        with np.load(self.paths[index], allow_pickle=False) as archive:
            return {name: archive[name] for name in names}


def load_run(root, *, expected_spis=289, min_coverage=.90):
    """Validate the saved execution identity, unchanged config and complete census.

    Cluster paths are relocated in memory; the frozen YAML bytes remain unchanged.
    Undefined SPIs are retained as missing features, never repaired or removed here.
    The production coverage gate matches full_p90_run; tiny fixtures may override it.
    """
    root = Path(root).resolve()
    identity = json.loads((root / "execution-identity.json").read_text())
    required = {"corpus_config_sha256", "pyspi_config_sha256", "code_sha256", "pyspi_version"}
    code_files = {"run_corpus.py", "corpus.py", "generated_corpus.py", "compute.py"}
    if (not required.issubset(identity) or set(identity["code_sha256"]) != code_files
            or identity["pyspi_version"]["computation"] != "3.0.0.r9"):
        raise ValueError("A frozen, complete r9 execution identity is required")
    config = ArchiveCorpus.from_file(root / "external.yaml")
    config = replace(config, archive=root / "observations.npz", base_output_dir=root / "mpis")
    manifest = json.loads((root / "manifest.json").read_text())
    for path, expected in ((config.path, identity["corpus_config_sha256"]),
                           (config.pyspi_config, identity["pyspi_config_sha256"]),
                           (config.archive, config.archive_sha256)):
        if file_sha256(path) != expected:
            raise ValueError(f"Frozen content hash mismatch: {path}")
    if manifest["archive_sha256"] != config.archive_sha256 or config.pair_tolerance is not None:
        raise ValueError("Manifest/config disagree or extraction used sampled pairs")
    rows, entries = manifest["rows"], config.entries()
    values, paths, sources, catalogue, revision = [], [], [], None, None
    with np.load(config.archive, allow_pickle=False) as observations:
        for entry, row in zip(entries, rows, strict=True):
            if (row["name"] != entry.name or row["topology"] not in COLORS
                    or (entry.M, entry.T, row["M"], row["T"], row["fs"], row["seconds"])
                    != (28, 5000, 28, 5000, 500, 10)):
                raise ValueError(f"Unexpected recording identity or sampling: {entry.name}")
            if error := completion_error(config, entry, identity=identity):
                raise ValueError(f"{entry.name}: {error}")
            directory = entry.output_dir(config)
            meta = json.loads((directory / "meta.json").read_text())
            if meta.get("job", {}).get("n_jobs") != 4:
                raise ValueError(f"Unexpected worker count for {entry.name}; expected n_jobs=4")
            member_hash = array_sha256(observations[entry.name])
            if member_hash != meta["source"]["member_sha256"] or member_hash != row["member_sha256"]:
                raise ValueError(f"Source member hash mismatch: {entry.name}")
            if meta["pyspi"]["version"] != identity["pyspi_version"]:
                raise ValueError(f"Conflicting pyspi version: {entry.name}")
            current_catalogue = meta["pyspi"]["spis"]
            revision = revision or meta["experiment"]["git_commit"]
            catalogue = current_catalogue if catalogue is None else catalogue
            if (catalogue != current_catalogue or len(catalogue) != expected_spis
                    or meta["experiment"]["git_commit"] != revision or meta["experiment"]["git_dirty"]):
                raise ValueError(f"Mixed catalogue/source revision or dirty runner: {entry.name}")
            names = [item["name"] for item in catalogue]
            path = directory / "spi_mpis.npz"
            with np.load(path, allow_pickle=False) as archive:
                mpis = {name: archive[name] for name in names}
            edge = ~np.eye(entry.M, dtype=bool)
            coverage = float(np.mean([np.isfinite(a[edge]).mean() for a in mpis.values()]))
            failures = len(meta["pyspi"]["errors"])
            if coverage < min_coverage or failures > 15:
                raise ValueError(f"Insufficient SPI coverage for {entry.name}: fraction {coverage:.6f}, {failures} failed SPIs")
            features = build_unified_features(mpis, names)
            values.append(features.z)
            paths.append(path)
            sources.append(dict(name=entry.name, mpi_sha256=file_sha256(path),
                                valid_features=int(features.valid.sum()),
                                finite_edge_fraction=coverage,
                                failed_spis=meta["pyspi"]["errors"], invalid_spis=features.invalid_reasons))
    if not values:
        raise ValueError("Empty corpus")
    return FullP90(root, tuple(rows), np.array(values), features.schema, tuple(paths), tuple(sources), identity)


@dataclass(frozen=True)
class FrozenPCA:
    features: np.ndarray
    center: np.ndarray
    components: np.ndarray
    variance_ratio: np.ndarray
    fit_indices: np.ndarray

    def missing_fraction(self, z):
        return (~np.isfinite(np.asarray(z)[:, self.features])).mean(axis=1)

    def project(self, z):
        """Whole-row rejection preserves the fitted coordinate; no mean imputation."""
        selected = np.asarray(z, dtype=float)[:, self.features]
        valid = np.isfinite(selected).all(axis=1)
        scores = np.full((len(selected), len(self.components)), np.nan)
        scores[valid] = (selected[valid] - self.center) @ self.components.T
        return scores


def fit_pca(z, fit_rows, *, feature_mask=None, components=2, spread_floor=1e-8):
    """Select finite/nonconstant coordinates and center using fit rows only.

    An externally supplied mask may condition on SW completeness; label that use.
    PC signs depend only on the largest absolute loading, never physical labels.
    """
    z, fit_rows = np.asarray(z, dtype=float), np.asarray(fit_rows, dtype=bool)
    if z.ndim != 2 or fit_rows.shape != (len(z),) or fit_rows.sum() < 3:
        raise ValueError("PCA requires at least three explicitly selected fit recordings")
    if np.any(np.abs(z[np.isfinite(z)]) > 1 + 1e-6):
        raise ValueError("Expected bounded SPI–SPI correlations, without Fisher transformation")
    keep = np.isfinite(z[fit_rows]).all(axis=0)
    if feature_mask is not None:
        mask = np.asarray(feature_mask, dtype=bool)
        if mask.shape != keep.shape:
            raise ValueError("Feature mask has the wrong shape")
        keep &= mask
    indices = np.flatnonzero(keep)
    indices = indices[z[fit_rows][:, indices].std(axis=0) > spread_floor]
    if len(indices) < 2 or components < 1:
        raise ValueError("Insufficient varying, complete fit features or invalid component count")
    train = z[fit_rows][:, indices]
    center = train.mean(axis=0)
    _, singular, loadings = np.linalg.svd(train - center, full_matrices=False)
    count = min(components, len(train) - 1, len(indices))
    loadings = loadings[:count]
    for vector in loadings:
        vector *= 1 if vector[np.argmax(abs(vector))] >= 0 else -1
    return FrozenPCA(indices, center, loadings, singular ** 2 / np.sum(singular ** 2), np.flatnonzero(fit_rows))


def rank_loadings(bank, model):
    rows = []
    for index, weight in zip(model.features, model.components[0], strict=True):
        spec = bank.schema[index]
        rows.append(dict(feature_index=int(index), spi_a=spec.spi_a, spi_b=spec.spi_b, weight=float(weight),
                         both_correlation=correlation_endpoint(spec.spi_a) and correlation_endpoint(spec.spi_b)))
    table = pd.DataFrame(rows).sort_values("weight", key=abs, ascending=False, kind="stable").reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return table


def loading_stability(z, model):
    """Leave one fit recording out, recenter/refit on the frozen feature set.

    Absolute loading cosines are sampling sensitivity, not independent replication.
    """
    train = np.asarray(z, float)[model.fit_indices][:, model.features]
    cosines = []
    for leave in range(len(train)):
        retained = np.delete(train, leave, axis=0)
        _, _, vectors = np.linalg.svd(retained - retained.mean(axis=0), full_matrices=False)
        cosines.append(float(abs(vectors[0] @ model.components[0])))
    return np.array(cosines)


def contrast_contributions(bank, model, left, right):
    """w_j Δz_j explains a chosen PC1 movement; |w_j| answers a different question."""
    if not np.isfinite(model.project(bank.z)[[left, right], 0]).all():
        raise ValueError("Contrast includes an unprojectable recording")
    table = rank_loadings(bank, model)
    table["delta_z"] = bank.z[right, table.feature_index] - bank.z[left, table.feature_index]
    table["pc1_contribution"] = table.weight * table.delta_z
    return table.sort_values("pc1_contribution", key=abs, ascending=False, kind="stable")


def _edge_corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return np.nan
    a, b = a - a.mean(), b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float((a / na) @ (b / nb)) if min(na, nb) >= 1e-12 else np.nan


def edge_diagnostics(bank, feature_index, row_indices=None):
    """All ordered edges; deletion filters cached edges, not estimator refits or CIs.

    Report scale and spread: numerical validity alone does not establish reliability.
    Directed endpoints remain directed, including in the original-edge scatterplots.
    """
    spec, rows = bank.schema[feature_index], []
    for index in range(len(bank.rows)) if row_indices is None else row_indices:
        matrices = bank.mpis(index, (spec.spi_a, spec.spi_b))
        a, b = matrices[spec.spi_a], matrices[spec.spi_b]
        i, j = np.where(~np.eye(len(a), dtype=bool))
        x, y = a[i, j], b[i, j]
        deleted = np.array([_edge_corr(x[(i != k) & (j != k)], y[(i != k) & (j != k)]) for k in range(len(a))])
        rows.append(dict(name=bank.rows[index]["name"], feature_index=feature_index, z=_edge_corr(x, y),
                         mean_a=float(x.mean()), mean_b=float(y.mean()), sd_a=float(x.std()), sd_b=float(y.std()),
                         range_a=float(np.ptp(x)), range_b=float(np.ptp(y)),
                         delete_min=float(deleted.min()), delete_max=float(deleted.max()),
                         undefined_deletions=int((~np.isfinite(deleted)).sum())))
    return pd.DataFrame(rows)


def frozen_benchmark(bank):
    """Recompute the frozen 2/20 ms feature and unchanged three-band explanation.

    Archive parity uses the full 10 s; physical reconstruction uses the central 9 s.
    This distinction is retained in the output and neither feature nor bands are fit.
    """
    from research.empirical.hardware_pilot.calibrate_hardware_pilot import lag_matrix
    from research.empirical.hardware_pilot.transfer_hardware_pilot import explain_record
    rows = []
    with np.load(bank.root / "observations.npz", allow_pickle=False) as observations:
        for index, row in enumerate(bank.rows):
            x = observations[row["name"]]
            archived = bank.mpis(index, LAG_PAIR)
            edge = ~np.eye(len(x), dtype=bool)
            parity = [np.max(abs(lag_matrix(x, tau)[edge] - archived[name][edge]))
                      for tau, name in zip((1, 10), LAG_PAIR, strict=True)]
            summary, _, _, _, _ = explain_record(x, row["fs"])
            rows.append(dict(name=row["name"], topology=row["topology"], step=row["step"],
                             archived_z=_edge_corr(*(archived[name][edge] for name in LAG_PAIR)),
                             archive_profile_max_error=float(np.max(parity)),
                             archive_parity_pass=bool(np.isfinite(parity).all() and np.max(parity) <= 1e-10), **summary))
    return pd.DataFrame(rows)


def analyse(root, *, exploratory=False):
    """Write compact numerical evidence, return objects for optional notebook plots.

    Primary ER fit and correlation-filtered refit use ER availability only. A common-
    complete ER fit is added only if primary projection loses SW recordings. Optional
    SW/pooled fits are descriptive sampling-design sensitivities, not held-out tests.
    """
    bank = load_run(root)
    er = np.array([row["topology"] == "ER" for row in bank.rows])
    no_double_corr = np.array([not (correlation_endpoint(s.spi_a) and correlation_endpoint(s.spi_b)) for s in bank.schema])
    models, unavailable = {"er": fit_pca(bank.z, er)}, {}
    variants = {"er_no_double_correlation": (er, no_double_corr)}
    if np.any(models["er"].missing_fraction(bank.z)[~er] > 0):
        variants["er_common_complete"] = (er, np.isfinite(bank.z).all(axis=0))
    if exploratory:
        variants.update(sw_exploratory=(~er, None), pooled_exploratory=(np.ones(len(er), bool), None))
    for name, (fit, mask) in variants.items():
        try:
            models[name] = fit_pca(bank.z, fit, feature_mask=mask)
        except ValueError as error:
            unavailable[name] = str(error)
    output = bank.root / "analysis"
    output.mkdir(exist_ok=True)
    spi_order = list(dict.fromkeys(name for spec in bank.schema for name in (spec.spi_a, spec.spi_b)))
    arrays, summaries = {"z": bank.z, "spi_order": np.array(spi_order)}, {}
    scores_table = pd.DataFrame(bank.rows)[["name", "topology", "step", "voltage"]]
    for name, model in models.items():
        scores, missing = model.project(bank.z), model.missing_fraction(bank.z)
        table = rank_loadings(bank, model)
        table.to_csv(output / f"{name}-loadings.csv", index=False)
        for component in range(scores.shape[1]):
            scores_table[f"{name}_pc{component + 1}"] = scores[:, component]
        scores_table[f"{name}_missing_fraction"] = missing
        arrays.update({f"{name}_{key}": getattr(model, key) for key in ("features", "center", "components", "variance_ratio", "fit_indices")})
        summaries[name] = dict(features=len(model.features), fit_names=[bank.rows[i]["name"] for i in model.fit_indices],
                               variance_ratio=model.variance_ratio.tolist(),
                               unprojectable=[bank.rows[i]["name"] for i in np.flatnonzero(missing)],
                               top20_double_correlation=int(table.head(20).both_correlation.sum()))
    scores_table.to_csv(output / "scores.csv", index=False)
    np.savez_compressed(output / "features.npz", **arrays)
    benchmark = frozen_benchmark(bank)
    benchmark.to_csv(output / "frozen-lag-benchmark.csv", index=False)
    primary = models["er"].project(bank.z)[:, 0]
    filtered = models["er_no_double_correlation"].project(bank.z)[:, 0] if "er_no_double_correlation" in models else np.full(len(er), np.nan)
    comparisons = {}
    for topology, mask in (("ER", er), ("SW", ~er)):
        paired = mask & np.isfinite(primary) & np.isfinite(filtered)
        similarity = abs(_edge_corr(primary[paired], filtered[paired])) if paired.sum() > 1 else np.nan
        comparisons[topology] = dict(paired_records=int(paired.sum()),
                                     absolute_pc1_correlation=float(similarity) if np.isfinite(similarity) else None)
    parity = benchmark.archive_profile_max_error.to_numpy()
    report = dict(records=len(bank.rows), computation=bank.identity["pyspi_version"],
                  schema_sha256=schema_sha256(bank.schema), execution_identity=bank.identity,
                  models=summaries, unavailable_sensitivities=unavailable,
                  er_leave_record_loading_cosines=loading_stability(bank.z, models["er"]).tolist(),
                  correlation_exclusion_sensitivity=comparisons, sources=bank.sources,
                  benchmark_archive_profile_max_error=float(parity.max()) if np.isfinite(parity).all() else None,
                  benchmark_archive_parity_failures=benchmark.loc[~benchmark.archive_parity_pass, "name"].tolist(),
                  benchmark_reconstruction_passes=int(benchmark.reconstruction_target_met.sum()),
                  benchmark_reconstruction_failures=benchmark.loc[~benchmark.reconstruction_target_met, "name"].tolist(),
                  script_sha256=file_sha256(Path(__file__)),
                  scope="One selected sweep per topology; SW fixed-feature results previously inspected. Equal-record PCA is sampling-design dependent.",
                  fitting="Centered bounded z; no scaling or imputation. Sign by largest absolute loading. Common-complete selection uses SW availability.",
                  interpretation="High loadings do not establish a mechanism. Inspect original profile spread and frozen reconstruction failures; channel-edge deletion is not refitting or a confidence interval.")
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return dict(bank=bank, models=models, benchmark=benchmark, summary=report)


def _panels(count):
    import matplotlib.pyplot as plt
    from research.empirical.hardware_pilot.plot_hardware_pilot import box, style
    if count < 1:
        raise ValueError("Select at least one panel")
    style()
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white"})
    fig, axes = plt.subplots(1, count, figsize=(2.7 * count, 2.8), squeeze=False)
    for ax in axes[0]:
        box(ax)
    return fig, axes[0]


def save_figure(fig, output, name):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output / f"{name}.{suffix}", dpi=180 if suffix == "png" else 600, bbox_inches="tight")
    return fig


def _voltage_axis(ax):
    ax.set_xscale("symlog", linthresh=.05)
    ax.set_xlim(0, 3.5)
    ax.set_xticks([0, .05, .2, 1, 3.5], ["0", "0.05", "0.2", "1", "3.5"])


def plot_overview(bank, models):
    """Primary frozen geometry, control trajectory and spectrum, with common scales."""
    model = models["er"]
    scores = model.project(bank.z)
    if scores.shape[1] < 2:
        raise ValueError("The geometry figure needs at least two fitted components")
    fig, axes = _panels(3)
    for topology, color in COLORS.items():
        indices = sorted((i for i, r in enumerate(bank.rows) if r["topology"] == topology), key=lambda i: bank.rows[i]["step"])
        if not indices:
            continue
        voltage = [bank.rows[i]["voltage"] for i in indices]
        valid = np.isfinite(scores[indices]).all(axis=1).sum()
        label = f"{topology} {'fit' if topology == 'ER' else 'projection'} ({valid}/{len(indices)})"
        axes[0].plot(scores[indices, 0], scores[indices, 1], "o-", color=color, label=label)
        axes[1].plot(voltage, scores[indices, 0], "o-", color=color)
    axes[0].set(xlabel="ER PC1", ylabel="ER PC2")
    axes[0].legend()
    axes[1].set(xlabel="Control voltage (V)", ylabel="ER PC1")
    _voltage_axis(axes[1])
    count = min(10, len(model.variance_ratio))
    axes[2].plot(np.arange(1, count + 1), model.variance_ratio[:count] * 100, "o-", color=".2")
    axes[2].set(xlabel="ER component", ylabel="Variance explained (%)")
    return fig


def plot_loadings(bank, model, feature_indices=None, *, count=4):
    """A small loading-ranked gallery; caller can pass frozen selected contrasts."""
    from research.empirical.hardware_pilot.plot_hardware_pilot import short
    table = rank_loadings(bank, model).set_index("feature_index")
    chosen = list(table.index[:count] if feature_indices is None else feature_indices)
    fig, axes = _panels(len(chosen))
    for ax, index in zip(axes, chosen, strict=True):
        spec = bank.schema[index]
        label = f"#{table.loc[index, 'rank']} · w={table.loc[index, 'weight']:+.3f}" if index in table.index else "Excluded from PCA"
        for topology, color in COLORS.items():
            rows = sorted((i for i, r in enumerate(bank.rows) if r["topology"] == topology), key=lambda i: bank.rows[i]["step"])
            if not rows:
                continue
            ax.plot([bank.rows[i]["voltage"] for i in rows], bank.z[rows, index], "o-", color=color, label=topology)
        ax.set(xlabel="Control voltage (V)", ylabel="Ordered-edge Pearson z", ylim=(-1.05, 1.05),
               title=f"{label}\n{short(spec.spi_a)}\n× {short(spec.spi_b)}")
        _voltage_axis(ax)
    axes[0].legend()
    return fig


def plot_edge_contrast(bank, feature_index, row_indices):
    """Matched axes show the original profiles, including any near-collapse."""
    from research.empirical.hardware_pilot.plot_hardware_pilot import short
    spec = bank.schema[feature_index]
    diagnostics = edge_diagnostics(bank, feature_index, row_indices).set_index("name")
    fig, axes = _panels(len(row_indices))
    profiles = []
    for ax, index in zip(axes, row_indices, strict=True):
        row = bank.rows[index]
        mpi = bank.mpis(index, (spec.spi_a, spec.spi_b))
        edge = ~np.eye(row["M"], dtype=bool)
        x, y = mpi[spec.spi_a][edge], mpi[spec.spi_b][edge]
        profiles.append((x, y))
        ax.scatter(x, y, s=5, alpha=.4, color=COLORS[row["topology"]], linewidths=0, rasterized=True)
        d = diagnostics.loc[row["name"]]
        ax.set(xlabel=short(spec.spi_a), ylabel=short(spec.spi_b),
               title=f"{row['topology']} · setting {row['step']}\nz={d.z:+.3f}; SD={d.sd_a:.2g}, {d.sd_b:.2g}")
    for dim, setter in ((0, "set_xlim"), (1, "set_ylim")):
        values = np.concatenate([p[dim] for p in profiles])
        values = values[np.isfinite(values)]
        if len(values):
            pad = max(float(np.ptp(values)) * .04, np.finfo(float).eps * max(1., abs(values).max()))
            for ax in axes:
                getattr(ax, setter)(values.min() - pad, values.max() + pad)
    return fig
