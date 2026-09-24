"""Cached-MPI baseline exploration and an exactly solvable VAR illustration.

All learned transforms fit development rows only, except the explicitly
transductive Zenodo atlas. No pyspi calls or new physical benchmark simulations.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

from src.corpus_geometry import fit_geometry_transform
from src.representation_screen import bootstrap_group_means
from scripts.order_parameter_simple_baselines import input_statistics

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/baseline-comparison_260921"
OUT = ROOT / "results/baseline-comparison_260921"
CONFIG = ROOT / "configs/analysis/spi-baseline-exploration-260921.yaml"
STATS = ("mean", "std", "q10", "q25", "median", "q75", "q90")
VIEWS = ("pearson", "mean", "distribution", "z", "mean+z", "distribution+z")
CML = ("chaotic-traveling-wave", "defect-turbulence", "fdstc", "frozen-chaos", "sti-i")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def finite_json(value):
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite_json(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def summarize(mpis, order, symmetric=False):
    """Whole-profile validity; constant finite MPIs retain meaningful means."""
    n = len(mpis[order[0]])
    assert all(np.asarray(mpis[k]).shape == (n, n) for k in order)
    mask = np.triu(np.ones((n, n), dtype=bool), 1) if symmetric else ~np.eye(n, dtype=bool)
    vectors = np.array([((mpis[k] + mpis[k].T) / 2 if symmetric else mpis[k])[mask] for k in order])
    result = np.full((len(order), len(STATS)), np.nan)
    finite = np.isfinite(vectors).all(axis=1)
    x = vectors[finite]
    result[finite] = np.column_stack((x.mean(axis=1), x.std(axis=1),
                                     np.quantile(x, [.1, .25, .5, .75, .9], axis=1).T))
    return result


def extract_proof():
    manifest_path = DATA / "proof-inputs.json"
    manifest = json.loads(manifest_path.read_text())
    records, order = manifest["records"], manifest["spi_order"]
    features, raw_stats, hashes = [], [], []
    for i, row in enumerate(records):
        path, meta = ROOT / row["mpi_path"], ROOT / row["meta_path"]
        assert sha(path) == row["mpi_sha256"] and sha(meta) == row["meta_sha256"]
        with np.load(path, allow_pickle=False) as a:
            assert set(a.files) == set(order)
            features.append(summarize(a, order, symmetric=True))
        raw = ROOT / row["raw_path"]
        x = np.load(raw, allow_pickle=False)
        assert x.shape == (row["T"], row["M"])
        summary = input_statistics(x.T)
        raw_stats.append([summary["mean_correlation"], summary["mean_abs_correlation"]])
        hashes.append(dict(row_id=row["row_id"], raw_sha256=sha(raw), mpi_sha256=row["mpi_sha256"]))
        if (i + 1) % 300 == 0:
            print(f"proof extraction {i + 1}/{len(records)}", flush=True)
    DATA.mkdir(parents=True, exist_ok=True)
    output = DATA / "proof-summaries.npz"
    np.savez_compressed(output, marginal=np.array(features), pearson=np.array(raw_stats),
        row_id=np.array([r["row_id"] for r in records]), spi_order=np.array(order))
    write_json(output.with_suffix(".json"), dict(input_manifest_sha256=sha(manifest_path),
        artifact_sha256=sha(output), sources=hashes, symmetric=True, stats=STATS))


@dataclass
class Projection:
    transform: object
    pca: PCA
    clip: float | None

    def project(self, x):
        x = self.transform.transform(x)
        if self.clip is not None:
            x = np.clip(x, -self.clip, self.clip)
        return self.pca.transform(x)


def project_features(train, query, *, standard=True, dimensions=50, valid=.95):
    transform = fit_geometry_transform(train, scaling="standard" if standard else "center",
        minimum_valid_fraction=valid, variance_threshold=1e-8)
    x = transform.transform(train)
    clip = 5.0 if standard else None
    if clip is not None:
        x = np.clip(x, -clip, clip)
    pca = PCA(n_components=min(dimensions, len(train)-1, x.shape[1]),
              svd_solver="randomized", random_state=260921).fit(x)
    model = Projection(transform, pca, clip)
    return model, model.project(train), model.project(query)


def balance(train, query):
    mean = train.mean(axis=0)
    scale = np.sqrt(np.var(train, axis=0).sum())
    assert scale > 0
    return (train-mean)/scale, (query-mean)/scale


def fuse(left, right):
    a, b = balance(*left), balance(*right)
    train, query = np.column_stack((a[0], b[0])), np.column_stack((a[1], b[1]))
    pca = PCA(n_components=min(50, train.shape[1], len(train)-1), svd_solver="full").fit(train)
    return balance(pca.transform(train), pca.transform(query))


def row_keys(labels, m, t, instance):
    return np.array([f"{l}|M{int(mi)}|T{int(ti)}|I{int(ii)}" for l, mi, ti, ii in zip(labels, m, t, instance)])


def retrieval(train, query, train_meta, query_meta):
    distances = cdist(query, train)
    allowed = ((query_meta["M"].to_numpy()[:, None] != train_meta["M"].to_numpy()) &
               (query_meta["T"].to_numpy()[:, None] != train_meta["T"].to_numpy()))
    distances[~allowed] = np.inf
    order = np.argsort(distances, axis=1, kind="stable")
    valid = np.take_along_axis(allowed, order, axis=1)
    match = (train_meta.label.to_numpy()[order] == query_meta.label.to_numpy()[:, None]) & valid
    precision = np.cumsum(match, axis=1) / np.arange(1, len(train)+1)
    assert np.all(match.sum(axis=1) > 0)
    ap = (precision * match).sum(axis=1) / match.sum(axis=1)
    return ap, match[:, 0].astype(float)


def proof_analysis():
    from umap import UMAP
    frame = pd.DataFrame(json.loads((DATA / "proof-inputs.json").read_text())["records"])
    with np.load(DATA / "proof-summaries.npz", allow_pickle=False) as a:
        assert np.array_equal(a["row_id"], frame.row_id)
        marg, raw = a["marginal"], a["pearson"]
    coordinate_file = ROOT / "results/cross_mt_transfer_260824/confirmation-coordinates.npz"
    with np.load(coordinate_file, allow_pickle=True) as a:
        keys_dev = a["development_row_keys"].astype(str)
        keys_eval = row_keys(a["confirmation_y"], a["confirmation_M"], a["confirmation_T"], a["confirmation_instance"])
        z = (a["development_pca_sym"], a["confirmation_pca_sym"])
    indices = pd.Series(np.arange(len(frame)), index=frame.row_id)
    dev, held = indices.loc[keys_dev].to_numpy(), indices.loc[keys_eval].to_numpy()
    train_meta, eval_meta = frame.iloc[dev].reset_index(drop=True), frame.iloc[held].reset_index(drop=True)
    assert (train_meta.instance < 10).all() and (eval_meta.instance >= 10).all()
    projections, information = {}, {}
    for name, x in (("pearson", raw), ("mean", marg[:, :, 0]), ("distribution", marg.reshape(len(marg), -1))):
        model, a, b = project_features(x[dev], x[held])
        projections[name] = balance(a, b)
        information[name] = dict(retained=len(model.transform.keep_indices),
            pca_dimensions=a.shape[1], explained_variance=float(model.pca.explained_variance_ratio_.sum()))
    projections["z"] = balance(*z)
    information["z"] = dict(source=str(coordinate_file.relative_to(ROOT)), source_sha256=sha(coordinate_file),
        pca_dimensions=50, original_frozen_projection=True)
    for name in ("mean", "distribution"):
        projections[f"{name}+z"] = fuse(projections[name], projections["z"])
    OUT.mkdir(parents=True, exist_ok=True)
    arrays, metrics, individual = {}, [], []
    labels = train_meta.label.to_numpy()
    for name, (a, b) in projections.items():
        arrays[f"{name}_dev"], arrays[f"{name}_eval"] = a, b
        for scope, classes in (("all14", np.unique(labels)), ("CML5", np.array(CML))):
            keep_dev, keep_eval = np.isin(labels, classes), np.isin(eval_meta.label, classes)
            model = LogisticRegression(C=1, max_iter=3000).fit(a[keep_dev], labels[keep_dev])
            predicted = model.predict(b[keep_eval])
            target = eval_meta.label.to_numpy()[keep_eval]
            ap, top1 = retrieval(a[keep_dev], b[keep_eval], train_meta[keep_dev], eval_meta[keep_eval])
            detail = eval_meta.loc[keep_eval, ["row_id", "label", "instance", "M", "T"]].copy()
            detail["correct"], detail["ap"], detail["top1"] = predicted == target, ap, top1
            detail["method"], detail["scope"] = name, scope
            individual.append(detail)
            groups = detail.groupby(["label", "instance"])[["correct", "ap", "top1"]].mean()
            boot = bootstrap_group_means(groups.to_numpy().T, groups.index.get_level_values(0).to_numpy(), 2000, 260921)
            ci = np.quantile(boot, [.025, .975], axis=1)
            metrics.append(dict(method=name, scope=scope, rows=len(detail),
                balanced_accuracy=balanced_accuracy_score(target, predicted), BA_low=ci[0,0], BA_high=ci[1,0],
                hard_mAP=ap.mean(), mAP_low=ci[0,1], mAP_high=ci[1,1], hard_top1=top1.mean()))
        if name in ("mean", "distribution", "z"):
            for scope, classes in (("inter", np.unique(labels[labels != "brownian-defect"])), ("cml", np.array(CML))):
                kd, ke = np.isin(labels, classes), np.isin(eval_meta.label, classes)
                pca = PCA(n_components=2, svd_solver="full").fit(a[kd])
                arrays[f"{name}_{scope}_pca"] = pca.transform(b[ke])
                mapper = UMAP(n_neighbors=30, min_dist=.1, random_state=260824, transform_seed=260824, n_jobs=1).fit(a[kd])
                arrays[f"{name}_{scope}_umap"] = mapper.transform(b[ke])
                arrays[f"{scope}_labels"] = eval_meta.label.to_numpy()[ke].astype(str)
        print(f"proof analysis {name}", flush=True)
    np.savez_compressed(OUT / "proof-projections.npz", **arrays)
    pd.DataFrame(metrics).to_csv(OUT / "proof-metrics.csv", index=False)
    details = pd.concat(individual, ignore_index=True)
    details.to_csv(OUT / "proof-individual.csv", index=False)
    paired = []
    for scope in details.scope.unique():
        g = details[details.scope == scope].groupby(["method", "label", "instance"])[["correct", "ap"]].mean()
        for left, right in (("z","mean"), ("z","distribution"), ("mean+z","mean"), ("distribution+z","distribution")):
            diff = g.loc[left] - g.loc[right]
            boot = bootstrap_group_means(diff.to_numpy().T, diff.index.get_level_values(0).to_numpy(), 2000, 260921)
            for j, measure in enumerate(diff.columns):
                lo, hi = np.quantile(boot[j], [.025, .975])
                paired.append(dict(scope=scope, comparison=f"{left} - {right}", metric=measure,
                                   difference=diff[measure].mean(), low=lo, high=hi))
    pd.DataFrame(paired).to_csv(OUT / "proof-paired.csv", index=False)
    write_json(OUT / "proof-provenance.json", dict(projections=information, config_sha256=sha(CONFIG),
        summaries_sha256=sha(DATA / "proof-summaries.npz"), development=1260, evaluation=2520,
        claim="exploratory new baseline comparison; pooled sizes, not original leave-cell-out validation"))


def var_population(strength, orientation, seed=0):
    """Prescribe C=Cov(x_t), L=Cov(x_t,x_{t-1}); Q=C-L C^-1 L^T."""
    n = 6
    upper = np.triu_indices(n, 1)
    v = np.random.default_rng(seed).permutation(np.linspace(-1, 1, len(upper[0])))
    C, L = np.eye(n), .2 * np.eye(n)
    C[upper] = strength + .035 * v
    L[upper] = .01 + orientation * .03 * v
    C[(upper[1], upper[0])] = C[upper]
    L[(upper[1], upper[0])] = L[upper]
    A = np.linalg.solve(C, L.T).T
    Q = C - A @ C @ A.T
    assert np.linalg.eigvalsh(C).min() > 0 and np.linalg.eigvalsh(Q).min() > 0
    assert np.max(abs(np.linalg.eigvals(A))) < 1
    np.testing.assert_allclose(A @ C @ A.T + Q, C, atol=1e-14)
    return C, L, A, Q


def toy_analysis():
    rows, population, matrices = [], [], {}
    for strength in (.02, .10):
        for orientation in (-1, 1):
            C, L, A, Q = var_population(strength, orientation)
            mask = ~np.eye(6, dtype=bool)
            label = f"strength={strength:.2f}, orientation={orientation:+d}"
            matrices[f"C_{strength}_{orientation}"], matrices[f"L_{strength}_{orientation}"] = C, L
            population.append(dict(strength=strength, orientation=orientation, mean_C=C[mask].mean(),
                mean_L=L[mask].mean(), z=np.corrcoef(C[mask], L[mask])[0,1],
                min_innovation_eigenvalue=np.linalg.eigvalsh(Q).min(), spectral_radius=max(abs(np.linalg.eigvals(A)))))
            for seed in range(24):
                C, L, A, Q = var_population(strength, orientation, seed)
                rng = np.random.default_rng(260921 + seed)
                # A stationary initial state removes any burn-in choice.
                x = np.empty((6000, 6))
                x[0] = np.linalg.cholesky(C) @ rng.normal(size=6)
                noise = rng.normal(size=x.shape) @ np.linalg.cholesky(Q).T
                for t in range(1, len(x)):
                    x[t] = A @ x[t-1] + noise[t]
                R = np.corrcoef(x.T)
                lag = np.corrcoef(x[1:].T, x[:-1].T)[:6, 6:]
                rows.append(dict(strength=strength, orientation=orientation, seed=seed,
                    mean_C=R[mask].mean(), mean_L=lag[mask].mean(),
                    z=np.corrcoef(R[mask], lag[mask])[0,1], label=label))
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "intuition-samples.csv", index=False)
    pd.DataFrame(population).to_csv(OUT / "intuition-population.csv", index=False)
    np.savez_compressed(OUT / "intuition-matrices.npz", **matrices)


def load_corpus(corpus, analysis, feature_path):
    """Join by recording identity and validate cached MPI provenance."""
    frame = pd.read_csv(analysis / "scores.csv")
    identity = json.loads((analysis / "eligibility.json").read_text())
    sources = {s["row_id"]: s for s in identity["sources"]}
    with np.load(feature_path, allow_pickle=False) as a:
        order = a["spi_order"].astype(str).tolist()
        index = pd.Series(np.arange(len(a["row_id"])), index=a["row_id"].astype(str))
        z = a["z"][index.loc[frame.row_id].to_numpy()]
    folders = {}
    for p in (corpus / "mpi").rglob("meta.json"):
        m = json.loads(p.read_text())
        assert m["status"] == "complete"
        folders[m["dataset_name"]] = p.parent
    with np.load(corpus / "observations.npz", allow_pickle=False) as raw:
        marg, stats, provenance = [], [], []
        for name in frame.row_id:
            path = folders[name] / "spi_mpis.npz"
            assert sha(path) == sources[name]["mpi_sha256"], name
            with np.load(path, allow_pickle=False) as a:
                assert set(a.files) == set(order)
                marg.append(summarize(a, order))
            x = raw[name]
            s = input_statistics(x)
            # Same-window population proxy: informative privileged baseline,
            # evaluated separately from the long, disjoint physical target.
            s["sample_period2"] = abs(x.mean(axis=0)[1::2] - x.mean(axis=0)[::2]).mean()
            stats.append(s)
            provenance.append(dict(row_id=name, mpi_sha256=sources[name]["mpi_sha256"]))
    for key in stats[0]:
        frame[key] = [s[key] for s in stats]
    if "control" not in frame:
        frame["control"] = frame.r
    return frame, np.array(marg), z, order, provenance


def fit_coordinate(matrix, dev, groups):
    """PC1 only, with no target/control argument; loading stability diagnostic."""
    pca = PCA(n_components=min(5, dev.sum()-1, matrix.shape[1]), svd_solver="full").fit(matrix[dev])
    w = pca.components_[0].copy()
    w *= 1 if w[np.argmax(abs(w))] >= 0 else -1
    values = (matrix-pca.mean_) @ w
    values /= values[dev].std()
    cosines = []
    for seed in np.unique(groups[dev]):
        local = dev & (groups != seed)
        other = PCA(n_components=1, svd_solver="full").fit(matrix[local]).components_[0]
        cosines.append(float(abs(w @ other)))
    evr = pca.explained_variance_ratio_
    return values, dict(evr1=float(evr[0]), evr_ratio=float(evr[0]/evr[1]),
        minimum_leave_seed_loading_cosine=min(cosines), features=matrix.shape[1])


def inference_analysis():
    kur = ROOT / "data/order_parameter/kuramoto_full_observation_260916/primary"
    pilot = ROOT / "data/order_parameter/cml2d_period_doubling_260911"
    confirm = ROOT / "data/order_parameter/cml2d_confirmation_260911"
    specifications = {
        "Kuramoto": [(kur, kur / "analysis", kur / "analysis/features.npz")],
        "CML2D": [(pilot / "primary", pilot / "primary-analysis", pilot / "primary-replay/features.npz"),
                  (confirm / "primary", confirm / "primary-analysis", confirm / "primary-analysis/features.npz")],
    }
    all_metrics, all_diagnostics, all_sources = [], {}, {}
    for system, parts in specifications.items():
        loaded = [load_corpus(*part) for part in parts]
        frame = pd.concat([x[0].assign(cohort=i) for i, x in enumerate(loaded)], ignore_index=True)
        marginal = np.concatenate([x[1] for x in loaded])
        z = np.concatenate([x[2] for x in loaded])
        assert all(x[3] == loaded[0][3] for x in loaded)
        all_sources[system] = [s for x in loaded for s in x[4]]
        dev = (frame.role == "development").to_numpy() & (frame.cohort == 0).to_numpy()
        evaluation = (frame.role == "evaluation").to_numpy() if system == "Kuramoto" else (frame.cohort == 1).to_numpy()
        frozen_model = parts[0][1] / "model.npz"
        with np.load(frozen_model, allow_pickle=False) as model:
            keep = model["keep"]
            z_selected = np.where(np.isfinite(z[:, keep]), z[:, keep], model["impute"]) - model["center"]
            reconstructed = (z_selected @ model["component"]) / float(model["score_scale"])
        np.testing.assert_allclose(reconstructed[frame.eligible], frame.loc[frame.eligible, "q"], atol=2e-6, rtol=2e-5)
        diagnostics, blocks, eligibility = {}, {}, frame.eligible.to_numpy().copy()
        for name, values in (("mean", marginal[:, :, 0]), ("distribution", marginal.reshape(len(frame), -1))):
            transformer = fit_geometry_transform(values[dev], scaling="standard", minimum_valid_fraction=.95)
            x = np.clip(transformer.transform(values), -5, 5)
            eligibility &= (np.mean(~np.isfinite(values[:, transformer.keep_indices]), axis=1) <= .05)
            x = (x-x[dev].mean(axis=0))/np.sqrt(np.var(x[dev], axis=0).sum())
            blocks[name] = x
            frame[f"{name}_PC1"], diagnostics[name] = fit_coordinate(x, dev, frame.seed.to_numpy())
        # A single declared sensitivity checks whether the comparison with the
        # historical q is driven by its centre-only preprocessing. This does
        # not replace the original q, its model or its gate outcome.
        z_transform = fit_geometry_transform(z[dev], scaling="standard", minimum_valid_fraction=.95)
        standardized_z = np.clip(z_transform.transform(z), -5, 5)
        eligibility &= (np.mean(~np.isfinite(z[:, z_transform.keep_indices]), axis=1) <= .05)
        frame["z_standard_PC1"], diagnostics["z_standard"] = fit_coordinate(standardized_z, dev, frame.seed.to_numpy())
        z_selected /= np.sqrt(np.var(z_selected[dev], axis=0).sum())
        joined = np.column_stack((blocks["mean"], z_selected))
        frame["mean+z_PC1"], diagnostics["mean+z"] = fit_coordinate(joined, dev, frame.seed.to_numpy())
        frame["development"], frame["evaluation"], frame["common_eligible"] = dev, evaluation, eligibility
        methods = ["q", "mean_PC1", "distribution_PC1", "mean+z_PC1", "z_standard_PC1", "mean_correlation", "mean_abs_correlation", "temporal_spectral_entropy"]
        methods += ["analytic_phase_coherence"] if system == "Kuramoto" else ["sample_period2"]
        eligible = evaluation & eligibility & np.isfinite(frame[methods + ["Q_reference"]]).all(axis=1)
        held = frame.loc[eligible].reset_index(drop=True)
        rng = np.random.default_rng(260921)
        groups = [np.flatnonzero(held.seed == seed) for seed in sorted(held.seed.unique())]
        draws = [np.concatenate([groups[j] for j in rng.integers(len(groups), size=len(groups))]) for _ in range(2000)]
        target = held.Q_reference.to_numpy()
        boots = {name: np.array([abs(spearmanr(held[name].to_numpy()[ix], target[ix]).statistic) for ix in draws]) for name in methods}
        for name in methods:
            observed = abs(spearmanr(held[name], target).statistic)
            residual = held[[name, "Q_reference"]] - held.groupby("control")[[name, "Q_reference"]].transform("mean")
            curve = held.groupby("control")[[name, "Q_reference"]].mean()
            low, high = np.nanquantile(boots[name], [.025, .975])
            diff = boots[name] - boots["q"]
            dlow, dhigh = np.nanquantile(diff, [.025, .975])
            all_metrics.append(dict(system=system, method=name, n=len(held), seeds=len(groups),
                evaluated_available=int(evaluation.sum()), abs_rho=observed, low=low, high=high,
                difference_vs_q=observed-abs(spearmanr(held.q, target).statistic), difference_low=dlow, difference_high=dhigh,
                control_mean_abs_rho=abs(spearmanr(curve[name], curve.Q_reference).statistic),
                within_control_abs_rho=abs(spearmanr(residual[name], residual.Q_reference).statistic)))
        # Display only: Q-informed sign is explicitly recorded and is never
        # used in feature fitting, evaluation, component selection or scores.
        signs = {name: (-1 if spearmanr(frame.loc[dev, name], frame.loc[dev, "Q_reference"], nan_policy="omit").statistic < 0 else 1)
                 for name in ("q", "mean_PC1", "distribution_PC1", "mean+z_PC1")}
        diagnostics["display_signs_from_development_Q"] = signs
        diagnostics["development_rows"] = int(dev.sum())
        diagnostics["common_eligible_evaluation_rows"] = int(eligible.sum())
        diagnostics["original_frozen_model_sha256"] = sha(frozen_model)
        frame.to_csv(OUT / f"inference-{system}.csv", index=False)
        np.savez_compressed(DATA / f"inference-{system}-summaries.npz", marginal=marginal,
            row_id=frame.row_id.to_numpy(dtype=str), spi_order=np.array(loaded[0][3]))
        all_diagnostics[system] = diagnostics
        print(f"inference {system}: {int(dev.sum())} development / {int(eligible.sum())} evaluation", flush=True)
    pd.DataFrame(all_metrics).to_csv(OUT / "inference-metrics.csv", index=False)
    write_json(OUT / "inference-provenance.json", finite_json(dict(diagnostics=all_diagnostics, sources=all_sources,
        config_sha256=sha(CONFIG), status="exploratory retrospective baseline comparison; original q reproduced")))
    transition_intervals()


def transition_intervals():
    """Descriptive finite-grid slopes, kept separate from rank association."""
    rows = []
    for system in ("Kuramoto", "CML2D"):
        frame = pd.read_csv(OUT / f"inference-{system}.csv").query("evaluation and common_eligible")
        names = ["Q_reference", "q", "mean_PC1", "distribution_PC1", "mean+z_PC1",
                 "z_standard_PC1", "mean_correlation", "mean_abs_correlation", "temporal_spectral_entropy"]
        for name in names:
            curve = frame.groupby("control")[name].mean()
            slope = abs(np.diff(curve.to_numpy()) / np.diff(curve.index.to_numpy()))
            i = int(np.argmax(slope))
            rows.append(dict(system=system, method=name, interval_start=curve.index[i], interval_end=curve.index[i+1]))
    pd.DataFrame(rows).to_csv(OUT / "inference-transition-intervals.csv", index=False)


def zenodo_analysis():
    from umap import UMAP
    bank_path = ROOT / "data/zenodo_7118947/features/pearson-unified-v3-seed1729.npz"
    with np.load(bank_path, allow_pickle=True) as a:
        order = a["spi_order"].astype(str).tolist()
        z = a["X"]
        original_paths = a["dataset_paths"].astype(str)
        sizes = (a["M"].astype(int), a["T"].astype(int))
        prior_sources = {Path(s["dataset_path"]).name: s for s in
                         json.loads(str(a["source_manifest_json"].item()))["entries"]}
    mpi_root = ROOT / "data/zenodo_7118947/runs/p90-zscore/zenodo-7118947-p90-zscore-seed1729"
    values, names, sources = [], [], []
    for path in original_paths:
        folder = mpi_root / Path(path).name
        original = prior_sources[folder.name]
        assert sha(folder / "spi_mpis.npz") == original["mpi_sha256"]
        assert sha(folder / "meta.json") == original["meta_sha256"]
        meta = json.loads((folder / "meta.json").read_text())
        assert meta["status"] == "complete" and meta["random_seed"] == 1729
        names.append(meta["dataset_name"])
        with np.load(folder / "spi_mpis.npz", allow_pickle=False) as a:
            assert set(a.files) == set(order)
            values.append(summarize(a, order))
        sources.append(dict(dataset=names[-1], mpi_sha256=sha(folder / "spi_mpis.npz")))
    marginal = np.array(values)
    arrays, metrics = {}, []
    for name, values in (("mean", marginal[:, :, 0]), ("distribution", marginal.reshape(len(marginal), -1)), ("z", z)):
        model, scores, _ = project_features(values, values, standard=name != "z")
        arrays[f"{name}_pca"] = scores[:, :2]
        arrays[f"{name}_scores"] = scores
        arrays[f"{name}_umap"] = UMAP(n_neighbors=30, min_dist=.1, random_state=260824, n_jobs=1).fit_transform(scores)
        metrics.append(dict(method=name, retained=len(model.transform.keep_indices),
            explained_variance=model.pca.explained_variance_ratio_.sum(),
            abs_PC1_M=spearmanr(scores[:, 0], sizes[0]).statistic.__abs__(),
            abs_PC1_T=spearmanr(scores[:, 0], sizes[1]).statistic.__abs__()))
        print(f"Zenodo {name}", flush=True)
    arrays.update(dataset=np.array(names), M=sizes[0], T=sizes[1])
    np.savez_compressed(OUT / "zenodo-projections.npz", **arrays)
    np.savez_compressed(DATA / "zenodo-summaries.npz", marginal=marginal, dataset=np.array(names), spi_order=np.array(order))
    pd.DataFrame(metrics).to_csv(OUT / "zenodo-descriptive.csv", index=False)
    write_json(OUT / "zenodo-provenance.json", dict(sources=sources, feature_bank_sha256=sha(bank_path),
        status="transductive exploration; no independent superiority criterion", config_sha256=sha(CONFIG)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["extract-proof", "proof", "toy", "inference", "zenodo", "intervals"])
    args = p.parse_args()
    with threadpool_limits(limits=4):
        {"extract-proof": extract_proof, "proof": proof_analysis, "toy": toy_analysis,
         "inference": inference_analysis, "zenodo": zenodo_analysis, "intervals": transition_intervals}[args.stage]()


if __name__ == "__main__":
    main()
