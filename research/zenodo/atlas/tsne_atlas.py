"""Corpus-atlas t-SNE (2026-10-21): one map, feature-space clusters, and the diagnostics behind both.

Settles the embedding and candidate clusters for the paper's corpus figure. Reads the canonical
seed-1729 `unified_ordered_v3` bank and its stored per-dataset MPIs; never runs pyspi. Clusters are
fitted in the full retained feature space and only displayed on the map.

Run: .venv/bin/python -m research.zenodo.atlas.tsne_atlas   (fits and caches; the notebook
zenodo_1053_tsne_261021.ipynb reads the cache, draws the figures and writes the explorer)
"""
from __future__ import annotations

import argparse
import base64
from hashlib import sha256
from importlib.metadata import version
import io
from itertools import combinations
import json
from pathlib import Path
import re
import time

from matplotlib import patheffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import procrustes
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, silhouette_samples
import seaborn as sns
from threadpoolctl import threadpool_limits
import yaml

from mtsspi.corpus_visualization import scale_timeseries
from mtsspi.spi_spi_contract import _pearson_corr_matrix, _upper_symmetrized, build_unified_feature_values
from research.zenodo.atlas.explore_zenodo_geometry import ROOT, density_overlay, historical_input, paper_style
from research.zenodo.atlas.explore_zenodo_v2 import corrected_tags, dataset_origin, distance_neighbors, marker_radii, recall
from research.zenodo.shared.atlas_analysis import fit_atlas_transform, load_unified_artifact

CONFIG = ROOT / "research/zenodo/atlas/configs/analysis/zenodo-tsne-261021.yaml"

# pyspi categories of the p90 catalogue (configs/pyspi/benchmarked_p90.yaml), keyed by SPI-name prefix.
SPI_CATEGORY = {
    **dict.fromkeys(["cov", "cov-sq", "prec", "prec-sq", "spearmanr", "spearmanr-sq", "kendalltau", "kendalltau-sq",
                     "corr", "xcorr", "xcorr-sq"], "basic"),
    **dict.fromkeys(["pdist", "dcorr", "dtw", "xpdist", "lcss", "bary", "bary-sq", "gwtau"], "distance"),
    **dict.fromkeys(["cds", "reci"], "causal"),
    **dict.fromkeys(["je", "ce", "cce", "xme", "di", "si", "mi", "tlmi", "te", "gc", "phi"], "infotheory"),
    **dict.fromkeys(["phase", "cohmag", "icoh", "psi", "plv", "pli", "wpli", "dspli", "dswpli", "ppc", "dtf", "dcoh",
                     "pdcoh", "gpdcoh", "ddtf", "gd", "sgc"], "spectral"),
    **dict.fromkeys(["lmfit", "coint", "pec", "ids"], "misc"),
}

# SPI pairs whose agreement the paper's case studies read as named character axes.
CASE_AXES = {
    "r–ρ (linear vs monotone)": ("cov_EmpiricalCovariance", "spearmanr"),
    "r–MI (linear vs general)": ("cov_EmpiricalCovariance", "mi_kernel_W-0.25"),  # Kraskov is undefined for 40% of rows
    "r–max xcorr (zero lag vs best lag)": ("cov_EmpiricalCovariance", "xcorr_max_sig-True"),
    "r–coherence (time vs frequency)": ("cov_EmpiricalCovariance", "cohmag_multitaper_mean_fs-1_fmin-0_fmax-0-5"),
    "Euclid–DTW (aligned vs warped shape)": ("pdist_euclidean", "dtw"),
}

# Ordered (pattern, source family) rules; first match wins. Labels are archive provenance, not dynamics.
SOURCE_RULES = [
    (r"^hcp_rsfMRI", "HCP rs-fMRI"), (r"^hcp_tfMRI", "HCP task fMRI"), (r"^UCLA_neuro", "UCLA fMRI"),
    (r"^mousefMRI", "Mouse fMRI"), (r"^sim\d+$", "NetSim fMRI"), (r"^ForEx", "ForEx"), (r"^DJIA", "DJIA"),
    (r"^SNP500", "S&P 500"), (r"^\d{4}-\d\d-\d\d", "Seismogram"), (r"^epidemic", "Epidemic incidence"),
    (r"^TestCLIM", "TestCLIM"), (r"^TestWEATH", "TestWEATH"), (r"^([A-Za-z0-9-]+)_C-", None),
    (r"^var_", "VAR"), (r"^vma_", "VMA"), (r"^noise_", "IID noise"), (r"^brownian", "Brownian motion"),
    (r"^ornstein", "Ornstein–Uhlenbeck"), (r"^kuramoto", "Kuramoto–Sakaguchi"), (r"^oscillator_sync", "Oscillator sync"),
    (r"^oscillator_legion", "LEGION"), (r"^oscillator_hysteresis", "Oscillator hysteresis"), (r"^vanderpol", "Van der Pol"),
    (r"^wilson-cowan", "Wilson–Cowan"), (r"^wave-1D", "Wave 1D"), (r"^wave-2D", "Wave 2D"),
    (r"^(spatiotemporal_intermittency_ii?|chaotic_brownian_motion_of_defect|defect_turbulence|pattern_selection|"
     r"spatiotemporal_chaos|traveling_wave)", "CML"),
]


def spi_category(name: str) -> str:
    return SPI_CATEGORY[name.split("_")[0]]


def source_family(name: str) -> str:
    for pattern, label in SOURCE_RULES:
        match = re.match(pattern, name)
        if match:
            return label or match.group(1)
    raise ValueError(f"no source rule for {name}")


def parent_key(name: str) -> str:
    """Conservative underlying-recording key: all M/T variants of one generator configuration, price and
    log-return versions of one market window, region subsets of one subject, and all epidemic chunks share
    a key. UEA instances, NetSim and CLIM/WEATH samples are separate draws."""
    rules = [(r"[_-]M-?\d+[_-]T-?\d+$", ""), (r"^(ForEx|DJIA|SNP500)_N-\d+_P-\d+_(\d{4}-\d\d-\d\d)(-lr)?$", r"\1_\2"),
             (r"^(hcp_(?:rs|tf)fMRI_S\d+)_R-.*$", r"\1"), (r"^(UCLA_neuro_[A-Z]+_S-\d+)_R-.*$", r"\1"),
             (r"^(mousefMRI_S-\d+)_R-.*$", r"\1"), (r"^epidemic_incidence_.*$", "epidemic_incidence")]
    for pattern, replacement in rules:
        name = re.sub(pattern, replacement, name)
    return name


def nested_parents(names, arrays):
    """Merge keys of arrays that are exact copies or channel/time prefixes of one another (union-find)."""
    keys = {n: parent_key(n) for n in names}
    root = {k: k for k in keys.values()}

    def find(k):
        while root[k] != k:
            root[k] = root[root[k]]
            k = root[k]
        return k

    order = sorted(names, key=lambda n: arrays[n].size)
    for i, small in enumerate(order):
        a = arrays[small]
        for large in order[i + 1:]:
            b = arrays[large]
            if a.shape[0] <= b.shape[0] and a.shape[1] <= b.shape[1] and find(keys[small]) != find(keys[large]):
                if np.array_equal(a, b[:a.shape[0], :a.shape[1]]):
                    root[find(keys[small])] = find(keys[large])
    return np.array([find(keys[n]) for n in names])


def load_corpus(config):
    """Bank, contract checks and corrected archive metadata, row-aligned."""
    artifact = load_unified_artifact(ROOT / config["features"])  # raises unless unified_ordered_v3 and K choose 2
    if str(artifact["diagonal_policy"]) != "excluded" or str(artifact["ordered_entry_order"]) != "C-row-major-i-ne-j":
        raise ValueError("bank does not declare all M(M-1) ordered off-diagonal entries")
    names = artifact["y"].astype(str)
    rules = yaml.safe_load((ROOT / config["label_corrections"]).read_text())["add_tags"]
    with np.load(ROOT / config["database"], allow_pickle=False) as archive:
        source_tags = {str(n): json.loads(str(t)) for n, t in zip(archive["__dataset_names__"], archive["__labels_json__"])}
        arrays = {n: archive[n] for n in names}
    tags = [corrected_tags(n, source_tags[n], rules) for n in names]
    meta = pd.DataFrame({"row": np.arange(len(names)), "dataset": names,
                         "M": [arrays[n].shape[0] for n in names], "T": [arrays[n].shape[1] for n in names],
                         "origin": [dataset_origin(t) for t in tags], "source": [source_family(n) for n in names],
                         "parent": nested_parents(list(names), arrays), "tags": [json.dumps(t) for t in tags]})
    np.testing.assert_array_equal(meta.M, artifact["M"].astype(int))
    spis = [str(s) for s in artifact["spi_order"]]
    pairs = np.array(list(combinations(range(len(spis)), 2)))
    return {"raw": artifact["X"].astype(np.float64), "names": names, "spis": spis, "pairs": pairs,
            "directed": artifact["directed_flags"].astype(bool), "meta": meta, "arrays": arrays}


def verify_contract(corpus, runs_dir: Path, check_every: int = 50):
    """Recompute z from every stored seeded MPI file and compare with the bank row by row.

    Also scores every `check_every`-th row under the upper-triangle symmetrised alternative, to show the
    comparison can tell the two contracts apart."""
    directories = {json.loads((d / "meta.json").read_text())["dataset_name"]: d for d in sorted(runs_dir.iterdir()) if d.is_dir()}
    worst, mismatched, alternative = 0., 0, []
    for i, name in enumerate(corpus["names"]):
        with np.load(directories[name] / "spi_mpis.npz", allow_pickle=False) as stored:
            mpis = {k: stored[k] for k in stored.files}
        z, _, _ = build_unified_feature_values(mpis, corpus["spis"])
        bank = corpus["raw"][i].astype(np.float32)
        finite = np.isfinite(bank) & np.isfinite(z)
        mismatched += int((np.isfinite(bank) != np.isfinite(z)).sum())
        worst = max(worst, float(np.abs(bank[finite] - z[finite]).max()))
        if i % check_every == 0:
            vectors = np.vstack([_upper_symmetrized(np.asarray(mpis[s], float)) for s in corpus["spis"]])
            sym = _pearson_corr_matrix(vectors)[np.triu_indices(len(corpus["spis"]), 1)]
            ok = np.isfinite(sym) & np.isfinite(bank)
            alternative.append(float(np.abs(sym[ok] - bank[ok]).max()))
    asymmetric = np.mean([not np.allclose(m, m.T, equal_nan=True) for m in
                          (np.asarray(mpis[s], float) for s, d in zip(corpus["spis"], corpus["directed"]) if d)])
    return {"rows": len(corpus["names"]), "max_abs_difference": worst, "finite_pattern_mismatches": mismatched,
            "upper_symmetrised_max_difference_min": min(alternative), "upper_symmetrised_rows_checked": len(alternative),
            "asymmetric_directed_mpis_last_row": float(asymmetric), "entries_per_mpi": "M(M-1)"}


def preprocess(raw, variant: str):
    """Return (matrix, description). Every variant keeps all 1,053 rows."""
    if variant == "historical":
        values, rows, columns = historical_input(raw)
        if len(rows) != len(raw):
            raise ValueError("historical recipe dropped rows")
        return values, {"features": values.shape[1], "imputed_fraction": float(np.mean(~np.isfinite(raw[:, columns])))}
    settings = {"primary": (.95, False), "complete": (1., False), "valid90": (.90, False), "standardised": (.95, True)}
    valid, standardise = settings[variant]
    transform = fit_atlas_transform(raw, minimum_valid_fraction=valid)
    values = transform.transform(raw)
    if standardise:
        values = values / values.std(axis=0)
    missing = ~np.isfinite(raw[:, transform.keep_indices])
    return values, {"features": values.shape[1], "imputed_fraction": float(missing.mean()),
                    "rows_over_20pct_imputed": int((missing.mean(axis=1) > .2).sum()), "keep_indices": transform.keep_indices}


def exact_scores(values):
    """Unwhitened full-rank PCA; Euclidean distances between rows equal those of `values`."""
    pca = PCA(svd_solver="full", whiten=False)
    scores = pca.fit_transform(values)[:, :len(values) - 1]
    return scores, pca.explained_variance_ratio_[:len(values) - 1]


class ExaggeratedTSNE(TSNE):
    """scikit-learn t-SNE with a constant exaggeration after the early-exaggeration phase.

    Exaggeration rho multiplies the attractive term for the remaining iterations (Linderman & Steinerberger
    2019; Kobak & Berens 2019; Böhm, Berens & Kobak 2022, attraction-repulsion spectrum). The early phase is
    unchanged: P is scaled by rho and the early factor divided by rho. rho = 1 is stock scikit-learn.
    Relies on TSNE._tsne(P, ...), checked against scikit-learn 1.9.
    """

    exaggeration = 1.

    def _tsne(self, P, *args, **kwargs):
        if self.exaggeration == 1:
            return super()._tsne(P, *args, **kwargs)
        early = self.early_exaggeration
        self.early_exaggeration = early / self.exaggeration
        try:
            return super()._tsne(P * self.exaggeration, *args, **kwargs)
        finally:
            self.early_exaggeration = early


def tsne(data, *, perplexity=30., exaggeration=1., init="pca", seed=1729, max_iter=1500, threads=8):
    model = ExaggeratedTSNE(n_components=2, perplexity=perplexity, early_exaggeration=12., learning_rate="auto",
                            init=init, max_iter=max_iter, random_state=seed, n_jobs=threads, method="barnes_hut", angle=.5)
    model.exaggeration = float(exaggeration)
    return model.fit_transform(data)


def assigned_silhouette(distances, labels):
    covered = labels >= 0
    if len(np.unique(labels[covered])) < 2:
        return np.nan
    return float(silhouette_samples(distances[np.ix_(covered, covered)], labels[covered], metric="precomputed").mean())


def map_clusters(xy, min_cluster_size=10, min_samples=5):
    """Descriptive 2-D regions: what a reader would call clusters on this map."""
    return HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, copy=True).fit_predict(xy)


def map_metrics(xy, reference, partition, settings):
    """Neighbour preservation, and visual versus full-space separation, for one map."""
    d2 = squareform(pdist(xy))
    nn2 = {k: distance_neighbors(d2, k) for k in settings["knn"]}
    regions = map_clusters(xy, **settings["map_hdbscan"])
    result = {f"recall{k}_full": recall(reference["nn"][k], nn2[k]) for k in settings["knn"]}
    result[f"recall15_complete"] = recall(reference["nn_complete"], nn2[15])
    result.update({"map_regions": int(regions.max() + 1), "map_coverage": float((regions >= 0).mean()),
                   "regions_sil_2d": assigned_silhouette(d2, regions),
                   "regions_sil_full": assigned_silhouette(reference["d"], regions),
                   "partition_sil_2d": assigned_silhouette(d2, partition),
                   "partition_nn15_same_2d": float(np.mean(partition[nn2[15]] == partition[:, None]))})
    return result, regions, nn2[15]


def seed_stability(maps, neighbours, regions):
    pairs = list(combinations(range(len(maps)), 2))
    return {"seed_nn15_overlap": float(np.mean([recall(neighbours[a], neighbours[b]) for a, b in pairs])),
            "seed_procrustes_disparity": float(np.mean([procrustes(maps[a], maps[b])[2] for a, b in pairs])),
            "seed_regions_ari": float(np.mean([adjusted_rand_score(regions[a], regions[b]) for a, b in pairs]))}


def ward(scores, k):
    return fcluster(linkage(scores, method="ward"), k, criterion="maxclust") - 1


def kmeans(scores, k, seed=1729):
    return KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(scores)


def grouped_jaccard(scores, labels, fit, parents, *, replicates, fraction, seed):
    """Hennig (2007) cluster-wise stability with subsets drawn by parent recording, not by row.

    Each replicate keeps a random `fraction` of parent keys, refits on those rows and records, for every
    original cluster, the best Jaccard match among the refitted clusters (restricted to retained rows)."""
    rng = np.random.default_rng(seed)
    keys = np.unique(parents)
    clusters = np.unique(labels[labels >= 0])
    values = np.full((replicates, len(clusters)), np.nan)
    for b in range(replicates):
        rows = np.flatnonzero(np.isin(parents, rng.choice(keys, size=round(fraction * len(keys)), replace=False)))
        refit = fit(scores[rows])
        for j, c in enumerate(clusters):
            original = set(rows[labels[rows] == c])
            if not original:
                continue
            values[b, j] = max(len(original & set(rows[refit == s])) / len(original | set(rows[refit == s]))
                               for s in np.unique(refit[refit >= 0]))
    return pd.DataFrame({"cluster": clusters, "jaccard_mean": np.nanmean(values, axis=0),
                         "dissolved_fraction": np.nanmean(values < .5, axis=0)})


def cluster_table(labels, distances, meta, jaccard):
    sil = silhouette_samples(distances, labels, metric="precomputed")
    rows = []
    for c in np.unique(labels):
        members = meta[labels == c]
        idx = members.row.to_numpy()
        medoid = idx[np.argmin(distances[np.ix_(idx, idx)].sum(axis=1))]
        origins = members.origin.value_counts()
        sources = members.source.value_counts()
        rows.append({"cluster": int(c), "size": len(idx), "parents": members.parent.nunique(),
                     "silhouette_full": float(sil[idx].mean()), "medoid": int(medoid),
                     "real": int(origins.get("real", 0)), "synthetic": int(origins.get("synthetic", 0)),
                     "sources": len(sources), "source_mix": "; ".join(f"{s} {n}" for s, n in sources.items()),
                     "M_range": f"{members['M'].min()}–{members['M'].max()}", "T_range": f"{members['T'].min()}–{members['T'].max()}"})
    return pd.DataFrame(rows).merge(jaccard, on="cluster")


def classify(row, dominant=.8):
    """Composition (by source family) and origin, as two separate axes."""
    composition = "homogeneous" if row["sources"] == 1 else "heterogeneous"
    origin = "mixed" if row["real"] and row["synthetic"] else "real" if row["real"] else "synthetic"
    return composition, origin


def contrasts(raw, keep, rows, pairs, spis):
    """Per-feature standardised difference, cluster versus rest, on observed values only (no imputation)."""
    values = raw[:, keep]
    inside = np.zeros(len(raw), bool)
    inside[rows] = True
    with np.errstate(invalid="ignore"):
        delta = (np.nanmean(values[inside], axis=0) - np.nanmean(values[~inside], axis=0)) / np.nanstd(values, axis=0)
    a, b = pairs[keep, 0], pairs[keep, 1]
    frame = pd.DataFrame({"spi_a": np.array(spis)[a], "spi_b": np.array(spis)[b], "effect": delta,
                          "cluster_median": np.nanmedian(values[inside], axis=0),
                          "corpus_median": np.nanmedian(values, axis=0)})
    cats = sorted(set(SPI_CATEGORY.values()))
    frame["family_pair"] = [" × ".join(sorted((spi_category(x), spi_category(y)), key=cats.index))
                            for x, y in zip(frame.spi_a, frame.spi_b)]
    return frame


def representatives(rows, distances, meta, n=8):
    """Medoid first, then round-robin over source families, nearest-to-medoid within each family."""
    rows = np.asarray(rows)
    medoid = rows[np.argmin(distances[np.ix_(rows, rows)].sum(axis=1))]
    queues = {s: sorted(rows[meta.source.to_numpy()[rows] == s], key=lambda r: distances[medoid, r])
              for s in meta.source.to_numpy()[rows][np.argsort(distances[medoid, rows])]}
    chosen = [medoid]
    while len(chosen) < min(n, len(rows)):
        for queue in queues.values():
            queue[:] = [r for r in queue if r not in chosen]
            if queue and len(chosen) < n:
                chosen.append(queue.pop(0))
    return chosen


def family_summary(frame, top=3):
    """Signed mean effect per SPI-category block, strongest first in each direction."""
    blocks = frame.groupby("family_pair").effect.mean().sort_values()
    return blocks.tail(top)[::-1], blocks.head(top)


def raw_descriptors(values):
    """Transparent sample-scale checks on one process x time array, used to test cluster readings."""
    x = np.asarray(values, float)
    z = (x - x.mean(axis=1, keepdims=True)) / np.where(x.std(axis=1, keepdims=True) > 0, x.std(axis=1, keepdims=True), 1)
    with np.errstate(all="ignore"):
        r = np.corrcoef(x)[~np.eye(len(x), dtype=bool)]
    r = r[np.isfinite(r)]
    strong = np.abs(r) >= np.quantile(np.abs(r), .75) if r.size else np.zeros(0, bool)
    t = np.arange(x.shape[1])
    trend = [np.corrcoef(t, row)[0, 1] ** 2 if row.std() > 0 else 0. for row in x]
    lag1 = [np.corrcoef(row[:-1], row[1:])[0, 1] if row[:-1].std() > 0 and row[1:].std() > 0 else np.nan for row in x]
    eigen = np.linalg.eigvalsh(np.cov(z)) if len(x) > 1 else np.ones(1)
    return {"lag1": float(np.nanmedian(lag1)), "trend_r2": float(np.median(trend)),
            "negative_share_strong": float(np.mean(r[strong] < 0)) if r.size else np.nan,
            "pc1_share": float(eigen[-1] / eigen.sum()), "repeat_fraction": float(np.mean(np.diff(x, axis=1) == 0)),
            "max_robust_z": float(np.max(np.abs(x - np.median(x, axis=1, keepdims=True))
                                         / np.maximum(1.4826 * np.median(np.abs(x - np.median(x, axis=1, keepdims=True)), axis=1, keepdims=True), 1e-12)))}


def case_axes(raw, spis):
    """Columns of the named case-study SPI pairs (raw z, NaN where undefined)."""
    index = {s: i for i, s in enumerate(spis)}
    k = len(spis)
    out = {}
    for label, (a, b) in CASE_AXES.items():
        i, j = sorted((index[a], index[b]))
        out[label] = raw[:, i * k - i * (i + 1) // 2 + (j - i - 1)]
    return pd.DataFrame(out)


ORIGIN_FILL = {"real": "0.2", "synthetic": "0.62"}


def notebook_style():
    """docs/figure-style.md without TeX: dataset names carry underscores, so use bundled Computer Modern."""
    import matplotlib
    from matplotlib import font_manager
    paper_style()
    font_manager.fontManager.addfont(str(Path(matplotlib.get_data_path()) / "fonts/ttf/cmr10.ttf"))
    plt.rcParams.update({"text.usetex": False, "font.family": ["cmr10", "DejaVu Serif"], "axes.formatter.use_mathtext": True,
                         "axes.unicode_minus": False, "figure.constrained_layout.use": True})


def plot_atlas(ax, xy, meta, labels=None, letters=None, *, shade=True, annotate=True, size=1.1, seed=1729):
    """Paper-style map: grey by origin (hollow if unlabelled), area by M, faint KDE around lettered clusters."""
    letters = letters or {}
    if shade and labels is not None and letters:
        density_overlay(ax, xy, np.where(np.isin(labels, list(letters)), labels, -1))
    radii = marker_radii(meta["M"].to_numpy(), meta["M"].min(), meta["M"].max())
    order = np.random.default_rng(seed).permutation(len(xy))  # no origin drawn systematically on top
    origin = meta["origin"].to_numpy()[order]
    fill = [ORIGIN_FILL.get(o, "none") for o in origin]
    edge = ["white" if o in ORIGIN_FILL else "0.3" for o in origin]
    ax.scatter(*xy[order].T, s=size * radii[order] ** 2, c=fill, edgecolors=edge, linewidths=.3, zorder=2)
    if annotate and labels is not None:
        for cluster, letter in letters.items():
            points = xy[labels == cluster]
            ax.annotate(letter, (points[:, 0].mean(), points[:, 1].max()), xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10, zorder=3,
                        path_effects=[patheffects.withStroke(linewidth=2.5, foreground="white")])
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    middle, radius = (lo + hi) / 2, max(float(np.max(hi - lo)), 1e-6) * .55
    ax.set(xlim=(middle[0] - radius, middle[0] + radius), ylim=(middle[1] - radius, middle[1] + radius), xticks=[], yticks=[])
    ax.set_aspect("equal", adjustable="box")
    ax.spines[["top", "right"]].set_visible(True)


def heatmap(ax, values, title=None):
    """Same per-process robust scaling and icefire map as the v1 heatmap assets."""
    scaled, limit = scale_timeseries(values, "robust")
    ax.imshow(scaled, aspect="auto", interpolation="nearest", cmap=sns.color_palette("icefire", as_cmap=True),
              vmin=-limit, vmax=limit, origin="lower", rasterized=True)
    ax.set(xticks=[], yticks=[])
    if title:
        ax.set_title(title, fontsize=7, loc="left")


def hover_text(meta, labels, letters):
    return [f"{r.dataset}<br>{r.source} · {r.origin}<br>M={r.M}, T={r.T}<br>cluster {labels[r.row]}"
            f"{' (' + letters[labels[r.row]] + ')' if labels[r.row] in letters else ''}" for r in meta.itertuples()]


def hover_figure(xy, meta, labels, letters, title):
    """Static plotly map: hover shows name, source, origin, M, T and cluster without a kernel."""
    import plotly.graph_objects as go
    text = np.array(hover_text(meta, labels, letters))
    figure = go.Figure()
    for origin, colour in [("synthetic", "#9e9e9e"), ("real", "#333333")]:
        mask = (meta.origin == origin).to_numpy()
        figure.add_trace(go.Scattergl(x=xy[mask, 0], y=xy[mask, 1], mode="markers", name=origin, text=text[mask],
                                      hovertemplate="%{text}<extra></extra>",
                                      marker={"color": colour, "size": 4 + meta.M[mask] / 5, "line": {"color": "white", "width": .5}}))
    figure.update_layout(title=title, template="simple_white", width=700, height=640, xaxis={"visible": False},
                         yaxis={"visible": False, "scaleanchor": "x"})
    return figure


def inspection_widget(xy, meta, labels, letters, arrays):
    """Live-kernel browser: click a point to show its metadata and robust-scaled heatmap."""
    import ipywidgets as widgets
    import plotly.graph_objects as go
    from IPython.display import clear_output, display
    figure = go.FigureWidget(hover_figure(xy, meta, labels, letters, "Click a point (live kernel required)"))
    output = widgets.Output(layout={"width": "700px"})
    rows = {origin: meta.row[(meta.origin == origin).to_numpy()].to_numpy() for origin in ["synthetic", "real"]}

    def show(row):
        r = meta.iloc[row]
        with output:
            clear_output(wait=True)
            print(f"{r.dataset} | {r.source} | {r.origin} | M={r.M} T={r.T} | cluster {labels[row]} | parent {r.parent}")
            fig, ax = plt.subplots(figsize=(6.5, 1.8), constrained_layout=True)
            heatmap(ax, arrays[r.dataset])
            display(fig)
            plt.close(fig)

    for trace, origin in zip(figure.data, ["synthetic", "real"]):
        trace.on_click(lambda _t, points, _s, origin=origin: points.point_inds and show(int(rows[origin][points.point_inds[0]])))
    return widgets.VBox((figure, output))


def best_jaccard(labels, other, cluster):
    members = set(np.flatnonzero(labels == cluster))
    return max(len(members & set(np.flatnonzero(other == o))) / len(members | set(np.flatnonzero(other == o)))
               for o in np.unique(other[other >= 0]))


def candidate_table(candidates, labels, clusters, arrays, meta, xy, corpus, keep):
    """One row per lettered cluster: composition, separation, stability, robustness and case-axis reading."""
    distances = arrays["full_distances"]
    nn5 = distance_neighbors(distances, 5)
    source = meta.source.to_numpy()
    silhouette_2d = silhouette_samples(squareform(pdist(xy)), labels, metric="precomputed")
    axes = case_axes(corpus["raw"], corpus["spis"])
    rows, readings = [], {}
    for cluster, (letter, title) in candidates.items():
        info = clusters.set_index("cluster").loc[cluster]
        members = np.flatnonzero(labels == cluster)
        composition, origin = classify(info)
        frame = contrasts(corpus["raw"], keep, members, corpus["pairs"], corpus["spis"])
        up, down = family_summary(frame, top=2)
        readings[cluster] = {"frame": frame, "axes": axes.iloc[members].median(), "corpus_axes": axes.median()}
        rows.append({"letter": letter, "cluster": cluster, "title": title, "composition": composition, "origin": origin,
                     "size": len(members), "parents": int(info.parents), "sources": info.source_mix,
                     "silhouette_full": info.silhouette_full, "silhouette_map": float(silhouette_2d[members].mean()),
                     "jaccard": info.jaccard_mean, "jaccard_kmeans40": best_jaccard(labels, arrays["partition__kmeans40"], cluster),
                     "jaccard_complete_features": best_jaccard(labels, arrays["partition__ward_complete"], cluster),
                     "cross_source_nn5": float(np.mean(source[nn5[members]] != source[members][:, None])),
                     "M": info.M_range, "T": info.T_range,
                     "families_up": "; ".join(f"{k} {v:+.2f}" for k, v in up.items()),
                     "families_down": "; ".join(f"{k} {v:+.2f}" for k, v in down.items())})
    return pd.DataFrame(rows), readings


def gallery_payload(meta, labels, clusters, table, readings, maps, default_map, methods):
    radii = marker_radii(meta["M"].to_numpy(), meta["M"].min(), meta["M"].max())
    legend = sorted({int(meta["M"].min()), int(meta["M"].median()), int(meta["M"].max())})
    info = clusters.set_index("cluster")
    rows = [{"row": int(r.row), "dataset": r.dataset, "M": int(r.M), "T": int(r.T), "origin": r.origin, "source": r.source,
             "parent": r.parent, "tags": json.loads(r.tags), "cluster": int(labels[r.row]), "marker_radius": float(radii[r.row])}
            for r in meta.itertuples()]
    candidates = []
    for c in table.itertuples():
        reading = readings[c.cluster]
        candidates.append({"letter": c.letter, "cluster": int(c.cluster), "title": c.title, "composition": c.composition,
                           "origin": c.origin, "size": int(c.size), "parents": int(c.parents), "sources": c.sources,
                           "silhouette": float(c.silhouette_full), "jaccard": float(c.jaccard), "note": c.note,
                           "families": f"higher agreement: {c.families_up}; lower: {c.families_down}",
                           "axes": [{"axis": a, "cluster": None if np.isnan(reading["axes"][a]) else float(reading["axes"][a]),
                                     "corpus": float(reading["corpus_axes"][a])} for a in CASE_AXES]})
    return {"rows": rows, "maps": maps, "default_map": default_map, "candidates": candidates, "methods": methods,
            "clusters": {str(c): {"size": int(i["size"]), "parents": int(i.parents), "silhouette": float(i.silhouette_full),
                                  "jaccard": float(i.jaccard_mean), "medoid": int(i.medoid), "source_mix": i.source_mix}
                         for c, i in info.iterrows()},
            "size_legend": [{"M": m, "radius": float(marker_radii(m, meta["M"].min(), meta["M"].max()))} for m in legend],
            "origin_styles": {"real": {"fill": "#333333", "stroke": "#ffffff"}, "synthetic": {"fill": "#9e9e9e", "stroke": "#ffffff"},
                              "unlabelled": {"fill": "none", "stroke": "#555555"}},
            "origin_counts": meta.origin.value_counts().to_dict()}


def thumbnail(png: Path, width=240, height=96, colours=64):
    """Browsing preview: at most width x height, 64-colour palette PNG; native assets stay linked."""
    image = Image.open(png).convert("RGB")
    image = image.resize((min(width, image.width), min(height, image.height)), Image.Resampling.BOX).quantize(colours)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def render_gallery(out: Path, payload: dict, heatmaps: Path):
    """Self-contained explorer: embedded thumbnails, links to native v1 heatmap assets."""
    index = pd.read_csv(heatmaps / "heatmap-index.csv")
    np.testing.assert_array_equal(index.dataset, [r["dataset"] for r in payload["rows"]])
    relative = Path("..") / heatmaps.name
    for row, (_, item) in zip(payload["rows"], index.iterrows()):
        row["thumb"] = thumbnail(heatmaps / item.png)
        row["png"], row["svg"] = (relative / item.png).as_posix(), (relative / item.svg).as_posix()
        row["display_limit"] = float(item.display_limit)
    here = Path(__file__).parent
    page = (here / "zenodo_gallery_v3.html").read_text().replace("__KDE_SCRIPT__", (here / "zenodo_kde.js").read_text())
    page = page.replace("__CORPUS_DATA__", json.dumps(payload, allow_nan=False).replace("</", "<\\/"))
    (out / "index.html").write_text(page)
    return out / "index.html"


def subsample_map_stability(data, params, parents, full_map, *, replicates, fraction, seed, threads):
    """Refit the map on parent-grouped subsets; compare each subset map's 15-NN with the full map's."""
    rng = np.random.default_rng(seed)
    keys = np.unique(parents)
    full = squareform(pdist(full_map))
    overlaps = []
    for b in range(replicates):
        rows = np.flatnonzero(np.isin(parents, rng.choice(keys, size=round(fraction * len(keys)), replace=False)))
        xy = tsne(data[rows], seed=seed + b, threads=threads, **params)
        overlaps.append(recall(distance_neighbors(full[np.ix_(rows, rows)], 15), distance_neighbors(squareform(pdist(xy)), 15)))
    return float(np.mean(overlaps)), float(np.std(overlaps))


def embedding_grid(config):
    """Declared grid: one-factor departures from the baseline, plus perplexity x exaggeration."""
    grid = [dict(group="perplexity × exaggeration", input="primary", perplexity=p, exaggeration=e, init="pca")
            for p in config["perplexities"] for e in config["exaggerations"]]
    grid += [dict(group="input dimension", input=f"pca{k}", perplexity=30, exaggeration=1, init="pca") for k in config["pca_dimensions"]]
    grid += [dict(group="initialisation", input="primary", perplexity=p, exaggeration=e, init="random")
             for p, e in [(30, 1), (30, 2)]]
    grid += [dict(group="preprocessing", input=v, perplexity=30, exaggeration=1, init="pca") for v in config["variants"]]
    for item in grid:
        item["key"] = f"{item['input']}-p{item['perplexity']}-x{item['exaggeration']}-{item['init']}"
    return grid


def compute(config, out: Path):
    """Fit everything the notebook reports; cache arrays and tables under `out`."""
    tick = time.monotonic()
    corpus = load_corpus(config)
    raw, meta = corpus["raw"], corpus["meta"]
    inputs, info, variance = {}, {}, {}
    for variant in ["primary", *config["variants"]]:
        values, info[variant] = preprocess(raw, variant)
        inputs[variant], variance[variant] = exact_scores(values)
        if variant == "primary":
            d = squareform(pdist(inputs[variant]))
            np.testing.assert_allclose(d, squareform(pdist(values)), atol=1e-8)  # full rank = all retained features
            keep = info[variant].pop("keep_indices")
        info[variant].pop("keep_indices", None)
    for k in config["pca_dimensions"]:
        inputs[f"pca{k}"] = inputs["primary"][:, :k]
    d_complete = squareform(pdist(inputs["complete"]))
    reference = {"d": d, "nn": {k: distance_neighbors(d, k) for k in config["knn"]}, "nn_complete": distance_neighbors(d_complete, 15)}
    for name in list(info):
        nn = distance_neighbors(squareform(pdist(inputs[name])), 15)
        info[name].update({"nn15_vs_primary": recall(reference["nn"][15], nn), "variance_pc2": float(variance[name][:2].sum())})
    partitions = {f"ward{k}": ward(inputs["primary"], k) for k in config["ward_k"]}
    partitions.update({f"kmeans{k}": kmeans(inputs["primary"], k) for k in config["ward_k"]})
    primary_partition = partitions[config["primary_partition"]]
    print(f"Inputs and partitions ready in {time.monotonic() - tick:.0f} s", flush=True)

    arrays, records = {}, []
    pc = inputs["primary"][:, :2]
    metrics, regions, _ = map_metrics(pc, reference, primary_partition, config)
    records.append({"key": "pc1-pc2", "group": "linear reference", "input": "primary", **metrics})
    arrays["map__pc1-pc2"], arrays["regions__pc1-pc2"] = pc, regions
    for item in embedding_grid(config):
        maps, neighbours, labelled, rows = [], [], [], []
        for seed in config["seeds"]:
            xy = tsne(inputs[item["input"]], perplexity=item["perplexity"], exaggeration=item["exaggeration"],
                      init=item["init"], seed=seed, max_iter=config["max_iter"], threads=config["threads"])
            metrics, reg, nn15 = map_metrics(xy, reference, primary_partition, config)
            maps.append(xy); neighbours.append(nn15); labelled.append(reg); rows.append(metrics)
        frame = pd.DataFrame(rows)
        summary = {**{k: v for k, v in item.items()}, **frame.iloc[0].to_dict(),
                   "recall15_full_seed_sd": float(frame.recall15_full.std()), **seed_stability(maps, neighbours, labelled)}
        records.append(summary)
        arrays[f"map__{item['key']}"] = np.stack(maps)
        arrays[f"regions__{item['key']}"] = np.stack(labelled)
        print(f"{item['key']}: recall15 {summary['recall15_full']:.3f} · regions {summary['map_regions']} "
              f"· {time.monotonic() - tick:.0f} s", flush=True)
    grid = pd.DataFrame(records)

    stability = []
    for key in config["subsample_maps"]:
        item = next(i for i in embedding_grid(config) if i["key"] == key)
        params = {k: item[k] for k in ["perplexity", "exaggeration", "init"]} | {"max_iter": config["max_iter"]}
        mean, sd = subsample_map_stability(inputs[item["input"]], params, meta.parent.to_numpy(), arrays[f"map__{key}"][0],
                                           replicates=config["subsample"]["map_replicates"], fraction=config["subsample"]["fraction"],
                                           seed=config["seed"], threads=config["threads"])
        stability.append({"key": key, "subsample_nn15_vs_full_map": mean, "sd": sd})
    grid = grid.merge(pd.DataFrame(stability), on="key", how="left")

    jaccard = {}
    for name, labels in partitions.items():
        k = int(re.sub(r"\D", "", name))
        fit = (lambda s, k=k: ward(s, k)) if name.startswith("ward") else (lambda s, k=k: kmeans(s, k))
        jaccard[name] = cluster_table(labels, d, meta, grouped_jaccard(
            inputs["primary"], labels, fit, meta.parent.to_numpy(), replicates=config["subsample"]["replicates"],
            fraction=config["subsample"]["fraction"], seed=config["seed"]))
        jaccard[name].insert(0, "partition", name)
    clusters = pd.concat(jaccard.values(), ignore_index=True)
    # Cross-preprocessing persistence: refit the primary partition's resolution on the complete-feature geometry.
    complete_labels = ward(inputs["complete"], int(re.sub(r"\D", "", config["primary_partition"])))
    partitions["ward_complete"] = complete_labels

    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "tsne-atlas.npz", names=corpus["names"], keep_indices=keep, scores_primary=inputs["primary"],
                        variance_primary=variance["primary"], full_distances=d,
                        **arrays, **{f"partition__{k}": v for k, v in partitions.items()})
    grid.to_csv(out / "embedding-grid.csv", index=False)
    clusters.to_csv(out / "partition-clusters.csv", index=False)
    meta.to_csv(out / "dataset-metadata.csv", index=False)
    pd.DataFrame(info).T.to_csv(out / "preprocessing-variants.csv")
    manifest = {"config": config, "features_sha256": sha256((ROOT / config["features"]).read_bytes()).hexdigest(),
                "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
                "versions": {p: version(p) for p in ["numpy", "scipy", "scikit-learn", "pandas"]},
                "seconds": time.monotonic() - tick}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return out


def load(out: Path):
    with np.load(out / "tsne-atlas.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    return {"arrays": arrays, "grid": pd.read_csv(out / "embedding-grid.csv"),
            "clusters": pd.read_csv(out / "partition-clusters.csv"), "meta": pd.read_csv(out / "dataset-metadata.csv"),
            "variants": pd.read_csv(out / "preprocessing-variants.csv", index_col=0),
            "manifest": json.loads((out / "manifest.json").read_text())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    with threadpool_limits(limits=config["threads"]):
        compute(config, ROOT / config["output"])


if __name__ == "__main__":
    main()
