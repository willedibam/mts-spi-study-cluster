"""Local Figures 4--5 exploration; run with --config, then inspect index.html.

Requires hdbscan==0.8.44 in addition to the repository environment. Historical
settings mean the old downstream code applied to our current Pearson features,
not a reconstruction of the draft's original Spearman SPI feature bank.
"""
from __future__ import annotations

import argparse
import base64
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits
from umap import UMAP
import hdbscan
import yaml

from src.atlas_analysis import fit_atlas_transform
from src.corpus_visualization import scale_timeseries

ROOT = Path(__file__).resolve().parents[1]


def paper_style() -> None:
    """Formatting from docs/benchmark-figure-style.md; neutral corpus colours."""
    plt.rcParams.update({
        "text.usetex": shutil.which("latex") is not None and shutil.which("dvipng") is not None,
        "font.family": "serif", "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm", "font.size": 9, "axes.labelsize": 9,
        "axes.titlesize": 10, "legend.fontsize": 8, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "axes.grid": False, "axes.spines.top": False,
        "axes.spines.right": False, "xtick.direction": "out", "ytick.direction": "out",
        "lines.linewidth": 1.7, "lines.markersize": 2.7, "legend.frameon": False,
        "figure.dpi": 180, "savefig.dpi": 600, "savefig.bbox": "tight",
        "figure.facecolor": "white", "axes.facecolor": "white", "svg.fonttype": "none",
    })
    # Matplotlib ships Computer Modern even when it is not installed as a system font.
    if not plt.rcParams["text.usetex"]:
        from matplotlib import font_manager
        font_manager.fontManager.addfont(str(Path(matplotlib.get_data_path()) / "fonts/ttf/cmr10.ttf"))
        plt.rcParams["font.serif"] = ["cmr10", "DejaVu Serif"]
        plt.rcParams["axes.formatter.use_mathtext"] = True


def historical_input(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirror available CorrelationFrame.get_feature_matrix() defaults and fillna.

    Drop duplicate feature rows first, keep datasets with >=20% finite entries,
    then features with >=80% finite entries, then fill with raw zero. No z-score.
    pandas preserves the historical threshold rounding and NaN duplicate semantics.
    """
    frame = pd.DataFrame(values.T).drop_duplicates()
    frame = frame.dropna(axis=1, thresh=0.2 * len(frame))
    frame = frame.dropna(axis=0, thresh=0.8 * frame.shape[1])
    return frame.fillna(0).T.to_numpy(), frame.columns.to_numpy(), frame.index.to_numpy()


def neighbors(values: np.ndarray, k: int = 15) -> np.ndarray:
    # Passing X explicitly includes self, so remove it by identity, including ties.
    indices = NearestNeighbors(n_neighbors=k + 1).fit(values).kneighbors(values, return_distance=False)
    return np.asarray([row[row != i][:k] for i, row in enumerate(indices)])


def overlap(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.mean([len(set(a) & set(b)) / len(a) for a, b in zip(first, second)]))


def density_overlay(ax, coordinates: np.ndarray, labels: np.ndarray) -> None:
    """50/80% KDE mass contours: visual guides, not confidence/cluster boundaries."""
    for label in np.unique(labels):
        if label < 0:
            continue
        points = coordinates[labels == label]
        if len(points) < 5 or np.linalg.matrix_rank(np.cov(points.T)) < 2:
            continue
        try:
            kde = gaussian_kde(points.T, bw_method="scott")
            kde.set_bandwidth(kde.factor * 1.2)
            padding = 3 * np.sqrt(np.diag(kde.covariance))
            lo, hi = points.min(0) - padding, points.max(0) + padding
            gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 55), np.linspace(lo[1], hi[1], 55))
            density = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
            sorted_density = np.sort(density.ravel())[::-1]
            mass = np.cumsum(sorted_density) / sorted_density.sum()
            levels = sorted(set(float(sorted_density[np.searchsorted(mass, p)]) for p in (.5, .8)))
            ax.contourf(gx, gy, density, levels=levels + [float(density.max()) + 1e-12],
                        colors=["0.65", "0.45"], alpha=.12, zorder=0)
        except np.linalg.LinAlgError:
            continue


def map_axis(ax, coordinates, labels=None, title="", axes_name="Dimension"):
    if labels is not None:
        density_overlay(ax, coordinates, labels)
    ax.scatter(*coordinates.T, s=7, c="0.16", alpha=.78, linewidth=0)
    ax.set(title=title, xlabel=f"{axes_name} 1", ylabel=f"{axes_name} 2")
    # Fix the viewport from the points: KDE tails must not change the apparent
    # point layout when comparing different partitions of the same embedding.
    lo, hi = coordinates.min(axis=0), coordinates.max(axis=0)
    midpoint = (lo + hi) / 2
    radius = max(float(np.max(hi - lo)), 1e-6) * .57
    ax.set_xlim(midpoint[0] - radius, midpoint[0] + radius)
    ax.set_ylim(midpoint[1] - radius, midpoint[1] + radius)
    ax.set_aspect("equal", adjustable="box")


def save_panel(out, stem, items, embeddings, partitions=None):
    cols = min(3, len(items)); rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.1 * rows),
                             squeeze=False, constrained_layout=True)
    for ax, item in zip(axes.flat, items):
        embedding, title, partition = item
        e = embeddings[embedding]
        labels = None if partition is None else partitions[partition][e["rows"]]
        map_axis(ax, e["coordinates"], labels, title, e["method"])
    for ax in axes.flat[len(items):]:
        ax.set_visible(False)
    fig.savefig(out / f"{stem}.svg")
    fig.savefig(out / f"{stem}.png", dpi=180)
    plt.close(fig)


def cluster_summary(values, labels):
    groups, counts = np.unique(labels[labels >= 0], return_counts=True)
    covered = labels >= 0
    return {
        "clusters": len(groups), "coverage": float(covered.mean()),
        "min_size": int(counts.min()) if len(counts) else 0,
        "median_size": float(np.median(counts)) if len(counts) else 0,
        "max_size": int(counts.max()) if len(counts) else 0,
        "silhouette": float(silhouette_score(values[covered], labels[covered]))
        if 1 < len(groups) < covered.sum() else None,
    }


def run(config_path: Path, figures_only: bool = False) -> None:
    config = yaml.safe_load(config_path.read_text())
    out = ROOT / config["output"]
    out.mkdir(parents=True, exist_ok=True)
    paper_style()
    if figures_only:
        with np.load(out / "exploration.npz", allow_pickle=False) as z:
            names = z["names"]
            manifest = json.loads((out / "manifest.json").read_text())
            embeddings = {key: {**info, "coordinates": z[f"embedding__{key}"],
                                "rows": z[f"rows__{key}"]} for key, info in manifest["embeddings"].items()}
            partitions = {key: z[f"cluster__{key}"] for key in manifest["partitions"]}
        render_outputs(out, config, names, embeddings, partitions, manifest)
        return
    start = time.monotonic()
    with np.load(ROOT / config["features"], allow_pickle=True) as z:
        values = z["X"].astype(np.float64)
        names = z["y"].astype(str)
        if values.shape != (1053, 41616) or str(z["metric"]) != "pearson":
            raise ValueError("Expected the 1053 x 41616 Pearson bank")
    with np.load(ROOT / config["atlas"], allow_pickle=True) as z:
        if not np.array_equal(names, z["dataset"].astype(str)):
            raise ValueError("Atlas row order differs from canonical bank")
        scores = z["pca_scores"].astype(float)
        variance = np.cumsum(z["pca_explained_variance_ratio"].astype(float))
        keep = z["feature_keep_indices"]
    transform = fit_atlas_transform(values)
    if not np.array_equal(keep, transform.keep_indices):
        raise ValueError("Atlas preprocessing has changed")
    centered = transform.transform(values)
    full_neighbors = neighbors(centered)
    all_rows = np.arange(len(values))
    old, old_rows, old_keep = historical_input(values)
    print(f"Historical defaults: {old.shape}; primary: {centered.shape}", flush=True)
    seed = int(config["seed"])
    embeddings, partitions, partition_info = {}, {}, {}
    metrics, cluster_metrics = [], []
    geometries = {f"pca{d}": scores[:, :d] for d in config["pca_dimensions"]}
    standard = centered / np.std(centered, axis=0)
    standard_pca = PCA(n_components=100, svd_solver="randomized", random_state=seed).fit_transform(standard)
    geometries["standard50"] = standard_pca[:, :50]
    pca_quality = [{"dimensions": d, "variance_explained": float(variance[d-1]),
                    "full_feature_knn15": overlap(full_neighbors, neighbors(scores[:, :d]))}
                   for d in config["pca_dimensions"]]
    pd.DataFrame(pca_quality).to_csv(out / "pca-sensitivity.csv", index=False)

    def embed(key, reference, method, rows=all_rows, **kwargs):
        stamp = time.monotonic()
        if method == "PCA":
            coordinates = reference[:, :2]
            actual = {"components_shown": [1, 2]}
        else:
            if method == "t-SNE":
                params = dict(n_components=2, init="pca", learning_rate="auto", max_iter=config["tsne_iterations"],
                              random_state=seed, n_jobs=config["threads"])
                params.update(kwargs)
                model = TSNE(**params)
            else:
                params = dict(n_components=2, n_neighbors=30, min_dist=.1, metric="euclidean",
                              random_state=seed, n_jobs=1)
                params.update(kwargs)
                model = UMAP(**params)
            coordinates = model.fit_transform(reference)
            actual = {k: v for k, v in model.get_params().items() if isinstance(v, (str, int, float, bool, type(None)))}
        embeddings[key] = {"method": method, "coordinates": coordinates, "rows": rows,
                           "input_dimensions": reference.shape[1], "parameters": actual}
        row = {"embedding": key, "rows": len(rows), "input_dimensions": reference.shape[1],
               "trustworthiness15_input": trustworthiness(reference, coordinates, n_neighbors=15),
               "knn15_input": overlap(neighbors(reference), neighbors(coordinates)),
               "knn15_primary_full": overlap(neighbors(centered[rows]), neighbors(coordinates)),
               "seconds": time.monotonic() - stamp}
        metrics.append(row)
        print(f"{key}: {row['seconds']:.1f}s, input recall={row['knn15_input']:.3f}", flush=True)

    embed("historical", old, "t-SNE", old_rows, perplexity=10, random_state=42, max_iter=1000)
    old_pca = PCA(n_components=50, svd_solver="randomized", random_state=42).fit_transform(old)
    embed("historical-pca50", old_pca, "t-SNE", old_rows, perplexity=10, random_state=42, max_iter=1000)
    embed("primary-full-old-tsne", centered, "t-SNE", perplexity=10, random_state=42, max_iter=1000)
    embed("primary-pca50-old-tsne", scores[:, :50], "t-SNE", perplexity=10, random_state=42, max_iter=1000)
    embed("pca", scores, "PCA")
    embed("standard-pca", standard_pca, "PCA")
    for d in config["pca_dimensions"]:
        for p in config["tsne_perplexities"]:
            embed(f"tsne-pca{d}-p{p}", scores[:, :d], "t-SNE", perplexity=p)
        for n in config["umap_neighbors"]:
            embed(f"umap-pca{d}-n{n}", scores[:, :d], "UMAP", n_neighbors=n, min_dist=config["umap_min_dist"])
    for m in config["umap_min_dist_sensitivity"]:
        embed(f"umap-pca50-m{m}", scores[:, :50], "UMAP", min_dist=m)
    embed("tsne-standard50", standard_pca[:, :50], "t-SNE", perplexity=30)
    embed("umap-standard50", standard_pca[:, :50], "UMAP")
    # Two additional stochastic realizations of the balanced inspection candidates.
    for replicate in [2718, 3141]:
        embed(f"tsne-seed{replicate}", scores[:, :50], "t-SNE", perplexity=30, random_state=replicate)
        embed(f"umap-seed{replicate}", scores[:, :50], "UMAP", random_state=replicate)

    def partition(key, reference, factory, space, rows=all_rows, predictive=False):
        model = factory(seed)
        labels = model.fit_predict(reference)
        full_labels = np.full(len(values), -2, dtype=int)
        full_labels[rows] = labels
        partitions[key] = full_labels
        stats = {"partition": key, "space": space, **cluster_summary(reference, labels)}
        params = {k: v for k, v in model.get_params().items() if isinstance(v, (str, int, float, bool, type(None)))}
        partition_info[key] = {"space": space, "parameters": params}
        if predictive:
            assignments = []
            for s in config["stability_seeds"]:
                subset = np.random.default_rng(s).choice(len(reference), round(.8 * len(reference)), replace=False)
                fitted = factory(s).fit(reference[subset])
                assignments.append(fitted.predict(reference))
            stats["subsample_ARI"] = float(np.mean([adjusted_rand_score(a, b)
                for i, a in enumerate(assignments) for b in assignments[i+1:]]))
        if isinstance(model, GaussianMixture):
            stats["BIC"] = float(model.bic(reference))
        cluster_metrics.append(stats)
        # Exact observed medoid in the fitted clustering space; all distances within cluster.
        exemplars = []
        for label in np.unique(labels):
            if label < 0:
                continue
            members = np.flatnonzero(labels == label)
            distances = cdist(reference[members], reference[members])
            medoid = members[np.argmin(distances.sum(axis=1))]
            exemplars.append({"partition": key, "cluster": int(label), "size": len(members),
                              "row": int(rows[medoid]), "dataset": names[rows[medoid]]})
        return exemplars

    exemplars = []
    for key in ["historical", "tsne-pca50-p30", "umap-pca50-n30"]:
        e = embeddings[key]
        exemplars += partition(f"map-{key}", e["coordinates"],
                               lambda s: hdbscan.HDBSCAN(min_cluster_size=5, min_samples=5),
                               key + " (2D descriptive)", e["rows"])
    for d in config["pca_dimensions"]:
        for k in config["kmeans_clusters"]:
            exemplars += partition(f"kmeans-pca{d}-k{k}", scores[:, :d],
                lambda s, k=k: KMeans(n_clusters=k, n_init=20, random_state=s), f"PCA{d}", predictive=True)
    for k in config["gmm_clusters"]:
        exemplars += partition(f"gmm-pca20-k{k}", scores[:, :20],
            lambda s, k=k: GaussianMixture(n_components=k, covariance_type="diag", reg_covar=1e-6,
                                            n_init=3, max_iter=500, random_state=s), "PCA20", predictive=True)
    for m in config["hdbscan_min_cluster_sizes"]:
        for method in ["eom", "leaf"]:
            exemplars += partition(f"hdbscan-pca50-{method}-m{m}", scores[:, :50],
                lambda s, m=m, method=method: hdbscan.HDBSCAN(min_cluster_size=m, min_samples=5,
                                                            cluster_selection_method=method), "PCA50")
    pd.DataFrame(metrics).to_csv(out / "embedding-metrics.csv", index=False)
    pd.DataFrame(cluster_metrics).to_csv(out / "cluster-metrics.csv", index=False)
    pd.DataFrame(exemplars).to_csv(out / "medoids.csv", index=False)
    manifest = {"config": config, "raw_shape": list(values.shape), "primary_shape": list(centered.shape),
        "historical_shape": list(old.shape), "old_feature_indices": old_keep.tolist(),
        "source_sha256": sha256((ROOT / config["features"]).read_bytes()).hexdigest(),
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {p: version(p) for p in ["numpy", "scipy", "pandas", "scikit-learn", "umap-learn", "hdbscan", "matplotlib"]},
        "embeddings": {key: {k: v for k, v in info.items() if k not in ["coordinates", "rows"]}
                       for key, info in embeddings.items()}, "partitions": partition_info,
        "analysis_seconds": time.monotonic() - start}
    payload = {"names": names, "pca_scores": scores, "standard_pca_scores": standard_pca,
               "primary_feature_indices": keep}
    for key, e in embeddings.items():
        payload[f"embedding__{key}"] = e["coordinates"]
        payload[f"rows__{key}"] = e["rows"]
    payload.update({f"cluster__{key}": value for key, value in partitions.items()})
    np.savez_compressed(out / "exploration.npz", **payload)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    pd.DataFrame({"dataset": names, **partitions}).to_csv(out / "memberships.csv", index=False)
    render_outputs(out, config, names, embeddings, partitions, manifest)


def render_outputs(out, config, names, embeddings, partitions, manifest):
    figures = out / "figures"; figures.mkdir(exist_ok=True)
    historical_items = [
        ("historical", "Historical preprocessing; direct t-SNE", "map-historical"),
        ("historical-pca50", "Same preprocessing + PCA50", None),
        ("primary-full-old-tsne", "Primary preprocessing; direct t-SNE", None),
        ("primary-pca50-old-tsne", "Primary preprocessing + PCA50", None),
        ("pca", "Covariance PCA", None), ("standard-pca", "Standardized PCA", None)]
    save_panel(figures, "historical-controls", historical_items, embeddings, partitions)
    for method, grid in [("tsne", config["tsne_perplexities"]), ("umap", config["umap_neighbors"])]:
        token = "p" if method == "tsne" else "n"
        items = [(f"{method}-pca{d}-{token}{p}", f"PCA{d}; {token}={p}", None)
                 for d in config["pca_dimensions"] for p in grid]
        save_panel(figures, method + "-sensitivity", items, embeddings)
    items = [("umap-pca50-m0.0", "UMAP: min distance = 0", None),
             ("umap-pca50-n30", "UMAP: min distance = 0.1", None),
             ("umap-pca50-m0.5", "UMAP: min distance = 0.5", None),
             ("tsne-pca50-p30", "Centre-only; t-SNE", None),
             ("tsne-standard50", "Standardized; t-SNE", None),
             ("umap-standard50", "Standardized; UMAP", None)]
    save_panel(figures, "scaling-and-spacing", items, embeddings)
    selected = ["kmeans-pca50-k8", "kmeans-pca50-k20", "kmeans-pca50-k40",
                "kmeans-pca50-k60", "gmm-pca20-k20", "hdbscan-pca50-eom-m10",
                "hdbscan-pca50-leaf-m10", "map-tsne-pca50-p30", "map-umap-pca50-n30"]
    titles = ["K-means: k=8", "K-means: k=20", "K-means: k=40", "K-means: k=60",
              "GMM: 20 components (PCA20)", "HDBSCAN: EOM, min size 10",
              "HDBSCAN: leaf, min size 10", "HDBSCAN fitted on t-SNE", "HDBSCAN fitted on UMAP"]
    items = [("umap-pca50-n30" if i == 8 else "tsne-pca50-p30", title, key)
             for i, (title, key) in enumerate(zip(titles, selected))]
    save_panel(figures, "cluster-resolutions", items, embeddings, partitions)
    items = [(key, key.replace("-", " "), None) for key in
             ["tsne-pca50-p30", "tsne-seed2718", "tsne-seed3141", "umap-pca50-n30", "umap-seed2718", "umap-seed3141"]]
    save_panel(figures, "seed-sensitivity", items, embeddings)
    for key in ["historical", "tsne-pca50-p30", "umap-pca50-n30"]:
        save_panel(figures, key, [(key, key.replace("-", " "), f"map-{key}")], embeddings, partitions)
    export_heatmaps(out, ROOT / config["database"], names)
    write_gallery(out, names, embeddings, partitions)
    manifest["rendering_script_sha256"] = sha256(Path(__file__).read_bytes()).hexdigest()
    manifest["figure_style_sha256"] = sha256((ROOT / "docs/benchmark-figure-style.md").read_bytes()).hexdigest()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Ready: {out / 'index.html'}", flush=True)


def export_heatmaps(out, database_path, names):
    import seaborn as sns
    heatmaps = out / "heatmaps"; heatmaps.mkdir(exist_ok=True)
    cmap = sns.color_palette("icefire", as_cmap=True)
    rows = []
    with np.load(database_path, allow_pickle=False) as database:
        for i, name in enumerate(names):
            data = database[name]
            scaled, limit = scale_timeseries(data, "robust")
            stem = f"{i:04d}"
            rgba = cmap(np.clip((scaled / limit + 1) / 2, 0, 1), bytes=True)
            # Native sample columns; process rows enlarged for readable gallery previews.
            picture = Image.fromarray(rgba).resize((data.shape[1], data.shape[0] * 12), Image.Resampling.NEAREST)
            picture.save(heatmaps / f"{stem}.png")
            encoded = base64.b64encode((heatmaps / f"{stem}.png").read_bytes()).decode()
            # Editor-friendly single raster inside SVG, at >= one pixel per sample.
            (heatmaps / f"{stem}.svg").write_text(
                f'<svg xmlns="http://www.w3.org/2000/svg" width="6in" height="{max(.5, .055*data.shape[0]):.3f}in" '
                f'viewBox="0 0 {data.shape[1]} {data.shape[0]*12}" preserveAspectRatio="none">'
                f'<image width="100%" height="100%" preserveAspectRatio="none" href="data:image/png;base64,{encoded}"/></svg>')
            rows.append({"row": i, "dataset": name, "M": data.shape[0], "T": data.shape[1],
                         "display_limit": limit, "png": f"heatmaps/{stem}.png", "svg": f"heatmaps/{stem}.svg"})
    pd.DataFrame(rows).to_csv(out / "heatmap-index.csv", index=False)


def write_gallery(out, names, embeddings, partitions):
    """Offline inspection report; embedded metadata means file:// works without a server."""
    template = Path(__file__).with_name("zenodo_cluster_gallery.html").read_text()
    table = pd.read_csv(out / "heatmap-index.csv").to_dict("records")
    payload = {"rows": table, "embeddings": {key: {"xy": e["coordinates"].tolist(), "rows": e["rows"].tolist()}
                for key, e in embeddings.items()}, "partitions": {key: v.tolist() for key, v in partitions.items()}}
    payload["shortlist"] = json.loads((out / "shortlist.json").read_text()) if (out / "shortlist.json").exists() else []
    (out / "index.html").write_text(template.replace("__CORPUS_DATA__", json.dumps(payload).replace("</", "<\\/")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/analysis/zenodo-visual-exploration.yaml")
    parser.add_argument("--figures-only", action="store_true", help="Re-render cached coordinates without refitting")
    args = parser.parse_args()
    with threadpool_limits(limits=8):
        run(args.config, args.figures_only)
