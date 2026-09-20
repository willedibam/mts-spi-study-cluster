"""Offline corpus explorer v2; v1 and its assets are read-only dependencies.

Run: .venv/bin/python -m scripts.explore_zenodo_v2 [--render-only]
No pyspi, cluster jobs, GMM or automatic scientific labels are involved.
"""
from __future__ import annotations

import argparse
from fnmatch import fnmatchcase
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import time

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, pairwise_distances, silhouette_samples
from threadpoolctl import threadpool_limits
from umap import UMAP
import yaml

from scripts.explore_zenodo_geometry import ROOT, historical_input, paper_style
from src.atlas_analysis import fit_atlas_transform


def distance_neighbors(distances, k=15):
    """Exclude self by identity, even when two datasets are exact duplicates."""
    work = distances.copy()
    np.fill_diagonal(work, np.inf)
    return np.argsort(work, axis=1, kind="stable")[:, :k]


def recall(reference, candidate):
    return float(np.mean([len(set(a) & set(b)) / len(a) for a, b in zip(reference, candidate)]))


def medoids_alternate(x, k, max_iter=300):
    """NumPy compatibility implementation of sklearn-extra's optional defaults.

    Euclidean, heuristic initialization, alternate updates, strict cost decrease.
    Follows sklearn-extra/cluster/_k_medoids.py (BSD-3-Clause), not PAM or KMeans.
    The heuristic initializer is deterministic; random_state=42 has no effect.
    Kept local because sklearn-extra's released binary does not support NumPy 2.
    """
    d = pairwise_distances(x)
    centers = np.argpartition(d.sum(axis=1), k - 1)[:k]
    for iteration in range(max_iter):
        before = centers.copy()
        labels = d[centers].argmin(axis=0)
        for group in range(k):
            members = np.flatnonzero(labels == group)
            if not len(members):
                continue
            costs = d[np.ix_(members, members)].sum(axis=1)
            best = costs.argmin()
            current = np.argmax(members == centers[group])
            if costs[best] < costs[current]:
                centers[group] = members[best]
        if np.array_equal(centers, before):
            return d[centers].argmin(axis=0), centers, iteration + 1
    raise RuntimeError("Optional historical K-medoids did not converge")


def summarize_partition(labels, distances):
    """All silhouettes/representatives use the same full primary geometry."""
    covered = labels >= 0
    groups, sizes = np.unique(labels[covered], return_counts=True)
    silhouettes = np.full(len(labels), np.nan)
    if 1 < len(groups) < covered.sum():
        silhouettes[covered] = silhouette_samples(
            distances[np.ix_(covered, covered)], labels[covered], metric="precomputed")
    members = {}
    for group, size in zip(groups, sizes):
        rows = np.flatnonzero(labels == group)
        medoid = rows[np.argmin(distances[np.ix_(rows, rows)].sum(axis=1))]
        members[str(group)] = {"size": int(size), "medoid": int(medoid),
            "silhouette": float(np.mean(silhouettes[rows])) if np.isfinite(silhouettes[rows]).all() else None}
    return {"clusters": len(groups), "coverage": float(covered.mean()),
            "median_size": float(np.median(sizes)) if len(sizes) else 0.,
            "full_silhouette_assigned": float(np.nanmean(silhouettes)) if np.isfinite(silhouettes).any() else None,
            "groups": members}


def compute(config, out):
    previous = ROOT / config["previous"]
    source = ROOT / config["features"]
    digest = sha256(source.read_bytes()).hexdigest()
    old_manifest = json.loads((previous / "manifest.json").read_text())
    if digest != old_manifest["source_sha256"]:
        raise ValueError("v1 and v2 must use the same canonical feature bank")
    with np.load(source, allow_pickle=False) as z:
        raw, names = z["X"].astype(float), z["y"].astype(str)
    transform = fit_atlas_transform(raw)
    x = transform.transform(raw)
    old, old_rows, old_columns = historical_input(raw)
    all_rows = np.arange(len(x))
    print(f"Primary {x.shape}; historical {old.shape}. Computing full reference and exact PCA…", flush=True)
    d_full = squareform(pdist(x))
    nn_full = distance_neighbors(d_full)
    start = time.monotonic()
    pca = PCA(svd_solver="full", whiten=False)
    scores = pca.fit_transform(x)
    pca_seconds = time.monotonic() - start
    # Last centered component is zero: N-1 scores are lossless for corpus distances.
    full_scores = scores[:, :len(x) - 1]
    np.testing.assert_allclose(squareform(pdist(full_scores)), d_full, atol=1e-8, rtol=1e-8)
    geometries = {f"pca{d}": scores[:, :d] for d in config["pca_dimensions"]}
    geometries["full"] = full_scores
    embeddings, partitions, partition_info = {}, {}, {}
    pca_metrics = []
    for dim in config["pca_dimensions"]:
        pca_metrics.append({"dimensions": dim,
            "variance": float(pca.explained_variance_ratio_[:dim].sum()),
            "knn15_full": recall(nn_full, distance_neighbors(squareform(pdist(scores[:, :dim]))))})

    def embed(key, data, method, section, title, rows=all_rows, cached=None, **kwargs):
        tick = time.monotonic()
        if cached is not None:
            coordinates, params = cached
            elapsed = None
            iterations = None
        elif method == "PCA":
            coordinates, params, iterations = data, {}, None
            elapsed = time.monotonic() - tick
        else:
            if method == "t-SNE":
                params = dict(n_components=2, init="pca", learning_rate="auto", early_exaggeration=12.,
                    metric="euclidean", method="barnes_hut", angle=.5, max_iter=config["tsne_iterations"],
                    random_state=config["seed"], n_jobs=config["threads"])
                params.update(kwargs)
                model = TSNE(**params)
            else:
                params = dict(n_components=2, min_dist=.1, metric="euclidean", random_state=config["seed"], n_jobs=1)
                params.update(kwargs)
                model = UMAP(**params)
            coordinates = model.fit_transform(data)
            elapsed = time.monotonic() - tick
            iterations = int(model.n_iter_) if hasattr(model, "n_iter_") else None
        reference = nn_full if len(rows) == len(x) and np.array_equal(rows, all_rows) else distance_neighbors(d_full[np.ix_(rows, rows)])
        embeddings[key] = {"coordinates": coordinates, "rows": rows, "method": method,
            "section": section, "title": title, "parameters": params, "seconds_fit": elapsed,
            "iterations": iterations, "cached_v1": cached is not None,
            "knn15_full": recall(reference, distance_neighbors(squareform(pdist(coordinates))))}
        print(f"Embedding {key}: {elapsed if elapsed is not None else 'cached'}", flush=True)

    with np.load(previous / "exploration.npz", allow_pickle=False) as v1:
        np.testing.assert_array_equal(names, v1["names"])
        np.testing.assert_array_equal(old_rows, v1["rows__historical"])
        embed("historical-p10", old, "t-SNE", "historical", "Original recipe · perplexity 10", rows=old_rows,
              cached=(v1["embedding__historical"], old_manifest["embeddings"]["historical"]["parameters"]))
        historical_labels = v1["cluster__map-historical"].copy()
    for p in config["historical_perplexities"]:
        if p != 10:
            embed(f"historical-p{p}", old, "t-SNE", "historical", f"Historical variation · perplexity {p}",
                  rows=old_rows, perplexity=p, random_state=42, max_iter=1000)
    for space in ["full", *[f"pca{d}" for d in config["pca_dimensions"]]]:
        for p in config["tsne_perplexities"]:
            embed(f"tsne-{space}-p{p}", x if space == "full" else geometries[space], "t-SNE", "preferred",
                  f"{space.upper()} → t-SNE · perplexity {p}", perplexity=p)
    embed("tsne-pca100-p100", geometries["pca100"], "t-SNE", "preferred", "PCA100 → t-SNE · perplexity 100", perplexity=100)
    for seed in config["embedding_seeds"]:
        embed(f"tsne-pca100-p30-s{seed}", geometries["pca100"], "t-SNE", "preferred",
              f"PCA100 → t-SNE · p30 · seed {seed}", perplexity=30, random_state=seed)
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        key = f"pc{a+1}-pc{b+1}"
        embed(key, scores[:, [a, b]], "PCA", "pca", f"PC{a+1} versus PC{b+1}")
        embeddings[key]["axis_labels"] = [f"PC{i+1} ({100*pca.explained_variance_ratio_[i]:.1f}%)" for i in (a, b)]
    for dim in [50, 100]:
        for n in config["umap_neighbors"]:
            embed(f"umap-pca{dim}-n{n}", geometries[f"pca{dim}"], "UMAP", "umap",
                  f"PCA{dim} → UMAP · neighbours {n} · min_dist 0.1", n_neighbors=n)
    for m in [0., .5]:
        embed(f"umap-pca100-n30-m{m}", geometries["pca100"], "UMAP", "umap",
              f"PCA100 → UMAP · neighbours 30 · min_dist {m}", n_neighbors=30, min_dist=m)

    def partition(key, labels, space, method, params, map_id=None, stability=None):
        partitions[key] = labels
        partition_info[key] = {"space": space, "method": method, "parameters": params,
            "map_id": map_id, "stability_seed_ARI": stability, **summarize_partition(labels, d_full)}
        print(f"Partition {key}: {partition_info[key]['clusters']} groups", flush=True)

    for space, data in geometries.items():
        for k in config["kmeans_clusters"]:
            params = dict(n_clusters=k, n_init=20, random_state=config["seed"], max_iter=300)
            model = KMeans(**params).fit(data)
            replicas = [KMeans(**{**params, "random_state": seed}).fit_predict(data) for seed in config["stability_seeds"]]
            ari = float(np.mean([adjusted_rand_score(model.labels_, r) for r in replicas]))
            partition(f"kmeans-{space}-k{k}", model.labels_, space, "K-means", params, stability=ari)
    for dim in config["hdbscan_dimensions"]:
        for size in config["hdbscan_min_cluster_sizes"]:
            for selection in ["eom", "leaf"]:
                params = dict(min_cluster_size=size, min_samples=5, cluster_selection_method=selection)
                labels = hdbscan.HDBSCAN(**params).fit_predict(geometries[f"pca{dim}"])
                partition(f"hdbscan-pca{dim}-{selection}-m{size}", labels, f"pca{dim}", "HDBSCAN", params)
    for key in [k for k, e in embeddings.items() if e["section"] == "historical"] + ["tsne-pca100-p30", "umap-pca100-n15"]:
        e = embeddings[key]
        for selection in ["eom", "leaf"]:
            params = dict(min_cluster_size=5, min_samples=5, cluster_selection_method=selection)
            labels = np.full(len(x), -2, dtype=int)
            labels[e["rows"]] = hdbscan.HDBSCAN(**params).fit_predict(e["coordinates"])
            if key == "historical-p10" and selection == "eom":
                np.testing.assert_array_equal(labels, historical_labels)
            partition(f"map-{key}-{selection}", labels, "2D map", "HDBSCAN", params, map_id=key)
    labels, centers, iterations = medoids_alternate(embeddings["historical-p10"]["coordinates"], 200)
    full_labels = np.full(len(x), -2, dtype=int)
    full_labels[old_rows] = labels
    partition("map-historical-p10-kmedoids200", full_labels, "2D map", "K-medoids (compatibility)",
              {"n_clusters": 200, "init": "heuristic", "method": "alternate", "max_iter": 300,
               "iterations": iterations, "implementation": "NumPy compatibility, not historical binary"}, map_id="historical-p10")

    manifest = {"config": config, "source_sha256": digest, "v1_manifest_sha256": sha256((previous / "manifest.json").read_bytes()).hexdigest(),
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {p: version(p) for p in ["numpy", "scipy", "scikit-learn", "umap-learn", "hdbscan"]},
        "primary_shape": list(x.shape), "historical_shape": list(old.shape), "pca_fit_seconds": pca_seconds,
        "pca_metrics": pca_metrics, "full_clustering": "Exact unwhitened N-1 PCA scores; full corpus Euclidean distances verified identical.",
        "historical_note": "Available downstream recipe on current Pearson bank. p10 cached unchanged from v1; HDBSCAN labels recomputed and verified identical. Original Spearman bank/environment unavailable.",
        "stability_note": "ARI across initialization seeds, all rows, fixed preprocessing. NOT grouped resampling or evidence of independent replication.",
        "embeddings": {k: {a: b for a, b in e.items() if a not in ["coordinates", "rows"]} for k, e in embeddings.items()},
        "partitions": partition_info}
    arrays = {"names": names, "pca_scores_full": full_scores, "pca_components_200": pca.components_[:200],
        "pca_variance_ratio": pca.explained_variance_ratio_, "feature_indices": transform.keep_indices,
        "impute_values": transform.impute_values, "feature_center": transform.center, "pca_mean": pca.mean_,
        "full_feature_distances": d_full, "historical_feature_indices": old_columns}
    for key, e in embeddings.items():
        arrays[f"embedding__{key}"] = e["coordinates"]
        arrays[f"rows__{key}"] = e["rows"]
    arrays.update({f"cluster__{k}": v for k, v in partitions.items()})
    np.savez_compressed(out / "exploration.npz", **arrays)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    pd.DataFrame({"dataset": names, **partitions}).to_csv(out / "memberships.csv", index=False)
    pd.DataFrame(pca_metrics).to_csv(out / "pca-sensitivity.csv", index=False)
    pd.DataFrame([{ "embedding": k, **{a: e[a] for a in ["knn15_full", "seconds_fit", "iterations", "cached_v1"]}}
                  for k, e in embeddings.items()]).to_csv(out / "embedding-metrics.csv", index=False)
    pd.DataFrame([{ "partition": k, **{a: e[a] for a in ["space", "clusters", "coverage", "median_size", "full_silhouette_assigned", "stability_seed_ARI"]}}
                  for k, e in partition_info.items()]).to_csv(out / "cluster-metrics.csv", index=False)


def corrected_tags(name, tags, rules):
    """Append documented metadata tags without changing the archived source."""
    result = list(tags)
    for rule in rules:
        if fnmatchcase(name, rule["pattern"]):
            result.extend(tag for tag in rule["tags"] if tag not in result)
    return result


def dataset_origin(tags):
    """Use explicit archive tags; absence is not evidence of synthetic origin."""
    real, synthetic = "real" in tags, "synthetic" in tags
    return "mixed" if real and synthetic else "real" if real else "synthetic" if synthetic else "unlabelled"


def marker_radii(process_counts, minimum, maximum):
    """Bounded area scaling: radii 2–4 SVG units, fixed across the corpus."""
    counts = np.asarray(process_counts, dtype=float)
    fraction = np.clip((counts - minimum) / max(maximum - minimum, 1), 0, 1)
    return np.sqrt(4 + 12 * fraction)


def render(config, out):
    manifest = json.loads((out / "manifest.json").read_text())
    previous = ROOT / config["previous"]
    rows = pd.read_csv(previous / "heatmap-index.csv").to_dict("records")
    database_path = ROOT / config["database"]
    corrections_path = ROOT / config["label_corrections"]
    rules = yaml.safe_load(corrections_path.read_text())["add_tags"]
    with np.load(database_path, allow_pickle=False) as archive:
        source_tags = {str(name): json.loads(str(tags))
                       for name, tags in zip(archive["__dataset_names__"], archive["__labels_json__"])}
    for row in rows:
        row["source_tags"] = source_tags[row["dataset"]]
        row["tags"] = corrected_tags(row["dataset"], row["source_tags"], rules)
        row["origin"] = dataset_origin(row["tags"])
    pd.DataFrame([{ "dataset": row["dataset"], "origin": row["origin"],
                    "source_tags": json.dumps(row["source_tags"]), "tags": json.dumps(row["tags"])}
                  for row in rows]).to_csv(out / "dataset-metadata.csv", index=False)
    origin_styles = {"real": {"fill": "#333333", "stroke": "#ffffff"},
                     "synthetic": {"fill": "#969696", "stroke": "#ffffff"},
                     "mixed": {"fill": "#777777", "stroke": "#333333"},
                     "unlabelled": {"fill": "none", "stroke": "#666666"}}
    origin_counts = {origin: sum(row["origin"] == origin for row in rows) for origin in origin_styles}
    process_counts = np.array([row["M"] for row in rows])
    m_min, m_max = int(process_counts.min()), int(process_counts.max())
    radii = marker_radii(process_counts, m_min, m_max)
    legend_m = sorted(set([m_min, int(np.median(process_counts)), m_max]))
    size_legend = [{"M": m, "radius": float(marker_radii(m, m_min, m_max))} for m in legend_m]
    for row, radius in zip(rows, radii):
        row["marker_radius"] = float(radius)
    asset_prefix = Path(os.path.relpath(previous, out)).as_posix()
    for row in rows:
        for extension in ["png", "svg"]:
            if not (previous / row[extension]).is_file():
                raise FileNotFoundError(row[extension])
            row[extension] = f"{asset_prefix}/{row[extension]}"
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    paper_style()
    with np.load(out / "exploration.npz", allow_pickle=False) as z:
        np.testing.assert_array_equal(z["names"], [r["dataset"] for r in rows])
        embeddings = {}
        for key, e in manifest["embeddings"].items():
            xy = z[f"embedding__{key}"]
            axes = e.get("axis_labels", [f"{e['method']} 1", f"{e['method']} 2"])
            embeddings[key] = {**e, "xy": xy.tolist(), "rows": z[f"rows__{key}"].tolist(), "axis_labels": axes}
            fig, ax = plt.subplots(figsize=(4.2, 4.2), constrained_layout=True)
            # Matplotlib's s is squared marker diameter; use the same relative areas.
            ax.scatter(*xy.T, c=".15", s=1.25 * radii[z[f"rows__{key}"]] ** 2, edgecolors="white", linewidths=.3, alpha=.75)
            handles = [ax.scatter([], [], c=".15", s=1.25 * item["radius"] ** 2, edgecolors="white", linewidths=.3,
                                  label=f"$M={item['M']}$") for item in size_legend]
            ax.legend(handles=handles, loc="upper right", ncol=len(handles),
                      handletextpad=.3, columnspacing=.7, borderaxespad=.5)
            def figure_text(text):
                if plt.rcParams["text.usetex"]:
                    return text.replace("_", r"\_").replace("%", r"\%").replace("→", r"$\to$").replace("·", r"$\cdot$")
                return text
            ax.set(xlabel=figure_text(axes[0]), ylabel=figure_text(axes[1]), title=figure_text(e["title"]))
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_box_aspect(1)
            ax.spines[["top", "right"]].set_visible(True)
            for extension in ["svg", "png"]:
                with plt.rc_context({"savefig.bbox": None}):
                    fig.savefig(figures / f"{key}.{extension}", dpi=180 if extension == "png" else 600)
            plt.close(fig)
        partitions = {k: {**p, "labels": z[f"cluster__{k}"].tolist()} for k, p in manifest["partitions"].items()}
    payload = {"rows": rows, "embeddings": embeddings, "partitions": partitions, "size_legend": size_legend,
               "origin_styles": origin_styles, "origin_counts": origin_counts,
               "pca_metrics": manifest["pca_metrics"], "historical_note": manifest["historical_note"]}
    template = Path(__file__).with_name("zenodo_gallery_v2.html").read_text()
    kde_script = Path(__file__).with_name("zenodo_kde.js").read_text()
    page = template.replace("__KDE_SCRIPT__", kde_script)
    (out / "index.html").write_text(page.replace("__CORPUS_DATA__", json.dumps(payload, allow_nan=False).replace("</", "<\\/")))
    manifest["rendering"] = {"template_sha256": sha256(template.encode()).hexdigest(),
        "kde_script_sha256": sha256(kde_script.encode()).hexdigest(),
        "origin_source": config["database"], "origin_source_sha256": sha256(database_path.read_bytes()).hexdigest(),
        "label_corrections": config["label_corrections"], "label_corrections_sha256": sha256(corrections_path.read_bytes()).hexdigest(),
        "origin_counts": origin_counts,
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "style_sha256": sha256((ROOT / "docs/benchmark-figure-style.md").read_bytes()).hexdigest()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Ready: {out / 'index.html'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/analysis/zenodo-visual-exploration-v2.yaml")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    out = ROOT / config["output"]
    if out.resolve() == (ROOT / config["previous"]).resolve():
        raise ValueError("v2 must not overwrite v1")
    out.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=config["threads"]):
        if not args.render_only:
            compute(config, out)
        render(config, out)


if __name__ == "__main__":
    main()
