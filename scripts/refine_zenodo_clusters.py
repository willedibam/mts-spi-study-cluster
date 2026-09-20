"""Extend cached corpus partitions and audit small groups without rerunning embeddings."""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_samples
from threadpoolctl import threadpool_limits

from scripts.explore_zenodo_geometry import ROOT, paper_style, save_panel, write_gallery
from src.atlas_analysis import fit_atlas_transform

OUT = ROOT / "results/zenodo_7118947/visual-exploration"


def best_jaccard(members, labels):
    """Best membership match, independent of arbitrary cluster label numbering."""
    groups, counts = np.unique(labels[members], return_counts=True)
    return max((count / (len(members) + np.sum(labels == group) - count)
                for group, count in zip(groups, counts) if group >= 0), default=0.)


def run():
    with np.load(OUT / "exploration.npz", allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    manifest = json.loads((OUT / "manifest.json").read_text())
    names, scores = payload["names"], payload["pca_scores"]
    with np.load(ROOT / manifest["config"]["features"], allow_pickle=True) as archive:
        if not np.array_equal(archive["y"].astype(str), names):
            raise ValueError("Feature and embedding row orders differ")
        values, tags = archive["X"].astype(float), archive["labels"].tolist()
    transform = fit_atlas_transform(values)
    transformed = transform.transform(values)
    distances = cdist(transformed, transformed)
    np.fill_diagonal(distances, 0.)
    complete_indices = np.flatnonzero(np.all(np.isfinite(values), axis=0))
    complete_indices = complete_indices[np.std(values[:, complete_indices], axis=0) >= 1e-8]
    complete_distances = cdist(values[:, complete_indices], values[:, complete_indices])
    np.fill_diagonal(complete_distances, 0.)
    raw_validity = np.mean(np.isfinite(values), axis=1)
    primary_validity = np.mean(np.isfinite(values[:, transform.keep_indices]), axis=1)
    source = ["mixed" if "real" in t and "synthetic" in t else
              "real" if "real" in t else "synthetic" if "synthetic" in t else "unspecified" for t in tags]
    all_rows = np.arange(len(names))
    seeds = [11, 23, 47, 71, 101]
    replicated = {}
    metrics = []

    # Re-audit two existing resolutions, without replacing their labels.
    jobs = [(d, k) for d in [20, 50, 100] for k in [80, 120, 160, 200]]
    jobs += [(20, 40), (50, 60)]
    for d, k in jobs:
        key = f"kmeans-pca{d}-k{k}"
        x = scores[:, :d]
        if f"cluster__{key}" not in payload:
            payload[f"cluster__{key}"] = KMeans(n_clusters=k, n_init=20, random_state=1729).fit_predict(x)
        labels = payload[f"cluster__{key}"]
        replicas = []
        for seed in seeds:
            rows = np.random.default_rng(seed).choice(len(x), round(.8 * len(x)), replace=False)
            model = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(x[rows])
            replicas.append(model.predict(x))
        replicated[key] = np.asarray(replicas)
        _, counts = np.unique(labels, return_counts=True)
        metrics.append({"partition": key, "dimensions": d, "k": k,
            "median_size": np.median(counts), "max_size": counts.max(),
            "clusters_under5": int(np.sum(counts < 5)),
            "rows_in_under5_fraction": float(counts[counts < 5].sum() / len(x)),
            "subsample_ARI": np.mean([adjusted_rand_score(a, b) for i, a in enumerate(replicas) for b in replicas[i+1:]])})
        manifest["partitions"].setdefault(key, {"space": f"PCA{d}", "parameters": {
            "n_clusters": k, "n_init": 20, "random_state": 1729}})
        print(metrics[-1], flush=True)
    for embedding in ["historical", "tsne-pca50-p30"]:
        for size in [5, 10]:
            key = f"map-{embedding}-leaf-m{size}"
            rows = payload[f"rows__{embedding}"]
            labels = np.full(len(names), -2)
            labels[rows] = hdbscan.HDBSCAN(min_cluster_size=size, min_samples=5,
                cluster_selection_method="leaf").fit_predict(payload[f"embedding__{embedding}"])
            payload[f"cluster__{key}"] = labels
            manifest["partitions"][key] = {"space": embedding + " (2D descriptive)",
                "parameters": {"min_cluster_size": size, "min_samples": 5, "cluster_selection_method": "leaf"}}

    audit, members_table, medoids = [], [], []
    keys = list(manifest["partitions"])
    for key in keys:
        labels = payload[f"cluster__{key}"]
        # Use a common full-feature reference for every partition's silhouette.
        mask = labels >= 0
        groups = np.unique(labels[mask])
        silhouette = np.full(len(names), np.nan)
        complete_silhouette = np.full(len(names), np.nan)
        if 1 < len(groups) < mask.sum():
            silhouette[mask] = silhouette_samples(distances[np.ix_(mask, mask)], labels[mask], metric="precomputed")
            complete_silhouette[mask] = silhouette_samples(complete_distances[np.ix_(mask, mask)], labels[mask], metric="precomputed")
        for group in groups:
            indices = np.flatnonzero(labels == group)
            medoid = indices[np.argmin(distances[np.ix_(indices, indices)].sum(axis=1))]
            medoids.append({"partition": key, "cluster": int(group), "size": len(indices),
                            "row": int(medoid), "dataset": names[medoid], "space": "full primary features"})
            if not 5 <= len(indices) <= 40:
                continue
            outside = np.setdiff1d(all_rows, indices)
            near_outside = outside[np.argmin(distances[medoid, outside])]
            radius = np.quantile(distances[medoid, indices], .9)
            tag_counts = Counter(tag for i in indices for tag in tags[i] if tag not in ["real", "synthetic"])
            composition = Counter(source[i] for i in indices)
            row = {"partition": key, "cluster": int(group), "size": len(indices),
                "full_silhouette": float(np.mean(silhouette[indices])),
                "complete_silhouette": float(np.mean(complete_silhouette[indices])),
                "primary_feature_validity": float(np.mean(primary_validity[indices])),
                "outside_distance_over_radius90": float(distances[medoid, near_outside] / radius) if radius else None,
                "medoid": names[medoid], "medoid_row": int(medoid),
                "nearest_outsider": names[near_outside],
                "real": composition["real"], "synthetic": composition["synthetic"],
                "unspecified": composition["unspecified"],
                "mean_raw_validity": float(np.mean(raw_validity[indices])),
                "source_tags": json.dumps(tag_counts.most_common(8)),
                "members": json.dumps(names[indices].tolist())}
            if key in replicated:
                matches = [best_jaccard(indices, replica) for replica in replicated[key]]
                row["subsample_jaccard_mean"] = float(np.mean(matches))
                row["subsample_jaccard_min"] = float(np.min(matches))
            audit.append(row)
            for i in indices:
                members_table.append({"partition": key, "cluster": int(group), "row": int(i),
                    "dataset": names[i], "source": source[i], "tags": json.dumps(tags[i])})
    pd.DataFrame(metrics).to_csv(OUT / "high-k-metrics.csv", index=False)
    pd.DataFrame(audit).to_csv(OUT / "small-cluster-audit.csv", index=False)
    pd.DataFrame(members_table).to_csv(OUT / "small-cluster-members.csv", index=False)
    pd.DataFrame(medoids).to_csv(OUT / "full-feature-medoids.csv", index=False)
    np.savez_compressed(OUT / "high-k-resamples.npz", **replicated)
    np.savez_compressed(OUT / "exploration.npz", **payload)
    partitions = {key: payload[f"cluster__{key}"] for key in keys}
    pd.DataFrame({"dataset": names, **partitions}).to_csv(OUT / "memberships.csv", index=False)
    manifest["high_k_extension"] = {"k": [80, 120, 160, 200], "pca_dimensions": [20, 50, 100],
        "seeds": seeds, "fraction": .8, "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "complete_case_features": len(complete_indices),
        "note": "Existing coordinates/partitions preserved. Resampling holds PCA fixed. Source tags used only after clustering."}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    embeddings = {key: {**meta, "rows": payload[f"rows__{key}"], "coordinates": payload[f"embedding__{key}"]}
                  for key, meta in manifest["embeddings"].items()}
    paper_style()
    items = [("tsne-pca50-p30", f"PCA{d}; k={k}", f"kmeans-pca{d}-k{k}")
             for d in [20, 50, 100] for k in [80, 120, 200]]
    save_panel(OUT / "figures", "high-k-resolutions", items, embeddings, partitions)
    items = []
    for embedding in ["historical", "tsne-pca50-p30"]:
        for suffix, label in [("", "EOM, size 5"), ("-leaf-m5", "Leaf, size 5"), ("-leaf-m10", "Leaf, size 10")]:
            key = f"map-{embedding}{suffix}"
            labels = partitions[key]
            count = len(set(labels) - {-1, -2})
            view = "Historical" if embedding == "historical" else "PCA50 / t-SNE p30"
            items.append((embedding, f"{view}\n{label}: {count} groups", key))
    save_panel(OUT / "figures", "historical-leaf-comparison", items, embeddings, partitions)
    write_gallery(OUT, names, embeddings, partitions)
    print("High-k extension complete", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=8):
        run()
