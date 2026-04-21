"""Quantitative comparison of four UMAP preprocessing combos for the
5-class embedding (cauchy-noise, gaussian-noise, kuramoto, var-ring-sym, sti-i).

Combos:
  (a) raw + euclidean
  (b) scaled + euclidean  (StandardScaler → UMAP, = plot_umap default)
  (c) raw + correlation
  (d) raw + cosine

Reports, per combo:
  - silhouette score in the 2D UMAP embedding (higher = better class separation)
  - global 1-NN same-class rate (local purity)
  - per-class 1-NN same-class rate (where does the embedding fail?)
  - kuramoto ↔ gaussian specific confusion rate
  - pairwise class-centroid distances in 2D (absolute) and ranks
  - pairwise class-centroid distances in the ORIGINAL high-dim feature space
    (provides a UMAP-independent reference)

Runs UMAP with fixed random_state=0 so results are reproducible.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from umap import UMAP

ROOT = Path(__file__).resolve().parents[1]
FEATURES_FILE = ROOT / "features/data-embeddings-proof_260420_pearson.npz"
CLASSES = ["cauchy-noise", "gaussian-noise", "kuramoto", "var-ring-sym", "sti-i"]
UMAP_KWARGS = dict(n_neighbors=9, min_dist=0.25, random_state=0, verbose=False)


def per_class_knn_purity(emb, y, k=1):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(emb)
    _, idx = nn.kneighbors(emb)
    neighbours = y[idx[:, 1:k + 1]]
    same = (neighbours == y[:, None]).mean(axis=1)
    out = {}
    for cls in np.unique(y):
        out[cls] = float(same[y == cls].mean())
    out["_global"] = float(same.mean())
    return out


def specific_confusion(emb, y, cls_a, cls_b):
    """Fraction of cls_a points whose 1-NN is cls_b, and vice versa."""
    nn = NearestNeighbors(n_neighbors=2).fit(emb)
    _, idx = nn.kneighbors(emb)
    nearest = y[idx[:, 1]]
    a_to_b = float((nearest[y == cls_a] == cls_b).mean()) if (y == cls_a).any() else float("nan")
    b_to_a = float((nearest[y == cls_b] == cls_a).mean()) if (y == cls_b).any() else float("nan")
    return a_to_b, b_to_a


def centroid_distance_matrix(X_or_emb, y, classes):
    n = len(classes)
    D = np.zeros((n, n))
    cents = {c: X_or_emb[y == c].mean(axis=0) for c in classes}
    for i, c1 in enumerate(classes):
        for j, c2 in enumerate(classes):
            D[i, j] = np.linalg.norm(cents[c1] - cents[c2])
    return pd.DataFrame(D, index=classes, columns=classes)


def main():
    if not FEATURES_FILE.exists():
        print(f"[ERROR] feature file not found: {FEATURES_FILE}")
        sys.exit(1)

    data = np.load(FEATURES_FILE, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    M = data["M"].astype(int)

    mask = np.isin(y, CLASSES)
    X = X[mask]
    y = y[mask]
    M = M[mask]
    print(f"Loaded: X={X.shape}  n_classes={len(np.unique(y))}  counts={dict(pd.Series(y).value_counts())}")
    print()

    # ----- build the 4 embeddings -----
    embeddings = {}
    print("Fitting UMAP (a) raw + euclidean...", flush=True)
    embeddings["(a) raw+euclid  "] = UMAP(metric="euclidean", **UMAP_KWARGS).fit_transform(X)
    print("Fitting UMAP (b) scaled + euclidean...", flush=True)
    xs = StandardScaler().fit_transform(X)
    embeddings["(b) scaled+euclid"] = UMAP(metric="euclidean", **UMAP_KWARGS).fit_transform(xs)
    print("Fitting UMAP (c) raw + correlation...", flush=True)
    embeddings["(c) raw+corr     "] = UMAP(metric="correlation", **UMAP_KWARGS).fit_transform(X)
    print("Fitting UMAP (d) raw + cosine...", flush=True)
    embeddings["(d) raw+cosine   "] = UMAP(metric="cosine", **UMAP_KWARGS).fit_transform(X)

    # ----- global metrics table -----
    print("\n" + "=" * 80)
    print("Global metrics (higher = better)")
    print("=" * 80)
    rows = []
    for name, emb in embeddings.items():
        sil = silhouette_score(emb, y)
        purity = per_class_knn_purity(emb, y, k=1)
        purity_k5 = per_class_knn_purity(emb, y, k=5)
        k_to_g, g_to_k = specific_confusion(emb, y, "kuramoto", "gaussian-noise")
        rows.append({
            "combo": name.strip(),
            "silhouette": sil,
            "1NN_same_global": purity["_global"],
            "5NN_same_global": purity_k5["_global"],
            "kuramoto→gaussian_1NN": k_to_g,
            "gaussian→kuramoto_1NN": g_to_k,
        })
    df_global = pd.DataFrame(rows)
    print(df_global.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ----- per-class 1-NN purity -----
    print("\n" + "=" * 80)
    print("Per-class 1-NN same-class rate (where each combo fails)")
    print("=" * 80)
    pur_rows = []
    for name, emb in embeddings.items():
        pur = per_class_knn_purity(emb, y, k=1)
        pur_rows.append({"combo": name.strip(), **{c: pur[c] for c in CLASSES}})
    print(pd.DataFrame(pur_rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ----- per-class 5-NN purity -----
    print("\n" + "=" * 80)
    print("Per-class 5-NN same-class rate (broader neighbourhood; 1.0 = pure local region)")
    print("=" * 80)
    pur_rows = []
    for name, emb in embeddings.items():
        pur = per_class_knn_purity(emb, y, k=5)
        pur_rows.append({"combo": name.strip(), **{c: pur[c] for c in CLASSES}})
    print(pd.DataFrame(pur_rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ----- reference: class-centroid distances in ORIGINAL feature space -----
    print("\n" + "=" * 80)
    print("Reference: centroid distances in ORIGINAL feature space (raw, euclidean)")
    print("(UMAP-independent; tells you which classes are intrinsically close)")
    print("=" * 80)
    D_raw = centroid_distance_matrix(X, y, CLASSES)
    print(D_raw.to_string(float_format=lambda v: f"{v:.2f}"))

    # same, z-scored
    print("\n(and in STANDARDISED feature space — matches what combo (b) sees)")
    D_z = centroid_distance_matrix(xs, y, CLASSES)
    print(D_z.to_string(float_format=lambda v: f"{v:.2f}"))

    # ----- centroid distances in each 2D embedding -----
    for name, emb in embeddings.items():
        print("\n" + "=" * 80)
        print(f"Centroid distances in 2D UMAP — {name.strip()}")
        print("=" * 80)
        D = centroid_distance_matrix(emb, y, CLASSES)
        print(D.to_string(float_format=lambda v: f"{v:.2f}"))

    # ----- save embeddings and summary -----
    out_dir = ROOT / "scripts"
    np.savez(
        out_dir / "compare_umap_combos_embeddings.npz",
        **{name.strip().replace(" ", "_").replace("+", "_"): emb for name, emb in embeddings.items()},
        y=y,
    )
    df_global.to_csv(out_dir / "compare_umap_combos_metrics.csv", index=False)
    print(f"\nWrote embeddings -> {out_dir/'compare_umap_combos_embeddings.npz'}")
    print(f"Wrote metrics    -> {out_dir/'compare_umap_combos_metrics.csv'}")


if __name__ == "__main__":
    main()
