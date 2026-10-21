"""Observation-size control for the corpus-atlas candidates (2026-10-21).

Asks whether the lettered Ward-40 candidates of zenodo_1053_tsne_261021 survive a control for the number
of processes M and samples T. Reads the cached tsne-261021 outputs (run `tsne_atlas` first); never runs pyspi.
The notebook zenodo_1053_mt_control_261021.ipynb reports the results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_samples

from research.zenodo.atlas.tsne_atlas import best_jaccard, ward

CANDIDATES = {35: "A", 21: "B", 20: "C", 4: "D", 34: "E", 39: "F", 6: "G"}  # Ward-40 id -> letter (selection notebook)
LEADING_SOURCE = {"A": "NATOPS", "B": "NetSim fMRI", "C": "CML", "D": "Cricket", "E": "PEMS-SF",
                  "F": "ArticularyWordRecognition", "G": "IID noise"}


def check_candidates(meta, labels):
    """Fail if the cached partition no longer matches the selection notebook's letters."""
    for cluster, letter in CANDIDATES.items():
        if meta.source[labels == cluster].value_counts().index[0] != LEADING_SOURCE[letter]:
            raise ValueError(f"cluster {cluster} is no longer candidate {letter}; re-select candidates")


def size_basis(M, T):
    """Full quadratic surface in (log M, log T); spans log(T/M) and its square."""
    lm, lt = np.log(np.asarray(M, float)), np.log(np.asarray(T, float))
    return np.column_stack([lm, lt, lm ** 2, lt ** 2, lm * lt])


def _demean(values, groups):
    codes = pd.factorize(groups)[0]
    sums = np.zeros((codes.max() + 1, values.shape[1]))
    np.add.at(sums, codes, values)
    return values - (sums / np.bincount(codes)[:, None])[codes]


def residualise(scores, meta, by, permute=None):
    """Remove a smooth M/T component from the full-rank scores.

    by="pooled": OLS on the size basis across all rows. Size is aliased with source in this archive, so this
    also removes source character correlated with size (an over-control bound).
    by="source" or "parent": the coefficients are estimated from variation within sources or within parent
    recordings only (fixed effects), then the fitted surface, clipped to the M/T range that varies within
    groups, is subtracted from every row. Parent variation is the closest available proxy for "same dynamics,
    different size".
    permute: a Generator for a placebo that shuffles (M, T) within the same grouping (globally if pooled).
    """
    M, T = meta["M"].to_numpy(), meta["T"].to_numpy()
    groups = np.zeros(len(meta), int) if by == "pooled" else meta[by].to_numpy()
    if permute is not None:
        order = np.arange(len(meta))
        for g in np.unique(groups):
            rows = np.flatnonzero(groups == g)
            order[rows] = permute.permutation(rows)
        M, T = M[order], T[order]
    basis = size_basis(M, T)
    total = float(((scores - scores.mean(axis=0)) ** 2).sum())
    if by == "pooled":
        design = np.column_stack([np.ones(len(scores)), basis])
        beta, *_ = np.linalg.lstsq(design, scores, rcond=None)
        component = design[:, 1:] @ beta[1:]
        component -= component.mean(axis=0)
        return scores - component, {"removed_share": float((component ** 2).sum() / total), "within_r2": np.nan,
                                    "M_support": (int(M.min()), int(M.max())), "T_support": (int(T.min()), int(T.max()))}
    centred = _demean(scores, groups)
    beta, *_ = np.linalg.lstsq(_demean(basis, groups), centred, rcond=None)
    frame = pd.DataFrame({"g": groups, "M": M, "T": T})
    varying = frame.groupby("g").filter(lambda s: len(s[["M", "T"]].drop_duplicates()) > 1)
    support_M, support_T = (varying.M.min(), varying.M.max()), (varying["T"].min(), varying["T"].max())
    clipped = size_basis(np.clip(M, *support_M), np.clip(T, *support_T))
    component = (clipped - clipped.mean(axis=0)) @ beta
    within_r2 = float(((_demean(basis, groups) @ beta) ** 2).sum() / (centred ** 2).sum())
    return scores - component, {"removed_share": float((component ** 2).sum() / total), "within_r2": within_r2,
                                "M_support": tuple(map(int, support_M)), "T_support": tuple(map(int, support_T)),
                                "varying_groups": int(varying.g.nunique()), "varying_rows": len(varying)}


def candidate_separation(scores, labels, candidates=CANDIDATES):
    """Per-candidate silhouette under the fixed Ward-40 labels, and best-match Jaccard of a Ward-40 refit."""
    sil = silhouette_samples(squareform(pdist(scores)), labels, metric="precomputed")
    refit = ward(scores, labels.max() + 1)
    return {letter: (float(sil[labels == c].mean()), best_jaccard(labels, refit, c)) for c, letter in candidates.items()}


def residual_controls(scores, meta, labels, candidates=CANDIDATES, placebo=50, seed=1729):
    """Candidate separation after each residualisation, with a same-procedure placebo on shuffled (M, T)."""
    rows, removed = [], []
    for by in ["parent", "source", "pooled"]:
        residual, info = residualise(scores, meta, by)
        observed = candidate_separation(residual, labels, candidates)
        rng, null, shares = np.random.default_rng(seed), [], []
        for _ in range(placebo):
            shuffled, shuffled_info = residualise(scores, meta, by, permute=rng)
            null.append(candidate_separation(shuffled, labels, candidates))
            shares.append(shuffled_info["removed_share"])
        removed.append({"control": by, **info, "placebo_removed_share": float(np.median(shares))})
        for letter, (sil, jac) in observed.items():
            placebo_j = np.array([n[letter][1] for n in null])
            rows.append({"letter": letter, "control": by, "silhouette": sil, "jaccard": jac,
                         "placebo_jaccard_median": float(np.median(placebo_j)), "placebo_jaccard_q05": float(np.quantile(placebo_j, .05)),
                         "placebo_silhouette_median": float(np.median([n[letter][0] for n in null]))})
    return pd.DataFrame(rows), pd.DataFrame(removed)


def size_coordinates(meta):
    u = np.column_stack([np.log(meta["M"].to_numpy(float)), np.log(meta["T"].to_numpy(float))])
    return u / u.std(axis=0)


def size_matches(u, member, pool, taken, k):
    """Pool rows tied within the k nearest to `member` in standardised (log M, log T), skipping taken rows."""
    free = pool[~taken[pool]]
    gap = np.linalg.norm(u[free] - u[member], axis=1)
    return free[gap <= np.sort(gap)[min(k, len(gap)) - 1] + 1e-12]


def embedded_silhouette(distances, labels, rows):
    """Mean silhouette of `rows` treated as one cluster while every other row keeps its Ward-40 label.

    For a candidate this equals its ordinary Ward-40 silhouette."""
    rows = np.asarray(rows)
    inside = np.zeros(len(labels), bool)
    inside[rows] = True
    a = distances[np.ix_(rows, rows)].sum(axis=1) / (len(rows) - 1)
    b = np.full(len(rows), np.inf)
    for c in np.unique(labels[~inside]):
        b = np.minimum(b, distances[np.ix_(rows, np.flatnonzero((labels == c) & ~inside))].mean(axis=1))
    return float(np.mean((b - a) / np.maximum(a, b)))


def other_cluster_rank(distances, labels):
    """rank[i, c]: position of cluster c among the clusters other than i's own, by mean distance from i (1 = nearest).

    Scale-free per row, so it is not driven by how noisy row i's z is."""
    k = labels.max() + 1
    mean = np.column_stack([(distances[:, labels == c].sum(axis=1)) / ((labels == c).sum() - (labels == c)) for c in range(k)])
    mean[np.arange(len(labels)), labels] = np.inf
    return mean.argsort(axis=1).argsort(axis=1) + 1


def matched_null(distances, labels, meta, candidates=CANDIDATES, replicates=1000, k=5, seed=1729):
    """Same-size sets from *other* source families, matched member by member on (log M, log T).

    Each replicate draws, for every member, one row among its k nearest pool rows in standardised
    (log M, log T) (ties included, no row twice). The pool excludes every source family present in the
    candidate, so a matched set shares the candidate's sizes but not its sources. Unmatched sets draw from the
    same pool at random. Matched sets also inherit whatever character the few sources at those sizes share,
    so `survival` is a lower bound on separation not explained by size.
    """
    rng = np.random.default_rng(seed)
    u = size_coordinates(meta)
    rank = other_cluster_rank(distances, labels)
    source, M, T = meta.source.to_numpy(), meta["M"].to_numpy(float), meta["T"].to_numpy(float)
    rows, draws = [], {}
    for cluster, letter in candidates.items():
        members = np.flatnonzero(labels == cluster)
        pool = np.flatnonzero(~np.isin(source, np.unique(source[members])))
        matched, random, m_ratio, t_ratio = [], [], [], []
        for _ in range(replicates):
            taken, chosen = np.zeros(len(labels), bool), []
            for i in rng.permutation(members):
                j = rng.choice(size_matches(u, i, pool, taken, k))
                taken[j] = True
                chosen.append(j)
                m_ratio.append(abs(np.log2(M[j] / M[i])))
                t_ratio.append(abs(np.log2(T[j] / T[i])))
            matched.append(embedded_silhouette(distances, labels, chosen))
            random.append(embedded_silhouette(distances, labels, rng.choice(pool, len(members), replace=False)))
        matched, random = np.array(matched), np.array(random)
        observed = embedded_silhouette(distances, labels, members)
        nearest = np.unique(np.concatenate([size_matches(u, i, pool, np.zeros(len(labels), bool), k) for i in members]))
        outsiders = np.flatnonzero(labels != cluster)
        draws[letter] = {"matched": matched, "random": random, "observed": observed}
        top = pd.Series(source[nearest]).value_counts()
        rows.append({"letter": letter, "silhouette": observed, "matched_mean": matched.mean(), "matched_q95": np.quantile(matched, .95),
                     "random_mean": random.mean(), "p_matched": (1 + np.sum(matched >= observed)) / (1 + replicates),
                     "survival": (observed - matched.mean()) / (observed - random.mean()),
                     "match_M_ratio": float(2 ** np.median(m_ratio)), "match_T_ratio": float(2 ** np.median(t_ratio)),
                     "rank_matched": float(np.median(rank[nearest, cluster])), "rank_outsiders": float(np.median(rank[outsiders, cluster])),
                     "matched_sources": "; ".join(f"{s} {n}" for s, n in top.head(3).items()) + ("; …" if len(top) > 3 else "")})
    return pd.DataFrame(rows), draws


def parent_pairs(distances, meta, labels):
    """All row pairs that share a parent recording, typed by what differs. 'transform' marks price versus
    log-return versions of one market window (same M, T differing by one), which are not size variants."""
    nn5 = np.sort(distances + np.diag(np.full(len(distances), np.inf)), axis=1)[:, :5].mean(axis=1)
    rows = []
    for _, group in meta.groupby("parent"):
        r = group.row.to_numpy()
        for a in range(len(r)):
            for b in range(a + 1, len(r)):
                i, j = r[a], r[b]
                same_m, d_t = meta.M[i] == meta.M[j], abs(int(meta["T"][i]) - int(meta["T"][j]))
                kind = ("same size" if same_m and d_t == 0 else "transform" if same_m and d_t == 1 else
                        "T only" if same_m else "M only" if d_t == 0 else "M and T")
                rows.append({"i": i, "j": j, "source": meta.source[i], "kind": kind, "distance": distances[i, j],
                             "relative": distances[i, j] / ((nn5[i] + nn5[j]) / 2),
                             "T_ratio": max(meta["T"][i], meta["T"][j]) / min(meta["T"][i], meta["T"][j]),
                             "M_ratio": max(meta.M[i], meta.M[j]) / min(meta.M[i], meta.M[j]), "same_cluster": labels[i] == labels[j]})
    return pd.DataFrame(rows)


def parent_membership(meta, labels, candidates=CANDIDATES):
    """For candidate members whose parent recording exists at several sizes: how many of those sizes the
    candidate keeps, and which it loses."""
    rows = []
    for cluster, letter in candidates.items():
        for parent in meta.parent[labels == cluster].unique():
            group = meta[meta.parent == parent]
            if len(group[["M", "T"]].drop_duplicates()) < 2 or group.source.iloc[0] in {"ForEx", "DJIA", "S&P 500"}:
                continue  # market parents differ by price/log-return transform, not size
            inside = labels[group.row] == cluster
            rows.append({"letter": letter, "parent": parent, "source": group.source.iloc[0], "sizes_in": int(inside.sum()),
                         "sizes": len(group), "lost (M×T)": ", ".join(f"{m}×{t}" for m, t in zip(group.M[~inside], group["T"][~inside])) or "–"})
    return pd.DataFrame(rows)


def family_drop(values, pairs, spis, labels, categories, cluster, probes):
    """Candidate separation, and the rank probe rows give it, after removing every SPI pair that involves
    one SPI category (or one name prefix)."""
    rows = []
    prefix = np.array([s.split("_")[0] for s in spis])
    for name, involved in [("none", np.zeros(len(spis), bool)), ("precision (prec, prec-sq)", np.isin(prefix, ["prec", "prec-sq"]))] + \
            [(c, np.array([categories[s] == c for s in spis])) for c in sorted(set(categories.values()))]:
        keep = ~(involved[pairs[:, 0]] | involved[pairs[:, 1]])
        x = values[:, keep]
        d = squareform(pdist(x))
        rank = other_cluster_rank(d, labels)
        sil = silhouette_samples(d, labels, metric="precomputed")
        rows.append({"removed": name, "features": int(keep.sum()), "silhouette": float(sil[labels == cluster].mean()),
                     "jaccard": best_jaccard(labels, ward(x, labels.max() + 1), cluster),
                     **{f"rank_{p}": float(np.median(rank[r, cluster])) for p, r in probes.items()},
                     "rank_outsiders": float(np.median(rank[labels != cluster, cluster]))})
    return pd.DataFrame(rows)
