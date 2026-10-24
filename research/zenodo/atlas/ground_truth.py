"""Synthetic ground truth (2026-10-24): do z-neighbourhoods follow the generators' coupling form?

Among the 487 synthetic recordings of the archive, asks how much of the PCA50 z-distance is explained by an a priori
coupling-form class, by generator family and by recording size (log M, log T), and whether a recording's nearest
other-parent neighbour shares its class but not its generator more often than chance. Classes are declared in
configs/analysis/zenodo-ground-truth-261024.yaml from the archive's own generator equations (Cliff et al.,
arXiv:2201.11941v2, Sec. S2). Reads the tsne-261021 cache and the raw archive; never runs pyspi.

Run: .venv/bin/python -m research.zenodo.atlas.ground_truth   (the notebook
zenodo_1053_ground_truth_261024.ipynb reads the cache)
"""
from __future__ import annotations

from hashlib import sha256
from itertools import combinations, combinations_with_replacement
import json
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from threadpoolctl import threadpool_limits
import yaml

from research.zenodo.atlas import neighbourhood_atlas as na
from research.zenodo.atlas import tsne_atlas as ta
from research.zenodo.atlas.explore_zenodo_geometry import ROOT
from research.zenodo.atlas.mt_control import size_basis

CONFIG = ROOT / "research/zenodo/atlas/configs/analysis/zenodo-ground-truth-261024.yaml"
FIELDS = ["family", "unit", "toolbox", "coupling", "interaction", "lag", "marginal", "observable", "coarse", "fine",
          "evidence", "caveat"]


def descriptors(meta, rules):
    """A priori generator descriptors per row (first matching rule wins). Fails on a row no rule matches or on a rule
    whose family disagrees with the archive's source label."""
    rows = []
    for name, source in zip(meta.dataset, meta.source):
        rule = next((r for r in rules if re.match(r["pattern"], name)), None)
        if rule is None or rule["family"] != source:
            raise ValueError(f"no generator rule for {name} ({source})")
        rows.append({f: rule.get(f, "") for f in FIELDS})
    return pd.DataFrame(rows, index=meta.index)


def size_strata(M, T, edges):
    """Size cell from fixed edges on M and T."""
    return np.digitize(M, edges["M"]) * (len(edges["T"]) + 1) + np.digitize(T, edges["T"])


def onehot(labels):
    codes, levels = pd.factorize(np.asarray(labels))
    return np.eye(len(levels))[codes]


class Partition:
    """Explained sums of squares of a (weighted) centred response for unions of design blocks. With Euclidean
    distances this is distance-based variance partitioning (PERMANOVA / db-RDA) on the coordinates themselves."""

    def __init__(self, Y, blocks, weights=None):
        self.w = np.ones(len(Y)) if weights is None else np.asarray(weights, float)
        self.Y = self.prepare(Y)
        self.total = float((self.Y ** 2).sum())
        self.blocks = blocks
        self.n = self.w.sum() ** 2 / (self.w ** 2).sum()  # Kish effective size; the row count when unweighted
        self._cache = {}

    def prepare(self, X):
        X = np.asarray(X, float)
        return (X - np.average(X, axis=0, weights=self.w)) * np.sqrt(self.w)[:, None]

    @staticmethod
    def basis(X):
        if X.shape[1] == 0:
            return X
        U, s, _ = np.linalg.svd(X, full_matrices=False)
        return U[:, s > max(s.max(), 1e-300) * max(X.shape) * np.finfo(float).eps]

    def given(self, names):
        """Orthonormal basis of the given blocks and the response's residual after them (cached)."""
        key = tuple(sorted(names))
        if key not in self._cache:
            G = self.basis(self.prepare(np.column_stack([self.blocks[b] for b in key]))) if key else np.zeros((len(self.Y), 0))
            self._cache[key] = (G, self.Y - G @ (G.T @ self.Y))
        return self._cache[key]

    def fit(self, names):
        G, residual = self.given(names)
        return self.total - float((residual ** 2).sum()), G.shape[1]

    def partial(self, X, given):
        """Sum of squares, df and pseudo-F of an effect design X after the given blocks."""
        G, residual = self.given(given)
        Xr = self.prepare(X)
        U = self.basis(Xr - G @ (G.T @ Xr))
        ss, q = float(((U.T @ residual) ** 2).sum()), U.shape[1]
        df = self.n - G.shape[1] - q - 1
        return ss, q, (ss / q) / ((float((residual ** 2).sum()) - ss) / df) if q else np.nan


def adjusted(r2, n, q):
    return 1 - (1 - r2) * (n - 1) / (n - q - 1)


def subsets(Y, blocks, weights=None):
    """R² and adjusted R² of every non-empty union of blocks, keyed by frozenset of block names."""
    part = Partition(Y, blocks, weights)
    out = {}
    for k in range(1, len(blocks) + 1):
        for names in combinations(blocks, k):
            ss, q = part.fit(names)
            out[frozenset(names)] = {"r2": ss / part.total, "adj_r2": adjusted(ss / part.total, part.n, q), "df": q}
    return out


def venn3(table, a, b, c):
    """Legendre's fractions of three blocks from the adjusted R² of their seven unions (they sum to [abc])."""
    R = lambda *k: table[frozenset(k)]["adj_r2"]
    full = R(a, b, c)
    return {f"[{a}]": full - R(b, c), f"[{b}]": full - R(a, c), f"[{c}]": full - R(a, b),
            f"[{a}·{b}]": R(a, c) + R(b, c) - R(c) - full, f"[{a}·{c}]": R(a, b) + R(b, c) - R(b) - full,
            f"[{b}·{c}]": R(a, b) + R(a, c) - R(a) - full,
            f"[{a}·{b}·{c}]": R(a) + R(b) + R(c) - R(a, b) - R(a, c) - R(b, c) + full, "residual": 1 - full}


class GroupShuffler:
    """Permute a group-level value among groups of the same stratum; rows follow their group."""

    def __init__(self, groups, values, strata=None):
        codes, _ = pd.factorize(np.asarray(groups))
        strata = np.zeros(len(codes), int) if strata is None else pd.factorize(np.asarray(strata))[0]
        first = pd.Series(np.arange(len(codes))).groupby(codes).first().to_numpy()
        values = np.asarray(values)
        for name, x in [("value", values), ("stratum", strata)]:
            if not (pd.Series(x).groupby(codes).nunique() == 1).all():
                raise ValueError(f"{name} varies within a group")
        self.codes, self.values = codes, values[first]
        self.members = [np.flatnonzero(strata[first] == s) for s in np.unique(strata)]

    def __call__(self, rng):
        v = self.values.copy()
        for m in self.members:
            v[m] = v[rng.permutation(m)]
        return v[self.codes]


def within_group_rows(groups, rng):
    """Row order permuted within each group (for size labels within parent recordings)."""
    order = np.arange(len(groups))
    for g in np.unique(groups):
        rows = np.flatnonzero(groups == g)
        order[rows] = rng.permutation(rows)
    return order


def permutation_test(part, effect, given, draw, permutations, rng):
    """One-sided pseudo-F test of an effect design after the given blocks; `draw(rng)` returns a permuted design."""
    ss, q, observed = part.partial(effect, given)
    null = np.array([part.partial(draw(rng), given) for _ in range(permutations)])
    return {"F": observed, "df": q, "null_F_median": float(np.median(null[:, 2])), "partial_r2": ss / part.total,
            "null_partial_r2_median": float(np.median(null[:, 0])) / part.total,
            "null_partial_r2_q975": float(np.quantile(null[:, 0], .975)) / part.total,
            "p": (1 + np.sum(null[:, 2] >= observed)) / (1 + permutations)}


def design_blocks(frame, scheme):
    return {"class": onehot(frame[scheme]), "family": onehot(frame.source), "size": size_basis(frame["M"], frame["T"]),
            "toolbox": onehot(frame.toolbox), "parent": onehot(frame.parent)}


def parent_weights(parents):
    return 1 / pd.Series(parents).map(pd.Series(parents).value_counts()).to_numpy()


def fractions(Y, frame, scheme, weighting):
    blocks = design_blocks(frame, scheme)
    w = parent_weights(frame.parent) if weighting == "parent" else None
    table = subsets(Y, {k: blocks[k] for k in ("class", "family", "size")}, w)
    out = venn3(table, "class", "family", "size")
    for k in ("class", "family", "size"):
        out[f"R2adj {k}"] = table[frozenset([k])]["adj_r2"]
    extra = subsets(Y, {k: blocks[k] for k in ("class", "toolbox", "size")}, w)
    out["[class | toolbox, size]"] = extra[frozenset(["class", "toolbox", "size"])]["adj_r2"] - extra[frozenset(["toolbox", "size"])]["adj_r2"]
    parent = subsets(Y, {"parent": blocks["parent"]}, w)[frozenset(["parent"])]
    out["R2adj parent"] = parent["adj_r2"]
    return out


def jackknife_fractions(Y, frame, scheme, weighting):
    """Delete-one-parent jackknife standard errors of the fractions. A bootstrap that resamples parents with
    replacement duplicates them, which shrinks the spread between parents of a family and inflates every
    between-parent fraction (its percentile intervals sat above the estimates); deleting parents avoids that."""
    reps = pd.DataFrame([fractions(Y[keep], frame[keep].reset_index(drop=True), scheme, weighting)
                         for keep in ((frame.parent != p).to_numpy() for p in frame.parent.unique())])
    return np.sqrt((len(reps) - 1) / len(reps) * ((reps - reps.mean()) ** 2).sum())


def fraction_tests(Y, frame, scheme, weighting, strata_of_parent, permutations, rng):
    """Permutation tests of the partial effects, each with the null that matches its question."""
    blocks = design_blocks(frame, scheme)
    w = parent_weights(frame.parent) if weighting == "parent" else None
    part = Partition(Y, blocks, w)
    cls, fam = frame[scheme].to_numpy(), frame.source.to_numpy()
    units = GroupShuffler(frame.unit, cls)
    in_family = GroupShuffler(frame.parent, cls, fam)
    in_size = GroupShuffler(frame.parent, cls, frame.parent.map(strata_of_parent))
    family_in_class = GroupShuffler(frame.parent, fam, cls)
    M, T, parents = frame["M"].to_numpy(), frame["T"].to_numpy(), frame.parent.to_numpy()
    resize = lambda r: (lambda o: size_basis(M[o], T[o]))(within_group_rows(parents, r))
    tests = [("class", ["size"], lambda r: onehot(units(r)), "class labels permuted across generator units"),
             ("class", ["size"], lambda r: onehot(in_size(r)), "class labels permuted across parents within size strata"),
             ("class", ["toolbox", "size"], lambda r: onehot(units(r)), "class labels permuted across generator units"),
             ("class", ["family", "size"], lambda r: onehot(in_family(r)), "class labels permuted across parents within family"),
             ("family", ["class", "size"], lambda r: onehot(family_in_class(r)), "family labels permuted across parents within class"),
             ("size", ["class", "family"], resize, "(M, T) permuted across windows within parent"),
             ("size", ["parent"], resize, "(M, T) permuted across windows within parent")]
    rows = []
    for effect, given, draw, null in tests:
        res = permutation_test(part, blocks[effect], given, draw, permutations, rng)
        rows.append({"scheme": scheme, "weighting": weighting, "effect": effect, "given": " + ".join(given), "null": null, **res})
    return pd.DataFrame(rows)


def other_parent_neighbours(D, parents, k, strata=None):
    """Directed k-NN edges skipping the own parent recording, optionally only within the same size stratum."""
    d = np.array(D, float, copy=True)
    d[parents[:, None] == parents[None, :]] = np.inf
    if strata is not None:
        d[strata[:, None] != strata[None, :]] = np.inf
    order = np.argsort(d, axis=1)[:, :k]
    valid = np.isfinite(np.take_along_axis(d, order, axis=1)).ravel()
    return np.repeat(np.arange(len(d)), k)[valid], order.ravel()[valid]


def edge_kinds(src, dst, family, cls):
    same_family = family[src] == family[dst]
    same_class = cls[src] == cls[dst]
    return same_family, ~same_family & same_class, ~same_family & ~same_class


def nn_test(edges, frame, scheme, strata_of_parent, permutations, bootstrap, rng):
    """Share of neighbour edges that join a different family of the same class, against class labels permuted
    (a) across generator units and (b) across parents within size strata; family labels and the graph stay fixed."""
    src, dst = edges
    fam, cls = frame.source.to_numpy(), frame[scheme].to_numpy()
    same_family, cross_same, cross_other = edge_kinds(src, dst, fam, cls)
    nulls = {"units": GroupShuffler(frame.unit, cls),
             "parents in size strata": GroupShuffler(frame.parent, cls, frame.parent.map(strata_of_parent))}
    units, unit = np.unique(frame.unit.to_numpy()[src], return_inverse=True)
    per_source = np.bincount(unit, minlength=len(units))
    per_unit = {"observed": np.bincount(unit, cross_same, len(units)) / per_source}
    rows = []
    by_parent = pd.Series(np.arange(len(src))).groupby(frame.parent.to_numpy()[src]).apply(np.array).to_numpy()
    boot = [cross_same[np.concatenate(rng.choice(by_parent, len(by_parent), replace=True))].mean() for _ in range(bootstrap)]
    for name, shuffle in nulls.items():
        null, unit_null = [], np.zeros(len(units))
        for _ in range(permutations):
            permuted = shuffle(rng)
            hit = ~same_family & (permuted[src] == permuted[dst])
            null.append(hit.mean())
            unit_null += np.bincount(unit, hit, len(units))
        null = np.array(null)
        per_unit[f"expected ({name})"] = unit_null / permutations / per_source
        rows.append({"scheme": scheme, "null": name, "edges": len(src), "same_family": same_family.mean(),
                     "cross_same_class": cross_same.mean(), "ci_low": np.quantile(boot, .025), "ci_high": np.quantile(boot, .975),
                     "cross_other_class": cross_other.mean(), "given_cross": cross_same.sum() / max((~same_family).sum(), 1),
                     "expected": null.mean(), "null_q975": np.quantile(null, .975), "ratio": cross_same.mean() / null.mean(),
                     "p": (1 + np.sum(null >= cross_same.mean())) / (1 + permutations)})
    return pd.DataFrame(rows), pd.DataFrame(per_unit, index=pd.Index(units, name="unit")).reset_index().assign(scheme=scheme)


def unit_table(frame):
    """One row per generator unit: family, toolbox, classes and counts."""
    info = frame.groupby("unit")[["source", "toolbox", "coarse", "fine"]].first()
    return info.assign(parents=frame.groupby("unit").parent.nunique(), recordings=frame.unit.value_counts())


def edge_table(edges, frame):
    """Cross-family neighbour edges by unordered generator-unit pair, with class agreement under both schemes."""
    src, dst = edges
    unit, parent = frame.unit.to_numpy(), frame.parent.to_numpy()
    cross = frame.source.to_numpy()[src] != frame.source.to_numpy()[dst]
    s, d = src[cross], dst[cross]
    swap = unit[s] > unit[d]
    a, b = np.where(swap, unit[d], unit[s]), np.where(swap, unit[s], unit[d])
    pairs = [tuple(sorted(p)) for p in zip(parent[s], parent[d])]
    table = (pd.DataFrame({"unit_a": a, "unit_b": b, "parent_pair": pairs}).groupby(["unit_a", "unit_b"])
             .agg(edges=("parent_pair", "size"), parent_pairs=("parent_pair", "nunique")).reset_index())
    info = unit_table(frame)
    for scheme in ("coarse", "fine"):
        table[f"{scheme}_a"], table[f"{scheme}_b"] = info.loc[table.unit_a, scheme].to_numpy(), info.loc[table.unit_b, scheme].to_numpy()
        table[f"{scheme}_match"] = table[f"{scheme}_a"] == table[f"{scheme}_b"]
    return table.sort_values("edges", ascending=False, ignore_index=True)


def unit_distances(D, frame):
    """Size-matched distance between generator units. Within each size stratum both units occupy, the mean
    other-parent distance between their recordings is divided by that stratum's median other-parent distance
    (z is noisier at small M); strata are then averaged with equal weight."""
    strata, unit, parent = frame.stratum.to_numpy(), frame.unit.to_numpy(), frame.parent.to_numpy()
    other = parent[:, None] != parent[None, :]
    scale = {s: np.median(D[np.ix_(strata == s, strata == s)][other[np.ix_(strata == s, strata == s)]]) for s in np.unique(strata)}
    rows = []
    for a, b in combinations_with_replacement(sorted(np.unique(unit)), 2):
        values = []
        for s in np.intersect1d(strata[unit == a], strata[unit == b]):
            ia, ib = np.flatnonzero((unit == a) & (strata == s)), np.flatnonzero((unit == b) & (strata == s))
            mask = other[np.ix_(ia, ib)]
            if mask.any():
                values.append(D[np.ix_(ia, ib)][mask].mean() / scale[s])
        rows.append({"unit_a": a, "unit_b": b, "distance": np.mean(values) if values else np.nan, "strata": len(values)})
    return pd.DataFrame(rows)


def proximity_test(distances, info, scheme, permutations, rng):
    """Between generator units of different families: 1 - mean size-matched distance of same-class pairs / mean of
    other-class pairs, against class labels permuted across units."""
    d = distances[(distances.unit_a != distances.unit_b) & distances.distance.notna()]
    d = d[info.loc[d.unit_a, "source"].to_numpy() != info.loc[d.unit_b, "source"].to_numpy()]
    index = {u: i for i, u in enumerate(info.index)}
    i, j, x = d.unit_a.map(index).to_numpy(), d.unit_b.map(index).to_numpy(), d.distance.to_numpy()
    cls = info[scheme].to_numpy()

    def stat(c):
        same = c[i] == c[j]
        return 1 - x[same].mean() / x[~same].mean() if same.any() else np.nan

    observed = stat(cls)
    null = np.array([stat(rng.permutation(cls)) for _ in range(permutations)])
    null = null[np.isfinite(null)]
    return {"scheme": scheme, "unit_pairs": len(x), "same_class_pairs": int((cls[i] == cls[j]).sum()),
            "proximity": observed, "expected": null.mean(), "null_q975": np.quantile(null, .975),
            "p": (1 + np.sum(null >= observed)) / (1 + len(null))}


def raw_checks(arrays, names, max_lag=5):
    """Descriptor verification on the raw series (never used to assign classes): range, lag-1 autocorrelation,
    linear-trend R², mean |zero-lag correlation|, and the share of channel pairs whose largest |cross-correlation|
    within ±max_lag samples is at a non-zero lag."""
    rows = []
    for name in names:
        x = np.asarray(arrays[name], float)
        z = (x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True)
        T = z.shape[1]
        ac1 = (z[:, :-1] * z[:, 1:]).mean(1)
        trend = np.corrcoef(np.vstack([np.arange(T), x]))[0, 1:] ** 2
        lagged = np.stack([z[:, :T - l] @ z[:, l:].T / (T - l) for l in range(max_lag + 1)])
        both = np.concatenate([lagged[:0:-1].transpose(0, 2, 1), lagged])  # lags -max_lag..max_lag
        iu = np.triu_indices(len(z), 1)
        pair = np.abs(both[:, iu[0], iu[1]])
        rows.append({"dataset": name, "min": x.min(), "max": x.max(), "ac1": ac1.mean(), "abs_ac1": np.abs(ac1).mean(),
                     "trend_r2": trend.mean(), "abs_r0": pair[max_lag].mean(), "peak_abs_xcorr": pair.max(0).mean(),
                     "lagged_peak_share": float(np.mean(pair.argmax(0) != max_lag))})
    return pd.DataFrame(rows)


def compute(config, out: Path):
    tick = time.monotonic()
    base = yaml.safe_load((ROOT / config["base"]).read_text())
    cached = ta.load(ROOT / base["output"])
    corpus = ta.load_corpus(base)
    meta = corpus["meta"]
    if not np.array_equal(meta.dataset.to_numpy(), cached["arrays"]["names"]):
        raise ValueError("cached scores and corpus metadata are not row-aligned")
    synthetic = (meta.origin == "synthetic").to_numpy()
    frame = meta[synthetic].reset_index(drop=True)
    frame = frame.join(descriptors(frame, config["generators"]).drop(columns="family"))
    frame["stratum"] = size_strata(frame["M"], frame["T"], config["size_strata"])
    median = frame.groupby("parent")[["M", "T"]].median()
    strata_of_parent = pd.Series(size_strata(median["M"], median["T"], config["size_strata"]), index=median.index)
    frame["parent_stratum"] = frame.parent.map(strata_of_parent)
    scores = cached["arrays"]["scores_primary"][:, :config["analysis_dims"]]
    Y = scores[synthetic]
    D = squareform(pdist(Y))
    parents = frame.parent.to_numpy()
    rng = lambda offset: np.random.default_rng(config["seed"] + offset)
    schemes, perms, boots = config["schemes"], config["permutations"], config["bootstrap"]

    rows, tests, lofo = [], [], []
    for scheme in schemes:
        for weighting in ("rows", "parent"):
            observed = fractions(Y, frame, scheme, weighting)
            se = jackknife_fractions(Y, frame, scheme, weighting)
            rows += [{"scheme": scheme, "weighting": weighting, "fraction": key, "value": value, "se": se[key],
                      "ci_low": value - 1.96 * se[key], "ci_high": value + 1.96 * se[key]} for key, value in observed.items()]
            tests.append(fraction_tests(Y, frame, scheme, weighting, strata_of_parent, perms, rng(2)))
            print(f"fractions {scheme}/{weighting}: {time.monotonic() - tick:.0f} s", flush=True)
        for family in sorted(frame.source.unique()):
            keep = (frame.source != family).to_numpy()
            lofo.append({"scheme": scheme, "dropped": family, **fractions(Y[keep], frame[keep].reset_index(drop=True), scheme, "rows")})
    fraction_table, test_table, lofo_table = pd.DataFrame(rows), pd.concat(tests, ignore_index=True), pd.DataFrame(lofo)

    nn, per_unit, edge_tables = [], [], []
    for k in config["knn"]:
        for graph, strata in [("any size", None), ("same size stratum", frame.stratum.to_numpy())]:
            edges = other_parent_neighbours(D, parents, k, strata)
            for scheme in schemes:
                summary, units = nn_test(edges, frame, scheme, strata_of_parent, perms, boots, rng(3))
                nn.append(summary.assign(k=k, graph=graph))
                per_unit.append(units.assign(k=k, graph=graph))
            edge_tables.append(edge_table(edges, frame).assign(k=k, graph=graph))
    print(f"nearest neighbours: {time.monotonic() - tick:.0f} s", flush=True)

    e = config["enrichment"]
    info = unit_table(frame)
    enrich = na.enrichment(na.neighbour_edges(D, parents, e["k"]), frame.assign(source=frame.unit), bins=e["bins"],
                           permutations=e["permutations"], seed=config["seed"])
    for scheme in schemes:
        enrich[f"{scheme}_match"] = info.loc[enrich.family_a, scheme].to_numpy() == info.loc[enrich.family_b, scheme].to_numpy()
    enrich["same_family"] = info.loc[enrich.family_a, "source"].to_numpy() == info.loc[enrich.family_b, "source"].to_numpy()
    distances = unit_distances(D, frame)
    proximity = pd.DataFrame([proximity_test(distances, info, scheme, perms, rng(4)) for scheme in schemes])
    all_src, all_dst = other_parent_neighbours(squareform(pdist(scores)), meta.parent.to_numpy(), 1)
    real_neighbour = float(np.mean(~synthetic[all_dst[synthetic[all_src]]]))
    raw = raw_checks(corpus["arrays"], frame.dataset)
    print(f"enrichment, unit distances, raw checks: {time.monotonic() - tick:.0f} s", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    frame.drop(columns=["tags"]).to_csv(out / "descriptors.csv", index=False)
    info.to_csv(out / "units.csv")
    fraction_table.to_csv(out / "fractions.csv", index=False)
    test_table.to_csv(out / "tests.csv", index=False)
    lofo_table.to_csv(out / "lofo.csv", index=False)
    pd.concat(nn, ignore_index=True).to_csv(out / "nn.csv", index=False)
    pd.concat(per_unit, ignore_index=True).to_csv(out / "nn-units.csv", index=False)
    pd.concat(edge_tables, ignore_index=True).to_csv(out / "nn-edges.csv", index=False)
    enrich.to_csv(out / "enrichment.csv", index=False)
    distances.to_csv(out / "unit-distances.csv", index=False)
    proximity.to_csv(out / "proximity.csv", index=False)
    raw.to_csv(out / "raw-checks.csv", index=False)
    (out / "manifest.json").write_text(json.dumps({
        "config": config, "seconds": time.monotonic() - tick, "recordings": len(frame), "parents": int(frame.parent.nunique()),
        "units": len(info), "families": int(frame.source.nunique()), "synthetic_nn_is_real_share": real_neighbour,
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "config_sha256": sha256(CONFIG.read_bytes()).hexdigest(),
        "tsne_manifest_sha256": sha256((ROOT / base["output"] / "manifest.json").read_bytes()).hexdigest()}, indent=1))
    return out


def load(out: Path):
    names = ["descriptors", "fractions", "tests", "lofo", "nn", "nn-units", "nn-edges", "enrichment", "unit-distances",
             "proximity", "raw-checks"]
    tables = {n: pd.read_csv(out / f"{n}.csv") for n in names}
    tables["units"] = pd.read_csv(out / "units.csv", index_col=0)
    tables["manifest"] = json.loads((out / "manifest.json").read_text())
    return tables


def main():
    config = yaml.safe_load(CONFIG.read_text())
    with threadpool_limits(limits=config["threads"]):
        compute(config, ROOT / config["output"])


if __name__ == "__main__":
    main()
