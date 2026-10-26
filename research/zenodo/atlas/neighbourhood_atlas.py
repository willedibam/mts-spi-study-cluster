"""Neighbourhoods (2026-10-24): cross-family neighbour enrichment and map-proposed, feature-validated candidates.

Density groups find the tightest places in z, which are mostly single sources. Two questions they cannot answer:
which source families are nearer each other than their sizes and nested windows explain (enrichment), and which
regions a reader sees on the 2-D map are real neighbourhoods in feature space (validation). Also reports the
uncertainty of the density groups' stability and how the pool moves with the cutoff and HDBSCAN setting.
Reads the tsne-261021 and consensus-261024 caches; never runs pyspi.

Run: .venv/bin/python -m research.zenodo.atlas.neighbourhood_atlas   (the notebook
zenodo_1053_neighbourhoods_261024.ipynb reads the cache)
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from threadpoolctl import threadpool_limits
import yaml

from research.zenodo.atlas import consensus_atlas as ca
from research.zenodo.atlas import tsne_atlas as ta
from research.zenodo.atlas.explore_zenodo_geometry import ROOT

CONFIG = ROOT / "research/zenodo/atlas/configs/analysis/zenodo-neighbourhoods-261024.yaml"


def neighbour_edges(distances, parents, k):
    """Directed k-NN edges (source row, neighbour row), skipping neighbours from the same parent recording."""
    d = np.array(distances, float, copy=True)
    d[parents[:, None] == parents[None, :]] = np.inf
    return np.repeat(np.arange(len(d)), k), np.argsort(d, axis=1)[:, :k].ravel()


def parent_strata(meta, bins):
    """Stratum per parent recording from quantile bins of its median log M and log T (one stratum if bins == 0)."""
    sizes = meta.assign(log_M=np.log(meta["M"]), log_T=np.log(meta["T"])).groupby("parent", sort=False)[["log_M", "log_T"]].median()
    if not bins:
        return pd.Series(0, index=sizes.index)
    cut = lambda v: pd.qcut(v, bins, labels=False, duplicates="drop")
    return pd.Series(cut(sizes["log_M"]) * bins + cut(sizes["log_T"]), index=sizes.index)


def enrichment(edges, meta, *, bins, permutations, seed):
    """Edges between each pair of source families against a null that permutes family labels across parent
    recordings within size strata. The graph is fixed, so degree, nesting and parent multiplicity are preserved."""
    src, dst = edges
    families = meta.source.astype("category")
    names, codes = list(families.cat.categories), families.cat.codes.to_numpy().astype(np.int64)  # int8 overflows f * f
    parent_index, parent_keys = pd.factorize(meta.parent)
    parent_family = pd.Series(codes).groupby(parent_index).agg(lambda v: v.iloc[0]).to_numpy()
    assert (pd.Series(codes).groupby(parent_index).nunique() == 1).all(), "a parent recording spans two families"
    strata = parent_strata(meta, bins).reindex(parent_keys).to_numpy()
    groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    f = len(names)

    def pair_counts(rec):
        c = np.bincount(rec[src] * f + rec[dst], minlength=f * f).reshape(f, f)
        return c + c.T - np.diag(np.diag(c))

    observed = pair_counts(codes)
    rng = np.random.default_rng(seed)
    total, square, exceed = np.zeros((f, f)), np.zeros((f, f)), np.zeros((f, f))
    for _ in range(permutations):
        shuffled = parent_family.copy()
        for g in groups:
            shuffled[g] = rng.permutation(parent_family[g])
        null = pair_counts(shuffled[parent_index])
        total += null
        square += null ** 2
        exceed += null >= observed
    mean = total / permutations
    sd = np.sqrt(np.maximum(square / permutations - mean ** 2, 0))
    origin = meta.groupby("source").origin.first()
    parents = meta.groupby("source").parent.nunique()
    recordings = meta.source.value_counts()
    linked = {}
    for i, j in zip(src, dst):
        a, b = sorted((codes[i], codes[j]))
        linked.setdefault((a, b), set()).add(tuple(sorted((parent_index[i], parent_index[j]))))
    rows = []
    for a in range(f):
        for b in range(a, f):
            oa, ob = origin[names[a]], origin[names[b]]
            kind = "within" if a == b else ("model–real" if oa != ob else "model–model" if oa == "synthetic" else "real–real")
            rows.append({"family_a": names[a], "family_b": names[b], "kind": kind, "observed": int(observed[a, b]),
                         "expected": mean[a, b], "null_sd": sd[a, b],
                         "ratio": observed[a, b] / mean[a, b] if mean[a, b] > 0 else np.inf if observed[a, b] else np.nan,
                         "p": (1 + exceed[a, b]) / (1 + permutations), "parent_pairs": len(linked.get((a, b), ())),
                         "parents_a": int(parents[names[a]]), "parents_b": int(parents[names[b]]),
                         "recordings_a": int(recordings[names[a]]), "recordings_b": int(recordings[names[b]])})
    table = pd.DataFrame(rows)
    between = table.kind != "within"
    table.loc[between, "q"] = bh(table.p[between].to_numpy())
    return table


def bh(p):
    """Benjamini–Hochberg adjusted p-values."""
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1)
    return out


def recurrence(regions, maps, settings):
    """Per map region: median and share >= .5 of its best Jaccard match among regions re-found on other maps."""
    clusters = np.unique(regions[regions >= 0])
    best = np.array([[ta.best_jaccard(regions, ta.map_clusters(xy, **settings), c) for c in clusters] for xy in maps])
    return pd.DataFrame({"region": clusters, "recurrence": np.median(best, axis=0), "recurs_share": np.mean(best >= .5, axis=0)})


def validate_regions(regions, meta, distances, edges, density, *, random_sets, seed):
    """Feature-space evidence for each map region: cohesion against random same-size sets, neighbour lift,
    cross-source neighbours, composition, size heterogeneity and overlap with density groups."""
    rng = np.random.default_rng(seed)
    src, dst = edges
    parents, source = meta.parent.to_numpy(), meta.source.to_numpy()
    n, rows = len(regions), []
    everyone = np.arange(n)
    for r in np.unique(regions[regions >= 0]):
        inside = regions == r
        members = np.flatnonzero(inside)
        rest = np.flatnonzero(~inside)
        mine = inside[src]
        expected = np.mean([np.sum(inside & (parents != parents[i])) / np.sum(parents != parents[i]) for i in members])
        within = mine & inside[dst]
        cross = within & (source[src] != source[dst])
        observed = ca.cohesion(distances, members, rest)
        null = []
        for _ in range(random_sets):
            pick = rng.choice(n, len(members), replace=False)
            null.append(ca.cohesion(distances, pick, np.setdiff1d(everyone, pick)))
        null = np.array(null)
        found = meta.iloc[members]
        per_source = found.groupby("source").parent.nunique().sort_values(ascending=False)
        groups = density[members]
        modal = np.bincount(groups[groups >= 0]).argmax() if (groups >= 0).any() else -1
        rows.append({
            "region": int(r), "size": len(members), "parents": found.parent.nunique(), "sources": len(per_source),
            "sources_2p": int((per_source >= 2).sum()),
            "origin": "mixed" if found.origin.nunique() > 1 else found.origin.iloc[0],
            "M_levels": found["M"].nunique(), "M_ratio": found["M"].max() / found["M"].min(),
            "T_levels": found["T"].nunique(), "T_ratio": found["T"].max() / found["T"].min(),
            "cohesion": observed, "cohesion_z": (observed - null.mean()) / null.std(),
            "p_random": (1 + np.sum(null >= observed)) / (1 + random_sets),
            "nn_inside": float(inside[dst][mine].mean()),
            "nn_lift": float(inside[dst][mine].mean() / expected) if expected > 0 else np.nan,  # nan: a one-parent region
            "cross_edges": int(cross.sum()), "within_edges": int(within.sum()),
            "cross_members": float(np.isin(members, np.concatenate([src[cross], dst[cross]])).mean()),
            "in_density": float(np.mean(groups >= 0)), "modal_group": int(modal),
            "modal_group_share": float(np.mean(groups == modal)) if modal >= 0 else 0.,
            "source_mix": "; ".join(f"{s} {int((found.source == s).sum())}/{p}p" for s, p in per_source.items()),
            "title": " + ".join(per_source.index[:3]) + (" + …" if len(per_source) > 3 else ""),
            "M": f"{found['M'].min()}–{found['M'].max()}", "T": f"{found['T'].min()}–{found['T'].max()}"})
    return pd.DataFrame(rows)


PRIORITY = ["A multi-source, size-heterogeneous", "B multi-source", "C second source from one parent", "D size-heterogeneous only"]


def priority(frame):
    """Candidate priority. Multi-source means at least two sources with two or more parent recordings each;
    size-heterogeneous means an M or T ratio of at least 2 (needs a `size_heterogeneous` column)."""
    multi = frame.sources_2p >= 2
    return np.select([multi & frame.size_heterogeneous, multi, frame.sources >= 2], PRIORITY[:3], PRIORITY[3])


def classify_regions(table, tiers):
    """Kind and priority of each validated map region; `tiers` maps density group to tier.
    core: ≥ 80% of members in one tier 1–2 density group. one parent: windows of one recording. neighbourhood: tighter
    than random sets (p ≤ .01), stability ≥ .6 and ≥ 30% of members' neighbours inside. Otherwise map-only.
    Candidates are core or neighbourhood regions that are multi-source (priority A or B)."""
    t = table.copy()
    t["kind"] = np.select([(t.modal_group_share >= .8) & t.modal_group.map(lambda g: tiers.get(g, 3) <= 2), t.parents == 1,
                           (t.p_random <= .01) & (t.stability >= .6) & (t.nn_inside >= .3)],
                          ["core", "one parent", "neighbourhood"], "map-only")
    t["multi_source"] = t.sources_2p >= 2
    t["size_heterogeneous"] = (t.M_ratio >= 2) | (t.T_ratio >= 2)
    t["priority"] = priority(t)
    t["candidate"] = t.kind.isin(["core", "neighbourhood"]) & t.multi_source
    return t


def setting_robustness(scores, labels, dims, density, settings, dims_list):
    """Best Jaccard match of each density group under alternative HDBSCAN settings and analysis dimensions."""
    clusters = np.unique(labels[labels >= 0])
    out = pd.DataFrame({"group": clusters})
    for mcs, ms in settings:
        alt = ca.density_groups(scores, dims, mcs, ms)
        out[f"mcs{mcs}-ms{ms}"] = [ta.best_jaccard(labels, alt, c) for c in clusters]
    for d in dims_list:
        alt = ca.density_groups(scores, d, **density)
        out[f"PCA{d}"] = [ta.best_jaccard(labels, alt, c) for c in clusters]
    return out


def compute(config, out: Path):
    tick = time.monotonic()
    cons = yaml.safe_load((ROOT / config["consensus"]).read_text())
    base = yaml.safe_load((ROOT / config["base"]).read_text())
    cached = ta.load(ROOT / base["output"])
    meta, scores = cached["meta"], cached["arrays"]["scores_primary"]
    C = ca.load(ROOT / cons["output"])
    density, regions, chosen = C["arrays"]["density"], C["arrays"]["chosen_regions"], C["manifest"]["chosen_map"]
    dims = config["analysis_dims"]
    parents = meta.parent.to_numpy()
    distances = squareform(pdist(scores[:, :dims]))

    e = config["enrichment"]
    tables, edges = [], {}
    for k in e["k"]:
        edges[k] = neighbour_edges(distances, parents, k)
        for bins in e["bins"]:
            tables.append(enrichment(edges[k], meta, bins=bins, permutations=e["permutations"], seed=config["seed"]).assign(k=k, bins=bins))
            print(f"enrichment k={k} bins={bins}: {time.monotonic() - tick:.0f} s", flush=True)
    enrich = pd.concat(tables, ignore_index=True)

    r = config["regions"]
    table = validate_regions(regions, meta, distances, edges[r["k"]], density, random_sets=r["random_sets"], seed=config["seed"])
    null = ca.size_null(distances, regions, meta, table.region, r["size_null"]["replicates"], r["size_null"]["k"], config["seed"])
    table = table.merge(null.rename(columns={"group": "region"})[["region", "matched_mean", "random_mean", "p_matched", "size_survival"]], on="region")
    sweep = C["sweep"]
    floor = cons["sweep"]["recall_floor"] * sweep.recall15_analysis.max()
    eligible = sweep.key[sweep.recall15_analysis >= floor].tolist()
    others = [xy for key in eligible for i, xy in enumerate(C["arrays"][f"map__{key}"]) if not (key == chosen and i == 0)]
    table = table.merge(recurrence(regions, others, cons["map_regions"]), on="region")
    name, p, x = chosen.split("-")
    inputs = scores[:, :int(name[3:])] if name.startswith("pca") else scores
    fit = lambda s: ta.map_clusters(ta.tsne(s, perplexity=float(p[1:]), exaggeration=float(x[1:]), seed=config["seed"],
                                            max_iter=cons["sweep"]["max_iter"], threads=config["threads"]), **cons["map_regions"])
    region_stability, region_values = ta.grouped_jaccard(inputs, regions, fit, parents, replicates=r["stability"]["replicates"],
                                                         fraction=r["stability"]["fraction"], seed=config["seed"], values=True)
    table = table.merge(region_stability.rename(columns={"cluster": "region", "jaccard_mean": "stability"}), on="region")
    print(f"regions validated and re-embedded: {time.monotonic() - tick:.0f} s", flush=True)

    d = cons["density"]
    density_stability, density_values = ta.grouped_jaccard(
        scores, density, lambda s: ca.density_groups(s, cons["analysis_dims"], **d), parents,
        replicates=cons["stability"]["replicates"], fraction=cons["stability"]["fraction"], seed=cons["seed"], values=True)
    groups = C["groups"].set_index("group")
    assert np.allclose(density_stability.jaccard_mean, groups.loc[density_stability.cluster, "stability"]), "stability cache mismatch"
    robust = setting_robustness(scores, density, cons["analysis_dims"], d, config["stability"]["settings"], config["stability"]["dims"])

    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "neighbourhoods.npz", names=cached["arrays"]["names"], regions=regions, density=density,
                        region_jaccard=region_values, density_jaccard=density_values,
                        **{f"edges_k{k}": np.stack(v) for k, v in edges.items()})
    enrich.to_csv(out / "enrichment.csv", index=False)
    table.to_csv(out / "regions.csv", index=False)
    robust.to_csv(out / "density-robustness.csv", index=False)
    (out / "manifest.json").write_text(json.dumps({
        "config": config, "chosen_map": chosen, "eligible_maps": eligible, "seconds": time.monotonic() - tick,
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "consensus_manifest_sha256": sha256((ROOT / cons["output"] / "manifest.json").read_bytes()).hexdigest()}, indent=1))
    return out


def load(out: Path):
    with np.load(out / "neighbourhoods.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    return {"arrays": arrays, "enrichment": pd.read_csv(out / "enrichment.csv"), "regions": pd.read_csv(out / "regions.csv"),
            "robustness": pd.read_csv(out / "density-robustness.csv"), "manifest": json.loads((out / "manifest.json").read_text())}


def main():
    config = yaml.safe_load(CONFIG.read_text())
    with threadpool_limits(limits=config["threads"]):
        compute(config, ROOT / config["output"])


if __name__ == "__main__":
    main()
