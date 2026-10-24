"""Consensus atlas (2026-10-24): feature-space density groups that also hold together on the 2-D map.

Groups are HDBSCAN cores in the first 50 principal components of z, kept if they are stable under
parent-grouped resampling and span at least three parent recordings. The 2-D map is then chosen from a
perplexity x input x exaggeration sweep by how many of those groups it shows as single map regions,
subject to a neighbour-recall floor. Criteria live in configs/analysis/zenodo-consensus-261024.yaml.
Reads the cached tsne-261021 outputs (run `tsne_atlas` first); never runs pyspi.

Run: .venv/bin/python -m research.zenodo.atlas.consensus_atlas   (fits and caches; the notebook
zenodo_1053_consensus_261024.ipynb reads the cache, characterises the groups and writes explorer v4)
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import HDBSCAN
from threadpoolctl import threadpool_limits
import yaml

from research.zenodo.atlas import tsne_atlas as ta
from research.zenodo.atlas.explore_zenodo_geometry import ROOT
from research.zenodo.atlas.explore_zenodo_v2 import distance_neighbors, recall
from research.zenodo.atlas.mt_control import size_coordinates, size_matches

CONFIG = ROOT / "research/zenodo/atlas/configs/analysis/zenodo-consensus-261024.yaml"


def density_groups(scores, dims, min_cluster_size, min_samples):
    return HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, copy=True).fit_predict(scores[:, :dims])


def source_agreement(scores, meta, dims=(5, 10, 20, 50, 100, 200, None), k=5):
    """5-NN agreement with source family, excluding neighbours from the same parent recording."""
    source, parent = meta.source.to_numpy(), meta.parent.to_numpy()
    rows = []
    for d in dims:
        dist = squareform(pdist(scores if d is None else scores[:, :d]))
        np.fill_diagonal(dist, np.inf)
        order = np.argsort(dist, axis=1)
        share = [np.mean(source[[j for j in order[i, :60] if parent[j] != parent[i]][:k]] == source[i]) for i in range(len(dist))]
        rows.append({"input": "full rank" if d is None else f"PCA{d}", "dims": d or scores.shape[1], "share": float(np.mean(share))})
    return pd.DataFrame(rows)


def group_table(labels, meta, d_analysis, d_full, stability):
    """One row per density group: composition, origin, size, parents, separation and stability."""
    from sklearn.metrics import silhouette_samples
    assigned = labels >= 0
    sil_a = np.full(len(labels), np.nan)
    sil_f = np.full(len(labels), np.nan)
    sil_a[assigned] = silhouette_samples(d_analysis[np.ix_(assigned, assigned)], labels[assigned], metric="precomputed")
    sil_f[assigned] = silhouette_samples(d_full[np.ix_(assigned, assigned)], labels[assigned], metric="precomputed")
    rows = []
    for g in np.unique(labels[assigned]):
        members = meta[labels == g]
        sources = members.source.value_counts()
        origins = members.origin.value_counts()
        real, synthetic = int(origins.get("real", 0)), int(origins.get("synthetic", 0))
        rows.append({"group": int(g), "size": len(members), "parents": members.parent.nunique(), "sources": len(sources),
                     "composition": "homogeneous" if len(sources) == 1 else "heterogeneous",
                     "origin": "mixed" if real and synthetic else "real" if real else "synthetic",
                     "cross_source": bool((sources >= 2).sum() >= 2),
                     "silhouette_pca": float(np.nanmean(sil_a[labels == g])), "silhouette_full": float(np.nanmean(sil_f[labels == g])),
                     "M": f"{members['M'].min()}–{members['M'].max()}", "T": f"{members['T'].min()}–{members['T'].max()}",
                     "source_mix": "; ".join(f"{s} {n}" for s, n in sources.items()),
                     "title": " + ".join(sources.index[:3]) + (" + …" if len(sources) > 3 else "")})
    table = pd.DataFrame(rows).merge(stability.rename(columns={"cluster": "group", "jaccard_mean": "stability"}), on="group")
    return table


def map_agreement(xy, labels, groups, modal_share, region_share, region_settings):
    """Per group: share of members in their modal map region, and that region's share of the group."""
    regions = ta.map_clusters(xy, **region_settings)
    out = {}
    for g in groups:
        members = regions[labels == g]
        if not (members >= 0).any():
            out[g] = (0., 0., -1, False)
            continue
        modal = np.bincount(members[members >= 0]).argmax()
        inside = float(np.mean(members == modal))
        own = float(np.mean(labels[regions == modal] == g))
        out[g] = (inside, own, int(modal), inside >= modal_share and own >= region_share)
    return out, regions


def cohesion(distances, rows, rest):
    """1 - mean within-set distance / mean distance from members to everyone else (higher = tighter)."""
    rows = np.asarray(rows)
    within = distances[np.ix_(rows, rows)].sum() / (len(rows) * (len(rows) - 1))
    return 1 - within / distances[np.ix_(rows, rest)].mean()


def size_null(distances, labels, meta, groups, replicates, k, seed):
    """Cohesion against same-size sets from other source families matched member by member on (log M, log T)."""
    rng = np.random.default_rng(seed)
    u = size_coordinates(meta)
    source = meta.source.to_numpy()
    rows = []
    for g in groups:
        members = np.flatnonzero(labels == g)
        pool = np.flatnonzero(~np.isin(source, np.unique(source[members])))
        rest = lambda chosen: np.setdiff1d(np.arange(len(labels)), chosen)
        observed = cohesion(distances, members, rest(members))
        matched, random = [], []
        for _ in range(replicates):
            taken, chosen = np.zeros(len(labels), bool), []
            for i in rng.permutation(members):
                j = rng.choice(size_matches(u, i, pool, taken, k))
                taken[j] = True
                chosen.append(j)
            matched.append(cohesion(distances, chosen, rest(chosen)))
            pick = rng.choice(pool, len(members), replace=False)
            random.append(cohesion(distances, pick, rest(pick)))
        matched, random = np.array(matched), np.array(random)
        rows.append({"group": int(g), "cohesion": observed, "matched_mean": matched.mean(), "random_mean": random.mean(),
                     "p_matched": (1 + np.sum(matched >= observed)) / (1 + replicates),
                     "size_survival": (observed - matched.mean()) / (observed - random.mean())})
    return pd.DataFrame(rows)


def sweep(inputs, labels, pool, reference, config):
    """Every map in the declared grid, scored by how many pool groups it shows as single map regions."""
    s, a = config["sweep"], config["agreement"]
    records, maps = [], {}
    for name in s["inputs"]:
        for p in s["perplexities"]:
            for rho in s["exaggerations"]:
                key = f"{name}-p{p}-x{rho}"
                per_seed, stack = [], []
                for seed in s["seeds"]:
                    xy = ta.tsne(inputs[name], perplexity=p, exaggeration=rho, seed=seed, max_iter=s["max_iter"], threads=config["threads"])
                    agree, regions = map_agreement(xy, labels, pool, a["modal_share"], a["region_share"], config["map_regions"])
                    nn = distance_neighbors(squareform(pdist(xy)), 15)
                    per_seed.append({"agree": sum(v[3] for v in agree.values()),
                                     "covered": int(sum((labels == g).sum() for g, v in agree.items() if v[3])),
                                     "recall15_analysis": recall(reference["analysis"], nn), "recall15_full": recall(reference["full"], nn),
                                     "regions": int(regions.max() + 1), "unassigned": int((regions < 0).sum())})
                    stack.append(xy)
                frame = pd.DataFrame(per_seed)
                records.append({"key": key, "input": name, "perplexity": p, "exaggeration": rho,
                                **frame.mean().to_dict(), "agree_min": int(frame.agree.min()), "agree_max": int(frame.agree.max())})
                maps[key] = np.stack(stack)
                print(f"{key}: agree {frame.agree.mean():.1f} ({frame.agree.min()}–{frame.agree.max()}), "
                      f"recall {frame.recall15_analysis.mean():.3f}", flush=True)
    return pd.DataFrame(records), maps


def choose(grid, floor):
    """Most pool groups shown as single regions (mean over seeds) among maps within the recall floor."""
    eligible = grid[grid.recall15_analysis >= floor * grid.recall15_analysis.max()]
    return eligible.sort_values(["agree", "agree_min", "recall15_analysis"], ascending=False).iloc[0].key


def compute(config, out: Path):
    tick = time.monotonic()
    base = yaml.safe_load((ROOT / config["base"]).read_text())
    cached = ta.load(ROOT / base["output"])
    meta, arrays = cached["meta"], cached["arrays"]
    scores, d_full = arrays["scores_primary"], arrays["full_distances"]
    dims = config["analysis_dims"]
    d_analysis = squareform(pdist(scores[:, :dims]))
    agreement_by_dims = source_agreement(scores, meta)
    density = config["density"]
    labels = density_groups(scores, dims, **density)
    stability = ta.grouped_jaccard(scores, labels, lambda s: density_groups(s, dims, **density), meta.parent.to_numpy(),
                                   replicates=config["stability"]["replicates"], fraction=config["stability"]["fraction"], seed=config["seed"])
    table = group_table(labels, meta, d_analysis, d_full, stability)
    table["pool"] = (table.stability >= config["pool"]["min_stability"]) & (table.parents >= config["pool"]["min_parents"])
    pool = table.group[table.pool].tolist()
    print(f"{len(table)} density groups, {len(pool)} in the pool; {time.monotonic() - tick:.0f} s", flush=True)
    inputs = {f"pca{k}": scores[:, :k] for k in (20, 50, 100)} | {"full": scores}
    reference = {"analysis": distance_neighbors(d_analysis, 15), "full": distance_neighbors(d_full, 15)}
    grid, maps = sweep(inputs, labels, pool, reference, config)
    chosen = choose(grid, config["sweep"]["recall_floor"])
    xy = maps[chosen][0]
    agree, regions = map_agreement(xy, labels, table.group, config["agreement"]["modal_share"], config["agreement"]["region_share"],
                                   config["map_regions"])
    table["in_modal_region"] = [agree[g][0] for g in table.group]
    table["region_share"] = [agree[g][1] for g in table.group]
    table["agrees_on_map"] = [agree[g][3] for g in table.group]
    null = size_null(d_analysis, labels, meta, table.group, config["size_null"]["replicates"], config["size_null"]["k"], config["seed"])
    table = table.merge(null, on="group")
    table["tier"] = np.where(table.pool & table.agrees_on_map, 1, np.where(table.pool, 2, 3))
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "consensus.npz", names=arrays["names"], density=labels, chosen_regions=regions,
                        **{f"map__{k}": v for k, v in maps.items()})
    table.to_csv(out / "groups.csv", index=False)
    grid.to_csv(out / "display-sweep.csv", index=False)
    agreement_by_dims.to_csv(out / "source-agreement-by-dims.csv", index=False)
    (out / "manifest.json").write_text(json.dumps({
        "config": config, "chosen_map": chosen, "seconds": time.monotonic() - tick,
        "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "tsne_cache_manifest_sha256": sha256((ROOT / base["output"] / "manifest.json").read_bytes()).hexdigest()}, indent=1))
    return out


def load(out: Path):
    with np.load(out / "consensus.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    return {"arrays": arrays, "groups": pd.read_csv(out / "groups.csv"), "sweep": pd.read_csv(out / "display-sweep.csv"),
            "source_agreement": pd.read_csv(out / "source-agreement-by-dims.csv"),
            "manifest": json.loads((out / "manifest.json").read_text())}


def main():
    config = yaml.safe_load(CONFIG.read_text())
    with threadpool_limits(limits=config["threads"]):
        compute(config, ROOT / config["output"])


if __name__ == "__main__":
    main()
