"""Render complete reviewed groups and add direct links to the offline gallery."""
from __future__ import annotations

import html
import json
from pathlib import Path
import textwrap
from urllib.parse import urlencode

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.signal import detrend
from threadpoolctl import threadpool_limits
from scripts.explore_zenodo_geometry import ROOT, paper_style, write_gallery

OUT = ROOT / "results/zenodo_7118947/visual-exploration"


def dynamical_descriptors(data):
    """Transparent sample-scale summaries; not mechanistic classifiers."""
    x = np.asarray(data, dtype=float)
    centered = x - x.mean(axis=1, keepdims=True)
    residual = detrend(centered, axis=1)
    variance = np.var(centered, axis=1)
    def common_fraction(a):
        scale = np.std(a, axis=1, keepdims=True)
        z = a / np.where(scale > 0, scale, 1)
        eigenvalues = np.linalg.eigvalsh(z @ z.T / z.shape[1])
        return float(eigenvalues[-1] / eigenvalues.sum()) if eigenvalues.sum() else np.nan
    lag = [np.corrcoef(row[:-1], row[1:])[0, 1] for row in centered]
    return {"M":x.shape[0], "T":x.shape[1], "median_lag1":float(np.median(lag)),
            "median_linear_trend_R2":float(np.median(1 - np.var(residual, axis=1) / variance)),
            "shared_PC1_fraction":common_fraction(centered),
            "detrended_shared_PC1_fraction":common_fraction(residual)}


def exact_prefix_pairs(arrays):
    """Identify unequal-length, exactly overlapping windows; no tolerance matching."""
    pairs = []
    names = list(arrays)
    for i, first in enumerate(names):
        for second in names[i+1:]:
            short, long = sorted([first, second], key=lambda name: arrays[name].shape[1])
            a, b = arrays[short], arrays[long]
            if a.shape[0] == b.shape[0] and a.shape[1] < b.shape[1] and np.array_equal(a, b[:, :a.shape[1]]):
                pairs.append({"short":short, "long":long, "shared_samples":a.shape[1]})
    return pairs


def run():
    paper_style()
    selections = json.loads((ROOT / "configs/analysis/zenodo-cluster-shortlist.json").read_text())
    audit = pd.read_csv(OUT / "small-cluster-audit.csv")
    images = pd.read_csv(OUT / "heatmap-index.csv").set_index("dataset")
    members = pd.read_csv(OUT / "small-cluster-members.csv")
    parts, summary, exported_members, descriptors = [], [], [], []
    needed = set(members[members.partition.isin([e["partition"] for e in selections])].dataset)
    needed.update(audit[(audit.partition.isin([e["partition"] for e in selections]))].nearest_outsider)
    with np.load(ROOT / "data/zenodo_7118947/database.npz", allow_pickle=False) as database:
        raw_descriptors = {name:dynamical_descriptors(database[name]) for name in sorted(needed)}
        prefix_relations = {}
        for entry in selections:
            r = audit[(audit.partition == entry["partition"]) & (audit.cluster == entry["cluster"])].iloc[0]
            names = members[(members.partition == entry["partition"]) & (members.cluster == entry["cluster"])].dataset.tolist()
            arrays = {name:database[name] for name in names + [r.nearest_outsider]}
            prefix_relations[entry["id"]] = exact_prefix_pairs(arrays)
    target = OUT / "shortlist"; target.mkdir(exist_ok=True)
    for entry in selections:
        r = audit[(audit.partition == entry["partition"]) & (audit.cluster == entry["cluster"])].iloc[0]
        group = members[(members.partition == entry["partition"]) & (members.cluster == entry["cluster"])].copy()
        group["shortlist"] = entry["id"]
        group["composition"] = entry["composition"]
        group["origin"] = entry["origin"]
        exported_members.extend(group.to_dict("records"))
        names = group.dataset.tolist()
        # Show the full-feature medoid first, followed by all other members.
        names = [r.medoid] + [name for name in names if name != r.medoid]
        nrows = int(np.ceil(len(names) / 3))
        fig, axes = plt.subplots(nrows, 3, figsize=(10.5, nrows * 1.65),
                                 constrained_layout=True, squeeze=False)
        for ax, name in zip(axes.flat, names):
            im = images.loc[name]
            with Image.open(OUT / im.png) as picture:
                ax.imshow(np.asarray(picture), aspect="auto", interpolation="nearest", extent=(0, im["T"], im.M, 0))
            label = name + (" [medoid]" if name == r.medoid else "")
            ax.set_title("\n".join(textwrap.wrap(label, 48)), fontsize=7, usetex=False)
            ax.set(xlabel="Observation", ylabel="Process")
            ax.tick_params(labelsize=7)
        for ax in axes.flat[len(names):]:
            ax.set_visible(False)
        fig.suptitle(entry["title"], fontsize=10)
        fig.savefig(target / f"{entry['id']}.svg")
        fig.savefig(target / f"{entry['id']}.png", dpi=180)
        plt.close(fig)
        link = "index.html?" + urlencode({"embedding":"tsne-pca50-p30", "partition":entry["partition"], "cluster":entry["cluster"]})
        row = {**entry, "size":int(r["size"]), "full_silhouette":float(r.full_silhouette),
               "complete_silhouette":float(r.complete_silhouette),
               "primary_feature_validity":float(r.primary_feature_validity),
               "jaccard":float(r.subsample_jaccard_mean), "min_jaccard":float(r.subsample_jaccard_min),
               "exact_prefix_pairs":prefix_relations[entry["id"]],
               "nearest_outsider":r.nearest_outsider, "medoid":r.medoid, "gallery":link}
        summary.append(row)
        description_rows = [{"dataset":name, "role":"member", **raw_descriptors[name]} for name in names]
        description_rows.append({"dataset":r.nearest_outsider, "role":"nearest excluded dataset", **raw_descriptors[r.nearest_outsider]})
        descriptors.extend({"shortlist":entry["id"], **item} for item in description_rows)
        descriptor_table = pd.DataFrame(description_rows).round(3).to_html(index=False, escape=True)
        outsider = images.loc[r.nearest_outsider]
        parts.append(f'<section><h2>{html.escape(entry["title"])}</h2><p>{html.escape(entry["composition"].title())} · {html.escape(entry["origin"])}; '
            f'{len(names)} members. Full-feature silhouette {r.full_silhouette:.2f}; mean matched Jaccard '
            f'{r.subsample_jaccard_mean:.2f} (minimum {r.subsample_jaccard_min:.2f}) across five 80% subsamples.</p>'
            f'<p>Classes used here: {html.escape(", ".join(entry["classes"]))}.</p>'
            f'<p>Observed entries in retained features: {r.primary_feature_validity:.1%}. '
            f'Silhouette using only 9,313 fully observed varying features: {r.complete_silhouette:.2f}.</p>'
            f'<p>{html.escape(entry["note"])}</p>'
            + ('<p><strong>Exact overlapping windows:</strong> ' +
               '; '.join(html.escape(p['short']) + ' is a prefix of ' + html.escape(p['long']) for p in prefix_relations[entry['id']]) +
               '. These rows are not independent simulations; row-resampling stability does not establish independent replication.</p>'
               if prefix_relations[entry['id']] else '') +
            f'<p><a href="{html.escape(link)}">Inspect this group</a> · '
            f'<a href="shortlist/{entry["id"]}.svg">SVG sheet</a></p>'
            f'<img src="shortlist/{entry["id"]}.png" alt="All members of {html.escape(entry["title"])}">'
            f'<details><summary>Full membership and source tags</summary>{group[["dataset","source","tags"]].to_html(index=False,escape=True)}</details>'
            f'<details><summary>Measured dynamics and nearest excluded dataset</summary>{descriptor_table}'
            f'<p>Nearest excluded dataset to the full-feature medoid: {html.escape(r.nearest_outsider)}.</p>'
            f'<img src="{html.escape(outsider.png)}" alt="Nearest excluded MTS">'
            f'<p><a href="{html.escape(outsider.svg)}">Excluded-series SVG</a>. This comparison tests how convincing the proposed boundary is.</p></details></section>')
    (OUT / "shortlist.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(summary).to_csv(OUT / "shortlist.csv", index=False)
    pd.DataFrame(exported_members).to_csv(OUT / "shortlist-members.csv", index=False)
    pd.DataFrame(descriptors).to_csv(OUT / "shortlist-dynamics.csv", index=False)
    (OUT / "shortlist.html").write_text('''<!doctype html><meta charset="utf-8"><title>Reviewed MTS groups</title>
<style>body{max-width:1100px;margin:35px auto;padding:0 24px;font:16px Georgia,serif;color:#222}h1,h2{font-weight:normal}
p{line-height:1.5}section{margin:45px 0}img{max-width:100%}table{font:12px sans-serif;border-collapse:collapse}td,th{padding:7px;text-align:left;overflow-wrap:anywhere}a{color:#36536b}</style>
<h1>Representative groups for inspection</h1><p><a href="index.html">Full gallery</a> · <a href="figures/high-k-resolutions.svg">Higher-k comparison</a> · <a href="shortlist-members.csv">Selected-group membership CSV</a></p>
<p>Post-hoc shortlist drawn from all tested resolutions. Every member of each selected group is shown, not just attractive examples.
Titles describe source composition; cross-model or cross-source mechanism remains a hypothesis. Per-process robust scaling and per-MTS colour limits match the gallery.
Jaccard measures membership reproducibility under subsampling with PCA fixed, not independent scientific validation. Shared recordings/source protocols and missing-feature patterns can also cause similarity.</p>
<p>Composition and origin are separate: homogeneous = one listed class, heterogeneous = several; real/synthetic/mixed describe provenance.
Class granularity is stated per group: changing M,T or a model parameter is not automatically a new class. These labels say nothing by themselves about dynamical coherence.</p>
<p>The new inspection tables compare every member with the nearest excluded dataset. Lag-1 correlation measures smoothness in sample units;
linear-trend R² measures channel-wise drift; the shared PC1 fraction summarizes linear cofluctuation of standardized channels, before/after removing a linear trend.
They do not measure physical frequency or establish synchronization, and M,T affect their estimates. These descriptive checks overlap with some SPI information and are not independent validation.</p>
''' + "\n".join(parts))
    with np.load(OUT / "exploration.npz", allow_pickle=False) as archive:
        data = {key:archive[key] for key in archive.files}
    manifest = json.loads((OUT / "manifest.json").read_text())
    embeddings = {key:{**meta,"rows":data[f"rows__{key}"],"coordinates":data[f"embedding__{key}"]}
                  for key,meta in manifest["embeddings"].items()}
    partitions = {key:data[f"cluster__{key}"] for key in manifest["partitions"]}
    write_gallery(OUT, data["names"], embeddings, partitions)
    print("Rendered shortlist:", len(summary), "complete groups")


if __name__ == "__main__":
    with threadpool_limits(limits=8):
        run()
