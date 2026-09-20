"""Independent numerical replay of baseline outputs and original-file integrity."""
from pathlib import Path
import hashlib
import json
import subprocess
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/spi_baseline_exploration_260921"
OUT = ROOT / "results/spi_baseline_exploration_260921"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def independent_standardization(values, dev):
    keep = np.flatnonzero(np.isfinite(values[dev]).mean(axis=0) >= .95)
    medians = np.nanmedian(values[dev][:, keep], axis=0)
    filled = np.where(np.isfinite(values[:, keep]), values[:, keep], medians)
    sd = filled[dev].std(axis=0)
    varying = np.isfinite(sd) & (sd >= 1e-8)
    keep, medians, filled, sd = keep[varying], medians[varying], filled[:, varying], sd[varying]
    center = filled[dev].mean(axis=0)
    x = np.clip((filled-center)/sd, -5, 5)
    return x, dict(keep=keep, impute=medians, center=center, sd=sd)


def pc1(x, dev):
    center = x[dev].mean(axis=0)
    _, _, vt = np.linalg.svd(x[dev]-center, full_matrices=False)
    w = vt[0] * (1 if vt[0, np.argmax(abs(vt[0]))] >= 0 else -1)
    scores = (x-center) @ w
    sd = scores[dev].std()
    return scores/sd, dict(component=w, pca_center=center, score_scale=np.array(sd))


def run():
    report = {"status": "passed", "checks": {}, "outputs": {}}
    source = json.loads((DATA / "proof-inputs.json").read_text())
    summary_manifest = json.loads((DATA / "proof-summaries.json").read_text())
    assert digest(DATA / "proof-summaries.npz") == summary_manifest["artifact_sha256"]
    assert digest(DATA / "proof-inputs.json") == summary_manifest["input_manifest_sha256"]
    records, order = source["records"], source["spi_order"]
    with np.load(DATA / "proof-summaries.npz") as a:
        marginal = a["marginal"]
        keys = a["row_id"].astype(str)
        raw = a["pearson"]
    meta = pd.DataFrame(records).set_index("row_id")
    assert len(keys) == 3780 and len(set(keys)) == 3780
    # Fixed stratification spans every source bank, size and several classes.
    check_rows = np.unique(np.r_[0, 899, 900, 1259, 1260, np.linspace(0, 3779, 36, dtype=int)])
    for i in check_rows:
        record = records[i]
        assert digest(ROOT / record["mpi_path"]) == record["mpi_sha256"]
        with np.load(ROOT / record["mpi_path"]) as a:
            m = record["M"]
            upper = np.triu_indices(m, 1)
            for j, spi in enumerate(order):
                matrix = a[spi]
                edges = (matrix[upper] + matrix[(upper[1], upper[0])])/2
                expected = ([edges.mean(), edges.std(), *np.quantile(edges,[.1,.25,.5,.75,.9])]
                            if np.isfinite(edges).all() else [np.nan]*7)
                np.testing.assert_allclose(marginal[i,j],expected,rtol=2e-14,atol=2e-14,equal_nan=True)
        x = np.load(ROOT / record["raw_path"])
        c = np.corrcoef(x.T)[np.triu_indices(x.shape[1],1)]
        np.testing.assert_allclose(raw[i],[c.mean(),abs(c).mean()],atol=1e-14)
    report["checks"]["independent_MPI_and_raw_replays"] = int(len(check_rows))
    # Every overlap agrees with a pre-existing independently extracted mean bank.
    with np.load(ROOT / "data/representation_stage_a_260907/proof-unified-controls.npz") as a:
        lookup = pd.Series(np.arange(len(keys)),index=keys)
        target = lookup.loc[a["row_id"].astype(str)].to_numpy()
        np.testing.assert_allclose(marginal[target,:,0],a["X_m"].reshape(-1,289,7)[:,:,0],
                                   rtol=1e-11,atol=1e-11,equal_nan=True)
    report["checks"]["existing_ordered_mean_bank_replays"] = len(target)
    with np.load(ROOT / "data/zenodo_7118947/features/pearson-unified-v3-seed1729.npz", allow_pickle=True) as a:
        zenodo_sources=json.loads(str(a["source_manifest_json"].item()))["entries"]
    base=ROOT / "data/zenodo_7118947/runs/p90-zscore/zenodo-7118947-p90-zscore-seed1729"
    recorded={r["mpi_sha256"] for r in json.loads((OUT / "zenodo-provenance.json").read_text())["sources"]}
    for row in zenodo_sources:
        folder=base / Path(row["dataset_path"]).name
        assert digest(folder / "spi_mpis.npz")==row["mpi_sha256"]
        assert digest(folder / "meta.json")==row["meta_sha256"]
        assert row["mpi_sha256"] in recorded
    report["checks"]["Zenodo_prior_MPI_and_metadata_hashes"]=len(zenodo_sources)
    with np.load(ROOT / "results/cross_mt_transfer_260824/confirmation-coordinates.npz",allow_pickle=True) as a:
        dev_keys = a["development_row_keys"].astype(str)
        eval_keys = np.array([f"{l}|M{int(m)}|T{int(t)}|I{int(i)}" for l,m,t,i in
            zip(a["confirmation_y"],a["confirmation_M"],a["confirmation_T"],a["confirmation_instance"])])
        original_z_dev, original_z_eval = a["development_pca_sym"],a["confirmation_pca_sym"]
    with np.load(OUT / "proof-projections.npz") as p:
        center, scale = original_z_dev.mean(axis=0),np.sqrt(original_z_dev.var(axis=0).sum())
        np.testing.assert_allclose(p["z_dev"],(original_z_dev-center)/scale)
        np.testing.assert_allclose(p["z_eval"],(original_z_eval-center)/scale)
        detail = pd.read_csv(OUT / "proof-individual.csv")
        checks=0
        for method in detail.method.unique():
            for scope in detail.scope.unique():
                rows=detail.query("method == @method and scope == @scope").set_index("row_id")
                accepted=set(meta.loc[rows.index,"label"])
                allowed_dev = np.array([k.split("|")[0] in accepted for k in dev_keys])
                gallery=meta.loc[dev_keys[allowed_dev]]
                gallery_scores=p[f"{method}_dev"][allowed_dev]
                for i in np.linspace(0,len(rows)-1,12,dtype=int):
                    key=rows.index[i]
                    position=int(np.flatnonzero(eval_keys==key)[0])
                    record=meta.loc[key]
                    admissible=(gallery["M"]!=record["M"]) & (gallery["T"]!=record["T"])
                    d=np.linalg.norm(gallery_scores[admissible]-p[f"{method}_eval"][position],axis=1)
                    rank=np.argsort(d,kind="stable")
                    matches=(gallery.loc[admissible,"label"].to_numpy()[rank] == record.label)
                    positive=np.flatnonzero(matches)
                    ap=np.mean(np.arange(1,len(positive)+1)/(positive+1))
                    np.testing.assert_allclose(ap,rows.loc[key,"ap"],atol=1e-12)
                    checks+=1
    report["checks"]["independent_hard_retrieval_replays"] = checks
    report["checks"]["original_proof_z_projection"] = "exact normalized replay"
    for system in ("Kuramoto","CML2D"):
        frame=pd.read_csv(OUT/f"inference-{system}.csv")
        dev=frame.development.to_numpy()
        with np.load(DATA/f"inference-{system}-summaries.npz") as a:
            marg=a["marginal"]
        errors={}
        matrices={}
        for name,values in (("mean",marg[:,:,0]),("distribution",marg.reshape(len(marg),-1))):
            x,transform=independent_standardization(values,dev)
            block_center=x[dev].mean(axis=0)
            block_scale=np.sqrt(x[dev].var(axis=0).sum())
            x=(x-block_center)/block_scale
            predicted,model=pc1(x,dev)
            errors[name]=float(np.max(abs(predicted-frame[f"{name}_PC1"])))
            np.testing.assert_allclose(predicted,frame[f"{name}_PC1"],atol=2e-10)
            matrices[name]=x
            np.savez_compressed(OUT/f"audit-{system}-{name}-model.npz",**transform,**model,
                                 block_center=block_center,block_scale=np.array(block_scale))
        report["checks"][f"{system}_independent_PC1_max_errors"]=errors
        metrics=pd.read_csv(OUT/"inference-metrics.csv").query("system == @system")
        held=frame.query("evaluation and common_eligible")
        for row in metrics.itertuples():
            r=abs(spearmanr(held[row.method],held.Q_reference).statistic)
            np.testing.assert_allclose(r,row.abs_rho,atol=1e-14)
            assert row.n==len(held)
    originals=["notebooks/embeddings/proof_p90_260824.ipynb","notebooks/embeddings/zenodo_1053_geometry.ipynb",
        "notebooks/inference/order-parameter-benchmark-comparison.ipynb",
        "notebooks/inference/order-parameter-benchmarks-lean.ipynb","notebooks/inference/order-parameter-benchmarks-baselines.ipynb"]
    for name in originals:
        before=subprocess.check_output(["git","show",f"9ba2094:{name}"],cwd=ROOT)
        assert before==(ROOT/name).read_bytes(),name
    report["checks"]["original_notebooks_unchanged"] = originals
    config=(ROOT/"configs/analysis/spi-baseline-exploration-260921.yaml").read_text()
    primary="\n".join(l for l in config.splitlines() if not l.startswith("  preprocessing_sensitivity:") and not l.startswith("  sensitivity_rationale:"))+"\n"
    (OUT/"protocol-primary.yaml").write_text(primary)
    (OUT/"protocol-with-preprocessing-sensitivity.yaml").write_text(config)
    assert digest(OUT/"protocol-primary.yaml")==json.loads((OUT/"proof-provenance.json").read_text())["config_sha256"]
    report["checks"]["primary_protocol_preserved"] = True
    for path in sorted(OUT.glob("*.csv")):
        report["outputs"][path.name]=digest(path)
    report["source_sha256"]={p:digest(ROOT/p) for p in ["scripts/spi_baseline_exploration.py",
        "scripts/prepare_spi_baseline_data.py","scripts/audit_spi_baseline_exploration.py",
        "scripts/plot_spi_baseline_exploration.py","scripts/build_spi_baseline_notebooks.py",
        "configs/analysis/spi-baseline-exploration-260921.yaml"]}
    (OUT/"verification.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report["checks"],indent=2))


if __name__=="__main__":
    with threadpool_limits(limits=4):
        run()
