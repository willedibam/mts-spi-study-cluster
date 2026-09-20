"""Extend the baseline notebook to the other lean benchmark systems, using cached MPIs."""
from pathlib import Path
import argparse
import json
import re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

from scripts.spi_baseline_exploration import ROOT, DATA, OUT, sha, summarize, fit_coordinate, write_json, finite_json
from scripts.order_parameter_simple_baselines import input_statistics
from src.corpus_geometry import fit_geometry_transform
from src.utils import slugify

ORDER_ROOT = ROOT / "data/order_parameter"
MIRROR = DATA / "downloads"
LEGACY = {
    "stuart_landau_development_full_p90": ("SL", "development", "stuart_landau_development_analysis", "gamma", "frequency_half_width"),
    "stuart_landau_confirmation_full_p90": ("SL", "evaluation", "stuart_landau_confirmation_analysis", "gamma", "frequency_half_width"),
    "stuart_landau_locking_boundary_confirmation_p90": ("SLfine", "evaluation", "stuart_landau_locking_boundary_confirmation_analysis", "gamma", "frequency_half_width"),
    "miller_huse_development_full_p90": ("MH", "development", "miller_huse_development_analysis", "g", "coupling"),
    "miller_huse_confirmation_full_p90": ("MH", "evaluation", "miller_huse_confirmation_analysis", "g", "coupling"),
    "quadratic_cml_development_full_p90": ("Kaneko", "mixed", "quadratic_cml_development_analysis", "alpha", "alpha"),
}


def prepare():
    inventory = json.loads((DATA / "legacy-feature-inventory.json").read_text())
    records, requests = [], set()
    def add(source, bank, extra):
        remote = source["dataset_path"].lstrip("/")
        row = dict(source, bank=bank, **extra)
        local_relative = remote.split("mts-spi-data/", 1)[1]
        for key, filename in (("mpi_path", "spi_mpis.npz"), ("meta_path", "meta.json"), ("raw_path", "timeseries.npy")):
            local = ROOT / "data" / local_relative / filename
            if not local.exists():
                local = MIRROR / remote / filename
                if not local.exists(): requests.add(f"{remote}/{filename}")
            row[key] = str(local.relative_to(ROOT))
        records.append(row)
    for bank, spec in LEGACY.items():
        for row in inventory[bank]["sources"]:
            path = Path(row["dataset_path"])
            match = re.search(r"M(\d+)_T(\d+)_I(\d+)", path.name)
            m, t, instance = map(int, match.groups())
            if bank.startswith("stuart_landau_development"):
                keep = path.parent.name == "stuart-landau-full-observation" and t >= 500 and instance < 4
            elif bank.startswith("stuart_landau_confirmation"):
                keep = path.parent.name == "stuart-landau-full-observation-confirmation" and m == 32 and t == 1000
            elif bank.startswith("stuart_landau_locking"):
                keep = True
            elif bank.startswith("miller_huse_development"):
                keep = instance < 4
            elif bank.startswith("miller_huse_confirmation"):
                keep = m == 32
            else:
                keep = path.parent.name.endswith("large-lattice") and (instance < 4 or m == 32)
            if keep: add(row, bank, dict(M=m, T=t, instance=instance))
    for arm in ("primary", "confirmation/primary"):
        relative = f"order_parameter/finite_regime_260915/rossler/{arm}"
        corpus = ROOT / "data" / relative
        source = {r["row_id"]:r for r in json.loads((corpus / "analysis/eligibility.json").read_text())["sources"]}
        manifest = json.loads((corpus / "manifest.json").read_text())
        for r in manifest["rows"]:
            if arm == "primary" and r["role"] != "development": continue
            folder = f"{r['corpus_index']:04d}-{slugify(r['row_id'], 'dataset')}"
            remote = f"/g/data/ql44/we2614/mts-spi-data/{relative}/mpi/finite-regime/{folder}"
            # Named-archive inputs already exist locally; only MPI/meta are needed.
            row = dict(dataset_path=remote, bank="Rossler", corpus=str(corpus.relative_to(ROOT)),
                       row_id=r["row_id"], mpi_sha256=source[r["row_id"]]["mpi_sha256"], **{k:r[k] for k in ("M","T","seed","role")})
            for key, filename in (("mpi_path","spi_mpis.npz"),("meta_path","meta.json")):
                local = MIRROR / remote.lstrip("/") / filename
                if not local.exists(): requests.add(f"{remote.lstrip('/')}/{filename}")
                row[key]=str(local.relative_to(ROOT))
            records.append(row)
    write_json(DATA / "extension-inputs.json", dict(records=records, catalogue=inventory[next(iter(LEGACY))]["spi_order"]))
    (DATA / "extension-transfer-files.txt").write_text("\n".join(sorted(requests))+"\n")
    print(pd.Series([r["bank"] for r in records]).value_counts().to_string())
    print(f"{len(requests)} cached files to retrieve")


def extract():
    manifest = json.loads((DATA / "extension-inputs.json").read_text())
    order = manifest["catalogue"]
    tables={name:pd.read_csv(ORDER_ROOT/spec[2]/"scores.csv") for name,spec in LEGACY.items()}
    tables["Rossler-dev"] = pd.read_csv(ORDER_ROOT/"finite_regime_260915/rossler/primary/analysis/scores.csv")
    tables["Rossler-eval"] = pd.read_csv(ORDER_ROOT/"finite_regime_260915/rossler/confirmation/primary/analysis/scores.csv")
    values, rows, provenance=[], [], []
    raw_archives={}
    for i, r in enumerate(manifest["records"]):
        mpi, meta_path = ROOT/r["mpi_path"], ROOT/r["meta_path"]
        assert sha(mpi)==r["mpi_sha256"]
        if "meta_sha256" in r: assert sha(meta_path)==r["meta_sha256"]
        meta=json.loads(meta_path.read_text())
        with np.load(mpi) as a:
            assert set(a.files)==set(order)
            values.append(summarize(a,order))
        if r["bank"] == "Rossler":
            table=tables["Rossler-dev" if r["role"]=="development" else "Rossler-eval"]
            matched=table[table.row_id==r["row_id"]]
            corpus=ROOT/r["corpus"]
            if str(corpus) not in raw_archives: raw_archives[str(corpus)]=np.load(corpus/"observations.npz")
            x=raw_archives[str(corpus)][r["row_id"]]
            system, role="Rossler",r["role"]
            assert meta["source"]["archive_sha256"]==json.loads((corpus/"manifest.json").read_text())["archive_sha256"]
        else:
            system, role, _, control, param=LEGACY[r["bank"]]
            table=tables[r["bank"]]
            params=meta["generator"]["resolved_params"]
            matched=table[(table.M==r["M"]) & (table["T"]==r["T"]) &
                          (table.instance==r["instance"]) & np.isclose(table[control], params[param],atol=1e-9,rtol=0)]
            if system.startswith("SL"): matched=matched[matched.arm=="full"]
            if system=="Kaneko":
                matched=matched[matched.arm=="large"]
                role="development" if r["instance"]<4 else "evaluation"
            x=np.load(ROOT/r["raw_path"]).T
        assert len(matched)==1,(r["bank"],r.get("row_id"),len(matched))
        row=matched.iloc[0].to_dict()
        assert x.shape==(int(row["M"]),int(row["T"]))
        stats=input_statistics(x)
        for key in ("mean_abs_correlation","analytic_phase_coherence","temporal_spectral_entropy"):
            if key in row: np.testing.assert_allclose(stats[key],row[key],rtol=2e-5,atol=2e-6)
        row.update(stats, system=system, role=role, row_id=r.get("row_id",r["dataset_path"]),bank=r["bank"])
        if system!="Rossler":
            row["seed"]=int(row["instance"])
            row["control"]=row[LEGACY[r["bank"]][3]]
        row["q"]=row["q1"] if system=="Kaneko" else row["q"]
        # Legacy score tables already encode their reported cohorts; do not
        # invent a new original-q row gate from their missingness diagnostic.
        row["original_eligible"]=bool(row.get("eligible",np.isfinite(row["q"])))
        rows.append(row)
        provenance.append(dict(row_id=row["row_id"],mpi_sha256=sha(mpi),meta_sha256=sha(meta_path)))
        if (i+1)%250==0: print(f"extension extraction {i+1}/{len(manifest['records'])}",flush=True)
    for a in raw_archives.values():a.close()
    DATA.mkdir(exist_ok=True)
    np.savez_compressed(DATA/"extension-summaries.npz",marginal=np.array(values),row_id=np.array([r["row_id"] for r in rows]),spi_order=np.array(order))
    pd.DataFrame(rows).to_csv(DATA/"extension-scores.csv",index=False)
    write_json(DATA/"extension-sources.json",dict(sources=provenance,inventory_sha256=sha(DATA/"extension-inputs.json")))


def analyze():
    frame=pd.read_csv(DATA/"extension-scores.csv")
    with np.load(DATA/"extension-summaries.npz") as a:
        np.testing.assert_array_equal(a["row_id"],frame.row_id)
        all_marg=a["marginal"]
    cohorts=[("StuartLandau","SL","Q_R_mean"),("StuartLandauFine","SLfine","Q_R_mean"),
             ("MillerHuse","MH","Q_spin_abs"),("KanekoBand","Kaneko","Q_selected_band_power"),
             ("KanekoEntropy","Kaneko","Q_temporal_entropy"),("Rossler","Rossler","Q_reference")]
    metrics, diagnostics={},{}
    for name,group,target in cohorts:
        mask=(frame.system==group) | ((frame.system=="SL") & (frame.role=="development") if group=="SLfine" else False)
        f=frame.loc[mask].reset_index(drop=True).copy();marg=all_marg[mask]
        f["Q_reference"]=f[target]
        dev=(f.role=="development").to_numpy();evaluation=~dev
        expected={"StuartLandau":240,"StuartLandauFine":240,"MillerHuse":144,"KanekoBand":492,"KanekoEntropy":492,"Rossler":84}
        assert dev.sum()==expected[name],(name,dev.sum())
        assert not set(f.loc[dev,'row_id']) & set(f.loc[evaluation,'row_id'])
        eligible=f.original_eligible.to_numpy().copy()
        diag={}
        for method,values in (("mean",marg[:,:,0]),("distribution",marg.reshape(len(f),-1))):
            transform=fit_geometry_transform(values[dev],scaling="standard",minimum_valid_fraction=.95)
            x=np.clip(transform.transform(values),-5,5)
            f[f"{method}_selected_missingness"]=np.mean(~np.isfinite(values[:,transform.keep_indices]),axis=1)
            eligible &= f[f"{method}_selected_missingness"].to_numpy()<=.05
            scores, info=fit_coordinate(x,dev,f.seed.to_numpy())
            f[f"{method}_PC1"]=scores
            diag[method]=info
            # Independently recover PC1 by direct NumPy SVD and save fitted state.
            center=x[dev].mean(axis=0)
            _,_,vt=np.linalg.svd(x[dev]-center,full_matrices=False)
            w=vt[0];w=w*(1 if w[np.argmax(abs(w))]>=0 else -1)
            replay=(x-center)@w;scale=replay[dev].std();replay/=scale
            np.testing.assert_allclose(scores,replay,atol=3e-10)
            np.savez_compressed(OUT/f"extension-{name}-{method}-model.npz",keep=transform.keep_indices,
                impute=transform.impute_values,location=transform.location,scale=transform.scale,
                component=w,center=center,score_scale=scale)
            info["independent_replay_max_error"]=float(abs(replay-scores).max())
        f["development"],f["evaluation"],f["common_eligible"]=dev,evaluation,eligible
        methods=["q","mean_PC1","distribution_PC1","mean_correlation","mean_abs_correlation","temporal_spectral_entropy","mean_channel_sd"]
        if group.startswith("SL"):methods.append("analytic_phase_coherence")
        h=f.loc[evaluation & eligible].copy()
        assert np.isfinite(h[methods+["Q_reference"]]).all().all()
        units=[np.flatnonzero(h.seed.to_numpy()==s) for s in sorted(h.seed.unique())]
        rng=np.random.default_rng(260921)
        draws=[np.concatenate([units[j] for j in rng.integers(len(units),size=len(units))]) for _ in range(2000)]
        qboot=np.array([abs(spearmanr(h.q.to_numpy()[ix],h.Q_reference.to_numpy()[ix]).statistic) for ix in draws])
        result=[]
        for method in methods:
            value=abs(spearmanr(h[method],h.Q_reference).statistic)
            boot=np.array([abs(spearmanr(h[method].to_numpy()[ix],h.Q_reference.to_numpy()[ix]).statistic) for ix in draws])
            means=h.groupby('control')[[method,'Q_reference']].mean()
            residual=h[[method,'Q_reference']]-h.groupby('control')[[method,'Q_reference']].transform('mean')
            low,high=np.nanquantile(boot,[.025,.975]);dlo,dhi=np.nanquantile(boot-qboot,[.025,.975])
            result.append(dict(system=name,method=method,n=len(h),seeds=len(units),evaluated_available=int(evaluation.sum()),
                abs_rho=value,low=low,high=high,difference_vs_q=value-abs(spearmanr(h.q,h.Q_reference).statistic),
                difference_low=dlo,difference_high=dhi,control_mean_abs_rho=abs(spearmanr(means[method],means.Q_reference).statistic),
                within_control_abs_rho=abs(spearmanr(residual[method],residual.Q_reference).statistic)))
        diag['display_signs_from_development_Q']={method:(-1 if spearmanr(f.loc[dev,method],f.loc[dev,'Q_reference']).statistic<0 else 1) for method in ['q','mean_PC1','distribution_PC1']}
        diag.update(development_rows=int(dev.sum()),evaluation_rows=int(evaluation.sum()),common_eligible_evaluation_rows=len(h),target=target,
                    original_q_all_evaluation_abs_rho=float(abs(spearmanr(f.loc[evaluation,'q'],f.loc[evaluation,'Q_reference']).statistic)))
        f.to_csv(OUT/f"inference-{name}.csv",index=False)
        metrics[name]=result;diagnostics[name]=diag
        print(name,'development',int(dev.sum()),'held',len(h),[(r['method'],round(r['abs_rho'],3)) for r in result[:3]],flush=True)
    pd.DataFrame([r for result in metrics.values() for r in result]).to_csv(OUT/'inference-extra-metrics.csv',index=False)
    write_json(OUT/'inference-extra-provenance.json',finite_json(dict(diagnostics=diagnostics,
        status='retrospective extension to existing lean benchmark systems; no new p90',
        protocol_sha256=sha(ROOT/'configs/analysis/spi-baseline-lean-extension-260921.yaml'),
        code_sha256=sha(Path(__file__)),summaries_sha256=sha(DATA/'extension-summaries.npz'),
        scores_sha256=sha(DATA/'extension-scores.csv'),sources_sha256=sha(DATA/'extension-sources.json'))))


def audit():
    base=pd.read_csv(DATA/'extension-scores.csv').set_index('row_id')
    metrics=pd.read_csv(OUT/'inference-extra-metrics.csv')
    report={'status':'passed','coverage':{},'identical_fitted_states':[]}
    for name in metrics.system.unique():
        frame=pd.read_csv(OUT/f'inference-{name}.csv')
        np.testing.assert_allclose(frame.q,base.loc[frame.row_id,'q'],rtol=0,atol=2e-14)
        assert not set(frame.loc[frame.development,'row_id']) & set(frame.loc[frame.evaluation,'row_id'])
        eligible=frame.original_eligible & frame.mean_selected_missingness.le(.05) & frame.distribution_selected_missingness.le(.05)
        np.testing.assert_array_equal(frame.common_eligible,eligible)
        held=frame[frame.evaluation & eligible]
        for row in metrics[metrics.system==name].itertuples():
            assert row.n==len(held)
            np.testing.assert_allclose(abs(spearmanr(held[row.method],held.Q_reference).statistic),row.abs_rho,atol=1e-14)
        report['coverage'][name]={'available':int(frame.evaluation.sum()),'retained':len(held)}
    for a,b in [('StuartLandau','StuartLandauFine'),('KanekoBand','KanekoEntropy')]:
        for method in ('mean','distribution'):
            with np.load(OUT/f'extension-{a}-{method}-model.npz') as first,np.load(OUT/f'extension-{b}-{method}-model.npz') as second:
                assert first.files==second.files
                for key in first.files:np.testing.assert_array_equal(first[key],second[key])
        report['identical_fitted_states'].append([a,b])
    originals=json.loads((DATA/'presentation-update-inputs.json').read_text())
    for file,digest in originals.items():assert sha(ROOT/file)==digest,file
    report['preserved_files']=originals
    report['validated_cached_MPI_count']=len(json.loads((DATA/'extension-sources.json').read_text())['sources'])
    import nbformat
    notebook=ROOT/'notebooks/embeddings/spi_baseline_exploration_260921.ipynb'
    n=nbformat.read(notebook,as_version=4)
    for c in n.cells:
        if c.cell_type=='code':assert c.execution_count is not None and all(o.output_type!='error' for o in c.outputs)
    report['notebook_cells']=len(n.cells)
    report['executed_code_cells']=sum(c.cell_type=='code' for c in n.cells)
    report['embedded_figures']=sum('image/png' in o.get('data',{}) for c in n.cells for o in c.get('outputs',[]))
    report['artifact_sha256']={str(p.relative_to(ROOT)):sha(p) for p in [notebook,OUT/'inference-extra-metrics.csv',OUT/'inference-extra-provenance.json']}
    report['code_sha256']=sha(Path(__file__))
    write_json(OUT/'extension-verification.json',report)
    print(json.dumps(report['coverage'],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','extract','analyze','audit']);args=p.parse_args()
    with threadpool_limits(limits=4):{'prepare':prepare,'extract':extract,'analyze':analyze,'audit':audit}[args.stage]()
