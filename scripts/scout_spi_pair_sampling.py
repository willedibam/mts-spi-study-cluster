"""Cached-MPI Monte Carlo diagnostic; no SPI recomputation or model fitting.

Sample unordered dyads without replacement, retain both ordered entries, and
use exactly the same sampled positions for every SPI. The reference is the
existing finite-record full MPI descriptor, not population/physical truth.
"""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.spi_spi_contract import _pearson_corr_matrix, build_unified_feature_values

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/spi_pair_sampling_260916'


def pair_vectors(matrices):
    """K x D x 2: unordered dyads, with forward and reverse values aligned."""
    k,m,_=matrices.shape
    i,j=np.triu_indices(m,1)
    return np.stack((matrices[:,i,j],matrices[:,j,i]),axis=-1)


def correlation_features(pairs, chosen):
    vectors=pairs[:,chosen,:].reshape(len(pairs),-1)
    correlation=_pearson_corr_matrix(vectors)
    return correlation[np.triu_indices(len(pairs),1)].astype(np.float32)


def sources():
    # Fixed available cases; no selection using convergence or target outcomes.
    selected=[]
    for label,pattern in [
        ('covariance-modulation','results/covariance_modulation_260909/replay-mpis/*/spi_mpis.npz'),
        ('Kuramoto','data/order_parameter/kuramoto_figure_examples/*/spi_mpis.npz'),
        ('2D CML','data/order_parameter/cml2d_period_doubling_260911/primary/mpi/primary/*s26091115*/spi_mpis.npz'),
        ('TASEP','results/spi_pair_sampling_260916/cached-m64/*/spi_mpis.npz')]:
        selected.extend((label,p) for p in sorted(ROOT.glob(pattern)))
    assert len(selected)==26, len(selected)
    return selected


def score_q(z,model):
    v=z[model['keep']]
    missing=np.mean(~np.isfinite(v))
    value=(np.where(np.isfinite(v),v,model['impute'])-model['center'])@model['component']/model['score_scale']
    return float(value),float(missing)


def run(repeats):
    OUT.mkdir(parents=True,exist_ok=True)
    records=[];provenance=[]
    with np.load(ROOT/'data/order_parameter/cml2d_period_doubling_260911/primary-analysis/model.npz') as a:
        model={key:a[key] for key in a.files}
    for case,(system,path) in enumerate(sources()):
        meta=json.loads(path.with_name('meta.json').read_text())
        order=[s['name'] for s in meta['pyspi']['spis']]
        assert len(order)==289
        with np.load(path,allow_pickle=False) as a:
            matrices=np.stack([a[name] for name in order]).astype(float)
            gold,valid,_=build_unified_feature_values(a,order)
        pairs=pair_vectors(matrices);D=pairs.shape[1]
        np.testing.assert_allclose(correlation_features(pairs,np.arange(D)),gold,atol=2e-7,equal_nan=True)
        qmodel=model if system=='2D CML' else None
        if qmodel is not None:
            assert order==model['spi_order'].tolist()
            qgold,_=score_q(gold,model)
        sizes=sorted({min(D,b) for b in [8,16,32,50,100,200,400,800,1600,D]})
        provenance.append(dict(case=case,system=system,path=str(path.relative_to(ROOT)),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),M=meta['M'],T=meta['T'],
            full_valid_features=int(valid.sum()),total_features=len(gold),dyads=D,
            full_q=qgold if qmodel is not None else None,
            pyspi_config_sha256=meta['pyspi'].get('config_sha256'),
            source_identity=meta.get('execution_identity',meta.get('experiment'))))
        for draw in range(repeats):
            permutation=np.random.default_rng(260916000+case*10000+draw).permutation(D)
            for b in sizes:
                z=correlation_features(pairs,permutation[:b])
                shared=valid&np.isfinite(z);error=z[shared]-gold[shared]
                row=dict(case=case,system=system,M=meta['M'],T=meta['T'],draw=draw,
                    dyads=b,ordered_entries=2*b,fraction=b/D,
                    rmse=float(np.sqrt(np.mean(error**2))),
                    median_absolute_error=float(np.median(abs(error))),
                    p95_absolute_error=float(np.quantile(abs(error),.95)),
                    maximum_absolute_error=float(max(abs(error))),
                    signed_mean_error=float(np.mean(error)),
                    lost_full_valid_fraction=float(np.mean(~np.isfinite(z[valid]))),
                    apparent_valid_among_full_invalid=float(np.mean(np.isfinite(z[~valid]))),
                    feature_vector_correlation=float(np.corrcoef(z[shared],gold[shared])[0,1]))
                if qmodel is not None:
                    q,missing=score_q(z,model)
                    row.update(q_error=q-qgold,q_selected_missingness=missing)
                records.append(row)
        print(system,meta['M'],path.parent.name,flush=True)
    frame=pd.DataFrame(records)
    frame.to_csv(OUT/'draws.csv',index=False)
    summary=frame.groupby(['system','M','dyads','ordered_entries']).agg(
        median_RMSE=('rmse','median'),worst_draw_RMSE=('rmse','max'),
        median_p95_error=('p95_absolute_error','median'),median_z_correlation=('feature_vector_correlation','median'),
        maximum_valid_loss=('lost_full_valid_fraction','max'),
        median_apparent_validity=('apparent_valid_among_full_invalid','median')).reset_index()
    summary.to_csv(OUT/'summary.csv',index=False)
    report=dict(status='exploratory cached-MPI approximation; not sparse-extraction equivalence or timing',
        design='simple random dyads without replacement; both directions; nested prefixes per draw; common positions across SPIs',
        repeats=repeats,cases=provenance,
        reference='all ordered off-diagonal finite-record MPI entries; exact current validity rule',
        caveats=['Different systems across M: not a controlled system-size scaling experiment.',
            'Error summaries use full-valid features and separately report loss/apparent validity.',
            'M32 CML is already a dispersed N65536 view; cannot validate a full-N descriptor.',
            'Paired orientations are not independent samples.'],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(12,3.6),constrained_layout=True)
    for (system,M),part in frame.groupby(['system','M']):
        curves=part.groupby('ordered_entries')
        for ax,key in zip(axes[:2],['rmse','p95_absolute_error']):
            median=curves[key].median();lo=curves[key].quantile(.1);hi=curves[key].quantile(.9)
            ax.plot(median.index,median,'o-',ms=3,label=f'{system}, M={M}')
            ax.fill_between(median.index,lo,hi,alpha=.12)
    cml=frame.query("system == '2D CML'").copy()
    cml['absolute_q_error']=abs(cml.q_error)
    curves=cml.groupby('ordered_entries').absolute_q_error
    axes[2].plot(curves.median().index,curves.median(),'o-',color='#35b779',ms=3)
    axes[2].fill_between(curves.median().index,curves.quantile(.1),curves.quantile(.9),color='#35b779',alpha=.16)
    for ax in axes:
        ax.set_xscale('log');ax.set_xlabel('Sampled ordered entries (2 per dyad)')
        ax.spines[['top','right']].set_visible(False)
        ax.axvline(100,color='.6',ls=':',lw=.8)
    axes[0].set(ylabel='RMSE of z coordinates',title='Typical full-descriptor error')
    axes[1].set(ylabel='95th percentile absolute coordinate error',title='Less forgiving feature-wise error')
    axes[2].set(ylabel='Absolute q error (frozen development SD)',title='2D CML: unchanged frozen q')
    axes[0].legend(frameon=False,fontsize=7)
    fig.suptitle(f'Cached p90 pilot: 26 records, {repeats} pair draws each; bands = 10–90% of record/draw values')
    fig.savefig(OUT/'convergence.png',dpi=180)
    fig.savefig(OUT/'convergence.svg')
    plt.close(fig)
    print(summary[summary.dyads.isin([50,200,800])].to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repeats',type=int,default=32)
    run(p.parse_args().repeats)
