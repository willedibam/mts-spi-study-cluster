"""Descriptive size/time and sensor audit for the collective period-doubling CML."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def analyze(inputs, output):
    rows, observations = [], []
    for folder in inputs:
        for path in sorted(folder.glob('case-*.npz')):
            with np.load(path,allow_pickle=False) as a:
                meta=json.loads(str(a['metadata_json'])); means=a['global_mean']; x=a['observed']
                assert means.shape==(meta['record_steps'],) and x.shape==(meta['observation_steps'],2,64)
                assert np.isfinite(means).all() and np.isfinite(x).all()
                future=np.abs(means[meta['observation_steps']+1::2]-means[meta['observation_steps']::2])
                np.testing.assert_allclose(future.mean(),meta['Q'],atol=1e-13,rtol=0)
                blocks=np.array(meta['Q_blocks'])
                row={k:meta[k] for k in ['L','N','r','g','seed','burn','record_steps','Q','binder','elapsed_seconds']}
                row.update(path=str(path),start='preordered' if meta.get('pre_steps',0) else 'random',
                    block_range=np.ptp(blocks),block_mean_se=blocks.std(ddof=1)/np.sqrt(len(blocks)),
                    half_difference=abs(meta['Q_first_half']-meta['Q_second_half']),
                    Q_input=np.abs(means[1:meta['observation_steps']:2]-means[:meta['observation_steps']:2]).mean())
                rows.append(row)
                for view,label in enumerate(meta['views']):
                    for M in (8,16,32,64):
                        for T in (100,500,1000):
                            window=x[:T,view,:M]; sd=window.std(axis=0)
                            edges=np.corrcoef(window.T)[np.triu_indices(M,1)]
                            sample_mean=window.mean(axis=1)
                            observations.append(dict(path=str(path),L=meta['L'],r=meta['r'],seed=meta['seed'],view=label,M=M,T=T,
                                constant_fraction=float(np.mean(sd<=1e-8)),mean_abs_correlation=float(np.abs(edges).mean()),
                                edge_correlation_sd=float(edges.std()),
                                sample_Q=float(np.abs(sample_mean[1::2]-sample_mean[::2]).mean()),
                                global_window_Q=float(np.abs(means[1:T:2]-means[:T:2]).mean())))
    frame=pd.DataFrame(rows).sort_values(['L','r','seed','burn','start'])
    obs=pd.DataFrame(observations)
    output.mkdir(parents=True,exist_ok=True)
    frame.to_csv(output/'physics.csv',index=False); obs.to_csv(output/'observations.csv',index=False)
    summary=dict(cases=len(frame),sizes=sorted(frame.N.unique().tolist()),
        max_half_difference=float(frame.half_difference.max()),max_block_se=float(frame.block_mean_se.max()),
        maximum_constant_fraction=float(obs.constant_fraction.max()),SPI_computed=False)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(12,3.5),constrained_layout=True)
    for L,group in frame.groupby('L'):
        c=group.groupby('r').agg(Q=('Q','mean'),lo=('Q','min'),hi=('Q','max'),se=('block_mean_se','max'),drift=('half_difference','max'))
        axes[0].plot(c.index,c.Q,'o-',ms=3,label=f'L={L}, N={L*L}')
        axes[0].fill_between(c.index,c.lo,c.hi,alpha=.12)
        axes[1].plot(c.index,c.se,'o-',ms=3)
        axes[2].plot(c.index,c.drift,'o-',ms=3)
    for ax in axes:
        ax.axvline(3.86212,color='0.5',ls=':',lw=1); ax.set_xlabel('Logistic parameter r')
        ax.spines[['top','right']].set_visible(False)
    axes[0].set(ylabel='Future collective Q',title='Mean and seed/start range (not CI)');axes[0].legend(fontsize=7)
    axes[1].set(ylabel='SD(block means) / sqrt(8)',title='Descriptive block precision; not mixing proof')
    axes[2].set(ylabel='Absolute difference',title='First/second future-half means')
    fig.savefig(output/'physics.png',dpi=170);plt.close(fig)
    print(frame[['L','r','seed','burn','start','Q','block_mean_se','half_difference']].to_string(index=False))
    print(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',type=Path,nargs='+',required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();analyze(a.inputs,a.output_dir)
