"""Stored-data composites: observed state portraits, MTS snapshots and unchanged readouts."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.benchmark_latex_style import latex_style, prepare_latex_figure

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'results/order-parameter-inference/locking-composites-261008'
ANCHORS = (0., .10, .20)
CASES = [
    ('clean', 'cross-frequency-locking-261006', None, None, None, 'Clean observations'),
    ('random-noise-m24', 'cross-frequency-locking-snr-261006', 'random-noise', 'eta', .6, 'Varying independent sensor noise'),
    ('random-noise-m48', 'cross-frequency-locking-confirm-261006', 'random-noise', 'eta', .6, 'Varying independent sensor noise · confirmation'),
    ('common-mode', 'cross-frequency-locking-confirm-261006', 'common-mode', 'common', .4, 'Varying shared observation signal'),
    ('internal-coupling', 'cross-frequency-locking-confirm-261006', 'internal-coupling', 'eps', .375, 'Varying internal coupling'),
]


def independent_limits(mean, low, high, margin=.08):
    """Display-only limits include the complete plotted band; never use Q to scale a readout."""
    values = np.concatenate([np.asarray(v, float).ravel() for v in (mean, low, high)])
    values = values[np.isfinite(values)]
    lo, hi = values.min(), values.max()
    span = hi-lo
    if span <= 1e-12:
        span = max(abs(lo)*.1, .01)
    return float(lo-margin*span), float(hi+margin*span)


def choose_records(scores, field=None, target=None):
    held = scores[scores.role.eq('evaluation') & scores.comparison_eligible]
    chosen = []
    for gamma in ANCHORS:
        part = held[np.isclose(held.control, gamma)].sort_values('row_id')
        if field is None:
            chosen.append(part.sort_values('seed').iloc[0])
        else:
            chosen.append(part.iloc[np.argmin(abs(part[field].to_numpy()-target))])
    return chosen


def tracking(ax, scores, key, title, reference=True):
    held = scores[scores.role.eq('evaluation') & scores.comparison_eligible]
    groups = held.groupby('control')
    q = groups.Q_lock
    left = q.mean()
    ax.plot(left.index, left, 'o-', color='#222222', ms=2.7, lw=1.7, label='$Q$')
    ax.fill_between(left.index,q.quantile(.1),q.quantile(.9),color='#222222',alpha=.09,lw=0)
    right = ax.twinx()
    right.spines['right'].set_visible(True)
    g = groups[key]
    mean, low, high = g.mean(), g.quantile(.1), g.quantile(.9)
    right.plot(mean.index,mean,'s-',color='#0072B2',ms=2.7,lw=1.7,label='readout')
    right.fill_between(mean.index,low,high,color='#0072B2',alpha=.16,lw=0)
    # Affine display range only: no fit, sign reversal, Q-dependent calibration or changed score.
    right.set_ylim(*independent_limits(mean,low,high))
    ax.set_ylim(-.08,1.08)
    if reference: ax.axvline(.1017,color='.6',ls=':',lw=.8)
    rho = abs(spearmanr(held[key],held.Q_lock).statistic)
    ax.set(xlabel=r'Cross-coupling $\gamma$',ylabel='Locking index $Q$',title=title+'\n'+rf'$|\rho_s|={rho:.2f}$')
    right.set_ylabel(r'$\langle|r|\rangle$' if key=='mean_abs_r' else '$q$ (original PC units)',color='#0072B2')
    right.tick_params(axis='y',labelcolor='#0072B2')
    ax.set_xticks([0,.05,.1,.15,.2])
    return dict(right_limits=list(right.get_ylim()),rho=float(rho),curve_min=float(mean.min()),curve_max=float(mean.max()))


def make_case(case):
    slug,run,arm,field,target,title = case
    results=ROOT/'results/order-parameter-inference'/run
    data=ROOT/'data/order-parameter-inference'/run
    scores=pd.read_csv(results/'scores.csv')
    if arm is not None:scores=scores[scores.system.eq(arm)].copy()
    chosen=choose_records(scores,field,target)
    with np.load(data/'observations.npz') as archive:
        observations=[np.array(archive[r.row_id],copy=True) for r in chosen]
    M,T=observations[0].shape;n=M//3
    assert all(x.shape==(M,T) and np.isfinite(x).all() for x in observations)
    snapshot_steps=200
    fig,axes=plt.subplots(3,3,figsize=(11.9,8.3),constrained_layout=True,
                          gridspec_kw={'height_ratios':[1,1,1.05]})
    bound=max(abs(x[[0,n]]).max() for x in observations)*1.04
    records=[]
    for col,(r,x) in enumerate(zip(chosen,observations)):
        a=axes[0,col]
        a.scatter(x[0],x[n],s=3,alpha=.22,c='#31688e',edgecolors='none',rasterized=True)
        symbol = {'eta':r'\eta','common':'s','eps':r'\varepsilon'}.get(field)
        nuisance='' if field is None else rf', ${symbol}={r[field]:.2f}$'
        a.set(title=rf'$\gamma={r.control:.2f}$'+nuisance,xlabel='Slow A channel (SD)',ylabel='Fast B channel (SD)',
              xlim=(-bound,bound),ylim=(-bound,bound),aspect='equal')
        image=axes[1,col].imshow(x[:,:snapshot_steps],origin='lower',aspect='auto',interpolation='nearest',
                                cmap='RdBu_r',vmin=-3,vmax=3,extent=(0,snapshot_steps*.5,-.5,M-.5))
        for boundary in [n-.5,2*n-.5]:axes[1,col].axhline(boundary,color='.2',lw=.5)
        axes[1,col].set(xlabel='Time (model units)',ylabel='Observed community',yticks=[(n-1)/2,(3*n-1)/2,(5*n-1)/2],
                        yticklabels=['A','B','C'])
        records.append(dict(row_id=r.row_id,seed=int(r.seed),control=float(r.control),Q=float(r.Q_lock),
                            nuisance=None if field is None else float(r[field]),
                            array_sha256=hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()))
    colorbar=fig.colorbar(image,ax=list(axes[1]),fraction=.025,pad=.015,extend='both')
    colorbar.set_label('Channel-standardized value')
    metrics={}
    for ax,(key,name) in zip(axes[2],[('z_PC1','SPI–SPI PC1 (centered)'),('mean_abs_r','Mean absolute Pearson'),('mean_PC1','Mean-SPI PC1')]):
        metrics[key]=tracking(ax,scores,key,name,reference=arm!='internal-coupling')
    fig.suptitle(f'{title} · M=N={M}, T={T}\n'
                 'Observed A–B portraits → first 200 samples → held tracking; bands: 10–90% of instances\n'
                 'Black: Q (left) · blue: readout (right); independently ranged axes, not calibration',fontsize=10)
    prepare_latex_figure(fig)
    fig.canvas.draw();fig.canvas.draw();fig.set_layout_engine('none')
    for ext in ['png','svg']:fig.savefig(OUT/f'{slug}-composite.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    return dict(run=run,arm=arm,M=M,T=T,selection='First held seed for clean; closest to prechosen nuisance midpoint otherwise; never select using Q or q.',
                nuisance_field=field,nuisance_target=target,anchors=list(ANCHORS),snapshot_steps=snapshot_steps,
                portrait_channels=[0,n],portraits_use='All T observed, standardized samples; not latent phase angles or smoothed signals.',
                scores_sha256=hashlib.sha256((results/'scores.csv').read_bytes()).hexdigest(),
                records=records,tracking=metrics)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'serif','font.serif':['CMU Serif','DejaVu Serif'],
                         'mathtext.fontset':'cm','font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                         'legend.frameon':False,'xtick.direction':'out','ytick.direction':'out'})
    latex_style()
    audit={case[0]:make_case(case) for case in CASES}
    audit['code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print('Wrote',len(CASES),'composites to',OUT)


if __name__=='__main__':main()
