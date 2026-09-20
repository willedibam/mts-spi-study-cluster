"""Cached full-catalogue pair-sampling curves, with explicit validity coverage.

No SPI extraction or catalogue fitting. Every replicate samples common nested
dyads without replacement and retains both directions. Records, not pair draws,
are the unit for assessing generalisation beyond the fixed recordings.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from scripts.scout_spi_pair_sampling import pair_vectors, correlation_features
from src.spi_spi_contract import build_unified_feature_values

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'results/large_m_pair_baselines_260917'
OUT = BASE/'sampling-m64-m100'
FAMILIES = ['CML', 'MEG', 'VAR sparse', 'VAR dense']
COLORS = {64:'#2879a8', 100:'#d46a29', 243:'#3b9b78', 256:'#9b5dab'}
X_LABEL = (r'$b$ sampled channel dyads $(i,j) = 2b$ directed channel-pairs: '
           r'$(i\to j)$ and $(j\to i)$')
# Prespecified for semantic contrast, not selected by measured error.
PAIRS = [('spearmanr','kendalltau'), ('cov_EmpiricalCovariance','corr_pearson_tau-1'),
         ('cov_EmpiricalCovariance','plv_multitaper_mean_fs-1_fmin-0_fmax-0-5')]
PAIR_LABELS = ['Spearman × Kendall', 'Covariance × lag-1 Pearson', 'Covariance × mean PLV (full band)']


def family(name):
    if name.startswith('cml-'): return 'CML'
    if name.startswith('meg-'): return 'MEG'
    if name.startswith('var-sparse-'): return 'VAR sparse'
    if name.startswith('var-dense-'): return 'VAR dense'
    raise ValueError(f'Unexpected record, including cancelled families: {name}')


def metrics(estimates, gold):
    """Conditional numerical error and strict fixed-reference validity are distinct."""
    estimates = np.atleast_2d(estimates)
    full_valid = np.isfinite(gold)
    shared = np.isfinite(estimates) & full_valid[None, :]
    error = np.where(shared, estimates-gold, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        rmse = np.sqrt(np.nanmean(error**2, axis=1))
        p95 = np.nanquantile(abs(error), .95, axis=1)
        coordinate_rmse = np.sqrt(np.nanmean(error**2, axis=0))
        bias = np.nanmean(error, axis=0)
    retained = shared.sum(axis=1)/full_valid.sum()
    if (~full_valid).any():
        apparent = np.isfinite(estimates[:, ~full_valid]).mean(axis=1)
    else:
        apparent = np.full(len(estimates), np.nan)
    correlations = np.array([np.corrcoef(z[v], gold[v])[0,1] if v.sum()>1 else np.nan
                             for z,v in zip(estimates,shared)])
    return dict(rmse=rmse, p95_absolute_error=p95, retained=retained,
        strict_rmse=np.where(retained==1,rmse,np.nan), apparent_validity=apparent,
        feature_vector_correlation=correlations, coordinate_rmse=coordinate_rmse,
        coordinate_bias=bias, coordinate_valid_fraction=shared.mean(axis=0))


def process_case(arguments):
    path, repeats, out = arguments
    path, out = Path(path), Path(out)
    meta = json.loads(path.with_name('meta.json').read_text())
    name, m, t = meta['dataset_name'], meta['M'], meta['T']
    label = family(name)
    order = [s['name'] for s in meta['pyspi']['spis']]
    assert len(order)==289 and t==1000 and m in [64,100,243,256]
    assert meta['execution_identity']['pyspi_config_sha256']=='bc4bafa16b4add8bb6283db490a9fa3d8dde4fde47093147d32579b989397965'
    with np.load(path, allow_pickle=False) as bank:
        pairs = pair_vectors(np.stack([bank[n] for n in order]).astype(float))
        reference, reference_valid, _ = build_unified_feature_values(bank,order)
    d = pairs.shape[1]
    gold = correlation_features(pairs,np.arange(d))
    np.testing.assert_allclose(gold,reference,rtol=0,atol=2e-7,equal_nan=True)
    np.testing.assert_array_equal(np.isfinite(gold),reference_valid)
    budgets = sampling_budgets(d)
    seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4],'little')
    permutations = [np.random.default_rng(np.random.SeedSequence([260917801,seed,r])).permutation(d)
                    for r in range(repeats)]
    ia,ib = np.triu_indices(len(order),1)
    lookup = {frozenset((order[a],order[b])):i for i,(a,b) in enumerate(zip(ia,ib))}
    example_indices = np.array([lookup[frozenset(pair)] for pair in PAIRS])
    records, examples, stats = [], [], []
    for b in budgets:
        if b==d:
            # Census is deterministic; no need to recompute it for every draw.
            estimates = np.broadcast_to(gold,(repeats,len(gold)))
        else:
            estimates = np.stack([correlation_features(pairs,p[:b]) for p in permutations])
        result = metrics(estimates,gold)
        stats.append({k:result[k] for k in ['coordinate_rmse','coordinate_bias','coordinate_valid_fraction']})
        examples.append(estimates[:,example_indices])
        for draw in range(repeats):
            records.append(dict(name=name, family=label, M=m, T=t, draw=draw, dyads=int(b),
                ordered_entries=int(2*b), fraction=float(b/d), full_valid=int(reference_valid.sum()),
                **{key:float(result[key][draw]) for key in ['rmse','p95_absolute_error','retained',
                    'strict_rmse','apparent_validity','feature_vector_correlation']}))
    stem = path.parent.name
    np.savez_compressed(out/f'{stem}-coordinates.npz', budgets=budgets, gold=gold,
        full_valid=reference_valid, spi_order=np.asarray(order), example_indices=example_indices,
        example_estimates=np.stack(examples),
        **{key:np.stack([s[key] for s in stats]) for key in stats[0]})
    pd.DataFrame(records).to_csv(out/f'{stem}-draws.csv',index=False)
    provenance = dict(name=name, family=label, M=m, T=t, repeats=repeats, dyads=d,
        full_valid=int(reference_valid.sum()), full_coordinates=len(gold), path=str(path.relative_to(ROOT)),
        mpi_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        meta_sha256=hashlib.sha256(path.with_name('meta.json').read_bytes()).hexdigest(),
        source=meta['source'], execution_identity=meta['execution_identity'], seed=seed,
        coordinate_file=f'{stem}-coordinates.npz', draw_file=f'{stem}-draws.csv')
    (out/f'{stem}-report.json').write_text(json.dumps(provenance,indent=2)+'\n')
    return provenance


def sampling_budgets(d):
    return np.array(sorted({min(d,b) for b in [8,16,32,50,100,200,400,800,1600,3200,6400,12800,25600,d]}))


def figure_style():
    if not (shutil.which('latex') and shutil.which('dvipng')):
        raise RuntimeError('Requested LaTeX rendering requires latex and dvipng.')
    plt.rcParams.update({'text.usetex': True, 'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'], 'mathtext.fontset': 'cm',
        'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
        'axes.spines.top': False, 'axes.spines.right': False,
        'xtick.direction': 'out', 'ytick.direction': 'out', 'legend.frameon': False,
        'lines.linewidth': 1.7, 'lines.markersize': 2.7,
        'figure.constrained_layout.use': True, 'savefig.bbox': 'tight'})


def draw_paths(ax, budgets, values):
    """All nested sample paths: faint context, not additional independent data."""
    ax.plot(budgets, values, color='.35', alpha=.05, lw=.45, zorder=1)


def style_axis(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.set_xscale('log')
    ax.grid(axis='y',alpha=.12)


def plot_meg_comparison(out, frame, matched=False):
    """Available records, optionally restricted to parents present at every M."""
    meg = frame[(frame.family == 'MEG') & frame.M.isin([64, 100, 243])].copy()
    meg['parent'] = meg.name.str.rsplit('-M', n=1).str[0]
    if matched:
        parents = set.intersection(*(set(p.parent) for _,p in meg.groupby('M')))
        meg = meg[meg.parent.isin(parents)].copy()
    counts = meg.groupby('M').name.nunique().to_dict()
    assert counts and set(counts) <= {64,100,243}, counts
    assert meg['T'].eq(1000).all() and meg.groupby('name').draw.nunique().eq(128).all()
    stem = 'meg-matched-full-vector-convergence' if matched else 'meg-full-vector-convergence'
    meg.to_csv(out/('meg-matched-draws.csv' if matched else 'meg-only-draws.csv'), index=False)
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.5), constrained_layout=True)
    for m, part in meg.groupby('M'):
        groups = part.groupby('dyads')
        for ax, key in zip(axes, ['rmse', 'p95_absolute_error', 'retained']):
            scale = 100 if key == 'retained' else 1
            med = groups[key].median()*scale
            lo, hi = groups[key].quantile(.1)*scale, groups[key].quantile(.9)*scale
            ax.plot(med.index, med, 'o-', color=COLORS[m], label=rf'$M={m}$ ($n={counts[m]}$)')
            ax.fill_between(med.index, lo, hi, color=COLORS[m], alpha=.14)
            ax.plot(med.index[-1], med.iloc[-1], 'D', color=COLORS[m], ms=4)
    for ax, title in zip(axes, [r'RMSE$(\hat{\mathbf{z}},\mathbf{z})$',
                               '95th-percentile absolute feature error',
                               r'Reference features retained (\%)']):
        ax.set_title(title)
        style_axis(ax)
    axes[0].set_ylabel('RMSE')
    axes[1].set_ylabel('Absolute error')
    axes[2].set_ylabel(r'Features retained (\%)')
    axes[0].axhline(.05, color='.55', ls=':', lw=.8)
    axes[2].set_ylim(100*meg.retained.min()-1, 100.3)
    handles, labels = axes[0].get_legend_handles_labels()
    handles += [Patch(facecolor='.5', alpha=.14), Line2D([], [], color='.3', marker='D', ls='none'),
                Line2D([], [], color='.55', ls=':', lw=.8)]
    labels += [r'10--90\% instance/draw spread', 'Full census', r'RMSE$=0.05$ (illustrative)']
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5,-.015), ncol=len(handles), fontsize=8)
    fig.supxlabel(X_LABEL)
    sizes=','.join(map(str,counts))
    count_label=str(next(iter(counts.values()))) if len(set(counts.values()))==1 else 'see legend'
    note = r' $\cdot$ matched instances' if matched else (r' $\cdot$ partial large-$M$ results' if 243 in counts and counts[243] < 4 else '')
    fig.suptitle(rf'data=MEG ($M={sizes}$, $T=1000$, instances={count_label}) $\cdot$ 128 draws per instance'+note, fontsize=10)
    for ext in ['png', 'svg']:
        fig.savefig(out/f'{stem}.{ext}', dpi=180)
    plt.close(fig)


def plot(out, reports):
    frame = pd.concat([pd.read_csv(out/r['draw_file']) for r in reports],ignore_index=True)
    frame.to_csv(out/'all-draws.csv',index=False)
    rows = []
    for (label,m,b),part in frame.groupby(['family','M','dyads']):
        rows.append(dict(family=label,M=m,dyads=b,ordered_entries=2*b,records=part.name.nunique(),
            median_rmse=part.rmse.median(),p95_rmse=part.rmse.quantile(.95),
            median_p95_coordinate_error=part.p95_absolute_error.median(),
            minimum_coverage=part.retained.min(),median_coverage=part.retained.median(),
            strict_complete_draw_fraction=part.strict_rmse.notna().mean(),
            strict_rmse_le_005_fraction=(part.strict_rmse<=.05).mean(),
            median_feature_vector_correlation=part.feature_vector_correlation.median()))
    summary = pd.DataFrame(rows)
    summary.to_csv(out/'summary.csv',index=False)
    figure_style()
    plot_meg_comparison(out, frame)
    if 243 in frame.M.unique():
        plot_meg_comparison(out, frame, matched=True)
    fig,axes = plt.subplots(3,4,figsize=(12.6,7.5),sharex='col',constrained_layout=True)
    for col,label in enumerate(FAMILIES):
        subset = frame[frame.family==label]
        counts = subset.groupby('M').name.nunique()
        count_text=', '.join(f'{m}:{n}' for m,n in counts.items())
        axes[0,col].set_title(label+'\n'+rf'$M:n$ = {count_text}')
        for m,part in subset.groupby('M'):
            groups = part.groupby('dyads')
            for row,key in enumerate(['rmse','p95_absolute_error','retained']):
                ax=axes[row,col]
                median=groups[key].median(); low=groups[key].quantile(.1); high=groups[key].quantile(.9)
                scale=100 if key=='retained' else 1
                ax.plot(median.index,scale*median,'o-',ms=2.7,color=COLORS[m],label=rf'$M={m}$')
                ax.fill_between(median.index,scale*low,scale*high,color=COLORS[m],alpha=.14)
                if row<2:
                    ax.plot(median.index[-1],scale*median.iloc[-1],'D',ms=5,color=COLORS[m])
        axes[0,col].axhline(.05,color='.55',ls=':',lw=.8)
        for row in range(3): style_axis(axes[row,col])
        axes[2,col].set_ylim(max(0,100*subset.retained.min()-2),100.3)
    for row,label in enumerate([r'RMSE$(\hat{\mathbf{z}},\mathbf{z})$'+'\n(evaluable features)',
                                '95th percentile absolute\nfeature error per draw',
                                r'Reference features retained (\%)']):
        axes[row,0].set_ylabel(label)
    handles=[Line2D([],[],color=COLORS[m],marker='o',ms=2.7) for m in sorted(frame.M.unique())]
    labels=[rf'$M={m}$' for m in sorted(frame.M.unique())]
    handles += [Patch(facecolor='.5',alpha=.14),Line2D([],[],color='.3',marker='D',ls='none'),
                Line2D([],[],color='.55',ls=':',lw=.8)]
    labels += [r'10--90\% instance/draw spread', 'Full census', r'RMSE$=0.05$ (illustrative)']
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,-.015),ncol=len(handles),fontsize=8)
    fig.supxlabel(X_LABEL)
    fig.suptitle(r'Full feature vector $\cdot$ $T=1000$ $\cdot$ 128 draws per instance'
                 r' $\cdot$ median curves $\cdot$ vertical scales differ',fontsize=11)
    for suffix in ['png','svg']: fig.savefig(out/f'full-vector-convergence.{suffix}',dpi=180)
    plt.close(fig)

    # Fixed first chronological MEG block: same raw parent and nested channel layouts.
    selected=[next(r for r in reports if r['M']==m and r['name']==f'meg-105923-run6-block01-M{m}') for m in [64,100]]
    fig,axes=plt.subplots(2,3,figsize=(12.6,6.8),constrained_layout=True)
    example_rows=[]
    for row,r in enumerate(selected):
        with np.load(out/r['coordinate_file']) as bank:
            budgets=bank['budgets']; indices=bank['example_indices']; gold=bank['gold']; values=bank['example_estimates']
        for col,(idx,label) in enumerate(zip(indices,PAIR_LABELS)):
            ax=axes[row,col]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                low,median,high=np.nanquantile(values[:,:,col],[.025,.5,.975],axis=1)
            draw_paths(ax,budgets,values[:,:,col])
            ax.fill_between(budgets,low,high,color=COLORS[r['M']],alpha=.12,label=r'2.5--97.5\% of draws')
            ax.plot(budgets,median,'o-',ms=3,color=COLORS[r['M']],label='Median of pair draws')
            ax.axhline(gold[idx],color='black',ls='--',lw=1,label='Exact all-pairs value')
            ax.set_title(label.replace(' × ',r' $\times$ ')+'\n'+rf'$M={r["M"]}$; exact $z={gold[idx]:.4f}$')
            style_axis(ax)
            example_rows.append(dict(name=r['name'],M=r['M'],pair=label,exact=float(gold[idx]),
                rmse_at_50_dyads=float(np.sqrt(np.nanmean((values[np.flatnonzero(budgets==50)[0],:,col]-gold[idx])**2)))))
    handles,labels=axes[0,0].get_legend_handles_labels()
    handles.append(Line2D([],[],color='.6',lw=.6)); labels.append('128 nested paths (faint)')
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,-.015),ncol=4,fontsize=8)
    fig.supxlabel(X_LABEL)
    fig.supylabel(r'Estimated SPI-SPI feature value $\hat{z}$')
    fig.suptitle(r'select examples $\cdot$ data=MEG ($M=64,100$, $T=1000$, 1 instance)'
                 r' $\cdot$ 128 draws $\cdot$ vertical scales differ',fontsize=11)
    for suffix in ['png','svg']: fig.savefig(out/f'individual-coordinate-convergence.{suffix}',dpi=180)
    plt.close(fig)
    (out/'examples.json').write_text(json.dumps(example_rows,indent=2)+'\n')

    # Distribution over every full-valid coordinate, with missing draws kept visible.
    fig,axes=plt.subplots(1,4,figsize=(14,3.6),constrained_layout=True)
    for col,label in enumerate(FAMILIES):
        r=next(r for r in reports if r['M']==100 and r['family']==label)
        with np.load(out/r['coordinate_file']) as bank:
            budgets=bank['budgets']; errors=bank['coordinate_rmse']; coverage=bank['coordinate_valid_fraction']; valid=bank['full_valid']
        q=np.array([np.quantile(e[valid&(c==1)],[.1,.5,.9,.95]) for e,c in zip(errors,coverage)])
        ax=axes[col]
        ax.fill_between(budgets,q[:,0],q[:,2],alpha=.18,color=COLORS[100],label=r'10--90\% of features')
        ax.plot(budgets,q[:,1],'o-',ms=3,color=COLORS[100],label='Median coordinate RMSE')
        ax.plot(budgets,q[:,3],':',color='black',label='95th-percentile coordinate')
        ax.set_title(label+r' $\cdot$ $M=100$')
        style_axis(ax)
    axes[0].set_ylabel('Per-coordinate RMSE over pair draws')
    axes[0].legend(fontsize=7,frameon=False)
    fig.supxlabel(X_LABEL)
    fig.suptitle(r'Feature error distribution $\cdot$ first instance per family $\cdot$ $T=1000$'
                '\nFeatures defined in every draw only',fontsize=10)
    for suffix in ['png','svg']: fig.savefig(out/f'coordinate-error-distribution.{suffix}',dpi=180)
    plt.close(fig)
    return summary


def replot(out):
    """Restyle saved draws without recomputing any numerical estimates."""
    report=json.loads((out/'report.json').read_text())
    plot(out,report['cases'])
    report['figure_style'] = dict(latex=True, individual_paths=128, path_alpha=.05,
        focused_figure='meg-full-vector-convergence.png', meg_blocks_per_M=4,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        note='Presentation-only redraw from saved numerical results; extraction and sampling unchanged.')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


def main(cache, out, repeats, workers, include_large=False, reuse_from=None):
    out.mkdir(parents=True,exist_ok=True)
    paths=sorted(cache.glob('large-m-var-panel-m*/**/spi_mpis.npz'))
    assert len(paths)==32, f'Expected all32 completed M64/M100 references, got {len(paths)}'
    if include_large:
        paths += sorted(cache.glob('large-m-var-panel-large-*/**/spi_mpis.npz'))
    reports=[]; pending=[]
    for path in paths:
        prior_report=out/f'{path.parent.name}-report.json'
        candidate=prior_report if prior_report.exists() else ((reuse_from/prior_report.name) if reuse_from else prior_report)
        if candidate.exists():
            r=json.loads(candidate.read_text())
            assert r['repeats']==repeats
            assert r['mpi_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
            assert r['meta_sha256']==hashlib.sha256(path.with_name('meta.json').read_bytes()).hexdigest()
            with np.load(candidate.parent/r['coordinate_file']) as previous:
                np.testing.assert_array_equal(previous['budgets'],sampling_budgets(r['dyads']))
            if candidate != prior_report:
                for filename in [r['coordinate_file'],r['draw_file'],candidate.name]:
                    shutil.copy2(candidate.parent/filename,out/filename)
            reports.append(r)
        else:
            pending.append(path)
    print(f'Reusing {len(reports)} cached diagnostics; computing {len(pending)} new records.',flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for r in pool.map(process_case,[(str(p),repeats,str(out)) for p in pending]):
            reports.append(r)
            print('DONE',r['name'],flush=True)
    summary=plot(out,reports)
    report=dict(records=len(reports),repeats=repeats,cases=reports,
        partial_panel=len(reports)<48, planned_records=48,
        completed_counts=pd.DataFrame(reports).groupby(['family','M']).size().rename('instances').reset_index().to_dict('records'),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        method='Uniform nested common dyads without replacement; both orientations; current ordered-Pearson contract',
        selected_pairs=PAIRS,reference='Each recording own exact finite-record all-pairs z; no population-truth claim',
        caveats=['Numerical errors conditional on full-valid coordinates finite on that sample; coverage and strict complete-draw metrics saved.',
                 'Pooled bands mix record and sampling variation; not independent-record confidence intervals.',
                 'CML and MEG layouts nested across M; VAR physical system changes with M. No generic dimension-invariance claim.',
                 'Two CML controls and two VAR topologies are paired by seed; MEG blocks from one run are dependent.',
                 'No pyspi recomputation, sparse-extraction speedup, q refitting or catalogue reduction.',
                 'Only completed, audited M243/256 records included; incomplete family/size counts are explicit. Old TASEP not pooled.',
                 'Large-M records exceeded the original memory planning guard; user authorized numerical analysis of their valid outputs, not a claim that this guard passed.'])
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(summary[summary.dyads.isin([50,200,800])].to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,default=BASE/'cached-mpis')
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--repeats',type=int,default=128)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--replot-only',action='store_true')
    parser.add_argument('--include-large',action='store_true')
    parser.add_argument('--reuse-from',type=Path)
    args=parser.parse_args()
    if args.replot_only:
        replot(args.output)
    else:
        main(args.cache,args.output,args.repeats,args.workers,args.include_large,args.reuse_from)
