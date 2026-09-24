"""Analyse and plot the focused 2026-09-24 case reruns, without recomputing SPIs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

R = 'cov_EmpiricalCovariance'
RHO = 'spearmanr'
MI = 'mi_kraskov_NN-4'
DCE = 'mi_kraskov_NN-4_DCE-AUTO'
ED = 'pdist_euclidean_rmse'
XD = 'xpdist_euclidean_tau-10_min_rmse'
DTW = 'dtw_rmse'
COLORS = ['#0072B2', '#D55E00', '#009E73']
CASE_LABELS = {'iter1_L': 'Linear', 'iter2_LM': 'Linear + monotone', 'iter3_LMNM': '+ quadratic'}
BETA_LABELS = {'beta-pi': r'$\beta=\pi$', 'beta-5': r'$\beta=5$', 'beta-2pi': r'$\beta=2\pi$'}


def style():
    plt.rcParams.update({'text.usetex': False, 'font.family': 'serif', 'font.serif': ['CMU Serif', 'DejaVu Serif'],
                        'mathtext.fontset': 'cm', 'font.size': 9, 'axes.titlesize': 10,
                        'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
                        'xtick.direction': 'out', 'ytick.direction': 'out', 'figure.dpi': 180,
                        'savefig.bbox': 'tight', 'axes.grid': False})


def save(fig, output: Path, name: str):
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f'{name}.png', dpi=180)
    fig.savefig(output / f'{name}.svg')
    return fig


def corr(a, b):
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3 or np.std(a[valid]) < 1e-12 or np.std(b[valid]) < 1e-12:
        return np.nan
    return float(np.corrcoef(a[valid], b[valid])[0, 1])


def load_records(root, expected=None):
    records = []
    for path in sorted(Path(root).rglob('meta.json')):
        meta = json.loads(path.read_text())
        with np.load(path.parent / 'spi_mpis.npz', allow_pickle=False) as bank:
            mpis = {k: bank[k] for k in bank.files}
        x = np.load(path.parent / 'timeseries.npy')
        assert x.shape == (meta['T'], meta['M']) and np.isfinite(x).all(), path
        records.append((path.parent, meta, mpis))
    if expected is not None and len(records) != expected:
        raise ValueError(f'Expected {expected} datasets, found {len(records)} under {root}')
    return records


def analyze(root, kind, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    records = load_records(root, 350)
    rows = []; catalogue = set(); configs = set(); hierarchy = 0; nonfinite = 0
    for path, meta, mpis in records:
        catalogue.add(tuple(sorted(mpis)))
        configs.add(meta['pyspi'].get('config_sha256', 'missing'))
        assert not meta['pyspi'].get('errors'), (path, meta['pyspi'].get('errors'))
        iu = np.triu_indices(meta['M'], 1)
        p = meta['generator']['params']
        row = {'path': str(path), 'class': meta['mts_class'], 'instance': meta['instance_index']}
        primary = [R, RHO, MI, DCE] if kind == 'r-rho-mi' else [ED, XD, DTW]
        for k in primary:
            assert k in mpis, (path, k)
            nonfinite += int((~np.isfinite(mpis[k][iu])).sum())
        if kind == 'r-rho-mi':
            variant = meta.get('variant')
            row['beta_tag'] = (variant.get('name') if isinstance(variant, dict) else variant) or 'baseline'
            row['beta'] = p['beta']
            row.update(r_rho=corr(mpis[R][iu], mpis[RHO][iu]), r_mi=corr(mpis[R][iu], mpis[MI][iu]),
                       rho_mi=corr(mpis[RHO][iu], mpis[MI][iu]), r_mi_dce=corr(mpis[R][iu], mpis[DCE][iu]),
                       rho_mi_dce=corr(mpis[RHO][iu], mpis[DCE][iu]),
                       mi_median_abs_change=float(np.median(abs(mpis[MI][iu] - mpis[DCE][iu]))))
            # Covariance is Pearson r only because these channels have unit variance.
            pearson = np.corrcoef(np.load(path / 'timeseries.npy'), rowvar=False)
            assert np.allclose(mpis[R][iu], pearson[iu], atol=1e-10), path
        else:
            row.update(p_step=p['p_step'], max_lag=p['max_lag'], ed_xd=corr(mpis[ED][iu], mpis[XD][iu]),
                       ed_dtw=corr(mpis[ED][iu], mpis[DTW][iu]), dtw_xd=corr(mpis[DTW][iu], mpis[XD][iu]))
            hierarchy += int(((mpis[DTW][iu] > mpis[XD][iu] + 1e-10) | (mpis[XD][iu] > mpis[ED][iu] + 1e-10)).sum())
        rows.append(row)
    frame = pd.DataFrame(rows)
    assert len(catalogue) == 1 and nonfinite == 0, (catalogue, nonfinite)
    assert len(configs) == 1 and 'missing' not in configs, configs
    groups = ['class', 'beta_tag'] if kind == 'r-rho-mi' else ['class']
    for _, group in frame.groupby(groups):
        assert sorted(group.instance) == list(range(50))
    assert hierarchy == 0, hierarchy
    assert not frame.drop(columns=['path', 'class', 'beta_tag'], errors='ignore').isna().any().any()
    frame.to_csv(output / 'per-instance.csv', index=False)
    audit = {'datasets': len(records), 'spis': list(next(iter(catalogue))), 'config_hashes': sorted(configs),
             'nonfinite_off_diagonal_values': nonfinite, 'distance_hierarchy_violations': hierarchy,
             'replication_unit': 'independent MTS instance'}
    if kind == 'r-rho-mi':
        audit['temporal_exclusion'] = {key: {'median_abs_change': float(np.median(abs(frame[key+'_dce']-frame[key]))),
                                                  'max_abs_change': float(np.max(abs(frame[key+'_dce']-frame[key])))}
                                        for key in ['r_mi', 'rho_mi']}
    (output / 'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    return frame


def raincloud(ax, values, position, color, rng, width=.2):
    values = np.asarray(values); values = values[np.isfinite(values)]
    if len(values) > 2 and np.std(values) > 1e-9:
        parts = ax.violinplot(values, positions=[position], widths=width, showextrema=False)
        for body in parts['bodies']:
            verts = body.get_paths()[0].vertices
            verts[:, 0] = np.maximum(verts[:, 0], position)
            body.set_facecolor(color); body.set_edgecolor('none'); body.set_alpha(.25)
    ax.scatter(position-rng.uniform(.025, width*.55, len(values)), values, color=color, s=7, alpha=.5, linewidths=0)
    ax.plot([position-width*.12, position+width*.12], [np.median(values)]*2, color=color, lw=1.6)


def plot_rainclouds(frame, output):
    style(); fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True, layout='constrained')
    keys = ['r_rho', 'r_mi', 'rho_mi']; labels = [r'$r$–$\rho$', r'$r$–MI', r'$\rho$–MI']
    for ax, (beta, label) in zip(axes, BETA_LABELS.items()):
        subset = frame[(frame.beta_tag == beta) | (frame['class'] == 'iter1_L')]
        for j, cls in enumerate(CASE_LABELS):
            for k, (key, color) in enumerate(zip(keys, COLORS)):
                raincloud(ax, subset[subset['class'] == cls][key], j+(k-1)*.25, color, np.random.default_rng(260924+j+k))
        ax.set(xticks=range(3), xticklabels=['Linear', '+ monotone', '+ quadratic'], ylim=(-1.05, 1.05), title=label)
        ax.axhline(0, color='.85', lw=.6)
    for color, label in zip(COLORS, labels): axes[0].plot([], [], color=color, label=label)
    axes[0].legend(loc='lower left'); axes[0].set_ylabel('Pearson agreement across channel pairs')
    return save(fig, Path(output), 'beta-rainclouds')


def plot_mi_sensitivity(frame, output):
    style(); fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True, layout='constrained')
    for ax, key, label in zip(axes, ['r_mi', 'rho_mi'], [r'$r$–MI', r'$\rho$–MI']):
        for j, cls in enumerate(['iter2_LM', 'iter3_LMNM']):
            for b, tag in enumerate(BETA_LABELS):
                d = frame[(frame['class']==cls) & (frame.beta_tag==tag)]
                raincloud(ax, d[key+'_dce']-d[key], b+(j-.5)*.28, COLORS[j], np.random.default_rng(b+j), width=.23)
            ax.plot([], [], color=COLORS[j], label=CASE_LABELS[cls])
        ax.axhline(0, color='.5', lw=.8); ax.set(xticks=range(3), xticklabels=[r'$\pi$', '5', r'$2\pi$'], xlabel=r'$\beta$', title=label)
    axes[0].set_ylabel('Change in agreement: exclusion − primary'); axes[0].legend(fontsize=7)
    return save(fig, Path(output), 'mi-temporal-exclusion')


def plot_spi_planes(frame, kind, output):
    style()
    if kind == 'r-rho-mi':
        subset = frame[(frame.instance==0) & ((frame.beta_tag=='beta-5') | (frame['class']=='iter1_L'))]
        selected = [subset[subset['class']==c].iloc[0] for c in CASE_LABELS]
        pairs = [(R,RHO),(R,MI),(RHO,MI)]; titles = list(CASE_LABELS.values()); labels={R:'Pearson r',RHO:r'Spearman $\rho$',MI:'MI (nats)'}
    else:
        selected = [frame[(frame.instance==0)&(frame.max_lag==lag)&np.isclose(frame.p_step,ps)].iloc[0] for lag,ps in [(0,0),(5,0),(5,.9)]]
        pairs=[(ED,XD),(ED,DTW),(DTW,XD)]; titles=['No lag / no warp','Lag only','Lag + heavy warp'];labels={ED:'Euclidean / √T',XD:'Shifted / √T',DTW:'DTW / √T'}
    fig, axes=plt.subplots(3,3,figsize=(9,8),layout='constrained')
    for row,(record,title) in enumerate(zip(selected,titles)):
        path=Path(record['path']);meta=json.loads((path/'meta.json').read_text());bank=np.load(path/'spi_mpis.npz');iu=np.triu_indices(meta['M'],1)
        if kind=='r-rho-mi':
            p=meta['generator']['params'];types=np.array(['L']*p['n_linear']+['M']*p['n_monotonic']+['Q']*p['n_nonmonotonic'])
            pairtypes=np.array([''.join(sorted((types[i],types[j]))) for i,j in zip(*iu)])
        else:
            lags=np.array(meta['generator']['lags']); lagdiff=abs(lags[iu[0]]-lags[iu[1]])
        for col,(a,b) in enumerate(pairs):
            ax=axes[row,col];x,y=bank[a][iu],bank[b][iu]
            if kind=='r-rho-mi':
                mixed=np.isin(pairtypes,['LM','LQ','MQ']);ax.scatter(x[~mixed],y[~mixed],s=7,c='.65',alpha=.25,linewidths=0)
                for tag,color in zip(['LM','LQ','MQ'],COLORS):
                    use=pairtypes==tag;ax.scatter(x[use],y[use],s=8,color=color,alpha=.45,linewidths=0,label=tag.replace('Q','quadratic'))
            else:
                sc=ax.scatter(x,y,c=lagdiff,cmap='viridis',vmin=0,vmax=5,s=9,alpha=.55,linewidths=0)
                limit=max(x.max(),y.max())*1.04;ax.plot([0,limit],[0,limit],color='.7',lw=.7,ls='--');ax.set(xlim=(0,limit),ylim=(0,limit))
            ax.set(xlabel=labels[a],ylabel=labels[b]);ax.set_box_aspect(1)
            ax.set_title(f'{title}\nagreement = {corr(x,y):.2f}',fontsize=9)
    if kind=='dtw':fig.colorbar(sc,ax=axes.ravel().tolist(),label='Absolute generating lag difference',shrink=.5)
    else:axes[-1,-1].legend(fontsize=7,loc='lower right')
    return save(fig,Path(output),'spi-planes')


def plot_dtw_sweep(frame, output):
    style();fig,ax=plt.subplots(figsize=(7.5,3.7),layout='constrained')
    conditions=[(0,0),(5,0),(5,.1),(5,.3),(5,.5),(5,.7),(5,.9)]
    for key,color,label in zip(['ed_xd','ed_dtw','dtw_xd'],COLORS,['Euclidean–shifted','Euclidean–DTW','DTW–shifted']):
        vals=[frame[(frame.max_lag==lag)&np.isclose(frame.p_step,ps)][key].to_numpy() for lag,ps in conditions]
        means=np.array([v.mean() for v in vals]);cis=np.array([student_t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v)) for v in vals])
        ax.errorbar(range(7),means,yerr=cis,color=color,label=label,marker='o',ms=3,lw=1.2,capsize=2)
    ax.set(xticks=range(7),xticklabels=['No lag','0','.1','.3','.5','.7','.9'],xlabel=r'Warp probability $p_{\rm step}$ (lag range 0–5 except first control)',ylabel='Pearson agreement across channel pairs',ylim=(-1.05,1.05))
    ax.legend(fontsize=8);ax.axvline(.5,color='.85',lw=.7);ax.axhline(0,color='.85',lw=.6)
    return save(fig,Path(output),'warp-sweep')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('kind',choices=['r-rho-mi','dtw']);parser.add_argument('root',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();frame=analyze(args.root,args.kind,args.output)
    if args.kind=='r-rho-mi':
        plot_rainclouds(frame,args.output);plot_mi_sensitivity(frame,args.output)
    else:plot_dtw_sweep(frame,args.output)
    plot_spi_planes(frame,args.kind,args.output);plt.close('all')


if __name__=='__main__':main()
