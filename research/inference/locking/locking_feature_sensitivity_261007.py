"""Retrospective feature-family sensitivity; preserves all original locking readouts."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'results/order-parameter-inference/cross-frequency-locking-confirm-261006'
OUT = ROOT / 'results/order-parameter-inference/locking-preview-261007'
LAG_PREFIXES = ('corr_', 'xcorr')
TEMPORAL_PREFIXES = (*LAG_PREFIXES, 'cohmag_', 'plv_')


def feature_masks(names):
    """Masks depend only on catalogue names, never on outcomes."""
    names = list(map(str, names))
    a, b = np.triu_indices(len(names), 1)
    lag = np.array([s.startswith(LAG_PREFIXES) for s in names])
    temporal = np.array([s.startswith(TEMPORAL_PREFIXES) for s in names])
    return ~(lag[a] | lag[b]), temporal, a, b


def project(x, fit, standard=False):
    """Original train-complete filter and median fill, in float64 for replay."""
    x = np.asarray(x, dtype=float)
    keep = np.isfinite(x[fit]).all(0)
    keep[keep] &= x[fit][:, keep].std(0) > 1e-10
    if not keep.any():
        raise ValueError('No eligible development features')
    selected = x[:, keep]
    missing = (~np.isfinite(selected)).mean(1)
    fill = np.median(selected[fit], axis=0)
    selected = np.where(np.isfinite(selected), selected, fill)
    center = selected[fit].mean(0)
    scale = selected[fit].std(0) if standard else np.ones(keep.sum())
    transformed = (selected - center) / scale
    model = PCA(1, svd_solver='full').fit(transformed[fit])
    q = model.transform(transformed)[:, 0]
    return q, missing, keep, model.components_[0], transformed, float(model.explained_variance_ratio_[0])


def analyze(source=SOURCE, out=OUT):
    out.mkdir(parents=True, exist_ok=True)
    bank = np.load(source / 'features.npz')
    rows = pd.read_csv(source / 'scores.csv')
    np.testing.assert_array_equal(bank['row_id'], rows.row_id)
    no_lag, temporal, a, b = feature_masks(bank['spi_order'])
    z = bank['z']
    means = bank['mean']
    summaries, records, groups = [], [], []
    audit = dict(source_features_sha256=hashlib.sha256((source/'features.npz').read_bytes()).hexdigest(),
                 script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 omitted_spi_names=bank['spi_order'][np.array([str(s).startswith(LAG_PREFIXES) for s in bank['spi_order']])].tolist(),
                 temporal_baseline_spi_names=bank['spi_order'][temporal].tolist(),
                 scope='Retrospective diagnostic; no new physical data or p90. Masks fixed by identifiers, no held-Q selection. Original PC1 unchanged.')
    for arm in dict.fromkeys(rows.system):
        index = np.flatnonzero(rows.system.eq(arm))
        part = rows.iloc[index].copy().reset_index(drop=True)
        fit = part.role.eq('development').to_numpy()
        held = part.role.eq('evaluation').to_numpy() & part.comparison_eligible.to_numpy()
        common = held.copy()
        fitted = {}
        for label, x, standard in [('full_z_PC1', z[index], False),
                                    ('nonlag_z_PC1', z[index][:, no_lag], False),
                                    ('temporal_mean_PC1', means[index][:, temporal], True)]:
            q, missing, keep, weights, transformed, evr = project(x, fit, standard)
            sign = 1 if spearmanr(q[fit], part.control[fit]).statistic >= 0 else -1
            q, weights = q * sign, weights * sign
            if label == 'full_z_PC1':
                error = float(abs(q - part.z_PC1.to_numpy()).max())
                assert error < 1e-4, error
                audit[arm] = dict(original_q_replay_max_error=error)
                lo = fit & np.isclose(part.control, .08)
                hi = fit & np.isclose(part.control, .12)
                contribution = weights * (transformed[hi].mean(0)-transformed[lo].mean(0)) / q[fit].std()
                assert np.isclose(contribution.sum(), (q[hi].mean()-q[lo].mean())/q[fit].std())
                active = np.flatnonzero(keep)
                def family(name):
                    name = str(name)
                    return 'lagged-correlation' if name.startswith(LAG_PREFIXES) else name.split('_')[0]
                labels = [' × '.join(sorted([family(bank['spi_order'][a[j]]), family(bank['spi_order'][b[j]])])) for j in active]
                gf = pd.DataFrame(dict(family_pair=labels, signed=contribution, absolute=abs(contribution)))
                gf = gf.groupby('family_pair').agg(signed_contribution=('signed','sum'),
                                                    absolute_contribution=('absolute','sum'), count=('signed','size')).reset_index()
                gf['absolute_share'] = gf.absolute_contribution / gf.absolute_contribution.sum()
                gf['arm'] = arm
                groups.append(gf)
            part[label] = q / q[fit].std()
            common &= missing <= .05
            fitted[label] = dict(features=int(keep.sum()), max_held_missing=float(missing[held].max()), evr=evr)
        counts = part[common].groupby('control').size()
        assert len(counts) == 21 and counts.min() >= 4
        part['sensitivity_eligible'] = common | fit
        for label in ['full_z_PC1','nonlag_z_PC1','temporal_mean_PC1','mean_PC1','mean_abs_r']:
            summaries.append(dict(arm=arm,method=label,n=int(common.sum()),
                                  abs_rho_Q=float(abs(spearmanr(part.loc[common,label],part.Q_lock[common]).statistic)),
                                  abs_rho_nuisance=float(abs(spearmanr(part.loc[common,label],part.nuisance[common]).statistic)),
                                  **fitted.get(label, {})))
        records.append(part)
    metrics = pd.DataFrame(summaries)
    scores = pd.concat(records, ignore_index=True)
    family = pd.concat(groups, ignore_index=True)
    metrics.to_csv(out/'sensitivity-metrics.csv',index=False)
    scores.to_csv(out/'sensitivity-scores.csv',index=False)
    family.to_csv(out/'family-contributions.csv',index=False)
    (out/'sensitivity-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    plot(scores, out)
    print(metrics[['arm','method','n','abs_rho_Q','abs_rho_nuisance','features']].round(3).to_string(index=False))
    print(family[family.arm.eq('random-noise')].nlargest(8,'absolute_share').round(4).to_string(index=False))
    return metrics


def plot(scores, out):
    from research.inference.locking.plot_locking_composites_261008 import independent_limits
    from scripts.benchmark_latex_style import latex_style, prepare_latex_figure
    plt.rcParams.update({'font.family':'serif','font.serif':['CMU Serif','DejaVu Serif'],
                         'mathtext.fontset':'cm','font.size':9,'axes.spines.top':False,
                         'axes.spines.right':False,'legend.frameon':False})
    latex_style()
    fig, axes = plt.subplots(1,3,figsize=(11.8,3.4),constrained_layout=True)
    colors = ['#0072B2','#D55E00','#009E73']
    labels = [('full_z_PC1','Full SPI–SPI PC1'),('nonlag_z_PC1','SPI–SPI: no lagged correlation'),
              ('temporal_mean_PC1','Lagged/spectral mean-PC1')]
    for ax,arm in zip(axes,dict.fromkeys(scores.system)):
        part=scores[scores.system.eq(arm)]
        held=part[part.role.eq('evaluation')&part.sensitivity_eligible]
        group=held.groupby('control');right=ax.twinx();right.spines['right'].set_visible(True)
        lines=ax.plot(group.Q_lock.mean().index,group.Q_lock.mean(),'o-',color='#222222',label='$Q$',ms=2.7)
        ranges=[]
        for (key,name),color in zip(labels,colors):
            g=group[key];m=g.mean()
            lines+=right.plot(m.index,m,'s-',color=color,label=name,ms=2.7,lw=1.7)
            right.fill_between(m.index,g.quantile(.1),g.quantile(.9),alpha=.1,color=color,lw=0)
            ranges.extend([m.to_numpy(),g.quantile(.1).to_numpy(),g.quantile(.9).to_numpy()])
        ax.set(title=arm,xlabel=r'Cross-coupling $\gamma$',ylabel='Locking index $Q$',ylim=(-.05,1.05))
        extent=np.concatenate(ranges)
        right.set(ylabel='$q$ (fit SD)',ylim=independent_limits(extent,extent,extent))
        if arm != 'internal-coupling':ax.axvline(.1017,color='.6',ls=':',lw=.8)
        if ax is axes[0]:ax.legend(lines,[l.get_label() for l in lines],fontsize=6.5,loc='upper left')
    fig.suptitle('Retrospective family sensitivity · M=N=48, T=1000 · held means and 10–90% instance bands\nIndependent right-axis ranges include all displayed curves and bands; scores unchanged',fontsize=10)
    prepare_latex_figure(fig)
    fig.canvas.draw();fig.set_layout_engine('none')
    for ext in ['png','svg']:fig.savefig(out/f'family-sensitivity.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        analyze()
