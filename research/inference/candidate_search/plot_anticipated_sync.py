"""Figure for the delayed -> anticipated synchronization (DS -> AS) p90 check, in the composite style.

Presentation only: q and baselines are the archived target-blind readouts in `scores.csv`; the PC1 refit here
reproduces them and is used only to rank features by |loading| (never Q). Snapshot records follow a fixed rule:
first held seed at the first, middle and last control. Run after `anticipated_sync_bank extract` and `analyze`.

Run: python -m research.inference.candidate_search.plot_anticipated_sync
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from scipy.stats import spearmanr

from mtsspi.spi_spi_contract import build_unified_schema
from research.inference.benchmarks.composite_figures import style, Q_COLOR, Z_COLOR, BASELINES
from research.inference.candidate_search import anticipated_sync_bank as bank
from research.inference.candidate_search import scout_anticipated_sync as gen

OUT = bank.OUT
SNAP_SEED, M = bank.FIT_SEEDS, gen.MODULES          # first held seed
READOUTS = [('z_PC1', Z_COLOR, r'$q$ (PC1 of $z$)'), ('mean_abs_r', BASELINES['mean-|r|'], r'mean-$|r|$'),
            ('mean_PC1', BASELINES['mean-MPI'], 'mean-MPI PC1')]
EDGE_COLORS = {'S$\\to$R': '#c51b7d', 'R$\\to$S': '#4d9221', 'unrelated': '#bdbdbd'}


def loadings(z, fit):
    """Replays `report_dependence_transition.coordinate` (centred) and returns scores, PC1 weights and kept mask."""
    from sklearn.decomposition import PCA
    keep = np.isfinite(z[fit]).all(0) & (np.nanstd(z[fit], 0) > 1e-10)
    x = z[:, keep]; fill = np.median(x[fit], 0); x = np.where(np.isfinite(x), x, fill)
    x = x - x[fit].mean(0); model = PCA(1, svd_solver='full').fit(x[fit])
    return model.transform(x)[:, 0], model.components_[0], keep


def classify(anti, mean_anti_fit, control_fit):
    """Symmetric (no module asymmetry), lead-following (asymmetry changes sign with the control) or coupling-following."""
    if np.nanmax(abs(anti)) < 1e-9:
        return 'symmetric'
    rho = spearmanr(anti, control_fit).statistic
    return 'lead' if abs(rho) > .8 and np.sign(anti.min()) != np.sign(anti.max()) else 'coupling'


def edge_types(order):
    pos = np.argsort(order); kind = np.full((2 * M, 2 * M), 'unrelated', dtype=object)
    for i in range(M):
        kind[pos[i], pos[i + M]] = 'S$\\to$R'; kind[pos[i + M], pos[i]] = 'R$\\to$S'
    return kind[~np.eye(2 * M, dtype=bool)]


def headline(ax, part, title, targets=('Q',)):
    held = part[part.role.eq('evaluation')]; fit = part[part.role.eq('development')]
    g = held.groupby('control').Q
    ax.plot(g.mean().index, g.mean(), 'o-', color=Q_COLOR, ms=2.2, lw=1.2, label='$Q$ (signed lag)')
    ax.axhline(0, color='.7', lw=.5); ax.set(xlabel='Receiver detuning $d$', ylabel='$Q$ (rad)', title=title)
    right = ax.twinx(); right.set_yticks([]); right.spines['right'].set_visible(False)
    for col, color, name in READOUTS:
        y = (held[col] - fit[col].mean()) / fit[col].std(); g = y.groupby(held.control)
        rhos = ' / '.join(f"{abs(spearmanr(held[col], held[t]).statistic):.2f}" for t in targets)
        right.plot(g.mean().index, g.mean(), 's-', color=color, ms=2, lw=1.1, label=f'{name}  {rhos}')
        right.fill_between(g.mean().index, g.quantile(.1), g.quantile(.9), color=color, alpha=.13, lw=0)
    lines = ax.get_lines()[:1] + right.get_lines()
    head = r'$|\rho_s|$ with ' + ' / '.join({'Q': '$Q$', 'absQ': '$|Q|$'}[t] for t in targets)
    ax.legend(lines, [l.get_label() for l in lines], title=head, fontsize=5.5, title_fontsize=5.5, loc='upper center',
              bbox_to_anchor=(.5, -.22), ncol=2, frameon=False)


def main():
    style()
    rows = pd.DataFrame(json.loads((bank.DATA / 'manifest.json').read_text())['rows'])
    scores = pd.read_csv(OUT / 'scores.csv'); feats = np.load(OUT / 'features.npz')
    order = feats['spi_order'].tolist(); schema = build_unified_schema(order)
    raw = np.load(bank.DATA / 'observations.npz')
    narrow = rows.arm.eq('narrow').to_numpy(); nfit = narrow & rows.role.eq('development').to_numpy()
    q, w, keep = loadings(feats['z'][narrow], rows.role[narrow].eq('development').to_numpy())
    ref = scores.set_index('row_id').loc[rows.row_id[narrow], 'z_PC1'].to_numpy()
    assert abs(abs(np.corrcoef(q, ref)[0, 1]) - 1) < 1e-6, 'PC1 replay does not reproduce the archived q'
    kept = np.flatnonzero(keep); top = kept[np.argsort(-abs(w))[:20]]
    zn = feats['z'][narrow][:, kept]; zn = np.where(np.isfinite(zn), zn, np.nanmedian(zn[nfit[narrow]], 0))
    dev_rho = np.array([abs(spearmanr(zn[nfit[narrow], j], rows.Q[nfit]).statistic) for j in range(len(kept))])
    sign_feature = kept[np.nanargmax(dev_rho)]  # selected on development Q: explanatory, not an unsupervised readout
    kinds = {k: classify(feats['anti'][nfit, i], None, rows.control[nfit].to_numpy()) for i, k in enumerate(order)}

    fig = plt.figure(figsize=(7.2, 9.0))
    gs = fig.add_gridspec(3, 6, height_ratios=[1, 1.15, 1.05], hspace=.8, wspace=1.6, left=.08, right=.97, top=.965, bottom=.075)

    # (a) system
    ax = fig.add_subplot(gs[0, :2]); ax.set_axis_off(); ax.set(xlim=(0, 1), ylim=(0, 1))
    for xy, name in [((.2, .72), 'S'), ((.8, .72), 'R')]:
        ax.add_patch(Circle(xy, .1, fc='white', ec=Q_COLOR, lw=.8)); ax.text(*xy, name, ha='center', va='center', fontsize=9)
    ax.add_patch(FancyArrowPatch((.31, .72), (.69, .72), arrowstyle='-|>', mutation_scale=8, lw=.8, color=Q_COLOR))
    ax.text(.5, .78, '$K$', ha='center', fontsize=7)
    ax.text(.5, .4, r'$\dot Z_S=(1+i\omega_S-|Z_S|^2)Z_S+\xi_S$' '\n' r'$\dot Z_R=(1+i(\omega_S+d)-|Z_R|^2)Z_R$' '\n'
            r'$\qquad\quad+K(Z_S-Z_R)+\xi_R$', ha='center', va='center', fontsize=6.2)
    ax.text(.5, .02, r'8 modules, $\omega_S\in[0.8,1.2]$, $K=0.4$; observe Re$\,Z$' '\n' r'$M=N=16$, $T=1000$, channels shuffled',
            ha='center', va='bottom', fontsize=5.6)
    ax.set_title('(a) Sender $\\to$ receiver modules', loc='left')

    # (b) traces: wide arm, first held seed, physical module 0
    ax = fig.add_subplot(gs[0, 2:4]); wide = rows[rows.arm.eq('wide') & rows.seed.eq(SNAP_SEED)].sort_values('control')
    for k, (_, r) in enumerate(wide.iloc[[0, 6, 12]].iterrows()):
        x = raw[r.row_id]; pos = np.argsort(r.channel_order); t = np.arange(60) * gen.STEP; off = -2.9 * k
        ax.plot(t, x[pos[0], :60] + off, color='#c51b7d', lw=.8, label='sender' if k == 0 else None)
        ax.plot(t, x[pos[M], :60] + off, color='#4d9221', lw=.8, ls='--', label='receiver' if k == 0 else None)
        ax.text(30.8, off, f"$d={r.control:+.2f}$\n$Q={r.Q:+.2f}$", fontsize=5.5, va='center')
    ax.set(xlabel='Time', yticks=[], xlim=(0, 40), xticks=[0, 10, 20, 30]); ax.legend(fontsize=5.5, loc='upper center', ncol=2, bbox_to_anchor=(.5, -.2), frameon=False)
    ax.set_title('(b) DS / zero lag / AS', loc='left')

    # (c) order parameter
    ax = fig.add_subplot(gs[0, 4:])
    for arm, color in [('wide', '.55'), ('narrow', Q_COLOR)]:
        g = rows[rows.arm.eq(arm)].groupby('control').Q
        ax.plot(g.mean().index, g.mean(), 'o-', color=color, ms=2, lw=1, label=f'{arm} arm')
    ax.axhline(0, color='.7', lw=.5); ax.axvline(0, color='.7', lw=.5)
    ax.text(-.1, .62, 'DS: R lags\n($Q>0$)', fontsize=5.5, ha='center'); ax.text(.13, -.62, 'AS: R leads\n($Q<0$)', fontsize=5.5, ha='center')
    ax.set(xlabel='Receiver detuning $d$', ylabel=r'$Q=\arg\langle Z_S\bar Z_R\rangle$ (rad)'); ax.legend(fontsize=5.5, loc='upper right')
    ax.set_title('(c) Order parameter', loc='left')

    # (d, e) headlines
    s = scores.merge(rows[['row_id', 'channel_order']], on='row_id')
    headline(fig.add_subplot(gs[1, :3]), s[s.arm.eq('narrow')], '(d) Narrow arm, $|Q|\\leq0.1$: held records', ('Q', 'absQ'))
    headline(fig.add_subplot(gs[1, 3:]), s[s.arm.eq('wide')], '(e) Wide arm, $|Q|\\leq0.8$: held records', ('Q', 'absQ'))
    for a in fig.axes[3:]:
        a.title.set_ha('left'); a.title.set_x(0)

    # (f) edge plane of the top-|loading| feature, three held narrow records
    a_name, b_name = schema[sign_feature].spi_a, schema[sign_feature].spi_b
    sub = gs[2, :3].subgridspec(1, 3, wspace=.12); snaps = rows[narrow & rows.seed.eq(SNAP_SEED)].sort_values('control').iloc[[0, 6, 12]]
    for k, (_, r) in enumerate(snaps.iterrows()):
        ax = fig.add_subplot(sub[k]); folder = bank.DATA / 'mpis' / bank.RUN / f"{r.corpus_index + 1:04d}-{r.row_id}"
        with np.load(folder / 'spi_mpis.npz') as mp:
            va, vb = (mp[n][~np.eye(2 * M, dtype=bool)] for n in (a_name, b_name))
        kind = edge_types(r.channel_order)
        for lab, color in EDGE_COLORS.items():
            sel = kind == lab; ax.scatter(va[sel], vb[sel], s=3 if lab != 'unrelated' else 1.2, color=color, lw=0, label=lab, zorder=3 if lab != 'unrelated' else 1)
        ax.set_title(f"$d={r.control:+.3f}$\n$z={np.corrcoef(va, vb)[0, 1]:+.2f}$", fontsize=6)
        first = ax if k == 0 else first
        ax.tick_params(labelsize=5); ax.set_xticks([]); ax.set_yticks([])
        if k == 0:
            ax.set_ylabel(b_name.replace('_', r'\_')[:28], fontsize=5)
        if k == 1:
            ax.set_xlabel(a_name.replace('_', r'\_')[:34], fontsize=5)
        if k == 2:
            ax.legend(fontsize=5, loc='lower right', handletextpad=.1, borderpad=.2, markerscale=1.5)
    box = first.get_position(); fig.text(box.x0 - .01, box.y1 + .045, '(f) Sign-carrying $z$ feature (chosen on development $Q$; explanatory)', fontsize=8)

    # (g) module asymmetry and ordered mean of the two SPIs, narrow arm
    ax = fig.add_subplot(gs[2, 3:5]); held = narrow & rows.role.eq('evaluation').to_numpy()
    for name, color in [(a_name, '#c51b7d'), (b_name, '#4d9221')]:
        i = order.index(name); norm = abs(feats['anti'][nfit, i]).max()
        g = pd.Series(feats['anti'][held, i] / norm).groupby(rows.control[held].to_numpy())
        ax.plot(g.mean().index, g.mean(), 'o-', color=color, ms=2, lw=1, label=f'{kinds[name]}: S$\\to$R $-$ R$\\to$S')
        mu = feats['mean'][:, i]; mfit = mu[nfit]
        g = pd.Series((mu[held] - mfit.mean()) / (mfit.std() or 1) / 10).groupby(rows.control[held].to_numpy())
        ax.plot(g.mean().index, g.mean(), ':', color=color, lw=1, label='ordered-edge mean (SD/10)')
    ax.axhline(0, color='.7', lw=.5); ax.set(xlabel='Receiver detuning $d$', ylabel='scaled value', title='(g) What flips, what does not')
    ax.title.set_ha('left'); ax.title.set_x(0); ax.legend(fontsize=4.6, loc='lower right', ncol=1, frameon=False)

    # (h) PC1 composition by pair class
    ax = fig.add_subplot(gs[2, 5]); cls = np.array(['-'.join(sorted((kinds[schema[j].spi_a], kinds[schema[j].spi_b]))) for j in kept])
    share = pd.Series(w ** 2).groupby(cls).sum(); base = pd.Series(np.ones(len(cls)) / len(cls)).groupby(cls).sum()
    labels = sorted(share.index, key=lambda c: -share[c])[:5]
    y = np.arange(len(labels)); ax.barh(y, [share[c] for c in labels], color=Z_COLOR, height=.4, label='PC1 $w^2$')
    ax.barh(y + .4, [base[c] for c in labels], color='.75', height=.4, label='feature count')
    short = {'symmetric': 'sym', 'lead': 'lead', 'coupling': 'coup'}
    ax.set_yticks(y + .2, [r' $\times$ '.join(short[t] for t in c.split('-')) for c in labels], fontsize=5.5); ax.invert_yaxis()
    ax.set_xlabel('share', fontsize=6); ax.tick_params(axis='x', labelsize=5.5)
    ax.legend(fontsize=4.8, loc='upper right', bbox_to_anchor=(1.05, -.18), frameon=False, ncol=1); ax.set_title('(h) PC1 by class', fontsize=8)
    ax.title.set_ha('left'); ax.title.set_x(-.6)

    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(OUT / f'anticipated-sync-figure.{ext}', dpi=300 if ext == 'png' else None)
    summary = dict(sign_feature=dict(a=a_name, b=b_name, dev_rho=float(np.nanmax(dev_rho)), kinds=[kinds[a_name], kinds[b_name]]),
                   top_features=[dict(a=schema[j].spi_a, b=schema[j].spi_b, w=float(w[np.searchsorted(kept, j)]),
                                     kinds=[kinds[schema[j].spi_a], kinds[schema[j].spi_b]]) for j in top],
                   class_counts=pd.Series(kinds).value_counts().to_dict(), pc1_share_by_class=share.round(4).to_dict(),
                   feature_share_by_class=base.round(4).to_dict())
    (OUT / 'figure-summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1)[:3000])


if __name__ == '__main__':
    main()
