"""ACF figure for preregs/robust-prereg.md (descriptive): decay in seconds, tau_1/e against sample spacing, N_eff against
window length and rate, AUTO Theiler window against its caps. Input: robust_acf.py compute output.

  python robust_acf.py figure <acf.npz> <stem>        writes <stem>.svg/.pdf/.png
"""
import shutil
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mtsspi.figures import SANS, SANS_PREAMBLE
from robust_acf import FS, ORDER, cross, group

mpl.rcParams.update({
    "text.usetex": bool(shutil.which("latex") and shutil.which("dvipng")),
    "text.latex.preamble": SANS_PREAMBLE, **SANS,
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.top": False, "ytick.right": False,
    "axes.grid": False, "legend.frameon": False,
    "lines.linewidth": 1.7, "lines.markersize": 2.7,
    "figure.constrained_layout.use": True,
    "figure.dpi": 180, "savefig.dpi": 600, "savefig.bbox": "tight",
})
# state colours as plot_character_draft.py
COL = {'awake_ec': '#8C969C', 'awake_eo': '#8C969C', 'KT': '#C8553D', 'KTMD': '#7B4B94', 'MD': '#2E86AB', 'PF': '#D9A21B',
       'sleep': '#3E5C76', 'recovery_ec': '#78838A'}
LS = {'awake_eo': '--', 'recovery_ec': ':'}
LAB = {'awake_ec': 'awake EC', 'awake_eo': 'awake EO', 'sleep': 'sleep', 'recovery_ec': 'recovery EC'}
INK = '#2B2B2B'


def figure(f, stem):
    D = np.load(f); g = group(D); S = D['S']; A = D['ACF']; WA = D['WA']; lag = np.arange(A.shape[-1]) / FS
    fig, ax = plt.subplots(1, 4, figsize=(7.4, 2.7), gridspec_kw=dict(width_ratios=[1.2, 1, 1.05, 1]))
    # (a) unit-averaged channel ACF, median over units x channels per state
    tau = {}
    for s in ORDER:
        m = g == s; cur = []
        for u in np.unique(D['unit'][m]):
            cur.append(A[m & (D['unit'] == u)].mean(0))
        cur = np.concatenate(cur); med = np.median(cur, 0)
        ax[0].plot(lag, med, color=COL[s], ls=LS.get(s, '-'), lw=1.2, label=LAB.get(s, s))
        tau[s] = S[m][:, :, 0].ravel() / FS
    ax[0].axhline(1 / np.e, color=INK, lw=.6, ls=(0, (2, 2))); ax[0].axhline(0, color=INK, lw=.5)
    ax[0].text(.6, 1 / np.e + .03, r'$1/e$', ha='right', fontsize=8)
    for dt, r in ((.05, '20 Hz'), (.1, '10 Hz')):
        ax[0].axvline(dt, color='#D9DDE0', lw=.8, zorder=0); ax[0].text(dt + .006, -.33, r, fontsize=6.5, ha='left', va='bottom', rotation=90)
    ax[0].set(xlim=(0, .6), ylim=(-.35, 1.05), xlabel='lag (s)', ylabel='ACF (median)')
    fig.legend(*ax[0].get_legend_handles_labels(), ncol=8, fontsize=7, handlelength=1.8, columnspacing=1.2, loc='outside lower center')
    # (b) tau_1/e per window x channel
    pos = np.arange(len(ORDER))
    bp = ax[1].boxplot([tau[s][np.isfinite(tau[s])] for s in ORDER], positions=pos, widths=.6, showfliers=False,
                       patch_artist=True, medianprops=dict(color=INK, lw=1), whiskerprops=dict(lw=.7), capprops=dict(lw=.7))
    for b, s in zip(bp['boxes'], ORDER):
        b.set(facecolor=COL[s], alpha=.75, lw=.6)
    for dt, r in ((.01, '100 Hz'), (.05, '20 Hz'), (.1, '10 Hz')):
        ax[1].axhline(dt, color=INK, lw=.5, ls=(0, (2, 2))); ax[1].text(1.01, dt, r, fontsize=6.5, va='center', ha='left', transform=ax[1].get_yaxis_transform())
    ax[1].set(xticks=pos, yscale='log', ylim=(.005, .5), ylabel=r'$\tau_{1/e}$ (s)')
    ax[1].set_xticklabels([LAB.get(s, s).replace('awake ', '').replace('recovery EC', 'rec.') for s in ORDER], rotation=60, fontsize=7)
    # (c) N_eff = T / (2 tau) against window length; raw N per rate in grey
    Ts = np.array([2, 10, 50, 100])
    for hz in (100, 20, 10):
        ax[2].plot(Ts, Ts * hz, color='#B9BFC3', lw=.9, ls=(0, (3, 1.5))); ax[2].text(115, Ts[-1] * hz, f'$N$ {hz} Hz', fontsize=6.5, va='center', clip_on=False)
    for s in ('awake_ec', 'KT', 'KTMD', 'MD', 'PF', 'sleep'):
        ax[2].plot(Ts, Ts / (2 * np.nanmedian(tau[s])), color=COL[s], ls=LS.get(s, '-'), lw=1.2, marker='o')
    ax[2].set(xscale='log', yscale='log', xticks=Ts, xlim=(1.6, 110), xlabel='window (s)', ylabel=r'$N_\mathrm{eff} = T / 2\tau_{1/e}$')
    ax[2].set_xticklabels([str(t) for t in Ts], fontsize=7); ax[2].minorticks_off()
    # (d) uncapped AUTO window per pair, samples at 100 Hz, against the caps
    bins = np.arange(0, 61, 1.5)
    for s in ('awake_ec', 'KT', 'KTMD', 'MD', 'PF', 'sleep'):
        w = WA[g == s].ravel(); h, _ = np.histogram(w, bins, density=True)
        ax[3].plot(bins[:-1] + .75, h, color=COL[s], ls=LS.get(s, '-'), lw=1.2)
    ax[3].axvline(20, color=INK, lw=.6, ls=(0, (2, 2))); ax[3].text(21, ax[3].get_ylim()[1] * .92, 'cap at 2 s', fontsize=6.5)
    ax[3].set(xlim=(0, 60), xlabel=r'AUTO $w = 2\langle a_i, a_j\rangle$ (samples)', ylabel='density')
    for a, l in zip(ax, 'abcd'):
        a.set_title(f'({l})', loc='left', fontsize=9, fontweight='bold')
    for ext in ('svg', 'pdf', 'png'):
        fig.savefig(f'{stem}.{ext}', dpi=180 if ext == 'png' else None)
