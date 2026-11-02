"""Working figure: (a) selectivity matrix, each measure's response to every knob relative to its own knob;
(b) own-knob response curves for the best measure per construct."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy import stats

from mtsspi.figures import save, style

KNOB_COLS = [('A', 'lag_level'), ('A', 'lag_disp'), ('A', 'band_level'), ('A', 'band_disp'), ('A', 'form_level'),
             ('A', 'form_disp'), ('A', 'warp_level'), ('A', 'warp_disp'), ('B', 'dir_level'), ('B', 'dir_disp'),
             ('C', 'med'), ('K', 'lag_level'), ('K', 'lag_disp'), ('K', 'band_level'),
             ('A', 'nuis_cv_s'), ('A', 'nuis_noise'), ('A', 'nuis_T'), ('A', 'nuis_r'), ('A', 'nuis_beta'),
             ('A', 'nuis_mix'), ('A', 'nuis_M')]
LABEL = {'lag_level': 'lag level', 'lag_disp': 'lag disp.', 'band_level': 'band level', 'band_disp': 'band disp.',
         'form_level': 'form level', 'form_disp': 'form disp.', 'warp_level': 'warp level', 'warp_disp': 'warp disp.',
         'dir_level': 'dir. level', 'dir_disp': 'dir. disp.', 'med': 'mediated share', 'nuis_cv_s': 'strength CV',
         'nuis_noise': 'noise', 'nuis_T': 'T', 'nuis_r': 'autocorr. r', 'nuis_beta': 'marginal map',
         'nuis_mix': 'mixing', 'nuis_M': 'M'}


def curve(df, fam, knob, col, baseline='B0', surrogate=False):
    s = df[(df.family == fam) & (df.knob == knob) & (df.baseline == baseline) & (df.grid == '')
           & (df.surrogate == surrogate)]
    if s.empty or col not in s:
        return None
    g = s.groupby('level')[col]
    m, sd, n = g.mean(), g.std(ddof=1), g.count()
    ci = stats.t.ppf(0.975, (n - 1).clip(lower=1)) * sd / np.sqrt(n)
    return m.index.to_numpy(), m.to_numpy(), ci.to_numpy()


def make(df, verdicts, resp, out: Path):
    from research.representation.character_axes.validation.analyse import OWN_LEVEL, OWN_ORG, SECOND, measures
    style()
    ms = [m for m in measures() if m[1] != 'control']
    ver = verdicts.set_index('measure')
    R = np.full((len(ms), len(KNOB_COLS)), np.nan)
    own_col = {}
    for i, (mid, construct, _, kind, col) in enumerate(ms):
        own = (OWN_ORG if kind == 'org' else OWN_LEVEL).get(construct)
        sel = resp[(resp.measure == mid) & (resp.grid.fillna('') == '') & (resp.baseline == 'B0')]
        own_mean = ver.loc[mid, 'own_mean'] if mid in ver.index else np.nan
        for j, (fam, knob) in enumerate(KNOB_COLS):
            v = sel[(sel.family == fam) & (sel.knob == knob)]['mean']
            if len(v) and np.isfinite(own_mean) and own_mean != 0:
                R[i, j] = v.iloc[0] / abs(own_mean)
            if own and (fam, knob) == own[:2]:
                own_col[i] = j
    fig = plt.figure(figsize=(11, 8.6))
    ax = fig.add_axes([0.25, 0.08, 0.42, 0.86])
    im = ax.imshow(np.clip(R, -3, 3), cmap='RdBu_r', vmin=-3, vmax=3, aspect='auto', interpolation='nearest')
    for i, j in own_col.items():
        ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, lw=1.2, ec='k'))
    ax.set_xticks(range(len(KNOB_COLS)), [f'{f}: {LABEL.get(k, k)}' for f, k in KNOB_COLS], rotation=90, fontsize=7)
    names = []
    for mid, *_ in ms:
        v = ver.loc[mid, 'verdict'] if mid in ver.index else ''
        names.append(f'{mid}  [{v}]')
    ax.set_yticks(range(len(ms)), names, fontsize=6.5)
    ax.axvline(13.5, color='k', lw=.6)
    ax.set_title('a  response to each knob / |own response| (B0); own knob outlined', fontsize=9, loc='left')
    cb = fig.colorbar(im, ax=ax, fraction=.03, pad=.01)
    cb.set_label('relative response (clipped at ±3)', fontsize=7)
    # (b) curves: best measure per construct by own paired d
    best = (verdicts[verdicts.verdict.isin(['core', 'sub-axis'])]
            .assign(ad=lambda t: t.own_d.abs()).sort_values('ad', ascending=False).drop_duplicates('construct'))
    cons = list(best.itertuples())[:6]
    for k, r in enumerate(cons):
        a2 = fig.add_axes([0.75, 0.83 - k * 0.15, 0.22, 0.11])
        mid = r.measure
        col = ('org|' if r.kind == 'org' else 'level|') + mid
        fam, knob = r.own_knob.split(':')
        for bl, ls, surr, lab in (('B0', '-', False, 'B0'), ('B1', '--', False, 'B1'), ('B0', ':', True, 'surrogate')):
            c = curve(df, fam, knob, col, bl, surr)
            if c is not None:
                a2.errorbar(c[0], c[1], yerr=c[2], ls=ls, lw=1.2, color='k' if not surr else '0.5', capsize=1.5,
                            label=lab)
        sec = SECOND.get((r.construct, r.kind))
        if sec:
            c = curve(df, sec[0], sec[1], col)
            if c is not None:
                a2.errorbar(c[0], c[1], yerr=c[2], color='tab:orange', lw=1.2, capsize=1.5, label=f'family {sec[0]}')
        a2.set_title(f'{mid} ({r.verdict})', fontsize=7, loc='left')
        a2.set_xticks(range(5), [str(i) for i in range(5)], fontsize=6)
        a2.set_xlabel(f'{fam}: {LABEL.get(knob, knob)} level', fontsize=6.5, labelpad=1)
        a2.tick_params(labelsize=6)
        if k == 0:
            a2.legend(fontsize=6, ncol=2, loc='best')
    fig.text(0.75, 0.97, 'b  own-knob response, mean ± 95% CI over replicates', fontsize=9)
    out.mkdir(parents=True, exist_ok=True)
    save(fig, out, 'selectivity')
    fig.savefig(out / 'selectivity.pdf')
    plt.close(fig)
