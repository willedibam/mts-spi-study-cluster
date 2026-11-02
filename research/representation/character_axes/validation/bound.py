"""Criterion 7: bound cohort organisation effects by measured nuisances.

For each cohort contrast and each organisation pair that passed criteria 1-2, per cell (unit x state):
raw within-window z; cross-window z over window pairs; profile reliabilities; reliability-corrected z;
strength-corrected dispersion where criterion 6 marked the correction usable; channel lag-1
autocorrelation. Effects are paired d_z (within-unit designs) or between-group d (participant means).
The autocorrelation-explained change is the simulated slope of organisation on lag-1 autocorrelation
(family A, r sweep, B0) times the measured cohort change in autocorrelation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mtsspi.character_axes import edge_profiles, is_antisymmetric

CONTRAST = {'neurotycho': (0, 1), 'chbmit': (0, 1), 'hmc': (0, 1), 'ad-ftd': (0, 1), 'lemon': (0, 1),
            'propofol': (0, 2)}
PAIRED = {'neurotycho', 'chbmit', 'hmc', 'propofol'}
CLIP = 0.05


def _std(X):
    X = X - X.mean(-1, keepdims=True)
    n = np.linalg.norm(X, axis=-1, keepdims=True)
    return X / np.where(n < 1e-12, np.nan, n)


def _cell_stats(A, B):
    """A, B: (windows, entries) profiles of one cell -> raw, cross, rel_a, rel_b."""
    A, B = _std(A), _std(B)
    CAB, CAA, CBB = A @ B.T, A @ A.T, B @ B.T
    off = ~np.eye(len(A), dtype=bool)
    return (np.nanmean(np.diag(CAB)), np.nanmean(CAB[off]), np.nanmean(CAA[off]), np.nanmean(CBB[off]))


def _effect(v, units, states, c0, c1, paired):
    v = np.asarray(v, float)
    if paired:
        diffs = []
        for u in np.unique(units):
            a, b = v[(units == u) & (states == c0)], v[(units == u) & (states == c1)]
            if len(a) and len(b) and np.isfinite(a[0]) and np.isfinite(b[0]):
                diffs.append(b[0] - a[0])
        diffs = np.array(diffs)
        if len(diffs) < 3:
            return np.nan, np.nan
        return diffs.mean() / diffs.std(ddof=1), diffs.mean()
    a, b = v[states == c0], v[states == c1]
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    return (b.mean() - a.mean()) / np.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1))), b.mean() - a.mean()


def ac_slope(df, key):
    """Simulated d(organisation)/d(lag-1 autocorrelation), family A r sweep at B0."""
    s = df[(df.family == 'A') & (df.knob == 'nuis_r') & (df.baseline == 'B0') & (df.grid == '') & ~df.surrogate]
    y, x = s[f'org|{key}'].to_numpy(float), s['ac1'].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    return np.polyfit(x[ok], y[ok], 1)[0] if ok.sum() > 3 else np.nan


def run_bound(df, verdicts, banks: Path, out: Path):
    from research.representation.character_axes.validation.analyse import PAIRS
    corr = pd.read_csv(out.parent / 'corrections.csv')
    usable = {(r.pair, r.correction) for r in corr.itertuples() if r.usable}
    keep = [m for m in verdicts.loc[verdicts.verdict.isin(['core', 'sub-axis']), 'measure'] if m in PAIRS]
    rows = []
    for name, (c0, c1) in CONTRAST.items():
        f = banks / name / 'bank.npz'
        if not f.exists():
            print('missing bank', f); continue
        b = np.load(f, allow_pickle=False)
        spi = [str(s) for s in b['spi_order']]
        M = int(b['M'][0])
        P = edge_profiles(b['aligned'], M, len(spi))
        y = b['y']
        rid = b['row_id'].astype(str)
        units = np.array([r.split('--')[0] for r in rid]) if name == 'neurotycho' else b['group'].astype(str)
        x = b['x']
        ac = np.array([np.mean([np.corrcoef(w[:-1, j], w[1:, j])[0, 1] for j in range(w.shape[1])]) for w in x])
        cells = sorted(set(zip(units, y)))
        cu = np.array([c[0] for c in cells]); cs = np.array([c[1] for c in cells])
        cell_ac = np.array([ac[(units == u) & (y == s)].mean() for u, s in cells])
        d_ac, m_ac = _effect(cell_ac, cu, cs, c0, c1, name in PAIRED)
        for key in keep:
            _, _, a, bb = PAIRS[key]
            if a not in spi or bb not in spi:
                continue
            A = P[:, spi.index(a)]; B = P[:, spi.index(bb)]
            if is_antisymmetric(a): A = np.abs(A)
            if is_antisymmetric(bb): B = np.abs(B)
            st = np.array([_cell_stats(A[(units == u) & (y == s)], B[(units == u) & (y == s)]) for u, s in cells])
            raw, cross, ra, rb = st.T
            rel = cross / np.sqrt(np.clip(ra, CLIP, None) * np.clip(rb, CLIP, None))
            rel_lowfloor = cross / np.sqrt(np.clip(ra, 1e-3, None) * np.clip(rb, 1e-3, None))
            cv = np.array([np.mean(np.nanstd(A[(units == u) & (y == s)], 1) / np.nanmean(A[(units == u) & (y == s)], 1))
                           for u, s in cells])
            sd = np.array([np.mean(np.nanstd(A[(units == u) & (y == s)], 1)) for u, s in cells])
            zz = np.clip(rel, -0.999, 0.999)
            excess = np.where(zz > 0, zz ** -2.0 - 1, np.nan)
            mult = np.sqrt(np.clip(excess, 0, None) / (1 + (np.sqrt(np.clip(ra, CLIP, None)) * np.abs(cv)) ** -2.0))
            add = np.sqrt(np.clip(ra, CLIP, None)) * sd * np.sqrt(np.clip(excess, 0, None))
            paired = name in PAIRED
            d_raw, m_raw = _effect(raw, cu, cs, c0, c1, paired)
            d_rel, _ = _effect(rel, cu, cs, c0, c1, paired)
            d_rel_low, _ = _effect(rel_lowfloor, cu, cs, c0, c1, paired)
            d_mult = _effect(mult, cu, cs, c0, c1, paired)[0] if (key, 'mult') in usable else np.nan
            d_add = _effect(add, cu, cs, c0, c1, paired)[0] if (key, 'add') in usable else np.nan
            d_ra, _ = _effect(ra, cu, cs, c0, c1, paired)
            d_rb, _ = _effect(rb, cu, cs, c0, c1, paired)
            pred_ac = ac_slope(df, key) * m_ac
            corrected = [d for d in (d_rel, -d_mult, -d_add) if np.isfinite(d)]
            stands = (np.isfinite(d_raw) and abs(d_raw) >= 0.5 and all(np.sign(d) == np.sign(d_raw) and abs(d) >= 0.5
                      for d in corrected) and abs(pred_ac) < abs(m_raw))
            rows.append(dict(cohort=name, contrast=f'{c0}->{c1}', pair=key, cells=len(cells),
                             mean_raw=np.nanmean(raw), d_raw=d_raw, delta_raw=m_raw, d_rel_corrected=d_rel, d_rel_corrected_floor_0001=d_rel_low,
                             d_strength_corrected_mult=d_mult, d_strength_corrected_add=d_add,
                             d_rel_a=d_ra, d_rel_b=d_rb, d_ac1=d_ac, delta_ac1=m_ac,
                             ac_explained_delta=pred_ac, ac_same_sign=bool(np.sign(pred_ac) == np.sign(m_raw)),
                             claim_stands=bool(stands)))
        print('bound done', name)
    t = pd.DataFrame(rows)
    t.to_csv(out, index=False, float_format='%.4g')
    with pd.option_context('display.width', 250, 'display.max_columns', 20):
        print(t[['cohort', 'pair', 'd_raw', 'd_rel_corrected', 'd_strength_corrected_mult', 'd_strength_corrected_add',
                 'd_rel_corrected_floor_0001', 'd_rel_a', 'd_rel_b', 'delta_raw', 'ac_explained_delta', 'ac_same_sign', 'claim_stands']].to_string(index=False))
    return t
