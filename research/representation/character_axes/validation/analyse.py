"""Score the validation runs, apply the pre-registered criteria, bound the cohort effects.

    python -m research.representation.character_axes.validation.analyse [--refresh] [--out DIR] [--raw DIR ...]
        (--banks DIR | --no-bound)

Reads the windows from --raw (stage 1, then stage 2), by default $VALIDATION_OUT (run.py's override) or
results/representation/character-axes-validation/raw, and caches per-recording scores in scores.pkl
beside the last directory.
Writes DIR/tables/*.csv and DIR/figures/selectivity.{svg,pdf,png}; DIR defaults to this folder.
On Gadi, jobs/analyse.pbs runs it.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from mtsspi.character_axes import edge_profiles, is_antisymmetric

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RAW = Path(os.environ.get('VALIDATION_OUT', ROOT / 'results' / 'representation' / 'character-axes-validation' / 'raw'))
B_, L_, H_ = 'fmin-0_fmax-0-5', 'fmin-0_fmax-0-25', 'fmin-0-25_fmax-0-5'
mt = lambda f, b: f'{f}_multitaper_mean_fs-1_{b}'
SGC = {B_: 'sgc_nonparametric_mean_fs-1_fmin-1e-05_fmax-0-5', L_: 'sgc_nonparametric_mean_fs-1_fmin-1e-05_fmax-0-25',
       H_: 'sgc_nonparametric_mean_fs-1_fmin-0-25_fmax-0-5'}
GC, TE = 'gc_gaussian_k-1_kt-1_l-1_lt-1', 'te_kraskov_NN-4_DCE-AUTO_k-1_kt-1_l-1_lt-1'
DTWSC, EUCL = 'dtw_constraint-sakoe-chiba_radius-auto', 'pdist_euclidean'

# name: (construct, estimator family, a, b); a is the strength-like member
PAIRS = {
    'form: mi_g|mi_ksgDCE': ('form', 'KSG', 'mi_gaussian', 'mi_kraskov_NN-4_DCE-AUTO'),
    'form: tlmi_g|tlmi_ksgDCE': ('form', 'KSG', 'tlmi_gaussian', 'tlmi_kraskov_NN-4_DCE-AUTO'),
    'form: gc|te_ksgDCE': ('form', 'KSG', GC, TE),
    'form: r2|dcorr': ('form', 'distance', 'cov-sq_EmpiricalCovariance', 'dcorr'),
    'form: mi_g|mi_kernel': ('form', 'kernel', 'mi_gaussian', 'mi_kernel_W-0.25'),
    'lag: coh|icoh 0-0.5': ('lag', 'coherency', mt('cohmag', B_), mt('icoh', B_)),
    'lag: coh|icoh 0-0.25': ('lag', 'coherency', mt('cohmag', L_), mt('icoh', L_)),
    'lag: coh|icoh 0.25-0.5': ('lag', 'coherency', mt('cohmag', H_), mt('icoh', H_)),
    'lag: plv|wpli': ('lag', 'phase locking', mt('plv', B_), mt('wpli', B_)),
    'lag: ppc|dswpli': ('lag', 'phase locking', mt('ppc', B_), mt('dswpli', B_)),
    'lag: pec|pec_orth': ('lag', 'envelope', 'pec', 'pec_orth'),
    'band: coh L|H': ('band', 'coherency', mt('cohmag', L_), mt('cohmag', H_)),
    'band: ppc L|H': ('band', 'phase locking', mt('ppc', L_), mt('ppc', H_)),
    'band: sgc L|H': ('band', 'spectral Granger', SGC[L_], SGC[H_]),
    'mediation: LW': ('mediation', 'shrinkage', 'cov-sq_LedoitWolf', 'prec-sq_LedoitWolf'),
    'mediation: empirical': ('mediation', 'empirical', 'cov-sq_EmpiricalCovariance', 'prec-sq_EmpiricalCovariance'),
    'direction: tlmi_g|gc': ('direction', 'Gaussian', 'tlmi_gaussian', GC),
    'warp: eucl|dtwSC': ('warp', 'alignment', EUCL, DTWSC),
    'warp: dtwSC|dtw': ('warp', 'alignment', DTWSC, 'dtw'),
    'control: ksg|ksgDCE': ('control', 'estimator', 'mi_kraskov_NN-4', 'mi_kraskov_NN-4_DCE-AUTO'),
    'control: r2|mi_g': ('control', 'strength range', 'cov-sq_EmpiricalCovariance', 'mi_gaussian'),
    'control: r|rho': ('control', 'monotone', 'cov_EmpiricalCovariance', 'spearmanr'),
}
SINGLE = {'direction: gc A|At': ('direction', 'Gaussian GC', GC), 'direction: te A|At': ('direction', 'KSG TE', TE),
          'direction: sgc A|At': ('direction', 'spectral Granger', SGC[B_])}
LEVELS = {'form: median(ksgDCE - mi_g)': ('form', 'mi_kraskov_NN-4_DCE-AUTO', 'mi_gaussian', 'diff'),
          'form: median(te - gc)': ('form', TE, GC, 'diff'),
          'lag: median icoh/sqrt(coh)': ('lag', mt('icoh', B_), mt('cohmag', B_), 'sinphi'),
          'band: median high share': ('band', mt('cohmag', H_), mt('cohmag', L_), 'share'),
          'warp: median dtwSC/eucl': ('warp', DTWSC, EUCL, 'ratio'),
          'direction: median gc asymmetry': ('direction', GC, None, 'asym')}
# Post hoc (not pre-registered): the same contrasts over the strongest quartile of edges, because a
# median over all entries reads uncoupled edges when coupling is sparse. Strength member and its sign.
TOPQ = {'form: median(ksgDCE - mi_g)': ('mi_gaussian', 1), 'form: median(te - gc)': (GC, 1),
        'lag: median icoh/sqrt(coh)': (mt('cohmag', B_), 1), 'band: median high share': (mt('cohmag', B_), 1),
        'warp: median dtwSC/eucl': (EUCL, -1), 'direction: median gc asymmetry': (GC, 1)}
TQ = ' [top quartile, post hoc]'
for _k in TOPQ:
    LEVELS[_k + TQ] = LEVELS[_k]
# own knob and predicted sign of the response (score at max level minus zero level)
OWN_ORG = {'form': ('A', 'form_disp', -1), 'lag': ('A', 'lag_disp', -1), 'band': ('A', 'band_disp', -1),
           'warp': ('A', 'warp_disp', -1), 'mediation': ('C', 'med', -1), 'direction': ('B', 'dir_level', -1)}
OWN_LEVEL = {'form': ('A', 'form_level', 1), 'lag': ('A', 'lag_level', 1), 'band': ('A', 'band_level', 1),
             'warp': ('A', 'warp_level', -1), 'direction': ('B', 'dir_level', 1)}
SECOND = {('lag', 'org'): ('K', 'lag_disp', -1), ('lag', 'level'): ('K', 'lag_level', 1),
          ('band', 'level'): ('K', 'band_level', 1)}
CONSTRUCT_KNOBS = {'lag': ('lag_level', 'lag_disp'), 'band': ('band_level', 'band_disp'),
                   'form': ('form_level', 'form_disp'), 'warp': ('warp_level', 'warp_disp')}
# Both knobs of each construct, for the response to the construct's other knob (reported; not in SR)
OWN_KNOBS = {**{c: ('A', k) for c, k in CONSTRUCT_KNOBS.items()}, 'direction': ('B', ('dir_level', 'dir_disp'))}
NUISANCES = ('nuis_cv_s', 'nuis_noise', 'nuis_T', 'nuis_r', 'nuis_beta', 'nuis_mix', 'nuis_M')
DIAGNOSTIC = {'direction: tlmi_g|gc'}       # no pre-registered sign; reported only
# Stage 2 (pre-registered 3 November 2026, README "Stage 2"): second families for mediation and direction,
# and level-aware spread readouts (SD of a per-edge contrast over the strongest quartile of entries).
SPREAD = {'lag: spread icoh/sqrt(coh) [level-aware]': ('lag', mt('icoh', B_), mt('cohmag', B_), 'sinphi'),
          'band: spread high share [level-aware]': ('band', mt('cohmag', H_), mt('cohmag', L_), 'share')}
SPREAD_TOP = mt('cohmag', B_)
OWN_SPREAD = {'lag': ('A', 'lag_disp', 1), 'band': ('A', 'band_disp', 1)}
SECOND[('lag', 'spread')] = ('K', 'lag_disp', 1)
STAGE2_OWN = {'mediation': [('A', 'med')], 'direction': [('A', 'dir_level'), ('K', 'dir_level')]}
STAGE2_KNOBS = {('A', 'med'): 'mediation', ('A', 'dir_level'): 'direction', ('K', 'dir_level'): 'direction'}
GRID2_NUIS = ('noise', 'mix', 'cv_s')


def own_of(construct, kind):
    """(family, knob, predicted sign) of a measure's own knob, or None."""
    return {'org': OWN_ORG, 'level': OWN_LEVEL, 'spread': OWN_SPREAD}[kind].get(construct)


def _prof(S, names, n):
    if n not in names:                      # dcorr is skipped where its pair is not evaluated
        return np.full(S.shape[1] * (S.shape[1] - 1), np.nan)
    P = S[names.index(n)][~np.eye(S.shape[1], dtype=bool)].astype(np.float64)
    return np.abs(P) if is_antisymmetric(n) else P


def _transpose_prof(S, names, n):
    return S[names.index(n)].T[~np.eye(S.shape[1], dtype=bool)].astype(np.float64)


def _corr(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5 or np.std(x[ok]) < 1e-12 or np.std(y[ok]) < 1e-12:
        return np.nan
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def _contrast_vec(a, b, kind):
    with np.errstate(divide='ignore', invalid='ignore'):
        if kind == 'diff':
            c = a - b
        elif kind == 'sinphi':
            c = a / np.sqrt(b)
        elif kind == 'share':
            c = a / (a + b)
        elif kind == 'ratio':
            c = a / b
    return np.where(np.isfinite(c), c, np.nan)


def _contrast(a, b, kind, mask=None):
    c = _contrast_vec(a, b, kind)
    return float(np.nanmedian(c if mask is None else c[mask]))


def _top(s, sign):
    s = sign * s
    return s >= np.nanquantile(s, 0.75)


def recording_scores(win):
    """win: list of (S, names) for the windows of one recording -> dict of scores."""
    out = {}
    need = {s for _, _, a, b in PAIRS.values() for s in (a, b)}
    profs = [{n: _prof(S, names, n) for n in need.union(names)} for S, names in win]  # NaN where absent
    for key, (_, _, a, b) in PAIRS.items():
        z = [_corr(p[a], p[b]) for p in profs]
        out[f'org|{key}'] = np.nanmean(z)
        if len(profs) == 2:
            p0, p1 = profs
            out[f'cross|{key}'] = np.nanmean([_corr(p0[a], p1[b]), _corr(p1[a], p0[b])])
            out[f'rela|{key}'] = _corr(p0[a], p1[a])
            out[f'relb|{key}'] = _corr(p0[b], p1[b])
        out[f'cva|{key}'] = np.nanmean([np.nanstd(p[a]) / np.nanmean(p[a]) for p in profs])
        out[f'sda|{key}'] = np.nanmean([np.nanstd(p[a]) for p in profs])
    for key, (_, _, a) in SINGLE.items():
        out[f'org|{key}'] = np.nanmean([_corr(_prof(S, names, a), _transpose_prof(S, names, a)) for S, names in win])
    for key, (_, a, b, kind) in LEVELS.items():
        top = TOPQ.get(key[:-len(TQ)]) if key.endswith(TQ) else None
        vals = []
        for S, names in win:
            if kind == 'asym':
                A = S[names.index(a)].astype(np.float64)
                off = ~np.eye(len(A), dtype=bool)
                with np.errstate(divide='ignore', invalid='ignore'):
                    c = (np.abs(A - A.T) / (np.abs(A) + np.abs(A.T)))[off]
                mask = _top((np.abs(A) + np.abs(A.T))[off], 1) if top else np.ones(c.shape, bool)
                vals.append(float(np.nanmedian(c[mask])))
            else:
                mask = _top(_prof(S, names, top[0]), top[1]) if top else None
                vals.append(_contrast(_prof(S, names, a), _prof(S, names, b), kind, mask))
        out[f'level|{key}'] = np.nanmean(vals)
    for key, (_, a, b, kind) in SPREAD.items():
        vals = []
        for S, names in win:
            c = _contrast_vec(_prof(S, names, a), _prof(S, names, b), kind)
            vals.append(float(np.nanstd(c[_top(_prof(S, names, SPREAD_TOP), 1)])))
        out[f'spread|{key}'] = np.nanmean(vals)
    return out


def load_scores(refresh=False, raws=(RAW,)):
    cache = Path(raws[-1]).parent / 'scores.pkl'
    files = sorted(f for raw in raws for f in glob.glob(str(Path(raw) / '*.npz')))
    if cache.exists() and not refresh:
        df = pd.read_pickle(cache)
        if len(df) * 2 >= len(files) - 1:
            return df
    groups = {}
    for f in files:
        stem = Path(f).name[:-4]
        surr = stem.endswith('-surr')
        base = stem[:-5] if surr else stem
        rec, w = base.rsplit('-w', 1)
        groups.setdefault(rec + ('-surr' if surr else ''), []).append(f)
    rows = []
    for rec, fs in groups.items():
        win, meta, ac1 = [], None, []
        for f in sorted(fs):
            d = np.load(f)
            win.append((d['S'], [str(s) for s in d['spi']]))
            meta = json.loads(str(d['meta'])); ac1.append(float(d['ac1']))
        meta.pop('win', None)
        rows.append({**meta, 'n_windows': len(win), 'ac1': np.mean(ac1), **recording_scores(win)})
    df = pd.DataFrame(rows)
    df.to_pickle(cache)
    return df


def corrected(df, key, kind):
    """Reliability-corrected organisation, and strength-corrected character dispersion (two toy forms)."""
    z = df[f'cross|{key}'] / np.sqrt(df[f'rela|{key}'].clip(lower=0.05) * df[f'relb|{key}'].clip(lower=0.05))
    if kind == 'rel':
        return z
    zz = z.clip(-0.999, 0.999)
    excess = np.where(zz > 0, zz ** -2.0 - 1, np.nan)
    rel_a = df[f'rela|{key}'].clip(lower=0.05)
    if kind == 'mult':
        cv = np.sqrt(rel_a) * df[f'cva|{key}'].abs()
        return np.sqrt(np.clip(excess, 0, None) / (1 + cv ** -2.0))
    return np.sqrt(rel_a) * df[f'sda|{key}'] * np.sqrt(np.clip(excess, 0, None))


def response(df, fam, knob, col, baseline='B0', grid='', surrogate=False, lo=0, hi=4):
    """Paired per-replicate change between knob levels: mean, 95% t-interval, paired d, n."""
    sel = df[(df.family == fam) & (df.knob == knob) & (df.baseline == baseline) & (df.grid == grid)
             & (df.surrogate == surrogate)]
    if callable(col):
        vals = col(sel)
    else:
        vals = sel[col] if col in sel else pd.Series(np.nan, index=sel.index)
    a = pd.Series(np.asarray(vals, float), index=sel.index)
    x0 = a[sel.level == lo].groupby(sel.loc[sel.level == lo, 'rep']).mean()
    x1 = a[sel.level == hi].groupby(sel.loc[sel.level == hi, 'rep']).mean()
    d = (x1 - x0).dropna()
    n = len(d)
    if n < 2:
        return dict(mean=np.nan, lo=np.nan, hi=np.nan, d=np.nan, n=n)
    m, s = d.mean(), d.std(ddof=1)
    t = stats.t.ppf(0.975, n - 1) * s / np.sqrt(n)
    return dict(mean=m, lo=m - t, hi=m + t, d=m / s if s > 0 else np.nan, n=n)


def _status(r, sign):
    if not np.isfinite(r['mean']):
        return 'missing'
    lo, hi = r['lo'], r['hi']
    if (sign > 0 and lo > 0) or (sign < 0 and hi < 0):
        return 'ok'
    if (sign > 0 and hi < 0) or (sign < 0 and lo > 0):
        return 'reversed'
    return 'insensitive'


def measures():
    """(measure id, construct, family label, kind, column)."""
    m = []
    for k, (c, fam, _, _) in PAIRS.items():
        m.append((k, c, fam, 'org', f'org|{k}'))
    for k, (c, fam, _) in SINGLE.items():
        m.append((k, c, fam, 'org', f'org|{k}'))
    for k, (c, _, _, _) in LEVELS.items():
        m.append((k, c, 'level', 'level', f'level|{k}'))
    for k, (c, _, _, _) in SPREAD.items():
        m.append((k, c, 'spread', 'spread', f'spread|{k}'))
    return m


def evaluate(df):
    grid_tags = sorted(g for g in df.grid.unique() if g)
    rows, resp = [], []
    for mid, construct, fam, kind, col in measures():
        own = own_of(construct, kind)
        rec = dict(measure=mid, construct=construct, method=fam, kind=kind)
        if construct == 'control' or own is None:
            rows.append(dict(rec, verdict='control' if construct == 'control' else 'no own knob')); continue
        ofam, oknob, sign = own
        if mid in DIAGNOSTIC:
            sign = 0
        main = {bl: response(df, ofam, oknob, col, bl) for bl in (('B0', 'B1') if ofam == 'A' else ('B0',))}
        for bl, r in main.items():
            resp.append(dict(measure=mid, knob=oknob, family=ofam, baseline=bl, grid='', **r))
        status = {f'main {bl}': _status(r, sign) if sign else 'diagnostic' for bl, r in main.items()}
        if ofam == 'A':
            for g in grid_tags:
                r = response(df, 'A', oknob, col, 'B0', g)
                resp.append(dict(measure=mid, knob=oknob, family='A', baseline='B0', grid=g, **r))
                status[f'grid {g}'] = _status(r, sign) if sign else 'diagnostic'
        sec = SECOND.get((construct, kind))
        if sec:
            r = response(df, sec[0], sec[1], col)
            resp.append(dict(measure=mid, knob=sec[1], family=sec[0], baseline='B0', grid='', **r))
            status[f'second {sec[0]}'] = _status(r, sec[2]) if sign else 'diagnostic'
        own_r = main['B0']
        sibling = {bl: np.nan for bl in main}
        fam2, knobs2 = OWN_KNOBS.get(construct, (None, ()))
        for k2 in knobs2:
            if k2 != oknob:
                for bl in main:
                    r = response(df, fam2, k2, col, bl)
                    if r['n'] or bl == 'B0':
                        resp.append(dict(measure=mid, knob=k2, family=fam2, baseline=bl, grid='', **r))
                    sibling[bl] = r['mean'] / abs(main[bl]['mean']) if main[bl]['mean'] else np.nan
        stage2 = {}                         # stage-2 knobs: reported for every measure
        for (f3, k3), c3 in STAGE2_KNOBS.items():
            r = response(df, f3, k3, col)
            if r['n']:
                resp.append(dict(measure=mid, knob=k3, family=f3, baseline='B0', grid='', **r))
                if f3 == 'A' and c3 != construct and np.isfinite(r['mean']):
                    stage2[k3] = abs(r['mean'])
        others = []
        for c2, knobs in CONSTRUCT_KNOBS.items():
            if c2 == construct:
                continue
            for k2 in knobs:
                r = response(df, 'A', k2, col)
                others.append((k2, abs(r['mean'])))
                resp.append(dict(measure=mid, knob=k2, family='A', baseline='B0', grid='', **r))
        nuis = {}
        for nk in NUISANCES:
            for bl in ('B0', 'B1'):
                r = response(df, 'A', nk, col, bl)
                resp.append(dict(measure=mid, knob=nk, family='A', baseline=bl, grid='', **r))
                if bl == 'B0':
                    nuis[nk] = r['mean'] / abs(own_r['mean']) if own_r['mean'] else np.nan
        worst_other = max(others, key=lambda t: t[1] if np.isfinite(t[1]) else -1)
        sr = abs(own_r['mean']) / worst_other[1] if worst_other[1] > 0 else np.inf
        reversed_at = [k for k, v in status.items() if v == 'reversed']
        insensitive_at = [k for k, v in status.items() if v == 'insensitive']
        main_ok = all(status[f'main {bl}'] == 'ok' for bl in main)
        if not sign:
            verdict = 'diagnostic'
        elif reversed_at:
            verdict = 'drop (sign reversal)'
        elif not main_ok:
            verdict = 'drop (insensitive)'
        else:
            verdict = 'core' if sr >= 2 else 'sub-axis'
        extra = {'other own knob/own': sibling['B0']}
        if 'B1' in sibling:
            extra['other own knob/own B1'] = sibling['B1']
        if kind == 'spread':                # pre-registered level-awareness: |level knob| <= 0.5 |own| at B0 and B1
            extra['level_aware'] = bool(all(np.isfinite(v) and abs(v) <= 0.5 for v in sibling.values()))
        if stage2:
            extra['SR incl. stage-2 knobs'] = abs(own_r['mean']) / max(worst_other[1], *stage2.values())
        rows.append(dict(rec, own_knob=f'{ofam}:{oknob}', own_mean=own_r['mean'], own_d=own_r['d'],
                         SR=sr, worst_other=worst_other[0], **extra, reversed_at='; '.join(reversed_at),
                         insensitive_at='; '.join(insensitive_at), verdict=verdict,
                         **{f'nuis/own {k[5:]}': v for k, v in nuis.items()}))
    return pd.DataFrame(rows), pd.DataFrame(resp)


def surrogate_table(df):
    rows = []
    for mid, construct, fam, kind, col in measures():
        if construct not in ('form', 'warp', 'lag'):
            continue
        own = own_of(construct, kind)
        if not own:
            continue
        for bl in ('B0', 'B1'):
            real = response(df, 'A', own[1], col, bl)
            surr = response(df, 'A', own[1], col, bl, surrogate=True)
            if not np.isfinite(surr['mean']):
                continue
            rows.append(dict(measure=mid, construct=construct, kind=kind, knob=own[1], baseline=bl,
                             real=real['mean'], surrogate=surr['mean'],
                             kept=surr['mean'] / real['mean'] if real['mean'] else np.nan))
    return pd.DataFrame(rows)


def posthoc_table(df):
    """Post hoc: form and warp measures at stronger coupling (baseline PHS), real and surrogate."""
    rows = []
    for mid, construct, fam, kind, col in measures():
        if construct not in ('form', 'warp'):
            continue
        own = own_of(construct, kind)
        if not own:
            continue
        for k2 in CONSTRUCT_KNOBS[construct]:
            real = response(df, 'A', k2, col, 'PHS')
            surr = response(df, 'A', k2, col, 'PHS', surrogate=True)
            if not np.isfinite(real['mean']):
                continue
            target = k2 == own[1]
            rows.append(dict(measure=mid, kind=kind, knob=k2, own=target, real=real['mean'], lo=real['lo'],
                             hi=real['hi'], d=real['d'], status=_status(real, own[2]) if target else '',
                             surrogate=surr['mean'],
                             kept=surr['mean'] / real['mean'] if real['mean'] else np.nan))
    return pd.DataFrame(rows)


def correction_table(df):
    """Criterion 6 for every organisation pair with an own knob in family A."""
    rows = []
    for key, (construct, fam, _, _) in PAIRS.items():
        own = OWN_ORG.get(construct)
        if not own or own[0] != 'A':
            continue
        for kind in ('raw', 'rel', 'mult', 'add'):
            col = f'org|{key}' if kind == 'raw' else (lambda s, k=key, kd=kind: corrected(s, k, kd))
            o = response(df, 'A', own[1], col)
            sign = own[2] if kind in ('raw', 'rel') else -own[2]
            rec = dict(pair=key, correction=kind, own_mean=o['mean'], own_d=o['d'] * sign)
            for nk in ('nuis_cv_s', 'nuis_noise', 'nuis_r'):
                r = response(df, 'A', nk, col)
                rec[f'{nk[5:]}/own'] = abs(r['mean']) / abs(o['mean']) if o['mean'] else np.nan
            rows.append(rec)
    t = pd.DataFrame(rows)
    use = []
    for key in t.pair.unique():
        s = t[t.pair == key].set_index('correction')
        raw = s.loc['raw']
        for kind in ('rel', 'mult', 'add'):
            c = s.loc[kind]
            cut = all(c[f'{n}/own'] <= 0.5 * raw[f'{n}/own'] for n in ('cv_s', 'noise'))
            keep = c['own_d'] >= 0.75 * raw['own_d'] and c['own_d'] > 0
            use.append(dict(pair=key, correction=kind, usable=bool(cut and keep)))
    return t.merge(pd.DataFrame(use), on=['pair', 'correction'], how='left')


def _cell_mean(df, fam, knob, col, level, grid=''):
    s = df[(df.family == fam) & (df.knob == knob) & (df.baseline == 'B0') & (df.grid == grid) & ~df.surrogate
           & (df.level == level)]
    return s[col].mean() if len(s) else np.nan


def second_family_table(df, verdicts):
    """Stage 2: mediation and direction measures in their second families, with the pre-registered verdict."""
    ver = verdicts.set_index('measure')
    rows = []
    for mid, construct, _, kind, col in measures():
        own = own_of(construct, kind) if construct in STAGE2_OWN else None
        if own is None or mid in DIAGNOSTIC:
            continue
        sign = own[2]
        for fam2, knob2 in STAGE2_OWN[construct]:
            main = {bl: response(df, fam2, knob2, col, bl) for bl in ('B0', 'B1')}
            if not main['B0']['n']:
                continue
            status = {f'main {bl}': _status(r, sign) for bl, r in main.items()}
            grids = sorted(g for g in df.loc[(df.family == fam2) & (df.knob == knob2), 'grid'].unique() if g)
            for g in grids:
                status[f'grid {g}'] = _status(response(df, fam2, knob2, col, 'B0', g), sign)
            rec = dict(measure=mid, kind=kind, family=fam2, knob=knob2, stage1=ver.loc[mid, 'verdict'],
                       own_B0=main['B0']['mean'], d_B0=main['B0']['d'], own_B1=main['B1']['mean'], d_B1=main['B1']['d'],
                       main_ok=all(status[f'main {bl}'] == 'ok' for bl in main),
                       reversed_at='; '.join(k for k, v in status.items() if v == 'reversed'),
                       insensitive_at='; '.join(k for k, v in status.items() if v == 'insensitive'))
            if fam2 == 'A':                 # SR against every other construct knob of family A
                ks = [k for c2, kk in CONSTRUCT_KNOBS.items() for k in kk]
                ks += [k for (f3, k), c3 in STAGE2_KNOBS.items() if f3 == 'A' and c3 != construct]
                others = {k: abs(response(df, 'A', k, col)['mean']) for k in ks}
                others = {k: v for k, v in others.items() if np.isfinite(v)}
                worst = max(others, key=others.get) if others else ''
                rec.update(SR=abs(main['B0']['mean']) / others[worst] if worst else np.nan, worst_other=worst)
                for nuis in GRID2_NUIS:     # unpaired: knob level 0 at the nuisance's two extremes
                    ends = {t: next((g for g in grids if g.startswith(f'{nuis}:{t}:')), '') for t in ('lo', 'hi')}
                    v = _cell_mean(df, 'A', knob2, col, 0, ends['hi']) - _cell_mean(df, 'A', knob2, col, 0, ends['lo'])
                    rec[f'nuis/own {nuis}'] = v / abs(main['B0']['mean']) if main['B0']['mean'] else np.nan
            rows.append(rec)
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    verdict = {}
    for mid, s in t.groupby('measure', sort=False):
        stage1, a = s.stage1.iloc[0], s[s.family == 'A']
        if stage1.startswith('drop'):
            verdict[mid] = stage1
        elif (s.reversed_at != '').any():
            verdict[mid] = 'drop (sign reversal in a second family)'
        elif a.empty:
            verdict[mid] = f'{stage1} (second family not run)'
        elif a.main_ok.iloc[0]:
            verdict[mid] = 'core, two generators' if stage1 == 'core' and a.SR.iloc[0] >= 2 else 'sub-axis, two generators'
        else:
            verdict[mid] = f'{stage1}, one generator (insensitive in A)'
    return t.assign(verdict=t.measure.map(verdict))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--refresh', action='store_true')
    ap.add_argument('--out', type=Path, default=HERE, help='writes OUT/tables and OUT/figures')
    ap.add_argument('--banks', type=Path, help='DIR/<cohort>/bank.npz for the cohort bound')
    ap.add_argument('--no-bound', action='store_true')
    ap.add_argument('--raw', type=Path, nargs='+', default=[RAW],
                    help='window directories (stage 1, then stage 2); scores.pkl is cached beside the last')
    a = ap.parse_args()
    if a.banks is None and not a.no_bound:
        ap.error('the cohort bound needs --banks DIR (or pass --no-bound)')
    tables = a.out / 'tables'
    df = load_scores(a.refresh, a.raw)
    print(df.groupby(['family', 'surrogate', df.grid != '']).size())
    tables.mkdir(parents=True, exist_ok=True)
    verdicts, resp = evaluate(df)
    verdicts.to_csv(tables / 'verdicts.csv', index=False, float_format='%.4g')
    resp.to_csv(tables / 'responses.csv', index=False, float_format='%.4g')
    surrogate_table(df).to_csv(tables / 'surrogates.csv', index=False, float_format='%.4g')
    correction_table(df).to_csv(tables / 'corrections.csv', index=False, float_format='%.4g')
    if (df.baseline == 'PHS').any():
        posthoc_table(df).to_csv(tables / 'posthoc_strong_coupling.csv', index=False, float_format='%.4g')
    second = second_family_table(df, verdicts)
    pd.set_option('display.width', 250, 'display.max_columns', 30, 'display.max_colwidth', 60)
    if not second.empty:
        second.to_csv(tables / 'second_family.csv', index=False, float_format='%.4g')
        print(second[['measure', 'family', 'own_B0', 'own_B1', 'reversed_at', 'insensitive_at', 'verdict']
                     + [c for c in ('SR', 'worst_other') if c in second]].to_string(index=False))
    print(verdicts[['measure', 'kind', 'own_knob', 'own_mean', 'own_d', 'SR', 'worst_other', 'verdict',
                    'reversed_at']].to_string(index=False))
    from research.representation.character_axes.validation.figure import make
    make(df, verdicts, resp, a.out / 'figures', second)
    if not a.no_bound:
        from research.representation.character_axes.validation.bound import run_bound
        run_bound(df, verdicts, a.banks, tables / 'bound.csv')


if __name__ == '__main__':
    main()
