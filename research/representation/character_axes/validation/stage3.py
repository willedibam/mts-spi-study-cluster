"""Stage 3 of the construction validation (README, "Stage 3"): spectral location within the low band and
joint-burst coupling, scored on the stage-3 corpus (its own pyspi subset, spis_stage3.yaml).

    python -m research.representation.character_axes.validation.stage3 --raw DIR [--out DIR] [--refresh]

Writes OUT/tables/stage3_{verdicts,responses,core}.csv; per-recording scores are cached in DIR/../scores3.pkl.
On Gadi, jobs/analyse3.pbs runs it. The extended readouts are scored with the cohort functions
(mtsspi.character_axes.axis_scores, level_scores and core_scores) on the same ordered off-diagonal entries.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mtsspi.character_axes import AXES, BAND_RANK, LEVELS, axis_scores, core_scores, level_scores
from research.representation.character_axes.validation.analyse import _corr, _status, response

HERE = Path(__file__).resolve().parent
AX = {a.name: a for a in AXES}
COV = 'cov_EmpiricalCovariance'
TAU = lambda k: f'corr_pearson_tau-{k}'

# id: (construct, kind, own knob, sign, how). sign +1/-1 is predicted; 0 is learned from family A at B0.
MEASURES = {
    'location: lag-1 ratio [top quartile]': ('location', 'level', 'loc_level', -1, ('lag1',)),
    'location: AR(2) frequency [top quartile]': ('location', 'level', 'loc_level', 1, ('ar2',)),
    'location: AR(2) frequency spread [level-aware]': ('location', 'spread', 'loc_disp', 1, ('ar2_sd',)),
    'location: spectral-peak level [top quartile]': ('location', 'level', 'loc_level', 0, ('topq', 'spectral_peak')),
    'location: spectral-peak level [all entries]': ('location', 'level', 'loc_level', 0, ('level', 'spectral_peak')),
    'location: best-shift level [top quartile]': ('location', 'level', 'loc_level', -1, ('topq', 'timing_best_shift')),
    'location: best-shift level [all entries]': ('location', 'level', 'loc_level', -1, ('level', 'timing_best_shift')),
    'location: fixed-lag axis': ('location', 'org', 'loc_disp', 0, ('axis', 'timing_fixed_lag')),
    'location: spectral-peak axis': ('location', 'org', 'loc_disp', 0, ('axis', 'spectral_peak')),
    'location: best-shift axis': ('location', 'org', 'loc_disp', 0, ('axis', 'timing_best_shift')),
    'burst: linear-rank level [top quartile]': ('burst', 'level', 'burst_level', -1, ('topq', 'form_monotone')),
    'burst: linear-rank level [all entries]': ('burst', 'level', 'burst_level', -1, ('level', 'form_monotone')),
    'burst: linear-rank axis': ('burst', 'org', 'burst_disp', -1, ('axis', 'form_monotone')),
    'burst: r|spearman': ('burst', 'org', 'burst_disp', -1, ('pair', COV, 'spearmanr')),
    'burst: r|kendall': ('burst', 'org', 'burst_disp', -1, ('pair', COV, 'kendalltau')),
    'burst: tail level [top quartile]': ('burst', 'level', 'burst_level', 0, ('topq', 'tail')),
    'burst: tail level [all entries]': ('burst', 'level', 'burst_level', 0, ('level', 'tail')),
    'burst: tail axis': ('burst', 'org', 'burst_disp', 0, ('axis', 'tail')),
}
# Core readouts (b56d908), reported for cross-talk: stage-1 own knob in family A and its predicted sign
CORE_OWN = {'band': ('band_disp', -1), 'band_high_share': ('band_level', 1), 'band_spread': ('band_disp', 1),
            'mediation_ledoit_wolf': ('med', -1), 'mediation_empirical': ('med', -1), 'lag': ('lag_disp', -1),
            'lag_sin_phi': ('lag_level', 1), 'form_te_minus_gc': ('form_level', 1)}
KNOBS = {'lag': ('lag_level', 'lag_disp'), 'band': ('band_level', 'band_disp'), 'form': ('form_level', 'form_disp'),
         'warp': ('warp_level', 'warp_disp'), 'location': ('loc_level', 'loc_disp'),
         'burst': ('burst_level', 'burst_disp'), 'direction': ('dir_level',), 'mediation': ('med',)}
# SR3 denominator: every other construct knob of family A plus the construct's named nuisance. Location
# leaves out the band knobs, which move the same variable (coupling frequency) over a wider range.
EXCLUDE = {'location': ('band',), 'burst': ()}
NAMED_NUISANCE = {'location': 'nuis_r', 'burst': 'nuis_beta'}
NUISANCES = ('nuis_cv_s', 'nuis_noise', 'nuis_r', 'nuis_beta', 'nuis_mix')
CORE_BAND = ('core: band_high_share', 'core: band_spread', 'core: band')
BASELINES = ('B0', 'B1')


def _contrast(x, y, kind):
    """The level contrasts of mtsspi.character_axes.level_scores, per entry."""
    with np.errstate(invalid='ignore', divide='ignore'):
        c = {'difference': lambda: x - y, 'abs_difference': lambda: np.abs(x) - np.abs(y),
             'ratio': lambda: np.abs(x) / np.abs(y), 'share': lambda: np.abs(x) / (np.abs(x) + np.abs(y))}[kind]()
    return np.where(np.isfinite(c), c, np.nan)


def ar2_frequency(s0, s1, s2, s3):
    """Per entry, the frequency (cycles per sample) of the recursion s_k = a1 s_(k-1) + a2 s_(k-2) fitted at
    k = 2, 3 to the lag-0 to lag-3 correlations; NaN where the fit has no complex pole pair."""
    with np.errstate(invalid='ignore', divide='ignore'):
        det = s1 * s1 - s0 * s2
        a1, a2 = (s1 * s2 - s0 * s3) / det, (s1 * s3 - s2 * s2) / det
        c = a1 / (2 * np.sqrt(-a2))
        f = np.arccos(c) / (2 * np.pi)
    return np.where((a2 < 0) & (np.abs(c) <= 1) & np.isfinite(f), f, np.nan)


def window_scores(S, names):
    E = ~np.eye(S.shape[1], dtype=bool)
    P = np.stack([S[k][E] for k in range(len(names))]).astype(np.float64)
    ix = {n: k for k, n in enumerate(names)}
    p = lambda n: P[ix[n]]
    rank = p(BAND_RANK)
    top = rank >= np.nanquantile(rank, .75)
    out = {}
    fh = ar2_frequency(p(COV), p(TAU(1)), p(TAU(2)), p(TAU(3)))[top]
    out['diag: AR(2) valid fraction'] = float(np.isfinite(fh).mean())
    for mid, (*_, how) in MEASURES.items():
        if how[0] == 'axis':
            v = axis_scores(P[None], names, axes=[AX[how[1]]])[0][0, 0]
        elif how[0] == 'level':
            v = level_scores(P[None], names, levels={how[1]: LEVELS[how[1]]})[0][0, 0]
        elif how[0] == 'topq':
            a, b, kind = LEVELS[how[1]]
            v = np.nanmedian(_contrast(p(a), p(b), kind)[top])
        elif how[0] == 'pair':
            v = _corr(p(how[1]), p(how[2]))
        elif how[0] == 'lag1':
            with np.errstate(invalid='ignore', divide='ignore'):
                v = np.nanmedian(np.where(np.isfinite(p(TAU(1)) / p(COV)), p(TAU(1)) / p(COV), np.nan)[top])
        else:
            ok = np.isfinite(fh).sum() >= 3
            v = (np.nanmedian(fh) if how[0] == 'ar2' else np.nanstd(fh)) if ok else np.nan
        out[mid] = float(v)
    core, cnames = core_scores(P[None], names)
    out.update({f'core: {n}': float(core[0, k]) for k, n in enumerate(cnames)})
    return out


def load(raw, refresh=False):
    cache = Path(raw).parent / 'scores3.pkl'
    files = sorted(glob.glob(str(Path(raw) / '*.npz')))
    if cache.exists() and not refresh:
        df = pd.read_pickle(cache)
        if len(df) * 2 >= len(files) - 1:
            return df
    groups = {}
    for f in files:
        stem = Path(f).name[:-4]
        kind = next((k for k in ('-surr', '-iaaft') if stem.endswith(k)), '')
        groups.setdefault(stem[:len(stem) - len(kind)].rsplit('-w', 1)[0] + kind, []).append(f)
    rows, versions, errors = [], set(), 0
    for fs in groups.values():
        scores, ac1 = [], []
        for f in sorted(fs):
            with np.load(f) as d:
                meta = json.loads(str(d['meta']))
                versions.add(str(d['pyspi']) if 'pyspi' in d.files else 'unrecorded')
                errors += bool(json.loads(str(d['errors'])))
                scores.append(window_scores(d['S'], [str(s) for s in d['spi']]))
                ac1.append(float(d['ac1']))
        meta.pop('win', None)
        meta['surrogate'] = {False: '', True: 'ft'}.get(meta['surrogate'], meta['surrogate'])
        rows.append({**meta, 'n_windows': len(fs), 'ac1': np.mean(ac1), **pd.DataFrame(scores).mean().to_dict()})
    print(f'{len(files)} windows, {len(rows)} recordings, {errors} windows with pyspi errors, pyspi {sorted(versions)}')
    df = pd.DataFrame(rows)
    df.to_pickle(cache)
    return df


def with_residuals(df):
    """Append real minus IAAFT per recording (surrogate 'resid'), as the cohort residual is computed."""
    key = ['family', 'knob', 'level', 'value', 'baseline', 'grid', 'rep']
    cols = list(MEASURES) + [c for c in df if c.startswith('core: ')]
    real = df[df.surrogate == ''].set_index(key)[cols]
    ia = df[df.surrogate == 'iaaft'].set_index(key)[cols]
    common = real.index.intersection(ia.index)
    res = (real.loc[common] - ia.loc[common]).reset_index().assign(surrogate='resid')
    return pd.concat([df, res], ignore_index=True)


class Responses:
    """Paired knob responses (analyse.response), memoised and logged once each."""

    def __init__(self, df):
        self.df, self.memo = df, {}

    def __call__(self, col, fam, knob, bl='B0', grid='', surr=''):
        k = (col, fam, knob, bl, grid, surr)
        if k not in self.memo:
            self.memo[k] = response(self.df, fam, knob, col, bl, grid, surrogate=surr)
        return self.memo[k]

    def table(self):
        return pd.DataFrame([dict(measure=k[0], family=k[1], knob=k[2], baseline=k[3], grid=k[4], surrogate=k[5], **r)
                             for k, r in self.memo.items() if r['n']])


def _ratio(a, b):
    return a / abs(b) if np.isfinite(a) and np.isfinite(b) and b else np.nan


def evaluate(df, R):
    grids = sorted(g for g in df.grid.unique() if g)
    rows = []
    for mid, (construct, kind, own, sign, _) in MEASURES.items():
        main = {bl: R(mid, 'A', own, bl) for bl in BASELINES}
        source = 'predicted' if sign else 'learned'
        if not sign:                        # learned: the sign of an interval excluding zero at A, B0
            b0 = main['B0']
            sign = int(np.sign(b0['mean'])) if np.isfinite(b0['mean']) and (b0['lo'] > 0 or b0['hi'] < 0) else 0
        st = lambda r: _status(r, sign) if sign else 'insensitive'
        status = {f'main {bl}': st(r) for bl, r in main.items()}
        for g in grids:
            status[f'grid {g}'] = st(R(mid, 'A', own, 'B0', g))
        rec = dict(measure=mid, construct=construct, kind=kind, own_knob=own, sign=sign, sign_source=source,
                   own_B0=main['B0']['mean'], d_B0=main['B0']['d'], own_B1=main['B1']['mean'], d_B1=main['B1']['d'])
        req = {}
        if construct == 'location':
            if kind == 'level':             # second families for the coupling frequency (criterion 1)
                for bl in BASELINES:
                    status[f'K {bl}'] = st(R(mid, 'K', 'loc_level', bl))
                    status[f'D {bl}'] = st(R(mid, 'D', 'fc_level', bl))
            selective = []
            for bl in BASELINES:            # family D: coupling frequency against channel-spectrum frequency
                fc, fp = R(mid, 'D', 'fc_level', bl), R(mid, 'D', 'fp_level', bl)
                rec[f'D fc {bl}'], rec[f'D fp/fc {bl}'] = fc['mean'], _ratio(fp['mean'], fc['mean'])
                s_d = sign if kind == 'level' else np.sign(fc['mean'])
                selective.append(_status(fc, s_d) == 'ok' and abs(fp['mean']) <= 0.5 * abs(fc['mean']))
            rec['reads coupling frequency'] = all(selective)
            if kind == 'level':
                req['reads coupling frequency'] = all(selective)
            core_d = {bl: np.nanmax([abs(R(c, 'A', own, bl)['d']) for c in CORE_BAND]) for bl in BASELINES}
            rec['max core band |d| B0'], rec['max core band |d| B1'] = core_d['B0'], core_d['B1']
            req['incremental over core'] = all(abs(main[bl]['d']) > core_d[bl] for bl in BASELINES)
            for k in KNOBS['band']:         # same variable over a wider range: reported, not in SR3
                rec[f'{k}/own'] = _ratio(R(mid, 'A', k)['mean'], main['B0']['mean'])
        other = next(k for k in KNOBS[construct] if k != own)
        for bl in BASELINES:
            rec[f'other own knob/own {bl}'] = _ratio(R(mid, 'A', other, bl)['mean'], main[bl]['mean'])
        if kind in ('org', 'spread'):
            req['level-aware'] = all(abs(rec[f'other own knob/own {bl}']) <= 0.5 for bl in BASELINES)
        others = {k: abs(R(mid, 'A', k)['mean']) for c2, ks in KNOBS.items()
                  if c2 != construct and c2 not in EXCLUDE[construct] for k in ks}
        others[NAMED_NUISANCE[construct]] = abs(R(mid, 'A', NAMED_NUISANCE[construct])['mean'])
        others = {k: v for k, v in others.items() if np.isfinite(v)}
        worst = max(others, key=others.get)
        rec.update(SR3=abs(main['B0']['mean']) / others[worst] if others[worst] else np.inf, worst_other=worst)
        req['SR3 >= 2'] = rec['SR3'] >= 2
        if construct == 'burst':
            for bl in BASELINES:
                for s in ('iaaft', 'ft'):
                    rec[f'{s} kept {bl}'] = R(mid, 'A', own, bl, surr=s)['mean'] / main[bl]['mean']
            req['IAAFT kept <= 0.5'] = all(rec[f'iaaft kept {bl}'] <= 0.5 for bl in BASELINES)
            beta = R(mid, 'A', 'nuis_beta')['mean']
            for s in ('iaaft', 'ft'):
                rec[f'beta {s} kept'] = R(mid, 'A', 'nuis_beta', surr=s)['mean'] / beta
            res = {bl: R(mid, 'A', own, bl, surr='resid') for bl in BASELINES}
            for bl in BASELINES:
                rec[f'resid own {bl}'], rec[f'resid d {bl}'] = res[bl]['mean'], res[bl]['d']
            rec['resid beta/own'] = _ratio(R(mid, 'A', 'nuis_beta', surr='resid')['mean'], res['B0']['mean'])
            rec['resid form_level/own'] = _ratio(R(mid, 'A', 'form_level', surr='resid')['mean'], res['B0']['mean'])
            rec['resid status'] = '; '.join(f'{bl} {st(r)}' for bl, r in res.items())
            rec['residual selective'] = (all(st(r) == 'ok' for r in res.values())
                                         and abs(rec['resid beta/own']) <= 0.5 and abs(rec['resid form_level/own']) <= 0.5)
        for nk in NUISANCES:
            rec[f'nuis/own {nk[5:]}'] = _ratio(R(mid, 'A', nk)['mean'], main['B0']['mean'])
        reversed_at = [k for k, v in status.items() if v == 'reversed']
        if reversed_at:
            verdict = 'drop (sign reversal)'
        elif not all(status[f'main {bl}'] == 'ok' for bl in BASELINES):
            verdict = 'drop (insensitive)'
        else:
            failed = [k for k, v in req.items() if not v]
            verdict = 'core' if not failed else f"sub-axis (fails {', '.join(failed)})"
        rows.append(dict(rec, reversed_at='; '.join(reversed_at),
                         insensitive_at='; '.join(k for k, v in status.items() if v == 'insensitive'), verdict=verdict))
    return pd.DataFrame(rows)


def core_table(R):
    """Core readouts: own response in the stage-3 corpus and selectivity once the stage-3 knobs are included."""
    rows = []
    new = KNOBS['location'] + KNOBS['burst']
    for name, (own, sign) in CORE_OWN.items():
        col = f'core: {name}'
        o = R(col, 'A', own)
        home = next(c for c, ks in KNOBS.items() if own in ks)
        others = {k: abs(R(col, 'A', k)['mean']) for c, ks in KNOBS.items() if c != home for k in ks}
        others = {k: v for k, v in others.items() if np.isfinite(v)}
        old = {k: v for k, v in others.items() if k not in new}
        worst = max(others, key=others.get)
        rec = dict(readout=name, own_knob=own, own_B0=o['mean'], d_B0=o['d'], status=_status(o, sign),
                   SR_stage12_knobs=abs(o['mean']) / max(old.values()), SR_with_stage3_knobs=abs(o['mean']) / others[worst],
                   worst_other=worst)
        rec.update({f'{k}/own': _ratio(R(col, 'A', k)['mean'], o['mean']) for k in new})
        rec.update({f'{k}/own B1': _ratio(R(col, 'A', k, 'B1')['mean'], o['mean']) for k in new})
        rec.update({f'D {k}/own': _ratio(R(col, 'D', k)['mean'], o['mean']) for k in ('fc_level', 'fp_level')})
        rec.update({f'K loc_level/own': _ratio(R(col, 'K', 'loc_level')['mean'], o['mean'])})
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', type=Path, required=True, help='stage-3 window directory')
    ap.add_argument('--out', type=Path, default=HERE, help='writes OUT/tables/stage3_*.csv')
    ap.add_argument('--refresh', action='store_true')
    a = ap.parse_args()
    df = with_residuals(load(a.raw, a.refresh))
    print(df.groupby(['family', 'surrogate', df.grid != '']).size())
    R = Responses(df)
    verdicts, core = evaluate(df, R), core_table(R)
    tables = a.out / 'tables'
    tables.mkdir(parents=True, exist_ok=True)
    verdicts.to_csv(tables / 'stage3_verdicts.csv', index=False, float_format='%.4g')
    core.to_csv(tables / 'stage3_core.csv', index=False, float_format='%.4g')
    R.table().to_csv(tables / 'stage3_responses.csv', index=False, float_format='%.4g')
    pd.set_option('display.width', 250, 'display.max_columns', 40, 'display.max_colwidth', 50)
    print(verdicts[['measure', 'sign', 'own_B0', 'd_B0', 'own_B1', 'd_B1', 'SR3', 'worst_other', 'verdict',
                    'reversed_at']].to_string(index=False))
    print(core.to_string(index=False))


if __name__ == '__main__':
    main()
