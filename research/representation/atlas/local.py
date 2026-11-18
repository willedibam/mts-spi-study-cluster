"""Channel-level (node-wise) z for the named SPI pairs: does a pair's agreement change locally, at a few channels,
even when the global z over all channel pairs barely moves? (findings.md, "Channel-level z".)

Node-wise z_i: the correlation of SPI A's and SPI B's values over the edges incident to channel i (both directions of
the stored off-diagonal matrix). Per session (unit) and state, maps are averaged over even and over odd windows
separately, and each half is contrasted with the same half of eyes-closed wake.

Readouts per pair and condition (agent:state), raw tier (stored pyspi, no surrogates):
  held-out local excess: the k channels with the largest (or, separately, smallest) even-half change, scored on the
      odd half as their mean change minus the odd half's all-channel mean change (sign-flipped for the lower tail, so
      positive means the local change reproduces). Subtracting the all-channel mean removes the global shift.
  null: the identical pipeline on the eyes-open contrast, pooled over every session; a session passes when its
      excess exceeds the 95th percentile of that pool.
  split-half map reliability: corr over channels of the even-half and odd-half change maps.
  within-animal agreement: corr of full change maps between sessions of the same animal and condition, on the
      electrodes both kept (channel index is not electrode identity across sessions).
  covariates: corr over channels of the change map with the change in channel spectral entropy, and with the change
      in the SD of A over incident edges (range restriction moves z without any change in character).

Decision rule, fixed before the run: a pair has a local signature in a condition if, for either tail, the held-out
excess passes in at least 75% of sessions and in every animal (an animal passes when at least half its sessions
do), and the mean within-animal agreement exceeds that of
the eyes-open contrast. A map that correlates (|r| > .5 on average) with either covariate gets the literal reading
only.

Residualised variant (``*_resid`` fields; added after the first run, so exploratory): each half's change map is
regressed on that half's changes in the spread of A, the spread of B and channel spectral entropy, and the held-out
test is rerun on the residuals against the same eyes-open pipeline (pass: the held-out clause of the rule). Pairs with a partner whose wake between-window reliability is below RELIABILITY are skipped (their node maps
are noise); the MI pair is kept as the known-positive reference. A null (stationary or zero-phase) is run only for
pairs that pass.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.signal import welch

from research.representation.atlas.discovery import PAIRS, _profiles
from research.representation.character_axes.episodic import sleep_tertiles

RELIABILITY = .3
K_TOP = 5
REFERENCE_PAIR = ('mi_gaussian', 'mi_kraskov_NN-4_DCE-AUTO')


def node_stats(a, b, rows, cols, M):
    """Per-channel correlation of a and b over incident edges, and the SDs of a and of b there."""
    s = lambda v: np.bincount(rows, v, M) + np.bincount(cols, v, M)
    n = s(np.ones_like(a))
    ma, mb = s(a) / n, s(b) / n
    va, vb = s(a * a) / n - ma ** 2, s(b * b) / n - mb ** 2
    cab = s(a * b) / n - ma * mb
    with np.errstate(invalid='ignore', divide='ignore'):
        return cab / np.sqrt(va * vb), np.sqrt(np.maximum(va, 0)), np.sqrt(np.maximum(vb, 0))


def spectral_entropy(X, fs):
    f, p = welch(X, fs=fs, nperseg=min(len(X), 128), axis=0)
    p = p[(f > 0) & (f <= 40)]
    p = p / p.sum(0)
    return -(p * np.log(p)).sum(0) / np.log(len(p))


def _corr(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 3 else np.nan


def local(bank, pyspi_dir, reliability_json, output, reference, jobs=8, k=K_TOP):
    manifest = json.loads((bank / 'manifest.json').read_text())
    rel = json.loads(reliability_json.read_text())['reference_state']
    shards = sorted(pyspi_dir.glob('shard-*.npz'))
    with np.load(shards[0], allow_pickle=True) as d:
        have = {str(x) for x in d['__spis__']}
    named, skipped = [], []
    for c, a, b in PAIRS:
        if a not in have or b not in have:
            continue
        r_ab = min(rel.get(a, {}).get('reliability', np.nan), rel.get(b, {}).get('reliability', np.nan))
        if (a, b) == REFERENCE_PAIR or r_ab >= RELIABILITY:
            named.append((c, a, b))
        else:
            skipped.append([c, a, b, r_ab])
    wanted = sorted({x for _, a, b in named for x in (a, b)})
    merged = {}
    for part in Parallel(n_jobs=jobs)(delayed(_profiles)(s, wanted) for s in shards):
        merged.update(part)
    rows = [r for r in manifest['rows'] if r['id'] in merged]
    state = np.array(sleep_tertiles(rows) if any('delta_fraction' in x for x in rows) else [x['state'] for x in rows])
    unit = np.array([r['unit'] for r in rows])
    agent = {r['unit']: r.get('agent', 'all') for r in rows}
    animal = {r['unit']: r.get('animal') or r['unit'] for r in rows}
    kept = {u['unit']: u['kept_channels'] for u in manifest.get('preprocessing', {}).get('units', []) if 'kept_channels' in u}
    with np.load(bank / 'windows.npz') as archive:
        entropy = {r['id']: spectral_entropy(np.asarray(archive[r['id']], dtype=np.float64), manifest['fs']) for r in rows}

    # per window: node z (pairs x M), SD of A (pairs x M), global z (pairs)
    def window(r):
        P = merged[r['id']]
        M = r['M']
        rr, cc = np.nonzero(~np.eye(M, dtype=bool))
        out_z, out_sd, out_sdb, glob = [], [], [], []
        for _, a, b in named:
            x, y = P[wanted.index(a)].astype(np.float64), P[wanted.index(b)].astype(np.float64)
            z, sd, sdb = node_stats(x, y, rr, cc, M)
            out_z.append(z), out_sd.append(sd), out_sdb.append(sdb), glob.append(_corr(x, y))
        return np.array(out_z), np.array(out_sd), np.array(glob), np.array(out_sdb)
    W = [window(r) for r in rows]

    half = np.zeros(len(rows), int)
    for u, s in set(zip(unit, state)):
        idx = np.flatnonzero((unit == u) & (state == s))
        half[idx] = np.arange(len(idx)) % 2
    mean = lambda idx, j: np.nanmean([W[i][j] for i in idx], 0)
    maps = {}  # (u, s) -> dict(z_even, z_odd, z, sd, ent, glob)
    for u, s in set(zip(unit, state)):
        idx = np.flatnonzero((unit == u) & (state == s))
        ev, od = idx[half[idx] == 0], idx[half[idx] == 1]
        if len(ev) == 0 or len(od) == 0:
            continue
        ent = lambda ix: np.mean([entropy[rows[i]['id']] for i in ix], 0)
        maps[(u, s)] = dict(even=mean(ev, 0), odd=mean(od, 0), z=mean(idx, 0), sd=mean(idx, 1), glob=mean(idx, 2),
                            ent=ent(idx), cov_even=(mean(ev, 1), mean(ev, 3), ent(ev)),
                            cov_odd=(mean(od, 1), mean(od, 3), ent(od)))

    def contrast(u, s):
        g, w = maps[(u, s)], maps[(u, reference)]
        de, do, d = g['even'] - w['even'], g['odd'] - w['odd'], g['z'] - w['z']
        res = dict(glob=g['glob'] - w['glob'], d=d, dsd=g['sd'] - w['sd'], dent=g['ent'] - w['ent'],
                   split=np.array([_corr(de[p], do[p]) for p in range(len(named))]))
        def residual(dz, half, p):
            cv = [g[half][0][p] - w[half][0][p], g[half][1][p] - w[half][1][p], g[half][2] - w[half][2]]
            X = np.column_stack([np.ones(len(dz))] + cv)
            ok = np.isfinite(dz) & np.isfinite(X).all(1)
            out = np.full(len(dz), np.nan)
            if ok.sum() > X.shape[1] + 2:
                out[ok] = dz[ok] - X[ok] @ np.linalg.lstsq(X[ok], dz[ok], rcond=None)[0]
            return out
        de_r = np.array([residual(de[p], 'cov_even', p) for p in range(len(named))])
        do_r = np.array([residual(do[p], 'cov_odd', p) for p in range(len(named))])
        for suffix, (E, O) in (('', (de, do)), ('_resid', (de_r, do_r))):
            for tail, sign in (('upper', 1), ('lower', -1)):
                ex = []
                for p in range(len(named)):
                    ok = np.isfinite(E[p]) & np.isfinite(O[p])
                    order = np.flatnonzero(ok)[np.argsort(-sign * E[p][ok])][:k]
                    ex.append(sign * (O[p][order].mean() - O[p][ok].mean()) if ok.any() else np.nan)
                res[tail + suffix] = np.array(ex)
        res['split_resid'] = np.array([_corr(de_r[p], do_r[p]) for p in range(len(named))])
        res['cov_entropy'] = np.array([_corr(d[p], res['dent']) for p in range(len(named))])
        res['cov_sd'] = np.array([_corr(d[p], res['dsd'][p]) for p in range(len(named))])
        absd = np.abs(d)
        res['top_share'] = np.array([np.sort(absd[p][np.isfinite(absd[p])])[::-1][:k].sum() / np.nansum(absd[p])
                                     for p in range(len(named))])
        return res

    C = {(u, s): contrast(u, s) for (u, s) in maps if s != reference and (u, reference) in maps}
    eo = [v for (u, s), v in C.items() if s == 'awake_eo']
    thresholds = {t: np.nanpercentile(np.array([v[t] for v in eo]), 95, axis=0)
                  for t in ('upper', 'lower', 'upper_resid', 'lower_resid')}

    def agreement(units, s):
        out = []
        for i, u in enumerate(units):
            for v in units[i + 1:]:
                if animal[u] != animal[v] or u not in kept or v not in kept:
                    continue
                common = sorted(set(kept[u]) & set(kept[v]))
                iu = [kept[u].index(e) for e in common]
                iv = [kept[v].index(e) for e in common]
                out.append([_corr(C[(u, s)]['d'][p][iu], C[(v, s)]['d'][p][iv]) for p in range(len(named))])
        return np.nanmean(out, 0) if out else np.full(len(named), np.nan)

    eo_units = sorted(u for (u, s) in C if s == 'awake_eo')
    eo_agree = agreement(eo_units, 'awake_eo')
    report = dict(bank=str(bank), reference=reference, k=k, reliability_floor=RELIABILITY,
                  skipped_unreliable=skipped, eyes_open_95=dict(upper=thresholds['upper'].tolist(),
                                                                lower=thresholds['lower'].tolist()),
                  pairs=[])
    conditions = sorted({f'{agent[u]}:{s}' for (u, s) in C})
    for p, (c, a, b) in enumerate(named):
        entry = dict(character=c, a=a, b=b, eyes_open_agreement=float(eo_agree[p]), conditions={})
        for cond in conditions:
            g, s = cond.split(':', 1)
            units = sorted(u for (u, s_) in C if s_ == s and agent[u] == g)
            if not units:
                continue
            e = {}
            for tail in ('upper', 'lower', 'upper_resid', 'lower_resid'):
                vals = np.array([C[(u, s)][tail][p] for u in units])
                passed = vals > thresholds[tail][p]
                by = {}
                for u, ok in zip(units, passed):
                    by.setdefault(animal[u], []).append(ok)
                e[tail] = dict(mean=float(np.nanmean(vals)), passed=int(passed.sum()), units=len(units),
                               animals_all_pass=int(sum(np.mean(v) >= .5 for v in by.values())), animals=len(by))
            agree = agreement(units, s)[p]
            e.update(global_dz=float(np.mean([C[(u, s)]['glob'][p] for u in units])),
                     split_half=float(np.nanmean([C[(u, s)]['split'][p] for u in units])),
                     within_animal_agreement=float(agree),
                     cov_entropy=float(np.nanmean([C[(u, s)]['cov_entropy'][p] for u in units])),
                     cov_sd=float(np.nanmean([C[(u, s)]['cov_sd'][p] for u in units])),
                     top_share=float(np.nanmean([C[(u, s)]['top_share'][p] for u in units])))
            frac = lambda t: e[t]['passed'] >= .75 * e[t]['units'] and e[t]['animals_all_pass'] == e[t]['animals']
            e['local_signature'] = bool((frac('upper') or frac('lower')) and agree > eo_agree[p])
            e['split_half_resid'] = float(np.nanmean([C[(u, s)]['split_resid'][p] for u in units]))
            e['local_signature_resid'] = bool(frac('upper_resid') or frac('lower_resid'))
            e['literal_only'] = bool(abs(e['cov_entropy']) > .5 or abs(e['cov_sd']) > .5)
            entry['conditions'][cond] = e
        report['pairs'].append(entry)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('bank', type=Path)
    p.add_argument('pyspi', type=Path)
    p.add_argument('--reliability', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reference', default='awake_ec')
    p.add_argument('--jobs', type=int, default=8)
    args = p.parse_args()
    local(args.bank, args.pyspi, args.reliability, args.output, args.reference, args.jobs)
