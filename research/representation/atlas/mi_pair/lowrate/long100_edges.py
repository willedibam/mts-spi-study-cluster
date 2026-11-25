"""Addendum B to preregs/long100-prereg.md: is a pair's departure from the G-K line persistent or one-off?

Reads the W50 and C10 score files (lowrate/score.py; normal-scored E = [real, surrogate] x [G, K] x 496 edges). No new
compute. These readouts address persistence only, not gain against coupling.

P1  Cross-window repeat at 50 s against 10 s. reorder.py v2's statistic: per window the pair-level residual (isotonic
    fit of K on G, then an additive node fit c + a_i + a_j), real and surrogate; within a session and state, the mean
    correlation of these 496-vectors over distinct window pairs, real minus surrogate. At 50 s over W50 windows; at
    10 s over C10 cuts from different parent windows only. Session delta = drug - awake_ec. A persistent pair property
    repeats more as estimator noise falls (50 s); one-off episodes do not.
P1w Within-parent repeat over the five contiguous cuts of each W50 window (descriptive: shared slow state inflates it).
P2  One cut or several. Per W50 window and edge, n_up = number of its five cuts in which the edge lies above the cut's
    own surrogate 95% band (band() of plot_character_draft, upward side), for the real cut and, as the baseline, for
    each cut's own surrogate about its own line. O = observed P(n_up >= 2 | n_up >= 1) / the same under independence
    across cuts at each cut's rate; persistence = O_real - O_surr; session delta = drug - awake_ec.
P2c P(edge above the band at 50 s | n_up = k), k = 0..5, drug and awake_ec windows (descriptive).

Reading (frozen, on P1 only; P2 is descriptive because O is noisy when few edges leave the band):
  PERSISTENT  r50 - r10 drug delta > 0 in >= 3/4 animals, and the pooled r50 drug delta > 0 with one-sided p < .05
  ONE-OFF     pooled r50 - r10 <= 0 and the r50 drug delta not significantly > 0
  MIXED       otherwise. The residual is net of an additive node fit, so this is persistence at the pair level.

    python long100_edges.py <scores dir with W50/ and C10/>
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import isotonic_regression
from scipy.stats import ttest_1samp

Q, CUTS, M = 95, 5, 32
AGENTS = ('KT', 'KTMD', 'MD', 'PF', 'SLP')


def resid(g, k):                                      # reorder.resid, verbatim (M = 32)
    iu = np.triu_indices(M, 1)
    o = np.argsort(g, kind='stable'); fit = np.empty_like(k); fit[o] = isotonic_regression(k[o]).x; e = k - fit
    B = np.zeros((len(k), M + 1)); B[:, 0] = 1; B[np.arange(len(k)), 1 + iu[0]] = 1; B[np.arange(len(k)), 1 + iu[1]] = 1
    return e - B @ np.linalg.lstsq(B, e, rcond=None)[0]


def up(e):                                            # band(): real above the surrogate's 95% half-width about its own line
    (g, k), (gs, ks) = e
    rs = ks - np.polyval(np.polyfit(gs, ks, 1), gs); lim = np.percentile(np.abs(rs), Q)
    return k - np.polyval(np.polyfit(g, k, 1), g) > lim, rs > lim


def load(d):
    L = [np.load(f) for f in sorted(glob.glob(f'{d}/zns_[AB]_*.npz'))]; cat = lambda k: np.concatenate([x[k] for x in L])
    return dict(E=cat('E')[:, 1].astype(float), chan=cat('chan'), **{k: cat(k) for k in ('id', 'unit', 'state', 'agent', 'animal')})


def rep(Rr, Rs, pairs):                               # mean corr over the given window pairs, real - surrogate
    if not len(pairs):
        return np.nan
    i, j = np.array(pairs).T; c = lambda R: np.corrcoef(R)[i, j].mean()
    return c(Rr) - c(Rs)


def p_ge2(p):                                         # independence: P(n >= 2 | n >= 1) at per-cut rates p
    p0 = np.prod(1 - p); p1 = sum(p[c] * np.prod(np.delete(1 - p, c)) for c in range(len(p)))
    return (1 - p0 - p1) / (1 - p0)


def persistence(U):                                   # U: (5 cuts, 496) booleans
    n = U.sum(0); hit = n >= 1
    return (np.mean(n[hit] >= 2) if hit.any() else np.nan) / p_ge2(U.mean(1))


def main(root):
    W, C = load(Path(root) / 'W50'), load(Path(root) / 'C10')
    pos = {k: i for i, k in enumerate(W['id'])}
    par = np.array([pos[i.rsplit(':c', 1)[0]] for i in C['id']]); cut = np.array([int(i.rsplit(':c', 1)[1]) for i in C['id']])
    assert (C['chan'] == W['chan'][par]).all(), 'cut and parent subsets differ'
    kids = {w: np.where(par == w)[0][np.argsort(cut[par == w])] for w in range(len(W['id']))}
    assert all(len(v) == CUTS for v in kids.values()), 'a W50 window lacks some of its five cuts'
    R = lambda X: (np.array([resid(*X['E'][w, 0]) for w in range(len(X['id']))]), np.array([resid(*X['E'][w, 1]) for w in range(len(X['id']))]))
    (Wr, Ws), (Cr, Cs) = R(W), R(C)
    Uw = [up(W['E'][w]) for w in range(len(W['id']))]; Uc = [up(C['E'][c]) for c in range(len(C['id']))]
    S = []
    for u in np.unique(W['unit']):
        ag, an = W['agent'][W['unit'] == u][0], W['animal'][W['unit'] == u][0]; tgt = 'sleep' if ag == 'SLP' else 'drug'
        row = dict(unit=u, ag=ag, an=an)
        for s in (tgt, 'awake_ec', 'awake_eo'):
            ws = np.where((W['unit'] == u) & (W['state'] == s))[0]
            if len(ws) < 2:
                continue
            assert (W['chan'][ws] == W['chan'][ws[0]]).all(), f'{u} {s}: subsets differ between windows'
            cs = np.concatenate([kids[w] for w in ws])
            p50 = [(a, b) for a in range(len(ws)) for b in range(a + 1, len(ws))]
            p10 = [(a, b) for a in range(len(cs)) for b in range(a + 1, len(cs)) if par[cs[a]] != par[cs[b]]]
            pin = [(a, b) for a in range(CUTS) for b in range(a + 1, CUTS)]
            row[s] = dict(r50=rep(Wr[ws], Ws[ws], p50), r10=rep(Cr[cs], Cs[cs], p10),
                          rin=np.mean([rep(Cr[kids[w]], Cs[kids[w]], pin) for w in ws]),
                          per=np.nanmean([persistence(np.array([Uc[c][0] for c in kids[w]])) -
                                          persistence(np.array([Uc[c][1] for c in kids[w]])) for w in ws]))
        if tgt in row and 'awake_ec' in row:
            S.append(row)
    keys = ('r50', 'r10', 'rin', 'per')
    D = {k: np.array([r[('sleep' if r['ag'] == 'SLP' else 'drug')][k] - r['awake_ec'][k] for r in S]) for k in keys}
    D['r50-r10'] = D['r50'] - D['r10']
    ag, an = np.array([r['ag'] for r in S]), np.array([r['an'] for r in S]); dr = ag != 'SLP'
    out = {}
    print(f'sessions {len(S)} (drug {dr.sum()}); awake_ec levels: ' + '  '.join(
        f'{k} {np.nanmean([r["awake_ec"][k] for r in S]):+.4f}' for k in keys))
    for k in ('r50', 'r10', 'r50-r10', 'rin', 'per'):
        A = {x: np.nanmean(D[k][dr & (an == x)]) for x in np.unique(an[dr])}; a = np.array(list(A.values()))
        t = ttest_1samp(a, 0, alternative='greater')
        eo = np.array([r['awake_eo'][k2] - r['awake_ec'][k2] for r in S if 'awake_eo' in r for k2 in [k if k != 'r50-r10' else 'r50']])
        out[k] = dict(animals=A, pooled=float(a.mean()), positive=int((a > 0).sum()), p=float(t.pvalue),
                      agents={g: float(np.nanmean(D[k][ag == g])) for g in AGENTS if (ag == g).any()})
        print(f'{k:8s} drug Δ pooled {a.mean():+.4f} | animals ' + ', '.join(f'{x} {v:+.4f}' for x, v in A.items()) +
              f' ({(a > 0).sum()}/{len(a)} > 0, one-sided p {t.pvalue:.3f}) | ' +
              '  '.join(f'{g} {v:+.4f}' for g, v in out[k]['agents'].items()) + (f' | EO SD {np.nanstd(eo, ddof=1):.4f}' if k != 'r50-r10' else ''))
    # P2c: P(above the band at 50 s | n_up = k)
    for s in ('awake_ec', 'drug'):
        ws = np.where(W['state'] == s)[0]
        n = np.concatenate([np.array([Uc[c][0] for c in kids[w]]).sum(0) for w in ws]); u50 = np.concatenate([Uw[w][0] for w in ws])
        print(f'P2c {s:8s} ' + '  '.join(f'n_up={k}: {u50[n == k].mean():.3f} (edges {np.sum(n == k)})' for k in range(CUTS + 1) if (n == k).any()))
    pers = out['r50-r10']['positive'] >= 3 and out['r50']['pooled'] > 0 and out['r50']['p'] < .05
    trans = out['r50-r10']['pooled'] <= 0 and not out['r50']['p'] < .05
    print('Addendum B reading (persistence only, not mechanism):',
          'PERSISTENT at the edge level' if pers else 'ONE-OFF / transient' if trans else 'MIXED')
    Path(root, 'long100-edges.json').write_text(json.dumps(out, indent=1, default=float) + '\n')


if __name__ == '__main__':
    main(sys.argv[1])
