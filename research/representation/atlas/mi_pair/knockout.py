"""Where the macaque MI-pair drug fall lives, and how the estimator responds to known structure (preregs/knockout-prereg.md).

map:    Part 0, drug-blind. Synthetic 16-channel signals with known structure (generators `block`, `gain_shared`,
        `gain_B4`, `xfreq`), phase-randomised from the periodograms of normal-scored W50 stage-A windows (awake
        eyes-closed of KTMD units, KTMD drug, PF drug; 32-channel subset as edges.py), 100 s long, so every window is
        cut from inside a longer signal (no circular ends). Each realisation is scored in every cell of CELLS (rate,
        duration, band, Theiler window: z(G, K) real and against its own spectrum-final joint IAAFT) and by the 14
        band-knockout variants of its anchor-like 10-s, 100 Hz cut. Per-edge G and K are stored for every evaluation.
        python knockout.py map <W50 stage-a bank> <out.npz> <n block> <n other> <workers>
score:  Part A, real data. The 14 knockout variants of each anchor window (edges.py recompute's rows, subsets, normal-
        scoring), phase seed default_rng([row, 2000]). Stores z and per-edge G, K per variant.
        python knockout.py score <bank dir> <out.npz> <agent> <first|row> <workers>
offset: Part B. long100.py's T = 1000 drug-blind calibration (calib_one: `block`, `gain_shared`) on a band-sweep arm's bank.
        python knockout.py offset <arm stage-a bank> <out.npz> <n block> <n gain_shared> <workers>
mread:  descriptive tables of a map file.  python knockout.py mread <map.npz>

The real-data reading is written at the freeze, after Part 0 returns.
PYTHONPATH=<repo>/src:<repo>/research/representation/atlas/mi_pair:<repo>/research/representation/atlas/mi_pair/lowrate
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

import edges as E
from mtsspi.surrogates import band_masks, band_phase_randomized, multivariate_iaaft

FS, TL, M16, DWELL = 100, 10000, 16, 100             # 100 Hz; 100-s realisations; gain dwell 1 s
BANDS = ((.1, 2), (2, 4), (4, 8), (8, 15), (15, 25), (25, 50))
BN = ('B1', 'B2', 'B3', 'B4', 'B5', 'B6')
VARIANTS = ('real',) + tuple(f'knock{b}' for b in BN) + tuple(f'keep{b}' for b in BN) + ('all',)
C10, C50 = slice(4500, 5500), slice(2500, 7500)        # 10 s and 50 s from the middle
# cell: (description, decimate to 20 Hz after the 8 Hz low-pass, filter at 100 Hz, cut, fixed Theiler window or None)
CELLS = {'A100x10': ('anchor-like: 100 Hz, 10 s, AUTO', 0, None, C10, None),
         'A100x50': ('100 Hz, 50 s', 0, None, C50, None),
         'D20x10': ('8 Hz low-pass, 20 Hz, 10 s (T 200)', 1, None, C10, None),
         'D20x50': ('8 Hz low-pass, 20 Hz, 50 s (T 1000)', 1, None, C50, None),
         'LP8': ('8 Hz low-pass at 100 Hz, 10 s (L20)', 0, ('lowpass', 8), C10, None),
         'HP8': ('8 Hz high-pass at 100 Hz, 10 s', 0, ('highpass', 8), C10, None),
         'BP8-15': ('8-15 Hz band-pass at 100 Hz, 10 s', 0, ('bandpass', (8, 15)), C10, None),
         'W0.25': ('100 Hz, 10 s, Theiler 0.25 s in every state', 0, None, C10, 25),
         'W1': ('100 Hz, 10 s, Theiler 1 s in every state', 0, None, C10, 100)}
SPECTRA = (('AW', 'KTMD', 'awake_ec'), ('KTMD', 'KTMD', 'drug'), ('PF', 'PF', 'drug'))
GENS = ('block', 'gain_shared', 'gain_B4', 'xfreq')


def filt(x, kind, c):                                  # order-8 zero-phase Butterworth, as bands.py (prepare.lowpass/highpass)
    return sosfiltfilt(butter(8, c, btype=kind, fs=FS, output='sos'), x, axis=0)


def edge_mi_w(x, w):                                  # edges.edge_mi with one fixed Theiler window for every pair
    x = (x - x.mean(0)) / x.std(0); T, M = x.shape; iu = np.triu_indices(M, 1); t = np.arange(T)
    td = np.abs(t[:, None] - t[None, :]); r = np.corrcoef(x.T)[iu]
    k = np.concatenate([E.ksg_rows(x, i, np.arange(i + 1, M), np.full(M - i - 1, w), td) for i in range(M - 1)])
    return -0.5 * np.log(1 - r ** 2), k


def variants(v, phase):
    """The 14 knockout variants of a normal-scored window, each normal-scored again (VARIANTS order)."""
    m = band_masks(len(v), FS, BANDS); a = np.any(m, 0)
    return [v] + [E.ns(band_phase_randomized(v, phase, b)) for b in m] + \
        [E.ns(band_phase_randomized(v, phase, a & ~b)) for b in m] + [E.ns(band_phase_randomized(v, phase, a))]


def knock_scores(v, phase):
    Ed = [E.edge_mi(u) for u in variants(v, phase)]
    return np.array([E.zc(g, k) for g, k in Ed]), np.array(Ed, np.float32)


def synth(kind, src, rng):
    """`gen` of long100.py on a TL-sample grid (periodogram interpolated), plus band-limited and cross-frequency gain."""
    s = src[:, rng.permutation(src.shape[1])[:M16]]; A = np.abs(np.fft.rfft(s, axis=0))
    fo, fi = np.fft.rfftfreq(TL), np.fft.rfftfreq(len(s))
    A = np.array([np.interp(fo, fi, A[:, m]) for m in range(M16)]).T
    ph = np.exp(2j * np.pi * rng.random(A.shape)); ph[0] = 1; ph[-1] = 1
    X = np.fft.irfft(A * ph, TL, axis=0); X = (X - X.mean(0)) / X.std(0)
    L_ = np.eye(M16)                                  # long100.gen's block mixing, verbatim
    for b in range(4):
        idx = np.arange(b * 4, b * 4 + 4); C = np.full((4, 4), rng.uniform(.2, .9)); np.fill_diagonal(C, 1)
        C = C + np.diag(rng.uniform(0, .3, 4)); d = np.sqrt(np.diag(C))
        L_[np.ix_(idx, idx)] = np.linalg.cholesky(C / np.outer(d, d))
    X = X @ L_.T
    sw = lambda: np.where(np.cumsum(rng.random(TL) < 1 / DWELL) % 2, 3., 1.)   # long100.markov, dwell DWELL
    if kind == 'gain_shared':
        X = X * sw()[:, None]
    elif kind in ('gain_B4', 'xfreq'):
        m = band_masks(TL, FS, BANDS); F = np.fft.rfft(X, axis=0)
        XB = np.fft.irfft(F * m[3][:, None], TL, axis=0)
        g = sw() if kind == 'gain_B4' else np.where(np.fft.irfft(F * m[0][:, None], TL, axis=0).mean(1) > 0, 3., 1.)
        X = X + XB * (g - 1)[:, None]
    return X


def map_one(a):
    kind, seed, src = a; rng = np.random.default_rng(seed); X = synth(kind, src, rng)
    Z, Ed = [], []
    for nm, (_, dec, fl, cut, w) in CELLS.items():
        x = filt(X, 'lowpass', 8)[cut][::5] if dec else (filt(X, *fl) if fl else X)[cut]
        v = E.ns(x); s = multivariate_iaaft(v, np.random.default_rng([seed, 1]), final='spectrum')[0]
        f = E.edge_mi if w is None else (lambda u: edge_mi_w(u, w))
        e = [f(v), f(s)]; Z.append([E.zc(*e[0]), E.zc(*e[1])]); Ed.append(e)
    v = E.ns(X[C10]); zk, ek = knock_scores(v, np.random.default_rng([seed, 2]).uniform(-np.pi, np.pi, len(v) // 2 + 1))
    return np.array(Z), np.array(Ed, np.float32), zk, ek


def sources(d):
    man = json.loads((Path(d) / 'manifest.json').read_text()); W = np.load(Path(d) / 'windows.npz'); src = {}
    for tag, ag, st in SPECTRA:
        rows = [r for r in man['rows'] if r['state'] in E.STATES and r['agent'] == ag]; S0 = E.sub32(rows[0]['M'])
        src[tag] = [E.ns(W[r['id']].astype(float)[:, S0]) for r in rows if r['state'] == st]
    return src


def run_map(d, out, nb, ng, nw):
    src = sources(d); jobs, lab = [], []
    for si, (tag, _, _) in enumerate(SPECTRA):
        for ki, kind in enumerate(GENS):
            for i in range(nb if kind == 'block' else ng):
                jobs.append((kind, 9 * 10 ** 6 + 10 ** 5 * si + 10 ** 4 * ki + i, src[tag][i % len(src[tag])]))
                lab.append((kind, tag, i))
    with Pool(nw) as p:
        R = p.map(map_one, jobs, chunksize=1)
    np.savez(out, Z=np.array([r[0] for r in R]), E=np.array([r[1] for r in R]), Zk=np.array([r[2] for r in R]),
             Ek=np.array([r[3] for r in R]), kind=np.array([k for k, _, _ in lab]), tag=np.array([t for _, t, _ in lab]),
             i=np.array([i for _, _, i in lab]), cells=np.array(list(CELLS)), variants=np.array(VARIANTS),
             n_sources=json.dumps({k: len(v) for k, v in src.items()}))
    print('done', out, flush=True)


def mread(f):
    """Descriptive. Drug-minus-awake contrasts use the generator on the drug spectra (mean of KTMD and PF) minus `block`
    on the awake spectrum, the real readout's construction. SEs over generator draws (bootstrap for ratios)."""
    D = np.load(f); Z, Zk, kind, tag = D['Z'], D['Zk'], D['kind'], D['tag']; cells = list(D['cells'])
    sel = lambda k, t: (kind == k) & (tag == t)
    se = lambda x: x.std(ddof=1) / np.sqrt(len(x))
    print('z_real - z_IAAFT per cell: mean +- SE by generator and spectrum')
    for c, nm in enumerate(cells):
        print(f'{nm:8s} ' + ' | '.join(f'{k} ' + ' '.join(f'{t} {(Z[sel(k, t), c, 0] - Z[sel(k, t), c, 1]).mean():+.4f}'
                                                         f'±{se(Z[sel(k, t), c, 0] - Z[sel(k, t), c, 1]):.4f}' for t in ('AW', 'KTMD', 'PF')) for k in GENS))
    dz = lambda k, c: np.mean([(Z[sel(k, t), c, 0] - Z[sel(k, t), c, 1]).mean() for t in ('KTMD', 'PF')]) - \
        (Z[sel('block', 'AW'), c, 0] - Z[sel('block', 'AW'), c, 1]).mean()
    print('drug - awake contrast per cell, ratio to A100x10:')
    for k in GENS[1:]:
        print(f'{k:12s} ' + '  '.join(f'{nm} {dz(k, c):+.4f} (R {dz(k, c) / dz(k, 0):.2f})' for c, nm in enumerate(cells)))
    print('block, knockout numerators per spectrum (end-mismatch and false-positive check): z_real - z_knockB')
    for t in ('AW', 'KTMD', 'PF'):
        b = Zk[sel('block', t)]
        print(f'{t:4s} ' + '  '.join(f'{BN[j]} {(b[:, 0] - b[:, 1 + j]).mean():+.4f}±{se(b[:, 0] - b[:, 1 + j]):.4f}' for j in range(6)) +
              f'  all {(b[:, 0] - b[:, 13]).mean():+.4f}±{se(b[:, 0] - b[:, 13]):.4f}')
    rng = np.random.default_rng(0)

    def ratios(k, boot):
        pick = lambda m: rng.choice(np.where(m)[0], m.sum()) if boot else np.where(m)[0]
        g = np.concatenate([Zk[pick(sel(k, t))] for t in ('KTMD', 'PF')]); a = Zk[pick(sel('block', 'AW'))]
        dl = lambda j0, j1: (g[:, j0] - g[:, j1]).mean() - (a[:, j0] - a[:, j1]).mean()
        Dd = dl(0, 13); return np.array([dl(0, 1 + j) / Dd for j in range(6)] + [dl(7 + j, 13) / Dd for j in range(6)] + [Dd])
    print('knockout loss l_B and sufficiency s_B (drug spectra minus awake block), bootstrap SE over draws:')
    for k in GENS[1:]:
        r, bs = ratios(k, False), np.array([ratios(k, True) for _ in range(500)]).std(0)
        print(f'{k:12s} D {r[12]:+.4f}±{bs[12]:.4f}' + (' (|D| < 3 SE: ratios unstable, read numerators)' if abs(r[12]) < 3 * bs[12] else '') + ' | l ' + ' '.join(f'{BN[j]} {r[j]:+.2f}±{bs[j]:.2f}' for j in range(6)) +
              ' | s ' + ' '.join(f'{BN[j]} {r[6 + j]:+.2f}±{bs[6 + j]:.2f}' for j in range(6)))
    rg, rb = ratios('gain_shared', False), ratios('gain_B4', False)
    bd = np.array([ratios('gain_B4', True) - ratios('gain_shared', True) for _ in range(500)])
    e = rb[:6] - rg[:6]; c = e[3] - np.max(np.abs(np.delete(e, 3)))
    cb = bd[:, 3] - np.max(np.abs(np.delete(bd[:, :6], 3, 1)), 1)
    print(f'identifiability: gain_B4 l_B - gain_shared l_B = {np.round(e, 2)}; B4 minus largest other |.| {c:+.2f}'
          f' (bootstrap 2.5% {np.percentile(cb, 2.5):+.2f}) ->', 'B4 singled out' if np.percentile(cb, 2.5) > 0 else 'not singled out')


def score(d, out, agent, rowmode, nw):                # rows, subsets as edges.py recompute
    man = json.load(open(f'{d}/manifest.json')); W = np.load(f'{d}/windows.npz')
    rows = [r for r in man['rows'] if r['state'] in E.STATES and r['agent'] in agent.split(',')]
    S0 = E.sub32(rows[0]['M']); sub = lambda r: S0 if rowmode == 'first' else E.sub32(r['M'])
    jobs = [(E.ns(W[r['id']].astype(float)[:, sub(r)]), k) for k, r in enumerate(rows)]
    with Pool(nw) as p:
        res = p.starmap(score_one, jobs, chunksize=1)
    m, kept = E.meta(rows, man)
    labels = np.array([np.array(kept[r['unit']][:r['M']])[sub(r)] for r in rows])
    np.savez(out, Zk=np.array([a for a, _ in res]), Ek=np.array([b for _, b in res]), chan=labels, rowmode=rowmode,
             variants=np.array(VARIANTS), **m)
    print('done', out, flush=True)


def score_one(v, row):
    return knock_scores(v, np.random.default_rng([row, 2000]).uniform(-np.pi, np.pi, len(v) // 2 + 1))


def offset(d, out, nb, ng, nw):                       # long100.calib at T = 1000 only, on this bank's own windows
    import long100 as LH
    src = sources(d); jobs, lab = [], []
    for si, (tag, _, _) in enumerate(SPECTRA):
        for ki, (kind, n) in enumerate((('block', nb), ('gain_shared', ng))):
            for i in range(n):
                jobs.append((kind, 8 * 10 ** 6 + 10 ** 5 * si + 10 ** 4 * ki + 2 * i, src[tag][i % len(src[tag])][:1000]))
                lab.append((kind, tag))
    with Pool(nw) as p:
        R = p.map(LH.calib_one, jobs, chunksize=1)
    np.savez(out, dz=np.array(R), kind=np.array([k for k, _ in lab]), tag=np.array([t for _, t in lab]))
    print('done', out, flush=True)


if __name__ == '__main__':
    a = sys.argv
    if a[1] == 'map':
        run_map(a[2], a[3], int(a[4]), int(a[5]), int(a[6]))
    elif a[1] == 'mread':
        mread(a[2])
    elif a[1] == 'score':
        score(a[2], a[3], a[4], a[5], int(a[6]))
    elif a[1] == 'offset':
        offset(a[2], a[3], int(a[4]), int(a[5]), int(a[6]))
