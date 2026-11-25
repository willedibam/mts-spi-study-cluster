"""100 Hz on 50-s windows for the macaque MI pair (preregs/long100-prereg.md): window duration at the analysis rate.

The 50-s windows are exactly the low-rate long arms' (D20L/S20L: same rows, `session`, `start_s`), rebuilt at 100 Hz
from the 1 kHz archives by pass 1 of prepare.nt_archive (as build.py and bands.py), CAR per window:
  W50     0.5-40 Hz, 50 s, T = 5000
  W50L8   low-pass 8 Hz (build.py's call for L20/D20L), 50 s, T = 5000
  C10     W50 cut into five contiguous 10-s pieces, T = 1000 (ids <id>:c<j>, start_s + 10 j)
  C10L8   W50L8 cut the same way
CAR is per sample, so it commutes with subsampling and cutting. Gate A: every W50 window's every 5th sample must
equal S20L's window and W50L8's must equal D20L's, or nothing is written.

  PYTHONPATH=<repo>:<repo>/src python .../lowrate/long100.py build <banks dir> <lowrate banks dir> <archives dir> <out dir> <workers>
  PYTHONPATH=<repo>:<repo>/src python .../lowrate/long100.py read <scores dir>
  PYTHONPATH=<repo>/src:<repo>/research/representation/atlas/mi_pair python .../lowrate/long100.py calib <W50 stage-a bank> <out.npz> <n block> <n gain_shared> <workers>
  PYTHONPATH=... python .../lowrate/long100.py cscore <calib.npz>

calib: the drug-blind stationary offset at T = 1000 and T = 5000 from the same 50-s sources, for the anchor estimator only
(edges.edge_mi: AUTO Theiler window, k = 4, psi(T)). Generators `block` (stationary linear coupling, MI 0 between blocks:
false-positive offset) and `gain_shared` (one shared gain switch: sensitivity) are the robustness session's
(ksgbias_calib.py on branch claude/mi-robustness-9d2fea, copied verbatim below), 16 channels phase-randomised from the
periodograms of normal-scored W50 stage-A windows (awake eyes-closed of KTMD units, KTMD drug, PF drug; 32-channel subset as
edges.py); T = 1000 is the first 1000 samples of the same source. Each synthetic window is normal-scored, given its own
spectrum-final joint IAAFT draw and scored z(G, K) real - surrogate.
"""
import json
import re
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

STAGES = ('stage-a', 'stage-b')
L, Q, CUTS, CUT8 = 5000, 5, 5, 8.0                    # 50 s at 100 Hz; 20 Hz factor; 10-s pieces; build.py CUT[20]


def archive(args):
    from research.representation.atlas import prepare as P
    path, rows, unit_meta, s20l, d20l = args
    with zipfile.ZipFile(path) as z:                  # pass 1, as prepare.nt_archive, build.py and bands.py
        names = set(z.namelist())
        sessions = sorted({m[1] for n in names if (m := re.fullmatch(r'(.*/Session\d+)/Condition\.mat', n))})
        used = [s for s in sessions if P.nt_intervals(z.read(f'{s}/Condition.mat'))]
        channels = sorted(int(m[1]) for n in names if (m := re.fullmatch(re.escape(used[0]) + r'/ECoG_ch(\d+)\.mat', n)))
        assert channels == list(range(1, len(channels) + 1))
        good = [c - 1 for c in unit_meta['kept_channels']] + [unit_meta['dropped_after_car'] - 1]
        need = {r['session'] for r in rows}
        low = {s.rsplit('/', 1)[1]: np.column_stack([P.continuous(np.nan_to_num(P.mat_vector(z.read(f'{s}/ECoG_ch{ch}.mat'))[:, None]),
                                                                  P.NT_FS, P.NT_MAINS)[:, 0] for ch in channels])[:, good]
               for s in used if s.rsplit('/', 1)[1] in need}
    ref = lambda x: P.reference(x, 'car')
    out, gate = {a: {} for a in ('W50', 'W50L8', 'C10', 'C10L8')}, []
    S, D = np.load(s20l), np.load(d20l)
    for s in low:
        lp = P.lowpass(low[s], P.FS, CUT8)            # as build.py: P.lowpass(low[s], P.FS, CUT[20])
        for row in rows:
            if row['session'] != s:
                continue
            k = int(round(row['start_s'] * P.FS)); w, w8 = ref(low[s][k:k + L]), ref(lp[k:k + L])
            for x, b in ((w, S[row['id']]), (w8, D[row['id']])):
                gate.append(float(np.max(np.abs(x[::Q] - b)) / b.std()) if x[::Q].shape == b.shape else np.inf)
            out['W50'][row['id']], out['W50L8'][row['id']] = w, w8
            for j in range(CUTS):
                out['C10'][f"{row['id']}:c{j}"] = w[j * L // CUTS:(j + 1) * L // CUTS]
                out['C10L8'][f"{row['id']}:c{j}"] = w8[j * L // CUTS:(j + 1) * L // CUTS]
    S.close(); D.close()
    return Path(path).stem, gate, out


def build(banks, lowrate, archives, out, nw):
    from research.representation.atlas import prepare as P
    units = {u['unit']: u for u in json.loads((banks / 'neurotycho-long-stage-a' / 'manifest.json').read_text())['preprocessing']['units']}
    man = {g: json.loads((lowrate / 'S20L' / g / 'manifest.json').read_text()) for g in STAGES}
    rows_of, stage_of = {}, {}
    for g in STAGES:
        for r in man[g]['rows']:
            rows_of.setdefault(r['unit'], []).append(r); stage_of[r['unit']] = g
    jobs = [(archives / f'{u}.zip', rows_of[u], units[u], lowrate / 'S20L' / stage_of[u] / 'windows.npz',
             lowrate / 'D20L' / stage_of[u] / 'windows.npz') for u in sorted(rows_of)]
    with ProcessPoolExecutor(nw) as pool:
        res = list(pool.map(archive, jobs))
    gate = {u: max(g) for u, g, _ in res}
    out.mkdir(parents=True, exist_ok=True)
    (out / 'gate_a_long100.json').write_text(json.dumps(dict(max_rel_diff=gate, worst=max(gate.values())), indent=1) + '\n')
    print('gate A worst max|rebuilt[::5] - S20L/D20L| / SD:', max(gate.values()), flush=True)
    if max(gate.values()) > 1e-9:
        sys.exit('gate A failed: rebuilt 50-s windows differ from the low-rate long banks')
    sha = P.sha256(Path(__file__))
    for a in ('W50', 'W50L8', 'C10', 'C10L8'):
        arrays = {k: v for _, _, o in res for k, v in o[a].items()}
        T = L if a.startswith('W') else L // CUTS
        for g in STAGES:
            rows = man[g]['rows'] if a.startswith('W') else [
                dict(r, id=f"{r['id']}:c{j}", start_s=r['start_s'] + j * T / P.FS, parent=r['id'], cut=j)
                for r in man[g]['rows'] for j in range(CUTS)]
            bank = out / a / g; bank.mkdir(parents=True, exist_ok=True)
            assert all(arrays[r['id']].shape == (T, r['M']) for r in rows)
            np.savez(bank / 'windows.partial.npz', **{r['id']: np.asarray(arrays[r['id']], np.float64) for r in rows})
            (bank / 'windows.partial.npz').replace(bank / 'windows.npz')
            rec = dict(man[g]['preprocessing'], arm=a, rate_hz=P.FS, lowpass_hz=CUT8 if a.endswith('L8') else None,
                       window_s=T / P.FS, source_rows=f'S20L/{g} (= D20L/{g})', build_sha256=sha)
            (bank / 'manifest.json').write_text(json.dumps(dict(dataset='neurotycho', arm=a, fs=P.FS, T=T, rows=rows,
                                                                preprocessing=rec), indent=1) + '\n')
        print(a, {g: len(man[g]['rows']) * (1 if a.startswith('W') else CUTS) for g in STAGES}, flush=True)
    print('ALLDONE', flush=True)


M16, NB = 16, 4


def phase_rand(X, rng):                               # verbatim, ksgbias_calib.py (robustness branch)
    T = len(X); F = np.fft.rfft(X, axis=0); ph = np.exp(2j * np.pi * rng.random(F.shape)); ph[0] = 1
    if T % 2 == 0: ph[-1] = 1
    Y = np.fft.irfft(np.abs(F) * ph, T, axis=0); return (Y - Y.mean(0)) / Y.std(0)


def markov(T, dwell, rng):                            # verbatim
    flip = rng.random(T) < 1 / dwell; flip[0] = rng.random() < .5; return np.cumsum(flip) % 2


def gen(kind, src, rng):                              # verbatim (M -> M16), block and gain_shared branches only
    T = len(src); X = phase_rand(src[:, rng.permutation(src.shape[1])[:M16]], rng)
    L_ = np.eye(M16)
    for b in range(NB):
        idx = np.arange(b * 4, b * 4 + 4); C = np.full((4, 4), rng.uniform(.2, .9)); np.fill_diagonal(C, 1)
        C = C + np.diag(rng.uniform(0, .3, 4)); d = np.sqrt(np.diag(C))
        L_[np.ix_(idx, idx)] = np.linalg.cholesky(C / np.outer(d, d))
    X = X @ L_.T
    if kind == 'gain_shared': X = X * np.where(markov(T, T / 10, rng), 3., 1.)[:, None]
    return X


def calib_one(a):
    import edges as E
    from mtsspi.surrogates import multivariate_iaaft
    kind, seed, src = a; x = E.ns(gen(kind, src, np.random.default_rng(seed)))
    s = multivariate_iaaft(x, np.random.default_rng(seed + 1), final='spectrum')[0]
    return E.zc(*E.edge_mi(x)) - E.zc(*E.edge_mi(s))


def calib(d, out, nb, ng, nw):
    import edges as E
    from multiprocessing import Pool
    man = json.loads((d / 'manifest.json').read_text()); W = np.load(d / 'windows.npz'); src = {}
    for tag, ag, st in (('PF', 'PF', 'drug'), ('KTMD', 'KTMD', 'drug'), ('AW', 'KTMD', 'awake_ec')):
        rows = [r for r in man['rows'] if r['state'] in E.STATES and r['agent'] == ag]; S0 = E.sub32(rows[0]['M'])
        src[tag] = [E.ns(W[r['id']].astype(float)[:, S0]) for r in rows if r['state'] == st]
    jobs, lab = [], []
    for ti, T in enumerate((1000, L)):
        for si, tag in enumerate(('AW', 'KTMD', 'PF')):
            for ki, (kind, n) in enumerate((('block', nb), ('gain_shared', ng))):
                for i in range(n):                    # same seed and source at both T
                    jobs.append((kind, 8 * 10 ** 6 + 10 ** 5 * si + 10 ** 4 * ki + 2 * i, src[tag][i % len(src[tag])][:T]))
                    lab.append((kind, tag, T))
    with Pool(nw) as p:
        R = p.map(calib_one, jobs[::-1], chunksize=1)[::-1]   # T = 5000 first (longest)
    np.savez(out, dz=np.array(R), kind=np.array([k for k, _, _ in lab]), tag=np.array([t for _, t, _ in lab]),
             T=np.array([T for _, _, T in lab]), n_sources={k: len(v) for k, v in src.items()})
    print('done', out, flush=True)


def cscore(f):
    D = np.load(f, allow_pickle=True); dz, kind, tag, TT = D['dz'], D['kind'], D['tag'], D['T']; off = {}
    for T in (1000, L):
        m = {}
        for t in ('AW', 'KTMD', 'PF'):
            b = dz[(kind == 'block') & (tag == t) & (TT == T)]; g = dz[(kind == 'gain_shared') & (tag == t) & (TT == T)]
            m[t] = (b.mean(), b.std(ddof=1) / np.sqrt(len(b)))
            print(f'T {T} {t:4s} block {b.mean():+.4f} +- {m[t][1]:.4f} (n {len(b)}) | gain_shared {g.mean():+.4f} (n {len(g)}) '
                  f'| detectability {(b.mean() - g.mean()) / b.std(ddof=1):.2f}')
        off[T] = (np.mean([m['KTMD'][0], m['PF'][0]]) - m['AW'][0], np.sqrt((m['KTMD'][1] ** 2 + m['PF'][1] ** 2) / 4 + m['AW'][1] ** 2))
        print(f'T {T} stationary drug - awake offset {off[T][0]:+.4f} +- {off[T][1]:.4f}')
    print(f'offset(5000) - offset(1000) {off[L][0] - off[1000][0]:+.4f} +- {np.hypot(off[L][1], off[1000][1]):.4f} (independent-SE bound)')


def read(root):
    """Frozen pair readings (preregs/long100-prereg.md): R = pooled drug delta(first) / pooled drug delta(second) over
    animal means; paired animal-level first - second, t(3). Neutral .75-1.33; smaller if R < .75 and t >= +2.35; larger
    if R > 1.33 and t <= -2.35; otherwise inconclusive."""
    sys.path.insert(0, str(Path(__file__).parent))
    from summarise import load, animal_drug
    from scipy.stats import ttest_1samp
    pairs = (('W50', 'C10', 'duration, 0.5-40 Hz (primary)'), ('W50L8', 'C10L8', 'duration, 0.5-8 Hz'),
             ('D20L', 'W50L8', 'rate/N, 0.5-8 Hz, same 50-s windows'), ('S20L', 'W50', 'subsampled vs 100 Hz, same 50-s windows'))
    for a, b, what in pairs:
        if not ((root / a).is_dir() and (root / b).is_dir()):
            continue
        xa, xb = load(root / a), load(root / b)
        for v, vn in ((1, 'ns'), (0, 'raw')):
            p, q = animal_drug(xa, v), animal_drug(xb, v); k = sorted(set(p) & set(q))
            d = np.array([p[i] - q[i] for i in k]); R = np.mean([p[i] for i in k]) / np.mean([q[i] for i in k])
            t = ttest_1samp(d, 0).statistic
            call = ('neutral' if .75 <= R <= 1.33 else f'smaller in {a}' if R < .75 and t >= 2.35
                    else f'larger in {a}' if R > 1.33 and t <= -2.35 else 'inconclusive')
            print(f'{a:6s}/{b:6s} {vn:3s} R {R:.2f}; {a}-{b} per animal {", ".join(f"{i} {x:+.4f}" for i, x in zip(k, d))}; '
                  f't(3) {t:+.2f} -> {call}   [{what}]')


if __name__ == '__main__':
    if sys.argv[1] == 'build':
        build(*map(Path, sys.argv[2:6]), int(sys.argv[6]))
    elif sys.argv[1] == 'calib':
        calib(Path(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))
    elif sys.argv[1] == 'cscore':
        cscore(sys.argv[2])
    else:
        read(Path(sys.argv[2]))
