"""Known-answer calibration of the cross-cumulant decomposition (stats.py) against the MI pair, on the segment calibration's
generators (../nonstat/calib.py: 32 channels, T = 1000, normal-scored, one spectrum-final joint IAAFT draw, edges.edge_mi) plus
  updown   shared two-state regime (DOWN 40% of the time, mean dwell .4 s): the shared part shrinks to .3 and shifts by a
           per-channel offset of +-1 (slow-wave-like mean-shifting regimes, which make odd cumulants non-zero).
Per window, real minus surrogate ("net"):
  zGK      the anchor z(G, K)
  zJ, zJ3, zJ4   z(G, G + J) with J = J3 + J4, J3, J4: does the cumulant estimate of I reproduce the anchor's organisation?
  zP       partial z(G, K | J3, J4): the anchor with the cumulant terms partialled out of both profiles
  link3, link4   across-edge r of the K excess (K - K_surr) with the J3 and J4 excess
  L3, L4, Lenv   mean edge excess of J3, J4 and env (p - |gamma|^2); zenv = z(|gamma|^2, p)
  python calib.py <n seeds> <n workers> <out.npz>
"""
import sys, time, numpy as np, warnings
from pathlib import Path
from joblib import Parallel, delayed
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'nonstat'), str(HERE.parent), str(HERE.parents[4] / 'src')]
from edges import edge_mi, zc
from nulls import ns, n0
import importlib.util as _u
_sp = _u.spec_from_file_location('nonstat_calib', HERE.parent / 'nonstat' / 'calib.py'); nc = _u.module_from_spec(_sp); _sp.loader.exec_module(nc)
from stats import edge_cum
warnings.filterwarnings('ignore')
KINDS = nc.KINDS + ('updown',)
KEYS = ('zGK', 'zJ', 'zJ3', 'zJ4', 'zP', 'link3', 'link4', 'L3', 'L4', 'Lenv', 'zenv')


def synth(kind, rng):
    if kind != 'updown': return nc.synth(kind, rng)
    T, M, K = nc.T, nc.M, nc.K
    s = np.column_stack([nc.ar(T, nc.PH, rng) for _ in range(K)]); A = rng.normal(size=(M, K)) * rng.uniform(.3, 1.5, (M, 1))
    sh = s @ A.T; sh /= sh.std(0); fl = np.column_stack([nc.ar(T, nc.PH, rng) for _ in range(M)])
    down = nc.episodes(T, .4, 40, rng)[:T, None]; off = rng.choice([-1., 1.], M)
    return np.where(down, .3 * sh + off, sh) + .6 * fl


def pz(g, k, C):
    X = np.column_stack([np.ones(len(g)), C]); res = lambda y: y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return zc(res(g), res(k))


def readouts(v):
    g, k = edge_mi(v); c = edge_cum(v); J = c['J3'] + c['J4']
    return dict(g=g, k=k, c=c, zGK=zc(g, k), zJ=zc(g, g + J), zJ3=zc(g, g + c['J3']), zJ4=zc(g, g + c['J4']),
                zP=pz(g, k, np.column_stack([c['J3'], c['J4']])), zenv=zc(c['g2'], c['p']))


def net(R, S):
    out = {key: R[key] - S[key] for key in ('zGK', 'zJ', 'zJ3', 'zJ4', 'zP', 'zenv')}
    ek = R['k'] - S['k']
    out['link3'] = zc(ek, R['c']['J3'] - S['c']['J3']); out['link4'] = zc(ek, R['c']['J4'] - S['c']['J4'])
    out.update(L3=np.mean(R['c']['J3'] - S['c']['J3']), L4=np.mean(R['c']['J4'] - S['c']['J4']), Lenv=np.mean(R['c']['env'] - S['c']['env']))
    return [out[key] for key in KEYS]


def one(kind, seed):
    rng = np.random.default_rng(seed); v = ns(synth(kind, rng)); s = n0(v, np.random.default_rng([seed, 0]))
    return net(readouts(v), readouts(s))


if __name__ == '__main__':
    B, nw, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]; t = time.time(); R = {}
    print('kind          ' + ' '.join(f'{k:>14s}' for k in KEYS))
    for kind in KINDS:
        R[kind] = np.array(Parallel(n_jobs=nw)(delayed(one)(kind, 1000 * (KINDS.index(kind) + 1) + s) for s in range(B)))
        m, sd = R[kind].mean(0), R[kind].std(0, ddof=1)
        print(f'{kind:13s} ' + ' '.join(f'{a:+.4f} ({a / b * np.sqrt(B):+5.1f})' for a, b in zip(m, sd)), f'[{time.time() - t:.0f} s]', flush=True)
    np.savez(out, keys=KEYS, **R)
