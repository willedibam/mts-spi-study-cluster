"""Known-answer calibration of the gain-over-floor null (nulls_env.py) for z(Gaussian MI, KSG MI); rule in
../preregs/envelope-prereg.md, fixed before this script's z outputs existed.

Pipeline as the anchor: 32 channels, T = 1000, real window normal-scored, edges.edge_mi, one draw per null; residual = z(real) - z(null).
Generators (code pinned to existing calibrations):
  nonstat/calib.synth: linear, burst, burst_slow, gain_cmfloor, gain_local, switch, decouple, quadratic, quad_weak
  reorder.py (window2 lines): gain_homog, gain_hetero (shared slow gain, heterogeneous white floor .05-2), pair_nonmono
  new here: linear_hetspec (sources and floors with heterogeneous AR(2) spectra), burst_quad and burst_decouple (the burst gain
            of nonstat 'burst' applied on top of 'quadratic' or 'decouple': additivity check)
Nulls: N0 (joint IAAFT, the anchor), GOF variants (sigma x floor), GOF 'decor' arm of each, N1 (nonstat n1) at sigma 25 and 100.

  python envcalib.py <phase: dev|confirm> <n windows> <n workers> <out.npz> [variant for confirm, e.g. 40-cm]
"""
import sys, time, importlib.util, numpy as np, warnings
from pathlib import Path
from multiprocessing import Pool
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parents[4] / 'src')]
from edges import edge_mi, zc
from mtsspi.surrogates import multivariate_iaaft
from nulls_env import ns, gof
warnings.filterwarnings('ignore')
_spec = importlib.util.spec_from_file_location('nscalib', HERE.parent / 'nonstat' / 'calib.py'); NC = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(NC)
from nonstat.nulls import n1  # noqa: E402  (namespace import from mi_pair)
import reorder as RO  # noqa: E402
M, T = 32, 1000
GAIN = ('gain_homog', 'gain_hetero', 'gain_cmfloor', 'burst', 'burst_slow', 'gain_local')
COUPLING = ('quadratic', 'decouple', 'pair_nonmono')
OTHER = ('linear', 'linear_hetspec', 'burst_quad', 'burst_decouple', 'switch', 'quad_weak')
KINDS = ('linear', 'linear_hetspec') + GAIN + COUPLING + OTHER[2:]
VARIANTS = tuple(f'{s}-{f}' for s in (15, 40) for f in ('diag', 'cm'))
SEED0 = dict(dev=50000, confirm=90000)


def ar2(rng, w, r=.9):
    return NC.ar(T, (2 * r * np.cos(w), -r * r), rng)


def synth(kind, rng):
    if kind in NC.KINDS: return NC.synth(kind, rng)
    if kind in ('gain_homog', 'gain_hetero', 'pair_nonmono'):  # reorder.window2, one window
        L = rng.standard_normal((M, 4)) * rng.uniform(.2, 1.2, (M, 1)); fl = rng.uniform(.05, 2., M)
        src = rng.choice(M, 8, replace=False); dst = rng.choice(np.setdiff1d(np.arange(M), src), 8, replace=False)
        u = RO.ar(rng, 4, .9) @ L.T + RO.ar(rng, M, .9); u = u / u.std(0)
        if kind == 'gain_homog': return RO.gain(rng)[:, None] * u + .3 * rng.standard_normal((T, M))
        if kind == 'gain_hetero': return RO.gain(rng)[:, None] * u + fl * rng.standard_normal((T, M))
        x = u + .3 * rng.standard_normal((T, M))
        for s_, d_ in zip(src, dst): x[:, d_] = x[:, d_] + 1.2 * (x[:, s_] ** 2 - 1)
        return x
    if kind == 'linear_hetspec':
        s = np.column_stack([ar2(rng, rng.uniform(.05, .4) * np.pi) for _ in range(4)]); A = rng.normal(size=(M, 4)) * rng.uniform(.3, 1.5, (M, 1))
        sh = s @ A.T; sh /= sh.std(0); return sh + .6 * np.column_stack([ar2(rng, rng.uniform(.05, .4) * np.pi) for _ in range(M)])
    if kind in ('burst_quad', 'burst_decouple'):  # nonstat 'burst' gain on top of the coupling generator
        x = NC.synth('quadratic' if kind == 'burst_quad' else 'decouple', rng)
        g = np.where(NC.episodes(T, .4, 80, rng)[:T], 1., .15)[:, None]; return g * x + .15 * rng.standard_normal((T, M))
    raise ValueError(kind)


def nulls(phase, variant):
    if phase == 'dev': return ['N0'] + [f'GOF {v}' for v in VARIANTS] + [f'DECOR {v}' for v in VARIANTS] + ['N1 s25', 'N1 s100']
    return ['N0', f'GOF {variant}', f'DECOR {variant}']


def one(args):
    kind, seed, names = args; rng = np.random.default_rng(seed); x = synth(kind, rng); v = ns(x); zr = zc(*edge_mi(v)); out = []
    for k, nm in enumerate(names):
        r = np.random.default_rng([seed, k])
        if nm == 'N0': s = multivariate_iaaft(v, r, final='spectrum')[0]
        elif nm.startswith('N1'): s = n1(v, r, int(nm[4:]))
        else:
            sg, fl = nm.split()[1].split('-'); s = gof(x, r, int(sg), fl, 'keep' if nm.startswith('GOF') else 'decor')
        out.append(zr - zc(*edge_mi(s)))
    return out


if __name__ == '__main__':
    phase, B, nw, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]; variant = sys.argv[5] if len(sys.argv) > 5 else None
    names = nulls(phase, variant); t = time.time(); R = {}
    jobs = [(k, SEED0[phase] + 1000 * i + s, names) for i, k in enumerate(KINDS) for s in range(B)]
    with Pool(nw) as p: res = p.map(one, jobs, chunksize=1)
    for i, k in enumerate(KINDS):
        R[k] = np.array(res[i * B:(i + 1) * B]); m, sd = R[k].mean(0), R[k].std(0, ddof=1)
        print(f'{k:15s} ' + '  '.join(f'{n} {a:+.4f} (t {a / b * np.sqrt(B):+.1f})' for n, a, b in zip(names, m, sd)), flush=True)
    print(f'[{time.time() - t:.0f} s]'); np.savez(out, nulls=names, kinds=KINDS, **R)
