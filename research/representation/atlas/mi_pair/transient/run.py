"""Transient cut on the macaque 10-s MI-pair windows (rule: ../preregs/transient-prereg.md).

Same rows, 32-channel subsets, normal-scoring and seeds as nonstat/run.py, so N0 is the anchor's own IAAFT draw
(reproduction gate against the stored zns z). Per window: z(Gaussian MI, KSG k=4) of the real window and its N0
surrogate; then the same after cutting the window's transients (CUT), and after cutting the same samples circularly
shifted to a random offset (SHIFT, the control: same number of samples and splice points, unrelated to the events).
After a cut the kept samples are concatenated, normal-scored again and given their own IAAFT draw (seed [row, 1|2]).

  python run.py <bank dir> <out.npz> <agent> <rowmode: first|row> <n workers>
  python run.py score <dir with tr_[AB]_*.npz>
"""
import sys, glob, json, numpy as np, warnings
from pathlib import Path
from multiprocessing import Pool
from scipy.stats import ttest_1samp
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent)]
from edges import edge_mi, zc, sub32, meta, ns, STATES
from mtsspi.surrogates import multivariate_iaaft
warnings.filterwarnings('ignore')
THR, PAD, MIN_KEEP = 5.0, 20, 300                  # robust SDs, samples either side (0.2 s), samples left to score


def transients(x):
    """(T,) True on samples within PAD of a sample beyond THR robust SDs (median, 1.4826 MAD) in any channel."""
    c = np.median(x, 0); s = 1.4826 * np.median(np.abs(x - c), 0)
    hit = (np.abs(x - c) > THR * s).any(1)
    return np.convolve(hit, np.ones(2 * PAD + 1), 'same') > 0


def pair_z(v, rng):
    return zc(*edge_mi(v)), zc(*edge_mi(multivariate_iaaft(v, rng, final='spectrum')[0]))


def one(args):
    x, seed = args; cut = transients(x)
    out = [*pair_z(ns(x), np.random.default_rng(seed)), cut.mean()]           # = edges.one_recompute, ns branch
    shift = np.roll(cut, np.random.default_rng([seed, 0]).integers(len(cut)))
    for b, m in ((1, cut), (2, shift)):
        out += list(pair_z(ns(x[~m]), np.random.default_rng([seed, b]))) if (~m).sum() >= MIN_KEEP else [np.nan, np.nan]
    return out                                     # z real, z N0, cut fraction, CUT real, CUT null, SHIFT real, SHIFT null


def score(d):
    F = sorted(glob.glob(f'{d}/tr_[AB]_*.npz')); Q = [np.load(f) for f in F]; cat = lambda k: np.concatenate([q[k] for q in Q])
    Z = cat('Z'); Vz = cat('Vzns'); unit, state, agent, animal = cat('unit'), cat('state'), cat('agent'), cat('animal')
    gap = np.nanmax(np.abs(Z[:, :2] - Vz)); print(f'reproduction gate: max |(real, N0) z - stored| = {gap:.2e} ->', 'PASS' if gap < 1e-4 else 'FAIL')
    if gap >= 1e-4: return
    frac = Z[:, 2]; sl = lambda s: frac[(state == s) | ((state == 'drug') & (agent == s))]
    print('cut fraction, mean (share of windows with < %d samples left):' % MIN_KEEP)
    for s in ('awake_ec', 'awake_eo', 'sleep', 'KT', 'KTMD', 'MD', 'PF'):
        f = sl(s); print(f'  {s:9s} {f.mean():.3f}  ({np.mean(f > 1 - MIN_KEEP / 1000):.2f})')
    out = {}
    for name, (a, b) in (('N0', (0, 1)), ('CUT', (3, 4)), ('SHIFT', (5, 6))):
        res = Z[:, a] - Z[:, b]; S = []
        for u in np.unique(unit):
            m = lambda s: (unit == u) & (state == s) & np.isfinite(res); ag = agent[unit == u][0]; s = 'sleep' if ag == 'SLP' else 'drug'
            if m(s).any() and m('awake_ec').any():
                S.append((ag, animal[unit == u][0], res[m(s)].mean() - res[m('awake_ec')].mean(),
                          res[m('awake_eo')].mean() - res[m('awake_ec')].mean() if m('awake_eo').any() else np.nan, u))
        ag, an = np.array([q[0] for q in S]), np.array([q[1] for q in S]); dz, eo = np.array([q[2] for q in S]), np.array([q[3] for q in S])
        dr = ag != 'SLP'; A = np.array([dz[dr & (an == x)].mean() for x in np.unique(an[dr])]); p = ttest_1samp(A, 0, alternative='less').pvalue
        agm = {g: (dz[ag == g].mean(), int(np.sum(dz[ag == g] < 0)), int(np.sum(ag == g))) for g in ('KT', 'KTMD', 'MD', 'PF', 'SLP')}
        ses = {q[4]: q[2] for q in S}
        out[name] = dict(pooled=float(dz[dr].mean()), animals=dict(zip(np.unique(an[dr]).tolist(), A.tolist())), neg=int((A < 0).sum()),
                         p=float(p), agents=agm, eo_sd=float(np.nanstd(eo, ddof=1)), sessions=ses)
        print(f'{name:6s} pooled drug Δ {dz[dr].mean():+.4f} | animal means {np.round(A, 4)} ({(A < 0).sum()}/4 < 0, one-sided t p = {p:.4f}) | eyes-open SD {np.nanstd(eo, ddof=1):.4f}')
        print('       ' + '  '.join(f'{g} {v[0]:+.4f} ({v[1]}/{v[2]} down)' for g, v in agm.items()))
    R = {k: out[k]['pooled'] / out['N0']['pooled'] for k in ('CUT', 'SHIFT')}
    print(f"\nretained: CUT {R['CUT']:.2f}  SHIFT {R['SHIFT']:.2f}  (pooled drug Δ over N0's)")
    for u in sorted(out['N0']['sessions']):
        if 'ktmd' in u and 'chibi' in u:
            print(f"  {u}: N0 {out['N0']['sessions'][u]:+.4f}  CUT {out['CUT']['sessions'].get(u, np.nan):+.4f}  SHIFT {out['SHIFT']['sessions'].get(u, np.nan):+.4f}")
    Path(d, 'transient-score.json').write_text(json.dumps(dict(retained=R, nulls=out), indent=1, default=float) + '\n')


if __name__ == '__main__':
    if sys.argv[1] == 'score': score(sys.argv[2]); sys.exit()
    d, out, agent, rowmode, nw = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
    man = json.load(open(f'{d}/manifest.json')); Wn = np.load(f'{d}/windows.npz')
    rows = [r for r in man['rows'] if r['state'] in STATES and r['agent'] in agent.split(',')]
    S0 = sub32(rows[0]['M']); sub = lambda r: S0 if rowmode == 'first' else sub32(r['M'])
    with Pool(nw) as p: Z = p.map(one, [(Wn[r['id']].astype(float)[:, sub(r)], k) for k, r in enumerate(rows)], chunksize=1)
    m, _ = meta(rows, man); zns = out.replace('/transient/tr_', '/zns/zns_')
    np.savez(out, Z=np.array(Z), Vzns=np.load(zns)['V'][:, 1], **m)
    print('done', out, flush=True)
