"""z(Gaussian MI, kNN MI), real and 3 joint-IAAFT draws, per selected window (rules: ../preregs/human-prereg.md).

    python score.py <windows dir> <out.npz> <n workers> [all<draws>]   (e.g. all10: every good channel, 10 draws; prereg full-data row)
    python score.py summary <out.npz> <prios|ds004541>

Variants: car = CAR over good channels, last dropped, then the evenly spaced 32 (primary); asis = recorded reference, same
channel indices. Out Z[window, variant, (real, surr1, surr2, surr3)].
"""
import sys, os, glob, json, zlib, numpy as np, warnings
from multiprocessing import Pool
from scipy.stats import ttest_1samp
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..')))
from edges import edge_mi, ns, zc, sub32
from mtsspi.surrogates import multivariate_iaaft
warnings.filterwarnings('ignore')


def channels(m, every=False):  # indices into the CAR-dropped good channels (m - 1 of them)
    return np.arange(m - 1) if every or m - 1 <= 32 else sub32(m - 1)


def one(args):
    """Stimulated electrodes (PRIOS rows' 'stim', Deviation 1) leave the CAR and the analysed set; none: as frozen."""
    w, seed, stim, nd, every = args; w = w.astype(float); keep = np.isin(np.arange(w.shape[1]), stim, invert=True)
    idx = [j for j in channels(w.shape[1], every) if keep[j]]; out = []
    for v in ((w - w[:, keep].mean(1, keepdims=True))[:, idx], w[:, idx]):
        v = ns(v); z = [zc(*edge_mi(v))]
        z += [zc(*edge_mi(multivariate_iaaft(v, np.random.default_rng(seed + d), final='spectrum')[0])) for d in range(nd)]
        out.append(z)
    return out


def run(src, out, nw, opt=''):
    nd, every = int(opt[3:] or 3) if opt.startswith('all') else 3, opt.startswith('all')
    jobs, rows = [], []
    for fn in sorted(glob.glob(f'{src}/sub-*.npz')):
        Z = np.load(fn); s = os.path.basename(fn)[:-4]
        for i, r in enumerate(json.loads(str(Z['rows']))):
            rows.append(dict(r, subject=s)); jobs.append((Z['W'][i], zlib.crc32(f'{s}:{i}'.encode()) * 10, r.get('stim', []), nd, every))
    with Pool(nw) as p: res = p.map(one, jobs, chunksize=1)
    np.savez(out, Z=np.array(res), rows=json.dumps(rows)); print('done', out, len(rows), flush=True)


def summary(fn, ds):
    d = np.load(fn); Z = d['Z']; rows = json.loads(str(d['rows'])); subj = np.array([r['subject'] for r in rows])
    state = np.array([r['state'] for r in rows]); res = Z[:, :, 0] - Z[:, :, 1:].mean(2); need = 20 if ds == 'prios' else 10
    keep = [s for s in np.unique(subj) if min(((subj == s) & (state == st)).sum() for st in ('awake', 'anaes')) >= need]
    print(f'{ds}: included {len(keep)}/{len(np.unique(subj))} subjects', {s: {st: int(((subj == s) & (state == st)).sum()) for st in ('awake', 'anaes')} for s in np.unique(subj)})
    for v, name in enumerate(('car (primary)', 'asis')):
        D = np.array([res[(subj == s) & (state == 'anaes'), v].mean() - res[(subj == s) & (state == 'awake'), v].mean() for s in keep])
        half = []
        for s in keep:
            a = res[(subj == s) & (state == 'awake'), v]; h = len(a) // 2; half.append(a[:h].mean() - a[h:2 * h].mean())
        sd = np.std(half, ddof=1); tl, tg = ttest_1samp(D, 0, alternative='less').pvalue, ttest_1samp(D, 0, alternative='greater').pvalue
        se = D.std(ddof=1) / np.sqrt(len(D)); from scipy.stats import t as tdist; q = tdist.ppf(.975, len(D) - 1)
        z0 = {st: np.mean([Z[(subj == s) & (state == st), v, 0].mean() for s in keep]) for st in ('awake', 'anaes')}
        print(f'\n[{name}] subject Δ {np.round(D, 4)}; mean {D.mean():+.4f} [{D.mean() - q * se:+.4f}, {D.mean() + q * se:+.4f}]; '
              f'{(D < 0).sum()}/{len(D)} negative; one-sided p(less) {tl:.4f}, p(greater) {tg:.4f}; null SD {sd:.4f} -> {D.mean() / sd:+.1f} SD; '
              f'real z awake {z0["awake"]:.3f}, anaes {z0["anaes"]:.3f}')
        if v == 0:
            rep = D.mean() < 0 and tl < .05 and (D > 0).sum() <= 1; opp = D.mean() > 0 and tg < .05
            print('VERDICT:', 'REPLICATES' if rep else 'OPPOSITE' if opp else 'NOT DETECTED')
    bi = np.array([r['burst'] for r in rows]); df = np.array([r['delta_fraction'] for r in rows])
    for s in keep:
        m = lambda st: (subj == s) & (state == st)
        print(f'  {s}: burst awake {np.median(bi[m("awake")]):.2f} anaes {np.median(bi[m("anaes")]):.2f}; delta awake {np.median(df[m("awake")]):.2f} anaes {np.median(df[m("anaes")]):.2f}')
    bins = np.floor(bi / .05); Dm = []
    for s in keep:
        cs = [b for b in np.unique(bins[subj == s]) if all(((subj == s) & (state == st) & (bins == b)).sum() >= 3 for st in ('awake', 'anaes'))]
        if cs: Dm.append(np.mean([res[(subj == s) & (state == 'anaes') & (bins == b), 0].mean() - res[(subj == s) & (state == 'awake') & (bins == b), 0].mean() for b in cs]))
    print(f'matched burst index (descriptive): {len(Dm)} subjects, mean {np.mean(Dm) if Dm else np.nan:+.4f}, {np.sum(np.array(Dm) < 0)}/{len(Dm)} negative')
    if ds == 'prios':
        late = np.array([r.get('delay_s', 0) >= 3.0 for r in rows]); Dl = []
        for s in keep:
            a, b = (subj == s) & (state == 'awake') & late, (subj == s) & (state == 'anaes') & late
            if a.sum() >= 5 and b.sum() >= 5: Dl.append(res[b, 0].mean() - res[a, 0].mean())
        print(f'epochs starting ≥ 3 s after a pulse (descriptive): {len(Dl)} subjects, mean {np.mean(Dl) if Dl else np.nan:+.4f}, {np.sum(np.array(Dl) < 0)}/{len(Dl)} negative')


if __name__ == '__main__':
    summary(sys.argv[2], sys.argv[3]) if sys.argv[1] == 'summary' else run(sys.argv[1], sys.argv[2], int(sys.argv[3]), *sys.argv[4:5])
