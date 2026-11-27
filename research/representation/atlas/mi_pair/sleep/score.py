"""Per-window MI pair (real and joint IAAFT) and covariates for the human-sleep test (rules: ../preregs/sleep-prereg.md).

    python score.py run <root> <out dir> <states> <shard i> <n shards> <n workers> [draws]
        states: comma list of task:acq, e.g. wake:1min,sleepN2:10min,sleepN3:10min (wake:1min alone = baseline-only run)
    python score.py plan <root>          inclusion table from metadata only (no samples read)

Per window, on the subject's fixed channel set (extract.channel_set): normal-score each channel; Gaussian MI from r and KSG MI
(k = 4, AUTO Theiler) for every pair (../edges.edge_mi); z = Pearson r of the two profiles; the same on `draws` spectrum-final
joint IAAFT draws (default 3) of the normal-scored window. Stored per window: z real and per draw, the real and per-draw (G, K)
profiles, strength covariates on the raw window (mean |r|, SD |r|, covariance participation ratio), burst index, delta
fraction, and the conventional metrics of bench_metrics.one. Out: <out dir>/<sub>.npz.
"""
import sys, os, json, numpy as np, pandas as pd, warnings
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, '..'), os.path.join(HERE, '../../../../..')]
from extract import windows, channel_set, segments, seed, T
from edges import edge_mi, ns, zc
from mtsspi.surrogates import multivariate_iaaft
from research.representation.atlas.prepare import delta_fraction
import bench_metrics
warnings.filterwarnings('ignore')
STATES = [('wake', '1min'), ('sleepN2', '10min'), ('sleepN3', '10min')]  # channel set: common to these three
M_MIN, WAKE_MIN = 8, 4


def plan(root):
    """Inclusion from metadata only: channels common to the three recordings, whole 10-s windows per state."""
    out = []
    for sub in sorted(d for d in os.listdir(root) if d.startswith('sub-')):
        try: ch = channel_set(root, sub, STATES)
        except FileNotFoundError: continue
        nw = {f'{t}:{a}': sum(n // (T * 2) for _, n in segments(root, sub, t, a)) for t, a in STATES + [('sleepN2', '1min'), ('sleepN3', '1min')]}
        out.append(dict(sub=sub, M=len(ch), **nw, included=len(ch) >= M_MIN and nw['wake:1min'] >= WAKE_MIN))
    return pd.DataFrame(out)


def covariates(w):
    C = np.corrcoef(w.T); iu = np.triu_indices(len(C), 1); a = np.abs(C[iu]); ev = np.linalg.eigvalsh(np.cov(w.T))
    return [a.mean(), a.std(), ev.sum() ** 2 / (ev ** 2).sum()]


def one(args):
    w, sd, nd = args; w = w.astype(float); v = ns(w); g, k = edge_mi(v); P = [g, k]; z = [zc(g, k)]
    for d in range(nd):
        gs, ks = edge_mi(multivariate_iaaft(v, np.random.default_rng(sd + d), final='spectrum')[0]); P += [gs, ks]; z.append(zc(gs, ks))
    b, rms = bench_metrics.one(w)
    return np.array(z), np.array(P, np.float32), np.array(covariates(w) + [delta_fraction(w)]), b, rms


def run(root, out, states, i, n, nw, nd=3):
    os.makedirs(out, exist_ok=True); pl = plan(root); subs = list(pl[pl.included]['sub'])[i::n]
    for sub in subs:
        fn = f'{out}/{sub}.npz'
        if os.path.exists(fn): continue
        ch = channel_set(root, sub, STATES); jobs, rows = [], []
        for t, a in states:
            try: ws = windows(root, sub, t, a, ch)
            except (ValueError, FileNotFoundError): print('skip', sub, t, a, 'channel set absent', flush=True); continue
            for q, (w, info) in enumerate(ws):
                jobs.append((w, seed(sub, t, a, q), nd)); rows.append(dict(info, state=f'{t}:{a}', index=q))
        with Pool(nw) as p: res = p.map(one, jobs, chunksize=1)
        np.savez(fn + '.part.npz', Z=np.array([r[0] for r in res]), P=np.array([r[1] for r in res]), C=np.array([r[2] for r in res]),
                 B=np.array([r[3] for r in res]), RMS=np.array([r[4] for r in res]), rows=json.dumps(rows), chans=json.dumps(ch),
                 cols=json.dumps(list(bench_metrics.COLS)))
        os.replace(fn + '.part.npz', fn); print('done', sub, len(rows), len(ch), flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'plan':
        pl = plan(sys.argv[2]); print(pl.to_string()); print('included', int(pl.included.sum())); sys.exit()
    root, out, st = sys.argv[2], sys.argv[3], [tuple(s.split(':')) for s in sys.argv[4].split(',')]
    run(root, out, st, int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]), *(int(x) for x in sys.argv[8:9]))
