"""Post hoc (26 November), descriptive: is the heavy-tailed difference axis (k04) episodic coupling, shared across pairs, and what
is in the raw signal when it happens? Same rows, 32-channel subsets, normal scoring and draw-0 IAAFT seeds as edges.py recompute.

Per window, 1-s segments (step 0.5 s, 19 segments), Fisher z of r per pair and segment, for the recording and its draw-0 surrogate:
  cv     mean over pairs of the SD over segments of z (coupling variability; the surrogate keeps spectrum and linear coupling)
  glob   share of the pair-by-segment z variance (pair means removed) on its first principal component (do pairs move together)
  gc     time course of the mean z over pairs (global coupling), correlated over segments (Spearman) with raw-window segment
         log RMS, log delta (1-4 Hz) power and log high-band (25-45 Hz) power, averaged over the 32 channels; gcv_r, gcv_s:
         the same with the normal-scored window's own segment RMS, for the recording and for its surrogate (the null)
  k04    mean difference-axis excess kurtosis (kens.stats), and per window the Spearman over pairs of the pair's cv excess
         (real minus surrogate) with its k04
  python episodic.py <bank dir> <out.npz> <agent> <rowmode: first|row> <n workers>
  python episodic.py score <dir with ep_*.npz>
"""
import sys, json, numpy as np, warnings
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
warnings.filterwarnings('ignore')
SEG, STEP, FS = 100, 50, 100


def segz(x):
    """(S, E) Fisher z of r per segment and pair i < j."""
    T, M = x.shape; iu = np.triu_indices(M, 1); out = []
    for s in range(0, T - SEG + 1, STEP):
        out.append(np.arctanh(np.clip(np.corrcoef(x[s:s + SEG].T)[iu], -.999999, .999999)))
    return np.array(out)


def glob_share(z):
    z = z - z.mean(0); s = np.linalg.svd(z, compute_uv=False) ** 2; return s[0] / s.sum()


def bands(w):
    """(S, 3) per segment: log RMS, log 1-4 Hz and log 25-45 Hz power, channel-averaged, on the raw (unscored) window."""
    f = np.fft.rfftfreq(SEG, 1 / FS); out = []
    for s in range(0, len(w) - SEG + 1, STEP):
        y = w[s:s + SEG] - w[s:s + SEG].mean(0); p = np.abs(np.fft.rfft(y * np.hanning(SEG)[:, None], axis=0)) ** 2
        out.append([np.log(np.sqrt(np.mean(y ** 2))), np.log(p[(f >= 1) & (f <= 4)].sum(0)).mean(), np.log(p[(f >= 25) & (f <= 45)].sum(0)).mean()])
    return np.array(out)


def one(args):
    from scipy.stats import spearmanr
    from edges import ns
    from kens import stats
    from mtsspi.surrogates import multivariate_iaaft
    w, seed = args; v = ns(w); s = multivariate_iaaft(v, np.random.default_rng(seed), final='spectrum')[0]
    zr, zs = segz(v), segz(s); cvr, cvs = zr.std(0), zs.std(0); k04 = stats(v)[0][3]; b = bands(w); gc = zr.mean(1)
    rb = [spearmanr(gc, b[:, j])[0] for j in range(3)]
    lr = lambda y: np.log([np.sqrt(np.mean(y[i:i + SEG] ** 2)) for i in range(0, len(y) - SEG + 1, STEP)])  # scored-scale RMS
    rv, rs = spearmanr(gc, lr(v))[0], spearmanr(zs.mean(1), lr(s))[0]
    return np.array([cvr.mean(), cvs.mean(), glob_share(zr), glob_share(zs), *rb, k04.mean(), spearmanr(cvr - cvs, k04)[0], rv, rs], np.float32)


COLS = ('cv_r', 'cv_s', 'glob_r', 'glob_s', 'gc_rms', 'gc_delta', 'gc_high', 'k04', 'rho_cv_k04', 'gcv_r', 'gcv_s')


def score(d):
    import pandas as pd
    fr = []
    for p in sorted(Path(d).glob('ep_*.npz')):
        with np.load(p) as a:
            f = pd.DataFrame(a['X'], columns=COLS)
            for k in ('id', 'unit', 'state', 'agent', 'animal'): f[k] = a[k]
            fr.append(f)
    df = pd.concat(fr, ignore_index=True); assert df.id.is_unique
    df['group'] = np.where(df.state == 'drug', df.agent, np.where(df.state == 'sleep', 'SLP', df.state))
    df['cv_x'], df['glob_x'], df['gcv_x'] = df.cv_r - df.cv_s, df.glob_r - df.glob_s, df.gcv_r - df.gcv_s
    pd.set_option('display.width', 200)
    print(df.groupby(['animal', 'group'])[['cv_x', 'glob_x', 'gc_rms', 'gc_delta', 'gc_high', 'gcv_r', 'gcv_s', 'gcv_x', 'k04', 'rho_cv_k04']].mean().round(3).to_string())
    print('\nall animals:'); print(df.groupby('group')[['cv_x', 'glob_x', 'gc_rms', 'gc_delta', 'gc_high', 'gcv_r', 'gcv_s', 'gcv_x', 'k04', 'rho_cv_k04']].mean().round(3).to_string())
    df.to_csv(Path(d) / 'ep-windows.csv.gz', index=False)


if __name__ == '__main__':
    if sys.argv[1] == 'score':
        score(sys.argv[2]); sys.exit()
    from multiprocessing import Pool
    from edges import STATES, sub32, meta
    d, out, agent, rowmode, nw = sys.argv[1:6]
    man = json.load(open(f'{d}/manifest.json')); W = np.load(f'{d}/windows.npz')
    rows = [r for r in man['rows'] if r['state'] in STATES and r['agent'] in agent.split(',')]
    S0 = sub32(rows[0]['M']); sub = lambda r: S0 if rowmode == 'first' else sub32(r['M'])
    jobs = [(W[r['id']].astype(float)[:, sub(r)], k) for k, r in enumerate(rows)]
    with Pool(int(nw)) as p: res = p.map(one, jobs, chunksize=1)
    m, _ = meta(rows, man); np.savez(out, X=np.array(res), **m); print('done', out, flush=True)
