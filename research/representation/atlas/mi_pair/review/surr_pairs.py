import sys, json, numpy as np, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '/Users/wedi0306/Code/mts-spi-study-cluster/.claude/worktrees/real-world-apps-9b53a1/src')
from mtsspi.surrogates import multivariate_iaaft
from scipy.special import digamma
from scipy.signal import hilbert, csd, welch
from joblib import Parallel, delayed
K_NN = 4
def acf(x):
    x = x - x.mean(); n = len(x); f = np.fft.rfft(x, 2 * n); a = np.fft.irfft(f * np.conj(f))[:n]; return a / a[0]
def ksg(x, y, w):
    n = len(x); dx = np.abs(x[:, None] - x[None, :]); dy = np.abs(y[:, None] - y[None, :]); d = np.maximum(dx, dy)
    t = np.arange(n); ex = np.abs(t[:, None] - t[None, :]) <= w
    d = np.where(ex, np.inf, d); eps = np.sort(d, 1)[:, K_NN - 1]
    nx = np.sum(np.where(ex, False, dx < eps[:, None]), 1); ny = np.sum(np.where(ex, False, dy < eps[:, None]), 1)
    return digamma(K_NN) + digamma(n) - np.mean(digamma(nx + 1) + digamma(ny + 1))
def pec(a, orth):
    M = a.shape[1]; env = np.abs(a); out = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            if i == j: continue
            if orth:
                yo = np.abs((a[:, j] * np.conj(a[:, i]) / np.abs(a[:, i])).imag)
                out[i, j] = np.corrcoef(env[:, i], yo)[0, 1]
            else: out[i, j] = np.corrcoef(env[:, i], env[:, j])[0, 1]
    return (out + out.T) / 2 if orth else out
def stats(x, fs):
    x = (x - x.mean(0)) / x.std(0); T, M = x.shape; iu = np.triu_indices(M, 1)
    r = np.corrcoef(x.T)[iu]; mig = -0.5 * np.log(1 - r ** 2)
    A = [acf(x[:, m]) for m in range(M)]
    mik = np.array([ksg(x[:, i], x[:, j], min(int(2 * np.dot(A[i], A[j])), T // 10)) for i, j in zip(*iu)])
    an = hilbert(x, axis=0); p0 = pec(an, False)[iu]; p1 = pec(an, True)[iu]
    f, S = csd(x[:, iu[0]], x[:, iu[1]], fs=fs, nperseg=50, axis=0); _, Pi = welch(x[:, iu[0]], fs=fs, nperseg=50, axis=0); _, Pj = welch(x[:, iu[1]], fs=fs, nperseg=50, axis=0)
    C = S / np.sqrt(Pi * Pj); band = (f > 0) & (f < fs / 2)
    coh, icoh = np.abs(C[band]).mean(0), np.abs(C[band].imag).mean(0)
    c = lambda a, b: np.corrcoef(a, b)[0, 1]
    return dict(gauss_vs_ksg=c(mig, mik), envelope_orth=c(p0, p1), coh_vs_icoh=c(coh, icoh), sd_abs_r=np.abs(r).std(), mean_abs_r=np.abs(r).mean())
def window(x, seed, fs):
    out = {'real': stats(x, fs)}
    for v in ('marginal', 'spectrum'):
        out[v] = stats(multivariate_iaaft(x, np.random.default_rng(seed), final=v)[0], fs)
    return out
if __name__ == '__main__':
    W = np.load('raw/nt/windows.npz'); man = json.load(open('raw/nt/manifest.json')); rows = man['rows']; fs = man['fs']
    M = rows[0]['M']; step = (M - 1) / 15; sub = np.round(np.arange(16) * step).astype(int)
    keep = [r for r in rows if r['state'] in ('awake_ec', 'awake_eo', 'drug', 'sleep', 'recovery_ec')]
    res = Parallel(n_jobs=8, verbose=0)(delayed(window)(W[r['id']].astype(float)[:, sub], k, fs) for k, r in enumerate(keep))
    json.dump(dict(rows=keep, res=res, subset=sub.tolist()), open('surr_pairs_nt.json', 'w'))
    print('done', len(res))
