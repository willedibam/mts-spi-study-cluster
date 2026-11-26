"""Score monkey B by the frozen rule (preregs/ferraina-prereg.md).    python fb_score.py Z.npz SEL_DIR [VARIANT]"""
import sys, json, numpy as np
from scipy.stats import spearmanr
Zf, d = np.load(sys.argv[1]), sys.argv[2]; var = sys.argv[3] if len(sys.argv) > 3 else 'ns'
rows = json.loads(str(Zf['rows'])); Z = Zf[f'Z_{var}']
real, res = Z[:, 0], Z[:, 0] - Z[:, 1:].mean(1)
st = np.array([r['state'] for r in rows]); kk = np.array([r['k'] for r in rows])
BI = np.array([r['burst_index'] for r in rows]); DF = np.array([r['delta_fraction'] for r in rows])
rng = np.random.default_rng(20261126); B, NB = 30, 10000


def mbb(x):
    n = len(x); nb = int(np.ceil(n / B)); st_ = rng.integers(0, n - B + 1, (NB, nb))
    return np.array([np.concatenate([x[s:s + B] for s in row])[:n].mean() for row in st_])


def blocks(s):
    m = st == s; b = kk[m] // B; v = res[m]
    return np.array([v[b == j].mean() for j in np.unique(b) if (b == j).sum() >= 15])


def report(res_, label, mask_d=None):
    w1, dd, w2 = res_[st == 'W1'], res_[st == 'D'] if mask_d is None else res_[(st == 'D') & mask_d], res_[st == 'W2']
    bw1, bw2, bd = mbb(w1), mbb(w2), mbb(dd)
    dD, dR = dd.mean() - w1.mean(), w2.mean() - dd.mean()
    ciD, ciR = np.percentile(bd - bw1, [2.5, 97.5]), np.percentile(bw2 - bd, [2.5, 97.5])
    print(f'{label}: n W1/D/W2 {len(w1)}/{len(dd)}/{len(w2)}  ΔD {dD:+.4f} [{ciD[0]:+.4f}, {ciD[1]:+.4f}]  Δrec {dR:+.4f} [{ciR[0]:+.4f}, {ciR[1]:+.4f}]  W2−W1 {w2.mean()-w1.mean():+.4f}')
    return dD, ciD, dR, ciR


print(f'variant {var}; residual = real z − mean of 3 joint-IAAFT draws; moving blocks of {B} windows, {NB} resamples')
for s in ('W1', 'D', 'W2'):
    m = st == s
    print(f'  {s}: n {m.sum()}  real z median {np.median(real[m]):.4f}  1−z² median {np.median(1-real[m]**2):.4f}  surr z median {np.median(Z[m,1:]):.4f}  '
          f'residual mean {res[m].mean():+.4f} sd {res[m].std(ddof=1):.4f}  BI median {np.median(BI[m]):.3f}  δ median {np.median(DF[m]):.3f}')
dD, ciD, dR, ciR = report(res, 'PRIMARY')
bw1, bd = blocks('W1'), blocks('D'); nsd = bw1.std(ddof=1)
fr = (bd < res[st == 'W1'].mean()).mean()
print(f'  W1 5-min block means {np.round(bw1, 4).tolist()}  null SD {nsd:.4f}; ΔD = {dD/nsd:+.2f} null SD')
print(f'  D 5-min block means {np.round(bd, 4).tolist()}  fraction below W1 mean {fr:.2f}')
c = [ciD[1] < 0, dD < -2 * nsd, fr >= .8, dR > 0]
ceiling = np.median(real[st == 'W1']) >= .97
if all(c): v = 'PRESENT'
elif dD > 0 and ciD[0] > 0: v = 'REVERSED'
elif ceiling: v = 'UNINFORMATIVE (ceiling)'
elif abs(dD) <= 2 * nsd and ciD[0] > -0.010: v = 'ABSENT'
else: v = 'INCONCLUSIVE'
print(f'  criteria [CI<0, beyond −2 SD, ≥80% D blocks below, Δrec>0] = {c}; W1 median real z ≥ .97: {ceiling}\nVERDICT ({var}): {v}')
if var != 'ns': sys.exit()
print('\nSECONDARY')
w1 = res[st == 'W1']; h = kk[st == 'W1'] < np.median(kk[st == 'W1'])
print(f'  W1 drift, later − earlier half: {w1[~h].mean() - w1[h].mean():+.4f}')
report(res, 'D windows with BI ≥ .85', BI >= .85)
md = st == 'D'; p = np.polyfit(BI[md], res[md], 1); at = np.polyval(p, np.median(BI[st == 'W1']))
print(f'  D residual on BI: slope {p[0]:+.4f}; at W1 median BI {np.median(BI[st=="W1"]):.3f}: ΔD {at - w1.mean():+.4f}; D BI range {BI[md].min():.3f}–{BI[md].max():.3f}')
for s in ('W1', 'D', 'W2'):
    m = st == s; print(f'  within {s}: Spearman(residual, BI) {spearmanr(res[m], BI[m])[0]:+.2f}, (residual, δ) {spearmanr(res[m], DF[m])[0]:+.2f}')
meta = json.load(open(f'{d}/sel.json'))['meta']; ch = np.array(meta['sel32_channels'])
G, K = Zf['G_ns'], Zf['K_ns']; iu = np.triu_indices(32, 1); a, b = ch[iu[0]] <= 96, ch[iu[1]] <= 96
for lab, sub in (('cross-block pairs', a != b), ('within-block pairs', a == b)):
    zs = np.array([[np.corrcoef(G[w, j, sub], K[w, j, sub])[0, 1] for j in range(4)] for w in range(len(rows))])
    print(f'  {lab} ({sub.sum()}):', end=' '); report(zs[:, 0] - zs[:, 1:].mean(1), lab)
