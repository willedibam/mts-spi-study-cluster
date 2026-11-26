"""Post hoc (26 November), descriptive: are Chibi's large-amplitude segments, and the decoupling in them, spread over the array or carried by a
few channels? Same subsets, normal scoring, 1-s segments as episodic.py; surrogate seeded k (not the zns seed, so a fresh null draw).
Per window: share of channels whose segment log RMS follows the global one (Spearman > .5); participation ratio over channels of the RMS
excess in the top-quartile segments; share of pairs whose |z| falls with global amplitude; that Spearman for pairs with and without the
4 burstiest channels; and the channel burstiness profile (SD over segments of log RMS), drug against eyes closed.
  python ep_spatial.py <bank dir>
"""
import sys, json, numpy as np, warnings
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path[:0] = [str(HERE), str(HERE.parent)]; warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from edges import ns, sub32
from episodic import segz, SEG, STEP
from mtsspi.surrogates import multivariate_iaaft
d = sys.argv[1]; man = json.load(open(f'{d}/manifest.json')); W = np.load(f'{d}/windows.npz')
def chrms(v): return np.log(np.array([np.sqrt((v[s:s+SEG]**2).mean(0)) for s in range(0, len(v)-SEG+1, STEP)]))  # (S, 32)
iu = np.triu_indices(32, 1)
for agent in ('KT', 'KTMD', 'PF'):
    rows = [r for r in man['rows'] if r['animal'] == 'Chibi' and r['agent'] == agent and r['state'] in ('drug', 'awake_ec')]
    if not rows: continue
    S0 = sub32([r for r in man['rows'] if r['agent'] == agent][0]['M'])
    out = {}
    for st in ('drug', 'awake_ec'):
        acc = []
        for k, r in enumerate([r for r in rows if r['state'] == st]):
            v = ns(W[r['id']].astype(float)[:, S0]); s = multivariate_iaaft(v, np.random.default_rng(k), final='spectrum')[0]
            res = []
            for x in (v, s):
                c = chrms(x); g = np.log(np.sqrt(np.mean(np.exp(2 * c), 1))); z = segz(x)
                rc = np.array([spearmanr(c[:, j], g)[0] for j in range(32)])           # channel follows global amplitude
                rp = np.array([spearmanr(np.abs(z[:, e]), g)[0] for e in range(z.shape[1])])  # pair |z| vs global amplitude
                top = np.argsort(c.std(0))[-4:]; inv = np.isin(iu[0], top) | np.isin(iu[1], top)
                hi = g >= np.quantile(g, .75); ex = np.clip(c[hi].mean(0) - c[~hi].mean(0), 0, None)
                pr = ex.sum() ** 2 / (32 * (ex ** 2).sum())                          # participation ratio of the burst over channels
                res.append([np.mean(rc > .5), pr, np.mean(rp < 0), rp[inv].mean(), rp[~inv].mean()])
            acc.append(res[0] + res[1] + [c.std(0)])
        out[st] = acc
    a = np.array([x[:10] for x in out['drug']]); b = np.array([x[:10] for x in out['awake_ec']])
    bd = np.mean([x[10] for x in out['drug']], 0); be = np.mean([x[10] for x in out['awake_ec']], 0)
    print(f"Chibi {agent}: n drug {len(a)}, n EC {len(b)}")
    lab = ['ch rho>.5 share', 'burst PR', 'pairs |z|-amp <0', 'rho pairs w/ top4 ch', 'rho other pairs']
    for i, l in enumerate(lab): print(f"  {l:22s} drug real {a[:, i].mean():+.2f} surr {a[:, i+5].mean():+.2f} | EC real {b[:, i].mean():+.2f} surr {b[:, i+5].mean():+.2f}")
    print(f"  channel burstiness (SD log RMS) drug vs EC across channels: r {np.corrcoef(bd, be)[0,1]:+.2f}; drug top-4 share of total {np.sort(bd)[-4:].sum()/bd.sum():.2f} (uniform .125)")
