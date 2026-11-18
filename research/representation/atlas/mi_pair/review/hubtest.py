"""George propofol hub test: select on one session, test on the other (both directions).
Selection: the 20 edges contributing most to the 2-s z drop (hubs_out 'd', 126-ch) in the selection session.
Controls: 3 per hub edge, outside the selection session's top 5%, nearest in (drug |r|, wake |r|) of the test session.
Test: 10-s windows of the test session (drug, awake_ec), all channels normal-scored, readouts from mech.feats beyond
B=10 joint 126-ch IAAFT draws per window (spectrum-final), as (real − surrogate mean) / surrogate SD.
Fixed readings (from the normal-scored calibration, mech_ns.py):
  curved function      : curv strongly +, cokurt ≈ 0
  decoupling episodes  : switch +, rkurt +, curv ≈ 0, gate ≤ 0 (local events: gate clearly −)
  gain / shared bursts : switch +, cokurt +, gate +, rkurt ≈ 0
Interpretation uses the hub − control difference of the drug − wake change."""
import json, sys, numpy as np, warnings; warnings.filterwarnings('ignore')
from joblib import Parallel, delayed
from scipy.stats import norm, rankdata
from surr_pairs import multivariate_iaaft
import mech
ns = lambda x: norm.ppf((np.apply_along_axis(rankdata, 0, x) - .5) / len(x))
HO = np.load('hubs_out.npy', allow_pickle=True).item()
SUB = np.load('edgemi_nt2s_126.npz')['sub']; M = len(SUB); IU = np.triu_indices(M, 1)
SESS = {'20120731': 'raw/nt10', '20120803': 'raw/nt10b'}
B, NH, NC = 10, 20, 3

def windows(date, state):
    d = SESS[date]; R = json.load(open(f'{d}/manifest.json'))['rows']; W = np.load(f'{d}/windows.npz')
    return [ns(W[r['id']].astype(float)[:, SUB]) for r in R if r['unit'].startswith(date) and r['state'] == state]

def edge_feats(x, E, seed):
    rng = np.random.default_rng(seed)
    f = np.array([mech.feats(x[:, i], x[:, j]) for i, j in E])
    S = []
    for _ in range(B):
        s = multivariate_iaaft(x, rng, final='spectrum')[0]; S.append([mech.feats(s[:, i], s[:, j]) for i, j in E])
    S = np.array(S); return (f - S.mean(0)) / S.std(0, ddof=1)

def run(sel, test):
    u = [k for k in HO if k.startswith(sel) and HO[k]['animal'] == 'George' and HO[k]['agent'] == 'PF'][0]
    d = HO[u]['d']; hub = np.argsort(d)[:NH]; top5 = set(np.argsort(d)[:int(.05 * len(d))])
    Xd, Xw = windows(test, 'drug'), windows(test, 'awake_ec')
    rd = np.abs(np.mean([np.corrcoef(x.T)[IU] for x in Xd], 0)); rw = np.abs(np.mean([np.corrcoef(x.T)[IU] for x in Xw], 0))
    cand = np.array([e for e in range(len(d)) if e not in top5]); used = set(); ctrl = []
    for h in hub:
        dist = np.hypot(rd[cand] - rd[h], rw[cand] - rw[h]); order = cand[np.argsort(dist)]
        pick = [e for e in order if e not in used][:NC]; used |= set(pick); ctrl += pick
    ctrl = np.array(ctrl); E = [(IU[0][e], IU[1][e]) for e in np.r_[hub, ctrl]]
    out = {}
    for st, X in (('drug', Xd), ('wake', Xw)):
        out[st] = np.array(Parallel(n_jobs=7)(delayed(edge_feats)(x, E, k) for k, x in enumerate(X)))   # win x edge x feat
    elec = lambda e: (SUB[IU[0][e]] + 1, SUB[IU[1][e]] + 1)
    return dict(hub=hub, ctrl=ctrl, rd=rd, rw=rw, hub_elec=[elec(e) for e in hub], **out)

if __name__ == '__main__':
    res = {}
    for sel, test in (('20120731', '20120803'), ('20120803', '20120731')):
        res[f'{sel}->{test}'] = run(sel, test); print('done', sel, '->', test, flush=True)
    np.save('hubtest_out.npy', res, allow_pickle=True)
