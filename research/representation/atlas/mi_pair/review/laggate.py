"""On hub and r-matched control edges (test session, normal-scored, 0.5-s blocks): does coupling lost in high-amplitude
blocks come back at a small lag (lag/phase dispersion) or not (non-shared local events)?
Per edge: gate0 = corr(block zero-lag r, block amplitude); gateL = same for block max |r| over lags ±5 samples (±50 ms);
loss = mean zero-lag r in the top-amplitude third minus bottom third; lossL = same for best-lag r."""
import numpy as np, hubtest
from mech import BL
R = np.load('hubtest_out.npy', allow_pickle=True).item(); IU = hubtest.IU
def blk(x, y):
    nb = len(x) // BL; bx, by = x[:nb*BL].reshape(nb, BL), y[:nb*BL].reshape(nb, BL)
    z = lambda a: (a - a.mean(1, keepdims=True)) / a.std(1, keepdims=True); bx, by = z(bx), z(by)
    r0 = (bx * by).mean(1)
    rl = np.max([np.abs((bx[:, max(0, l):BL+min(0, l)] * by[:, max(0, -l):BL-max(0, l)]).mean(1)) for l in range(-5, 6)], 0)
    amp = (np.log(x[:nb*BL].reshape(nb, BL).std(1)) + np.log(y[:nb*BL].reshape(nb, BL).std(1))) / 2
    return r0, rl, amp
for k, v in R.items():
    test = k.split('->')[1]; print(k)
    for st, state in (('drug', 'drug'), ('wake', 'awake_ec')):
        X = hubtest.windows(test, state)
        for g, E in (('hub', v['hub']), ('ctrl', v['ctrl'])):
            out = []
            for e in E:
                i, j = IU[0][e], IU[1][e]; r0, rl, amp = map(np.concatenate, zip(*[blk(x[:, i], x[:, j]) for x in X]))
                q = np.quantile(amp, [1/3, 2/3]); hi, lo = amp >= q[1], amp <= q[0]
                out.append([np.corrcoef(r0, amp)[0, 1], np.corrcoef(rl, amp)[0, 1], r0[hi].mean() - r0[lo].mean(), rl[hi].mean() - rl[lo].mean(), r0.mean()])
            o = np.mean(out, 0); print(f'  {st:4s} {g:4s} gate0 {o[0]:+.2f} gateL {o[1]:+.2f} | r hi−lo amp: zero-lag {o[2]:+.3f} best-lag {o[3]:+.3f} | mean block r {o[4]:.3f}')
