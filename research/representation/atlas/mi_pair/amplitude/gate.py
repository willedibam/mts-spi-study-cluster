"""Edge-averaged amplitude-coupling readouts, vectorised mech.feats definitions (0.5-s blocks of BL = 50 samples at 100 Hz):
 gate   : corr over blocks of (block Fisher r, block mean log-SD of the two channels)   + shared gain, − local events
 imbal  : corr over blocks of (block Fisher r, |log-SD_i − log-SD_j|)                   − local events
 switch : variance over blocks of block Fisher r
Block r is oriented by the sign of the edge's window r, so negative edges count the same way.
Input x is T × M, already prepared (normal-scored); edges E is an index into triu(M, 1). Returns the mean over E of each."""
import numpy as np
BL = 50


def block_feats(x, E):
    x = (x - x.mean(0)) / x.std(0); T, M = x.shape; nb = T // BL; iu = np.triu_indices(M, 1)
    b = x[:nb * BL].reshape(nb, BL, M); b = b - b.mean(1, keepdims=True); sd = b.std(1)
    z = b / sd[:, None, :]; R = np.einsum('ktm,ktn->kmn', z, z) / BL
    rb = np.arctanh(np.clip(R[:, iu[0][E], iu[1][E]], -.999, .999)); ls = np.log(sd)
    rb = rb * np.sign(np.corrcoef(x.T)[iu[0][E], iu[1][E]])     # orient: + = stronger coupling, whatever the edge's sign
    amp = (ls[:, iu[0][E]] + ls[:, iu[1][E]]) / 2; imb = np.abs(ls[:, iu[0][E]] - ls[:, iu[1][E]])
    c = lambda u, v: ((u - u.mean(0)) * (v - v.mean(0))).mean(0) / (u.std(0) * v.std(0))
    return np.array([np.nanmean(c(rb, amp)), np.nanmean(c(rb, imb)), rb.var(0).mean()])


def top_edges(xs, q=.25):
    """Fixed edge set: top q of |r| averaged over a list of prepared windows (chosen once per session from the reference state)."""
    M = xs[0].shape[1]; iu = np.triu_indices(M, 1)
    r = np.abs(np.mean([np.corrcoef(x.T)[iu] for x in xs], 0)); return np.argsort(r)[::-1][:int(np.ceil(q * len(r)))]
