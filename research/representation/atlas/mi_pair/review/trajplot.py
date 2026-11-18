import numpy as np, style, matplotlib.pyplot as plt
def centre(Z, unit, state, ref):
    Zc = Z.copy()
    for u in np.unique(unit):
        m = unit == u; r = m & (state == ref)
        if r.any(): Zc[m] -= Z[r].mean(0)
        else: Zc[m] = np.nan
    return Zc
def pca(X, fit=None):
    fit = np.ones(len(X), bool) if fit is None else fit
    mu = X[fit].mean(0); _, s, Vt = np.linalg.svd(X[fit] - mu, full_matrices=False)
    return (X - mu) @ Vt[:2].T, s ** 2 / np.sum(s ** 2)
def panel(ax, P, ev, unit, state, cond, paths, colours, title, faint=True):
    """paths: {condition: [states in order]}; draws faint unit centroids and a bold mean path per condition."""
    for c, order in paths.items():
        cu = np.unique(unit[cond == c]); col = colours[c]
        cents = np.array([[P[(unit == u) & (state == s)].mean(0) for s in order] for u in cu])
        if faint:
            for k in range(1, len(order)): ax.plot(cents[:, k, 0], cents[:, k, 1], 'o', color=col, alpha=.25, ms=3, mec='none')
        m = np.nanmean(cents, 0)
        ax.plot(m[:, 0], m[:, 1], '-', color=col, lw=1.7)
        for k in range(1, len(order)):
            ax.annotate('', m[k], m[k - 1], arrowprops=dict(arrowstyle='-|>', color=col, lw=1.7, shrinkA=0, shrinkB=0))
        ax.plot(m[-1, 0], m[-1, 1], 'o', color=col, ms=4, mec='none')
    ax.plot(0, 0, 'k+', ms=6) if faint else None
    ax.set_xlabel(f'PC1 ({ev[0]:.0%})'); ax.set_ylabel(f'PC2 ({ev[1]:.0%})'); ax.set_title(title); ax.set_box_aspect(1)
