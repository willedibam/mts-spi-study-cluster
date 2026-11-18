"""Why two CML-intermittency recordings embed with Gaussian noise in the proof figure.

Run ``python -m research.proof.p90.sti_strays``; writes to ``results/proof/figure-261029/sti-strays/``.
- ``sti-strays-lattice.png``: the full 100-site lattice of each stray and of reference recordings,
  laminar/turbulent, with the observed 8-site window marked.
- ``sti-continuum.{csv,png}``: every intermittency, developed-turbulence and Gaussian-noise recording,
  with laminar fraction, cross-channel dependence (mean over ordered pairs of max |lagged r|, lags
  0-3), its spread across pairs (``het``), non-finite z share, and PCA50 distance to its own class
  median over distance to the Gaussian-noise median (``noise_ratio``; > 1 sits nearer noise).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from research.cases.composite_style import carpet, style
from research.proof.p90.proof_figure import _specs, load_config
from research.proof.p90.refresh_proof_figures import load_bank, preprocess

OUT = Path('results/proof/figure-261029/sti-strays')
STRAYS = [(8, 1000, 4), (8, 500, 8)]  # (M, T, instance) of sti-i.
LAMINAR = .05  # |x(t) - x(t-3)| below this: on the period-3 laminar orbit.


def _series(spec, full=False):
    from mtsspi.generators.dynamical import generate_cml_logistic
    from mtsspi.generators.harness import generate_from_spec
    x = np.asarray(generate_from_spec(spec)[0], float); x = x if x.shape[0] == spec.T else x.T
    if not full:
        return x
    obs, lattice = generate_cml_logistic(M=spec.M, T=spec.T, rng=np.random.default_rng(spec.rng_seed),
                                         return_full_lattice=True, **spec.generator_params)
    assert np.allclose(obs, x)
    return x, lattice


def dependence(x):
    """Mean and spread over ordered channel pairs of the max |lagged Pearson r| at lags 0-3."""
    x = (x - x.mean(0))/(x.std(0) + 1e-12); n, m = x.shape
    v = [max(abs(np.corrcoef(x[k:, i], x[:n-k, j])[0, 1]) for k in range(4))
         for i in range(m) for j in range(m) if i != j]
    return float(np.mean(v)), float(np.std(v))


def continuum(config, bank):
    names = [c['name'] for c in config['classes']]
    mask = np.isin(bank['y'], names)
    y, M, T, I = (bank[k][mask] for k in ('y', 'M', 'T', 'instance'))
    Y, keep = preprocess(bank['X'][mask])
    from sklearn.decomposition import PCA
    Z = PCA(50, svd_solver='randomized', random_state=260924).fit_transform(Y)
    Xk = bank['X'][mask][:, keep]; finite = np.isfinite(Xk)
    median = {n: np.median(Z[y == n], 0) for n in names}
    specs = {(s.mts_class, s.M, s.T, s.instance): s for s in _specs(config['panel_a']['generation'])}
    rows = []
    for r in np.flatnonzero(np.isin(y, ['sti-i', 'fdstc', 'gaussian-noise'])):
        x = _series(specs[y[r], int(M[r]), int(T[r]), int(I[r])])
        lam = (np.abs(x[3:] - x[:-3]) < LAMINAR).mean() if y[r] != 'gaussian-noise' else np.nan
        dep, het = dependence(x)
        rows.append(dict(cls=y[r], M=int(M[r]), T=int(T[r]), instance=int(I[r]), laminar=lam, dep=dep, het=het,
                         nonfinite=1 - finite[r].mean(), noise_ratio=np.linalg.norm(Z[r] - median[y[r]]) /
                         np.linalg.norm(Z[r] - median['gaussian-noise'])))
    frame = pd.DataFrame(rows)
    # The laminar stray without imputation: neighbours over its finite coordinates only.
    r = np.flatnonzero((y == 'sti-i') & (M == 8) & (T == 1000) & (I == 4))[0]; c = finite[r]
    d = np.sqrt(np.nanmean(np.where(finite[:, c], (Xk[:, c] - Xk[r, c])**2, np.nan), 1)); d[r] = np.inf
    print('Fully laminar stray, %d finite coordinates only; 10 nearest:' % c.sum(),
          pd.Series(y[np.argsort(d)[:10]]).value_counts().to_dict())
    return frame


def plot_lattices(config):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    specs = {(s.mts_class, s.M, s.T, s.instance): s for s in _specs(config['panel_a']['generation'])}
    rows = [('sti-i', 8, 1000, 4, 'stray, fully laminar'), ('sti-i', 8, 500, 8, 'stray, most turbulent'),
            ('sti-i', 8, 500, 3, 'in class, one bursting site'), ('sti-i', 8, 2000, 0, 'in class, as weak as the stray'),
            ('sti-i', 8, 1000, 2, 'in class, typical'), ('fdstc', 8, 500, 8, 'developed turbulence'),
            ('gaussian-noise', 8, 500, 8, 'Gaussian noise')]
    style()
    fig, axes = plt.subplots(len(rows), 2, figsize=(7.2, 1.2*len(rows)), gridspec_kw={'width_ratios': [3, 1.2]})
    for (cls, m, t, i, label), (a0, a1) in zip(rows, axes):
        spec = specs[cls, m, t, i]; title = f'{cls} M={m} T={t} instance {i}: {label}'
        if spec.generator == 'cml_logistic':
            x, lattice = _series(spec, full=True); turb = np.abs(lattice[3:] - lattice[:-3]) >= LAMINAR
            a0.imshow(turb.T, aspect='auto', cmap=ListedColormap(['#f2efe6', '#222222']), interpolation='nearest',
                      rasterized=True)
            lo = (lattice.shape[1] - m)//2
            a0.add_patch(plt.Rectangle((-.5, lo - .5), turb.shape[0], m, fill=False, ec='#d62728', lw=1))
            a0.set_yticks([0, lo, lattice.shape[1] - 1]); a0.set_ylabel('site', fontsize=6)
            title += f'; laminar window {1 - turb[:, lo:lo+m].mean():.2f}, lattice {1 - turb.mean():.2f}'
        else:
            x = _series(spec); a0.axis('off')
        a0.set_title(title, fontsize=6.5, loc='left'); a0.set_xticks([]); a0.tick_params(labelsize=5)
        carpet(a1, x[:300].T); a1.set_title('window, first 300 steps', fontsize=6)
    axes[-1, 0].set_xlabel('time; black: turbulent, $|x(t) - x(t-3)| \\geq 0.05$; red box: observed sites', fontsize=6)
    fig.tight_layout(); fig.savefig(OUT/'sti-strays-lattice.png', dpi=200); plt.close(fig)


def plot_continuum(frame):
    import matplotlib.pyplot as plt
    style(); fig, ax = plt.subplots(figsize=(4.2, 2.8))
    markers = {500: '^', 1000: 's', 2000: 'o'}
    noise = frame[frame.cls == 'gaussian-noise'].dep  # Its ratio is 1 by construction: show its range only.
    ax.axvspan(noise.min(), noise.max(), color='#5aa9d6', alpha=.3, lw=0, label='Gaussian noise (range)')
    for cls, color in [('fdstc', '#1b7a45'), ('sti-i', '#8fbf3f')]:
        for t, mk in markers.items():
            f = frame[(frame.cls == cls) & (frame['T'] == t)]
            ax.scatter(f.dep, f.noise_ratio, s=4*np.sqrt(f.M), marker=mk, color=color, alpha=.75, lw=0,
                       label=cls if t == 500 else None)
    for m, t, i in STRAYS:
        f = frame[(frame.cls == 'sti-i') & (frame.M == m) & (frame['T'] == t) & (frame.instance == i)].iloc[0]
        ax.annotate(f'M={m}, T={t}', (f.dep, f.noise_ratio), xytext=(6, -2) if t == 500 else (-6, 4),
                    textcoords='offset points', ha='left' if t == 500 else 'right', fontsize=6)
    ax.axhline(1, color='.5', lw=.5, ls='--')
    ax.set(xlabel=r'cross-channel dependence (mean max $|r|$, lags 0--3)', ylabel='distance to own class / to noise',
           yscale='log')
    ax.legend(fontsize=6, frameon=False); fig.tight_layout(); fig.savefig(OUT/'sti-continuum.png', dpi=200); plt.close(fig)


if __name__ == '__main__':
    sys.path.insert(0, '.')
    OUT.mkdir(parents=True, exist_ok=True)
    config = load_config()
    bank = load_bank('data/proof/refresh-261014/features.npz', expected_spis=283, contract='unified_ordered_v3')
    frame = continuum(config, bank); frame.to_csv(OUT/'sti-continuum.csv', index=False)
    plot_lattices(config); plot_continuum(frame)
