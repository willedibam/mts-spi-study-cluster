"""Which SPIs and SPI–SPI coordinates register the A–B coupling in 2:1 locking? Retrospective, descriptive.

Inputs: the M24 varying-sensor-noise arm (`cross-frequency-locking-snr-261006`). The 24 held records at
gamma = 0, .10 and .20 (8 seeds each) have MPIs fetched read-only from Gadi into
`results/order-parameter-inference/locking-ab-audit-261022/mpis/`, each checked against `sources.json`.
Held-record z and Q/eta come from the archived feature bank and scores. No refitting; nothing here
selected q or the figure.

Per SPI: block means over ordered pairs (A–B, A–A, B–B, cross pairs with C) per record. The coupling
signal is the change in mean(A–B) minus mean(A–C, B–C) from gamma = 0 to .20, in units of the SPI's
gamma = 0 spread over all pairs. Per SPI–SPI coordinate: z over all ordered pairs and without the A–B
pairs; the A–B share of the change is 1 - change(without A–B) / change(all).
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RUN = 'cross-frequency-locking-snr-261006'
RESULTS = ROOT/'results/order-parameter-inference'
MPI_DIR = RESULTS/'locking-ab-audit-261022/mpis'
CONTROLS = (0., .10, .20)

FAMILIES = [  # first matching prefix wins; labels describe estimator families, not mechanisms
    (('corr', 'xcorr', 'cov', 'spearmanr', 'kendalltau'), 'linear and rank correlation'),
    (('prec',), 'partial correlation (precision)'),
    (('pdcoh', 'gpdcoh', 'dtf', 'ddtf', 'sgc', 'gc'), 'directed spectral / Granger'),
    (('cohmag', 'icoh', 'dcoh', 'phase', 'plv', 'pli', 'wpli', 'dspli', 'dswpli', 'ppc', 'psi', 'gd'), 'spectral coherence and phase'),
    (('te', 'di'), 'transfer and directed information'),
    (('mi', 'tlmi', 'ce', 'cce', 'je', 'si', 'xme', 'phi'), 'other information-theoretic'),
    (('dtw', 'lcss', 'pdist', 'xpdist', 'bary', 'gwtau'), 'distance and shape similarity'),
    (('coint',), 'cointegration'),
    (('dcorr', 'ids'), 'kernel and distance dependence'),
    (('pec',), 'envelope correlation'),
    (('cds', 'reci', 'lmfit'), 'model-based asymmetry'),
]


def family(name):
    head = name.split('_')[0].removesuffix('-sq')
    for prefixes, label in FAMILIES:
        if head in prefixes:
            return label
    return 'other'


def masks(n=8):
    i, j = np.where(~np.eye(3*n, dtype=bool))
    a, b = i//n, j//n
    ab = ((a == 0) & (b == 1)) | ((a == 1) & (b == 0))
    return dict(AB=ab, AA=(a == 0) & (b == 0), BB=(a == 1) & (b == 1), cross_C=(a != b) & ~ab, without_AB=~ab)


def records():
    scores = pd.read_csv(RESULTS/RUN/'scores.csv')
    sources = {r['row_id']: r for r in json.loads((RESULTS/RUN/'sources.json').read_text())['sources']}
    with np.load(RESULTS/RUN/'features.npz') as bank:
        order = list(bank['spi_order'])
    held = scores[scores.system.eq('random-noise') & scores.role.eq('evaluation') & scores.comparison_eligible]
    rows = held[held.control.round(3).isin(CONTROLS)].sort_values(['control', 'seed'])
    out = []
    off = ~np.eye(24, dtype=bool)
    for row in rows.itertuples():
        path = MPI_DIR/f'{row.row_id}.npz'
        if hashlib.sha256(path.read_bytes()).hexdigest() != sources[row.row_id]['mpi_sha256']:
            raise ValueError(f'MPI changed: {row.row_id}')
        with np.load(path) as m:
            V = np.array([np.asarray(m[k], float)[off] for k in order])
        out.append(dict(control=round(row.control, 3), seed=row.seed, row_id=row.row_id, V=V))
    return order, out


def spi_shifts(order, recs):
    m = masks()
    rows = []
    for r in recs:
        V = r['V']
        entry = dict(control=r['control'], seed=r['seed'])
        for k, name in enumerate(order):
            v = V[k]
            if not np.isfinite(v).all() or v.std() == 0:
                continue
            rows.append(dict(entry, spi=name, sd=v.std(), **{g: v[mask].mean() for g, mask in m.items() if g != 'without_AB'}))
    blocks = pd.DataFrame(rows)
    out = []
    for name, part in blocks.groupby('spi'):
        start, end = part[part.control.eq(0)], part[part.control.eq(.2)]
        if len(start) < 4 or len(end) < 4:
            continue
        scale = start.sd.median()
        signal = (end.AB-end.cross_C).to_numpy(), (start.AB-start.cross_C).to_numpy()
        diff = signal[0].mean()-signal[1].mean()
        spread = np.sqrt(signal[0].var(ddof=1)/len(signal[0])+signal[1].var(ddof=1)/len(signal[1]))
        t = diff/spread if spread > 0 else np.nan
        out.append(dict(spi=name, family=family(name), coupling=diff/scale, coupling_t=t,
                        AA=(end.AA.mean()-start.AA.mean())/scale, BB=(end.BB.mean()-start.BB.mean())/scale,
                        AB=(end.AB.mean()-start.AB.mean())/scale, cross_C=(end.cross_C.mean()-start.cross_C.mean())/scale))
    return pd.DataFrame(out).sort_values('coupling_t', key=abs, ascending=False, ignore_index=True)


def _block_z(V, mask):
    X = V[:, mask]
    finite = np.isfinite(X).all(1) & (X.std(1) > 0)
    X = (X-X.mean(1, keepdims=True))/np.where(finite, X.std(1), 1)[:, None]
    C = X @ X.T/X.shape[1]
    C[~finite] = np.nan; C[:, ~finite] = np.nan
    return C


def coordinate_changes(order, recs):
    """All eligible PC1 coordinates: held |rho| with Q and eta, and how the A–B pairs carry their change."""
    ranks = pd.read_csv(RESULTS/'locking-features-m24-261009/all-feature-rankings.csv').rename(columns={'rank_weight': 'rank'})
    a, b = np.triu_indices(len(order), 1)
    idx = ranks.feature_index.to_numpy()
    m = masks()
    z = {key: {c: [] for c in CONTROLS} for key in ('all', 'without_AB')}
    for r in recs:
        for key in z:
            z[key][r['control']].append(_block_z(r['V'], m[key] if key != 'all' else np.ones(552, bool))[a[idx], b[idx]])
    mean = {key: {c: np.nanmean(v, 0) for c, v in d.items()} for key, d in z.items()}
    change_all = mean['all'][.2]-mean['all'][0.]
    change_without = mean['without_AB'][.2]-mean['without_AB'][0.]
    scores = pd.read_csv(RESULTS/RUN/'scores.csv')
    held = scores.system.eq('random-noise') & scores.role.eq('evaluation') & scores.comparison_eligible
    with np.load(RESULTS/RUN/'features.npz') as bank:
        Z = pd.DataFrame(bank['z'][held.to_numpy()][:, idx].astype(float))
    H = scores[held].reset_index(drop=True)
    R = Z.rank()
    def rho(target):
        t = target.rank().to_numpy()
        Rc = R-R.mean(); tc = t-t.mean()
        return (Rc.mul(tc, axis=0).sum()/np.sqrt((Rc**2).sum()*(tc**2).sum())).abs().to_numpy()
    out = ranks[['rank', 'feature_index', 'spi_a', 'spi_b', 'weight']].copy()
    out['family_a'] = out.spi_a.map(family); out['family_b'] = out.spi_b.map(family)
    out['abs_rho_Q'] = rho(H.Q_lock); out['abs_rho_eta'] = rho(H.eta)
    out['z_gamma0'] = mean['all'][0.]; out['z_gamma20'] = mean['all'][.2]
    out['change'] = change_all; out['change_without_AB'] = change_without
    out['AB_share'] = 1-change_without/np.where(abs(change_all) > 1e-12, change_all, np.nan)
    return out


def classify(coords, strong=.8, clean=.2, moving=.3):
    """Clean trackers: |rho_Q| >= strong, |rho_eta| <= clean, |change| >= moving; carrier by A–B share."""
    c = coords[(coords.abs_rho_Q >= strong) & (coords.abs_rho_eta <= clean) & (coords.change.abs() >= moving)].copy()
    c['carrier'] = np.select([c.AB_share >= .6, c.AB_share <= .2], ['A–B pairs', 'within communities'], 'mixed')
    c['families'] = [' × '.join(sorted((x, y))) for x, y in zip(c.family_a, c.family_b)]
    return c


def frequency_check(cases=((0., 209, 'g000-s209'), (.20, 215, 'g200-s215'))):
    """Within a synchronised community, lagged correlation ~ cos(omega tau): compare the Pearson lag-2 and
    squared lag-20 block means (1 and 10 time units) with the replayed community mean-field frequencies."""
    from research.inference.benchmarks.composite_figures import locking_trajectory
    rows = []
    off = ~np.eye(8, dtype=bool)
    for gamma, seed, stem in cases:
        _, _, field = locking_trajectory(gamma, seed)
        omega = {c: np.polyfit(field['t'], field[k], 1)[0] for c, k in (('A', 'theta'), ('B', 'psi'))}
        with np.load(RESULTS/'locking-features-m24-261009/mpi-examples'/f'{stem}.npz') as m:
            for g, c in enumerate('AB'):
                block = slice(8*g, 8*g+8)
                rows.append(dict(gamma=gamma, community=c, omega=omega[c], ratio=omega['B']/omega['A'],
                                 r_lag2=m['corr_pearson_tau-2'][block, block][off].mean(), cos_1u=np.cos(omega[c]),
                                 r2_lag20=m['corr_pearson_tau-20-sq'][block, block][off].mean(), cos2_10u=np.cos(10*omega[c])**2))
    return pd.DataFrame(rows)
