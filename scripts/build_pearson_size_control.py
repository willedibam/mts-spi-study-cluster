"""Nested observation-size control; normalization uses covariance mean only."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from threadpoolctl import threadpool_limits
from src.generators import generate_varma, generate_cml_logistic, generate_kuramoto, generate_wave_1d
from scripts.spi_baseline_exploration import ROOT, sha
from scripts.build_pearson_strength_match import normalize

RUN = 'pearson-size-control-261004'
DATA = ROOT/'data/representation'/RUN
OUT = ROOT/'results/representation'/RUN
LABELS = ['VAR-0.7', 'Wave', 'CML-1.895', 'Kuramoto-fast',
          'Gaussian-correlated', 'Cauchy-correlated', 'Gaussian-independent', 'Cauchy-independent']


def parents(instance):
    result = {}
    for index, label in enumerate(LABELS[:4]):
        rng = np.random.default_rng(np.random.SeedSequence([261010, instance, index]))
        common = dict(M=32, T=1000, zscore=False, rng=rng)
        if index == 0:
            x = generate_varma(**common, phi=.7, coupling=.1, ma_phi=0, ma_coupling=0, noise_std=.1, transients=2000)
        elif index == 1:
            x = generate_wave_1d(**common, c=10*np.sqrt(.2/.08), dt=.00125, n_modes=5, ic_decay=1.5, noise_std=.01)
        elif index == 2:
            x = generate_cml_logistic(**common, alpha=1.895, eps=.109375, lattice_size=100, transients=2000)
        else:
            x = generate_kuramoto(**common, dt=.00625, K=-50., omega_mean=3., omega_std=1.73205, eta=0,
                                  connectivity='all-to-all', transients=2000, output='sin')
        result[label] = x.T
    for index, name in enumerate(['Gaussian', 'Cauchy']):
        rng = np.random.default_rng(np.random.SeedSequence([261010, instance, index+4]))
        draw = rng.normal if index == 0 else rng.standard_cauchy
        epsilon, common = draw(size=(32, 1000)), draw(size=(1, 1000))
        loadings = rng.permutation(np.linspace(.5, 1.5, 32))[:, None]
        result[name+'-independent'] = epsilon
        result[name+'-correlated'] = .75*epsilon + .25*loadings*common
    return result


def candidate_signs(m, instance):
    if m == 8:
        signs = np.ones((128, m))
        signs[:, 1:] = 1-2*((np.arange(128)[:, None] >> np.arange(m-1)) & 1)
    else:
        rng = np.random.default_rng(np.random.SeedSequence([261010, instance, m, 98]))
        signs = rng.choice([-1., 1.], size=(2048, m))
        signs[:, 0] = 1
        signs[0] = 1
        signs = np.unique(signs, axis=0)
    return signs


def match_mean(x, noise, target, instance):
    """Find the earliest noise-scale bracket admitting a signed-mean match."""
    x, noise = normalize(x), normalize(noise)
    m, t = x.shape
    signs = candidate_signs(m, instance)
    c, e = x@x.T/t, noise@noise.T/t
    cross = (x@noise.T + noise@x.T)/t
    def correlation(sigma):
        a = c + sigma*cross + sigma*sigma*e
        sd = np.sqrt(np.diag(a))
        return a/sd[:, None]/sd[None, :]
    def means(sigma):
        r = correlation(sigma)
        return (np.einsum('bi,ij,bj->b', signs, r, signs, optimize=True)-m)/(m*(m-1))
    left, fleft = 0., means(0.)-target
    for right in [.01, .03, .1, .3, 1., 3., 10., 30., 100.]:
        fright = means(right)-target
        crossing = np.flatnonzero(fleft*fright <= 0)
        if len(crossing):
            # Choose by scalar calibration only, never other SPI outcomes.
            fraction = np.abs(fleft[crossing])/(np.abs(fleft[crossing])+np.abs(fright[crossing])+1e-300)
            index = int(crossing[np.argmin(fraction)])
            sign = signs[index]
            def value(sigma):
                return (sign@correlation(sigma)@sign-m)/(m*(m-1))-target
            sigma = brentq(value, left, right, xtol=1e-13)
            return sign[:, None]*normalize(x+sigma*noise), float(sigma), sign.astype(int).tolist()
        left, fleft = right, fright
    raise ValueError(f'No mean-covariance bracket for M={m}, T={t}, parent={instance}; do not replace the seed')


def build():
    if (DATA/'manifest.json').exists():
        raise FileExistsError('Immutable raw bank exists')
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    arrays, original, rows = {}, {}, []
    for instance in range(5):
        bank = parents(instance)
        target = float(np.random.default_rng(np.random.SeedSequence([261010, instance, 99])).uniform(.003, .007))
        for label, x in bank.items():
            original[f'{label}-parent-{instance}'] = x
        for m in [8, 16, 32]:
            start = (32-m)//2
            for t in [100, 500, 1000]:
                for index, label in enumerate(LABELS):
                    parent = bank[label]
                    x = parent[start:start+m, :t]
                    matched = index < 6
                    sigma, sign = 0., [1]*m
                    if matched:
                        rng = np.random.default_rng(np.random.SeedSequence([261010, instance, index, 97]))
                        noise = rng.normal(size=(32, 1000))[start:start+m, :t]
                        x, sigma, sign = match_mean(x, noise, target, instance)
                    else:
                        x = normalize(x)
                    covariance = x@x.T/t
                    mask = ~np.eye(m, dtype=bool)
                    mean = float(covariance[mask].mean())
                    if matched:
                        assert abs(mean-target) < 1e-10
                    np.testing.assert_allclose(x.mean(1), 0, atol=1e-12)
                    np.testing.assert_allclose(x.var(1), 1, atol=1e-12)
                    name = f'{label}-M{m}-T{t}-I{instance}'
                    arrays[name] = x
                    rows.append(dict(row_id=name, label=label, instance=instance, parent=f'{label}-parent-{instance}',
                        M=m, T=t, cell=f'M{m}-T{t}', corpus_index=len(rows), panel='matched' if matched else 'independent_control',
                        mean_covariance=mean, mean_abs_Pearson=float(np.abs(covariance[mask]).mean()),
                        target_signed=target if matched else None, noise_sigma=sigma, polarity=sign))
        print('Parent block', instance, 'complete', flush=True)
    metadata = dict(__dataset_names__=np.array([r['row_id'] for r in rows]),
                    __labels_json__=np.array([json.dumps([r['label']]) for r in rows]),
                    __shapes__=np.array([[r['M'], r['T']] for r in rows]), __axis_order__=np.array(['process', 'observation']))
    np.savez_compressed(DATA/'observations.npz', **arrays, **metadata)
    np.savez_compressed(DATA/'parents.npz', **original)
    manifest = dict(rows=rows, archive_sha256=sha(DATA/'observations.npz'), parents_sha256=sha(DATA/'parents.npz'),
                    script_sha256=sha(Path(__file__)), status='raw feasibility only; no SPI job authorized by this script',
                    qualification='Five independent parents/class; nine nested observation windows/parent. Split by parent instance. Only signed covariance mean matched; absolute correlation is a diagnostic.')
    (DATA/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    pd.DataFrame(rows).drop(columns='polarity').to_csv(OUT/'calibration.csv', index=False)
    print('Saved', len(rows), 'records', manifest['archive_sha256'])


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        build()
