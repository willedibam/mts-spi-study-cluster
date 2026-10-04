"""Development-only mean matching across six generator families; no pyspi jobs.

Select native settings on scout means only, then assess fresh realizations.
The observational calibration is explicit and is not physical gain matching.
"""
import argparse
import itertools
import json
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.signal import hilbert, stft
from scipy.stats import rankdata
from sklearn.covariance import GraphicalLasso
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from scripts.build_pearson_size_match import absolute, joint_match
from scripts.build_pearson_strength_match import normalize
from scripts.spi_baseline_exploration import ROOT, project_features, sha
from src.corpus_geometry import fit_geometry_transform
from src.generators import generate_cml_logistic, generate_kuramoto, generate_varma, generate_wave_1d

RUN = 'diverse-marginal-scout-261005'
OUT = ROOT/'results/representation'/RUN
FAMILIES = ['VAR', 'Wave', 'CML', 'Kuramoto', 'Gaussian', 'Cauchy']
SEED = 261025
M, T = 16, 1000
MASK = ~np.eye(M, dtype=bool)


def configurations():
    return {
        'VAR': [dict(phi=a, gain=b) for a, b in itertools.product([.2, .5, .8], [.05, .1, .15])],
        'Wave': [dict(courant=a, modes=b) for a, b in itertools.product([.05, .2, .6], [2, 4, 8])],
        'CML': [dict(alpha=a, eps=b) for a, b in itertools.product([1.69, 1.895, 2.], [.03, .15, .3])],
        'Kuramoto': [dict(K=a, dt=b) for a, b in itertools.product([.5, 2., 8.], [.01, .03, .1])],
        **{name: [dict(max_delay=a, heterogeneity=b) for a, b in itertools.product([0, 1, 5], [0., .5, 1.])]
           for name in ['Gaussian', 'Cauchy']},
    }


def recording(family, config, block, stage):
    """Generate one record and match both observed correlation summaries."""
    index = FAMILIES.index(family)
    rng = np.random.default_rng(np.random.SeedSequence([SEED, stage, block, index]))
    common = dict(M=M, T=T, rng=rng, zscore=False)
    if family == 'VAR':
        x = generate_varma(**common, phi=config['phi'], coupling=config['gain']/2,
                          ma_phi=0, ma_coupling=0, noise_std=1, transients=2000).T
    elif family == 'Wave':
        x = generate_wave_1d(**common, c=1, dt=config['courant']/M,
                            n_modes=config['modes'], ic_decay=.75).T
    elif family == 'CML':
        x = generate_cml_logistic(**common, **config, lattice_size=100, transients=2000).T
    elif family == 'Kuramoto':
        x = generate_kuramoto(**common, **config, omega_mean=3, omega_std=1,
                             connectivity='all-to-all', transients=2000, output='sin').T
    else:
        draw = rng.normal if family == 'Gaussian' else rng.standard_cauchy
        independent = draw(size=(M, T))
        latent = draw(size=T+config['max_delay'])
        delays = np.arange(M) % (config['max_delay']+1)
        loadings = 1 + config['heterogeneity']*np.linspace(-.5, .5, M)
        shared = np.array([latent[d:d+T]*w for d, w in zip(delays, loadings)])
        # The common-driver component need not be rank one when delays differ.
        parts = dict(signal=shared, first=independent, second=draw(size=(M, T)), kind='shared_input')
        target_rng = np.random.default_rng(np.random.SeedSequence([SEED, stage, block, 99]))
        a, b = target_rng.uniform(.065, .075), target_rng.uniform(.003, .007)
        y, settings = joint_match(parts, a, b, block+stage*10000)
        return y, dict(target_absolute=a, target_signed=b, native_absolute=np.nan,
                       common_boost=0., retained_native_variance=np.nan, **settings)
    x = normalize(x)
    native_absolute = absolute(x)
    # If native correlation is too weak, add an explicit shared observation input.
    # Always record its size: similarity achieved by overwhelming a system is not success.
    target_rng = np.random.default_rng(np.random.SeedSequence([SEED, stage, block, 99]))
    a, b = target_rng.uniform(.065, .075), target_rng.uniform(.003, .007)
    boost = 0.
    shared = normalize(rng.normal(size=(1, T)))
    if native_absolute <= a + .01:
        boost = brentq(lambda q: absolute(x + q*shared) - (a+.03), 0, 10)
    signal = x + boost*shared
    parts = dict(signal=signal, first=rng.normal(size=(M, T)), second=rng.normal(size=(M, T)), kind='attenuation')
    y, settings = joint_match(parts, a, b, block+stage*10000)
    sigma = settings['parameter']
    retention = float(np.mean(1/(1+boost**2+sigma**2)))
    return y, dict(target_absolute=a, target_signed=b, native_absolute=native_absolute,
                   common_boost=boost, retained_native_variance=retention, **settings)


def probes(x):
    """Inexpensive pairwise probes, explicitly not the p90 catalogue."""
    x = normalize(x)
    r = x@x.T/T
    ranks = normalize(rankdata(x, axis=1))
    rho = ranks@ranks.T/T
    precision = np.linalg.inv(r)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        gl = GraphicalLasso().fit(x.T)
    matrices = dict(covariance=r, absolute_Pearson=abs(r), Pearson_squared=r*r,
                    Spearman=rho, Spearman_squared=rho*rho,
                    euclidean=np.sqrt(np.maximum(2-2*r, 0)),
                    Gaussian_plugin_MI=-.5*np.log(np.maximum(1-r*r, 1e-12)),
                    precision=precision, precision_squared=precision**2,
                    GL_covariance=gl.covariance_, GL_covariance_squared=gl.covariance_**2,
                    GL_precision=gl.precision_, GL_precision_squared=gl.precision_**2)
    for lag in [1, 5]:
        a, b = normalize(x[:, lag:]), normalize(x[:, :-lag])
        lag_r = a@b.T/(T-lag)
        a, b = normalize(ranks[:, lag:]), normalize(ranks[:, :-lag])
        lag_s = a@b.T/(T-lag)
        matrices.update({f'Pearson_lag{lag}': lag_r, f'Pearson_lag{lag}_squared': lag_r**2,
                         f'Spearman_lag{lag}': lag_s, f'Spearman_lag{lag}_squared': lag_s**2})
    frequencies, _, fourier = stft(x, nperseg=128, noverlap=64, boundary=None, padded=False)
    spectrum = np.einsum('ift,jft->ijf', fourier, fourier.conj())/fourier.shape[-1]
    power = np.real(spectrum[np.arange(M), np.arange(M)])
    coherence = abs(spectrum)**2/np.maximum(power[:, None, :]*power[None, :, :], 1e-30)
    matrices['coherence_low'] = coherence[:, :, (frequencies>0)&(frequencies<=.25)].mean(axis=2)
    matrices['coherence_high'] = coherence[:, :, frequencies>.25].mean(axis=2)
    phase = np.exp(1j*np.angle(hilbert(x, axis=1)))
    matrices['PLV'] = abs(phase@phase.conj().T/T)
    edges = np.array([a[MASK] for a in matrices.values()])
    assert np.isfinite(edges).all()
    z = np.corrcoef(edges)[np.triu_indices(len(edges), 1)]
    distributions = np.column_stack([edges.mean(1), edges.std(1),
                                     np.quantile(edges, [.1, .25, .5, .75, .9], axis=1).T]).ravel()
    return dict(mean=edges.mean(1), distribution=distributions, z=z), list(matrices), len(caught)


def bank(configs, blocks, stage, folder):
    rows, features, failures, names = [], {}, [], None
    for family, candidates in configs.items():
        for candidate, config in enumerate(candidates):
            for block in blocks:
                try:
                    x, settings = recording(family, config, block, stage)
                    f, names, nwarn = probes(x)
                    assert abs(f['mean'][0]-settings['target_signed']) < 1e-9
                    assert abs(f['mean'][1]-settings['target_absolute']) < 1e-9
                    row = dict(family=family, candidate=candidate, block=block, stage=stage, M=M, T=T,
                               graphical_warnings=nwarn, **settings)
                    rows.append(row)
                    for key, value in f.items():
                        features.setdefault(key, []).append(value)
                except (ValueError, np.linalg.LinAlgError, AssertionError) as exc:
                    failures.append(dict(family=family, candidate=candidate, block=block, error=str(exc)))
            print('Generated', stage, family, candidate, flush=True)
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(folder/'rows.csv', index=False)
    (folder/'failures.json').write_text(json.dumps(failures, indent=2)+'\n')
    arrays = {k: np.array(v) for k, v in features.items()}
    np.savez_compressed(folder/'features.npz', names=np.array(names), **arrays)
    return pd.DataFrame(rows), arrays


def select(rows, means, configs):
    # Scale each detector once using all scout observations; no z is inspected.
    scale = means.std(axis=0)
    keep = scale > 1e-7
    standardized = (means[:, keep]-means[:, keep].mean(0))/scale[keep]
    centers, indices = {}, {}
    for family in FAMILIES:
        ids, averages = [], []
        for candidate in range(len(configs[family])):
            use = (rows.family==family)&(rows.candidate==candidate)
            if use.sum() == 4:
                ids.append(candidate)
                averages.append(standardized[use].mean(0))
        if not ids:
            raise ValueError(f'No fully valid scout candidate for {family}')
        indices[family], centers[family] = ids, np.array(averages)
    solutions = []
    for initial in np.vstack(list(centers.values())):
        centroid = initial.copy()
        for _ in range(50):
            chosen = {f: int(np.argmin(((centers[f]-centroid)**2).sum(1))) for f in FAMILIES}
            points = np.array([centers[f][chosen[f]] for f in FAMILIES])
            new = points.mean(0)
            if np.allclose(new, centroid, atol=1e-12, rtol=0):
                break
            centroid = new
        solutions.append((float(((points-points.mean(0))**2).sum()), chosen))
    objective, chosen = min(solutions, key=lambda p: p[0])
    settings = {f: [configs[f][indices[f][chosen[f]]]] for f in FAMILIES}
    return dict(configurations=settings, objective=objective,
                indices={f: indices[f][chosen[f]] for f in FAMILIES},
                criterion='Minimize within-six-family squared distances between scaled scout mean vectors; deterministic multi-start alternating optimization, not a guaranteed global optimum. No z used.')


def evaluate():
    rows = pd.read_csv(OUT/'fresh/rows.csv')
    with np.load(OUT/'fresh/features.npz') as a:
        bank = {k:a[k] for k in a.files}
    assert len(rows) == 6*64, 'Do not silently drop failed seeds from evaluation'
    tr = rows.block.to_numpy()<32
    y = rows.family.to_numpy()
    scores, predictions = [], []
    def score(name, model, train, test):
        model.fit(train, y[tr])
        pred = model.predict(test)
        scores.append(dict(method=name, BA=balanced_accuracy_score(y[~tr], pred)))
        f = rows.loc[~tr, ['family','block']].copy()
        f['method'], f['prediction'] = name, pred
        predictions.append(f)
    for key in ['mean','distribution']:
        transform = fit_geometry_transform(bank[key][tr], scaling='standard', minimum_valid_fraction=1.)
        values = np.clip(transform.transform(bank[key]), -5, 5)
        for name, model in [('linear', LogisticRegression(C=1,max_iter=5000)), ('RBF', SVC(C=1)),
                            ('trees', ExtraTreesClassifier(n_estimators=400,min_samples_leaf=2,random_state=SEED,n_jobs=4))]:
            score(key+'_'+name, model, values[tr], values[~tr])
        fvalues, _ = f_classif(values[tr], y[tr])
        top = int(np.nanargmax(fvalues))
        score(key+'_selected_tree', DecisionTreeClassifier(max_depth=4,min_samples_leaf=5,random_state=SEED), values[tr,top,None], values[~tr,top,None])
    for standard in [False, True]:
        _, train, query = project_features(bank['z'][tr],bank['z'][~tr],dimensions=20,standard=standard)
        score('z_standard' if standard else 'z_center',LogisticRegression(C=1,max_iter=5000),train,query)
    # Exact matched scalar values: remove only roundoff well below matching tolerance.
    scalar = np.round(bank['mean'][:,:2], 9)
    scalar = (scalar-scalar[tr].mean(0))/scalar[tr].std(0)
    score('two_strength_scalars',SVC(C=1),scalar[tr],scalar[~tr])
    score('z_validity_only',LogisticRegression(C=1,max_iter=5000),np.isfinite(bank['z'][tr]).astype(float),np.isfinite(bank['z'][~tr]).astype(float))
    pd.DataFrame(scores).to_csv(OUT/'metrics.csv',index=False)
    pd.concat(predictions).to_csv(OUT/'predictions.csv',index=False)
    pd.DataFrame(scores).to_string(index=False)
    best_mean = max(r['BA'] for r in scores if r['method'].startswith('mean_'))
    primary_z = next(r['BA'] for r in scores if r['method']=='z_center')
    decision = dict(best_mean=best_mean, primary_z=primary_z, pass_gate=best_mean<=.30 and primary_z>=.70,
                    gate='All mean readouts <=.30 and centered z >=.70; heuristic development gate, chance1/6. Failure stops this fixed grid without p90 submission.',
                    scope='Fresh development realizations; selected using separate scout means. No existing held banks opened; not p90.')
    (OUT/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    print(pd.DataFrame(scores).to_string(index=False),flush=True)


def run(stage):
    if stage == 'scout':
        OUT.mkdir(parents=True,exist_ok=True)
        if (OUT/'protocol.json').exists():
            raise FileExistsError('Preserve the existing scout')
        protocol = dict(seed=SEED,M=M,T=T,configurations=configurations(),
            source_sha256=sha(__file__),helper_hashes={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'scripts/build_pearson_size_match.py',ROOT/'scripts/build_pearson_strength_match.py',ROOT/'src/generators/dynamical.py',ROOT/'src/generators/linear.py',ROOT/'src/generators/pde.py']},
            scout='Four realizations per candidate; common random numbers within family across settings. Select settings by means only. Invalid candidate if any of four fails; preserve failures.',
            fresh='64 new paired class blocks, first32 training and last32 validation. All six families retained. Report failure instead of dropping seeds. No held confirmation created.',
            calibration='Exact per-record signed/absolute correlation target shared across families within block; observation noise plus explicit shared drive if native strength too low. No claim of physical coupling equality. Record native variance fraction ignoring empirical cross-terms as a diagnostic, not exact variance decomposition.',
            gate='All focused mean readouts<=.30 and centered z>=.70. No p90 if screen fails. Distributions secondary. No fresh-validation selection of parameters, families, seeds or embeddings.',
            references=['https://www.nature.com/articles/s43588-023-00519-x'],
            qualifications=['Focused probes are not p90; Gaussian plug-in MI is not KSG MI.', 'Cauchy covariance is finite-sample only.', 'M16/T1000 isolates generator changes; size grid deferred until feasibility.', 'Wave speed uses fixed observation cadence via Courant number; positive Kuramoto K is attractive in this implementation.'])
        (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
        rows, features = bank(configurations(),range(4),0,OUT/'scout')
        selection = select(rows,features['mean'],configurations())
        selection['scout_features_sha256'] = sha(OUT/'scout/features.npz')
        (OUT/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
        print(json.dumps(selection,indent=2),flush=True)
    elif stage == 'fresh':
        if (OUT/'fresh/features.npz').exists():
            raise FileExistsError('Preserve fresh validation bank')
        selected = json.loads((OUT/'selection.json').read_text())
        bank(selected['configurations'],range(64),1,OUT/'fresh')
        evaluate()
    elif stage == 'evaluate':
        evaluate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',choices=['scout','fresh','evaluate'])
    with threadpool_limits(limits=4):
        run(parser.parse_args().stage)
