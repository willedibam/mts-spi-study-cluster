"""Input-only baseline audit; no SPI refits, target fitting, or new simulation.

Run with the project Python. Generated CSVs contain original scores plus baselines.
Pearson is recomputed exactly as the zero-lag Pearson MPI implementation does.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.signal import hilbert
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/order_parameter'
OUT = DATA / 'simple_baselines_260917'
FEATURES = ['mean_correlation', 'mean_abs_correlation', 'temporal_spectral_entropy',
            'analytic_phase_coherence', 'mean_channel_sd', 'leading_correlation_fraction']
LABELS = dict(zip(FEATURES, ['Mean Pearson', 'Mean abs. Pearson', 'Spectral entropy',
                           'Hilbert coherence', 'Mean channel SD', 'Leading corr. fraction']))
OLD = {
    'sl': ('stuart_landau_confirmation', 'stuart_landau_confirmation_analysis', 'gamma'),
    'sl_fine': ('stuart_landau_locking_boundary_confirmation', 'stuart_landau_locking_boundary_confirmation_analysis', 'gamma'),
    'mh': ('miller_huse_confirmation', 'miller_huse_confirmation_analysis', 'g'),
    'kaneko': ('quadratic_cml_development', 'quadratic_cml_development_analysis', 'alpha'),
}
NEW = {
    'kuramoto_full': ('kuramoto_full_observation_260916/primary', 'analysis'),
    'rossler': ('finite_regime_260915/rossler/confirmation/primary', 'analysis'),
    'cml_confirm': ('cml2d_confirmation_260911/primary', '../primary-analysis'),
    'cml_confirm_mt': ('cml2d_confirmation_260911/sensitivity', '../sensitivity-analysis'),
    'cml_pilot': ('cml2d_period_doubling_260911/primary', '../primary-analysis'),
    'cml_mt': ('cml2d_period_doubling_260911/sensitivity', '../sensitivity-analysis'),
    'cml_short': ('cml2d_period_doubling_260911/short-T', '../short-T-analysis'),
    'cml_patch': ('cml2d_period_doubling_260911/contiguous', '../contiguous-analysis'),
    'cml_L6': ('cml2d_full_observation_260914/L6/primary', '../analysis'),
}


def input_statistics(x):
    """x is M x T; undefined correlation from constant channels stays undefined."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) < 2 or not np.isfinite(x).all():
        raise ValueError('Expected finite M x T')
    centred = x - x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1)
    out = {'mean_channel_sd': sd.mean(), 'minimum_channel_sd': sd.min()}
    if np.any(sd <= 1e-12):
        return {**dict.fromkeys(FEATURES, np.nan), **out}
    corr = np.corrcoef(x)
    pairs = corr[np.triu_indices(len(x), 1)]
    power = abs(np.fft.rfft(centred, axis=1)[:, 1:]) ** 2
    prob = power / power.sum(axis=1, keepdims=True)
    logp = np.zeros_like(prob)
    np.log(prob, out=logp, where=prob > 0)
    out.update(mean_correlation=pairs.mean(), mean_abs_correlation=abs(pairs).mean(),
        temporal_spectral_entropy=(-(prob * logp).sum(axis=1) / np.log(prob.shape[1])).mean(),
        analytic_phase_coherence=abs(np.exp(1j * np.angle(hilbert(centred, axis=1))).mean(axis=0)).mean(),
        leading_correlation_fraction=np.linalg.eigvalsh(corr)[-1] / len(x))
    return out


def key(name, M, T, instance, value):
    return (str(name), int(M), int(T), int(instance), round(float(value), 6))


def legacy_inputs(bank, control):
    lookup = {}
    for path in sorted((DATA / bank).rglob('timeseries.npy')):
        meta = json.loads(path.with_name('meta.json').read_text())
        params = meta['generator']['resolved_params']
        value = params[{'gamma': 'frequency_half_width', 'g': 'coupling', 'alpha': 'alpha', 'kappa': 'K'}[control]]
        if control == 'kappa':
            value /= np.sqrt(8 / np.pi)
        k = key(meta['mts_class'], meta['M'], meta['T'], meta['instance_index'], value)
        assert k not in lookup
        lookup[k] = path
    return lookup


def attach(frame, statistics):
    stats = pd.DataFrame(statistics, index=frame.index)
    # This is also a row-identity/orientation audit against previously stored results.
    for col in set(FEATURES) & set(frame):
        np.testing.assert_allclose(frame[col], stats[col], atol=2e-6, rtol=2e-5, equal_nan=True,
                                   err_msg=f'Existing {col} does not reproduce')
    return pd.concat([frame.drop(columns=list(set(stats) & set(frame))), stats], axis=1)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = {}
    banks = {}
    for name, (raw, scores, control) in OLD.items():
        source = DATA / scores / 'scores.csv'
        frame = pd.read_csv(source)
        inputs = legacy_inputs(raw, control)
        statistics = []
        for row in frame.to_dict('records'):
            if 'class_name' in row:
                candidates = [inputs[key(row['class_name'], row['M'], row['T'], row['instance'], row[control])]]
            else:
                candidates = [p for k, p in inputs.items() if k[1:] == key('', row['M'], row['T'], row['instance'], row[control])[1:]]
            assert len(candidates) == 1, (name, row, candidates)
            statistics.append(input_statistics(np.load(candidates[0]).T))
        banks[name] = attach(frame, statistics)
        paths[str(source.resolve())] = name
        print(name, len(frame), flush=True)
    with np.load(DATA / 'kuramoto_full_catalogue_reanalysis/results.npz', allow_pickle=True) as a:
        ix = a['primary_row_indices']
        frame = pd.DataFrame({c: a[c][ix] for c in ['class_name', 'instance', 'kappa', 'mean_abs_correlation',
                                                   'analytic_phase_coherence', 'temporal_spectral_entropy']})
        frame['q'] = a['coordinate_pc1'][ix]
        frame['Q'] = a['target_full_future_R'][ix]
        frame['M'], frame['T'] = 20, 1000
    inputs = legacy_inputs('kuramoto_final_confirmation', 'kappa')
    statistics = [input_statistics(np.load(inputs[key(r.class_name, 20, 1000, r.instance, r.kappa)]).T)
                  for r in frame.itertuples()]
    banks['kuramoto_partial'] = attach(frame, statistics)
    for name, (raw, scores) in NEW.items():
        corpus = DATA / raw
        source = (corpus / scores / 'scores.csv').resolve()
        frame = pd.read_csv(source)
        rows = {r['row_id']: r for r in json.loads((corpus / 'manifest.json').read_text())['rows']}
        archive = corpus / 'observations.npz'
        primary = None
        if not archive.exists():
            archive = corpus.parent / 'primary/observations.npz'
            primary = json.loads((corpus.parent / 'primary/manifest.json').read_text())['rows']
        statistics = []
        with np.load(archive) as a:
            for row in frame.itertuples():
                r = rows[row.row_id]
                if primary is None:
                    x = a[row.row_id]
                else:
                    matches = [p for p in primary if p['seed'] == r['seed'] and p['r'] == r['r'] and p['view'] == r['view']]
                    assert len(matches) == 1
                    x = a[matches[0]['row_id']][:r['M'], :r['T']]
                assert x.shape == (r['M'], r['T'])
                statistics.append(input_statistics(x))
        banks[name] = attach(frame, statistics)
        paths[str(source)] = name
        print(name, len(frame), flush=True)
    # Fixed affine display transforms, shared across each M/T/layout family. These
    # use observed score distributions, never Q, and are not prediction models.
    reference = {k: k for k in banks}
    reference.update(cml_confirm_mt='cml_confirm', cml_mt='cml_pilot', cml_short='cml_pilot', cml_patch='cml_pilot')
    for name, frame in banks.items():
        ref = banks[reference[name]]
        if 'role' in ref:
            ref = ref.query("role == 'evaluation' and eligible")
        if name.startswith('sl'):
            ref = ref.query('M == 32 and T == 1000')
        if name == 'kaneko':
            ref = ref.query("arm == 'large' and instance >= 4 and M == 32")
        for col in ['q1' if name == 'kaneko' else 'q', *FEATURES]:
            mu, sd = ref[col].mean(), ref[col].std(ddof=0)
            frame['display_' + col] = (frame[col] - mu) / sd if sd > 1e-12 else np.nan
        frame['_baseline_bank'] = name
        frame.to_csv(OUT / f'{name}.csv', index=False)
    (OUT / 'sources.json').write_text(json.dumps(paths, indent=2) + '\n')
    return banks


def rho(x, y):
    x, y = rankdata(x), rankdata(y)
    return float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else np.nan


def evaluation_cohorts(banks):
    specs = [
        ('Kuramoto partial', 'kuramoto_partial', '', 'kappa', 'Q', 'q'),
        ('Kuramoto full', 'kuramoto_full', "role == 'evaluation' and eligible", 'control', 'Q_reference', 'q'),
        ('Stuart-Landau broad', 'sl', 'M == 32 and T == 1000', 'gamma', 'Q_R_mean', 'q'),
        ('Stuart-Landau fine', 'sl_fine', '', 'gamma', 'Q_R_mean', 'q'),
        ('Miller-Huse', 'mh', 'M == 32', 'g', 'Q_spin_abs', 'q'),
        ('Kaneko band power', 'kaneko', "arm == 'large' and instance >= 4 and M == 32", 'alpha', 'Q_selected_band_power', 'q1'),
        ('Kaneko entropy', 'kaneko', "arm == 'large' and instance >= 4 and M == 32", 'alpha', 'Q_temporal_entropy', 'q1'),
        ('2D CML confirmation', 'cml_confirm', 'eligible', 'r', 'Q_reference', 'q'),
        ('2D CML contiguous (failed)', 'cml_patch', "role == 'evaluation' and eligible", 'r', 'Q_reference', 'q'),
        ('2D CML L6 (failed)', 'cml_L6', "role == 'evaluation' and eligible", 'r', 'Q_reference', 'q'),
        ('Rossler', 'rossler', 'eligible', 'control', 'Q_reference', 'q')]
    for label, bank, query, control, Q, q in specs:
        f = banks[bank]
        yield label, bank, f.query(query).copy() if query else f.copy(), control, Q, q


def metrics(banks, n_boot=2000):
    results = []
    for label, bank, original, control, Q, q in evaluation_cohorts(banks):
        group = 'seed' if 'seed' in original else 'instance'
        # All methods receive exactly the same rows; no favourable pairwise deletion.
        frame = original.replace([np.inf, -np.inf], np.nan).dropna(subset=[q, Q, *FEATURES])
        targets = frame[Q].to_numpy()
        clusters = [np.flatnonzero(frame[group].to_numpy() == s) for s in sorted(frame[group].unique())]
        rng = np.random.default_rng(260917)
        resamples = [np.concatenate([clusters[j] for j in rng.integers(len(clusters), size=len(clusters))]) for _ in range(n_boot)]
        qboot = np.array([abs(rho(frame[q].to_numpy()[ix], targets[ix])) for ix in resamples])
        for feature in [q, *FEATURES]:
            values = frame[feature].to_numpy()
            boot = np.array([abs(rho(values[ix], targets[ix])) for ix in resamples])
            residual = frame[[feature, Q]] - frame.groupby(control)[[feature, Q]].transform('mean')
            means = frame.groupby(control)[[feature, Q]].mean()
            results.append(dict(system=label, bank=bank, feature='q' if feature == q else feature,
                n=len(frame), original_n=len(original), clusters=len(clusters), rho=rho(values, targets),
                abs_rho=abs(rho(values, targets)), ci_low=np.nanquantile(boot, .025), ci_high=np.nanquantile(boot, .975),
                delta_vs_q=abs(rho(values, targets))-abs(rho(frame[q], targets)),
                delta_low=np.nanquantile(boot-qboot, .025), delta_high=np.nanquantile(boot-qboot, .975),
                control_mean_abs_rho=abs(rho(means[feature], means[Q])),
                within_control_rho=rho(residual[feature], residual[Q])))
        print('metrics', label, len(frame), flush=True)
    pd.DataFrame(results).to_csv(OUT / 'metrics.csv', index=False)
    by_mt = []
    for bank, original in banks.items():
        f = original
        if 'role' in f:
            f = f.query("role == 'evaluation' and eligible")
        if bank == 'kaneko':
            f = f.query("arm == 'large' and instance >= 4")
        q = 'q1' if bank == 'kaneko' else 'q'
        targets = (['Q_selected_band_power', 'Q_temporal_entropy'] if bank == 'kaneko' else
                   ['Q_R_mean'] if bank.startswith('sl') else ['Q_spin_abs'] if bank == 'mh' else
                   ['Q'] if bank == 'kuramoto_partial' else ['Q_reference'])
        for (M, T), part in f.groupby(['M', 'T']):
            for Q in targets:
                paired = part.dropna(subset=[q, Q, *FEATURES])
                for col in [q, *FEATURES]:
                    by_mt.append(dict(bank=bank, M=M, T=T, target=Q, feature='q' if col == q else col,
                                      n=len(paired), abs_rho=abs(rho(paired[col], paired[Q]))))
    pd.DataFrame(by_mt).to_csv(OUT / 'metrics_by_mt.csv', index=False)
    provenance = {'analysis': 'paired retrospective input-only comparison; no refit',
                  'bootstrap_resamples': n_boot, 'bootstrap_seed': 260917,
                  'source_scores_sha256': {path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                      for path in json.loads((OUT / 'sources.json').read_text())},
                  'input_banks': {name: len(frame) for name, frame in banks.items()},
                  'existing_baseline_reproduction': 'all overlapping baseline columns checked to atol 2e-6, rtol 2e-5'}
    (OUT / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')


if __name__ == '__main__':
    banks = prepare()
    metrics(banks)
