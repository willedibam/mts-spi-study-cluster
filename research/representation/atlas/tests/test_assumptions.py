import json

import numpy as np

from research.representation.atlas.assumptions import classify, edges, links, parse, score

SPIS = ['cov_EmpiricalCovariance', 'cov-sq_EmpiricalCovariance', 'spearmanr', 'spearmanr-sq', 'corr_pearson_tau-1',
        'prec_EmpiricalCovariance', 'cov_MinCovDet', 'mi_gaussian', 'mi_kraskov_NN-4_DCE-AUTO',
        'cohmag_multitaper_mean_fs-1_fmin-0_fmax-0-25', 'icoh_multitaper_mean_fs-1_fmin-0_fmax-0-25',
        'psi_multitaper_mean_fs-1_fmin-0_fmax-0-25']


def test_parse_and_classify_minimal_pairs():
    assert parse('corr_pearson_tau-3-sq') == dict(stat='corr', est='Empirical', sign='sq', lag=3, band=None, agg=None, opt=None)
    assert parse('cohmag_multitaper_max_fs-1_fmin-0-25_fmax-0-5')['band'] == 'high'
    assert classify('cov_EmpiricalCovariance', 'cov-sq_EmpiricalCovariance') == 'sign handling: signed vs sq'
    assert classify('spearmanr', 'cov_EmpiricalCovariance') == 'estimator: Empirical vs spearman'
    assert classify('cov_EmpiricalCovariance', 'prec_EmpiricalCovariance') == 'mediation'
    assert classify('mi_gaussian', 'mi_kraskov_NN-4_DCE-AUTO') == 'estimator: gaussian vs kraskov_DCE'
    assert classify('cohmag_multitaper_mean_fs-1_fmin-0_fmax-0-25', 'icoh_multitaper_mean_fs-1_fmin-0_fmax-0-25') == 'zero-lag removal (coherence)'
    assert classify('cov-sq_EmpiricalCovariance', 'mi_gaussian') == 'saturation control'
    assert classify('spearmanr-sq', 'cov_EmpiricalCovariance') is None                    # two differences
    assert classify('psi_multitaper_mean_fs-1_fmin-0_fmax-0-25', 'cohmag_multitaper_mean_fs-1_fmin-0_fmax-0-25') is None
    chain = links(SPIS, 'spearmanr-sq', 'cov_EmpiricalCovariance')
    assert len(chain) == 2 and {c for c, _, _ in chain} == {'sign handling: signed vs sq', 'estimator: Empirical vs spearman'}


def test_score_finds_a_planted_class_effect(tmp_path):
    rng = np.random.default_rng(0)
    K = len(SPIS)
    iu = np.triu_indices(K, 1)
    col = {frozenset((SPIS[i], SPIS[j])): k for k, (i, j) in enumerate(zip(*iu))}
    signed = [col[frozenset((a, b))] for c, a, b in edges(SPIS) if c == 'sign handling: signed vs sq']
    rows, Z = [], []
    for u in range(6):
        for state in ('awake_ec', 'awake_eo', 'drug'):
            for w in range(8):
                z = .5 + .02 * rng.normal(size=len(iu[0]))
                if state == 'drug':
                    z[signed] -= .1
                rows.append(dict(id=f'{u}-{state}-{w}', unit=f'u{u}', agent='KT', animal=f'a{u % 2}', state=state))
                Z.append(z)
    Z = np.array(Z, dtype=np.float32)
    n = len(rows)
    np.savez(tmp_path / 'reduced.npz', rows=json.dumps(rows), spis=np.array(SPIS), valid=np.ones((n, K), bool), z=Z,
             node_r2=np.zeros((n, K)), covariates=rng.normal(size=(n, 6)),
             covariate_names=np.array(['mean_abs_r', 'sd_abs_r', 'cov_pr', 'delta_fraction', 'spectral_entropy', 'peak_hz']))
    score(tmp_path / 'reduced.npz', tmp_path / 'out.json', 'awake_ec')
    report = json.loads((tmp_path / 'out.json').read_text())
    cls = {c['cls']: c for c in report['classes']}
    sign = cls['sign handling: signed vs sq']['conditions']
    assert sign['KT:drug']['raw']['median_sd'] < -3 and sign['KT:drug']['raw']['same_sign'] == 1
    assert abs(sign['eyes_open']['raw']['median_sd']) < 3
    assert abs(cls['mediation']['conditions']['KT:drug']['raw']['median_sd']) < 3
    assert cls['sign handling: signed vs sq']['tier']['KT:drug'] == 'partial'  # two members, one statistic
    assert set(report['rankings']) >= {'all_drugs', 'all_but_one', 'only:KT:drug'}


def test_weighting_chain_and_null(tmp_path):
    from research.representation.atlas.assumptions import _weighting_profiles, weighting_null
    rng = np.random.default_rng(3)
    x = rng.normal(size=(400, 5))
    x[:, 1] += x[:, 0]                                         # zero-lag coupling: r^2 equals the phase-free version
    p = _weighting_profiles(x)
    assert p['r2'][0] <= p['power_weighted_coh'][0] < 1.25 * p['r2'][0]   # phase-free >= r^2; gap = |S| estimation bias
    rows, arrays = [], {}
    for s in ('awake_ec', 'drug'):
        for w in range(4):
            wid = f'u-{s}-{w}'
            arrays[wid] = rng.normal(size=(100, 6))
            rows.append(dict(id=wid, unit='u', state=s, M=6))
    np.savez(tmp_path / 'windows.npz', **arrays)
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(fs=100, rows=rows)))
    weighting_null(tmp_path, tmp_path / 'o.json', 'awake_ec', draws=1)
    c = json.loads((tmp_path / 'o.json').read_text())['contrasts']['all:drug']
    assert {'real:r2|low_band_coh', 'stationary:r2|low_band_coh:mean_profile', 'zero_phase:r2|power_weighted_coh'} <= set(c)
