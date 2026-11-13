import json

import numpy as np

from research.representation.atlas import discovery, prepare


def fake_atlas(tmp_path, M=8, T=400):
    """A bank and extract-format shards: 3 units x (awake_ec, awake_eo, drug) x 5 windows, 6 SPIs."""
    rng = np.random.default_rng(0)
    spis = ['cov_EmpiricalCovariance', 'spearmanr', 'kendalltau', 'prec-sq_LedoitWolf', 'cov-sq_LedoitWolf', 'coint_x']
    modules = ['basic', 'basic', 'basic', 'basic', 'basic', 'misc']
    rows, arrays, shard = [], {}, 0
    out = tmp_path / 'pyspi'
    out.mkdir()
    for u in range(3):
        for state in ('awake_ec', 'awake_eo', 'drug'):
            for k in range(5):
                wid = f'u{u}:{state}:{k:02d}'
                x = rng.normal(size=(T, M))
                if state == 'drug':
                    x[:, 1:] += .8 * x[:, :1]
                rows.append(dict(id=wid, unit=f'u{u}', state=state, agent='PF', M=M, delta_fraction=rng.uniform(.3, .7),
                                 start_s=100.0 * k * (2 if state == 'drug' else 1)))
                arrays[wid] = x
                c = np.corrcoef(x.T)
                slow = rng.normal(size=M)
                mats = [c, c + .05 * rng.normal(size=(M, M)), .9 * c, np.linalg.inv(c) ** 2, c ** 2,
                        slow[:, None] + slow[None, :]]  # last: purely node-additive
                for m in mats:
                    np.fill_diagonal(m, np.nan)  # as pyspi stores MPIs
                meta = dict(pyspi=dict(spis=[dict(module=f'pyspi.statistics.{m}') for m in modules]))
                np.savez(out / f'shard-{shard:05d}.npz', **{wid: np.array(mats)}, __spis__=np.array(spis),
                         __meta__=np.array(json.dumps(meta)))
                shard += 1
    prepare.write_bank(tmp_path / 'bank', 'toy', 'long', T, rows, arrays, {})
    return tmp_path / 'bank', out


def test_offdiag_stats_node_additive():
    rng = np.random.default_rng(1)
    a = rng.normal(size=6)
    _, resid, r2 = discovery.offdiag_stats(a[:, None] + a[None, :])
    assert r2 > .999 and np.abs(resid).max() < 1e-9
    _, _, r2_noise = discovery.offdiag_stats(rng.normal(size=(6, 6)))
    assert r2_noise < .8


def test_reduce_and_analyse(tmp_path):
    bank, pyspi = fake_atlas(tmp_path)
    discovery.reduce(bank, pyspi, tmp_path / 'reduced.npz', jobs=1)
    discovery.analyse(tmp_path / 'reduced.npz', tmp_path / 'report.json', 'awake_ec')
    rep = json.loads((tmp_path / 'report.json').read_text())
    assert 'coint_x' in rep['node_additive_spis']
    c = rep['contrasts']['PF:drug']
    assert c['cov_mean_abs_r']['raw']['mean_difference'] > .3 and c['cov_mean_abs_r']['raw']['units_positive'] == 3
    assert c['cov_mean_abs_r']['raw']['d'] is not None and c['effective_rank']['adjusted'] is not None
    assert c['effective_rank']['raw']['units'] == 3 and 'PF:awake_eo' in rep['contrasts']
    assert 'PF:drug' in rep['directions'] and len(rep['top_direction_pairs']['PF:drug']) > 0


def test_screen(tmp_path):
    bank, pyspi = fake_atlas(tmp_path)
    discovery.reduce(bank, pyspi, tmp_path / 'reduced.npz', jobs=1)
    discovery.screen(tmp_path / 'reduced.npz', tmp_path / 'screen.json', 'awake_ec')
    rep = json.loads((tmp_path / 'screen.json').read_text())
    assert rep['agents'] == ['PF'] and 'coint_x' in rep['excluded_node_additive']
    assert all('coint_x' not in (h['a'], h['b']) for h in rep['hits'])


def test_pairs(tmp_path):
    bank, pyspi = fake_atlas(tmp_path)
    discovery.reduce(bank, pyspi, tmp_path / 'reduced.npz', jobs=1)
    named = [('monotone', 'cov_EmpiricalCovariance', 'spearmanr'), ('absent', 'mi_gaussian', 'dcorr')]
    discovery.pairs(tmp_path / 'reduced.npz', tmp_path / 'pairs.json', 'awake_ec', named)
    rep = json.loads((tmp_path / 'pairs.json').read_text())
    assert rep['missing'] == [['absent', 'mi_gaussian', 'dcorr']] and len(rep['pairs']) == 1
    e = rep['pairs'][0]['effects']['PF:drug']
    assert e['raw']['units'] == 3 and e['adjusted'] is not None and e['spectral'] is not None
    assert np.isfinite(rep['pairs'][0]['eyes_open_null_sd'])


def test_reliability(tmp_path):
    bank, pyspi = fake_atlas(tmp_path)
    named = [('monotone', 'cov_EmpiricalCovariance', 'spearmanr'), ('absent', 'mi_gaussian', 'dcorr')]
    discovery.reliability(bank, pyspi, tmp_path / 'rel.json', 'awake_ec', named, jobs=1)
    rep = json.loads((tmp_path / 'rel.json').read_text())
    assert rep['spis'] == ['cov_EmpiricalCovariance', 'spearmanr']
    e = rep['contrasts']['PF:drug']
    assert e['cov_EmpiricalCovariance|spearmanr:raw']['units'] == 3
    assert np.isfinite(e['cov_EmpiricalCovariance|spearmanr:disattenuated']['mean'])
    assert abs(rep['reference_state']['spearmanr']['reliability']) < .2 and rep['states']['PF:drug']['spearmanr'][0] > .5


def test_explore(tmp_path):
    bank, pyspi = fake_atlas(tmp_path)
    discovery.reduce(bank, pyspi, tmp_path / 'reduced.npz', jobs=1)
    discovery.explore(tmp_path / 'reduced.npz', tmp_path / 'explore.json', 'awake_ec', ceiling=1.01)
    rep = json.loads((tmp_path / 'explore.json').read_text())
    assert rep['conditions'] == ['PF:drug'] and rep['scale'] == 'eyes-open null SD'
    assert all('coint_x' not in (h['a'], h['b']) for h in rep['single']['PF:drug'])
    assert rep['opposite'] == {} and rep['majority'] == []


def test_stability(tmp_path):
    bank, pyspi = fake_atlas(tmp_path)
    discovery.stability(bank, pyspi, tmp_path / 'stab.json', 'awake_ec', ('cov_EmpiricalCovariance', 'spearmanr'), jobs=1)
    rep = json.loads((tmp_path / 'stab.json').read_text())
    assert rep['states']['PF:drug']['median_separation_s'] > rep['states']['PF:awake_ec']['median_separation_s']
    c = rep['contrasts']['PF:drug']
    assert c['spearmanr:150-400']['units'] == 3 and c['spearmanr:150-400']['mean'] > .3  # structured drug profiles agree
    assert 'spearmanr:0-150' not in c  # drug windows are 200 s apart: no matched bin, no contrast


def test_stationary_draws_match_mean_cross_spectrum():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(6, 128, 5))
    X[:, :, 1] += np.roll(X[:, :, 0], 2, axis=1)  # lagged coupling
    S = discovery.stationary_draws(X, 4000, rng)
    F, G = np.fft.rfft(S, axis=1), np.fft.rfft((X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True), axis=1)
    cs = np.einsum('nfi,nfj->fij', F, F.conj()) / len(F)
    target = np.einsum('nfi,nfj->fij', G, G.conj()) / len(G)
    assert np.abs(cs - target).mean() / np.abs(target).mean() < .1
    assert S.shape == (4000, 128, 5)


def test_spectral_null(tmp_path):
    rng = np.random.default_rng(4)
    M, T = 10, 300
    mix = rng.normal(size=(M, M))
    stat = np.stack([rng.normal(size=(T, M)) @ mix for _ in range(8)])  # one coupling pattern
    drift = np.stack([rng.normal(size=(T, M)) @ rng.normal(size=(M, M)) for _ in range(8)])  # new pattern per window
    R = discovery._r_profiles
    null = lambda X: discovery._agreement(R(discovery.stationary_draws(X, len(X), rng))).mean()
    assert abs(discovery._agreement(R(stat)).mean() - null(stat)) < .05
    assert discovery._agreement(R(drift)).mean() < null(drift) - .3
    bank, _ = fake_atlas(tmp_path)
    discovery.spectral_null(bank, tmp_path / 'null.json', 'awake_ec', draws=2)
    rep = json.loads((tmp_path / 'null.json').read_text())
    c = rep['contrasts']['PF:drug']
    assert c['r:excess']['units'] == 3 and abs(rep['states']['PF:drug']['r:excess']) < .1


def test_gaussian_draws_and_zero_phase():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(6, 128, 4))
    X[:, :, 1] += np.roll(X[:, :, 0], 3, axis=1)
    S = discovery._mean_periodogram(X)
    D = discovery.gaussian_draws(S, 3000, 128, rng)
    assert np.abs(discovery._mean_periodogram(D) - S).mean() / np.abs(S).mean() < .1
    Z = discovery._zero_phase(S)
    assert np.allclose(np.angle(Z[:, 0, 1])[np.abs(Z[:, 0, 1]) > 1e-9], 0, atol=1e-6)


def test_lag_profiles_separate_lag_from_zero_lag():
    rng = np.random.default_rng(6)
    T = 400
    s = np.convolve(rng.normal(size=T + 50), np.ones(4) / 4, 'same')[:T + 50]
    x = np.stack([s[50:], s[50:] + .5 * rng.normal(size=T), s[47:T + 47] + .5 * rng.normal(size=T),
                  rng.normal(size=T)], 1)           # 0-1 zero lag, 0-2 lag 3, 3 independent
    prof, C = discovery.lag_profiles(x)
    a, b = np.triu_indices(4, 1)
    e = {(i, j): n for n, (i, j) in enumerate(zip(a, b))}
    assert prof['icoh_low'][e[0, 2]] > 2 * prof['icoh_low'][e[0, 1]]
    assert prof['dswpli'][e[0, 2]] > .15 > 3 * abs(prof['dswpli'][e[0, 1]])  # full band dilutes the lag
    assert prof['xcorr2_max'][e[0, 2]] > 2 * prof['cov2'][e[0, 2]]
    assert prof['pec'][e[0, 1]] > prof['pec_orth'][e[0, 1]]
    assert C.shape[1] == 6


def test_lag_null(tmp_path):
    bank, _ = fake_atlas(tmp_path)
    discovery.lag_null(bank, tmp_path / 'lag.json', 'awake_ec', draws=1, channels=6)
    rep = json.loads((tmp_path / 'lag.json').read_text())
    c = rep['contrasts']['PF:drug']
    assert c['cohmag_low|icoh_low:residual']['units'] == 3 and len(rep['channels']) == 6
    assert abs(rep['states']['PF:drug']['phase_dispersion:bias']) < .1


def test_lag_null_calibration():
    """Phase dispersion beyond the stationary null responds to window-varying lag, not to zero-lag bursts or to
    window-to-window strength change; magnitude dispersion falls with varying lag and rises with strength change.
    (The ppc|dswpli residual is not lag-specific: strength change raises it too, and it is noisy at M = 10.)"""
    rng = np.random.default_rng(0)
    T, M, N = 200, 10, 40
    gain = rng.uniform(.3, 1.2, M)

    def window(kind):
        e = rng.normal(size=T + 70)
        s = np.zeros(T + 70)
        for t in range(1, T + 70):
            s[t] = .9 * s[t - 1] + e[t]
        lags, g, a = np.zeros(M, int), np.ones(T), gain.copy()
        if kind == 'varying_lag':
            lags = rng.integers(0, 6, M)
        if kind == 'bursts':
            g = np.full(T, .15)
            st = rng.integers(0, T - 50)
            g[st:st + 50] = 1
        if kind == 'strength':
            lags, a = np.arange(M) % 4, gain * rng.uniform(.2, 1.5)
        x = np.stack([a[i] * g * s[25 - lags[i]:25 - lags[i] + T] for i in range(M)], 1)
        return x + rng.normal(size=(T, M))

    res = {}
    for kind in ('zero_lag', 'varying_lag', 'bursts', 'strength'):
        X = np.stack([window(kind) for _ in range(N)])
        real, S = discovery._lag_state(X), discovery._mean_periodogram(X)
        null = [discovery._lag_state(discovery.gaussian_draws(S, N, T, rng)) for _ in range(2)]
        res[kind] = {k: real[k] - np.mean([n[k] for n in null]) for k in real}
    pd = {k: v['phase_dispersion'] - res['zero_lag']['phase_dispersion'] for k, v in res.items()}
    assert pd['varying_lag'] > .05 and pd['bursts'] < .02 and pd['strength'] < .02
    md = {k: v['magnitude_dispersion'] - res['zero_lag']['magnitude_dispersion'] for k, v in res.items()}
    assert md['varying_lag'] < -.03 and md['strength'] > .005


def test_null_bank(tmp_path):
    bank, _ = fake_atlas(tmp_path)
    real = json.loads((bank / 'manifest.json').read_text())
    for kind in discovery.NULL_KINDS:
        out = tmp_path / kind
        discovery.null_bank(bank, out, kind, units=['u0'])
        m = json.loads((out / 'manifest.json').read_text())
        assert m['rows'] == [r for r in real['rows'] if r['unit'] == 'u0'] and m['null']['kind'] == kind
        assert {g['n'] for g in m['null']['groups'].values()} == {5}
        with np.load(out / 'windows.npz') as a, np.load(bank / 'windows.npz') as b:
            assert sorted(a.files) == [r['id'] for r in m['rows']]
            drug = np.stack([a[f'u0:drug:{k:02d}'] for k in range(5)])
            assert drug.shape == b['u0:drug:00'][None].repeat(5, 0).shape
        assert discovery._r_profiles(drug).mean() > .3  # the drug state's common drive survives
