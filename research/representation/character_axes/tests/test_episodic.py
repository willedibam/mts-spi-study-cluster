import numpy as np

from research.representation.character_axes import episodic


def gaussian(T=2000, M=12, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(M, M)) * .3
    return rng.normal(size=(T, M)) @ (np.eye(M) + a)


def episodic_window(T=2000, M=12, seed=0):
    """Bulk network plus a second, different network active in clustered 40-sample episodes (16% of time)."""
    rng = np.random.default_rng(seed)
    x = gaussian(T, M, seed)
    b = np.zeros((M, M))
    b[0, 1:4] = b[5, 6:9] = 3.0
    on = np.zeros(T, bool)
    for start in rng.choice(T - 40, 8, replace=False):
        on[start:start + 40] = True
    x[on] += rng.normal(size=(on.sum(), M)) @ b * 2
    return x


def test_mcd_flags_clustered_distinct_events_not_gaussian_noise():
    g = episodic.mcd_metrics(gaussian())
    e = episodic.mcd_metrics(episodic_window())
    assert e['fraction'] > g['fraction']
    assert e['clustering'] > .5 > g['clustering']
    assert e['distinct_network'] < -.1 and abs(g['distinct_network']) < .1


def test_burst_split_and_events():
    x = episodic_window()
    m = episodic.burst_metrics(x)
    assert {'q0_linear_rank', 'q0_marginal', 'q0_copula', 'q5_linear_rank'} <= set(m)
    assert episodic.event_metrics(x)['event_clustering'] > episodic.event_metrics(gaussian())['event_clustering']


def test_character_finds_single_channel_artefact():
    x = gaussian()
    rng = np.random.default_rng(3)
    for start in rng.choice(1900, 10, replace=False):
        x[start:start + 20, 4] += 15 * np.sin(np.arange(20) * 2 * np.pi * 30 / 100)  # brief fast bursts on one channel
    c = episodic.character_metrics(x, fs=100)
    assert c['max_channel_share'] > .5 and c['dominated'] > .5 and c['high_band_ratio'] > c['low_band_ratio']


def test_atlas_mode_end_to_end(tmp_path):
    import json
    from research.representation.atlas import prepare
    rows, arrays = [], {}
    for u in range(3):
        for state in ('awake_ec', 'drug'):
            for k in range(2):
                wid = f'u{u}:{state}:{k:02d}'
                x = episodic_window(T=400, M=20, seed=10 * u + k) if state == 'drug' else gaussian(T=400, M=20, seed=10 * u + k)
                rows.append(dict(id=wid, unit=f'u{u}', state=state, agent='PF', M=20))
                arrays[wid] = x
    prepare.write_bank(tmp_path / 'bank', 'toy', 'long', 400, rows, arrays, {})
    episodic.atlas(tmp_path / 'bank', tmp_path / 'out.json', 'awake_ec', jobs=1)
    report = json.loads((tmp_path / 'out.json').read_text())
    entry = report['contrasts']['real']['PF:drug']
    assert set(report['subsets']) == {'a', 'b'} and entry['a|fraction']['units'] == 3
    assert {'real_minus_marginal', 'real_minus_spectrum'} <= set(report['contrasts'])


def test_gain_control_removes_pure_gain_modulation():
    """A stationary process times a slow common gain gives excess flags over IAAFT; normalising the gain removes them."""
    T, fs = 2000, 100
    gain = np.exp(1.2 * np.sin(2 * np.pi * np.arange(T) / 700))[:, None]
    x = gaussian(T=T, M=12, seed=5) * gain
    raw = episodic.versions_metrics(x, 1)
    norm = episodic.versions_metrics(episodic.gain_normalise(x, fs), 1)
    excess = lambda v: v['real']['fraction'] - v['spectrum']['fraction']
    assert excess(raw) > .05 and abs(excess(norm)) < .5 * excess(raw)


def test_index_mode_and_sleep_tertiles(tmp_path):
    import json
    n = 8
    x = np.stack([gaussian(T=300, M=6, seed=k) for k in range(n)])
    np.savez(tmp_path / 'bank.npz', x=x, y=np.array([0, 1] * 4),
             row_id=np.array([f'20110112ktmdx--S1--w{k}' for k in range(n)]), group=np.array(['George'] * n))
    episodic.index('neurotycho', tmp_path / 'bank.npz', tmp_path / 'out.json', jobs=1, gain_fs=100)
    out = json.loads((tmp_path / 'out.json').read_text())
    assert out['gain_normalised'] and {'real', 'real_minus_marginal', 'real_minus_spectrum'} <= set(out['contrasts'])
    rows = [dict(unit='u', state='sleep', delta_fraction=d) for d in np.linspace(0, 1, 9)] + [dict(unit='u', state='awake_ec', delta_fraction=.1)]
    states = episodic.sleep_tertiles(rows)
    assert states[0] == 'sleep_light' and states[8] == 'sleep_deep' and states[-1] == 'awake_ec'
