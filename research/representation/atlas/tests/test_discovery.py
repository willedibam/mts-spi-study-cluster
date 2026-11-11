import json

import numpy as np

from research.representation.atlas import discovery, prepare


def fake_atlas(tmp_path, M=8, T=400):
    """A bank and extract-format shards: 3 units x (awake_ec, awake_eo, drug) x 3 windows, 6 SPIs."""
    rng = np.random.default_rng(0)
    spis = ['cov_EmpiricalCovariance', 'spearmanr', 'kendalltau', 'prec-sq_LedoitWolf', 'cov-sq_LedoitWolf', 'coint_x']
    modules = ['basic', 'basic', 'basic', 'basic', 'basic', 'misc']
    rows, arrays, shard = [], {}, 0
    out = tmp_path / 'pyspi'
    out.mkdir()
    for u in range(3):
        for state in ('awake_ec', 'awake_eo', 'drug'):
            for k in range(3):
                wid = f'u{u}:{state}:{k:02d}'
                x = rng.normal(size=(T, M))
                if state == 'drug':
                    x[:, 1:] += .8 * x[:, :1]
                rows.append(dict(id=wid, unit=f'u{u}', state=state, agent='PF', M=M, delta_fraction=.5))
                arrays[wid] = x
                c = np.corrcoef(x.T)
                slow = rng.normal(size=M)
                mats = [c, c + .05 * rng.normal(size=(M, M)), .9 * c, np.linalg.inv(c) ** 2, c ** 2,
                        slow[:, None] + slow[None, :]]  # last: purely node-additive
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
    assert c['effective_rank']['raw']['units'] == 3 and 'PF:awake_eo' in rep['contrasts']
    assert 'PF:drug' in rep['directions'] and len(rep['top_direction_pairs']['PF:drug']) > 0
