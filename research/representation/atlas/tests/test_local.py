import json

import numpy as np

from research.representation.atlas import local as L


def test_node_stats_matches_brute_force():
    rng = np.random.default_rng(0)
    M = 6
    rr, cc = np.nonzero(~np.eye(M, dtype=bool))
    a, b = rng.normal(size=len(rr)), rng.normal(size=len(rr))
    z, sd, _ = L.node_stats(a, b, rr, cc, M)
    for i in range(M):
        m = (rr == i) | (cc == i)
        assert np.isclose(z[i], np.corrcoef(a[m], b[m])[0, 1])
        assert np.isclose(sd[i], a[m].std())


def test_local_finds_a_planted_hub(tmp_path):
    """Pair agreement broken at two channels in the drug state only: the upper/lower held-out excess must flag it."""
    rng = np.random.default_rng(1)
    M, T = 12, 200
    spis = list(L.REFERENCE_PAIR)
    rows, arrays, shard = [], {}, {}
    for u in range(4):
        for s in ('awake_ec', 'awake_eo', 'drug'):
            for w in range(6):
                wid = f'u{u}-{s}-{w}'
                a = rng.normal(size=(M, M))
                a = a + a.T
                b = a + .3 * rng.normal(size=(M, M))
                if s == 'drug':
                    b[:2, :] = rng.normal(size=(2, M))
                    b[:, :2] = b[:2, :].T
                shard[wid] = np.stack([a, b])
                arrays[wid] = rng.normal(size=(T, M))
                rows.append(dict(id=wid, unit=f'u{u}', state=s, M=M, agent='X', animal=f'a{u % 2}'))
    (tmp_path / 'pyspi').mkdir()
    np.savez(tmp_path / 'pyspi' / 'shard-0.npz', __spis__=np.array(spis), **shard)
    np.savez(tmp_path / 'windows.npz', **arrays)
    units = [dict(unit=f'u{u}', kept_channels=list(range(1, M + 1))) for u in range(4)]
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(fs=100, rows=rows, preprocessing=dict(units=units))))
    (tmp_path / 'rel.json').write_text(json.dumps(dict(reference_state={x: dict(reliability=.9) for x in spis})))
    L.local(tmp_path, tmp_path / 'pyspi', tmp_path / 'rel.json', tmp_path / 'out.json', 'awake_ec', jobs=1, k=2)
    e = json.loads((tmp_path / 'out.json').read_text())['pairs'][0]['conditions']['X:drug']
    assert e['lower']['passed'] == 4 and e['lower']['mean'] > .2 and e['local_signature']


def test_sign_null_runs_on_a_tiny_bank(tmp_path):
    from research.representation.atlas import discovery as D
    rng = np.random.default_rng(2)
    rows, arrays = [], {}
    for s in ('awake_ec', 'drug'):
        for w in range(4):
            wid = f'u-{s}-{w}'
            arrays[wid] = rng.normal(size=(100, 6))
            rows.append(dict(id=wid, unit='u', state=s, M=6))
    np.savez(tmp_path / 'windows.npz', **arrays)
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(fs=100, rows=rows)))
    D.sign_null(tmp_path, tmp_path / 'o.json', 'awake_ec', draws=2)
    c = json.loads((tmp_path / 'o.json').read_text())['contrasts']['all:drug']
    assert {'real', 'real_mean_profile', 'null', 'null_mean_profile'} <= set(c)
