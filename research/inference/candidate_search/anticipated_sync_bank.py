"""Full-p90 check of the delayed -> anticipated synchronization (DS -> AS) candidate.

Generator: `scout_anticipated_sync.simulate` (eight noisy Stuart-Landau sender -> receiver modules, M = N = 16,
receiver detuning d as control, channel order permuted per recording). Q is the signed module-mean phase lag
arg<Z_S conj Z_R> on a disjoint future window (> 0 DS, < 0 AS). Mechanism: Dalla Porta et al. 2019, Front. Syst.
Neurosci. 13:41 (AS when the receiver's free-running frequency exceeds the sender's).

Two arms, fixed before any p90 outcome:
- narrow, d in [-.04, .04] (|Q| <= .1 rad): primary. The window was chosen after the numpy scout showed that centred
  probe-z PC1 follows sign(Q) only for small lags; first-principles reason: sign-odd z terms scale with the lag,
  sign-even (strength-like) terms with its square.
- wide, d in [-.3, .3] (|Q| <= .8 rad): sensitivity. Prediction: q follows |Q|, as the scout did.
Seeds 0-7 fit every target-blind coordinate and supervised ceiling; seeds 8-15 evaluate. Arms are analysed separately.

Stages: prepare (local, writes archive + corpus.yaml), Gadi farm, extract (after rsync of mpis/), analyze.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from research.inference.candidate_search import scout_anticipated_sync as gen

ROOT = Path(__file__).resolve().parents[3]
RUN = 'anticipated-sync-261118'
DATA = ROOT / 'data/order-parameter-inference' / RUN
OUT = ROOT / 'results/order-parameter-inference' / RUN
REMOTE = '/scratch/ql44/we2614/mts-spi-study/order-parameter-inference/' + RUN
ARMS = {'narrow': (.04, 261120), 'wide': (.3, 261121)}
CONTROLS, SEEDS, FIT_SEEDS, PYSPI_SEED = 13, 16, 8, 261122


def prepare():
    import yaml
    from research.proof.p90.spi_baseline_exploration import sha
    if (DATA / 'manifest.json').exists():
        raise FileExistsError(DATA)
    DATA.mkdir(parents=True, exist_ok=True)
    rows, raw, gates = [], {}, {}
    for arm, (dmax, seed) in ARMS.items():
        d = np.round(np.linspace(-dmax, dmax, CONTROLS), 4)
        x, q, lock, order, q_obs = gen.simulate(d, SEEDS, seed=seed, observed_q=True)
        _, q_half, _, _ = gen.simulate(d, SEEDS, seed=seed, dt=gen.DT / 2)   # half-step refinement
        gates[arm] = dict(lock_min=float(lock.min()), max_abs_Q_future_minus_observed=float(abs(q - q_obs).max()),
                          max_abs_control_mean_Q_halfstep=float(abs(q.mean(1) - q_half.mean(1)).max()),
                          Q_seed_sd_max=float(q.std(1).max()), Q_control_means=q.mean(1).round(4).tolist())
        for c in range(CONTROLS):
            for s in range(SEEDS):
                name = f'as-{arm}-c{c:02d}-s{s:02d}'
                y = x[c, s][:, order[c, s]]
                y = (y - y.mean(0)) / y.std(0)
                assert y.shape == (gen.T, 2 * gen.MODULES) and np.isfinite(y).all()
                raw[name] = y.T
                rows.append(dict(row_id=name, corpus_index=len(rows), arm=arm, M=2 * gen.MODULES, N=2 * gen.MODULES, T=gen.T,
                                 seed=s, control=float(d[c]), role='development' if s < FIT_SEEDS else 'evaluation',
                                 Q=float(q[c, s]), absQ=float(abs(q[c, s])), Q_observed=float(q_obs[c, s]),
                                 lock_min=float(lock[c, s]), channel_order=order[c, s].tolist(),
                                 mean_abs_r=float(abs(np.corrcoef(y.T)[~np.eye(len(y.T), dtype=bool)]).mean())))
    np.savez_compressed(DATA / 'observations.npz', **raw, __dataset_names__=np.array(list(raw)),
                        __labels_json__=np.array([json.dumps([RUN, r['arm']]) for r in rows]),
                        __shapes__=np.array([[2 * gen.MODULES, gen.T]] * len(rows)), __axis_order__=np.array(['process', 'observation']))
    manifest = dict(rows=rows, corpus=RUN, archive_sha256=sha(DATA / 'observations.npz'), generator_sha256=sha(Path(gen.__file__)),
                    bank_sha256=sha(Path(__file__)), physics_gates=gates,
                    parameters=dict(modules=gen.MODULES, omega_range=gen.OMEGA_RANGE, K=gen.K, noise=gen.NOISE, sensor=gen.SENSOR,
                                    dt=gen.DT, burn=gen.BURN, future=gen.FUTURE, sample_interval=gen.STEP,
                                    channels='Re Z, sender k and receiver k, order permuted per recording, z-scored'),
                    analysis_scope=__doc__)
    (DATA / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    config = dict(name=RUN, source=dict(format='named-npz-v1', archive=REMOTE + '/observations.npz', sha256=manifest['archive_sha256'],
                  axis_order=['process', 'observation']), base_output_dir=REMOTE + '/mpis', pyspi_config='benchmarked_p90',
                  normalise=False, random_seed=PYSPI_SEED)
    (DATA / 'corpus.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    (DATA / 'smoke-indices.txt').write_text(f'1\n{len(rows)}\n')
    (DATA / 'rest-indices.txt').write_text(''.join(f'{i}\n' for i in range(2, len(rows))))
    print(json.dumps(gates, indent=1), len(rows), manifest['archive_sha256'])


def extract():
    """Hash-checked MPIs -> per-SPI means, seven-summary distributions and ordered-contract z (no fixed SPI count)."""
    from mtsspi.provenance import array_sha256
    from mtsspi.spi_spi_contract import build_unified_feature_values
    from mtsspi.utils import slugify
    from research.proof.p90.spi_baseline_exploration import sha, summarize
    manifest = json.loads((DATA / 'manifest.json').read_text()); rows = manifest['rows']
    means, dists, zs, antis, sources, order = [], [], [], [], [], None
    with np.load(DATA / 'observations.npz') as raw:
        assert sha(DATA / 'observations.npz') == manifest['archive_sha256']
        for r in rows:
            folder = DATA / 'mpis' / RUN / f"{r['corpus_index'] + 1:04d}-{slugify(r['row_id'], 'dataset')}"
            meta = json.loads((folder / 'meta.json').read_text()); p = folder / 'spi_mpis.npz'
            assert meta['status'] == 'complete' and meta['dataset_name'] == r['row_id']
            assert meta['source']['member_sha256'] == array_sha256(raw[r['row_id']])
            names = [x['name'] for x in meta['pyspi']['spis']]
            order = order or names
            assert names == order
            with np.load(p) as a:
                mpis = {k: a[k] for k in order}
            marginal = summarize(mpis, order)
            z, _, _ = build_unified_feature_values(mpis, order)
            means.append(marginal[:, 0]); dists.append(marginal.reshape(-1)); zs.append(z)
            pos = np.argsort(r['channel_order']); m = gen.MODULES  # pos[physical] = displayed index; senders 0..m-1
            antis.append([np.mean([mpis[k][pos[i], pos[i + m]] - mpis[k][pos[i + m], pos[i]] for i in range(m)]) for k in order])
            sources.append(dict(row_id=r['row_id'], mpi_sha256=sha(p), pyspi=meta['pyspi'].get('version'), commit=meta['pyspi'].get('commit')))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / 'features.npz', mean=means, distribution=dists, z=zs, anti=antis, row_id=np.array([r['row_id'] for r in rows]),
                        spi_order=np.array(order))
    (OUT / 'sources.json').write_text(json.dumps(dict(sources=sources, n_spis=len(order), manifest_sha256=sha(DATA / 'manifest.json'),
                                      features_sha256=sha(OUT / 'features.npz'), code_sha256=sha(Path(__file__))), indent=1) + '\n')


def analyze():
    from research.inference.candidate_search.report_dependence_transition import coordinate
    from research.inference.candidate_search.dependence_transition_pipeline import predict_means
    rows = pd.DataFrame(json.loads((DATA / 'manifest.json').read_text())['rows'])
    bank = np.load(OUT / 'features.npz'); np.testing.assert_array_equal(bank['row_id'], rows.row_id)
    metrics, scores = [], []
    for arm, part in rows.groupby('arm', sort=False):
        ix = part.index.to_numpy(); fit = part.role.eq('development').to_numpy(); held = ~fit
        out = part.copy(); info = {}
        for label, key, standard in [('z_PC1', 'z', False), ('z_standard_PC1', 'z', True), ('mean_PC1', 'mean', True),
                                     ('distribution_PC1', 'distribution', True)]:
            q, evr, missing = coordinate(bank[key][ix], fit, standard)
            out[label] = q * (1 if spearmanr(q[fit], part.control[fit]).statistic >= 0 else -1)
            info[label] = dict(evr=round(evr, 4), max_missing_held=float(missing[held].max()))
        for label, nonlinear in [('mean_ridge', False), ('mean_RBF', True)]:
            out[label], info[label] = predict_means(bank['mean'][ix], part.Q.to_numpy(), part.seed.to_numpy(), fit, nonlinear)
        for label in ['z_PC1', 'z_standard_PC1', 'mean_PC1', 'distribution_PC1', 'mean_abs_r', 'mean_ridge', 'mean_RBF']:
            h = out[held]
            metrics.append(dict(arm=arm, readout=label, abs_rho_Q=abs(spearmanr(h[label], h.Q).statistic),
                                abs_rho_absQ=abs(spearmanr(h[label], h.absQ).statistic), **{k: v for k, v in info.get(label, {}).items() if k in ('evr',)}))
        scores.append(out)
    metrics = pd.DataFrame(metrics); pd.concat(scores).drop(columns='channel_order').to_csv(OUT / 'scores.csv', index=False)
    metrics.to_csv(OUT / 'metrics.csv', index=False)
    print(metrics.round(3).to_string(index=False))


if __name__ == '__main__':
    stage = argparse.ArgumentParser(); stage.add_argument('stage', choices=['prepare', 'extract', 'analyze'])
    {'prepare': prepare, 'extract': extract, 'analyze': analyze}[stage.parse_args().stage]()
