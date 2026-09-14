"""Prespecified small-lattice inference and unchanged-model transfer diagnostic."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from scripts.prepare_cml2d_corpus import prepare
from scripts.cml2d_confirmation import verify_frozen
from scripts.analyze_cml2d_spi import run, corr
from src.order_parameter_analysis import clustered_bootstrap_spearman

DEVELOPMENT = list(range(260914001, 260914009))
EVALUATION = list(range(260914101, 260914133))
ANCHORS = (3.84, 3.86212, 3.89)


def export(physics, root):
    paths = sorted(physics.glob('case-*.npz'))
    assert len(paths) == 1360
    records = []
    for path in paths:
        with np.load(path, allow_pickle=False) as a:
            meta = json.loads(str(a['metadata_json']))
            x = a['observed'][:1000, 0].T
            assert meta['views'] == ['full'] and x.shape == (meta['N'], 1000)
            np.testing.assert_array_equal(a['sensor_indices'][0], np.arange(meta['N']))
            np.testing.assert_allclose(x.mean(axis=0), a['global_mean'][:1000], atol=1e-14, rtol=0)
            records.append(dict(L=meta['L'], r=meta['r'], seed=meta['seed'], Q=meta['Q'],
                half_difference=abs(meta['Q_first_half']-meta['Q_second_half']),
                block_mean_se=float(np.std(meta['Q_blocks'], ddof=1)/np.sqrt(8)),
                minimum_channel_sd=float(x.std(axis=1).min()),
                spatial_sd_mean=float(x.std(axis=0).mean()),
                Q_window=float(np.abs(x.mean(axis=0)[1::2]-x.mean(axis=0)[::2]).mean())))
            if meta['seed'] == EVALUATION[0] and meta['r'] in ANCHORS:
                target = root / f"L{meta['L']}" / 'primary-masters'
                target.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target / path.name)
    frame = pd.DataFrame(records)
    for L in (6, 8):
        arm = root / f'L{L}'
        arm.mkdir(parents=True, exist_ok=True)
        part = frame[frame.L == L]
        assert len(part) == 680 and part.r.nunique() == 17
        assert set(part.seed) == set(DEVELOPMENT + EVALUATION)
        assert (part.groupby('r').size() == 40).all()
        part.to_csv(arm / 'physics.csv', index=False)
        if (part.minimum_channel_sd <= 1e-8).any():
            raise ValueError(f'L={L}: raw constant channels; see physics.csv; no cases discarded')
        prepare(physics, arm / 'primary', arm / 'corpus.yaml', [L*L], [1000], ['full'],
            DEVELOPMENT, select_L=L)
        rows = json.loads((arm / 'primary/manifest.json').read_text())['rows']
        smoke = [str(row['corpus_index']) for row in rows
                 if row['seed'] == DEVELOPMENT[0] and row['r'] == ANCHORS[1]]
        node = [str(row['corpus_index']) for row in rows
                if row['seed'] in DEVELOPMENT and row['r'] in ANCHORS and str(row['corpus_index']) not in smoke]
        assert len(smoke) == 1 and len(node) == 23
        (arm / 'smoke-indices.txt').write_text('\n'.join(smoke)+'\n')
        (arm / 'node-indices.txt').write_text('\n'.join(node)+'\n')


def coverage_gate(frame, geometry, require_geometry=True):
    held = frame[frame.role == 'evaluation']
    dev = frame[frame.role == 'development']
    counts = held.groupby('r').eligible.agg(['sum', 'size'])
    assert len(held) == 544 and len(counts) == 17 and (counts['size'] == 32).all()
    dev_counts = dev.groupby('r').eligible.agg(['sum', 'size'])
    assert len(dev) == 136 and (dev_counts['size'] == 8).all()
    eligible = bool((~held.eligible).mean() <= .1 and (counts['sum'] >= 24).all())
    development_ok = bool((dev_counts['sum'] >= 6).all())
    geometry_ok = bool(geometry['passes_one_coordinate_gate'])
    return dict(passes=eligible and (not require_geometry or (development_ok and geometry_ok)),
        evaluation_coverage_passes=eligible, development_coverage_passes=development_ok,
        geometry_passes=geometry_ok, evaluation_excluded=int((~held.eligible).sum()),
        evaluation_planned=544, minimum_evaluation_cell=int(counts['sum'].min()),
        minimum_development_cell=int(dev_counts['sum'].min()))


def report(root, L, frozen):
    verify_frozen(frozen)
    arm = root / f'L{L}'
    for name, source in [('analysis', None), ('transfer-analysis', frozen)]:
        output = arm / name
        run(arm / 'primary', arm / 'primary/mpi/primary', output, frozen=source)
        summary = json.loads((output / 'summary.json').read_text())
        frame = pd.read_csv(output / 'scores.csv')
        gate = coverage_gate(frame, summary['geometry'], require_geometry=source is None)
        (output / 'diagnostic-gate.json').write_text(json.dumps(gate, indent=2)+'\n')
        result = dict(L=L, M=L*L, N=L*L, T=1000, gate=gate,
            mode='independent small-L fit' if source is None else 'unchanged L256 model transfer',
            status='held-out diagnostic' if gate['passes'] else 'gate failed; descriptive only')
        if gate['passes']:
            part = frame.query("role == 'evaluation' and eligible")
            sign = summary['display_sign']
            boot, within = clustered_bootstrap_spearman(sign*part.q, part.Q_reference,
                part.r, part.seed, n_resamples=2000, seed=260914)
            means = part.groupby('r')[['q', 'Q_reference']].mean()
            result.update(rho=corr(sign*part.q, part.Q_reference),
                rho_ci=np.nanquantile(boot, [.025,.975]).tolist(),
                control_mean_rho=corr(sign*means.q, means.Q_reference),
                within_r_ci=np.nanquantile(within, [.025,.975]).tolist(),
                descriptive_endpoints=summary['results'])
        (output / 'diagnostic-report.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['export', 'report'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--physics', type=Path)
    p.add_argument('--frozen', type=Path)
    p.add_argument('--L', type=int, choices=[6,8])
    a = p.parse_args()
    if a.stage == 'export':
        export(a.physics, a.root)
    else:
        report(a.root, a.L, a.frozen)
