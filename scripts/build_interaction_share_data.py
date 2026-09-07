"""Build fresh interaction-share masters, nested SPI views and raw controls."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import yaml

from scripts.check_interaction_share_feasibility import interaction_share
from scripts.check_interaction_share_references import parameters, simulate
from src.cross_mt_transfer import pooled_baseline_features
from src.interaction_share_reference import energy_share, fit_map, own_memory
from src.representation_state_data import file_hash, observed_view


def build(config_path, output):
    if output.exists():
        raise FileExistsError(output)
    config = yaml.safe_load(config_path.read_text())
    if config['generator']['N_full'] != 32:
        raise ValueError('This bounded generator uses physical N=32')
    rows, records, arrays, targets, corpus, raw_features, references, memories = [], [], [], [], {}, [], [], []
    start = time.perf_counter()
    grid = config['methods']['raw_ridge_fractions']
    for split, role, count_key in [(0, 'training_pool', 'training_masters_per_coupling'),
                                    (1, 'evaluation', 'evaluation_masters_per_coupling')]:
        for fi, family in enumerate(config['generator']['families']):
            for k, r in enumerate(config['generator']['nominal_shares']):
                for replicate in range(config['sampling'][count_key]):
                    seed = [config['sampling']['master_seed'], split, fi, k, replicate]
                    a, b, gain = parameters(r, 'independent_gain', seed + [2])
                    states, jac = simulate(family, a, b, seed + [0])
                    order = np.random.default_rng(seed + [1]).permutation(32)
                    raw = states[:1000, order]
                    target = interaction_share(jac[1000:])
                    assert np.isfinite(raw).all() and 0 <= target <= 1
                    master_id = f's{split}-{family}-k{k}-r{replicate:02d}'
                    master_index = len(arrays)
                    records.append(dict(master_id=master_id, role=role, family=family, seed_parts=seed,
                                        coupling_index=k, replicate=replicate, a=a, b=b, gain=gain,
                                        nominal_share=r, sensor_order=order.tolist(),
                                        past_share=interaction_share(jac[:1000])))
                    arrays.append(raw); targets.append(target)
                    source = config['observations']['source']
                    cells = [source] if split == 0 else [source, config['observations']['shift']]
                    for cell in cells:
                        m, t = cell['M'], cell['T']
                        view = observed_view(raw, m, t)
                        name = f'{master_id}-M{m}-T{t}'
                        corpus[name] = view
                        rows.append(dict(row_id=name, master_id=master_id, master_index=master_index,
                                         role=role, family=family, coupling_index=k, M=m, T=t,
                                         target=target, corpus_index=len(rows)+1))
                        raw_features.append(pooled_baseline_features(view)['pooled_combined'][0])
                        memories.append(own_memory(view))
                        estimates = []
                        for nonlinear in [False, True]:
                            for ridge in grid:
                                model = fit_map(view, nonlinear=nonlinear, ridge_fraction=ridge)
                                d, o = model.energies(view)
                                estimates.append(energy_share(d, o, m, 32))
                        references.append(estimates)
            print(f'Generated {role}/{family}: {len(arrays)} masters', flush=True)
    output.mkdir(parents=True)
    np.save(output/'masters.npy', np.stack(arrays), allow_pickle=False)
    np.save(output/'targets.npy', np.asarray(targets), allow_pickle=False)
    # Compatibility with the shared neural runner; these never enter its input.
    np.save(output/'observables.npy', np.asarray(memories)[:, None], allow_pickle=False)
    np.savez_compressed(output/'raw-controls.npz', X_u=np.asarray(raw_features), memory=np.asarray(memories),
                        references=np.asarray(references), ridge_fractions=np.asarray(grid),
                        row_id=np.asarray([r['row_id'] for r in rows]))
    corpus.update(__dataset_names__=np.asarray([r['row_id'] for r in rows]),
                  __labels_json__=np.asarray(['[]']*len(rows)),
                  __shapes__=np.asarray([[r['T'], r['M']] for r in rows]),
                  __axis_order__=np.asarray(['observation', 'process']))
    np.savez_compressed(output/'views.npz', **corpus)
    files = ['masters.npy','targets.npy','observables.npy','raw-controls.npz','views.npz']
    manifest = dict(config_sha256=file_hash(config_path), artifacts={p:file_hash(output/p) for p in files},
                    rows=rows, masters=records, seconds=time.perf_counter()-start,
                    status='fresh_exploratory_pilot_not_confirmation',
                    code_sha256={p:file_hash(Path(p)) for p in [__file__,
                        'scripts/check_interaction_share_references.py', 'scripts/check_interaction_share_feasibility.py',
                        'src/interaction_share_reference.py', 'src/cross_mt_transfer.py']})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(masters=len(arrays), views=len(rows), seconds=manifest['seconds'])))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args(); build(a.config,a.output)
