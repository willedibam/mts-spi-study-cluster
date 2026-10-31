"""Select an individual aligned SPI on fMRI sources, then test adding z.

This is one catalogue-selection comparator, not a new data/architecture grid.
The much larger selection budget is disclosed; confirmation remains untouched.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import yaml

from mtsspi.provenance import atomic_json, atomic_savez, file_sha256
from research.representation.empirical_transfer.fit_source import SPEC, fit_fmri


def source_inputs(root):
    digest = file_sha256(root / 'bank.npz')
    if digest != json.loads((root / 'bank.json').read_text())['sha256']:
        raise ValueError('bank checksum mismatch')
    with np.load(root / 'bank.npz', allow_pickle=False) as raw:
        keep = raw['role'] == 'source'
        bank = {k: raw[k][keep] for k in ['aligned', 'z', 'y', 'group', 'row_id', 'M']}
        order = raw['spi_order'].tolist()
    sizes = set(bank.pop('M'))
    if len(sizes) != 1:
        raise ValueError('aligned single-SPI comparison needs fixed channel identities/dimension')
    m = sizes.pop()
    if bank['aligned'].shape[1] != len(order) * m * (m - 1):
        raise ValueError('aligned MPI column layout mismatch')
    return bank, order, int(m * (m - 1)), digest


def save_model(root, method, fitted, report, spi_index, spi_name, edge_count, bank, digest):
    path = root / 'source-models' / f'{method}.joblib'
    if path.exists() or path.with_suffix('.json').exists():
        raise FileExistsError(path)
    fitted.update(spi_index=spi_index, spi_name=spi_name, edges_per_spi=edge_count)
    joblib.dump(fitted, path)
    report.update(method=method, excluded_animal=None, source_rows=len(bank['y']),
                  source_groups=sorted(set(bank['group'])), target_data_used=False,
                  spi_index=spi_index, spi_name=spi_name, edges_per_spi=edge_count,
                  bank_sha256=digest, model_sha256=file_sha256(path),
                  config_sha256=file_sha256(SPEC), source_runner_sha256=file_sha256(Path(__file__)),
                  selection_scope='SPI identity chosen from all catalogue SPIs on source CV; z addition retains that identity')
    atomic_json(path.with_suffix('.json'), report)


def main(root):
    output = root / 'single-spi-selection.json'
    if output.exists() or (root / 'source-models/single-spi.joblib').exists():
        raise FileExistsError('single-SPI selection already exists')
    bank, order, edges, digest = source_inputs(root)
    config = yaml.safe_load(SPEC.read_text())
    grid, predictions, best = [], [], None
    index_by_id = {row: i for i, row in enumerate(bank['row_id'])}
    for index, name in enumerate(order):
        candidate = {**bank, 'u': bank['aligned'][:, index * edges:(index + 1) * edges]}
        fitted, report = fit_fmri(candidate, 'single-spi', config)
        record = dict(spi_index=index, spi_name=name, selected=report['selected'],
                      selected_source_validation_brier=report['selected_source_validation_brier'],
                      candidate_losses=[dict(choice=r['choice'], mean_brier=r['mean_brier']) for r in report['candidates']])
        grid.append(record)
        p = np.full((len(report['candidates']), len(bank['y'])), np.nan)
        for ci, item in enumerate(report['candidates']):
            for fold in item['folds']:
                rows = [index_by_id[row] for row in fold['validation_ids']]
                p[ci, rows] = fold['probability']
        if not np.isfinite(p).all():
            raise ValueError('missing source-CV predictions')
        predictions.append(p)
        if best is None or report['selected_source_validation_brier'] < best[2]['selected_source_validation_brier']:
            best = index, fitted, report
        if (index + 1) % 32 == 0:
            print('Source single-SPI selection', index + 1, '/', len(order), flush=True)
    index, fitted, report = best
    save_model(root, 'single-spi', fitted, report, index, order[index], edges, bank, digest)
    candidate = {**bank, 'u': bank['aligned'][:, index * edges:(index + 1) * edges]}
    combined, combined_report = fit_fmri(candidate, 'single-spi-z', config)
    save_model(root, 'single-spi-z', combined, combined_report, index, order[index], edges, bank, digest)
    archive = root / 'single-spi-source-grid.npz'
    atomic_savez(archive, dict(probability=np.array(predictions), row_id=bank['row_id'], y=bank['y'],
                              group=bank['group'], spi_order=np.array(order)))
    atomic_json(output, dict(candidates=grid, selected_spi=order[index], selected_index=index,
                source_rows=bank['row_id'].tolist(), bank_sha256=digest,
                probability_archive_sha256=file_sha256(archive), target_data_used=False,
                source_runner_sha256=file_sha256(Path(__file__)),
                selection_budget=dict(spis=len(order), choices_per_spi=len(report['candidates']), folds=3),
                limitation='Catalogue-wide source selection is optimistic and has a larger tuning budget than the single z model. Only fixed-model confirmation assesses incremental utility.'))
    print(json.dumps(dict(selected_spi=order[index], source_brier=report['selected_source_validation_brier'],
                          with_z_source_brier=combined_report['selected_source_validation_brier'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    main(parser.parse_args().root)
