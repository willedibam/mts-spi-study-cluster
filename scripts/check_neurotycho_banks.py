"""Verify assembled source banks without fitting or using evaluation data."""
import hashlib
import json
from pathlib import Path

import numpy as np

from src.utils import slugify


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    base = Path('/g/data/ql44/we2614')
    with np.load(base / 'neurotycho_neural_260910/matched.npz') as raw:
        references = {r: (int(y), a, d) for r, y, a, d in
                      zip(raw['record_id'], raw['y'], raw['animal'], raw['archive'], strict=True)}
    assert len(references) == 352
    observed = set()
    reports = []
    upper = np.triu_indices(289, 1)
    for tag, expected in [('scout', 256), ('extra', 448)]:
        folder = base / f'neurotycho_source_f64_{tag}_260910'
        provenance = json.loads((folder / 'bank.json').read_text())
        manifest = json.loads((folder / 'manifest.json').read_text())
        assert sha(folder / 'bank.npz') == provenance['sha256']
        assert sha(folder / 'manifest.json') == provenance['manifest_sha256']
        with np.load(folder / 'bank.npz') as archive:
            bank = {key: archive[key] for key in archive.files}
        assert bank['z'].shape == (expected, 41616)
        assert bank['m'].shape == (expected, 289 * 23)
        assert bank['g'].shape == (expected, 289 * 9)
        assert len(set(bank['spi_order'])) == 289
        assert len(provenance['sources']) == expected
        error = 0.0
        for i, row in enumerate(manifest['rows']):
            assert bank['row_id'][i] == row['row_id']
            record = str(bank['record_id'][i])
            assert (int(bank['y'][i]), bank['animal'][i], bank['archive'][i]) == references[record]
            m, t = int(bank['M'][i]), int(bank['T'][i])
            assert (m, t) in [(16, 2000), (8, 1000)]
            key = (record, m, t)
            assert key not in observed
            observed.add(key)
            length = m * (m - 1)
            assert bank['lengths'][i] == length
            values = bank['edges'][i, :length].astype(np.float64)
            valid = bank['validity'][i]
            assert np.isfinite(values).all()
            assert not np.any(bank['edges'][i, length:])
            assert not np.any(values[:, ~valid])
            np.testing.assert_allclose(values[:, valid].mean(0), 0, atol=2e-7)
            np.testing.assert_allclose((values[:, valid]**2).mean(0), 1, atol=2e-7)
            pair_valid = valid[upper[0]] & valid[upper[1]]
            np.testing.assert_array_equal(np.isfinite(bank['z'][i]), pair_valid)
            gram = (values.T @ values / length)[upper]
            difference = np.max(np.abs(gram[pair_valid] - bank['z'][i, pair_valid]))
            error = max(error, float(difference))
            assert difference < 2e-6
            root = folder / 'pyspi' / manifest['name'] / f'{row["corpus_index"]:04d}-{slugify(row["row_id"], "dataset")}'
            source = provenance['sources'][i]
            assert source['row_id'] == row['row_id']
            assert sha(root / 'spi_mpis.npz') == source['mpi_sha256']
            assert sha(root / 'meta.json') == source['meta_sha256']
            if i in [0, 1, expected - 2, expected - 1]:
                with np.load(root / 'spi_mpis.npz') as matrices:
                    edges = np.stack([matrices[name][~np.eye(m, dtype=bool)] for name in bank['spi_order']])
                indices = np.flatnonzero(valid)
                direct = np.corrcoef(edges[valid])
                expected_matrix = np.full((289, 289), np.nan)
                expected_matrix[np.ix_(indices, indices)] = direct
                np.testing.assert_allclose(expected_matrix[upper], bank['z'][i], atol=2e-12, rtol=2e-12, equal_nan=True)
        counts = bank['validity'].sum(1)
        reports.append(dict(bundle=tag, rows=expected, sha256=provenance['sha256'],
                            valid_spis_min=int(counts.min()), valid_spis_median=float(np.median(counts)),
                            valid_spis_max=int(counts.max()), max_edge_gram_difference=error,
                            independent_mpi_correlation_replays=4))
    assert observed == {(r, m, t) for r in references for m, t in [(16, 2000), (8, 1000)]}
    result = dict(status='passed', rows=704, independent_record_ids=352, banks=reports,
                  verifier_sha256=sha(Path(__file__)))
    output = base / 'neurotycho_source_f64_scout_260910/bank-verification.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
