"""Pack fixed spectral features for both views; no fitted preprocessing."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from src.neurotycho_pilot import spectral_features


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    rows, features, sources = [], [], []
    for path in sorted(args.input.glob('201*.json')):
        metadata = json.loads(path.read_text())
        agent = 'ktmd' if args.phase == 'source' else 'pf'
        assert agent in metadata['archive']
        if not metadata['usable']:
            continue
        with np.load(path.with_suffix('.npz')) as bank:
            x = bank['x']
            original = bank['spectral']
        for record in metadata['records']:
            if not record['quality']['accepted']:
                continue
            for m, t in [(16, 2000), (8, 1000)]:
                feature = spectral_features(x[record['array_row'], :m, -t:])
                if m == 16:
                    np.testing.assert_allclose(feature, original[record['array_row']], atol=1e-12, rtol=1e-12)
                features.append(feature)
                rows.append(dict(record_id=f'{record["archive"]}/{record["session"]}/{record["state"]}/{record["start"]}',
                                 animal=record['animal'], archive=record['archive'], y=record['target'], M=m, T=t))
        sources.append(dict(metadata=str(path), metadata_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            bank_sha256=hashlib.sha256(path.with_suffix('.npz').read_bytes()).hexdigest()))
    assert rows
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, spectral=np.asarray(features), **{k: np.asarray([r[k] for r in rows]) for k in rows[0]})
    args.output.with_suffix('.json').write_text(json.dumps(dict(phase=args.phase, rows=len(rows), sources=sources,
        sha256=hashlib.sha256(args.output.read_bytes()).hexdigest()), indent=2)+'\n')
    print('Packed', len(rows), 'spectral views', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=['source', 'evaluation'], required=True)
    main(parser.parse_args())
