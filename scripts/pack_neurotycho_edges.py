"""Pack verified full source SPI edges for the already specified pooling model."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from src.spi_edge_pool import pack_inputs


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    pieces, sources = [], []
    for path in args.banks:
        report = json.loads(path.with_suffix('.json').read_text())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == report['sha256'] and 'KTMD' in report['phase']
        with np.load(path) as bank:
            keep = (bank['M'] == 16) & (bank['T'] == 2000)
            assert np.all(bank['lengths'][keep] == 240)
            piece = {key: bank[key][keep] for key in ['record_id', 'y', 'animal', 'archive']}
            piece['x'] = pack_inputs(bank['edges'][keep], bank['validity'][keep])
        pieces.append(piece); sources.append(dict(path=str(path), sha256=digest))
    packed = {key: np.concatenate([piece[key] for piece in pieces]) for key in pieces[0]}
    assert packed['x'].shape == (352, 240, 578) and len(set(packed['record_id'])) == 352
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **packed)
    args.output.with_suffix('.json').write_text(json.dumps(dict(phase='KTMD source only', sources=sources,
        sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(), shape=list(packed['x'].shape)), indent=2)+'\n')
    print('Packed source edges', packed['x'].shape, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--banks', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
