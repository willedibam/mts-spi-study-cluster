"""Pack source-only normalized waveforms at the declared neural float32 boundary."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    xs, rows, sources = [], [], []
    for path in sorted(args.input.glob('201*.json')):
        meta = json.loads(path.read_text())
        if not meta['usable']:
            continue
        bank_path = path.with_suffix('.npz')
        with np.load(bank_path) as bank:
            for record in meta['records']:
                if not record['quality']['accepted']:
                    continue
                raw = bank['x'][record['array_row']].T.astype(np.float64)
                x = (raw - raw.mean(0)) / raw.std(0)
                if x.shape != (2000,16) or not np.isfinite(x).all():
                    raise ValueError('unexpected full source observation')
                xs.append(x.astype(np.float32))
                rows.append(dict(animal=record['animal'],archive=record['archive'],target=record['target'],
                    id=f'{record["archive"]}/{record["session"]}/{record["state"]}/{record["start"]}'))
        sources.append(dict(metadata=str(path),metadata_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            bank_sha256=hashlib.sha256(bank_path.read_bytes()).hexdigest(),sampling=meta.get('window_sampling','fixed16')))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,x=np.array(xs),y=np.array([r['target'] for r in rows]),
        animal=np.array([r['animal'] for r in rows]),archive=np.array([r['archive'] for r in rows]),
        record_id=np.array([r['id'] for r in rows]))
    report=dict(rows=len(rows),sources=sources,phase='KTMD source only',
        array_dtype='float32 for neural input only; SPI observations retain float64',
        sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(dict(rows=len(rows),sha256=report['sha256']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())
