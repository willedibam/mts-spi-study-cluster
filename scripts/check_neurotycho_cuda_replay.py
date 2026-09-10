"""Source-only numerical diagnostic; never loads PF data or changes a fit."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.neurotycho_learning import predict_binary
from src.representation_state_neural import make_encoder, seed_torch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    torch.set_num_threads(2)
    seed_torch(1729)  # Same precision flags as training, including no TF32.
    assert torch.cuda.is_available()
    with np.load(args.data / 'dense.npz') as bank:
        source = {k: bank[k] for k in ['x', 'record_id', 'animal']}
    lookup = {r: i for i, r in enumerate(source['record_id'])}
    rows = []
    for path in sorted(args.fits.glob('enriched-*.json')):
        if '.progress.' in path.name:
            continue
        report = json.loads(path.read_text())
        assert sha(path.with_suffix('.pt')) == report['checkpoint_sha256']
        assert sha(path.with_suffix('.npz')) == report['predictions_sha256']
        assert sha(args.data / 'dense.npz') == report['identity']['inputs']['dense.npz']
        checkpoint = torch.load(path.with_suffix('.pt'), map_location='cpu', weights_only=True)
        model = make_encoder(checkpoint['spec'])
        model.load_state_dict(checkpoint['state_dict'])
        with np.load(path.with_suffix('.npz')) as saved:
            ix = np.array([lookup[r] for r in saved['training_ids']])
            assert np.all(source['animal'][ix] != report['identity']['animal'])
            expected = saved['probability'].copy()
        x = torch.from_numpy(source['x'][ix])
        cpu = predict_binary(model, x)
        gpu = predict_binary(model.cuda(), x.cuda())
        # Audit every source prediction on both devices; inspect the largest
        # CPU/GPU disagreements in double precision without selecting on PF.
        worst = np.argsort(np.abs(cpu - expected))[-32:]
        double = predict_binary(model.cpu().double(), x[worst].double())
        row = dict(model=path.name, checkpoint_sha256=report['checkpoint_sha256'],
                   source_rows=len(ix), cpu_saved_max=float(np.max(np.abs(cpu-expected))),
                   cuda_saved_max=float(np.max(np.abs(gpu-expected))),
                   cpu_cuda_max=float(np.max(np.abs(cpu-gpu))),
                   cpu_saved_threshold_disagreements=int(np.sum((cpu >= .5) != (expected >= .5))),
                   diagnostic_ids=source['record_id'][ix[worst]].tolist(),
                   cpu_double_max=float(np.max(np.abs(cpu[worst]-double))),
                   cuda_double_max=float(np.max(np.abs(gpu[worst]-double))),
                   saved_double_max=float(np.max(np.abs(expected[worst]-double))))
        rows.append(row)
        print(json.dumps(row), flush=True)
    assert len(rows) == 6
    args.output.write_text(json.dumps(dict(torch=torch.__version__,
        gpu=torch.cuda.get_device_name(0), tf32=False, target_data_used=False,
        script_sha256=sha(Path(__file__)), models=rows), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--fits', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
