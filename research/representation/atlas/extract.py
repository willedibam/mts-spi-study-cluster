"""pyspi over an atlas window bank, written as one archive per shard of windows instead of one directory per window.

A window bank is a directory holding `windows.npz` (one (T, M) array per window id; M may differ between
windows) and `manifest.json` (dataset, arm, fs, T and one row per window). Windows are taken in sorted id
order and cut into shards of `--shard-size`. Shard k computes each of its windows with
`mtsspi.compute.run_pyspi`, exactly as `mtsspi.run_corpus` does (same arguments, a fixed seed per window,
per-channel z-scoring along time), and writes `<out>/shard-<k>.npz`. Each window's MPIs are stacked as
(K, M, M) in the order of `__spis__`, and `__meta__` is a JSON record with the identity, provenance,
errors, timings and resources. A shard is complete only if it holds every assigned window under the
current identity (bank checksum, pyspi config checksum, this file's checksum and the pyspi computation
version), so a rerun skips finished shards and recomputes stale or partial ones.

    python -m research.representation.atlas.extract BANK --out DIR --shard 3 --shard-size 25 --config CONFIG
    python -m research.representation.atlas.extract BANK --out DIR --shard-size 25 --count   # number of shards
    python -m research.representation.atlas.extract BANK --out DIR --shard-size 25 --config CONFIG --audit
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from mtsspi.compute import run_pyspi
from mtsspi.provenance import (atomic_savez, file_sha256, peak_memory_mib, pyspi_commit, pyspi_version,
                               repository_provenance, worker_resources)
from mtsspi.utils import timestamp


def window_ids(bank):
    manifest = json.loads((bank / 'manifest.json').read_text())
    return sorted(row['id'] for row in manifest['rows'])


def shard_ids(bank, shard, size):
    ids = window_ids(bank)
    return ids[shard * size:(shard + 1) * size]


def seed(window):
    return int(hashlib.sha256(window.encode()).hexdigest()[:8], 16)


def identity(bank, config):
    return dict(bank_sha256=file_sha256(bank / 'windows.npz'), manifest_sha256=file_sha256(bank / 'manifest.json'),
                pyspi_config_sha256=file_sha256(config), code_sha256=file_sha256(Path(__file__)),
                pyspi_version=pyspi_version(), normalise=True)


def shard_path(out, shard):
    return out / f'shard-{shard:05d}.npz'


def complete(path, ids, ident):
    """None if the shard holds every assigned window under this identity, else the reason it does not."""
    if not path.is_file():
        return 'missing'
    try:
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive['__meta__']))
            keys = set(archive.files)
    except Exception as error:  # an unreadable archive is incomplete, not fatal
        return f'unreadable: {error}'
    if meta.get('identity') != ident:
        return 'stale identity'
    missing = [w for w in ids if w not in keys]
    return f'{len(missing)} windows missing' if missing else None


def run_shard(bank, out, shard, size, config, n_jobs=1):
    ids = shard_ids(bank, shard, size)
    if not ids:
        raise ValueError(f'shard {shard} is empty')
    ident, path = identity(bank, config), shard_path(out, shard)
    if complete(path, ids, ident) is None:
        print(f'[SKIP] shard {shard}', flush=True)
        return path
    arrays, windows, spis, spi_info = {}, {}, None, None
    with np.load(bank / 'windows.npz', allow_pickle=False) as archive:
        for w in ids:
            x = np.asarray(archive[w], dtype=np.float64)
            start = time.perf_counter()
            result = run_pyspi(x, config_path=config, normalise=True, n_jobs=n_jobs, random_seed=seed(w))
            elapsed = time.perf_counter() - start
            names = list(result.matrices)
            if spis is None:
                spis = names
                spi_info = [dict(name=i.name, directed=i.directed, labels=i.labels, family=i.family,
                                 module=i.module, class_name=i.class_name) for i in result.metadata]
            elif names != spis:
                raise RuntimeError(f'SPI order changed within shard {shard} at window {w}')
            arrays[w] = np.stack([result.matrices[n] for n in spis])
            windows[w] = dict(T=int(x.shape[0]), M=int(x.shape[1]), seed=seed(w), compute_seconds=round(elapsed, 3),
                              errors=result.errors or {}, rng_policy=result.rng_policy)
            print(f'[DONE] shard {shard} {w} M={x.shape[1]} T={x.shape[0]} seconds={elapsed:.1f} '
                  f'errors={len(result.errors or {})}', flush=True)
    meta = dict(status='complete', identity=ident, shard=shard, shard_size=size, windows=windows,
                timestamp=timestamp(), experiment=repository_provenance(),
                pyspi=dict(config=str(config), commit=pyspi_commit(), n_spis=len(spis), spis=spi_info),
                job=dict(n_jobs=n_jobs, **worker_resources(n_jobs), peak_memory_mib=peak_memory_mib()))
    out.mkdir(parents=True, exist_ok=True)
    atomic_savez(path, {**arrays, '__spis__': np.array(spis), '__meta__': np.array(json.dumps(meta))})
    error = complete(path, ids, ident)
    if error is not None:
        raise RuntimeError(f'post-write validation failed for shard {shard}: {error}')
    return path


def count(bank, size):
    return -(-len(window_ids(bank)) // size)


def audit(bank, out, size, config):
    ident, bad = identity(bank, config), {}
    for k in range(count(bank, size)):
        error = complete(shard_path(out, k), shard_ids(bank, k, size), ident)
        if error is not None:
            bad[k] = error
    print(json.dumps(dict(shards=count(bank, size), incomplete=bad)))
    return not bad


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('bank', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--shard', type=int)
    parser.add_argument('--shard-size', type=int, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--n-jobs', type=int, default=1)
    parser.add_argument('--count', action='store_true')
    parser.add_argument('--audit', action='store_true')
    a = parser.parse_args()
    if a.count:
        print(count(a.bank, a.shard_size))
    elif a.audit:
        raise SystemExit(0 if audit(a.bank, a.out, a.shard_size, a.config) else 1)
    else:
        run_shard(a.bank, a.out, a.shard, a.shard_size, a.config, a.n_jobs)
