"""Retrospective frozen transfer using already-audited features, without new SPIs."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.finite_regime_pipeline import analyze_assembled, digest, write_json


def load_verified(corpus, cached, frozen):
    manifest = json.loads((corpus/'manifest.json').read_text())
    eligibility = json.loads((cached/'eligibility.json').read_text())
    if digest(corpus/'manifest.json') != eligibility['manifest_sha256']:
        raise ValueError('cached manifest changed')
    if digest(corpus/'observations.npz') != manifest['archive_sha256']:
        raise ValueError('observation archive changed')
    source_eligibility = json.loads((frozen/'eligibility.json').read_text())
    core_keys = ('pyspi_config_sha256', 'runner_sha256', 'compute_sha256', 'pyspi_version')
    def core(source):
        return {key:source['execution_identity'][key] for key in core_keys}
    expected = core(source_eligibility['sources'][0])
    if not all(core(s) == expected for s in eligibility['sources']+source_eligibility['sources']):
        raise ValueError('different p90 execution core')
    rows = manifest['rows']
    with np.load(cached/'features.npz', allow_pickle=False) as archive:
        z, order = archive['z'], archive['spi_order'].tolist()
        if archive['row_id'].tolist() != [r['row_id'] for r in rows]:
            raise ValueError('cached row order changed')
    if [s['row_id'] for s in eligibility['sources']] != [r['row_id'] for r in rows]:
        raise ValueError('source row order changed')
    with np.load(frozen/'model.npz', allow_pickle=False) as model:
        if model['spi_order'].tolist() != order:
            raise ValueError('frozen SPI order differs')
    if z.shape != (len(rows),len(order)*(len(order)-1)//2):
        raise ValueError('unexpected feature shape')
    return rows, z, order, eligibility['sources']


def run(corpus, cached, frozen, output):
    rows, z, order, sources = load_verified(corpus, cached, frozen)
    before = {f:digest(frozen/f) for f in ('model.npz','geometry.json','summary.json')}
    analyze_assembled(rows,z,order,sources,corpus,output,frozen,
                      status='retrospective frozen cross-size sensitivity; not independent confirmation')
    after = {f:digest(frozen/f) for f in before}
    if before != after:
        raise RuntimeError('original frozen model changed')
    write_json(output/'cached-provenance.json',dict(
        cached_features_sha256=digest(cached/'features.npz'),
        cached_eligibility_sha256=digest(cached/'eligibility.json'), frozen_files=before,
        scorer_sha256=digest(__file__), no_new_spi_extraction=True, no_refit=True,
        caveat='N and M change together; native target results were previously examined'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('corpus','cached','frozen','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    run(args.corpus,args.cached,args.frozen,args.output)
