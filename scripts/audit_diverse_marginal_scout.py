"""Replay inputs and focused features without opening any held data."""
import json
import argparse
import shutil
import tempfile
from pathlib import Path
from importlib.metadata import version

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from scripts.scout_diverse_marginal_match import OUT, ROOT, recording, probes, FAMILIES, bank, configurations
from scripts.spi_baseline_exploration import sha


def audit(rebuild=False):
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert protocol['source_sha256']==sha(ROOT/'scripts/scout_diverse_marginal_match.py')
    for path, digest in protocol['helper_hashes'].items():
        assert sha(ROOT/path)==digest, path
    selection=json.loads((OUT/'selection.json').read_text())
    if rebuild:
        for folder, configs, blocks, stage in [('scout',configurations(),range(4),0),
                                              ('fresh',selection['configurations'],range(64),1)]:
            target=OUT/folder/'features.npz'
            if target.exists():
                continue
            with tempfile.TemporaryDirectory() as temp:
                replay=Path(temp)
                bank(configs,blocks,stage,replay)
                pd.testing.assert_frame_equal(pd.read_csv(replay/'rows.csv'),pd.read_csv(OUT/folder/'rows.csv'))
                assert json.loads((replay/'failures.json').read_text())==[]
                shutil.copyfile(replay/'features.npz',target)
    assert selection['scout_features_sha256']==sha(OUT/'scout/features.npz')
    rows=pd.read_csv(OUT/'fresh/rows.csv')
    assert len(rows)==384 and rows.groupby('family').size().eq(64).all()
    assert json.loads((OUT/'fresh/failures.json').read_text())==[]
    with np.load(OUT/'fresh/features.npz') as archive:
        features_bank={k:archive[k] for k in archive.files}
    replays=[]
    for family in FAMILIES:
        index=rows.index[(rows.family==family)&(rows.block==0)][0]
        x, _=recording(family,selection['configurations'][family][0],0,1)
        features, _, _=probes(x)
        np.testing.assert_allclose(x.mean(1),0,atol=1e-12)
        np.testing.assert_allclose(x.var(1),1,atol=1e-12)
        for key, value in features.items():
            np.testing.assert_array_equal(value,features_bank[key][index])
        cov=np.cov(x,bias=True)
        off=~np.eye(16,dtype=bool)
        np.testing.assert_allclose(cov[off].mean(),features['mean'][0],atol=1e-12)
        np.testing.assert_allclose(np.corrcoef(cov[off],abs(cov[off]))[0,1],features['z'][0],atol=1e-12)
        replays.append(dict(family=family,bit_exact_features=True))
    report=dict(source_bindings_verified=True,source_freeze_commit='e1c39ae',
        fresh_features_sha256=sha(OUT/'fresh/features.npz'),fresh_rows_sha256=sha(OUT/'fresh/rows.csv'),
        scout_rows=216,fresh_rows=384,failures=0,probe_count=len(features_bank['names']),
        max_signed_error=float(abs(features_bank['mean'][:,0]-rows.target_signed).max()),
        max_absolute_error=float(abs(features_bank['mean'][:,1]-rows.target_absolute).max()),
        nonfinite_z=int((~np.isfinite(features_bank['z'])).sum()),replays=replays,
        package_versions={name:version(name) for name in ['numpy','scipy','scikit-learn','umap-learn']},
        scope='Six selected fresh block0 records replay bit-exact. Independent covariance and first-z formula checks. No complete p90 run or held confirmation.')
    (OUT/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Passed source bindings and six bit-exact feature replays.')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--rebuild-missing-caches',action='store_true')
    with threadpool_limits(limits=4):
        audit(parser.parse_args().rebuild_missing_caches)
