"""Independent sampled MPI-to-bank replay for both classes and observation sizes."""
import json
import tarfile
from pathlib import Path
import numpy as np
from src.mpi_representation_baselines import summarize_mpis
from src.representation_attribution import rich_marginals
from src.representation_state_data import file_hash
from src.run_external_corpus import _array_sha256
from src.spi_spi_contract import build_unified_features


def main():
    root=Path('results/oscillatory_coorganization_pilot_260909');data=Path('data/oscillatory_coorganization_pilot_260909')
    manifest=json.loads((data/'manifest.json').read_text());rows=manifest['rows'];lookup={r['row_id']:i for i,r in enumerate(rows)}
    replay=root/'replay-mpis';replay.mkdir(exist_ok=True)
    with tarfile.open(root/'replay-mpis.tar.gz') as archive:archive.extractall(replay,filter='data')
    bank_path=root/'gadi-analysis/features.npz'
    assert file_hash(bank_path)==json.loads(bank_path.with_suffix('.json').read_text())['artifact_sha256']
    checks=[]
    with np.load(bank_path,allow_pickle=False) as bank,np.load(data/'views.npz',allow_pickle=False) as raw:
        order=bank['spi_order'].tolist();matrices={k:bank['X_'+k] for k in ['m','g','z','validity']}
        for path in sorted(replay.glob('*/meta.json')):
            meta=json.loads(path.read_text());name=meta['dataset_name'];index=lookup[name]
            assert meta['source']['member_sha256']==_array_sha256(raw[name])
            assert meta['source']['archive_sha256']==manifest['artifacts']['views.npz']
            with np.load(path.parent/'spi_mpis.npz',allow_pickle=False) as a:mpis={k:a[k] for k in order}
            _,g,valid=summarize_mpis(mpis,order)
            values=dict(m=rich_marginals(mpis,order),g=g,validity=valid,z=build_unified_features(mpis,order,metric='pearson').z)
            errors={}
            for key,value in values.items():
                np.testing.assert_allclose(value,matrices[key][index],atol=1e-10,rtol=1e-10,equal_nan=True)
                delta=abs(np.asarray(value,dtype=float)-matrices[key][index]);finite=delta[np.isfinite(delta)]
                errors[key]=float(finite.max()) if len(finite) else 0.
            checks.append(dict(row_id=name,maximum_differences=errors))
    assert len(checks)==8
    (root/'feature-verification.json').write_text(json.dumps(dict(status='passed',checks=checks,script_sha256=file_hash(Path(__file__))),indent=2)+'\n')
    print('Eight sampled raw/MPI/feature replays pass')


if __name__=='__main__':main()
