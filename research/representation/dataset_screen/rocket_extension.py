"""One bounded source-only ridge extension after every MiniRocket optimum hit its upper bound."""
import argparse
import json
from pathlib import Path
import numpy as np

from mtsspi.provenance import atomic_json,file_sha256
from research.representation.dataset_screen.raw import input_manifest,load_role,rocket,screen


def main(root):
    output=root/'rocket-extended';output.mkdir(exist_ok=True)
    if (output/'minirocket.json').exists(): raise FileExistsError(output/'minirocket.json')
    original_path=root/'screen-models/minirocket.json'
    original=json.loads(original_path.read_text())
    assert original['selected_alpha']==max(c['alpha'] for c in original['candidates'])
    manifest=input_manifest(root)
    x,y,groups,rows=load_role(root,manifest,'source')
    identity=dict(manifest_sha256=file_sha256(root/'manifest.json'),raw_sha256=file_sha256(root/'raw-views.npz'),
        source_rows=[r['row_id'] for r in rows],source_groups=sorted(set(groups)),target_data_used=False,
        runner_sha256=file_sha256(Path(__file__)),original_report_sha256=file_sha256(original_path),
        purpose='Post-discovery comparator adequacy check: same kernels, groups and preprocessing; add penalties 1e4, 1e5 and 1e6 because the original source optimum was 1e3 in all four datasets. No target-based model selection.')
    report=rocket(root,x,y,groups,identity,output,
                  alpha_grid=[1e-3,1e-2,1e-1,1.,10.,100.,1000.,1e4,1e5,1e6])
    previous={c['alpha']:c['mean_brier'] for c in original['candidates']}
    for candidate in report['candidates']:
        if candidate['alpha'] in previous and not np.isclose(candidate['mean_brier'],previous[candidate['alpha']],atol=1e-9,rtol=1e-9):
            raise ValueError('original source CV scores changed during the ridge extension')
    atomic_json(output/'minirocket.json',report)
    screen(root,'minirocket',report,manifest,output,destination=root/'minirocket-extended-screen.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root)
