"""HCP one-person feasibility: frozen PC1, observation stability, simple controls."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import welch
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from src.run_external_corpus import ExternalCorpusConfig, load_inventory, completion_error
from src.spi_spi_contract import build_unified_features, schema_sha256
from src.representation_attribution import rich_marginals


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_pc1(features, train):
    a = np.asarray(features, dtype=float)
    mask = np.isfinite(a[train]).sum(axis=0) >= len(train)-1
    selected = a[:, mask]
    medians = np.nanmedian(selected[train], axis=0)
    filled = np.where(np.isfinite(selected), selected, medians)
    means, scales = filled[train].mean(axis=0), filled[train].std(axis=0)
    keep = scales > 1e-10
    assert keep.any()
    normalized = (filled[:, keep]-means[keep])/scales[keep]
    _, values, vectors = np.linalg.svd(normalized[train], full_matrices=False)
    loading = vectors[0].copy()
    if loading[np.argmax(np.abs(loading))] < 0:
        loading *= -1
    q = normalized @ loading
    score_scale = q[train].std()
    q /= score_scale
    return q, normalized, dict(retained_features=int(keep.sum()), training_explained_variance=float(values[0]**2/np.sum(values**2)),
                              sign_rule='Largest absolute loading positive; no label-based orientation'), dict(mask=mask, medians=medians, means=means, scales=scales, keep=keep, loading=loading, score_scale=score_scale)


def main(root):
    config = ExternalCorpusConfig.from_file(root/'external.yaml')
    manifest = json.loads((root/'manifest.json').read_text())
    assert sha(config.archive) == manifest['archive_sha256']
    entries = load_inventory(config)
    rows = manifest['rows']
    assert len(entries) == len(rows) == 64
    data = np.load(config.archive, allow_pickle=False)
    arrays = {'z': [], 'm': [], 'spectra': []}
    sources, order = [], None
    for entry, row in zip(entries, rows, strict=True):
        assert entry.name == row['name']
        error = completion_error(config, entry)
        assert error is None, error
        directory = entry.output_dir(config)
        meta = json.loads((directory/'meta.json').read_text())
        names = [s['name'] for s in meta['pyspi']['spis']]
        if order is None:
            order = names
        assert names == order and len(order) == 289
        with np.load(directory/'spi_mpis.npz') as bank:
            matrices = {name: bank[name] for name in order}
        features = build_unified_features(matrices, order, metric='pearson')
        arrays['z'].append(features.z)
        arrays['m'].append(rich_marginals(matrices, order))
        x = data[entry.name]
        f, power = welch(x, fs=250, nperseg=1000, axis=0)
        total = power[(f >= 1)&(f <= 100)].sum(axis=0)
        ratios = np.array([power[(f >= lo)&(f < hi)].sum(axis=0)/total for lo, hi in [(1,4),(4,8),(8,13),(13,30),(30,45),(65,100)]])
        arrays['spectra'].append(np.quantile(ratios,[.1,.5,.9],axis=1).ravel())
        sources.append(dict(name=entry.name, finite_z=int(np.isfinite(features.z).sum()),
                            mpi_sha256=sha(directory/'spi_mpis.npz'), meta_sha256=sha(directory/'meta.json'),
                            compute_seconds=meta['job']['compute_seconds']))
    arrays = {k:np.asarray(v) for k,v in arrays.items()}
    primary = [i for i,r in enumerate(rows) if r['layout']=='coverage_a' and r['T']==4000]
    train, test = primary[:8], primary[8:]
    assert len(train)==len(test)==8
    output=root/'analysis';output.mkdir(exist_ok=True)
    summaries, scores, models = {}, {}, {}
    for name, features in arrays.items():
        q, space, info, model = frozen_pc1(features, train)
        scores[name] = q
        models.update({name+'_'+k:v for k,v in model.items()})
        info['held_later_blocks'] = dict(load_auc_arbitrary_orientation=float(roc_auc_score([rows[i]['memory']==2 for i in test],q[test])),
                                        image_auc_arbitrary_orientation=float(roc_auc_score([rows[i]['image']==2 for i in test],q[test])),
                                        chronological_spearman=float(spearmanr(np.arange(8),q[test]).statistic))
        comparisons=[]
        for layout,length in [('coverage_b',4000),('coverage_a',2000),('coverage_b',2000)]:
            alt=[next(i for i,r in enumerate(rows) if r['block']==rows[j]['block'] and r['layout']==layout and r['T']==length) for j in primary]
            refdistance=pdist(space[primary])
            comparisons.append(dict(layout=layout,T=length,
                                    block_distance_spearman=float(spearmanr(refdistance,pdist(space[alt])).statistic),
                                    median_displacement_over_between_block_distance=float(np.median(np.linalg.norm(space[alt]-space[primary],axis=1))/np.median(refdistance)),
                                    q_spearman_all_blocks=float(spearmanr(q[primary],q[alt]).statistic),
                                    q_spearman_held_blocks=float(spearmanr(q[test],q[alt[8:]]).statistic)))
        info['observation_comparisons']=comparisons
        summaries[name]=info
    np.savez_compressed(output/'feature-bank.npz',**arrays,names=np.array([r['name'] for r in rows]),spi_order=np.array(order))
    np.savez_compressed(output/'pc1-models.npz',**models)
    report=dict(status='complete',scope='One-participant feasibility diagnostic; no cross-person transfer or physical order-parameter claim.',
                training_names=[rows[i]['name'] for i in train],evaluation_names=[rows[i]['name'] for i in test],
                pc1_rule='First component only, no component selection using labels; labels used only for descriptive held-block associations.',
                limitations=['Eight fit blocks and eight held blocks from one run; correlations are descriptive.',
                             'The 8-second window is contained in the 16-second window, so agreement is partly due to shared data.',
                             'Block centres do not measure onset timing or a continuous phase transition.',
                             'PCA scaling and orientation frozen on primary training blocks.'],
                summaries=summaries,sources=sources,
                manifest_sha256=sha(root/'manifest.json'),script_sha256=sha(Path(__file__)))
    # Schema is determined by the frozen SPI ordering, independently of feature family.
    report['schema_sha256']=schema_sha256(build_unified_features(matrices,order,metric='pearson').schema)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'scores.json').write_text(json.dumps([dict(r,**{k:float(v[i]) for k,v in scores.items()}) for i,r in enumerate(rows)],indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root)
