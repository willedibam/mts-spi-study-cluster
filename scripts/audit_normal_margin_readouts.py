"""Post-outcome mechanism audit; does not select another component or alter results."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
from scripts.spi_baseline_exploration import sha


def main():
    root=Path(__file__).resolve().parents[1]
    out=root/'results/order-parameter-inference/dependence-transition-261005/normal-margin-p90'
    data=root/'data/order-parameter-inference/t-copula-normal-margins-261006'
    bank=np.load(out/'features.npz');scores=pd.read_csv(out/'scores.csv')
    assert sha(out/'features.npz')==json.loads((out/'analysis.json').read_text())['features_sha256']
    with np.load(data/'observations.npz') as raw:
        expected=[np.corrcoef(raw[k])[~np.eye(16,dtype=bool)].mean() for k in scores.row_id]
    np.testing.assert_allclose(expected,scores.empirical_mean_r,atol=1e-14)
    fit=scores.role.eq('development').to_numpy();x=bank['z'];finite=np.isfinite(x[fit]).all(0)
    keep=finite.copy();keep[finite]=np.std(x[fit][:,finite],axis=0)>1e-10
    x=x[:,keep].astype(float);sd=x[fit].std(0);x-=x[fit].mean(0)
    ii,jj=np.triu_indices(len(bank['spi_order']),1)
    first=bank['spi_order'][ii[keep]];second=bank['spi_order'][jj[keep]]
    diagnostics=[];loadings=[]
    for standard in [False,True]:
        model=PCA(n_components=3,svd_solver='full').fit(x[fit]/(sd if standard else 1))
        w=model.components_[0];top=np.argsort(abs(w))[-20:][::-1]
        label='standardized' if standard else 'centered'
        diagnostics.append(dict(scaling=label,evr=model.explained_variance_ratio_.tolist(),
            eigengap_ratio=float(model.explained_variance_[0]/model.explained_variance_[1])))
        loadings.extend(dict(scaling=label,spi_a=first[j],spi_b=second[j],loading=float(w[j]),training_sd=float(sd[j])) for j in top)
    pd.DataFrame(loadings).to_csv(out/'loading-audit.csv',index=False)
    (out/'readout-audit.json').write_text(json.dumps(dict(source_sha256=sha(__file__),records=len(scores),
        features_sha256=sha(out/'features.npz'),all_observed_Pearson_replayed=True,retained_coordinates=int(keep.sum()),
        diagnostics=diagnostics,qualification='Training-only eigenspectra and largest loading magnitudes, inspected after outcomes. Not ablation or causal attribution; no readout was changed.'),indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
