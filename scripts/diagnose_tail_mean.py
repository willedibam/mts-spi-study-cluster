"""Post-outcome diagnostic: separate robust scatter scale from correlation."""
import json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from sklearn.covariance import EllipticEnvelope
from scipy.stats import spearmanr
from scripts.scout_tail_alignment import recording,population,OUT
from scripts.spi_baseline_exploration import sha


def case(task):
    swaps,seed=task;x=recording(swaps,seed)
    covariance=EllipticEnvelope(random_state=261062).fit(x.T).covariance_
    sd=np.sqrt(np.diag(covariance));correlation=covariance/np.outer(sd,sd)
    mask=~np.eye(len(x),dtype=bool)
    return dict(swaps=swaps,seed=seed,Q=population(swaps)['Q_tail'],
        robust_cov_sq=np.mean(covariance[mask]**2),robust_cor_sq=np.mean(correlation[mask]**2),
        mean_robust_variance=np.mean(sd**2),min_robust_variance=np.min(sd**2),
        max_robust_variance=np.max(sd**2),empirical_cor_sq=np.mean(np.corrcoef(x)[mask]**2))


if __name__=='__main__':
    tasks=[(j,s) for j in range(5) for s in range(32,64)]
    with ProcessPoolExecutor(max_workers=4) as pool:rows=list(pool.map(case,tasks))
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'robust-mean-diagnostic.csv',index=False)
    metrics={k:float(abs(spearmanr(frame[k],frame.Q).statistic)) for k in
             ['robust_cov_sq','robust_cor_sq','mean_robust_variance','empirical_cor_sq']}
    result=dict(held_abs_rho=metrics,source_sha256=sha(__file__),
        qualification='Post-outcome diagnostic on already seen held seeds, not new confirmation. Independent deterministic robust fit; does not claim bit-exact pyspi estimator replay.')
    (OUT/'robust-mean-diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
