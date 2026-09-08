"""Post-result, two-statistic population compression diagnostic; not full-p90 proof."""
import json
from pathlib import Path
import numpy as np
from src.covariance_modulation import population_covariance
from src.representation_state_data import file_hash


def main():
    rng=np.random.default_rng(26090991);alphas=np.linspace(.05,.95,19);checks=[]
    for replicate in range(100):
        c=rng.uniform(.05,.15);d=rng.uniform(.25,.6)*rng.uniform(.8,1.2,32)
        group=np.arange(32)>=16;between=group[:,None]!=group[None,:]
        mask=~np.eye(32,dtype=bool);r=population_covariance(c,d)[mask]
        agreements=[];magnitudes=[];estimates=[]
        for alpha in alphas:
            # Wick's identity conditional on the balanced Gaussian mixture:
            # cum(X_i,X_i,X_j,X_j)=2 Var[Cov(X_i,X_j | S)].
            kappa=2*alpha**2*np.outer(d,d)*between
            agreements.append(float(np.corrcoef(r,kappa[mask])[0,1]))
            magnitudes.append(float(np.mean(kappa[mask])))
            # Oracle loadings/groups are used ONLY for this algebra verification.
            estimates.append(float(np.sqrt(np.mean(kappa[between]/(2*np.outer(d,d)[between])))))
        error=float(abs(np.asarray(estimates)-alphas).max())
        variation=float(np.ptp(agreements))
        assert variation<1e-12 and error<1e-12
        checks.append(dict(replicate=replicate,agreement_range=variation,
                           oracle_alpha_error=error,cumulant_mean_ratio=magnitudes[-1]/magnitudes[0]))
    result=dict(status='passed',seed=26090991,alphas=alphas.tolist(),checks=checks,
                scope='Population Pearson/cross-cumulant pair only; not an assertion that all289SPIs or fullz are invariant',
                interpretation='Multiplicative target signal in a dependence matrix is erased by Pearson normalization',
                checker_sha256=file_hash(Path(__file__)))
    path=Path('results/covariance_modulation_260909/population-compression-diagnostic.json')
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(dict(nuisance_draws=100,max_agreement_change=max(r['agreement_range'] for r in checks),
               max_oracle_error=max(r['oracle_alpha_error'] for r in checks),magnitude_ratio=checks[0]['cumulant_mean_ratio']))


if __name__=='__main__':main()
