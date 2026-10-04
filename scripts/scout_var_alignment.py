"""Cheap feasibility check: matched C/L marginals do not imply matched SPI means."""
import json
import numpy as np
import pandas as pd
from scipy.signal import stft
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from scripts.analyze_band_swap import fitted_logistic
from scripts.spi_baseline_exploration import ROOT

OUT=ROOT/'results/representation/var-alignment-scout-261004'


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(M=16,T=1000,blocks=32,training_blocks=[0,23],validation_blocks=[24,31],seed=261006,
        C='diag1, offdiag .03+.06v',L='diag.15, offdiag h*.06v; h in{-1,+1}',
        v='uniform grid[-1,1] on120 unordered edges; independent permutation per block, shared between classes',
        rationale='Illustrative heterogeneous profiles with spread comparable to T1000 estimation noise; validate stationarity and positive innovations, not parameter optimization',
        spectral_probe='Welch-style magnitude-squared coherence from Hann128/overlap64; mean over offdiagonal edges and frequencies(0,.25] and(.25,.5]',
        claim='Supporting feasibility test only; not the full289 catalogue; population C/L marginal matching is exact, spectral matching is not assumed')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    upper=np.triu_indices(16,1);mask=~np.eye(16,dtype=bool);records=[]
    for block in range(32):
        v=np.random.default_rng(np.random.SeedSequence([261006,block,99])).permutation(np.linspace(-1,1,120))
        C=np.eye(16);C[upper]=.03+.06*v;C[(upper[1],upper[0])]=C[upper]
        for h in [-1,1]:
            L=.15*np.eye(16);L[upper]=h*.06*v;L[(upper[1],upper[0])]=L[upper]
            np.testing.assert_allclose(np.sort(L[mask]),np.sort(-L[mask]),atol=1e-14)
            A=np.linalg.solve(C,L.T).T;Q=C-A@C@A.T
            radius=max(abs(np.linalg.eigvals(A)));mineig=np.linalg.eigvalsh(Q).min()
            assert radius<1 and mineig>0
            np.testing.assert_allclose(A@C@A.T+Q,C,atol=1e-12)
            rng=np.random.default_rng(np.random.SeedSequence([261006,block,h+1]))
            x=np.empty((1000,16));x[0]=rng.normal(size=16)@np.linalg.cholesky(C).T
            noise=rng.normal(size=x.shape)@np.linalg.cholesky(Q).T
            for t in range(1,1000):x[t]=A@x[t-1]+noise[t]
            x-=x.mean(axis=0);c=x.T@x/1000;l=x[1:].T@x[:-1]/999
            frequencies,_,F=stft(x.T,fs=1,nperseg=128,noverlap=64,boundary=None,padded=False)
            spectrum=np.einsum('ift,jft->ijf',F,F.conj())/F.shape[-1]
            power=np.real(spectrum[np.arange(16),np.arange(16)])
            coherence=np.abs(spectrum)**2/(power[:,None,:]*power[None,:,:])
            bands=[(frequencies>0)&(frequencies<=.25),frequencies>.25]
            records.append(dict(block=block,h=h,mean_C=c[mask].mean(),mean_L=l[mask].mean(),
                z=np.corrcoef(c[mask],l[mask])[0,1],coherence_low=coherence[mask][:,bands[0]].mean(),
                coherence_high=coherence[mask][:,bands[1]].mean(),radius=float(radius),minimum_Q_eigenvalue=float(mineig)))
    frame=pd.DataFrame(records);frame.to_csv(OUT/'features.csv',index=False);train=frame.block.lt(24);metrics=[]
    for name,columns in [('two_MPI_means',['mean_C','mean_L']),('SPI_SPI_C_L',['z']),('means_plus_spectral_means',['mean_C','mean_L','coherence_low','coherence_high'])]:
        scale=StandardScaler().fit(frame.loc[train,columns]);x=scale.transform(frame[columns]);model=fitted_logistic(x[train],frame.loc[train,'h'])
        metrics.append(dict(method=name,BA=balanced_accuracy_score(frame.loc[~train,'h'],model.predict(x[~train]))))
    pd.DataFrame(metrics).to_csv(OUT/'metrics.csv',index=False)
    print(pd.DataFrame(metrics).to_string(index=False))
    print(frame.groupby('h')[['mean_C','mean_L','z','coherence_low','coherence_high']].mean().to_string())


if __name__=='__main__':
    with threadpool_limits(limits=4):run()
