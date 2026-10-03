import numpy as np
from src.generators import generate_cml_logistic
from scripts.calibrate_native_coupling import generate, calibrate


def test_cml_gain_matches_finite_difference_cross_channel_response():
    alpha,g=1.895,.09
    _,full=generate_cml_logistic(M=16,T=4,alpha=alpha,eps=g,transients=2000,
        rng=np.random.default_rng(17),zscore=False,return_full_lattice=True)
    # Derive the response by perturbing the states supplied to one actual map
    # update, rather than differentiating the analytical gain expression.
    def update(x):
        fx=1-alpha*x*x
        return (1-g)*fx+g/2*(np.roll(fx,1)+np.roll(fx,-1))
    numeric=[]
    for x in full:
        columns=[]
        for j in range(len(x)):
            step=np.zeros(len(x));step[j]=1e-6
            columns.append((update(x+step)-update(x-step))/(2e-6))
        jacobian=np.array(columns).T;np.fill_diagonal(jacobian,0)
        numeric.append(np.abs(jacobian[42:58]).sum(axis=1).mean())
    _,diagnostics=generate('CML',alpha,g,17,t=4)
    np.testing.assert_allclose(np.mean(numeric),diagnostics['strength'],rtol=1e-8)


def test_var_strength_is_exact_and_not_state_dependent():
    for phi in [.2,.7]:
        g,x,diagnostics,attempts=calibrate('VAR',phi,9)
        assert g==.2 and diagnostics['strength']==.2 and attempts==1
        assert np.isfinite(x).all()
