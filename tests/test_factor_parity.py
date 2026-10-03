import numpy as np
from scripts.build_factor_parity import covariance,simulate,CLASSES,WEIGHTS


def test_factor_strength_controls_and_different_sign_strength_correspondence():
    mask=~np.eye(16,dtype=bool)
    for strength in [.2,.35,.5]:
        a,b=[covariance(label,strength) for label in CLASSES]
        assert np.linalg.eigvalsh(a).min()>0 and np.linalg.eigvalsh(b).min()>0
        np.testing.assert_allclose(np.diag(a),1)
        np.testing.assert_allclose(np.diag(b),1)
        np.testing.assert_allclose(np.linalg.eigvalsh(a),np.linalg.eigvalsh(b))
        np.testing.assert_allclose(np.sort(np.abs(a[mask])),np.sort(np.abs(b[mask])))
        np.testing.assert_allclose(a[mask].mean(),b[mask].mean())
        for c in [a,b]:
            np.testing.assert_allclose(np.abs(c-np.eye(16)).sum(axis=1),strength*(16*WEIGHTS[0]-1))
        assert (a[mask]<0).sum()==(b[mask]<0).sum()
        za=np.corrcoef(a[mask],a[mask]**2)[0,1]
        zb=np.corrcoef(b[mask],b[mask]**2)[0,1]
        assert za>.25 and zb<-.15


def test_factor_generator_is_reproducible_and_finite():
    a,m=simulate(CLASSES[0],3);b,n=simulate(CLASSES[0],3)
    np.testing.assert_array_equal(a,b)
    assert m==n and a.shape==(1000,16) and np.isfinite(a).all()


def test_selected_dominant_factor_matches_magnitudes_and_regularized_means():
    from sklearn.covariance import graphical_lasso
    mask=~np.eye(16,dtype=bool)
    a,b=[covariance(label,.25,[.9,.075,.025]) for label in CLASSES]
    np.testing.assert_allclose(np.sort(np.abs(a[mask])),np.sort(np.abs(b[mask])))
    np.testing.assert_allclose(np.abs(a-np.eye(16)).sum(axis=1),3.35)
    np.testing.assert_allclose(np.abs(b-np.eye(16)).sum(axis=1),3.35)
    for alpha in [.01,.1]:
        first=graphical_lasso(a,alpha=alpha);second=graphical_lasso(b,alpha=alpha)
        for x,y in zip(first,second):np.testing.assert_allclose(x[mask].mean(),y[mask].mean(),atol=1e-6)
    assert abs(np.corrcoef(a[mask],a[mask]**2)[0,1]-np.corrcoef(b[mask],b[mask]**2)[0,1])>.08


def test_longer_bank_uses_declared_fresh_rng_blocks():
    x,m=simulate(CLASSES[0],0,.25,[.9,.075,.025],2000,64)
    y,n=simulate(CLASSES[0],64,.25,[.9,.075,.025],2000,0)
    np.testing.assert_array_equal(x,y)
    assert m==n and m['rng_block']==64 and x.shape==(2000,16)
    old,_=simulate(CLASSES[0],0,.25,[.9,.075,.025],2000,0)
    assert not np.array_equal(x,old)
