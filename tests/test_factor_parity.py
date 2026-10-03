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
