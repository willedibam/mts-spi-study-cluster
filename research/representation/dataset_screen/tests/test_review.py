import numpy as np
from research.representation.dataset_screen.review import grouped_scores, aggregate, bootstrap, contrast


def test_review_does_not_count_repeated_windows_as_independent_people():
    y=np.array([0,1,0,1]);g=np.array(['a','a','b','b'])
    p=np.array([[.9,.1],[.1,.9],[.5,.5],[.5,.5]])
    a,s=grouped_scores(y,p,g)
    repeated=np.r_[np.tile([0,1],20),[2,3]]
    b,t=grouped_scores(y[repeated],p[repeated],g[repeated])
    for group in a:
        np.testing.assert_allclose(list(a[group].values()),list(b[group].values()),atol=1e-15)
    np.testing.assert_allclose(aggregate([v['brier'] for v in a.values()],s),.13)
    np.testing.assert_allclose(bootstrap([v['brier'] for v in a.values()],s),
                             bootstrap([v['brier'] for v in b.values()],t),atol=1e-15)
    result=contrast(a,b,s)
    np.testing.assert_allclose(result['brier']['conditional_95_interval'],[0.,0.],atol=1e-15)


def test_review_bootstraps_whole_configurations_within_class():
    g=np.array(['a','a','b','c','d']);y=np.array([0,0,0,1,1])
    p=np.array([[1.,0.],[1.,0.],[.8,.2],[.1,.9],[.3,.7]])
    scores,strata=grouped_scores(y,p,g)
    values=np.array([v['brier'] for v in scores.values()])
    np.testing.assert_allclose(aggregate(values,strata),.035)
    draws=bootstrap(values,strata)
    assert draws.min()>=.005-1e-12 and draws.max()<=.065+1e-12
