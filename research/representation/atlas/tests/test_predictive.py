import numpy as np

from research.representation.atlas import predictive as P


def test_gauss_te_matches_linear_coupling():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    y = np.zeros(5000)
    for t in range(1, 5000):
        y[t] = .5 * y[t - 1] + .5 * x[t - 1] + rng.normal()
    # residual variance 1 with the source, 1 + .25 * var(x | y_past) without it
    assert .08 < P.gauss_te(x, y) < .14
    assert abs(P.gauss_te(y, x)) < .01


def pair_te(x, y, w=0):
    z, yy = y[1:], y[:-1]
    dist = (np.abs(z[:, None] - z[None, :]), np.abs(yy[:, None] - yy[None, :]))
    return P.ksg_te(x, dist, w)


def test_ksg_agrees_on_linear_and_sees_quadratic_coupling():
    rng = np.random.default_rng(1)
    T = 800
    x = rng.normal(size=T)
    lin, quad = np.zeros(T), np.zeros(T)
    for t in range(1, T):
        lin[t] = .5 * lin[t - 1] + .5 * x[t - 1] + rng.normal()
        quad[t] = .5 * quad[t - 1] + .5 * (x[t - 1] ** 2 - 1) + rng.normal()
    assert abs(pair_te(x, lin) - P.gauss_te(x, lin)) < .04
    assert P.gauss_te(x, quad) < .01 and pair_te(x, quad) > .05


def test_theiler_matches_pyspi_rule():
    A = np.array([P.acf(np.random.default_rng(2).normal(size=300)) for _ in range(2)])
    assert P.theiler(A, 0, 1) == max(int(2 * np.dot(A[0], A[1])), 0)


def test_readout_null_separates_quadratic_from_linear():
    rng = np.random.default_rng(3)
    pairs = P.coupled(4)
    lin = P.window_readout(P.simulate('linear_fast', 400, M=4, rng=rng), rng, marked=pairs)
    quad = P.window_readout(P.simulate('quadratic_fast', 400, M=4, rng=rng), rng, marked=pairs)
    assert quad['diff_marked'] > lin['diff_marked'] + .02
    r = P.readout_with_null(P.simulate('quadratic_fast', 400, M=4, rng=rng), seed=0, draws=1, marked=pairs)
    assert set(r) == {'real', 'null', 'residual'} and r['residual']['diff_marked'] > .02
