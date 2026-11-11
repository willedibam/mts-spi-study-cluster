import numpy as np

from research.representation.character_axes import episodic


def gaussian(T=2000, M=12, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(M, M)) * .3
    return rng.normal(size=(T, M)) @ (np.eye(M) + a)


def episodic_window(T=2000, M=12, seed=0):
    """Bulk network plus a second, different network active in clustered 40-sample episodes (16% of time)."""
    rng = np.random.default_rng(seed)
    x = gaussian(T, M, seed)
    b = np.zeros((M, M))
    b[0, 1:4] = b[5, 6:9] = 3.0
    on = np.zeros(T, bool)
    for start in rng.choice(T - 40, 8, replace=False):
        on[start:start + 40] = True
    x[on] += rng.normal(size=(on.sum(), M)) @ b * 2
    return x


def test_mcd_flags_clustered_distinct_events_not_gaussian_noise():
    g = episodic.mcd_metrics(gaussian())
    e = episodic.mcd_metrics(episodic_window())
    assert e['fraction'] > g['fraction']
    assert e['clustering'] > .5 > g['clustering']
    assert e['distinct_network'] < -.1 and abs(g['distinct_network']) < .1


def test_burst_split_and_events():
    x = episodic_window()
    m = episodic.burst_metrics(x)
    assert {'q0_linear_rank', 'q0_marginal', 'q0_copula', 'q5_linear_rank'} <= set(m)
    assert episodic.event_metrics(x)['event_clustering'] > episodic.event_metrics(gaussian())['event_clustering']
