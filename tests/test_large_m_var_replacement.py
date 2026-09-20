import numpy as np
import pytest
from scripts.large_m_var_replacement import generate, transition_matrix


@pytest.mark.parametrize('topology',['sparse','dense'])
@pytest.mark.parametrize('m',[4,16,64])
def test_stability_density_and_direction(topology,m):
    matrix = transition_matrix(m,topology,12)
    np.testing.assert_allclose(matrix.sum(axis=1),.85)
    np.testing.assert_allclose(np.diag(matrix),.55)
    assert np.max(abs(np.linalg.eigvals(matrix))) == pytest.approx(.85)
    off = matrix-np.diag(np.diag(matrix))
    assert np.count_nonzero(off) == (m if topology=='sparse' else m*(m-1))
    assert not np.allclose(matrix,matrix.T)


@pytest.mark.parametrize('topology',['sparse','dense'])
def test_covariance_matches_independent_kronecker_solution(topology):
    x, oracle, meta = generate(4,topology,16,steps=1000)
    matrix = oracle['transition']
    independent = np.linalg.solve(np.eye(16)-np.kron(matrix,matrix),np.eye(4).ravel()).reshape(4,4)
    np.testing.assert_allclose(oracle['stationary_covariance'], independent,atol=1e-12)
    np.testing.assert_allclose(x.mean(axis=0),0,atol=1e-14)
    np.testing.assert_allclose(x.std(axis=0),1)
    assert meta['M']==meta['N']==4 and meta['lyapunov_relative_residual']<1e-10
    np.testing.assert_array_equal(x,generate(4,topology,16)[0])


def test_long_simulation_agrees_with_stationary_covariance():
    x,oracle,_=generate(4,'dense',23,steps=100000)
    raw=x*oracle['sample_sd']+oracle['sample_mean']
    np.testing.assert_allclose(np.cov(raw,rowvar=False,bias=True),oracle['stationary_covariance'],atol=.08,rtol=.03)
