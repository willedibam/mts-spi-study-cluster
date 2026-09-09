import numpy as np
import torch

from src.representation_mechanism import permute_dyads
from src.spi_edge_pool import SPIEdgePool, pack_inputs
from src.spi_pooling_nulls import canonical_dyads, permute_edge_columns


def test_matches_independent_matrix_implementation_and_preserves_dyads():
    rng = np.random.default_rng(122)
    for m in [4, 8]:
        mask = ~np.eye(m, dtype=bool)
        matrices = {str(i): rng.normal(size=(m, m)) for i in range(5)}
        matrices["1"] += matrices["1"].T.copy()
        matrices["4"][:] = 0
        values = np.stack([a[mask] for a in matrices.values()], axis=1)
        for shared in [False, True]:
            moved = permute_edge_columns(values, m, np.random.default_rng(503), shared=shared)
            reference = permute_dyads(matrices, list(matrices), np.random.default_rng(503), shared=shared)
            np.testing.assert_array_equal(moved, np.stack([a[mask] for a in reference.values()], axis=1))
            np.testing.assert_array_equal(canonical_dyads(values, m), canonical_dyads(moved, m))


def test_common_permutation_preserves_model_output():
    torch.manual_seed(122)
    values = np.random.default_rng(122).normal(size=(56, 5)).astype(np.float32)
    valid = np.array([[True, True, False, True, True]])
    values[:, 2] = 0
    moved = permute_edge_columns(values, 8, np.random.default_rng(503), shared=True)
    model = SPIEdgePool(dict(input_width=10, edge_width=32, head_width=32)).eval()
    with torch.no_grad():
        a = model(torch.tensor(pack_inputs(values[None], valid)))
        b = model(torch.tensor(pack_inputs(moved[None], valid)))
    torch.testing.assert_close(a, b, atol=1e-6, rtol=0)
