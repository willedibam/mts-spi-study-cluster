import numpy as np
from scipy.stats import spearmanr

from src.diffusion_map import fit_diffusion_map


def test_frozen_diffusion_map_extends_training_coordinate() -> None:
    rng = np.random.default_rng(17)
    latent = np.linspace(-2.0, 2.0, 80)
    matrix = np.column_stack(
        [latent, latent**2, np.sin(latent), np.cos(2.0 * latent)]
    )
    matrix += rng.normal(0.0, 0.01, size=matrix.shape)

    model, coordinate = fit_diffusion_map(matrix, max_components=4)
    extended = model.transform(matrix)

    assert model.pca_components.shape[0] <= 4
    assert model.bandwidth > 0.0
    assert abs(spearmanr(coordinate, extended).statistic) > 0.99
