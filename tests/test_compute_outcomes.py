"""run_pyspi keeps pyspi's per-pair refusals and SPI warnings, not only errors."""
import numpy as np

from mtsspi.compute import run_pyspi

SGC = """.statistics.spectral:
  SpectralGrangerCausality:
    labels: [directed]
    configs:
      - method: nonparametric
        fmin: 0
        fmax: 0.5
"""


def test_a_dependent_pair_is_refused_and_warned(tmp_path):
    config = tmp_path / "sgc.yaml"
    config.write_text(SGC)
    x = np.random.default_rng(0).standard_normal((256, 3))
    x[:, 2] = 2 * x[:, 0]  # linearly dependent: no spectral factor for (0, 2)
    result = run_pyspi(x, config_path=config, normalise=False)
    (spi,) = result.matrices
    assert result.errors == {} and result.not_appropriate == {}
    refused = result.pair_not_appropriate[spi]
    assert sorted(pair[:2] for pair in refused) == [["proc-0", "proc-2"], ["proc-2", "proc-0"]]
    assert "linearly dependent" in refused[0][2]
    assert any("2 of 6 pairs are undefined" in message for message in result.warnings[spi])
