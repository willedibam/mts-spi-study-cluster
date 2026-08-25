import numpy as np
import pytest

from src.run_catch22_corpus import SUMMARY_NAMES, aggregate_catch22


def test_aggregated_catch22_contract() -> None:
    pytest.importorskip("pycatch22")
    time = np.linspace(0, 20, 300)
    values = np.column_stack((np.sin(time), np.cos(time), np.sin(2 * time)))
    features, schema, errors = aggregate_catch22(values)

    assert features.shape == (110,)
    assert len(schema) == 110
    assert not errors
    assert schema[:5] == [f"DN_HistogramMode_5__{name}" for name in SUMMARY_NAMES]
    assert np.isfinite(features).all()
