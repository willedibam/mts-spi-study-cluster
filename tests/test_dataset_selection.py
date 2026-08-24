from pathlib import Path

from scripts.select_dataset_indices import select_indices


def test_dataset_selection_filters_cell_and_instance_range(tmp_path: Path) -> None:
    config = tmp_path / "selection.yaml"
    config.write_text(
        """
base_output_dir: data/test-selection
timestamp: false
pyspi_config: configs/pyspi/benchmarked_p90.yaml
normalise: false
rng_seed: 7
save_heatmap: false
defaults:
  M_values: [8, 16]
  T_values: [500, 1000]
  instances: [10, 11, 12]
mts_classes:
  - name: gaussian-noise
    generator: gaussian_noise
    base_params: {zscore: false}
  - name: cauchy-noise
    generator: cauchy_noise
    base_params: {zscore: false}
""",
        encoding="utf-8",
    )

    indices = select_indices(
        str(config), M=16, T=1000, instance_min=11, instance_max=12
    )

    assert len(indices) == 4
    assert indices == sorted(indices)
