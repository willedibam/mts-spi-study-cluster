from src.run_experiments import _heatmap_enabled, parse_args


def test_heatmap_cli_explicitly_overrides_config() -> None:
    assert _heatmap_enabled(None, True)
    assert not _heatmap_enabled(None, False)
    assert not _heatmap_enabled(False, True)
    assert _heatmap_enabled(True, False)


def test_heatmap_cli_default_is_unspecified() -> None:
    assert parse_args(["--experiment-config", "example.yaml"]).heatmap is None
    assert not parse_args(
        ["--experiment-config", "example.yaml", "--no-heatmap"]
    ).heatmap
    assert parse_args(
        ["--experiment-config", "example.yaml", "--heatmap"]
    ).heatmap


def test_finished_dataset_from_another_pyspi_revision_stops_the_run(tmp_path) -> None:
    """--skip-existing must not top up a corpus with a different revision."""
    import json
    from types import SimpleNamespace

    import pytest

    from src.run_experiments import _dataset_complete, file_sha256, pyspi_version

    config = tmp_path / "spis.yaml"
    config.write_text(".statistics.basic:\n  Covariance:\n    labels: [undirected]\n")
    spec = SimpleNamespace(dataset_dir=tmp_path / "dataset", pyspi_config=config,
                           generator="varma", generator_params={})
    spec.dataset_dir.mkdir()
    assert not _dataset_complete(spec)  # nothing written yet: run it

    def write(computation, config_sha256):
        (spec.dataset_dir / "meta.json").write_text(json.dumps({"pyspi": {
            "version": {"computation": computation}, "config_sha256": config_sha256}}))

    for name in ("spi_mpis.npz", "timeseries.npy"):
        (spec.dataset_dir / name).touch()
    current = pyspi_version()["computation"]
    write(current, file_sha256(config))
    assert _dataset_complete(spec)
    write("3.0.0.r7", file_sha256(config))
    with pytest.raises(RuntimeError, match="3.0.0.r7"):
        _dataset_complete(spec)
    write(current, "0" * 64)
    with pytest.raises(RuntimeError, match="config 000000000000"):
        _dataset_complete(spec)
    # Metadata from before the version was recorded is stale too.
    (spec.dataset_dir / "meta.json").write_text("{}")
    with pytest.raises(RuntimeError, match="pyspi None"):
        _dataset_complete(spec)
