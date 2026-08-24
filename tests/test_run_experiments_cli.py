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
