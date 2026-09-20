import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from scripts.cml2d_confirmation import OLD_CONTROLS, confirmation_gate
from scripts.scout_cml2d_period_doubling import cases_from_config


def test_confirmation_is_fresh_interleaved_replication():
    path = Path(__file__).resolve().parents[1] / "configs/scout/cml2d-period-doubling-confirmation.yaml"
    config = yaml.safe_load(path.read_text())
    cases = cases_from_config(config)
    assert len(cases) == 544
    controls = sorted({case["r"] for case in cases})
    expected = sorted(OLD_CONTROLS + [(a+b)/2 for a, b in zip(OLD_CONTROLS[:-1], OLD_CONTROLS[1:])])
    np.testing.assert_allclose(controls, expected, rtol=0, atol=1e-14)
    assert {case["seed"] for case in cases} == set(range(260911101, 260911133))


def test_gate_requires_replication_and_coverage():
    frame = pd.DataFrame(dict(view=["dispersed"]*64, M=32, T=1000,
        r=np.repeat([3.84, 3.89], 32), eligible=True))
    assert confirmation_gate(frame)["passes"]
    frame.loc[:5, "eligible"] = False
    assert confirmation_gate(frame)["passes"]
    frame.loc[6:8, "eligible"] = False
    assert not confirmation_gate(frame)["passes"]


def test_launcher_uses_archive_runner_and_shape_index_files():
    source = (Path(__file__).resolve().parents[1] / "jobs/gadi/submit_cml2d_confirmation.sh").read_text()
    assert "jobs/gadi/run_dataset_farm.pbs" not in source
    assert source.count("jobs/gadi/run_external_corpus_farm.pbs") == 4
    for shape in ("m16-t500", "m16-t1000", "m32-t500"):
        assert f"indices-{shape}.txt" in source
    assert "--validate-source" in source and "CORPUS_CONFIG=" in source
