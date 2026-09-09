"""Check all target views/hashes and replay four boundary-index master realizations."""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from src.oscillatory_mechanism import draw_parameters, simulate
from src.representation_state_data import file_hash, observed_view


def main(config, data, output):
    cfg = yaml.safe_load(config.read_text())
    manifest = json.loads((data / "manifest.json").read_text())
    assert manifest["config_sha256"] == file_hash(config)
    for name, digest in manifest["artifacts"].items():
        assert file_hash(data / name) == digest
    masters = np.load(data / "masters.npy", mmap_mode="r")
    assert masters.shape == (200, 1000, 32)
    seeds = [tuple(m["seed_parts"]) for m in manifest["masters"]]
    assert len(set(seeds)) == 200 and {s[0] for s in seeds} == {260910211}
    assert len(manifest["rows"]) == 400
    assert all(r["role"] == "evaluation" for r in manifest["rows"])
    np.testing.assert_array_equal(np.bincount([m["target"] for m in manifest["masters"]]), [100, 100])
    with np.load(data / "views.npz", allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["__dataset_names__"], [r["row_id"] for r in manifest["rows"]])
        for r in manifest["rows"]:
            np.testing.assert_array_equal(archive[r["row_id"]], observed_view(masters[r["master_index"]], r["M"], r["T"]))
    replayed = []
    for index in [0, 99, 100, 199]:
        m = manifest["masters"][index]
        parameters = draw_parameters(m["seed_parts"] + [0], cfg["generator"]["settings"])
        assert all(parameters[k] == m[k] for k in parameters)
        x = simulate(bool(m["target"]), m["seed_parts"] + [1], parameters, cfg["generator"]["settings"])
        np.testing.assert_array_equal(x, masters[index])
        replayed.append(m["master_id"])
    result = dict(status="passed", all_artifact_hashes=True, exact_view_replays=400,
                  independent_master_seeds=200, exact_generator_replays=replayed,
                  manifest_sha256=file_hash(data / "manifest.json"), checker_sha256=file_hash(Path(__file__)))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["config", "data", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    main(args.config, args.data, args.output)
