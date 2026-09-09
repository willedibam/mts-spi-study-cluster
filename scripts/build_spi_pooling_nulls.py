"""Build three verified correspondence null banks without recomputing SPIs."""
import argparse
import json
from pathlib import Path

import numpy as np

from src.representation_state_data import file_hash
from src.spi_pooling_nulls import canonical_dyads, permute_edge_columns


def main(data, bank, output):
    if output.exists():
        raise FileExistsError(output)
    rows = json.loads((data / "manifest.json").read_text())["rows"]
    meta = json.loads(bank.with_suffix(".json").read_text())
    assert meta["artifact_sha256"] == file_hash(bank)
    assert meta["manifest_sha256"] == file_hash(data / "manifest.json")
    with np.load(bank, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    np.testing.assert_array_equal(arrays["row_id"], [r["row_id"] for r in rows])
    output.mkdir(parents=True)
    checks = []
    for seed in [503, 509, 521]:
        moved = arrays["edges"].copy()
        for i, row in enumerate(rows):
            n, m = int(arrays["lengths"][i]), row["M"]
            assert n == m * (m - 1)
            values = arrays["edges"][i, :n]
            moved[i, :n] = permute_edge_columns(values, m, np.random.default_rng([seed, row["corpus_index"]]))
            np.testing.assert_array_equal(np.sort(moved[i, :n], axis=0), np.sort(values, axis=0))
            np.testing.assert_array_equal(canonical_dyads(moved[i, :n], m), canonical_dyads(values, m))
        np.testing.assert_array_equal(moved[np.arange(moved.shape[1])[None, :] >= arrays["lengths"][:, None]],
                                      arrays["edges"][np.arange(moved.shape[1])[None, :] >= arrays["lengths"][:, None]])
        path = output / f"null-{seed}.npz"
        np.savez_compressed(path, **{**arrays, "edges": moved})
        result = dict(artifact_sha256=file_hash(path), manifest_sha256=file_hash(data / "manifest.json"),
            base_bank_sha256=file_hash(bank), permutation_seed=seed, records=len(rows),
            full_column_and_reciprocal_multisets_preserved=True, validity_and_padding_unchanged=True,
            code_sha256={p: file_hash(Path(p)) for p in [__file__, "src/spi_pooling_nulls.py"]})
        path.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
        checks.append(result)
        print(f"Verified and saved null {seed}: {len(rows)} records", flush=True)
    (output / "verification.json").write_text(json.dumps(checks, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["data", "bank", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    main(args.data, args.bank, args.output)
