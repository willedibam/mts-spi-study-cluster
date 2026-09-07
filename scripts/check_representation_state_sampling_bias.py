"""Physics-informed sampling correction; diagnostic, not a tuned comparator.

For a uniform size-M sample without replacement from N unit phasors,
E[(M r_M^2 - 1)/(M-1)] equals the full-population mean pairwise cosine.
Consequently q_N=1/N+(1-1/N)*(M*r_M^2-1)/(M-1) is unbiased for r_N^2
when phases are known. Analytic phases estimated from cosines, clipping, square
roots and future averaging do NOT inherit that unbiasedness guarantee.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.signal import hilbert
import yaml

from src.representation_screen import evaluation_cells
from src.representation_state_data import file_hash, load_state_data, observed_view


def corrected_squared_coherence(r_squared, m, n):
    if not 2 <= m <= n:
        raise ValueError("require 2 <= observed count <= population count")
    return 1 / n + (1 - 1 / n) * (m * r_squared - 1) / (m - 1)


def check(config, data, output):
    protocol = yaml.safe_load(config.read_text())
    manifest, masters = load_state_data(data, config)
    # Exhaustively verify the finite-population identity on a disjoint fixture.
    phase = np.asarray([0., .2, .8, 1.3, 2.5, 4.1])
    unit = np.exp(1j * phase)
    for m in range(2, len(unit) + 1):
        estimates = [corrected_squared_coherence(abs(unit[list(subset)].mean()) ** 2, m, len(unit))
                     for subset in itertools.combinations(range(len(unit)), m)]
        np.testing.assert_allclose(np.mean(estimates), abs(unit.mean()) ** 2, atol=1e-14)
    rows = [r for r in manifest["rows"] if r["role"] == "evaluation"]
    prediction = []
    for row in rows:
        raw = observed_view(masters[row["master_index"]], row["M"], row["T"])
        analytic = hilbert(raw, axis=0)
        unit = analytic / np.maximum(np.abs(analytic), np.finfo(float).eps)
        r2 = abs(unit.mean(axis=1)) ** 2
        corrected = corrected_squared_coherence(r2, row["M"], protocol["generator"]["N_full"])
        prediction.append(float(np.sqrt(np.clip(corrected, 0, 1)).mean()))
    target = np.asarray([r["target"] for r in rows])
    masks = evaluation_cells(np.asarray([r["M"] for r in rows]), np.asarray([r["T"] for r in rows]), protocol["observations"]["source"])
    error = abs(np.asarray(prediction) - target)
    report = {"status": "additional_exploratory_physics_diagnostic", "protocol_sha256": file_hash(config),
              "code_sha256": file_hash(Path(__file__)), "labels_used": 0,
              "population_count_known": protocol["generator"]["N_full"],
              "fixture_identity_verified": True, "MAE": {name: float(error[mask].mean()) for name, mask in masks.items()},
              "row_id": [r["row_id"] for r in rows], "prediction": prediction,
              "limitations": "Uses the known fixed population size and uniform sampling. Only the unclipped squared quantity with true phases is unbiased; estimated analytic phases, clipping, square roots and future labels have no such guarantee. Added after seeing the initial simple controls; not preregistered or a general real-data procedure."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["MAE"]))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    check(args.config, args.data, args.output)
