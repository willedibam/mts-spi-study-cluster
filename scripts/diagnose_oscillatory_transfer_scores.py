"""Post-hoc known-prevalence batch decisions; original frozen scores stay intact."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.representation_state_data import file_hash


def balanced_batch_assignments(scores):
    """Label-blind top-half decisions, with uniform expected assignments at ties."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or len(scores) % 2 or not np.isfinite(scores).all():
        raise ValueError("require an even nonempty vector of finite scores")
    positives = len(scores) // 2
    cutoff = np.sort(scores)[-positives]
    above, tied = scores > cutoff, scores == cutoff
    assignments = above.astype(float)
    assignments[tied] = (positives - above.sum()) / tied.sum()
    return assignments, float(cutoff)


def expected_balanced_accuracy(truth, assignments):
    return float(.5 * (assignments[truth == 1].mean() + 1 - assignments[truth == 0].mean()))


def main(data, predictions, output):
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((data / "manifest.json").read_text())
    rows = manifest["rows"]
    truth = np.array([r["target"] for r in rows])
    sizes = np.array([r["M"] for r in rows])
    records, inputs = [], []
    for path in sorted(predictions.glob("*.json")):
        info = json.loads(path.read_text())
        assert info["identity"]["manifest_sha256"] == file_hash(data / "manifest.json")
        digest = file_hash(path.with_suffix(".npz"))
        assert digest == info["predictions_sha256"]
        inputs.append(dict(path=str(path),metadata_sha256=file_hash(path),predictions_sha256=digest))
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["row_id"], [r["row_id"] for r in rows])
            np.testing.assert_array_equal(archive["target"], truth)
            scores = archive["prediction"]
        for m in [16, 8]:
            mask = sizes == m
            target, score = truth[mask], scores[mask]
            assert len(score) == 200
            assignments, cutoff = balanced_batch_assignments(score)
            np.testing.assert_allclose(assignments.sum(), 100, atol=1e-10)
            before = (score >= .5).astype(float)
            records.append(dict(method=info["identity"]["method"], labels=info["labels_total"],
                seed=info["identity"]["seed"], M=m, cutoff=cutoff,
                fixed_BA=expected_balanced_accuracy(target, before),
                batch_expected_BA=expected_balanced_accuracy(target, assignments),
                AUROC=float(roc_auc_score(target, score)),
                fixed_false_positive=float(before[target == 0].mean()),
                fixed_false_negative=float(1 - before[target == 1].mean()),
                batch_false_positive=float(assignments[target == 0].mean()),
                batch_false_negative=float(1 - assignments[target == 1].mean())))
        assert file_hash(path.with_suffix(".npz")) == digest
    assert len(inputs) == 99 and len(records) == 198
    frame = pd.DataFrame(records)
    assert not frame.duplicated(["method", "labels", "seed", "M"]).any()
    summary = frame.groupby(["M", "method", "labels"]).agg(
        fits=("seed", "size"), fixed_BA=("fixed_BA", "mean"),
        batch_expected_BA=("batch_expected_BA", "mean"), AUROC=("AUROC", "mean")).reset_index()
    output.mkdir(parents=True)
    frame.to_csv(output / "per-fit.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    (output / "provenance.json").write_text(json.dumps(dict(
        status="post_hoc_diagnostic", positive_prevalence_assumed=.5,
        target_information="unlabelled score batch and known design prevalence; labels only for evaluation",
        tie_rule="expected accuracy under uniform random tie resolution", original_predictions_unchanged=True,
        script_sha256=file_hash(Path(__file__)), inputs=inputs), indent=2) + "\n")
    print(summary[summary.M == 8].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["data", "predictions", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    main(args.data, args.predictions, args.output)
