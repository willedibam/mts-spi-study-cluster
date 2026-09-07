"""Independent tiny-data optimization and throughput check; no pilot eval data."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from src.representation_state_data import file_hash
from src.representation_state_neural import fit_encoder, predict, seed_torch


def check(config, output, device, learning_rate=.001):
    spec = yaml.safe_load(config.read_text())["methods"]["raw_encoder_spec"]
    torch.set_num_threads(2)
    # Deliberately simple phase organization at identical marginal waveforms.
    # Diagnostic inputs are disjoint from Stage B masters and no test MAE enters
    # this implementation check. This is not evidence of Kuramoto generalization.
    t = np.arange(1000) * .1
    phases = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    spread = np.array([.05, .3, .6, 1.])
    raw = np.stack([np.cos(t[:, None] + scale * phases) for scale in spread])
    labels = np.abs(np.exp(1j * spread[:, None] * phases).mean(axis=1))
    x = torch.as_tensor(raw, dtype=torch.float32, device=device)
    y = torch.as_tensor(labels, dtype=torch.float32, device=device)
    model, log = fit_encoder(x, y, spec, learning_rate, .0001, 911, epochs=200)
    prediction = predict(model, x, 4)
    mae = float(np.abs(prediction - labels).mean())
    report = {"config_sha256": file_hash(config), "model_sha256": file_hash(Path("src/representation_state_neural.py")),
              "device": device, "torch": torch.__version__, "learning_rate": learning_rate, "training_MAE": mae,
              "constant_training_MAE": float(np.abs(labels - labels.mean()).mean()),
              "targets": labels.tolist(), "predictions": prediction.tolist(), "fit": log,
              "passed": mae < .05, "claim": "Optimization diagnostic only; no held-out pilot data."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "fit"}))
    print(f"seconds={log['seconds']:.1f}", flush=True)
    if not report["passed"]:
        raise RuntimeError("tiny-data fit failed; do not interpret a negative neural comparison")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--learning-rate", type=float, default=.001)
    args = parser.parse_args()
    check(args.config, args.output, args.device, args.learning_rate)
