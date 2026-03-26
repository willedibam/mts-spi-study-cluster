"""Generate sweep configs from a base YAML by overriding one parameter at a time.

Usage:
    python scripts/gen_sweep_configs.py
"""

import copy
import yaml
from pathlib import Path

BASE_CONFIG = "configs/generate/r_rho_mi/260323_g-tanhsin_small.yaml"
OUT_DIR = Path("configs/generate/r_rho_mi/sweep_260323")

# Each sweep: (param_name, {yaml_key_in_base_params: value}, output_dir_suffix)
# Baseline: ar1_a=0.8, noise_std=0.1, k=6.283 (2pi)
import math

SWEEPS = {
    "ar1_a": {
        "param_key": "ar1_a",
        "values": [0.5, 0.6, 0.9],  # skip 0.8 (baseline)
        "dir_token": "a",
    },
    "noise_std": {
        "param_key": "noise_std",
        "values": [0.01, 0.05, 0.2, 0.5],  # skip 0.1 (baseline)
        "dir_token": "noise",
    },
    "k": {
        "param_key": "k",
        "values": [
            round(1.0, 4),
            round(math.pi, 4),
            round(4 * math.pi, 4),
            round(8 * math.pi, 4),
        ],  # skip 2pi=6.283 (baseline)
        "dir_token": "k",
    },
}


def main():
    with open(BASE_CONFIG) as f:
        base = yaml.safe_load(f)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    generated = []

    for sweep_name, spec in SWEEPS.items():
        for val in spec["values"]:
            cfg = copy.deepcopy(base)
            # Update output dir
            val_str = f"{val:.4f}".rstrip("0").rstrip(".")
            cfg["base_output_dir"] = (
                f"data/r_rho_mi/sweep_260323/{spec['dir_token']}_{val_str}/"
            )
            # Update all mts_classes
            for cls in cfg["mts_classes"]:
                cls["base_params"][spec["param_key"]] = val

            fname = f"{spec['dir_token']}_{val_str}.yaml"
            out_path = OUT_DIR / fname
            with open(out_path, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
            generated.append(str(out_path))
            print(f"  {out_path}")

    print(f"\nGenerated {len(generated)} configs in {OUT_DIR}/")
    # Also write a manifest for the PBS script
    manifest = OUT_DIR / "manifest.txt"
    # Include the baseline config too
    all_configs = [BASE_CONFIG] + generated
    with open(manifest, "w") as f:
        for p in all_configs:
            f.write(p + "\n")
    print(f"Manifest: {manifest} ({len(all_configs)} configs including baseline)")


if __name__ == "__main__":
    main()
