"""
Pre-seed timeseries for the explosive (hysteretic) Kuramoto sweep.

Hysteresis is a path-dependent property: in the bistable window two stable states
coexist and which one is observed depends on history. The standard harness generates
each (variant, instance) independently from a random IC, so it samples only the lower
branch and cannot produce the loop. This script does the required quasi-static
continuation -- per instance, on a FIXED network -- and writes `timeseries.npy` into
each dataset dir. The standard harness (run_experiments) then finds the cached
timeseries, skips regeneration, and runs pyspi in parallel as usual.

Protocol per (M, T, instance):
  network_seed = instance  (same BA network for both branches)
  forward : K ascending, start from a random IC, chain each K from the previous final state
  backward: K descending, start from the forward branch's final (synchronised) state

Usage:
  python -m src.preseed_explosive --config configs/generate/embeddings/kuramoto-explosive-sweep.yaml
  python -m src.preseed_explosive --config <yaml> --dry-run   # show grouping + target dirs only
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

from .generators import generate_kuramoto_explosive
from .mapping import DatasetMapping, ExperimentConfig

LOGGER = logging.getLogger(__name__)
GENERATOR = "kuramoto_explosive"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-seed adiabatic-continuation timeseries for explosive Kuramoto.")
    p.add_argument("--config", required=True, help="Path to the kuramoto-explosive sweep YAML.")
    p.add_argument("--force", action="store_true", help="Overwrite existing timeseries.npy.")
    p.add_argument("--dry-run", action="store_true", help="Print grouping and target dirs without generating.")
    return p.parse_args(argv)


def _branch_of(spec) -> str:
    branch = str(spec.generator_params.get("branch", "")).lower()
    if branch not in {"forward", "backward"}:
        raise ValueError(f"spec {spec.name} missing/invalid branch param: {branch!r}")
    return branch


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = ExperimentConfig.from_file(Path(args.config))
    specs = [s for s in DatasetMapping(config).specs if s.generator == GENERATOR]
    if not specs:
        raise RuntimeError(f"No {GENERATOR} specs found in {args.config}")

    # group by (M, T, instance); each group has a forward and a backward branch
    groups: dict[tuple[int, int, int], list] = defaultdict(list)
    for s in specs:
        groups[(s.M, s.T, s.instance)].append(s)

    LOGGER.info("Pre-seeding %d datasets across %d (M,T,instance) groups", len(specs), len(groups))
    written = skipped = 0

    for (M, T, inst), gspecs in sorted(groups.items()):
        fwd = sorted((s for s in gspecs if _branch_of(s) == "forward"),
                     key=lambda s: float(s.generator_params["K"]))
        bwd = sorted((s for s in gspecs if _branch_of(s) == "backward"),
                     key=lambda s: float(s.generator_params["K"]), reverse=True)

        # shared generator params (identical across the group's variants)
        gp = gspecs[0].generator_params
        common = dict(
            M=M, T=T,
            m_ba=int(gp.get("m_ba", 3)),
            omega_scale=float(gp.get("omega_scale", 1.0)),
            dt=float(gp.get("dt", 0.005)),
            transients=int(gp.get("transients", 3000)),
            output=str(gp.get("output", "sin")),
            zscore=bool(gp.get("zscore", True)),
            network_seed=inst,                       # network fixed per instance
        )
        if args.dry_run:
            LOGGER.info("(M%d T%d I%d) fwd=%d bwd=%d net_seed=%d e.g. %s",
                        M, T, inst, len(fwd), len(bwd), inst,
                        fwd[0].dataset_dir if fwd else "-")
            continue

        rng = np.random.default_rng(config.rng_seed + inst)
        dK = float(np.median(np.diff(sorted(float(s.generator_params["K"]) for s in fwd)))) if len(fwd) > 1 else 0.0

        def _run_branch(branch_specs, branch_name, init_state):
            nonlocal written, skipped
            theta = init_state
            for s in branch_specs:
                ts_path = Path(s.dataset_dir) / "timeseries.npy"
                K = float(s.generator_params["K"])
                Y, theta = generate_kuramoto_explosive(
                    K=K, init_state=theta, return_final_state=True, rng=rng, **common
                )
                if ts_path.exists() and not args.force:
                    skipped += 1
                    continue
                ts_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(ts_path, Y.astype(np.float32))
                # authoritative provenance for the cached-TS path (harness can't see it)
                prov = {k: v for k, v in common.items()}
                prov.update({
                    "generator": GENERATOR, "K": K, "branch": branch_name,
                    "instance": inst, "dK": round(dK, 6), "rng_seed": int(config.rng_seed + inst),
                    "protocol": "adiabatic_continuation",
                    "init": "random_IC" if (branch_name == "forward" and s is branch_specs[0]) else "continued",
                })
                (ts_path.parent / "gen_provenance.json").write_text(json.dumps(prov, indent=2))
                written += 1
            return theta

        theta_end = _run_branch(fwd, "forward", None)           # forward from random IC
        _run_branch(bwd, "backward", theta_end)                 # backward from synchronised state
        LOGGER.info("(M%d T%d I%d) done (written=%d skipped=%d)", M, T, inst, written, skipped)

    LOGGER.info("Pre-seed complete: %d timeseries written, %d skipped.", written, skipped)
    if not args.dry_run:
        LOGGER.info("Next: run_experiments on the same config (cached timeseries -> pyspi only).")


if __name__ == "__main__":
    main()
