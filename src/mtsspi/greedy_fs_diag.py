"""
Post-hoc diagnostics for a greedy_fs run.

Re-evaluates each prefix of final_S to produce train / CV-val / holdout log-loss
with 5-fold CV std error bars. Does not re-run selection. Writes diagnostics.csv
and diagnostics_curve.png into the run dir.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold

from .greedy_fs import (
    _features_for_subset,
    _make_lr_pipeline,
    build_group_tensors,
    stratified_holdout,
)
from .process_features import load_samples_with_flags

LOGGER = logging.getLogger(__name__)


def _cv_fold_metrics(X, y, classes, seed=42, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    tr_ll, va_ll, tr_ac, va_ac = [], [], [], []
    for tr, va in skf.split(X, y):
        pipe = _make_lr_pipeline()
        pipe.fit(X[tr], y[tr])
        tr_ll.append(log_loss(y[tr], pipe.predict_proba(X[tr]), labels=classes))
        va_ll.append(log_loss(y[va], pipe.predict_proba(X[va]), labels=classes))
        tr_ac.append(accuracy_score(y[tr], pipe.predict(X[tr])))
        va_ac.append(accuracy_score(y[va], pipe.predict(X[va])))
    return tr_ll, va_ll, tr_ac, va_ac


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, help="greedy/<run>/ folder with final_S.json")
    p.add_argument("--data-path", default="data/embeddings/proof_benchmarked_260505")
    p.add_argument("--metric", choices=["spearman", "pearson"], default="spearman")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-per-stratum", type=int, default=2)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    run_dir = Path(args.run_dir)
    S_names = json.loads((run_dir / "final_S.json").read_text())

    samples, spi_order, directed_flags = load_samples_with_flags(args.data_path)
    y = np.array([s["label"] for s in samples])
    M = np.array([s["M"] for s in samples])
    T = np.array([s["T"] for s in samples])

    LOGGER.info("Building tensors for %d selected SPIs (metric=%s)", len(S_names), args.metric)
    groups, spi_idx_map = build_group_tensors(
        samples, spi_order, directed_flags, S_names, metric=args.metric,
    )

    train_idx, test_idx = stratified_holdout(y, M, T, args.test_per_stratum, args.seed)
    y_train, y_test = y[train_idx], y[test_idx]
    classes = np.unique(y)

    rows = []
    for n in range(2, len(S_names) + 1):
        S_prefix = S_names[:n]
        S_idx = [spi_idx_map[name] for name in S_prefix]
        X_full = _features_for_subset(groups, S_idx, len(samples))
        X_train, X_test = X_full[train_idx], X_full[test_idx]

        tr_ll, va_ll, tr_ac, va_ac = _cv_fold_metrics(X_train, y_train, classes, args.seed)
        pipe = _make_lr_pipeline()
        pipe.fit(X_train, y_train)
        ho_ll = log_loss(y_test, pipe.predict_proba(X_test), labels=classes)
        ho_ac = accuracy_score(y_test, pipe.predict(X_test))

        rows.append({
            "S_size": n, "added_spi": S_prefix[-1],
            "train_ll_mean": float(np.mean(tr_ll)), "train_ll_std": float(np.std(tr_ll)),
            "val_ll_mean": float(np.mean(va_ll)), "val_ll_std": float(np.std(va_ll)),
            "train_acc_mean": float(np.mean(tr_ac)), "val_acc_mean": float(np.mean(va_ac)),
            "holdout_ll": float(ho_ll), "holdout_acc": float(ho_ac),
        })
        LOGGER.info(
            "|S|=%2d  train=%.4f±%.4f  val=%.4f±%.4f  holdout=%.4f",
            n, rows[-1]["train_ll_mean"], rows[-1]["train_ll_std"],
            rows[-1]["val_ll_mean"], rows[-1]["val_ll_std"], ho_ll,
        )

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "diagnostics.csv", index=False)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.errorbar(df["S_size"], df["train_ll_mean"], yerr=df["train_ll_std"],
                fmt="o-", capsize=3, label="train (5-fold mean ± std)", color="C2")
    ax.errorbar(df["S_size"], df["val_ll_mean"], yerr=df["val_ll_std"],
                fmt="s-", capsize=3, label="CV val (5-fold mean ± std)", color="C0")
    ax.plot(df["S_size"], df["holdout_ll"], "^--", label=f"holdout (n={len(test_idx)})", color="C3")
    ax.set_xlabel("|S|")
    ax.set_ylabel("log-loss (log scale)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    ax.set_title(f"{run_dir.name}: log-loss vs |S|")
    fig.tight_layout()
    fig.savefig(run_dir / "diagnostics_curve.png")
    plt.close(fig)
    LOGGER.info("Wrote %s and %s", run_dir / "diagnostics.csv", run_dir / "diagnostics_curve.png")


if __name__ == "__main__":
    main()
