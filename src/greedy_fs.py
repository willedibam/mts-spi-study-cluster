"""
Greedy forward SPI selection (vectorized).

For a subset S of SPIs, build a per-MTS feature vector of pairwise Spearman
correlations between off-diagonal MPI vectors. Grow S one SPI at a time to
minimize 5-fold stratified CV multinomial log-loss on the train split.
Anchor on cov_EmpiricalCovariance.

Pipeline (per CV fold): clip -> atanh -> StandardScaler -> LogisticRegression.
Holdout: per (class, M, T) cell, send `--test-per-stratum` instances to test.

Vectorization: MTS are grouped by M (edge-vector length). Within a group, all
SPIs share the same edge-vector length, so pairwise correlations across MTS
reduce to einsums on (n_mts_in_group, n_spis, edge_len) tensors.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Literal, Sequence, Tuple

MetricType = Literal["spearman", "pearson"]

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .process_features import _edge_vectors, load_samples_with_flags

LOGGER = logging.getLogger(__name__)

ANCHOR_DEFAULT = "cov_EmpiricalCovariance"
EPS = 1e-6
CONST_FRAC_THRESHOLD = 0.1
DENOM_TOL = 1e-12


def _rank(x: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    order = np.argsort(x)
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)
    return ranks


def _is_constant(v: np.ndarray) -> bool:
    if v.size == 0 or not np.all(np.isfinite(v)):
        return True
    return float(v.max() - v.min()) < 1e-12


# ----------------------------
# Pool determination + tensor cache
# ----------------------------
def determine_pool(
    samples: List[Dict], spi_order: List[str], directed_flags: List[bool],
    frac_threshold: float = CONST_FRAC_THRESHOLD,
) -> List[str]:
    n_const = {name: 0 for name in spi_order}
    for i, s in enumerate(samples, 1):
        with np.load(s["path"] / "spi_mpis.npz") as npz:
            for name, directed in zip(spi_order, directed_flags):
                _, vec = _edge_vectors(name, npz[name], directed, split_directed=False)[0]
                if _is_constant(vec):
                    n_const[name] += 1
        if i % 100 == 0 or i == len(samples):
            LOGGER.info("scan: %d/%d", i, len(samples))
    n = len(samples)
    pool = [name for name in spi_order if n_const[name] / n < frac_threshold]
    LOGGER.info(
        "dropped %d/%d SPIs constant on >=%.0f%% of MTS",
        len(spi_order) - len(pool), len(spi_order), 100 * frac_threshold,
    )
    return pool


def build_group_tensors(
    samples: List[Dict], spi_order: List[str], directed_flags: List[bool], pool: List[str],
    *, metric: MetricType = "spearman",
) -> Tuple[List[Dict], Dict[str, int]]:
    """
    Per (M-group, pool-SPI): centered tensor + L2 norms. Returns (groups, name->pool_idx).
    metric='spearman' → centers ranks (Spearman = Pearson on ranks).
    metric='pearson'  → centers raw edge vectors.
    """
    spi_idx_map = {name: i for i, name in enumerate(pool)}
    directed_lookup = dict(zip(spi_order, directed_flags))
    by_M: Dict[int, List[int]] = {}
    for i, s in enumerate(samples):
        by_M.setdefault(s["M"], []).append(i)

    groups: List[Dict] = []
    for M_val in sorted(by_M):
        mts_idx = np.array(by_M[M_val])
        n_mts = len(mts_idx)
        edge_len = M_val * (M_val - 1) // 2
        Rc = np.zeros((n_mts, len(pool), edge_len), dtype=np.float64)
        N_ = np.zeros((n_mts, len(pool)), dtype=np.float64)
        for j, sample_idx in enumerate(mts_idx):
            with np.load(samples[sample_idx]["path"] / "spi_mpis.npz") as npz:
                for name in pool:
                    directed = directed_lookup[name]
                    _, vec = _edge_vectors(name, npz[name], directed, split_directed=False)[0]
                    if _is_constant(vec):
                        continue  # leave Rc, N_ as zero
                    base = _rank(vec) if metric == "spearman" else vec.astype(np.float64)
                    centered = base - base.mean()
                    Rc[j, spi_idx_map[name], :] = centered
                    N_[j, spi_idx_map[name]] = float(np.linalg.norm(centered))
        groups.append({"M": M_val, "mts_idx": mts_idx, "Rc": Rc, "N": N_})
        LOGGER.info("group M=%d: %d MTS, edge_len=%d", M_val, n_mts, edge_len)
    return groups, spi_idx_map


# ----------------------------
# Vectorized feature construction
# ----------------------------
def _features_for_subset(groups: List[Dict], S_idx: List[int], N_total: int) -> np.ndarray:
    """N_total x C(|S|,2) feature matrix of pairwise Spearman correlations."""
    k = len(S_idx)
    n_pairs = k * (k - 1) // 2
    if n_pairs == 0:
        return np.empty((N_total, 0), dtype=np.float64)
    iu = np.triu_indices(k, k=1)
    X = np.empty((N_total, n_pairs), dtype=np.float64)
    for g in groups:
        Rc_S = g["Rc"][:, S_idx, :]                   # (g_mts, k, edge_len)
        N_S = g["N"][:, S_idx]                        # (g_mts, k)
        gram = np.einsum("mae,mbe->mab", Rc_S, Rc_S)  # (g_mts, k, k)
        denom = N_S[:, :, None] * N_S[:, None, :]
        corr = np.divide(gram, denom, out=np.zeros_like(gram), where=denom > DENOM_TOL)
        X[g["mts_idx"], :] = corr[:, iu[0], iu[1]]
    return X


def _new_pair_cols(
    groups: List[Dict], S_idx: List[int], C_idx: List[int], N_total: int,
) -> np.ndarray:
    """Return (N_total, n_C, k) tensor of corr(s in S, c in C) per MTS."""
    k, n_C = len(S_idx), len(C_idx)
    if k == 0:
        return np.empty((N_total, n_C, 0), dtype=np.float64)
    out = np.empty((N_total, n_C, k), dtype=np.float64)
    for g in groups:
        Rc_S = g["Rc"][:, S_idx, :]
        N_S = g["N"][:, S_idx]
        Rc_C = g["Rc"][:, C_idx, :]
        N_C = g["N"][:, C_idx]
        num = np.einsum("mce,mse->mcs", Rc_C, Rc_S)
        denom = N_C[:, :, None] * N_S[:, None, :]
        corr = np.divide(num, denom, out=np.zeros_like(num), where=denom > DENOM_TOL)
        out[g["mts_idx"], :, :] = corr
    return out


# ----------------------------
# Holdout split
# ----------------------------
def stratified_holdout(
    y: np.ndarray, M: np.ndarray, T: np.ndarray,
    test_per_stratum: int, seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"y": y, "M": M, "T": T, "idx": np.arange(len(y))})
    train_idx, test_idx = [], []
    for _, grp in df.groupby(["y", "M", "T"], sort=True):
        idx = grp["idx"].to_numpy()
        rng.shuffle(idx)
        test_idx.extend(idx[:test_per_stratum].tolist())
        train_idx.extend(idx[test_per_stratum:].tolist())
    return np.array(sorted(train_idx)), np.array(sorted(test_idx))


# ----------------------------
# Pipeline + CV
# ----------------------------
def _atanh_clip(X: np.ndarray) -> np.ndarray:
    return np.arctanh(np.clip(X, -1.0 + EPS, 1.0 - EPS))


def _make_lr_pipeline() -> Pipeline:
    return Pipeline([
        ("atanh", FunctionTransformer(_atanh_clip, validate=False)),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(solver="lbfgs", C=1.0, max_iter=1000)),
    ])


def cv_logloss_acc(
    X: np.ndarray, y: np.ndarray, classes: np.ndarray, seed: int,
) -> Tuple[float, float]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    losses, accs = [], []
    for tr, va in skf.split(X, y):
        pipe = _make_lr_pipeline()
        pipe.fit(X[tr], y[tr])
        proba = pipe.predict_proba(X[va])
        losses.append(log_loss(y[va], proba, labels=classes))
        accs.append(accuracy_score(y[va], pipe.predict(X[va])))
    return float(np.mean(losses)), float(np.mean(accs))


def _eval_candidate(
    X_old: np.ndarray, new_col: np.ndarray,
    y: np.ndarray, classes: np.ndarray, seed: int,
) -> Tuple[float, float]:
    X = np.concatenate([X_old, new_col], axis=1)
    return cv_logloss_acc(X, y, classes, seed)


# ----------------------------
# Greedy
# ----------------------------
def greedy_forward(
    groups: List[Dict], spi_idx_map: Dict[str, int], pool: List[str],
    train_idx_arr: np.ndarray, y_train: np.ndarray,
    anchor: str, n_max: int, seed: int, n_jobs: int, N_total: int,
) -> Tuple[List[Dict], List[str], bool]:
    classes = np.unique(y_train)
    if anchor not in spi_idx_map:
        raise ValueError(f"Anchor {anchor} not in pool")

    S_names: List[str] = [anchor]
    S_idx: List[int] = [spi_idx_map[anchor]]
    trace: List[Dict] = []
    stopped_early = False

    for n in range(2, n_max + 1):
        t0 = time.time()
        cand_names = [name for name in pool if name not in S_names]
        cand_idxs = [spi_idx_map[name] for name in cand_names]

        X_old_full = _features_for_subset(groups, S_idx, N_total)
        X_old = X_old_full[train_idx_arr]
        new_cols_full = _new_pair_cols(groups, S_idx, cand_idxs, N_total)
        new_cols_train = new_cols_full[train_idx_arr]  # (n_train, n_C, k)

        if n_jobs == 1:
            results = [
                _eval_candidate(X_old, new_cols_train[:, ci, :], y_train, classes, seed)
                for ci in range(len(cand_idxs))
            ]
        else:
            results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_eval_candidate)(
                    X_old, new_cols_train[:, ci, :], y_train, classes, seed,
                )
                for ci in range(len(cand_idxs))
            )

        scored = [
            (cand_names[ci], cand_idxs[ci], ll, acc)
            for ci, (ll, acc) in enumerate(results)
        ]
        scored.sort(key=lambda r: r[2])
        best_name, best_idx, best_ll, best_acc = scored[0]
        S_names.append(best_name)
        S_idx.append(best_idx)
        trace.append({
            "step": n, "added_spi": best_name,
            "log_loss": best_ll, "accuracy": best_acc, "S_size": n,
        })
        LOGGER.info(
            "step %2d (%.1fs) | +%-30s | logloss=%.4f acc=%.4f",
            n, time.time() - t0, best_name, best_ll, best_acc,
        )

        if len(trace) >= 4:
            lls = [t["log_loss"] for t in trace[-4:]]
            deltas = [lls[i] - lls[i + 1] for i in range(3)]
            if float(np.mean(deltas)) < 1e-3:
                LOGGER.info(
                    "stopping: mean Δlog-loss last 3 = %.5f < 1e-3",
                    float(np.mean(deltas)),
                )
                stopped_early = True
                break

    return trace, S_names, stopped_early


# ----------------------------
# Holdout eval
# ----------------------------
def holdout_eval(
    groups: List[Dict], spi_idx_map: Dict[str, int],
    y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, S_names: List[str],
) -> Dict:
    S_idx = [spi_idx_map[name] for name in S_names]
    X_full = _features_for_subset(groups, S_idx, len(y))
    X_train, X_test = X_full[train_idx], X_full[test_idx]
    classes = np.unique(y)
    out: Dict[str, Dict[str, float]] = {}

    lr = _make_lr_pipeline()
    lr.fit(X_train, y[train_idx])
    out["lr"] = {
        "log_loss": float(log_loss(y[test_idx], lr.predict_proba(X_test), labels=classes)),
        "accuracy": float(accuracy_score(y[test_idx], lr.predict(X_test))),
    }
    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X_train, y[train_idx])
    out["rf"] = {
        "log_loss": float(log_loss(y[test_idx], rf.predict_proba(X_test), labels=classes)),
        "accuracy": float(accuracy_score(y[test_idx], rf.predict(X_test))),
    }
    return out


# ----------------------------
# Plot
# ----------------------------
def plot_curve(trace: List[Dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt
    df = pd.DataFrame(trace)
    fig, ax_ll = plt.subplots(figsize=(8, 5), dpi=150)
    ax_acc = ax_ll.twinx()
    ax_ll.plot(df["S_size"], df["log_loss"], "o-", color="C0", label="CV log-loss")
    ax_acc.plot(df["S_size"], df["accuracy"], "s--", color="C1", label="CV accuracy")
    ax_ll.set_xlabel("|S|")
    ax_ll.set_ylabel("CV log-loss", color="C0")
    ax_acc.set_ylabel("CV accuracy", color="C1")
    ax_ll.grid(alpha=0.3)
    fig.suptitle("Greedy forward SPI selection")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ----------------------------
# CLI
# ----------------------------
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Greedy forward SPI selection.")
    p.add_argument("--data-path", default="data/embeddings/proof_benchmarked_260505")
    p.add_argument("--output-dir", default=None,
                   help="Output dir. If omitted, defaults to greedy/{metric}_N-{n_max}.")
    p.add_argument("--anchor", default=ANCHOR_DEFAULT)
    p.add_argument("--n-max", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--test-per-stratum", type=int, default=2,
                   help="Per (class,M,T) cell, this many to test (default 2/10).")
    p.add_argument("--dataset-limit", type=int, default=None,
                   help="Optional MTS limit for smoke tests.")
    p.add_argument("--metric", choices=["spearman", "pearson"], default="spearman",
                   help="Inter-SPI similarity for feature construction (default: spearman).")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    out_dir = Path(args.output_dir) if args.output_dir else Path("greedy") / f"{args.metric}_N-{args.n_max}"
    out_dir.mkdir(parents=True, exist_ok=True)

    samples, spi_order, directed_flags = load_samples_with_flags(
        args.data_path, limit=args.dataset_limit,
    )
    if args.anchor not in spi_order:
        raise ValueError(f"Anchor SPI '{args.anchor}' not in dataset (n_spis={len(spi_order)}).")

    y = np.array([s["label"] for s in samples])
    M = np.array([s["M"] for s in samples])
    T = np.array([s["T"] for s in samples])

    LOGGER.info("Scanning %d MTS to determine pool", len(samples))
    pool = determine_pool(samples, spi_order, directed_flags, frac_threshold=CONST_FRAC_THRESHOLD)
    if args.anchor not in pool:
        LOGGER.warning("Anchor %s filtered by constancy rule; restoring.", args.anchor)
        pool = [args.anchor] + pool
    LOGGER.info("Pool size: %d", len(pool))

    LOGGER.info("Building group tensors (metric=%s)...", args.metric)
    groups, spi_idx_map = build_group_tensors(
        samples, spi_order, directed_flags, pool, metric=args.metric,
    )

    train_idx, test_idx = stratified_holdout(y, M, T, args.test_per_stratum, args.seed)
    LOGGER.info("Holdout: %d train / %d test", len(train_idx), len(test_idx))

    y_train = y[train_idx]
    LOGGER.info("Greedy: anchor=%s | pool=%d | n_max=%d", args.anchor, len(pool), args.n_max)
    trace, S_names, stopped_early = greedy_forward(
        groups, spi_idx_map, pool,
        train_idx, y_train,
        args.anchor, args.n_max, args.seed, args.n_jobs, N_total=len(samples),
    )

    LOGGER.info("Holdout eval on |S|=%d", len(S_names))
    ho = holdout_eval(groups, spi_idx_map, y, train_idx, test_idx, S_names)

    pd.DataFrame(trace).to_csv(out_dir / "selection_trace.csv", index=False)
    (out_dir / "final_S.json").write_text(json.dumps(S_names, indent=2))
    (out_dir / "final_S.txt").write_text("\n".join(S_names) + "\n")
    (out_dir / "holdout_eval.json").write_text(json.dumps(ho, indent=2))
    plot_curve(trace, out_dir / "selection_curve.png")

    last_deltas: List[float] = []
    if len(trace) >= 4:
        lls = [t["log_loss"] for t in trace[-4:]]
        last_deltas = [lls[i] - lls[i + 1] for i in range(3)]
    summary = [
        "# Greedy forward SPI selection",
        f"timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"data_path: {args.data_path}",
        f"n_samples: {len(samples)} (train={len(train_idx)}, test={len(test_idx)})",
        f"n_spis_total: {len(spi_order)}",
        f"n_spis_pool: {len(pool)}  (constancy threshold {CONST_FRAC_THRESHOLD})",
        f"metric: {args.metric}",
        f"anchor: {args.anchor}",
        f"final_|S|: {len(S_names)}",
        f"stopped_early: {stopped_early}",
        f"final_S: {S_names}",
        "",
        "# Holdout evaluation on final_S:",
        f"  LR: log_loss={ho['lr']['log_loss']:.4f} acc={ho['lr']['accuracy']:.4f}",
        f"  RF: log_loss={ho['rf']['log_loss']:.4f} acc={ho['rf']['accuracy']:.4f}",
    ]
    if last_deltas:
        summary.append(f"\nstopping rule: mean Δlog-loss last 3 = {float(np.mean(last_deltas)):.5f}")
    (out_dir / "summary.txt").write_text("\n".join(summary))

    LOGGER.info("Done. Outputs -> %s", out_dir.resolve())


if __name__ == "__main__":
    main()
