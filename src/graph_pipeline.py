"""
Graph Pipeline for Topology Classification via Edge-Conditioned MPNN.

Loads pyspi output from the standard pipeline format (spi_mpis.npz + meta.json),
constructs PyG graphs with K-dimensional edge features, trains a GINEConv-based
MPNN for graph classification, and compares against SPI-SPI correlation baselines.

Usage:
    python -m src.graph_pipeline --data_dir data/topology_<timestamp> --n_folds 5 --epochs 200

Expected input format in data_dir:
    <class_name>/M<M>_T<T>_I<instance>/
        spi_mpis.npz   — keys are SPI names, values are (M, M) matrices
        meta.json      — metadata including class labels

    OR: provide pre-assembled arrays:
        mpi_tensors.npy — (N, K, M, M)
        labels.npy      — (N,)
        spi_names.npy   — (K,)

Requirements:
    pip install torch torch-geometric scikit-learn numpy scipy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GINEConv, global_mean_pool

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "torch and torch-geometric are required for graph_pipeline. "
            "Install with: pip install 'mts-spi-study-cluster[graph]'"
        )


# =============================================================================
# 1. DATA LOADING
# =============================================================================

def load_data_from_pipeline(data_dir: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load pyspi output from the standard pipeline directory structure.

    Scans data_dir for subdirectories containing spi_mpis.npz and meta.json,
    assembles them into the (N, K, M, M) tensor format.
    Class labels are auto-assigned from sorted directory names.

    Returns:
        mpi_tensors: (N, K, M, M) array
        labels: (N,) integer labels
        spi_names: list of str, length K
    """
    data_dir = Path(data_dir)

    # Auto-discover class directories (those containing >=1 spi_mpis.npz)
    class_dirs = sorted([
        d for d in data_dir.iterdir()
        if d.is_dir() and any(
            (sd / "spi_mpis.npz").exists()
            for sd in d.iterdir() if sd.is_dir()
        )
    ])
    if not class_dirs:
        raise FileNotFoundError(
            f"No class directories with spi_mpis.npz found in {data_dir}."
        )
    label_map = {d.name: i for i, d in enumerate(class_dirs)}
    print(f"Discovered classes: {label_map}")

    all_tensors = []
    all_labels = []
    spi_names = None

    for class_dir in class_dirs:
        class_name = class_dir.name
        label = label_map[class_name]

        for sample_dir in sorted(class_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            npz_path = sample_dir / "spi_mpis.npz"
            if not npz_path.exists():
                continue

            npz = np.load(npz_path)
            if spi_names is None:
                spi_names = sorted(npz.files)
                M = npz[spi_names[0]].shape[0]
                print(f"Detected K={len(spi_names)} SPIs, M={M}")

            tensor = np.stack([npz[s] for s in spi_names], axis=0)  # (K, M, M)
            all_tensors.append(tensor)
            all_labels.append(label)

    if not all_tensors:
        raise FileNotFoundError(
            f"No spi_mpis.npz files found in {data_dir}. "
            "Run the pipeline first, or provide pre-assembled arrays."
        )

    mpi_tensors = np.stack(all_tensors)  # (N, K, M, M)
    labels = np.array(all_labels)
    print(
        f"Loaded {mpi_tensors.shape[0]} samples, K={mpi_tensors.shape[1]} SPIs, "
        f"M={mpi_tensors.shape[2]}, {len(set(labels))} classes"
    )
    return mpi_tensors, labels, spi_names


def load_data_from_arrays(data_dir: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load pre-assembled arrays (mpi_tensors.npy, labels.npy, spi_names.npy).
    """
    data_dir = Path(data_dir)
    mpi_tensors = np.load(data_dir / "mpi_tensors.npy")
    labels = np.load(data_dir / "labels.npy")
    spi_names_path = data_dir / "spi_names.npy"
    if spi_names_path.exists():
        spi_names = list(np.load(spi_names_path, allow_pickle=True))
    else:
        spi_names = [f"SPI_{k}" for k in range(mpi_tensors.shape[1])]
    print(
        f"Loaded {mpi_tensors.shape[0]} samples, K={mpi_tensors.shape[1]} SPIs, "
        f"M={mpi_tensors.shape[2]}, {len(set(labels))} classes"
    )
    return mpi_tensors, labels, spi_names


def load_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load data, auto-detecting format (pipeline output vs pre-assembled arrays)."""
    data_dir_path = Path(data_dir)
    if (data_dir_path / "mpi_tensors.npy").exists():
        return load_data_from_arrays(data_dir)
    return load_data_from_pipeline(data_dir)


# =============================================================================
# 2. GRAPH CONSTRUCTION
# =============================================================================

def normalise_edge_features(mpi_tensors: np.ndarray) -> np.ndarray:
    """Z-score normalise each SPI dimension across all edges in all graphs."""
    N, K, M, _ = mpi_tensors.shape
    normalised = mpi_tensors.copy()
    mask = ~np.eye(M, dtype=bool)
    for k in range(K):
        vals = mpi_tensors[:, k][:, mask].ravel()
        mu, std = vals.mean(), vals.std()
        if std < 1e-10:
            std = 1.0
        normalised[:, k, :, :] = (mpi_tensors[:, k, :, :] - mu) / std
    return normalised


def mpi_to_pyg_graph(mpi_tensor: np.ndarray, label: int) -> "Data":
    """
    Convert a single (K, M, M) MPI tensor to a PyG Data object.

    Returns Data with:
        x:          (M, 2K) node features (mean/std of outgoing edges)
        edge_index: (2, num_edges) directed edges for all i != j
        edge_attr:  (num_edges, K) edge feature vectors
        y:          (1,) label
    """
    _require_torch()
    K, M, _ = mpi_tensor.shape

    src, dst, edge_attr = [], [], []
    for i in range(M):
        for j in range(M):
            if i != j:
                src.append(i)
                dst.append(j)
                edge_attr.append(mpi_tensor[:, i, j])

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr_t = torch.tensor(np.stack(edge_attr), dtype=torch.float32)

    node_features = []
    for i in range(M):
        out_edges = mpi_tensor[:, i, :]  # (K, M)
        mask = np.ones(M, dtype=bool)
        mask[i] = False
        out_edges = out_edges[:, mask]  # (K, M-1)
        feat = np.concatenate([out_edges.mean(axis=1), out_edges.std(axis=1)])
        node_features.append(feat)
    x = torch.tensor(np.stack(node_features), dtype=torch.float32)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr_t,
        y=torch.tensor([label], dtype=torch.long),
    )


def build_dataset(mpi_tensors: np.ndarray, labels: np.ndarray) -> list:
    """Convert all samples to PyG Data objects."""
    return [mpi_to_pyg_graph(mpi_tensors[n], labels[n]) for n in range(len(labels))]


def build_shuffled_dataset(
    mpi_tensors: np.ndarray, labels: np.ndarray, seed: int = 0,
) -> list:
    """Build PyG graphs with edge features randomly permuted across edges.

    For each sample and each SPI, the off-diagonal values are shuffled
    independently.  This destroys which-edge-has-which-value while
    preserving the per-graph feature distribution.
    """
    rng = np.random.default_rng(seed)
    shuffled = mpi_tensors.copy()
    N, K, M, _ = shuffled.shape
    mask = ~np.eye(M, dtype=bool)
    for n in range(N):
        for k in range(K):
            vals = shuffled[n, k][mask].copy()
            rng.shuffle(vals)
            shuffled[n, k][mask] = vals
    return [mpi_to_pyg_graph(shuffled[n], labels[n]) for n in range(N)]


def build_flat_edge_dataset(mpi_tensors: np.ndarray) -> np.ndarray:
    """Flatten upper-triangle of each SPI matrix into a feature vector.

    Shape: (N, K * M*(M-1)/2).  Retains per-edge values but discards
    graph structure.
    """
    N, K, M, _ = mpi_tensors.shape
    iu = np.triu_indices(M, k=1)
    features = np.empty((N, K * len(iu[0])))
    for n in range(N):
        vecs = [mpi_tensors[n, k][iu] for k in range(K)]
        features[n] = np.concatenate(vecs)
    return features


# =============================================================================
# 3. MPNN MODEL
# =============================================================================

if _TORCH_AVAILABLE:

    class InteractionGraphNet(nn.Module):
        """
        GINEConv-based MPNN for graph classification on interaction graphs.

        Architecture:
            - 2 GINEConv layers with edge feature projection
            - BatchNorm + ReLU after each layer
            - Global mean pooling
            - 2-layer MLP classifier with dropout
        """

        def __init__(
            self,
            num_node_features: int,
            num_edge_features: int,
            hidden_dim: int = 64,
            num_classes: int = 4,
            dropout: float = 0.3,
        ):
            super().__init__()
            self.node_proj = nn.Linear(num_node_features, hidden_dim)
            self.edge_proj1 = nn.Linear(num_edge_features, hidden_dim)

            nn1 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.conv1 = GINEConv(nn1)
            self.bn1 = nn.BatchNorm1d(hidden_dim)

            self.edge_proj2 = nn.Linear(num_edge_features, hidden_dim)
            nn2 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.conv2 = GINEConv(nn2)
            self.bn2 = nn.BatchNorm1d(hidden_dim)

            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_classes),
            )

        def forward(self, data):
            x, edge_index, edge_attr, batch = (
                data.x, data.edge_index, data.edge_attr, data.batch
            )
            x = self.node_proj(x)
            e1 = self.edge_proj1(edge_attr)
            x = F.relu(self.bn1(self.conv1(x, edge_index, e1)))
            e2 = self.edge_proj2(edge_attr)
            x = F.relu(self.bn2(self.conv2(x, edge_index, e2)))
            x = global_mean_pool(x, batch)
            return self.classifier(x)


if _TORCH_AVAILABLE:

    class FlatMLP(nn.Module):
        """MLP on flattened MPI features (no graph structure)."""

        def __init__(
            self, input_dim: int, hidden_dim: int = 128,
            num_classes: int = 4, dropout: float = 0.3,
        ):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_classes),
            )

        def forward(self, x):
            return self.net(x)


# =============================================================================
# 4. TRAINING
# =============================================================================

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out, data.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        pred = out.argmax(dim=1)
        correct += (pred == data.y.view(-1)).sum().item()
        total += data.num_graphs
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for data in loader:
        data = data.to(device)
        out = model(data)
        pred = out.argmax(dim=1)
        correct += (pred == data.y.view(-1)).sum().item()
        total += data.num_graphs
    return correct / total


def run_mpnn_fold(
    train_dataset, test_dataset, num_node_features, num_edge_features,
    num_classes, hidden_dim=64, epochs=200, lr=1e-3, patience=20, device="cpu",
):
    """Train and evaluate MPNN on one fold.

    Returns:
        best_acc: best test accuracy
        best_epoch: epoch at best test accuracy
        curves: dict with keys 'train_loss', 'train_acc', 'test_acc' (lists)
    """
    _require_torch()
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = InteractionGraphNet(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=10, factor=0.5)

    best_acc, no_improve = 0.0, 0
    best_epoch = 0
    curves = {"train_loss": [], "train_acc": [], "test_acc": []}

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        test_acc = evaluate(model, test_loader, device)
        scheduler.step(train_loss)

        curves["train_loss"].append(train_loss)
        curves["train_acc"].append(train_acc)
        curves["test_acc"].append(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    return best_acc, best_epoch, curves


def run_mlp_fold(
    X_train, y_train, X_test, y_test, num_classes,
    hidden_dim=128, epochs=200, lr=1e-3, patience=20, device="cpu",
):
    """Train a flat MLP on numpy feature arrays (one fold)."""
    _require_torch()
    scaler = StandardScaler()
    X_tr = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32).to(device)
    X_te = torch.tensor(scaler.transform(X_test), dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train, dtype=torch.long).to(device)
    y_te = torch.tensor(y_test, dtype=torch.long).to(device)

    model = FlatMLP(X_tr.shape[1], hidden_dim, num_classes).to(device)
    optimizer = Adam(model.parameters(), lr=lr)

    best_acc, no_improve, best_epoch = 0.0, 0, 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = F.cross_entropy(model(X_tr), y_tr)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = model(X_te).argmax(dim=1)
            acc = (pred == y_te).float().mean().item()
        if acc > best_acc:
            best_acc, best_epoch, no_improve = acc, epoch, 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break
    return best_acc, best_epoch


# =============================================================================
# 5. PLOTTING
# =============================================================================

def plot_training_curves(
    all_curves: Dict[str, List[dict]],
    save_path: Path | str | None = None,
) -> None:
    """Plot train loss and test accuracy per fold for each method that has curves.

    all_curves: {method_name: [fold_0_curves, fold_1_curves, ...]}
    Each fold_curves is a dict with keys 'train_loss', 'train_acc', 'test_acc'.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARN] matplotlib not installed — skipping plot.")
        return

    methods = list(all_curves.keys())
    n_methods = len(methods)
    n_folds = len(next(iter(all_curves.values())))

    fig, axes = plt.subplots(n_methods, 2, figsize=(12, 3.5 * n_methods),
                             squeeze=False)

    for row, method in enumerate(methods):
        ax_loss, ax_acc = axes[row, 0], axes[row, 1]
        for fold_idx, curves in enumerate(all_curves[method]):
            epochs = range(1, len(curves["train_loss"]) + 1)
            ax_loss.plot(epochs, curves["train_loss"],
                         alpha=0.6, label=f"Fold {fold_idx+1}")
            ax_acc.plot(epochs, curves["test_acc"],
                        alpha=0.6, label=f"Fold {fold_idx+1}")

        ax_loss.set_title(f"{method} — Train Loss")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.legend(fontsize=7)

        ax_acc.set_title(f"{method} — Test Accuracy")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_ylim(0, 1.05)
        ax_acc.legend(fontsize=7)

    fig.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved training curves to {save_path}")
    plt.close(fig)
    return fig


# =============================================================================
# 6. SPI-SPI BASELINE
# =============================================================================

def compute_spi_spi_features(mpi_tensor: np.ndarray) -> np.ndarray:
    """
    SPI-SPI feature vector: Pearson correlations between all pairs of
    SPI off-diagonal vectors. Returns length K*(K-1)/2.
    """
    K, M, _ = mpi_tensor.shape
    off_diag = []
    for k in range(K):
        vec = [mpi_tensor[k, i, j] for i in range(M) for j in range(M) if i != j]
        off_diag.append(np.array(vec))

    features = []
    for k1 in range(K):
        for k2 in range(k1 + 1, K):
            r, _ = pearsonr(off_diag[k1], off_diag[k2])
            features.append(0.0 if np.isnan(r) else r)
    return np.array(features)


def build_spi_spi_dataset(mpi_tensors: np.ndarray) -> np.ndarray:
    """Compute SPI-SPI features for all samples."""
    return np.array([compute_spi_spi_features(mpi_tensors[n]) for n in range(len(mpi_tensors))])


def run_baseline_fold(X_train, y_train, X_test, y_test, method="svm"):
    """Run SPI-SPI baseline with SVM or Random Forest on one fold."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if method == "svm":
        clf = SVC(kernel="rbf", C=1.0, gamma="scale")
    elif method == "rf":
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")

    clf.fit(X_train_s, y_train)
    return accuracy_score(y_test, clf.predict(X_test_s))


# =============================================================================
# 6. MAIN
# =============================================================================

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Topology classification via MPNN")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing pipeline output or pre-assembled arrays")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ablations", action="store_true",
        help="Run ablation controls (shuffled-edge MPNN, flat MLP, flat SVM/RF)",
    )
    parser.add_argument(
        "--plot", nargs="?", const="auto", default=None,
        metavar="PATH",
        help="Save training curves plot. 'auto' saves to <data_dir>/training_curves.png",
    )
    args = parser.parse_args(argv)

    _require_torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    mpi_tensors, labels, spi_names = load_data(args.data_dir)
    N, K, M, _ = mpi_tensors.shape
    num_classes = len(set(labels))

    print("Normalising edge features...")
    mpi_tensors_norm = normalise_edge_features(mpi_tensors)

    print("Building PyG graphs...")
    pyg_dataset = build_dataset(mpi_tensors_norm, labels)
    num_node_features = pyg_dataset[0].x.shape[1]
    num_edge_features = pyg_dataset[0].edge_attr.shape[1]

    print("Computing SPI-SPI features...")
    spi_spi_features = build_spi_spi_dataset(mpi_tensors_norm)
    print(f"  SPI-SPI feature dimension: {spi_spi_features.shape[1]}")

    # Ablation-specific pre-computation
    if args.ablations:
        print("Computing ablation features...")
        flat_edge_features = build_flat_edge_dataset(mpi_tensors_norm)
        print(f"  Flat-edge feature dimension: {flat_edge_features.shape[1]}")
        pyg_shuffled = build_shuffled_dataset(mpi_tensors_norm, labels, seed=args.seed)

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    results = {"mpnn": [], "spi_spi_svm": [], "spi_spi_rf": []}
    all_curves: Dict[str, list] = {"mpnn": []}
    if args.ablations:
        results.update({
            "mpnn_shuffled": [], "flat_mlp": [],
            "flat_svm": [], "flat_rf": [],
        })
        all_curves["mpnn_shuffled"] = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(N), labels)):
        print(f"\n--- Fold {fold + 1}/{args.n_folds} ---")

        train_ds = [pyg_dataset[i] for i in train_idx]
        test_ds = [pyg_dataset[i] for i in test_idx]

        mpnn_acc, best_epoch, mpnn_curves = run_mpnn_fold(
            train_ds, test_ds,
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            num_classes=num_classes,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
        )
        results["mpnn"].append(mpnn_acc)
        all_curves["mpnn"].append(mpnn_curves)
        print(f"  MPNN:         {mpnn_acc:.4f} (best epoch: {best_epoch})")

        svm_acc = run_baseline_fold(
            spi_spi_features[train_idx], labels[train_idx],
            spi_spi_features[test_idx], labels[test_idx], method="svm",
        )
        results["spi_spi_svm"].append(svm_acc)
        print(f"  SPI-SPI+SVM:  {svm_acc:.4f}")

        rf_acc = run_baseline_fold(
            spi_spi_features[train_idx], labels[train_idx],
            spi_spi_features[test_idx], labels[test_idx], method="rf",
        )
        results["spi_spi_rf"].append(rf_acc)
        print(f"  SPI-SPI+RF:   {rf_acc:.4f}")

        if args.ablations:
            # Ablation 1: MPNN with shuffled edge features
            shuf_train = [pyg_shuffled[i] for i in train_idx]
            shuf_test = [pyg_shuffled[i] for i in test_idx]
            shuf_acc, shuf_ep, shuf_curves = run_mpnn_fold(
                shuf_train, shuf_test,
                num_node_features=num_node_features,
                num_edge_features=num_edge_features,
                num_classes=num_classes,
                hidden_dim=args.hidden_dim,
                epochs=args.epochs,
                lr=args.lr,
                device=device,
            )
            results["mpnn_shuffled"].append(shuf_acc)
            all_curves["mpnn_shuffled"].append(shuf_curves)
            print(f"  MPNN-shuf:    {shuf_acc:.4f} (best epoch: {shuf_ep})")

            # Ablation 2: Flat MLP on raw upper-triangle features
            mlp_acc, mlp_ep = run_mlp_fold(
                flat_edge_features[train_idx], labels[train_idx],
                flat_edge_features[test_idx], labels[test_idx],
                num_classes=num_classes, device=device,
            )
            results["flat_mlp"].append(mlp_acc)
            print(f"  Flat-MLP:     {mlp_acc:.4f} (best epoch: {mlp_ep})")

            # Ablation 3: Flat SVM/RF on raw upper-triangle features
            flat_svm = run_baseline_fold(
                flat_edge_features[train_idx], labels[train_idx],
                flat_edge_features[test_idx], labels[test_idx], method="svm",
            )
            results["flat_svm"].append(flat_svm)
            print(f"  Flat-SVM:     {flat_svm:.4f}")

            flat_rf = run_baseline_fold(
                flat_edge_features[train_idx], labels[train_idx],
                flat_edge_features[test_idx], labels[test_idx], method="rf",
            )
            results["flat_rf"].append(flat_rf)
            print(f"  Flat-RF:      {flat_rf:.4f}")

    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    for method, accs in results.items():
        print(f"  {method:20s}: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")

    if args.ablations:
        print("\n" + "=" * 50)
        print("ABLATION INTERPRETATION")
        print("=" * 50)
        m = {k: np.mean(v) for k, v in results.items()}

        # Q1: Does message-passing help, or is it just having per-edge features?
        if m["mpnn"] > m["flat_mlp"] + 0.03:
            print("  MPNN > Flat-MLP  =>  message-passing adds value beyond raw features.")
        elif m["flat_mlp"] > m["mpnn"] + 0.03:
            print("  Flat-MLP > MPNN  =>  graph structure not helping; MPNN may be underfitting.")
        else:
            print("  MPNN ~ Flat-MLP  =>  graph structure provides no clear advantage.")

        # Q2: Does the MPNN use coherent edge-node assignment?
        if m["mpnn"] > m["mpnn_shuffled"] + 0.03:
            print("  MPNN > MPNN-shuf =>  MPNN exploits which-edge-has-which-value.")
        else:
            print("  MPNN ~ MPNN-shuf =>  MPNN only uses aggregate edge statistics.")

        # Q3: Is the SPI-SPI baseline bad because of topology loss or just lossy compression?
        if m["flat_svm"] > m["spi_spi_svm"] + 0.03:
            print("  Flat-SVM > SPI-SPI+SVM =>  SPI-SPI Pearson compression is the bottleneck.")
        else:
            print("  Flat-SVM ~ SPI-SPI+SVM =>  SPI-SPI compression retains most information.")

    # Plot training curves if requested
    if args.plot:
        plot_path = (
            Path(args.data_dir) / "training_curves.png"
            if args.plot == "auto"
            else Path(args.plot)
        )
        # Only plot methods that have curves (neural net methods)
        curves_to_plot = {k: v for k, v in all_curves.items() if v}
        if curves_to_plot:
            plot_training_curves(curves_to_plot, save_path=plot_path)


if __name__ == "__main__":
    sys.exit(main())
