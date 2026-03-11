"""
Training loop for the EEML MPNN pipeline.

Handles:
    - Train / val / test split
    - Early stopping on validation macro-F1
    - L1 regularisation on SPI mixture weights
    - Gradient clipping
    - Per-epoch logging
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from .model import SPIEdgeMPNN


@dataclass
class TrainConfig:
    lr: float = 2e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 150
    patience: int = 20
    grad_clip: float = 1.0
    l1_lambda: float = 1e-3
    device: str = "cpu"


@dataclass
class TrainResult:
    best_val_f1: float = 0.0
    best_epoch: int = 0
    test_f1: float = 0.0
    test_acc: float = 0.0
    learned_w: np.ndarray = field(default_factory=lambda: np.array([]))
    learned_b: float = 0.0
    train_losses: list[float] = field(default_factory=list)
    val_f1s: list[float] = field(default_factory=list)
    train_seconds: float = 0.0


def split_data(
    dataset: list[Data],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 0,
) -> tuple[list[Data], list[Data], list[Data]]:
    """
    Stratified-ish split by shuffling then slicing.

    For proper stratification across classes, the caller should ensure
    balanced class representation in the dataset.
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(dataset))
    n_train = int(len(dataset) * train_ratio)
    n_val = int(len(dataset) * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    return (
        [dataset[i] for i in train_idx],
        [dataset[i] for i in val_idx],
        [dataset[i] for i in test_idx],
    )


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    """Evaluate model, return (macro_f1, accuracy)."""
    model.eval()
    all_preds = []
    all_labels = []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        preds = logits.argmax(dim=1).cpu().numpy()
        labels = batch.y.cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    return float(f1), float(acc)


def train_model(
    model: SPIEdgeMPNN,
    train_data: list[Data],
    val_data: list[Data],
    test_data: list[Data],
    config: TrainConfig,
) -> TrainResult:
    """
    Full training loop with early stopping on validation macro-F1.

    Returns TrainResult with metrics and learned parameters.
    """
    device = torch.device(config.device)
    model = model.to(device)

    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=config.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=config.batch_size,
        shuffle=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    result = TrainResult()
    best_state = None
    patience_counter = 0
    start_time = time.perf_counter()

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            logits = model(batch)
            loss = criterion(logits, batch.y)

            # L1 on SPI mixture weights
            if hasattr(model, "spi_w"):
                loss = loss + config.l1_lambda * model.spi_w.abs().sum()

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        result.train_losses.append(avg_loss)

        # Validation
        val_f1, val_acc = _evaluate(model, val_loader, device)
        result.val_f1s.append(val_f1)

        if val_f1 > result.best_val_f1:
            result.best_val_f1 = val_f1
            result.best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            w_str = ""
            if hasattr(model, "spi_w"):
                w = model.spi_w.detach().cpu().numpy()
                top3 = np.argsort(np.abs(w))[-3:][::-1]
                w_str = f" | top-3 |w|: {', '.join(f'{w[i]:+.3f}' for i in top3)}"
            print(
                f"  [Epoch {epoch:3d}] loss={avg_loss:.4f}  "
                f"val_f1={val_f1:.4f}  val_acc={val_acc:.4f}{w_str}"
            )

        if patience_counter >= config.patience:
            print(f"  Early stopping at epoch {epoch} (best={result.best_epoch})")
            break

    result.train_seconds = time.perf_counter() - start_time

    # Restore best model and evaluate on test
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    test_f1, test_acc = _evaluate(model, test_loader, device)
    result.test_f1 = test_f1
    result.test_acc = test_acc

    if hasattr(model, "spi_w"):
        result.learned_w = model.spi_w.detach().cpu().numpy().copy()
        result.learned_b = model.spi_b.detach().cpu().item()

    print(
        f"  [Result] test_f1={test_f1:.4f}  test_acc={test_acc:.4f}  "
        f"best_epoch={result.best_epoch}  train_time={result.train_seconds:.1f}s"
    )

    return result
