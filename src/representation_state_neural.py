"""Small raw encoder that interacts across channels before invariant pooling."""
from __future__ import annotations

import random
import time

import numpy as np
import torch
from torch import nn


class AlignedChannelEncoder(nn.Module):
    """Input B x T x M; output one unbounded regression score per record.

    Temporal patch locations remain aligned during attention. Channel identity
    embeddings are deliberately absent for this exchangeable-oscillator pilot.
    No input normalization or observation augmentation is performed.
    """

    def __init__(self, spec: dict):
        super().__init__()
        layers = []
        width = 1
        for layer in spec["temporal_convolutions"]:
            layers.extend([nn.Conv1d(width, layer["width"], layer["kernel"],
                                     stride=layer["stride"], padding=layer["padding"]),
                           nn.GELU(), nn.Dropout(spec["dropout"])])
            width = layer["width"]
        if width != spec["attention_width"]:
            raise ValueError("temporal and attention widths must match")
        self.temporal = nn.Sequential(*layers)
        self.channel = nn.ModuleList([
            nn.TransformerEncoderLayer(width, spec["attention_heads"], spec["feedforward_width"],
                                       dropout=spec["dropout"], activation="gelu", batch_first=True)
            for _ in range(spec["channel_attention_blocks"])])
        post = spec["post_attention_temporal_convolution"]
        self.post = nn.Conv1d(width, post["width"], post["kernel"],
                              stride=post["stride"], padding=post["padding"])
        self.head = nn.Sequential(nn.Linear(2 * post["width"], spec["readout"][0]),
                                  nn.GELU(), nn.Linear(spec["readout"][0], 1))

    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        b, t, m = x.shape
        temporal = self.temporal(x.permute(0, 2, 1).reshape(b * m, 1, t))
        d, p = temporal.shape[1:]
        # Each attention set contains all channels at ONE common time patch.
        tokens = temporal.reshape(b, m, d, p).permute(0, 3, 1, 2).reshape(b * p, m, d)
        for layer in self.channel:
            tokens = layer(tokens)
        return tokens.reshape(b, p, m, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokens(x)
        b, p, m, d = tokens.shape
        local = self.post(tokens.permute(0, 2, 3, 1).reshape(b * m, d, p))
        local = local.reshape(b, m, local.shape[1], local.shape[2])
        pooled = torch.cat([local.mean(dim=(1, 3)), local.amax(dim=(1, 3))], dim=1)
        return self.head(pooled).squeeze(-1)


class PairRelationEncoder(nn.Module):
    """Shared raw two-channel temporal encoder, then invariant dyad pooling.

    Both orientations of every off-diagonal pair are used. No channel identities,
    simulator labels, precomputed dependence measures or pair subsampling enter.
    """
    def __init__(self, spec: dict):
        super().__init__()
        pointwise = spec.get('architecture') == 'pointwise_pair'
        k1,s1,p1 = (1,1,0) if pointwise else (7,4,3)
        k2,s2,p2 = (1,1,0) if pointwise else (5,4,2)
        self.temporal = nn.Sequential(nn.Conv1d(2,16,k1,stride=s1,padding=p1), nn.GELU(),
                                      nn.Conv1d(16,32,k2,stride=s2,padding=p2), nn.GELU())
        self.head = nn.Sequential(nn.Linear(128,32),nn.GELU(),nn.Linear(32,1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b,t,m=x.shape
        mask=~torch.eye(m,dtype=torch.bool,device=x.device)
        i,j=mask.nonzero(as_tuple=True)
        pairs=torch.stack((x[:,:,i],x[:,:,j]),dim=2).permute(0,3,2,1)
        encoded=self.temporal(pairs.reshape(b*len(i),2,t))
        per_pair=torch.cat((encoded.mean(-1),torch.sqrt(encoded.var(-1,unbiased=False)+1e-6)),dim=1)
        per_pair=per_pair.reshape(b,len(i),64)
        pooled=torch.cat((per_pair.mean(1),torch.sqrt(per_pair.var(1,unbiased=False)+1e-6)),dim=1)
        return self.head(pooled).squeeze(-1)


def make_encoder(spec: dict) -> nn.Module:
    architecture=spec.get('architecture','aligned_channel')
    if architecture=='spi_edge_pool':
        from src.spi_edge_pool import SPIEdgePool
        return SPIEdgePool(spec)
    if architecture=='aligned_channel': return AlignedChannelEncoder(spec)
    if architecture in ('pair_relation','pointwise_pair'): return PairRelationEncoder(spec)
    raise ValueError(f'Unknown encoder architecture: {architecture}')


def seed_torch(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
    # Record hardware/runtime; exact bitwise cross-device reproducibility is not
    # claimed. Disallow TF32 as a silent comparison change on CUDA hardware.
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def predict(model: nn.Module, x: torch.Tensor, batch_size: int, *, clip: bool = True) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        values = torch.cat([model(x[start:start + batch_size])
                            for start in range(0, len(x), batch_size)]).detach().cpu().numpy()
    return np.clip(values, 0, 1) if clip else values


def fit_encoder(x: torch.Tensor, y: torch.Tensor, spec: dict, lr: float, weight_decay: float,
                seed: int, validation: tuple[torch.Tensor, torch.Tensor] | None = None,
                epochs: int | None = None) -> tuple[nn.Module, dict]:
    """Early stop only on an inner fold; final refits receive an epoch count.

    No held-out evaluation data are accepted by this function. For final refits
    the caller uses median best epoch selected entirely inside the label budget.
    """
    if validation is None and epochs is None:
        raise ValueError("final fitting requires a preselected epoch count")
    stopping_metric = spec.get('early_stopping_metric', 'validation_MAE')
    if stopping_metric not in ('validation_MAE', 'validation_unclipped_MAE'):
        raise ValueError(f'Unknown early stopping metric: {stopping_metric}')
    seed_torch(seed)
    model = make_encoder(spec).to(x.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    maximum = spec["maximum_epochs"] if epochs is None else epochs
    batch_size = spec["batch_size"]
    best, best_epoch, best_state = float("inf"), 0, None
    best_reported_mae = None
    history = []
    start = time.perf_counter()
    for epoch in range(1, maximum + 1):
        model.train()
        order = torch.randperm(len(x), generator=generator).to(x.device)
        losses = []
        for indices in order.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(x[indices]), y[indices])
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite neural training loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()) * len(indices))
        row = {"epoch": epoch, "training_MSE": sum(losses) / len(x)}
        if validation is not None:
            vx, vy = validation
            raw_prediction = predict(model, vx, batch_size, clip=False)
            validation_target = vy.detach().cpu().numpy()
            mae = float(np.abs(np.clip(raw_prediction, 0, 1) - validation_target).mean())
            raw_mae = float(np.abs(raw_prediction - validation_target).mean())
            stopping_error = raw_mae if stopping_metric == 'validation_unclipped_MAE' else mae
            row["validation_MAE"] = mae
            row['validation_unclipped_MAE'] = raw_mae
            if stopping_error < best:
                best, best_epoch = stopping_error, epoch
                best_reported_mae = mae
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            if epoch >= spec["minimum_epochs"] and epoch - best_epoch >= spec["early_stopping_patience"]:
                history.append(row)
                break
        history.append(row)
    if validation is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = maximum
    return model, {"best_epoch": best_epoch, "validation_MAE": best_reported_mae,
                   "stopping_metric": stopping_metric,
                   "best_stopping_error": best if validation is not None else None,
                   "epochs_run": len(history), "history": history, "seconds": time.perf_counter() - start,
                   "parameter_count": sum(p.numel() for p in model.parameters())}
