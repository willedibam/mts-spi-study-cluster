"""InceptionTime architecture; independent PyTorch implementation.

Specification: Fawaz et al., arXiv:1909.04939. This reproduces the original
40/20/10 kernel architecture, not tsai's odd-kernel/identity-shortcut variant.
Training is explicitly source-CV adapted; see docs/inceptiontime-followup.md.
"""
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.neurotycho_learning import balanced_brier, date_state_weights, predict_binary
from src.representation_state_neural import seed_torch


class SameConv(nn.Conv1d):
    def forward(self, x):
        total = self.kernel_size[0] - 1
        return super().forward(F.pad(x, (total // 2, total - total // 2)))


def normalization(width, spec):
    return nn.BatchNorm1d(width, eps=spec['batch_norm_epsilon'],
                          momentum=spec['batch_norm_update_fraction'])


class InceptionModule(nn.Module):
    def __init__(self, channels, spec):
        super().__init__()
        width, bottleneck = spec['filters'], spec['bottleneck']
        self.bottleneck = nn.Conv1d(channels, bottleneck, 1, bias=False) if channels > 1 else nn.Identity()
        self.branches = nn.ModuleList([SameConv(bottleneck if channels > 1 else channels,
                                                width, k, bias=False) for k in spec['kernels']])
        self.pool = nn.Sequential(nn.MaxPool1d(3, stride=1, padding=1),
                                  nn.Conv1d(channels, width, 1, bias=False))
        self.norm = normalization(4 * width, spec)

    def forward(self, x):
        narrow = self.bottleneck(x)
        return F.relu(self.norm(torch.cat([*[branch(narrow) for branch in self.branches], self.pool(x)], 1)))


class InceptionNetwork(nn.Module):
    """B,T,M input; returns logit(class1)-logit(class0)."""
    def __init__(self, spec):
        super().__init__()
        self.channels = spec['input_channels']
        self.every = spec['residual_every']
        width = 4 * spec['filters']
        self.modules_list = nn.ModuleList([InceptionModule(self.channels if d == 0 else width, spec)
                                           for d in range(spec['depth'])])
        self.shortcuts = nn.ModuleList([
            nn.Sequential(nn.Conv1d(self.channels if d == 0 else width, width, 1, bias=False),
                          normalization(width, spec)) for d in range(spec['depth'] // self.every)])
        self.output = nn.Linear(width, spec['output_classes'])
        if spec['output_classes'] != 2 or spec['depth'] % self.every:
            raise ValueError('this benchmark requires binary output and complete residual blocks')
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        if x.ndim != 3 or x.shape[2] != self.channels:
            raise ValueError('fixed-channel InceptionTime expects B,T,M with the declared M')
        x = residual = x.transpose(1, 2)
        for d, module in enumerate(self.modules_list):
            x = module(x)
            if (d + 1) % self.every == 0:
                x = F.relu(x + self.shortcuts[d // self.every](residual))
                residual = x
        logits = self.output(x.mean(-1))
        return logits[:, 1] - logits[:, 0]


def fit_network(x, y, archives, spec, training, lr, decay, seed, validation=None, epochs=None):
    if validation is None and epochs is None:
        raise ValueError('refit requires a source-selected epoch count')
    seed_torch(seed)
    model = InceptionNetwork(spec).to(x.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=decay, eps=training['adam_epsilon'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=.5, patience=50, min_lr=.0001, threshold=1e-4, threshold_mode='abs')
    generator = torch.Generator().manual_seed(seed)
    target = torch.as_tensor(y, dtype=torch.float32, device=x.device)
    weights = torch.as_tensor(date_state_weights(archives, y) if archives is not None else np.ones(len(y)),
                              dtype=torch.float32, device=x.device)
    best, best_epoch, state = float('inf'), 0, None
    history, started = [], time.perf_counter()
    maximum = training['maximum_epochs'] if epochs is None else epochs
    for epoch in range(1, maximum + 1):
        model.train(); total = 0.
        for ix in torch.randperm(len(x), generator=generator).to(x.device).split(training['batch_size']):
            optimizer.zero_grad(set_to_none=True)
            loss = (F.binary_cross_entropy_with_logits(model(x[ix]), target[ix], reduction='none') * weights[ix]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite loss')
            loss.backward(); optimizer.step()
            total += float(loss.detach()) * len(ix)
        record = dict(epoch=epoch, training_bce=total / len(x), learning_rate=optimizer.param_groups[0]['lr'])
        scheduler.step(record['training_bce'])
        if validation is not None:
            vx, vy = validation
            score = balanced_brier(vy, predict_binary(model, vx, training['batch_size']))
            record['validation_brier'] = score
            if score < best - 1e-8:
                best, best_epoch = score, epoch
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(record)
        if validation is not None and epoch >= training['minimum_epochs'] and epoch - best_epoch >= training['patience']:
            break
    if validation is not None:
        model.load_state_dict(state)
    else:
        best_epoch = maximum
    return model, dict(history=history, best_epoch=best_epoch, epochs_run=len(history),
        selected_epoch_at_ceiling=validation is not None and best_epoch == maximum,
        validation_brier=None if validation is None else best,
        seconds=time.perf_counter()-started, parameters=sum(p.numel() for p in model.parameters()))
