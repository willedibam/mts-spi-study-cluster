"""
MPNN with edge network for graph-level MTS classification.

Architecture:
    1. Global adjacency prior: A_0(i,j) = softplus(b + sum_k w_k * e_ij_k)
    2. Top-d sparsification per node (directed, no self-loops)
    3. Edge network MLP on e_ij → phi_e
    4. Message MLP on [h_i, h_j, phi_e] → message
    5. Aggregation with A_0 weighting
    6. Residual + LayerNorm update
    7. Mean+max pool → classifier MLP

Learnable global SPI mixture weights w are shared across all graphs
in a batch and across all layers.

PyG batching note:
    spi_tensor is stored as (M, M, K) per graph. PyG concatenates it
    along dim 0 to (B*M, M, K) during batching. We recover per-graph
    tensors using to_dense_batch → (B, M, M, K).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.utils import to_dense_batch


def _unbatch_spi(batch: Batch) -> torch.Tensor:
    """
    Recover per-graph SPI tensors from PyG's concatenated batching.

    PyG stores spi_tensor as (B*M, M, K) after batching individual
    (M, M, K) tensors. This function returns (B, M, M, K).
    """
    # spi_tensor: (B*M, M, K), batch.batch: (B*M,) with graph indices
    spi_flat = batch.spi_tensor  # (B*M, M, K)
    # to_dense_batch pads variable-length sequences; for fixed M it just reshapes
    # It expects (N, *) and batch vector (N,), returns (B, M_max, *)
    dense, _ = to_dense_batch(spi_flat, batch.batch)  # (B, M, M, K)
    return dense


class SPIEdgeMPNN(nn.Module):
    """
    Full model: adjacency prior → sparsify → message passing → classify.

    Args:
        n_spi: K, number of SPI dimensions (edge feature size).
        n_node_features: F_n, node feature dimension.
        n_classes: C, number of output classes.
        hidden: H, hidden width. Default 64.
        n_layers: Number of message passing layers. Default 3.
        top_d: Number of outgoing edges to retain per node. Default 4.
        dropout: Dropout rate. Default 0.1.
    """

    def __init__(
        self,
        n_spi: int,
        n_node_features: int,
        n_classes: int,
        hidden: int = 64,
        n_layers: int = 3,
        top_d: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_spi = n_spi
        self.hidden = hidden
        self.top_d = top_d
        self.n_layers = n_layers

        # Global learnable SPI mixture weights (shared across graphs and layers)
        self.spi_w = nn.Parameter(torch.zeros(n_spi))
        self.spi_b = nn.Parameter(torch.tensor(-2.0))

        # Node projection
        self.node_proj = nn.Linear(n_node_features, hidden)

        # Edge network: K → 64
        self.edge_net = nn.Sequential(
            nn.Linear(n_spi, 32),
            nn.GELU(),
            nn.Linear(32, 64),
            nn.GELU(),
        )

        # Message passing layers
        self.mp_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.mp_layers.append(MPLayer(hidden, edge_dim=64, dropout=dropout))

        # Classifier: mean+max pool (2*H) → 64 → C
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def compute_adjacency_prior(self, spi_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute A_0(i,j) = softplus(b + sum_k w_k * e_ij_k).

        Args:
            spi_tensor: (M, M, K)

        Returns:
            (M, M) adjacency weights (non-negative).
        """
        scores = torch.einsum("ijk,k->ij", spi_tensor, self.spi_w) + self.spi_b
        return F.softplus(scores)

    def sparsify(
        self, adj: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Top-d outgoing edges per node, excluding self-loops.

        Args:
            adj: (M, M) adjacency weights.

        Returns:
            edge_index: (2, E) long tensor
            edge_weight: (E,) float tensor
        """
        M = adj.shape[0]
        mask = ~torch.eye(M, dtype=torch.bool, device=adj.device)
        adj_masked = adj * mask.float()

        d = min(self.top_d, M - 1)
        _, top_indices = adj_masked.topk(d, dim=1)  # (M, d)

        src = torch.arange(M, device=adj.device).unsqueeze(1).expand(-1, d).reshape(-1)
        dst = top_indices.reshape(-1)
        edge_index = torch.stack([src, dst], dim=0)  # (2, M*d)
        edge_weight = adj_masked[src, dst]

        return edge_index, edge_weight

    def forward(self, batch: Batch) -> torch.Tensor:
        """
        Forward pass for a batched set of graphs.

        Returns logits (B, C).
        """
        device = batch.x.device
        batch_size = batch.num_graphs

        # Recover per-graph SPI tensors: (B, M, M, K)
        spi_dense = _unbatch_spi(batch)

        all_edge_index = []
        all_edge_attr = []
        all_edge_weight = []
        all_batch_vec = []
        node_offset = 0

        for i in range(batch_size):
            node_mask = batch.batch == i
            M_i = node_mask.sum().item()
            spi_i = spi_dense[i, :M_i, :M_i, :]  # (M_i, M_i, K)

            # Adjacency prior + sparsify
            adj = self.compute_adjacency_prior(spi_i)
            edge_index_i, edge_weight_i = self.sparsify(adj)

            # Edge features from SPI tensor
            src, dst = edge_index_i
            edge_spi = spi_i[src, dst]  # (E_i, K)
            edge_attr_i = self.edge_net(edge_spi)  # (E_i, 64)

            all_edge_index.append(edge_index_i + node_offset)
            all_edge_attr.append(edge_attr_i)
            all_edge_weight.append(edge_weight_i)
            all_batch_vec.append(torch.full((M_i,), i, dtype=torch.long, device=device))
            node_offset += M_i

        edge_index = torch.cat(all_edge_index, dim=1)
        edge_attr = torch.cat(all_edge_attr, dim=0)
        edge_weight = torch.cat(all_edge_weight, dim=0)
        batch_vec = torch.cat(all_batch_vec, dim=0)

        # Node projection
        h = self.node_proj(batch.x)

        # Message passing
        for layer in self.mp_layers:
            h = layer(h, edge_index, edge_attr, edge_weight)

        # Graph readout
        h_mean = global_mean_pool(h, batch_vec)
        h_max = global_max_pool(h, batch_vec)
        h_graph = torch.cat([h_mean, h_max], dim=1)

        return self.classifier(h_graph)


class MPLayer(nn.Module):
    """
    Single message passing layer with edge network.

    m_i = sum_j A(i,j) * MLP([h_i, h_j, phi_e(e_ij)])
    h_i = h_i + LayerNorm(m_i)   (residual + LN update)
    """

    def __init__(self, hidden: int, edge_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * hidden + edge_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, hidden),
        )
        self.layer_norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index

        msg_input = torch.cat([h[dst], h[src], edge_attr], dim=1)
        messages = self.message_mlp(msg_input)
        messages = messages * edge_weight.unsqueeze(1)

        N = h.shape[0]
        agg = torch.zeros(N, messages.shape[1], device=h.device, dtype=h.dtype)
        agg.scatter_add_(0, dst.unsqueeze(1).expand_as(messages), messages)

        h = h + self.dropout(self.layer_norm(agg))
        return h


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


class CorrelationOnlyMPNN(SPIEdgeMPNN):
    """
    Baseline 1: Same architecture but edge features are correlation-only (K=1).

    Expects spi_tensor with K=1 (e.g., just Pearson correlation).
    """
    pass  # Identical architecture; difference is in the data (K=1)


class DenseLearnedGraphMPNN(nn.Module):
    """
    Baseline 2: No SPI prior. Learn adjacency from node embeddings.

    A_learn(i,j) = softplus(u_i^T u_j), then top-d sparsify.
    """

    def __init__(
        self,
        n_node_features: int,
        n_classes: int,
        hidden: int = 64,
        n_layers: int = 3,
        top_d: int = 4,
        dropout: float = 0.1,
        adj_dim: int = 16,
    ):
        super().__init__()
        self.hidden = hidden
        self.top_d = top_d

        self.adj_embed = None  # Initialized lazily based on M
        self.adj_dim = adj_dim

        self.node_proj = nn.Linear(n_node_features, hidden)

        self.mp_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.mp_layers.append(_ScalarEdgeMPLayer(hidden, dropout=dropout))

        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def _ensure_adj_embed(self, M: int, device: torch.device):
        if self.adj_embed is None or self.adj_embed.shape[0] != M:
            self.adj_embed = nn.Parameter(
                torch.randn(M, self.adj_dim, device=device) * 0.1
            )

    def forward(self, batch: Batch) -> torch.Tensor:
        device = batch.x.device
        batch_size = batch.num_graphs

        all_edge_index = []
        all_edge_weight = []
        all_batch_vec = []
        node_offset = 0

        for i in range(batch_size):
            node_mask = batch.batch == i
            M_i = node_mask.sum().item()
            self._ensure_adj_embed(M_i, device)

            adj = F.softplus(self.adj_embed @ self.adj_embed.T)
            mask = ~torch.eye(M_i, dtype=torch.bool, device=device)
            adj = adj * mask.float()

            d = min(self.top_d, M_i - 1)
            _, top_idx = adj.topk(d, dim=1)
            src = torch.arange(M_i, device=device).unsqueeze(1).expand(-1, d).reshape(-1)
            dst = top_idx.reshape(-1)

            all_edge_index.append(torch.stack([src + node_offset, dst + node_offset]))
            all_edge_weight.append(adj[src, dst])
            all_batch_vec.append(torch.full((M_i,), i, dtype=torch.long, device=device))
            node_offset += M_i

        edge_index = torch.cat(all_edge_index, dim=1)
        edge_weight = torch.cat(all_edge_weight)
        batch_vec = torch.cat(all_batch_vec)

        h = self.node_proj(batch.x)
        for layer in self.mp_layers:
            h = layer(h, edge_index, edge_weight)

        h_mean = global_mean_pool(h, batch_vec)
        h_max = global_max_pool(h, batch_vec)
        h_graph = torch.cat([h_mean, h_max], dim=1)
        return self.classifier(h_graph)


class _ScalarEdgeMPLayer(nn.Module):
    """Message passing with scalar edge weights only (no edge features)."""

    def __init__(self, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * hidden, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, hidden),
        )
        self.layer_norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, edge_index, edge_weight):
        src, dst = edge_index
        msg_input = torch.cat([h[dst], h[src]], dim=1)
        messages = self.message_mlp(msg_input) * edge_weight.unsqueeze(1)
        N = h.shape[0]
        agg = torch.zeros(N, messages.shape[1], device=h.device, dtype=h.dtype)
        agg.scatter_add_(0, dst.unsqueeze(1).expand_as(messages), messages)
        return h + self.dropout(self.layer_norm(agg))


class EdgeAblationMPNN(SPIEdgeMPNN):
    """
    Baseline 3: SPI-derived A_0, but no edge features in message function.

    Only scalar A_0(i,j) is used as edge weight; the edge network is bypassed.
    Zero vectors are passed as edge attributes.
    """

    def forward(self, batch: Batch) -> torch.Tensor:
        device = batch.x.device
        batch_size = batch.num_graphs

        spi_dense = _unbatch_spi(batch)

        all_edge_index = []
        all_edge_weight = []
        all_batch_vec = []
        node_offset = 0

        for i in range(batch_size):
            node_mask = batch.batch == i
            M_i = node_mask.sum().item()
            spi_i = spi_dense[i, :M_i, :M_i, :]

            adj = self.compute_adjacency_prior(spi_i)
            edge_index_i, edge_weight_i = self.sparsify(adj)

            all_edge_index.append(edge_index_i + node_offset)
            all_edge_weight.append(edge_weight_i)
            all_batch_vec.append(torch.full((M_i,), i, dtype=torch.long, device=device))
            node_offset += M_i

        edge_index = torch.cat(all_edge_index, dim=1)
        edge_weight = torch.cat(all_edge_weight)
        batch_vec = torch.cat(all_batch_vec)

        h = self.node_proj(batch.x)

        # Zero edge features — bypasses edge network
        zero_attr = torch.zeros(edge_index.shape[1], 64, device=device)
        for layer in self.mp_layers:
            h = layer(h, edge_index, zero_attr, edge_weight)

        h_mean = global_mean_pool(h, batch_vec)
        h_max = global_max_pool(h, batch_vec)
        h_graph = torch.cat([h_mean, h_max], dim=1)
        return self.classifier(h_graph)
