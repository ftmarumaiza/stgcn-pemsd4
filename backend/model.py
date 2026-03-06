from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


def normalized_adjacency(adj: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    deg = adj.sum(dim=-1)
    deg_inv_sqrt = torch.pow(deg + eps, -0.5)
    d_mat = torch.diag(deg_inv_sqrt)
    return d_mat @ adj @ d_mat


class SpectralGraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        adj_norm = normalized_adjacency(adj)
        support = torch.einsum("nm,btnc->btmc", adj_norm, x)
        return self.proj(support)


class SpatialBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        self.gconv = SpectralGraphConv(in_channels, out_channels)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = self.gconv(x, adj)
        x = self.act(x)
        x = self.drop(x)
        return x


class TemporalConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, steps, nodes, channels = x.shape
        x = x.permute(0, 2, 3, 1).reshape(bsz * nodes, channels, steps)
        x = self.conv(x)
        x = self.act(x)
        x = x.reshape(bsz, nodes, -1, steps).permute(0, 3, 1, 2)
        return x


class STGCN(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        in_channels: int = 1,
        spatial_hidden: int = 32,
        temporal_hidden: int = 64,
        horizon: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        if int(horizon) != 1:
            raise ValueError(f"Traffexplainer paper setting requires horizon=1, got {horizon}")
        self.horizon = 1

        self.spatial1 = SpatialBlock(in_channels, spatial_hidden, dropout=dropout)
        self.spatial2 = SpatialBlock(spatial_hidden, spatial_hidden, dropout=dropout)
        self.temporal = TemporalConvBlock(spatial_hidden, temporal_hidden)
        self.fc = nn.Linear(temporal_hidden, 1)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        adj1: torch.Tensor | None = None,
        adj2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        a1 = adj if adj1 is None else adj1
        a2 = adj if adj2 is None else adj2

        x = self.spatial1(x, a1)
        x = self.spatial2(x, a2)
        x = self.temporal(x)

        x_last = x[:, -1, :, :]
        out = self.fc(x_last)
        return out


def load_adjacency(adj_path: str, device: torch.device) -> torch.Tensor:
    adj = np.load(adj_path).astype(np.float32)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"Expected square adjacency matrix, got {adj.shape}")
    return torch.from_numpy(adj).to(device)


@dataclass
class Metrics:
    MAE: float
    RMSE: float
    MAPE: float


def evaluate_metrics(model: STGCN, loader, adj: torch.Tensor, scaler, device: torch.device) -> Metrics:
    model.eval()
    preds, tgts = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x, adj).detach().cpu().numpy().squeeze(-1)
            y_np = y.detach().cpu().numpy().squeeze(-1)
            out = scaler.inverse_transform(out)
            y_np = scaler.inverse_transform(y_np)
            preds.append(out)
            tgts.append(y_np)

    pred = np.concatenate(preds, axis=0)
    tgt = np.concatenate(tgts, axis=0)
    diff = pred - tgt
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    mape = float(np.mean(np.abs(diff) / np.clip(np.abs(tgt), 1e-5, None)) * 100.0)
    return Metrics(MAE=mae, RMSE=rmse, MAPE=mape)
