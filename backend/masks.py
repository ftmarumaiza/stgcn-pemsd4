from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam


class HierarchicalMasks(nn.Module):
    def __init__(self, num_nodes: int, history: int):
        super().__init__()
        self.ms1 = nn.Parameter(torch.zeros(num_nodes, num_nodes))
        self.ms2 = nn.Parameter(torch.zeros(num_nodes, num_nodes))
        self.mt = nn.Parameter(torch.zeros(history, num_nodes))

    def sym_ms1(self) -> torch.Tensor:
        return 0.5 * (self.ms1 + self.ms1.T)

    def sym_ms2(self) -> torch.Tensor:
        return 0.5 * (self.ms2 + self.ms2.T)

    def masked_adjacencies(self, adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a1 = adj * (torch.tanh(self.sym_ms1()) + 1.0)
        a2 = adj * (torch.tanh(self.sym_ms2()) + 1.0)
        return a1, a2

    def masked_input(self, x: torch.Tensor) -> torch.Tensor:
        mt_scale = (torch.tanh(self.mt) + 1.0).unsqueeze(0).unsqueeze(-1)
        return x * mt_scale

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        x_masked = self.masked_input(x)
        a1, a2 = self.masked_adjacencies(adj)
        return x_masked, a1, a2


def train_masks_only(
    predictor: torch.nn.Module,
    loaders,
    adj: torch.Tensor,
    device: torch.device,
    history: int,
    epochs: int = 30,
    lr: float = 1e-2,
    save_dir: str = "artifacts",
) -> HierarchicalMasks:
    os.makedirs(save_dir, exist_ok=True)

    predictor.eval()
    for p in predictor.parameters():
        p.requires_grad = False

    num_nodes = adj.shape[0]
    masks = HierarchicalMasks(num_nodes=num_nodes, history=history).to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(masks.parameters(), lr=lr)

    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        masks.train()
        train_loss = 0.0
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            x_masked, a1, a2 = masks(x, adj)
            pred = predictor(x_masked, adj=adj, adj1=a1, adj2=a2)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)

        train_loss /= len(loaders["train"].dataset)

        masks.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in loaders["val"]:
                x, y = x.to(device), y.to(device)
                x_masked, a1, a2 = masks(x, adj)
                pred = predictor(x_masked, adj=adj, adj1=a1, adj2=a2)
                val_loss += criterion(pred, y).item() * x.size(0)
        val_loss /= len(loaders["val"].dataset)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in masks.state_dict().items()}

        print(f"[Masks] epoch={epoch:03d} train_mse={train_loss:.6f} val_mse={val_loss:.6f}")

    if best_state is not None:
        masks.load_state_dict(best_state)

    ms1 = masks.sym_ms1().detach().cpu().numpy()
    ms2 = masks.sym_ms2().detach().cpu().numpy()
    mt = masks.mt.detach().cpu().numpy()

    np.save(os.path.join(save_dir, "MS1.npy"), ms1)
    np.save(os.path.join(save_dir, "MS2.npy"), ms2)
    np.save(os.path.join(save_dir, "MT.npy"), mt)
    torch.save({"mask_state": masks.state_dict()}, os.path.join(save_dir, "best_masks.pt"))
    return masks


def high_influence_edges_from_masks(
    adj: np.ndarray, ms1: np.ndarray, ms2: np.ndarray, threshold: float | None = None
) -> tuple[list[dict], float]:
    score = np.abs(ms1 * ms2) * adj
    edge_indices = np.argwhere(np.triu(adj > 0, k=1))
    if edge_indices.size == 0:
        return [], 0.0

    edge_scores = np.array([score[i, j] for i, j in edge_indices], dtype=np.float32)
    if threshold is None:
        threshold = float(np.mean(edge_scores))

    selected = edge_scores >= float(threshold)
    filtered_edges = []
    for idx in np.where(selected)[0]:
        i, j = edge_indices[idx]
        filtered_edges.append(
            {"from": int(i), "to": int(j), "importance": float(edge_scores[idx])}
        )
    return filtered_edges, float(threshold)
