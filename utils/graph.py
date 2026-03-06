from __future__ import annotations

import numpy as np
import torch


def load_adjacency(adj_path: str, device: torch.device) -> torch.Tensor:
    adj = np.load(adj_path)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"Expected square adjacency matrix, got {adj.shape}")
    return torch.from_numpy(adj.astype(np.float32)).to(device)
