from __future__ import annotations

import torch
import torch.nn as nn


class HierarchicalMasks(nn.Module):
    def __init__(self, num_nodes: int, history: int):
        super().__init__()
        self.ms1 = nn.Parameter(torch.zeros(num_nodes, num_nodes))
        self.ms2 = nn.Parameter(torch.zeros(num_nodes, num_nodes))
        self.mt = nn.Parameter(torch.zeros(history, num_nodes))

    def symmetrized_ms1(self) -> torch.Tensor:
        return 0.5 * (self.ms1 + self.ms1.T)

    def symmetrized_ms2(self) -> torch.Tensor:
        return 0.5 * (self.ms2 + self.ms2.T)

    def masked_adjacencies(self, adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ms1_sym = self.symmetrized_ms1()
        ms2_sym = self.symmetrized_ms2()
        a1 = adj * (torch.tanh(ms1_sym) + 1.0)
        a2 = adj * (torch.tanh(ms2_sym) + 1.0)
        return a1, a2

    def masked_input(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N, C], mt_scale: [1, T, N, 1]
        mt_scale = (torch.tanh(self.mt) + 1.0).unsqueeze(0).unsqueeze(-1)
        return x * mt_scale

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a1, a2 = self.masked_adjacencies(adj)
        x_masked = self.masked_input(x)
        return x_masked, a1, a2
