from __future__ import annotations

import torch
import torch.nn as nn


def normalized_laplacian(adj: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Build symmetric normalized Laplacian:
    L = I - D^{-1/2} A D^{-1/2}
    """
    n = adj.shape[0]
    deg = torch.sum(adj, dim=1)
    deg_inv_sqrt = torch.pow(deg + eps, -0.5)
    d_inv_sqrt = torch.diag(deg_inv_sqrt)
    identity = torch.eye(n, device=adj.device, dtype=adj.dtype)
    return identity - d_inv_sqrt @ adj @ d_inv_sqrt


def scaled_laplacian(adj: torch.Tensor) -> torch.Tensor:
    """
    Scale Laplacian into [-1, 1] domain for Chebyshev recurrence.

    Uses lambda_max = 2.0 (theoretical upper bound for symmetric
    normalized Laplacian) instead of computing eigenvalues.
    This is more stable and avoids expensive eigval computation.
    """
    n = adj.shape[0]
    identity = torch.eye(n, device=adj.device, dtype=adj.dtype)
    lap = normalized_laplacian(adj)
    # Normalized Laplacian eigenvalues are in [0, 2], so lambda_max <= 2.
    lambda_max = 2.0
    return (2.0 * lap / lambda_max) - identity


def cheb_polynomials(adj: torch.Tensor, k_order: int) -> torch.Tensor:
    """
    Return Chebyshev bases T_k(\\tilde{L}) for k in [0, K-1], shape [K, N, N].
    """
    if k_order < 1:
        raise ValueError(f"k_order must be >= 1, got {k_order}")
    n = adj.shape[0]
    dtype = adj.dtype
    device = adj.device

    t0 = torch.eye(n, device=device, dtype=dtype)
    if k_order == 1:
        return t0.unsqueeze(0)

    l_tilde = scaled_laplacian(adj)
    t1 = l_tilde
    cheb = [t0, t1]
    for _ in range(2, k_order):
        tk = 2.0 * l_tilde @ cheb[-1] - cheb[-2]
        cheb.append(tk)
    return torch.stack(cheb, dim=0)


class TemporalConvLayer(nn.Module):
    """
    Gated temporal convolution (GLU) on (B, C, T, N).
    out = conv_filter(x) * sigmoid(conv_gate(x))
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        pad_t = kernel_size // 2
        self.conv_filter = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad_t, 0),
        )
        self.conv_gate = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad_t, 0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        filt = self.conv_filter(x)
        gate = torch.sigmoid(self.conv_gate(x))
        return filt * gate


class ChebGraphConv(nn.Module):
    """
    Chebyshev graph convolution (K-order) on (B, C, T, N).
    """

    def __init__(self, in_channels: int, out_channels: int, k_order: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k_order = k_order
        self.theta = nn.Parameter(torch.empty(k_order, in_channels, out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.xavier_uniform_(self.theta)

    def forward(self, x: torch.Tensor, cheb_basis: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, N], cheb_basis: [K, N, N]
        supports = []
        for k in range(self.k_order):
            # Apply graph propagation for each Chebyshev basis.
            support_k = torch.einsum("nm,bctm->bctn", cheb_basis[k], x)
            supports.append(support_k)
        support = torch.stack(supports, dim=1)  # [B, K, C, T, N]
        out = torch.einsum("bkctn,kco->botn", support, self.theta)  # [B, O, T, N]
        return out + self.bias.view(1, -1, 1, 1)


class STBlock(nn.Module):
    """
    Paper-style ST-Block:
    TemporalConv(GLU) -> ChebGraphConv -> ReLU -> TemporalConv(GLU) -> ReLU -> Dropout -> Residual

    No LayerNorm (not in the original STGCN paper).
    """

    def __init__(self, in_channels: int, out_channels: int, k_order: int = 3, dropout: float = 0.3):
        super().__init__()
        self.temp1 = TemporalConvLayer(in_channels, out_channels, kernel_size=3)
        self.graph = ChebGraphConv(out_channels, out_channels, k_order=k_order)
        self.temp2 = TemporalConvLayer(out_channels, out_channels, kernel_size=3)
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        )

    def forward(self, x: torch.Tensor, cheb_basis: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, N]
        residual = self.residual(x)
        out = self.temp1(x)
        out = self.graph(out, cheb_basis)
        out = torch.relu(out)
        out = self.temp2(out)
        out = torch.relu(out)
        out = self.dropout(out)
        return out + residual


class STGCN(nn.Module):
    """
    Paper-style STGCN for input [B, T, N, F] and output [B, N, 1].
    Internal format: [B, C, T, N].

    Chebyshev basis is precomputed once and stored as a buffer to avoid
    expensive recomputation (including eigenvalue calculation) every forward pass.
    """

    def __init__(
        self,
        num_nodes: int,
        in_channels: int = 1,
        spatial_hidden: int = 32,   # Kept for training-script compatibility.
        temporal_hidden: int = 64,  # Kept for training-script compatibility.
        horizon: int = 1,
        dropout: float = 0.3,
        hidden_channels: int | None = None,
        k_order: int = 3,
        blocks: int = 2,
        readout_mode: str = "last",
    ):
        super().__init__()
        if int(horizon) != 1:
            raise ValueError(f"Traffexplainer paper setting requires horizon=1, got {horizon}")
        if blocks != 2:
            raise ValueError(f"This implementation expects blocks=2, got {blocks}")
        if readout_mode not in {"last", "mean"}:
            raise ValueError(f"readout_mode must be one of ['last', 'mean'], got {readout_mode}")

        self.num_nodes = num_nodes
        self.horizon = 1
        self.k_order = k_order
        self.hidden_channels = int(hidden_channels if hidden_channels is not None else temporal_hidden)
        self.readout_mode = readout_mode

        self.block1 = STBlock(in_channels, self.hidden_channels, k_order=k_order, dropout=dropout)
        self.block2 = STBlock(self.hidden_channels, self.hidden_channels, k_order=k_order, dropout=dropout)
        self.final_temporal = TemporalConvLayer(self.hidden_channels, self.hidden_channels, kernel_size=3)
        self.readout = nn.Conv2d(self.hidden_channels, 1, kernel_size=(1, 1))

        # Chebyshev basis will be precomputed on first call to init_graph()
        # or lazily on first forward if not initialised.
        self._cheb_ready = False

    def init_graph(self, adj: torch.Tensor) -> None:
        """Precompute and register Chebyshev polynomial bases from adjacency."""
        cheb = cheb_polynomials(adj, self.k_order)  # [K, N, N]
        self.register_buffer("cheb_basis", cheb)
        self._cheb_ready = True

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        adj1: torch.Tensor | None = None,
        adj2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Input x: [B, T, N, F] -> [B, F, T, N]
        x = x.permute(0, 3, 1, 2)

        # If custom adjacencies are provided (mask training), compute on-the-fly.
        if adj1 is not None or adj2 is not None:
            a1 = adj if adj1 is None else adj1
            a2 = adj if adj2 is None else adj2
            cheb1 = cheb_polynomials(a1, self.k_order)
            cheb2 = cheb_polynomials(a2, self.k_order)
        else:
            # Use precomputed basis for standard forward pass.
            if not self._cheb_ready:
                self.init_graph(adj)
            cheb1 = self.cheb_basis
            cheb2 = self.cheb_basis

        x = self.block1(x, cheb1)
        x = self.block2(x, cheb2)
        x = self.final_temporal(x)

        if self.readout_mode == "last":
            # Paper-style one-step forecasting uses the latest temporal state.
            x = x[:, :, -1:, :]  # [B, C, 1, N]
        else:
            # Backward-compatible readout for checkpoints trained with mean pooling.
            x = torch.mean(x, dim=2, keepdim=True)  # [B, C, 1, N]
        x = self.readout(x)  # [B, 1, 1, N]
        return x.squeeze(2).permute(0, 2, 1)  # [B, N, 1]
