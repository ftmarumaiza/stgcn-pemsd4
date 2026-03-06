from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.traffic_dataset import StandardScaler
def _compute_metrics_raw(
    pred_raw: np.ndarray,
    true_raw: np.ndarray,
    mape_min_actual: float = 10.0,
    mape_eps: float = 1e-5,
) -> Dict[str, float]:
    diff = pred_raw - true_raw
    mae_v = float(np.mean(np.abs(diff)))
    rmse_v = float(np.sqrt(np.mean(diff**2)))

    # Mask tiny true values to prevent MAPE explosion on near-zero traffic flow.
    mask = np.abs(true_raw) >= float(mape_min_actual)
    if np.any(mask):
        pred_m = pred_raw[mask]
        true_m = true_raw[mask]
        mape_v = float(np.mean(np.abs((pred_m - true_m) / (true_m + float(mape_eps)))) * 100.0)
        if not np.isfinite(mape_v):
            mape_v = 0.0
    else:
        mape_v = 0.0
    return {"MAE": mae_v, "RMSE": rmse_v, "MAPE": mape_v}


def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    adj: torch.Tensor,
    device: torch.device,
    scaler: StandardScaler,
    mape_min_actual: float = 10.0,
    mape_eps: float = 1e-5,
) -> Dict[str, float]:
    """
    Evaluate on full split with paper-aligned raw-scale metrics:
    1) Collect normalized predictions/targets across all batches.
    2) Inverse-transform both to original traffic flow scale.
    3) Compute MAE/RMSE/Masked-MAPE once on concatenated arrays.
    """
    model.eval()
    all_pred_raw = []
    all_true_raw = []
    num_nodes = None

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            out = model(x, adj)  # normalized output, shape [B, N, 1]

            pred_norm = out.detach().cpu().numpy().squeeze(-1)  # [B, N]
            true_norm = y.detach().cpu().numpy().squeeze(-1)  # [B, N]

            if num_nodes is None:
                num_nodes = int(pred_norm.shape[-1])

            pred_raw = scaler.inverse_transform(pred_norm.reshape(-1, num_nodes))
            true_raw = scaler.inverse_transform(true_norm.reshape(-1, num_nodes))
            all_pred_raw.append(pred_raw)
            all_true_raw.append(true_raw)

    pred_raw_all = np.concatenate(all_pred_raw, axis=0)
    true_raw_all = np.concatenate(all_true_raw, axis=0)
    return _compute_metrics_raw(
        pred_raw=pred_raw_all,
        true_raw=true_raw_all,
        mape_min_actual=mape_min_actual,
        mape_eps=mape_eps,
    )
