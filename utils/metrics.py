from __future__ import annotations

import numpy as np
import torch


def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target))


def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((pred - target) ** 2))


def mape(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-5,
    min_actual: float = 10.0,
) -> torch.Tensor:
    mask = torch.abs(target) > float(min_actual)
    if not torch.any(mask):
        return torch.tensor(0.0, dtype=pred.dtype, device=pred.device)
    pred_m = pred[mask]
    target_m = target[mask]
    mape_v = torch.mean(torch.abs((pred_m - target_m) / (target_m + float(eps)))) * 100.0
    if not torch.isfinite(mape_v):
        return torch.tensor(0.0, dtype=pred.dtype, device=pred.device)
    return mape_v


def compute_metrics_np(
    pred: np.ndarray,
    target: np.ndarray,
    mape_min_actual: float = 10.0,
    mape_eps: float = 1e-5,
) -> dict:
    diff = pred - target
    mae_v = np.mean(np.abs(diff))
    rmse_v = np.sqrt(np.mean(diff**2))

    mask = np.abs(target) > float(mape_min_actual)
    if np.any(mask):
        pred_m = pred[mask]
        target_m = target[mask]
        mape_v = np.mean(np.abs((pred_m - target_m) / (target_m + float(mape_eps)))) * 100.0
        if not np.isfinite(mape_v):
            mape_v = 0.0
    else:
        mape_v = 0.0
    return {"MAE": float(mae_v), "RMSE": float(rmse_v), "MAPE": float(mape_v)}
