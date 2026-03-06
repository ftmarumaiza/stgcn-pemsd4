from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.dataloader import PeMSDataModule
from backend.masks import HierarchicalMasks, high_influence_edges_from_masks
from models.stgcn import STGCN
from utils.graph import load_adjacency

PAPER_HISTORY = 12
PAPER_HORIZON = 1


class PredictRequest(BaseModel):
    date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    time: str = Field(..., pattern=r"^\d{2}:\d{2}$")
    horizon_steps: int = Field(1, description="Forecast steps ahead: 1, 3, or 6")


class ExplainRequest(BaseModel):
    date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    time: str = Field(..., pattern=r"^\d{2}:\d{2}$")
    threshold: float | None = None


class PredictResponse(BaseModel):
    prediction: dict
    metrics: dict


class ExplainResponse(BaseModel):
    spatial_importance: list[dict]
    spatial_explanation: dict
    temporal_explanation: dict
    metrics: dict


class MetricsResponse(BaseModel):
    metrics: dict


@dataclass
class Settings:
    flow_path: str = os.getenv("TRAFFEX_FLOW_PATH", "data/pemsd4_flow.npy")
    adj_path: str = os.getenv("TRAFFEX_ADJ_PATH", "data/pemsd4_adj.npy")
    checkpoint_path: str = os.getenv("TRAFFEX_MODEL_CKPT", "artifacts/best_stgcn.pt")
    masks_dir: str = os.getenv("TRAFFEX_MASK_DIR", "artifacts")
    metrics_dir: str = os.getenv("TRAFFEX_METRICS_DIR", "artifacts")
    history: int = PAPER_HISTORY
    horizon: int = PAPER_HORIZON
    base_datetime: str = os.getenv("TRAFFEX_BASE_DATETIME", "2018-01-01 00:00")
    mape_min_actual: float = float(os.getenv("TRAFFEX_MAPE_MIN_ACTUAL", "10.0"))
    mape_eps: float = float(os.getenv("TRAFFEX_MAPE_EPS", "1e-5"))
    flow_score_max_flow: float = float(os.getenv("TRAFFEX_FLOW_SCORE_MAX", "500"))


def _first_existing_path(candidates: list[str]) -> str | None:
    seen = set()
    for path in candidates:
        if not path:
            continue
        norm_path = os.path.normpath(path)
        if norm_path in seen:
            continue
        seen.add(norm_path)
        if os.path.exists(norm_path):
            return norm_path
    return None


def _first_complete_masks_dir(candidates: list[str]) -> str | None:
    seen = set()
    for base_dir in candidates:
        if not base_dir:
            continue
        norm_dir = os.path.normpath(base_dir)
        if norm_dir in seen:
            continue
        seen.add(norm_dir)
        ms1_path = os.path.join(norm_dir, "MS1.npy")
        ms2_path = os.path.join(norm_dir, "MS2.npy")
        mt_path = os.path.join(norm_dir, "MT.npy")
        if os.path.exists(ms1_path) and os.path.exists(ms2_path) and os.path.exists(mt_path):
            return norm_dir
    return None


def _first_complete_metrics_dir(candidates: list[str]) -> str | None:
    seen = set()
    for base_dir in candidates:
        if not base_dir:
            continue
        norm_dir = os.path.normpath(base_dir)
        if norm_dir in seen:
            continue
        seen.add(norm_dir)
        locked_path = os.path.join(norm_dir, "locked_mae_rmse.json")
        mape_path = os.path.join(norm_dir, "mape_corrected.json")
        if os.path.exists(locked_path) and os.path.exists(mape_path):
            return norm_dir
    return None


def compute_metrics(
    pred_raw: torch.Tensor,
    true_raw: torch.Tensor,
    min_actual: float = 1.0,
    epsilon: float = 1e-5,
) -> tuple[float, float, float]:
    """
    Compute MAE/RMSE/MAPE on raw traffic flow values.
    Uses masking to avoid MAPE explosion on near-zero targets.
    """
    pred = pred_raw.to(dtype=torch.float32)
    true = true_raw.to(dtype=torch.float32)

    # Paper-aligned raw-scale evaluation, with stable MAPE masking.
    mask = torch.abs(true) > float(min_actual)
    if not torch.any(mask):
        return 0.0, 0.0, 0.0

    pred_m = pred[mask]
    true_m = true[mask]

    diff = pred_m - true_m
    mae = torch.mean(torch.abs(diff))
    rmse = torch.sqrt(torch.mean(diff * diff))
    mape = torch.mean(torch.abs(diff / (true_m + float(epsilon)))) * 100.0

    if not torch.isfinite(mape):
        return float(mae.item()), float(rmse.item()), 0.0
    return float(mae.item()), float(rmse.item()), float(mape.item())


@torch.no_grad()
def evaluate_full_test(
    model: STGCN,
    test_loader,
    adj: torch.Tensor,
    scaler,
    device: torch.device,
    num_nodes: int,
    min_actual: float = 1.0,
    epsilon: float = 1e-5,
) -> dict[str, float]:
    """
    Compute paper-style global metrics on full test split:
    - aggregate all timestamps and all nodes
    - inverse-transform to raw flow
    - compute MAE/RMSE/MAPE once (not per batch, not per request)
    """
    model.eval()
    all_preds = []
    all_trues = []

    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)

        pred_scaled = model(x, adj)  # [B, N, 1]
        pred_scaled = pred_scaled.detach().cpu().squeeze(-1)  # [B, N]
        true_scaled = y.detach().cpu().squeeze(-1)  # [B, N]

        pred_raw_np = scaler.inverse_transform(pred_scaled.numpy().reshape(-1, num_nodes))
        true_raw_np = scaler.inverse_transform(true_scaled.numpy().reshape(-1, num_nodes))

        all_preds.append(torch.tensor(pred_raw_np, dtype=torch.float32))
        all_trues.append(torch.tensor(true_raw_np, dtype=torch.float32))

    pred_raw = torch.cat(all_preds, dim=0)
    true_raw = torch.cat(all_trues, dim=0)
    mae, rmse, mape = compute_metrics(
        pred_raw=pred_raw,
        true_raw=true_raw,
        min_actual=min_actual,
        epsilon=epsilon,
    )
    return {"MAE": float(mae), "RMSE": float(rmse), "MAPE": float(mape)}


@torch.no_grad()
def evaluate_full_test_with_masks(
    model: STGCN,
    masks: HierarchicalMasks,
    test_loader,
    adj: torch.Tensor,
    scaler,
    device: torch.device,
    num_nodes: int,
    min_actual: float = 1.0,
    epsilon: float = 1e-5,
) -> dict[str, float]:
    """
    Compute paper-style global metrics on the full test split using the
    learned hierarchical masks during inference.
    """
    model.eval()
    masks.eval()
    all_preds = []
    all_trues = []

    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)

        x_masked, a1, a2 = masks(x, adj)
        pred_scaled = model(x_masked, adj=adj, adj1=a1, adj2=a2)
        pred_scaled = pred_scaled.detach().cpu().squeeze(-1)
        true_scaled = y.detach().cpu().squeeze(-1)

        pred_raw_np = scaler.inverse_transform(pred_scaled.numpy().reshape(-1, num_nodes))
        true_raw_np = scaler.inverse_transform(true_scaled.numpy().reshape(-1, num_nodes))

        all_preds.append(torch.tensor(pred_raw_np, dtype=torch.float32))
        all_trues.append(torch.tensor(true_raw_np, dtype=torch.float32))

    pred_raw = torch.cat(all_preds, dim=0)
    true_raw = torch.cat(all_trues, dim=0)
    mae, rmse, mape = compute_metrics(
        pred_raw=pred_raw,
        true_raw=true_raw,
        min_actual=min_actual,
        epsilon=epsilon,
    )
    return {"MAE": float(mae), "RMSE": float(rmse), "MAPE": float(mape)}


def compute_traffic_score(pred_raw: torch.Tensor, max_flow: float = 500.0) -> torch.Tensor:
    """
    UI-only conversion from raw flow to 0-100 score.
    Does not affect training or evaluation metrics.
    """
    denom = max(float(max_flow), 1e-8)
    score = (pred_raw.to(dtype=torch.float32) / denom) * 100.0
    return torch.clamp(score, 0.0, 100.0)


def normalize_scores(scores: torch.Tensor) -> torch.Tensor:
    """
    Min-max normalize to [0, 100]. Returns zeros when all values are equal.
    """
    if scores.numel() == 0:
        return scores
    s_min = torch.min(scores)
    s_max = torch.max(scores)
    if float((s_max - s_min).item()) <= 1e-12:
        return torch.zeros_like(scores)
    return (scores - s_min) / (s_max - s_min) * 100.0


class TrafficService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = self._resolve_checkpoint_path()
        self.masks_dir = self._resolve_masks_dir()
        self.metrics_dir = self._resolve_metrics_dir()
        self.checkpoint_meta = self._load_checkpoint_metadata()

        train_ratio = float(self.checkpoint_meta.get("train_ratio", 0.6))
        val_ratio = float(self.checkpoint_meta.get("val_ratio", 0.2))
        if not (0.0 < train_ratio < 1.0 and 0.0 < val_ratio < 1.0 and (train_ratio + val_ratio) < 1.0):
            raise RuntimeError(
                "Invalid train/val split values in checkpoint metadata: "
                f"train_ratio={train_ratio}, val_ratio={val_ratio}"
            )
        print(
            "[Startup] Using split from checkpoint metadata: "
            f"train={train_ratio:.2f}, val={val_ratio:.2f}, test={1.0 - train_ratio - val_ratio:.2f}"
        )

        self.data = PeMSDataModule(
            flow_path=settings.flow_path,
            history=settings.history,
            horizon=settings.horizon,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            base_datetime=settings.base_datetime,
        )
        self.loaders = self.data.dataloaders(batch_size=64)
        self.adj = load_adjacency(settings.adj_path, self.device)
        self.model: STGCN | None = None

        self._load_model_only()
        assert self.model is not None
        self.masks = self._load_masks_if_present()
        locked_metrics = self._load_locked_metrics_if_present()
        if locked_metrics is not None:
            self.metrics = locked_metrics
            self.metrics_source = "locked"
            self.metrics_source_label = "Locked Colab metrics"
        elif self.masks is None:
            self.metrics = evaluate_full_test(
                model=self.model,
                test_loader=self.loaders["test"],
                adj=self.adj,
                scaler=self.data.scaler,
                device=self.device,
                num_nodes=self.data.num_nodes,
                min_actual=self.settings.mape_min_actual,
                epsilon=self.settings.mape_eps,
            )
            self.metrics_source = "predictor"
            self.metrics_source_label = "Predictor full test split"
        else:
            self.metrics = evaluate_full_test_with_masks(
                model=self.model,
                masks=self.masks,
                test_loader=self.loaders["test"],
                adj=self.adj,
                scaler=self.data.scaler,
                device=self.device,
                num_nodes=self.data.num_nodes,
                min_actual=self.settings.mape_min_actual,
                epsilon=self.settings.mape_eps,
            )
            self.metrics_source = "masked"
            self.metrics_source_label = "Mask-evaluated full test split"
        print(
            f"===== {self.metrics_source_label} (PeMSD4) =====\n"
            f"MAE: {self.metrics['MAE']:.3f}\n"
            f"RMSE: {self.metrics['RMSE']:.3f}\n"
            f"MAPE: {self.metrics['MAPE']:.3f}%"
        )

    def _checkpoint_is_stgcn_compatible(self, checkpoint_path: str) -> tuple[bool, str]:
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
        except Exception as exc:  # pragma: no cover - startup guard
            return False, f"unable to load checkpoint ({exc})"

        model_state = ckpt.get("model_state")
        if not isinstance(model_state, dict):
            return False, "missing 'model_state' dictionary"

        keys = set(model_state.keys())
        required_keys = {
            "block1.temp1.conv_filter.weight",
            "block1.graph.theta",
            "block2.temp1.conv_filter.weight",
            "final_temporal.conv_filter.weight",
            "readout.weight",
        }
        missing = sorted(required_keys - keys)
        if missing:
            return False, "missing STGCN keys: " + ", ".join(missing)
        return True, ""

    def _resolve_checkpoint_path(self) -> str:
        candidates = [
            self.settings.checkpoint_path,
            "artifacts/best_stgcn.pt",
            "artifacts/backend_best_stgcn.pt",
            "artifacts/backend/best_stgcn.pt",
        ]
        checked_paths = []
        seen = set()
        for path in candidates:
            if not path:
                continue
            norm_path = os.path.normpath(path)
            if norm_path in seen:
                continue
            seen.add(norm_path)
            checked_paths.append(norm_path)

        found_any = False
        for candidate in checked_paths:
            if not os.path.exists(candidate):
                continue
            found_any = True
            compatible, reason = self._checkpoint_is_stgcn_compatible(candidate)
            if compatible:
                print(f"[Startup] Using model checkpoint: {candidate}")
                return candidate
            print(f"[Startup] Skipping incompatible checkpoint: {candidate} ({reason})")

        if found_any:
            raise RuntimeError(
                "Found checkpoint file(s), but none are compatible with the current STGCN model. "
                "Retrain via train_predictor.py and point TRAFFEX_MODEL_CKPT to the new file."
            )
        return os.path.normpath(self.settings.checkpoint_path)

    def _load_checkpoint_metadata(self) -> dict:
        if not os.path.exists(self.checkpoint_path):
            raise RuntimeError(
                "Checkpoint not found. Train predictor first and set TRAFFEX_MODEL_CKPT "
                "or place file at one of: "
                f"{os.path.normpath(self.settings.checkpoint_path)}, "
                "artifacts/best_stgcn.pt, artifacts/backend_best_stgcn.pt, artifacts/backend/best_stgcn.pt"
            )
        ckpt = torch.load(self.checkpoint_path, map_location="cpu")
        if "model_state" not in ckpt:
            raise RuntimeError(f"Checkpoint is missing 'model_state': {self.checkpoint_path}")
        return ckpt

    def _resolve_masks_dir(self) -> str:
        candidates = [
            self.settings.masks_dir,
            "artifacts",
            "artifacts/backend",
        ]
        resolved = _first_complete_masks_dir(candidates)
        if resolved is None:
            return os.path.normpath(self.settings.masks_dir)
        print(f"[Startup] Using masks directory: {resolved}")
        return resolved

    def _resolve_metrics_dir(self) -> str:
        candidates = [
            self.settings.metrics_dir,
            self.masks_dir,
            "artifacts",
            "artifacts/backend",
        ]
        resolved = _first_complete_metrics_dir(candidates)
        if resolved is None:
            return os.path.normpath(self.settings.metrics_dir)
        print(f"[Startup] Using locked metrics directory: {resolved}")
        return resolved

    def _load_model_only(self) -> None:
        ckpt = self.checkpoint_meta
        ckpt_history = int(ckpt.get("history", PAPER_HISTORY))
        if ckpt_history != PAPER_HISTORY:
            raise RuntimeError(
                f"Checkpoint history must be {PAPER_HISTORY} for paper setting, got {ckpt_history}."
            )
        ckpt_horizon = int(ckpt.get("horizon", PAPER_HORIZON))
        if ckpt_horizon != PAPER_HORIZON:
            raise RuntimeError(
                f"Checkpoint horizon must be {PAPER_HORIZON} for paper setting, got {ckpt_horizon}."
            )
        self.model = STGCN(
            num_nodes=self.data.num_nodes,
            in_channels=1,
            spatial_hidden=int(ckpt.get("spatial_hidden", 32)),
            temporal_hidden=int(ckpt.get("temporal_hidden", 64)),
            horizon=PAPER_HORIZON,
            dropout=float(ckpt.get("dropout", 0.3)),
            readout_mode=str(ckpt.get("readout_mode", "last")),
        ).to(self.device)
        self.model.init_graph(self.adj)
        missing_keys, unexpected_keys = self.model.load_state_dict(ckpt["model_state"], strict=False)
        disallowed_missing = [k for k in missing_keys if k != "cheb_basis"]
        if disallowed_missing or unexpected_keys:
            raise RuntimeError(
                "Checkpoint architecture mismatch. "
                f"Missing keys: {disallowed_missing}; unexpected keys: {unexpected_keys}"
            )
        if set(missing_keys) == {"cheb_basis"}:
            print("[Startup] Checkpoint has no precomputed cheb_basis; generated at runtime.")
        self.model.eval()

    def _load_masks_if_present(self) -> HierarchicalMasks | None:
        ms1_path = os.path.join(self.masks_dir, "MS1.npy")
        ms2_path = os.path.join(self.masks_dir, "MS2.npy")
        mt_path = os.path.join(self.masks_dir, "MT.npy")
        if not (os.path.exists(ms1_path) and os.path.exists(ms2_path) and os.path.exists(mt_path)):
            return None

        masks = HierarchicalMasks(num_nodes=self.data.num_nodes, history=self.settings.history).to(self.device)
        ms1 = np.load(ms1_path)
        ms2 = np.load(ms2_path)
        mt = np.load(mt_path)
        with torch.no_grad():
            masks.ms1.copy_(torch.from_numpy(ms1).to(self.device))
            masks.ms2.copy_(torch.from_numpy(ms2).to(self.device))
            masks.mt.copy_(torch.from_numpy(mt).to(self.device))
        masks.eval()
        print(
            "[Startup] Loaded hierarchical masks from "
            f"{ms1_path}, {ms2_path}, {mt_path}"
        )
        return masks

    def _load_locked_metrics_if_present(self) -> dict[str, float | str] | None:
        locked_path = os.path.join(self.metrics_dir, "locked_mae_rmse.json")
        mape_path = os.path.join(self.metrics_dir, "mape_corrected.json")
        if not (os.path.exists(locked_path) and os.path.exists(mape_path)):
            return None

        with open(locked_path, "r", encoding="utf-8") as f:
            locked_payload = json.load(f)
        with open(mape_path, "r", encoding="utf-8") as f:
            mape_payload = json.load(f)

        metric_group = "interpreted" if self.masks is not None else "original"
        mae_rmse = (
            locked_payload.get(metric_group)
            or locked_payload.get("interpreted")
            or locked_payload.get("original")
        )
        if not isinstance(mae_rmse, dict):
            raise RuntimeError(
                "locked_mae_rmse.json must contain an 'interpreted' or 'original' object with MAE/RMSE."
            )

        if metric_group == "interpreted":
            mape_value = mape_payload.get("interpreted_mape_percent")
            if mape_value is None:
                mape_value = mape_payload.get("original_mape_percent")
        else:
            mape_value = mape_payload.get("original_mape_percent")
            if mape_value is None:
                mape_value = mape_payload.get("interpreted_mape_percent")
        if mape_value is None:
            raise RuntimeError(
                "mape_corrected.json must contain 'interpreted_mape_percent' or 'original_mape_percent'."
            )

        note = mape_payload.get("note")
        if note:
            print(f"[Startup] Locked metric note: {note}")
        print(
            "[Startup] Loaded locked metrics from "
            f"{locked_path} and {mape_path} using '{metric_group}' values"
        )
        return {
            "MAE": float(mae_rmse["MAE"]),
            "RMSE": float(mae_rmse["RMSE"]),
            "MAPE": float(mape_value),
            "note": str(note) if note else "",
        }

    def metrics_payload(self) -> dict:
        payload = {
            "MAE": float(self.metrics["MAE"]),
            "RMSE": float(self.metrics["RMSE"]),
            "MAPE": float(self.metrics["MAPE"]),
            "source": self.metrics_source,
            "source_label": self.metrics_source_label,
        }
        note = self.metrics.get("note")
        if note:
            payload["note"] = str(note)
        return payload

    def _recursive_predict_raw(self, window_raw: np.ndarray, steps: int) -> np.ndarray:
        if self.model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        x_norm = self.data.scaler.transform(window_raw).astype(np.float32)  # [12, N]
        pred_norm = None
        with torch.no_grad():
            for _ in range(steps):
                x_t = torch.from_numpy(x_norm).unsqueeze(0).unsqueeze(-1).to(self.device)
                pred_norm = self.model(x_t, self.adj).cpu().numpy()[0, :, 0]  # [N]
                x_norm = np.concatenate([x_norm[1:], pred_norm[None, :]], axis=0)
        assert pred_norm is not None
        return self.data.scaler.inverse_transform(pred_norm[None, :])[0]

    def predict(self, date: str, time: str, horizon_steps: int = 1) -> PredictResponse:
        if horizon_steps not in (1, 3, 6):
            raise HTTPException(status_code=400, detail="horizon_steps must be one of: 1, 3, 6")
        try:
            idx = self.data.datetime_to_target_index(date, time)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        target_idx = idx + horizon_steps - 1
        if target_idx >= self.data.total_time:
            raise HTTPException(status_code=400, detail="Requested horizon exceeds dataset range.")

        window_raw = self.data.window_for_target_index(idx)
        pred = self._recursive_predict_raw(window_raw, steps=horizon_steps)
        risk = self.data.get_risk_summary(pred)
        avg_flow = float(np.mean(pred))
        flow_score_0_100 = float(
            torch.mean(
                compute_traffic_score(
                    pred_raw=torch.from_numpy(pred),
                    max_flow=self.settings.flow_score_max_flow,
                )
            ).item()
        )
        predicted_dt = self.data.base_datetime + (target_idx * self.data.interval)
        minutes_ahead = horizon_steps * int(self.data.interval.total_seconds() // 60)
        input_var = float(np.var(window_raw))
        pred_var = float(np.var(pred))
        print(
            f"[PredictDebug] date={date} time={time} idx={idx} horizon_steps={horizon_steps} "
            f"input_var={input_var:.6f} pred_var={pred_var:.6f}"
        )

        return PredictResponse(
            prediction={
                "values": [float(v) for v in pred.tolist()],
                "risk_summary": risk,
                "horizon_steps": horizon_steps,
                "minutes_ahead": minutes_ahead,
                "avg_flow": avg_flow,
                "flow_score_0_100": flow_score_0_100,
                "forecast_note": f"Predicting traffic for next {minutes_ahead} minutes",
                "predicted_timestamp": predicted_dt.strftime("%Y-%m-%d %H:%M"),
            },
            metrics=self.metrics_payload(),
        )

    def explain(self, date: str, time: str, threshold: float | None = None) -> ExplainResponse:
        if self.masks is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Mask files not found. Train masks after base model and place "
                    f"MS1/MS2/MT under: {self.settings.masks_dir}"
                ),
            )
        try:
            idx = self.data.datetime_to_target_index(date, time)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        window_raw = self.data.window_for_target_index(idx)  # [T, N]
        x_norm = self.data.scaler.transform(window_raw)
        x = torch.from_numpy(x_norm.astype(np.float32)).unsqueeze(0).unsqueeze(-1).to(self.device)
        node_activity = np.mean(np.abs(window_raw), axis=0)  # [N]
        if self.model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")

        ms1 = self.masks.sym_ms1().detach().cpu().numpy()
        ms2 = self.masks.sym_ms2().detach().cpu().numpy()

        adj_np = self.adj.detach().cpu().numpy()
        base_edge_score = np.abs(ms1 * ms2) * adj_np
        node_factor = 0.5 * (node_activity[:, None] + node_activity[None, :])
        weighted_edge_score = base_edge_score * node_factor

        high_edges, threshold_used = high_influence_edges_from_masks(
            adj=adj_np,
            ms1=weighted_edge_score,
            ms2=np.ones_like(weighted_edge_score),
            threshold=threshold,
        )

        raw_scores_t = torch.tensor(
            [float(edge["importance"]) for edge in high_edges], dtype=torch.float32
        )
        norm_scores_t = normalize_scores(raw_scores_t)
        spatial_importance = []
        top_edges_ui = []
        for i, edge in enumerate(high_edges):
            raw_score = float(raw_scores_t[i].item())
            norm_score = float(norm_scores_t[i].item())
            from_node = int(edge["from"])
            to_node = int(edge["to"])
            spatial_importance.append(
                {
                    "edge": f"{from_node} -> {to_node}",
                    "raw_score": raw_score,
                    "normalized_score": norm_score,
                }
            )
            # Keep existing shape for frontend while exposing raw/normalized values.
            top_edges_ui.append(
                {
                    "from": from_node,
                    "to": to_node,
                    "importance": norm_score,
                    "raw_score": raw_score,
                    "normalized_score": norm_score,
                }
            )

        # Time-conditioned perturbation importance:
        # for each lag, ablate that timestep and measure prediction shift.
        with torch.no_grad():
            x_masked, a1, a2 = self.masks(x, self.adj)
            pred_base = self.model(x_masked, adj=self.adj, adj1=a1, adj2=a2)

            temporal_scores = []
            for t in range(self.settings.history):
                x_pert = x.clone()
                # 0 in normalized space corresponds to train-mean flow.
                x_pert[:, t, :, :] = 0.0
                x_pert_masked, a1p, a2p = self.masks(x_pert, self.adj)
                pred_pert = self.model(x_pert_masked, adj=self.adj, adj1=a1p, adj2=a2p)
                delta = torch.mean(torch.abs(pred_pert - pred_base)).item()
                temporal_scores.append(delta)

        temporal_importance = np.array(temporal_scores, dtype=np.float32)
        temporal_sum = float(np.sum(temporal_importance))
        if temporal_sum <= 1e-12:
            # Fallback when perturbation deltas collapse to zero:
            # derive relative timestep influence from learned temporal mask.
            mt_scale = torch.tanh(self.masks.mt.detach()) + 1.0  # [T, N]
            mt_importance = torch.mean(torch.abs(mt_scale), dim=1).cpu().numpy().astype(np.float32)
            mt_sum = float(np.sum(mt_importance))
            if mt_sum <= 1e-12:
                temporal_importance = np.full(self.settings.history, 1.0 / self.settings.history, dtype=np.float32)
            else:
                temporal_importance = mt_importance / mt_sum
        else:
            temporal_importance = temporal_importance / temporal_sum
        max_idx = int(np.argmax(temporal_importance))
        most_important_window = f"t-{self.settings.history - max_idx}"
        print(
            f"[ExplainDebug] date={date} time={time} idx={idx} "
            f"window_var={float(np.var(window_raw)):.6f} temp_var={float(np.var(temporal_importance)):.6f}"
        )

        return ExplainResponse(
            spatial_importance=spatial_importance,
            spatial_explanation={
                "top_edges": top_edges_ui,
                "threshold_used": float(threshold_used),
                "label": "High-Influence Road Pairs",
            },
            temporal_explanation={
                "importance_vector": [float(v) for v in temporal_importance.tolist()],
                "most_important_window": most_important_window,
            },
            metrics=self.metrics_payload(),
        )

    def nodes(self) -> list[dict]:
        return self.data.node_positions()


settings = Settings()
service: TrafficService | None = None
app = FastAPI(title="Traffexplainer API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    global service
    service = TrafficService(settings)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/nodes")
def nodes():
    if service is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return {"nodes": service.nodes()}


@app.get("/metrics", response_model=MetricsResponse)
def metrics():
    if service is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return MetricsResponse(metrics=service.metrics_payload())


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    if service is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return service.predict(date=payload.date, time=payload.time, horizon_steps=payload.horizon_steps)


@app.post("/explain", response_model=ExplainResponse)
def explain(payload: ExplainRequest):
    if service is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return service.explain(date=payload.date, time=payload.time, threshold=payload.threshold)
