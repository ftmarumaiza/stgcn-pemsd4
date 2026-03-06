from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PAPER_HISTORY = 12
PAPER_HORIZON = 1


@dataclass
class StandardScaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


class WindowDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]


class PeMSDataModule:
    def __init__(
        self,
        flow_path: str,
        history: int = PAPER_HISTORY,
        horizon: int = PAPER_HORIZON,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        base_datetime: str = "2018-01-01 00:00",
        interval_minutes: int = 5,
    ):
        self.flow_path = flow_path
        if history != PAPER_HISTORY or horizon != PAPER_HORIZON:
            raise ValueError(
                f"Traffexplainer paper setting requires history={PAPER_HISTORY}, horizon={PAPER_HORIZON}. "
                f"Got history={history}, horizon={horizon}."
            )
        self.history = history
        self.horizon = horizon
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.base_datetime = datetime.strptime(base_datetime, "%Y-%m-%d %H:%M")
        self.interval = timedelta(minutes=interval_minutes)

        self.flow_raw = np.load(self.flow_path).astype(np.float32)
        if self.flow_raw.ndim != 2:
            raise ValueError(f"Expected flow matrix [time, nodes], got {self.flow_raw.shape}")

        self.total_time, self.num_nodes = self.flow_raw.shape
        self.train_end = int(self.total_time * self.train_ratio)
        self.val_end = int(self.total_time * (self.train_ratio + self.val_ratio))

        mean = self.flow_raw[: self.train_end].mean(axis=0, keepdims=True)
        std = self.flow_raw[: self.train_end].std(axis=0, keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        self.scaler = StandardScaler(mean=mean, std=std)
        self.flow_norm = self.scaler.transform(self.flow_raw)

        self.risk_thresholds = np.quantile(self.flow_raw[: self.train_end], [0.33, 0.66])

    def _windows_from_segment(self, segment: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for t in range(self.history, segment.shape[0]):
            x = segment[t - self.history : t]
            y = segment[t]
            xs.append(x[..., None])
            ys.append(y[..., None])
        if not xs:
            raise ValueError(
                f"Insufficient split length {segment.shape[0]} for history={self.history}, "
                f"horizon={self.horizon}"
            )
        return np.stack(xs, axis=0), np.stack(ys, axis=0)

    def build_windows(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        segments = {
            "train": self.flow_norm[: self.train_end],
            "val": self.flow_norm[self.train_end : self.val_end],
            "test": self.flow_norm[self.val_end :],
        }
        return {split: self._windows_from_segment(seg) for split, seg in segments.items()}

    def dataloaders(self, batch_size: int = 64, num_workers: int = 0) -> dict[str, DataLoader]:
        windows = self.build_windows()
        loaders = {}
        for split, (x, y) in windows.items():
            loaders[split] = DataLoader(
                WindowDataset(x, y),
                batch_size=batch_size,
                shuffle=(split == "train"),
                num_workers=num_workers,
                drop_last=False,
            )
        return loaders

    def datetime_to_target_index(self, date_str: str, time_str: str) -> int:
        target_dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
        delta = target_dt - self.base_datetime
        if delta.total_seconds() < 0:
            raise ValueError("Requested datetime is before the dataset start.")
        if delta.total_seconds() % (self.interval.total_seconds()) != 0:
            raise ValueError("Time must align to 5-minute intervals.")

        idx = int(delta.total_seconds() // self.interval.total_seconds())
        if idx < self.history:
            raise ValueError(f"Need at least {self.history} past timesteps before target time.")
        if idx + self.horizon > self.total_time:
            raise ValueError("Requested datetime exceeds dataset range.")
        return idx

    def window_for_target_index(self, idx: int) -> np.ndarray:
        return self.flow_raw[idx - self.history : idx].copy()

    def get_risk_summary(self, predicted_values: np.ndarray) -> str:
        avg_flow = float(np.mean(predicted_values))
        low, high = self.risk_thresholds
        if avg_flow < low:
            return "Low"
        if avg_flow < high:
            return "Moderate"
        return "High"

    def node_positions(self) -> list[dict]:
        # Synthetic coordinates inside California bounds. Replace with real sensor geo if available.
        lats = np.linspace(32.5, 41.8, 20)
        lons = np.linspace(-124.2, -114.0, 20)
        coords = []
        for i in range(self.num_nodes):
            r = i // 20
            c = i % 20
            lat = lats[r % len(lats)] + 0.05 * np.sin(i / 7.0)
            lon = lons[c % len(lons)] + 0.05 * np.cos(i / 9.0)
            coords.append({"id": i, "lat": float(lat), "lon": float(lon)})
        return coords
