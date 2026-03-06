from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize hierarchical masks")
    p.add_argument("--ms1_path", type=str, required=True)
    p.add_argument("--ms2_path", type=str, required=True)
    p.add_argument("--mt_path", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="artifacts/figures")
    return p.parse_args()


def save_heatmap(arr: np.ndarray, title: str, save_path: str, cmap: str = "viridis") -> None:
    plt.figure(figsize=(7, 6))
    plt.imshow(arr, aspect="auto", cmap=cmap)
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    ms1 = np.load(args.ms1_path)
    ms2 = np.load(args.ms2_path)
    mt = np.load(args.mt_path)

    ms12 = ms1 * ms2
    temporal_importance = mt.mean(axis=1)

    save_heatmap(ms1, "MS1 Heatmap", os.path.join(args.save_dir, "ms1_heatmap.png"))
    save_heatmap(ms2, "MS2 Heatmap", os.path.join(args.save_dir, "ms2_heatmap.png"))
    save_heatmap(ms12, "MS1 x MS2 Heatmap", os.path.join(args.save_dir, "ms1_ms2_heatmap.png"))

    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(len(temporal_importance)), temporal_importance, marker="o")
    plt.title("Temporal Mask Importance (mean over nodes)")
    plt.xlabel("Time Step")
    plt.ylabel("Importance")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "temporal_importance.png"), dpi=200)
    plt.close()

    print(f"Saved visualizations to: {args.save_dir}")


if __name__ == "__main__":
    main()
