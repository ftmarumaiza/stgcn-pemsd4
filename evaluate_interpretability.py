from __future__ import annotations

import argparse

import numpy as np
import torch

from data.traffic_dataset import build_dataloaders
from models.stgcn import STGCN
from utils.engine import evaluate_loader
from utils.graph import load_adjacency

PAPER_HISTORY = 12
PAPER_HORIZON = 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate fidelity and sparsity")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--adj_path", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--ms1_path", type=str, required=True)
    p.add_argument("--ms2_path", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--history", type=int, default=12)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--topk_ratio", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--mape_min_actual", type=float, default=10.0)
    p.add_argument("--mape_eps", type=float, default=1e-5)
    return p.parse_args()


def topk_edges_from_masks(adj: np.ndarray, ms1: np.ndarray, ms2: np.ndarray, topk_ratio: float):
    score = np.abs(ms1 * ms2) * adj
    upper = np.triu(adj > 0, k=1)
    edge_idx = np.argwhere(upper)

    if edge_idx.shape[0] == 0:
        return edge_idx, 0

    edge_scores = np.array([score[i, j] for i, j in edge_idx])
    k = max(1, int(len(edge_scores) * topk_ratio))
    topk = np.argsort(edge_scores)[-k:]
    return edge_idx[topk], k


def main() -> None:
    args = parse_args()
    if args.history != PAPER_HISTORY or args.horizon != PAPER_HORIZON:
        raise ValueError(
            f"Traffexplainer paper setting requires --history {PAPER_HISTORY} --horizon {PAPER_HORIZON}. "
            f"Got history={args.history}, horizon={args.horizon}."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders, scaler = build_dataloaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        history=args.history,
        horizon=args.horizon,
        num_workers=args.num_workers,
    )

    sample_x, _ = next(iter(loaders["train"]))
    num_nodes = sample_x.shape[2]

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = STGCN(
        num_nodes=num_nodes,
        in_channels=1,
        spatial_hidden=ckpt.get("spatial_hidden", 32),
        temporal_hidden=ckpt.get("temporal_hidden", 64),
        horizon=ckpt.get("horizon", args.horizon),
        dropout=ckpt.get("dropout", 0.1),
        readout_mode=ckpt.get("readout_mode", "mean"),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    adj_t = load_adjacency(args.adj_path, device)
    adj_np = adj_t.detach().cpu().numpy()

    ms1 = np.load(args.ms1_path)
    ms2 = np.load(args.ms2_path)

    base_metrics = evaluate_loader(
        model,
        loaders["test"],
        adj_t,
        device,
        scaler,
        mape_min_actual=args.mape_min_actual,
        mape_eps=args.mape_eps,
    )

    topk_edges, selected = topk_edges_from_masks(adj_np, ms1, ms2, args.topk_ratio)

    perturbed_adj = adj_np.copy()
    for i, j in topk_edges:
        perturbed_adj[i, j] = 0.0
        perturbed_adj[j, i] = 0.0

    perturbed_adj_t = torch.from_numpy(perturbed_adj.astype(np.float32)).to(device)
    perturbed_metrics = evaluate_loader(
        model,
        loaders["test"],
        perturbed_adj_t,
        device,
        scaler,
        mape_min_actual=args.mape_min_actual,
        mape_eps=args.mape_eps,
    )

    total_edges = int(np.argwhere(np.triu(adj_np > 0, k=1)).shape[0])
    sparsity = 1.0 - (selected / max(total_edges, 1))
    fidelity = perturbed_metrics["RMSE"] - base_metrics["RMSE"]

    print("Prediction Performance (Base Test)")
    print(
        "MAE: {MAE:.4f}, RMSE: {RMSE:.4f}, MAPE: {MAPE:.2f}%".format(
            **base_metrics
        )
    )
    print(
        f"(MAPE computed with |true| >= {args.mape_min_actual} and epsilon={args.mape_eps})"
    )
    print("\nInterpretability")
    print(f"Fidelity (RMSE drop): {fidelity:.4f}")
    print(f"Sparsity: {sparsity:.4f}")


if __name__ == "__main__":
    main()
