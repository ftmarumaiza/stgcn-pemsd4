from __future__ import annotations

import argparse
import os

import torch

from data.traffic_dataset import PAPER_HISTORY, PAPER_HORIZON, build_dataloaders
from models.stgcn import STGCN
from utils.engine import evaluate_loader
from utils.graph import load_adjacency


REQUIRED_STGCN_KEYS = {
    "block1.temp1.conv_filter.weight",
    "block1.graph.theta",
    "block2.temp1.conv_filter.weight",
    "final_temporal.conv_filter.weight",
    "readout.weight",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate STGCN checkpoint on PeMSD4 with raw-scale metrics")
    p.add_argument("--checkpoint", type=str, default="artifacts/best_stgcn.pt")
    p.add_argument("--data_path", type=str, default="data/pemsd4_flow.npy")
    p.add_argument("--adj_path", type=str, default="data/pemsd4_adj.npy")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--mape_min_actual", type=float, default=10.0)
    p.add_argument("--mape_eps", type=float, default=1e-5)
    p.add_argument("--output", type=str, default="artifacts/eval_output.txt")
    return p.parse_args()


def validate_checkpoint(ckpt: dict, checkpoint_path: str) -> None:
    model_state = ckpt.get("model_state")
    if not isinstance(model_state, dict):
        raise RuntimeError(f"Checkpoint has no valid 'model_state': {checkpoint_path}")

    keys = set(model_state.keys())
    missing = sorted(REQUIRED_STGCN_KEYS - keys)
    if missing:
        raise RuntimeError(
            "Checkpoint is not compatible with current models/stgcn.py implementation. "
            f"Missing keys: {missing}"
        )


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    validate_checkpoint(ckpt, args.checkpoint)

    history = int(ckpt.get("history", PAPER_HISTORY))
    horizon = int(ckpt.get("horizon", PAPER_HORIZON))
    if history != PAPER_HISTORY or horizon != PAPER_HORIZON:
        raise RuntimeError(
            f"Expected history={PAPER_HISTORY}, horizon={PAPER_HORIZON}; got history={history}, horizon={horizon}"
        )

    train_ratio = float(ckpt.get("train_ratio", 0.6))
    val_ratio = float(ckpt.get("val_ratio", 0.2))
    strict_split = bool(ckpt.get("strict_split", True))
    if not (0.0 < train_ratio < 1.0 and 0.0 < val_ratio < 1.0 and (train_ratio + val_ratio) < 1.0):
        raise RuntimeError(
            f"Invalid split values in checkpoint: train_ratio={train_ratio}, val_ratio={val_ratio}"
        )

    loaders, scaler = build_dataloaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        history=history,
        horizon=horizon,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        strict_split=strict_split,
    )
    sample_x, _ = next(iter(loaders["train"]))
    num_nodes = int(sample_x.shape[2])

    model = STGCN(
        num_nodes=num_nodes,
        in_channels=1,
        spatial_hidden=int(ckpt.get("spatial_hidden", 32)),
        temporal_hidden=int(ckpt.get("temporal_hidden", 64)),
        horizon=horizon,
        dropout=float(ckpt.get("dropout", 0.3)),
        readout_mode=str(ckpt.get("readout_mode", "last")),
    ).to(device)

    adj = load_adjacency(args.adj_path, device)
    model.init_graph(adj)
    missing_keys, unexpected_keys = model.load_state_dict(ckpt["model_state"], strict=False)
    disallowed_missing = [k for k in missing_keys if k != "cheb_basis"]
    if disallowed_missing or unexpected_keys:
        raise RuntimeError(
            "Checkpoint architecture mismatch after load_state_dict. "
            f"Missing keys: {disallowed_missing}; unexpected keys: {unexpected_keys}"
        )

    test_metrics = evaluate_loader(
        model=model,
        loader=loaders["test"],
        adj=adj,
        device=device,
        scaler=scaler,
        mape_min_actual=args.mape_min_actual,
        mape_eps=args.mape_eps,
    )
    val_metrics = evaluate_loader(
        model=model,
        loader=loaders["val"],
        adj=adj,
        device=device,
        scaler=scaler,
        mape_min_actual=args.mape_min_actual,
        mape_eps=args.mape_eps,
    )

    lines = []
    lines.append("=== Checkpoint Metadata ===")
    lines.append(f"  checkpoint: {os.path.normpath(args.checkpoint)}")
    lines.append(f"  history: {history}")
    lines.append(f"  horizon: {horizon}")
    lines.append(f"  readout_mode: {ckpt.get('readout_mode', 'last')}")
    lines.append(f"  dropout: {ckpt.get('dropout', 0.3)}")
    lines.append("")
    lines.append(
        "Split: train={:.2f}, val={:.2f}, test={:.2f}, strict_split={}".format(
            train_ratio, val_ratio, 1.0 - train_ratio - val_ratio, strict_split
        )
    )
    lines.append("")
    lines.append("=== Test Metrics (raw-scale) ===")
    lines.append("  MAE:  {:.4f}".format(test_metrics["MAE"]))
    lines.append("  RMSE: {:.4f}".format(test_metrics["RMSE"]))
    lines.append("  MAPE: {:.2f}%".format(test_metrics["MAPE"]))
    lines.append("")
    lines.append("=== Val Metrics (raw-scale) ===")
    lines.append("  MAE:  {:.4f}".format(val_metrics["MAE"]))
    lines.append("  RMSE: {:.4f}".format(val_metrics["RMSE"]))
    lines.append("  MAPE: {:.2f}%".format(val_metrics["MAPE"]))
    lines.append("")
    lines.append("Note: No fixed paper target is printed; results depend on data preprocessing and graph construction.")

    result = "\n".join(lines)
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(result)
    print(result)


if __name__ == "__main__":
    main()
