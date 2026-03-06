from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data.traffic_dataset import build_dataloaders
from models.stgcn import STGCN
from utils.engine import evaluate_loader
from utils.graph import load_adjacency

PAPER_HISTORY = 12
PAPER_HORIZON = 1

# ---------- Paper-aligned defaults for PeMSD4 ----------
PAPER_DEFAULTS = dict(
    batch_size=64,
    epochs=100,
    lr=3e-3,                   # higher LR for faster convergence (paper-tuned)
    spatial_hidden=32,
    temporal_hidden=64,
    dropout=0.3,
    readout_mode="last",       # paper uses last-step readout
    weight_decay=1e-4,         # stronger regularisation
    grad_clip=5.0,
    seed=42,
    train_ratio=0.6,           # 60 / 20 / 20 (common PeMSD4 protocol)
    val_ratio=0.2,
    lr_factor=0.5,
    lr_patience=5,
    lr_min=1e-6,
    early_stop_patience=10,    # stop if val loss doesn't improve for N epochs
    mape_min_actual=10.0,
    mape_eps=1e-5,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _naive_last_step_mse(loader, device: torch.device) -> float:
    total, count = 0.0, 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = x[:, -1, :, :]  # naive persistence baseline, shape [B, N, 1]
        mse = torch.mean((pred - y) ** 2).item()
        total += mse * x.size(0)
        count += x.size(0)
    return total / max(count, 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train STGCN predictor (paper protocol)")

    # Paths
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--adj_path", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="artifacts")

    # Data / windowing
    p.add_argument("--batch_size", type=int, default=PAPER_DEFAULTS["batch_size"])
    p.add_argument("--history", type=int, default=PAPER_HISTORY)
    p.add_argument("--horizon", type=int, default=PAPER_HORIZON)
    p.add_argument("--train_ratio", type=float, default=PAPER_DEFAULTS["train_ratio"],
                    help="Fraction of data for training (paper: 0.6)")
    p.add_argument("--val_ratio", type=float, default=PAPER_DEFAULTS["val_ratio"],
                    help="Fraction of data for validation (paper: 0.2)")

    # Model architecture
    p.add_argument("--spatial_hidden", type=int, default=PAPER_DEFAULTS["spatial_hidden"])
    p.add_argument("--temporal_hidden", type=int, default=PAPER_DEFAULTS["temporal_hidden"])
    p.add_argument("--dropout", type=float, default=PAPER_DEFAULTS["dropout"])
    p.add_argument("--readout_mode", type=str, default=PAPER_DEFAULTS["readout_mode"],
                    choices=["last", "mean"],
                    help="Temporal readout: 'last' (paper) or 'mean'")

    # Optimisation
    p.add_argument("--epochs", type=int, default=PAPER_DEFAULTS["epochs"])
    p.add_argument("--lr", type=float, default=PAPER_DEFAULTS["lr"])
    p.add_argument("--weight_decay", type=float, default=PAPER_DEFAULTS["weight_decay"])
    p.add_argument("--grad_clip", type=float, default=PAPER_DEFAULTS["grad_clip"])
    p.add_argument("--lr_factor", type=float, default=PAPER_DEFAULTS["lr_factor"])
    p.add_argument("--lr_patience", type=int, default=PAPER_DEFAULTS["lr_patience"])
    p.add_argument("--lr_min", type=float, default=PAPER_DEFAULTS["lr_min"])
    p.add_argument("--early_stop_patience", type=int, default=PAPER_DEFAULTS["early_stop_patience"],
                    help="Stop training if val loss doesn't improve for N epochs (0 = disabled)")

    # Metric params
    p.add_argument("--mape_min_actual", type=float, default=PAPER_DEFAULTS["mape_min_actual"])
    p.add_argument("--mape_eps", type=float, default=PAPER_DEFAULTS["mape_eps"])

    # Misc
    p.add_argument("--seed", type=int, default=PAPER_DEFAULTS["seed"])
    p.add_argument("--num_workers", type=int, default=0)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ---- Enforce paper windowing ----
    if args.history != PAPER_HISTORY or args.horizon != PAPER_HORIZON:
        raise ValueError(
            f"Traffexplainer paper setting requires --history {PAPER_HISTORY} --horizon {PAPER_HORIZON}. "
            f"Got history={args.history}, horizon={args.horizon}."
        )

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- Data ----------
    loaders, scaler = build_dataloaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        history=args.history,
        horizon=args.horizon,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
        strict_split=True,
    )

    sample_x, _ = next(iter(loaders["train"]))
    if sample_x.ndim != 4:
        raise ValueError(f"Expected x shape [B,T,N,C], got {tuple(sample_x.shape)}")
    if sample_x.shape[1] != args.history:
        raise ValueError(f"Expected history={args.history}, got {sample_x.shape[1]}")
    num_nodes = sample_x.shape[2]

    print(f"\n===== Training Config =====")
    print(f"  Device        : {device}")
    print(f"  Nodes         : {num_nodes}")
    print(f"  History       : {args.history}")
    print(f"  Horizon       : {args.horizon}")
    print(f"  Split         : train={args.train_ratio:.0%} / val={args.val_ratio:.0%} "
          f"/ test={1 - args.train_ratio - args.val_ratio:.0%}")
    print(f"  Readout       : {args.readout_mode}")
    print(f"  Dropout       : {args.dropout}")
    print(f"  LR            : {args.lr}  (factor={args.lr_factor}, patience={args.lr_patience})")
    print(f"  Early stop    : {args.early_stop_patience} epochs")
    print(f"  Epochs (max)  : {args.epochs}")
    print(f"============================\n")

    # ---------- Model ----------
    model = STGCN(
        num_nodes=num_nodes,
        in_channels=1,
        spatial_hidden=args.spatial_hidden,
        temporal_hidden=args.temporal_hidden,
        horizon=args.horizon,
        dropout=args.dropout,
        readout_mode=args.readout_mode,
    ).to(device)

    adj = load_adjacency(args.adj_path, device)
    if adj.shape[0] != num_nodes or adj.shape[1] != num_nodes:
        raise ValueError(
            f"Adjacency shape {tuple(adj.shape)} does not match data nodes ({num_nodes})"
        )

    # ---------- Optimiser / scheduler ----------
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.lr_min,
    )

    best_val = float("inf")
    best_path = os.path.join(args.save_dir, "best_stgcn.pt")
    update_verified = False
    epochs_no_improve = 0

    naive_val = _naive_last_step_mse(loaders["val"], device)
    print(f"Naive baseline val_mse (last-step persistence): {naive_val:.6f}\n")

    # ---------- Training loop ----------
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        grad_norm_epoch = 0.0

        for x, y in loaders["train"]:
            x = x.to(device)
            y = y.to(device)

            if x.shape[1:] != (args.history, num_nodes, 1):
                raise ValueError(f"Unexpected input shape in batch: {tuple(x.shape)}")
            if y.shape[1:] != (num_nodes, 1):
                raise ValueError(f"Unexpected target shape in batch: {tuple(y.shape)}")

            before_first = None
            if not update_verified:
                before_first = next(model.parameters()).detach().clone()

            optimizer.zero_grad()
            pred = model(x, adj)
            loss = criterion(pred, y)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            grad_norm_epoch += float(grad_norm)
            optimizer.step()

            if not update_verified and before_first is not None:
                after_first = next(model.parameters()).detach()
                delta = torch.mean(torch.abs(after_first - before_first)).item()
                if delta > 0.0:
                    update_verified = True

            train_loss += loss.item() * x.size(0)

        train_loss /= len(loaders["train"].dataset)
        grad_norm_epoch /= max(len(loaders["train"]), 1)

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in loaders["val"]:
                x = x.to(device)
                y = y.to(device)
                pred = model(x, adj)
                loss = criterion(pred, y)
                val_loss += loss.item() * x.size(0)
        val_loss /= len(loaders["val"].dataset)
        scheduler.step(val_loss)

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "adj_path": args.adj_path,
                    "history": args.history,
                    "horizon": args.horizon,
                    "spatial_hidden": args.spatial_hidden,
                    "temporal_hidden": args.temporal_hidden,
                    "dropout": args.dropout,
                    "readout_mode": args.readout_mode,
                    "strict_split": True,
                    "train_ratio": args.train_ratio,
                    "val_ratio": args.val_ratio,
                },
                best_path,
            )
        else:
            epochs_no_improve += 1

        current_lr = optimizer.param_groups[0]["lr"]
        marker = " *" if improved else ""
        print(
            f"Epoch {epoch:03d} | train_mse={train_loss:.6f} | val_mse={val_loss:.6f} "
            f"| grad_norm={grad_norm_epoch:.4f} | lr={current_lr:.6g}{marker}"
        )

        # ---- Early stopping ----
        if args.early_stop_patience > 0 and epochs_no_improve >= args.early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.early_stop_patience} epochs)")
            break

    if not update_verified:
        raise RuntimeError("Model parameters were not updated. Check gradients and optimizer.")

    # ---------- Final evaluation on raw-scale metrics ----------
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    val_metrics = evaluate_loader(
        model=model,
        loader=loaders["val"],
        adj=adj,
        device=device,
        scaler=scaler,
        mape_min_actual=args.mape_min_actual,
        mape_eps=args.mape_eps,
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

    rmse = test_metrics["RMSE"]
    if not (10 < rmse < 20):
        print(f"\nWarning: RMSE {rmse:.4f} is outside expected PeMSD4 range (10, 20).")

    print("\n========== Prediction Performance ==========")
    print(
        "Val  -> MAE: {MAE:.4f}, RMSE: {RMSE:.4f}, MAPE: {MAPE:.2f}%".format(
            **val_metrics
        )
    )
    print(
        "Test -> MAE: {MAE:.4f}, RMSE: {RMSE:.4f}, MAPE: {MAPE:.2f}%".format(
            **test_metrics
        )
    )
    print("=============================================")


if __name__ == "__main__":
    main()
