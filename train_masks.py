from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from data.traffic_dataset import build_dataloaders
from interpret.hierarchical_masks import HierarchicalMasks
from models.stgcn import STGCN
from utils.engine import evaluate_loader
from utils.graph import load_adjacency

PAPER_HISTORY = 12
PAPER_HORIZON = 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Freeze STGCN and train hierarchical masks")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--adj_path", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="artifacts")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--history", type=int, default=12)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--num_workers", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.history != PAPER_HISTORY or args.horizon != PAPER_HORIZON:
        raise ValueError(
            f"Traffexplainer paper setting requires --history {PAPER_HISTORY} --horizon {PAPER_HORIZON}. "
            f"Got history={args.history}, horizon={args.horizon}."
        )
    os.makedirs(args.save_dir, exist_ok=True)

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

    for p in model.parameters():
        p.requires_grad = False

    masks = HierarchicalMasks(num_nodes=num_nodes, history=args.history).to(device)
    optimizer = Adam(masks.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    adj = load_adjacency(args.adj_path, device)

    best_val = float("inf")
    best_path = os.path.join(args.save_dir, "best_masks.pt")

    for epoch in range(1, args.epochs + 1):
        masks.train()
        train_loss = 0.0
        for x, y in loaders["train"]:
            x = x.to(device)
            y = y.to(device)

            x_masked, a1, a2 = masks(x, adj)
            pred = model(x_masked, adj=adj, adj1=a1, adj2=a2)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)

        train_loss /= len(loaders["train"].dataset)

        masks.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in loaders["val"]:
                x = x.to(device)
                y = y.to(device)
                x_masked, a1, a2 = masks(x, adj)
                pred = model(x_masked, adj=adj, adj1=a1, adj2=a2)
                loss = criterion(pred, y)
                val_loss += loss.item() * x.size(0)
        val_loss /= len(loaders["val"].dataset)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"mask_state": masks.state_dict()}, best_path)

        print(f"Epoch {epoch:03d} | mask_train_mse={train_loss:.6f} | mask_val_mse={val_loss:.6f}")

    best_masks = torch.load(best_path, map_location=device)
    masks.load_state_dict(best_masks["mask_state"])
    masks.eval()

    # Optional post-mask prediction performance using masked forward.
    @torch.no_grad()
    def evaluate_with_masks(loader_name: str) -> dict:
        model.eval()
        preds, tgts = [], []
        for x, y in loaders[loader_name]:
            x = x.to(device)
            y = y.to(device)
            x_masked, a1, a2 = masks(x, adj)
            out = model(x_masked, adj=adj, adj1=a1, adj2=a2)

            out_np = out.detach().cpu().numpy().squeeze(-1)
            y_np = y.detach().cpu().numpy().squeeze(-1)
            out_np = scaler.inverse_transform(out_np)
            y_np = scaler.inverse_transform(y_np)
            preds.append(out_np)
            tgts.append(y_np)

        pred = np.concatenate(preds, axis=0)
        tgt = np.concatenate(tgts, axis=0)
        from utils.metrics import compute_metrics_np

        return compute_metrics_np(pred, tgt)

    test_metrics = evaluate_with_masks("test")

    ms1 = masks.symmetrized_ms1().detach().cpu().numpy()
    ms2 = masks.symmetrized_ms2().detach().cpu().numpy()
    mt = masks.mt.detach().cpu().numpy()

    np.save(os.path.join(args.save_dir, "MS1.npy"), ms1)
    np.save(os.path.join(args.save_dir, "MS2.npy"), ms2)
    np.save(os.path.join(args.save_dir, "MT.npy"), mt)

    print("\nMasked Prediction Performance (Test)")
    print("MAE: {MAE:.4f}, RMSE: {RMSE:.4f}, MAPE: {MAPE:.2f}%".format(**test_metrics))
    print(f"Saved masks: {os.path.join(args.save_dir, 'MS1.npy')}, {os.path.join(args.save_dir, 'MS2.npy')}, {os.path.join(args.save_dir, 'MT.npy')}")


if __name__ == "__main__":
    main()
