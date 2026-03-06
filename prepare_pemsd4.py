from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


def load_flow(npz_path: str, channel: int = 0, min_timesteps: int = 16000) -> np.ndarray:
    obj = np.load(npz_path)
    if "data" not in obj:
        raise ValueError(f"{npz_path} must contain key 'data'")
    data = obj["data"]

    if data.ndim == 2:
        flow = data
    elif data.ndim == 3:
        # common shape: [time, nodes, features]
        flow = data[:, :, channel]
    else:
        raise ValueError(f"Unsupported data shape: {data.shape}")

    if flow.shape[1] != 307:
        raise ValueError(f"Expected 307 nodes, got shape {flow.shape}")
    if flow.shape[0] < min_timesteps:
        raise ValueError(
            f"Flow appears truncated: got {flow.shape[0]} timesteps, "
            f"expected at least {min_timesteps} for full PeMSD4."
        )
    return flow.astype(np.float32)


def _detect_cols(df: pd.DataFrame) -> tuple[str, str, str]:
    cols = {c.lower(): c for c in df.columns}
    from_col = (
        cols.get("origin_id")
        or cols.get("from")
        or cols.get("src")
        or cols.get("i")
        or df.columns[0]
    )
    to_col = (
        cols.get("destination_id")
        or cols.get("to")
        or cols.get("dst")
        or cols.get("j")
        or df.columns[1]
    )
    value_col = (
        cols.get("distance")
        or cols.get("cost")
        or cols.get("dist")
        or cols.get("weight")
        or df.columns[2]
    )
    return from_col, to_col, value_col


def build_adjacency_from_file(
    graph_path: str,
    num_nodes: int = 307,
    sigma2: float = 0.1,
    epsilon: float = 0.5,
) -> np.ndarray:
    df = pd.read_csv(graph_path)
    from_col, to_col, value_col = _detect_cols(df)
    is_distance = value_col.lower() in {"distance", "cost", "dist"}

    a = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    for _, row in df.iterrows():
        i = int(row[from_col])
        j = int(row[to_col])
        value = float(row[value_col])

        if i < 0 or i >= num_nodes or j < 0 or j >= num_nodes:
            continue

        if is_distance:
            # Distance -> Gaussian kernel.
            w = float(np.exp(-(value**2) / sigma2))
            if w < epsilon:
                continue
        else:
            # rel/weight style input.
            w = value

        a[i, j] = w
        a[j, i] = w

    np.fill_diagonal(a, 1.0)
    return a


def build_sparse_corr_adjacency(flow: np.ndarray, top_k: int = 20) -> np.ndarray:
    # flow: [time, nodes]
    corr = np.corrcoef(flow, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    corr = np.maximum(corr, 0.0)
    np.fill_diagonal(corr, 0.0)

    n = corr.shape[0]
    a = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        idx = np.argpartition(corr[i], -top_k)[-top_k:]
        a[i, idx] = corr[i, idx]
    a = np.maximum(a, a.T)
    np.fill_diagonal(a, 1.0)
    return a


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare PeMSD4 flow/adj npy files")
    p.add_argument("--npz", required=True, help="Path to pemsd4 npz containing key 'data'")
    p.add_argument(
        "--graph_csv",
        required=True,
        help="Path to graph file (.csv/.rel) with columns like origin_id,destination_id,weight or distance",
    )
    p.add_argument("--out_dir", default="data")
    p.add_argument("--channel", type=int, default=0, help="Feature channel for flow when data is 3D")
    p.add_argument("--sigma2", type=float, default=0.1)
    p.add_argument("--epsilon", type=float, default=0.5)
    p.add_argument("--min_timesteps", type=int, default=16000)
    p.add_argument("--fallback_topk", type=int, default=20)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    flow = load_flow(args.npz, channel=args.channel, min_timesteps=args.min_timesteps)
    adj = build_adjacency_from_file(
        args.graph_csv,
        num_nodes=flow.shape[1],
        sigma2=args.sigma2,
        epsilon=args.epsilon,
    )
    off_diag = adj[~np.eye(adj.shape[0], dtype=bool)]
    is_dense_uniform = (
        float(np.std(off_diag)) < 1e-8 and float(np.mean(off_diag > 0)) > 0.95
    )
    if is_dense_uniform:
        print(
            "[Warning] Graph file produced near-uniform dense adjacency. "
            f"Building sparse correlation adjacency with top_k={args.fallback_topk}."
        )
        adj = build_sparse_corr_adjacency(flow, top_k=args.fallback_topk)

    flow_path = os.path.join(args.out_dir, "pemsd4_flow.npy")
    adj_path = os.path.join(args.out_dir, "pemsd4_adj.npy")

    np.save(flow_path, flow)
    np.save(adj_path, adj)

    print(f"Saved flow: {flow_path} shape={flow.shape} dtype={flow.dtype}")
    print(f"Saved adj:  {adj_path} shape={adj.shape} dtype={adj.dtype}")
    print(
        "Checks -> timesteps: {}, nodes: {}, adj_nonzero: {}".format(
            flow.shape[0], flow.shape[1], int(np.count_nonzero(adj))
        )
    )


if __name__ == "__main__":
    main()
