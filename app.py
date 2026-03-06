from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Tuple

import numpy as np
import streamlit as st
import torch

from models.stgcn import STGCN
from utils.graph import load_adjacency

PAPER_HISTORY = 12
PAPER_HORIZON = 1


@dataclass
class Scaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


def build_scaler(flow_path: str, train_ratio: float = 0.7) -> Scaler:
    flow = np.load(flow_path)
    if flow.ndim != 2:
        raise ValueError(f"Expected [time, nodes], got {flow.shape}")
    train_end = int(flow.shape[0] * train_ratio)
    mean = flow[:train_end].mean(axis=0, keepdims=True)
    std = flow[:train_end].std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return Scaler(mean=mean, std=std)


@st.cache_resource
def load_inference_objects(
    checkpoint_path: str,
    adj_path: str,
    flow_path: str,
) -> Tuple[STGCN, torch.Tensor, Scaler, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    ckpt_horizon = int(ckpt.get("horizon", PAPER_HORIZON))
    if ckpt_horizon != PAPER_HORIZON:
        raise ValueError(
            f"Checkpoint horizon must be {PAPER_HORIZON} for paper setting, got {ckpt_horizon}."
        )
    adj = load_adjacency(adj_path, device)
    num_nodes = int(adj.shape[0])

    model = STGCN(
        num_nodes=num_nodes,
        in_channels=1,
        spatial_hidden=ckpt.get("spatial_hidden", 32),
        temporal_hidden=ckpt.get("temporal_hidden", 64),
        horizon=PAPER_HORIZON,
        dropout=ckpt.get("dropout", 0.1),
        readout_mode=ckpt.get("readout_mode", "mean"),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    scaler = build_scaler(flow_path)
    return model, adj, scaler, device


def parse_uploaded_window(file, expected_history: int, expected_nodes: int) -> np.ndarray:
    name = file.name.lower()
    if name.endswith(".npy"):
        arr = np.load(file)
    elif name.endswith(".csv"):
        arr = np.loadtxt(file, delimiter=",")
    else:
        raise ValueError("Only .npy or .csv are supported")

    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr.squeeze(-1)

    if arr.shape != (expected_history, expected_nodes):
        raise ValueError(
            f"Expected shape ({expected_history}, {expected_nodes}), got {arr.shape}"
        )
    return arr.astype(np.float32)


def predict_next_step(
    model: STGCN,
    adj: torch.Tensor,
    scaler: Scaler,
    device: torch.device,
    window_raw: np.ndarray,
) -> np.ndarray:
    x_norm = scaler.transform(window_raw)
    x = torch.from_numpy(x_norm.astype(np.float32)).unsqueeze(0).unsqueeze(-1).to(device)

    with torch.no_grad():
        y_norm = model(x, adj).detach().cpu().numpy()[0, :, 0]

    y_raw = scaler.inverse_transform(y_norm[None, :])[0]
    return y_raw


def main() -> None:
    st.set_page_config(page_title="Traffexplainer Predictor", layout="wide")
    st.title("Traffexplainer Interactive Predictor")
    st.caption("Input a 12x307 window and get next-step traffic prediction (1x307).")

    with st.sidebar:
        st.header("Model Setup")
        checkpoint_path = st.text_input("Checkpoint", value="artifacts/best_stgcn.pt")
        adj_path = st.text_input("Adjacency (.npy)", value="data/pemsd4_adj.npy")
        flow_path = st.text_input("Flow data (.npy)", value="data/pemsd4_flow.npy")
        load_clicked = st.button("Load Model", type="primary")

    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False

    if load_clicked:
        st.session_state.model_loaded = True

    if not st.session_state.model_loaded:
        st.info("Set paths in sidebar and click Load Model.")
        return

    try:
        model, adj, scaler, device = load_inference_objects(
            checkpoint_path=checkpoint_path,
            adj_path=adj_path,
            flow_path=flow_path,
        )
    except Exception as exc:
        st.error(f"Failed to load model setup: {exc}")
        return

    history = PAPER_HISTORY
    nodes = int(adj.shape[0])
    st.success(f"Model loaded on {device}. Expected input shape: ({history}, {nodes}).")
    st.markdown("### Input Template")
    template = np.zeros((history, nodes), dtype=np.float32)
    template_csv = "\n".join([",".join([str(v) for v in row]) for row in template])
    template_npy_buffer = BytesIO()
    np.save(template_npy_buffer, template)
    template_npy_buffer.seek(0)
    c_t1, c_t2 = st.columns(2)
    with c_t1:
        st.download_button(
            label="Download Template (.csv)",
            data=template_csv,
            file_name="input_template_12x307.csv",
            mime="text/csv",
        )
    with c_t2:
        st.download_button(
            label="Download Template (.npy)",
            data=template_npy_buffer.getvalue(),
            file_name="input_template_12x307.npy",
            mime="application/x-npy",
        )

    mode = st.radio(
        "Input mode",
        options=["Upload input window", "Select a window from flow dataset"],
        horizontal=True,
    )

    window = None
    if mode == "Upload input window":
        up = st.file_uploader("Upload .npy or .csv with shape [12, 307]", type=["npy", "csv"])
        if up is not None:
            try:
                window = parse_uploaded_window(up, expected_history=history, expected_nodes=nodes)
                st.write("Input preview (first 3 timesteps x first 8 nodes):")
                st.dataframe(window[:3, :8])
            except Exception as exc:
                st.error(str(exc))
                return
    else:
        try:
            flow = np.load(flow_path)
        except Exception as exc:
            st.error(f"Could not read flow file: {exc}")
            return

        max_start = flow.shape[0] - history - 1
        if max_start < 0:
            st.error("Flow file is too short for 12-step history.")
            return

        start_idx = st.slider("Window start index", min_value=0, max_value=max_start, value=0)
        window = flow[start_idx : start_idx + history].astype(np.float32)
        st.write("Input preview (first 3 timesteps x first 8 nodes):")
        st.dataframe(window[:3, :8])

    if window is None:
        st.warning("Provide input to enable prediction.")
        return

    if st.button("Predict Next Step", type="primary"):
        pred = predict_next_step(model, adj, scaler, device, window)

        st.subheader("Predicted Next Step (1x307)")
        st.dataframe(pred.reshape(1, -1))

        c1, c2 = st.columns(2)
        with c1:
            st.write("Top 10 predicted nodes")
            top_idx = np.argsort(pred)[-10:][::-1]
            top_vals = np.stack([top_idx, pred[top_idx]], axis=1)
            st.dataframe(top_vals, column_config={0: "node", 1: "prediction"})

        with c2:
            focus_node = st.number_input("Inspect node index", min_value=0, max_value=nodes - 1, value=0)
            st.line_chart(
                {
                    "history": window[:, int(focus_node)],
                    "predicted_next": np.r_[np.full(history - 1, np.nan), pred[int(focus_node)]],
                }
            )

        npy_buffer = BytesIO()
        np.save(npy_buffer, pred.astype(np.float32))
        npy_buffer.seek(0)
        csv_data = ",".join([str(v) for v in pred.tolist()])

        st.download_button(
            label="Download prediction (.npy)",
            data=npy_buffer.getvalue(),
            file_name="prediction_1x307.npy",
            mime="application/x-npy",
        )
        st.download_button(
            label="Download prediction (.csv)",
            data=csv_data,
            file_name="prediction_1x307.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
