# Traffexplainer Mini Project (PeMSD4)

This project implements:
- STGCN traffic prediction
- Perturbation-based hierarchical mask learning (`MS1`, `MS2`, `MT`)
- Fidelity and sparsity interpretability metrics
- Mask visualization
- Full-stack explainable dashboard (FastAPI + React + Tailwind + Leaflet + Chart.js)

## Data format

Place files under `data/` (or anywhere and pass full paths):
- Traffic flow matrix: `pemsd4_flow.npy` with shape `[time, nodes]` (nodes=307)
- Adjacency matrix: `pemsd4_adj.npy` with shape `[nodes, nodes]`

## Install

```bash
pip install -r requirements.txt
```

## Full-Stack Web App

### Backend

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Notes:
- Backend auto-loads data from:
  - `data/pemsd4_flow.npy`
  - `data/pemsd4_adj.npy`
- If model/masks are missing, it trains:
  - STGCN for 50 epochs
  - masks for 30 epochs

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Optional API URL override:

```bash
set VITE_API_BASE_URL=http://localhost:8000
```

Frontend URL: `http://localhost:5173`

### API Endpoints

- `POST /predict`
  - request:
    - `{\"date\":\"YYYY-MM-DD\",\"time\":\"HH:MM\"}`
  - response:
    - prediction values for all 307 nodes + risk summary
    - MAE/RMSE/MAPE
- `POST /explain`
  - response:
    - top spatial edges
    - temporal importance vector (12 values)
- `GET /nodes`
  - returns 307 node coordinates for map rendering

## Interactive UI (User Input -> Predicted Output)

```bash
streamlit run app.py
```

In the UI:
- Set `Checkpoint`, `Adjacency`, and `Flow data` paths.
- Click `Load Model`.
- Choose input mode:
  - Upload `.npy`/`.csv` with shape `[12, 307]`, or
  - Pick a window by index from the flow file.
- Click `Predict Next Step`.
- Download prediction as `.npy` or `.csv`.

## 1) Train predictor

```bash
python train_predictor.py \
  --data_path data/pemsd4_flow.npy \
  --adj_path data/pemsd4_adj.npy \
  --epochs 50 \
  --history 12 \
  --horizon 1 \
  --save_dir artifacts
```

## 2) Freeze predictor and train masks

```bash
python train_masks.py \
  --data_path data/pemsd4_flow.npy \
  --adj_path data/pemsd4_adj.npy \
  --checkpoint artifacts/best_stgcn.pt \
  --epochs 30 \
  --history 12 \
  --horizon 1 \
  --save_dir artifacts
```

This saves:
- `artifacts/MS1.npy`
- `artifacts/MS2.npy`
- `artifacts/MT.npy`

## 3) Evaluate interpretability

```bash
python evaluate_interpretability.py \
  --data_path data/pemsd4_flow.npy \
  --adj_path data/pemsd4_adj.npy \
  --checkpoint artifacts/best_stgcn.pt \
  --ms1_path artifacts/MS1.npy \
  --ms2_path artifacts/MS2.npy \
  --topk_ratio 0.1
```

Outputs:
- Prediction Performance (MAE, RMSE, MAPE)
- Interpretability (Fidelity, Sparsity)

## 4) Visualize masks

```bash
python visualize_masks.py \
  --ms1_path artifacts/MS1.npy \
  --ms2_path artifacts/MS2.npy \
  --mt_path artifacts/MT.npy \
  --save_dir artifacts/figures
```

Generates:
- `ms1_heatmap.png`
- `ms2_heatmap.png`
- `ms1_ms2_heatmap.png`
- `temporal_importance.png`
