import axios from "axios";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export const api = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
});

export async function fetchNodes() {
  const { data } = await api.get("/nodes");
  return data.nodes || [];
}

export async function fetchMetrics() {
  const { data } = await api.get("/metrics");
  return data.metrics || data;
}

export async function postPredict(payload) {
  const { data } = await api.post("/predict", payload);
  return data;
}

export async function postExplain(payload) {
  const { data } = await api.post("/explain", payload);
  return data;
}
