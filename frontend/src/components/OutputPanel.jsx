import React, { useMemo } from "react";
import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Tooltip,
  Legend,
} from "chart.js";

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip, Legend);

function formatFlow(value) {
  if (value == null || Number.isNaN(value)) return "-";
  return Number(value).toFixed(2);
}

export default function OutputPanel({ prediction, metrics, explanation }) {
  const values = prediction?.values || [];
  const avg = values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;

  const temporal = useMemo(
    () => explanation?.temporal_explanation?.importance_vector || [],
    [explanation]
  );
  const chartData = useMemo(
    () => ({
      labels: temporal.map((_, i) => `t-${12 - i}`),
      datasets: [
        {
          label: "Importance",
          data: temporal,
          backgroundColor: "rgba(6, 182, 212, 0.75)",
          borderRadius: 4,
        },
      ],
    }),
    [temporal]
  );
  const chartOptions = useMemo(
    () => ({
      responsive: false,
      maintainAspectRatio: false,
      animation: false,
      events: [],
      interaction: {
        mode: null,
      },
      plugins: {
        tooltip: {
          enabled: false,
        },
        legend: {
          display: false,
        },
      },
      transitions: {
        active: {
          animation: {
            duration: 0,
          },
        },
      },
    }),
    []
  );

  return (
    <section className="h-full space-y-4 rounded-2xl bg-traffic-panel p-5 shadow-panel">
      <h2 className="text-lg font-semibold text-traffic-ink">Output</h2>

      <article className="rounded-xl border border-slate-200 bg-white p-4">
        <h3 className="text-sm font-semibold uppercase text-slate-500">Prediction</h3>
        <p className="mt-3 text-2xl font-bold text-slate-900">{formatFlow(avg)}</p>
        <p className="text-sm text-slate-600">Average predicted flow</p>
        <p className="mt-2 text-sm font-medium text-cyan-700">
          Congestion: {prediction?.risk_summary || "-"}
        </p>
      </article>

      <article className="rounded-xl border border-slate-200 bg-white p-4">
        <h3 className="text-sm font-semibold uppercase text-slate-500">Metrics</h3>
        <div className="mt-3 grid grid-cols-3 gap-2 text-sm">
          <div className="rounded-lg bg-slate-50 p-2">
            <p className="text-slate-500">MAE</p>
            <p className="font-semibold text-slate-900">{formatFlow(metrics?.MAE)}</p>
          </div>
          <div className="rounded-lg bg-slate-50 p-2">
            <p className="text-slate-500">RMSE</p>
            <p className="font-semibold text-slate-900">{formatFlow(metrics?.RMSE)}</p>
          </div>
          <div className="rounded-lg bg-slate-50 p-2">
            <p className="text-slate-500">MAPE</p>
            <p className="font-semibold text-slate-900">{formatFlow(metrics?.MAPE)}</p>
          </div>
        </div>
      </article>

      <article className="rounded-xl border border-slate-200 bg-white p-4">
        <h3 className="text-sm font-semibold uppercase text-slate-500">High-Influence Road Pairs</h3>
        <ul className="mt-3 space-y-2 text-sm">
          {(explanation?.spatial_explanation?.top_edges || []).map((edge, idx) => (
            <li key={`${edge.from}-${edge.to}-${idx}`} className="rounded-lg bg-slate-50 px-3 py-2">
              Road {edge.from} {"->"} {edge.to} <span className="font-semibold">({edge.importance.toFixed(4)})</span>
            </li>
          ))}
          {!(explanation?.spatial_explanation?.top_edges || []).length && (
            <li className="text-slate-500">No explanation loaded.</li>
          )}
        </ul>
      </article>

      <article className="sticky top-4 rounded-xl border border-slate-200 bg-white p-4">
        <h3 className="text-sm font-semibold uppercase text-slate-500">Temporal Explanation</h3>
        {temporal.length ? (
          <div className="mt-2 overflow-x-auto">
            <Bar data={chartData} options={chartOptions} width={360} height={180} />
          </div>
        ) : (
          <p className="mt-3 text-sm text-slate-500">No explanation loaded.</p>
        )}
        <p className="mt-3 text-sm text-cyan-700">
          Most important window: {explanation?.temporal_explanation?.most_important_window || "-"}
        </p>
      </article>
    </section>
  );
}
