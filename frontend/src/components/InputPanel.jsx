import React from "react";

export default function InputPanel({ date, time, onDateChange, onTimeChange, onPredict, onExplain, loading }) {
  return (
    <section className="h-full rounded-2xl bg-traffic-panel p-5 shadow-panel">
      <h2 className="text-lg font-semibold text-traffic-ink">Input Controls</h2>
      <p className="mt-1 text-sm text-slate-600">Select target timestamp and run prediction/explanation.</p>

      <div className="mt-5 space-y-4">
        <label className="block">
          <span className="text-sm font-medium text-slate-700">Date</span>
          <input
            type="date"
            className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm focus:border-cyan-500 focus:outline-none"
            value={date}
            onChange={(e) => onDateChange(e.target.value)}
          />
        </label>

        <label className="block">
          <span className="text-sm font-medium text-slate-700">Time</span>
          <input
            type="time"
            step="300"
            className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm focus:border-cyan-500 focus:outline-none"
            value={time}
            onChange={(e) => onTimeChange(e.target.value)}
          />
        </label>
      </div>

      <div className="mt-6 space-y-3">
        <button
          className="w-full rounded-lg bg-cyan-600 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-cyan-500 disabled:cursor-not-allowed disabled:bg-slate-400"
          onClick={onPredict}
          disabled={loading}
        >
          {loading ? "Predicting..." : "Predict"}
        </button>
        <button
          className="w-full rounded-lg bg-slate-800 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:bg-slate-400"
          onClick={onExplain}
          disabled={loading}
        >
          {loading ? "Running..." : "Explain"}
        </button>
      </div>
    </section>
  );
}
