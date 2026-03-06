import React, { useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Bar,
  BarChart,
  Cell,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { AnimatedCounter } from "./AnimatedCounter";

function avg(values = []) {
  if (!values.length) return 245.82;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function riskColor(level = "") {
  const lower = level.toLowerCase();
  if (lower.includes("high") || lower.includes("congested")) return "bg-red-500/20 text-red-200 border-red-400/40";
  if (lower.includes("moderate") || lower.includes("medium")) return "bg-amber-500/20 text-amber-100 border-amber-400/40";
  return "bg-emerald-500/20 text-emerald-100 border-emerald-400/40";
}

function makeTemporalData(vector, selectedDate, selectedTime) {
  const history = 12;
  const minutesPerStep = 5;
  const base = new Date(`${selectedDate}T${selectedTime}:00`);
  const hasValidBase = !Number.isNaN(base.getTime());

  const labels = Array.from({ length: history }, (_, idx) => {
    if (!hasValidBase) return `${idx * 5}-${(idx + 1) * 5}`;
    const minutesBefore = (history - idx) * minutesPerStep;
    const ts = new Date(base.getTime() - minutesBefore * 60 * 1000);
    return ts.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hour12: false });
  });

  const source = vector?.length ? vector.slice(0, 12) : [];
  const maxValue = Math.max(...source.map((v) => Number(v || 0)), 0);
  return labels.map((slot, idx) => {
    const value = Number(source[idx] ?? 0);
    return {
      slot,
      value,
      // Display-scaled height for readability while preserving raw values in tooltip.
      displayValue: maxValue > 0 ? value / maxValue : 0,
    };
  });
}

function sparklineData(seed) {
  return Array.from({ length: 10 }, (_, i) => ({
    p: i,
    v: 0.2 + (((seed + 3) * (i + 5)) % 17) / 20,
  }));
}

function metricValueOrNull(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

export function AnalyticsPanel({
  prediction,
  metrics,
  explanation,
  viewMode,
  loadingExplain,
  selectedDate,
  selectedTime,
}) {
  const predictionValues = prediction?.values || [];
  const avgFlow = avg(predictionValues);
  const flowScore = Number(prediction?.flow_score_0_100 ?? avgFlow);
  const congestion = prediction?.risk_summary || "Moderate";
  const edges = explanation?.spatial_explanation?.top_edges || [];

  const highInfluencePairs = useMemo(() => {
    return edges.map((edge, idx) => ({
      ...edge,
      id: `${edge.from}-${edge.to}-${idx}`,
      score: Number(edge.importance || 0),
      spark: sparklineData(idx + Number(edge.from || 1)),
    }));
  }, [edges]);

  const temporalData = useMemo(
    () => makeTemporalData(
      explanation?.temporal_explanation?.importance_vector || [],
      selectedDate,
      selectedTime
    ),
    [explanation, selectedDate, selectedTime]
  );

  const metricItems = [
    { label: "MAE", value: metricValueOrNull(metrics?.MAE) },
    { label: "RMSE", value: metricValueOrNull(metrics?.RMSE) },
    { label: "MAPE", value: metricValueOrNull(metrics?.MAPE), suffix: "%" },
  ];

  return (
    <aside className="space-y-4">
      <Card className="glass-card glow-hover rounded-3xl">
        <CardHeader>
          <CardTitle className="text-base font-semibold text-slate-100">Prediction Card</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-4xl font-black leading-none text-white">
            <AnimatedCounter value={flowScore} decimals={2} />
          </p>
          <p className="mt-2 text-sm text-slate-300">Predicted Traffic Score (0-100)</p>
          <p className="mt-1 text-xs text-slate-300">Raw average flow: {avgFlow.toFixed(2)}</p>
          <p className="mt-1 text-xs text-slate-300">{prediction?.forecast_note || "Predicting traffic for next 5 minutes"}</p>
          <p className="mt-1 text-xs text-cyan-200">Predicted timestamp: {prediction?.predicted_timestamp || "-"}</p>
          <Badge className={`mt-4 border ${riskColor(congestion)}`}>Congestion Level: {congestion}</Badge>
        </CardContent>
      </Card>

      <Card className="glass-card glow-hover rounded-3xl">
        <CardHeader>
          <CardTitle className="text-base font-semibold text-slate-100">Performance Metrics</CardTitle>
        </CardHeader>
        <CardContent className="grid grid-cols-3 gap-2">
          {metricItems.map((item) => (
            <motion.div
              key={item.label}
              whileHover={{ scale: 1.04 }}
              className="rounded-2xl border border-cyan-300/25 bg-slate-900/45 p-2 text-center"
            >
              <p className="text-[11px] text-slate-300">{item.label}</p>
              <p className="text-lg font-bold text-cyan-200">
                {item.value == null ? "--" : (
                  <>
                    <AnimatedCounter value={item.value} decimals={2} />
                    {item.suffix || ""}
                  </>
                )}
              </p>
            </motion.div>
          ))}
        </CardContent>
      </Card>

      <AnimatePresence mode="wait">
        {(viewMode === "explanation" || edges.length > 0) && (
          <motion.div
            key="spatial-card"
            initial={{ opacity: 0, x: 28 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 28 }}
            transition={{ duration: 0.35 }}
          >
            <Card className="glass-card glow-hover rounded-3xl">
              <CardHeader>
                <CardTitle className="text-base font-semibold text-slate-100">High-Influence Road Pairs</CardTitle>
              </CardHeader>
              <CardContent className="max-h-[400px] overflow-y-auto pr-1 scroll-smooth">
                <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
                  {highInfluencePairs.map((pair, index) => (
                    <motion.div
                      key={pair.id}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: index * 0.03 }}
                      className="grid grid-cols-[1fr_72px_auto] items-center gap-1 rounded-xl border border-cyan-200/15 bg-slate-900/45 px-2.5 py-1.5"
                    >
                      <div>
                        <p className="text-xs text-slate-200">Road {pair.from} {"->"} {pair.to}</p>
                      </div>
                      <div className="h-7">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={pair.spark}>
                            <Line type="monotone" dataKey="v" stroke="#22D3EE" strokeWidth={1.5} dot={false} />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                      <span className={`justify-self-end text-sm font-bold ${pair.score > 0.8 ? "text-red-300" : pair.score > 0.65 ? "text-amber-200" : "text-emerald-200"}`}>
                        {pair.score.toFixed(2)}
                      </span>
                    </motion.div>
                  ))}
                </div>
                {!highInfluencePairs.length && (
                  <p className="text-xs text-slate-400">No high-influence road pairs for current threshold.</p>
                )}
              </CardContent>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>

      <Card className="glass-card glow-hover rounded-3xl">
        <CardHeader>
          <CardTitle className="text-base font-semibold text-slate-100">Temporal Importance</CardTitle>
        </CardHeader>
        <CardContent className="h-[200px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={temporalData} margin={{ top: 4, right: 0, left: -20, bottom: 0 }}>
              <XAxis
                dataKey="slot"
                tick={{ fill: "#94A3B8", fontSize: 10 }}
                axisLine={false}
                tickLine={false}
                interval={0}
              />
              <YAxis hide domain={[0, 1]} />
              <Tooltip
                cursor={false}
                formatter={(v, _name, ctx) => [Number(ctx?.payload?.value ?? v).toFixed(6), "value"]}
                contentStyle={{ background: "#0F172A", border: "1px solid #164E63" }}
              />
              <Bar dataKey="displayValue" radius={[6, 6, 0, 0]} animationDuration={900} background={{ fill: "rgba(56, 189, 248, 0.12)" }}>
                {temporalData.map((entry) => (
                  <Cell
                    key={entry.slot}
                    fill="#38BDF8"
                    opacity={0.85}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          {loadingExplain && <p className="text-xs text-cyan-200">Updating explanation metrics...</p>}
        </CardContent>
      </Card>
    </aside>
  );
}
