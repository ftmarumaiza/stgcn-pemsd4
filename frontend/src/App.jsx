import React, { useEffect, useMemo, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Activity, BrainCircuit, Cpu, Radar, Sparkles } from "lucide-react";
import { fetchMetrics, fetchNodes, postExplain, postPredict } from "./lib/api";
import { Badge } from "./components/ui/badge";
import { ControlPanel } from "./components/ControlPanel";
import { FuturisticMap } from "./components/FuturisticMap";
import { AnalyticsPanel } from "./components/AnalyticsPanel";

function getInitialTime() {
  const d = new Date();
  const h = String(d.getHours()).padStart(2, "0");
  const m = String(Math.floor(d.getMinutes() / 5) * 5).padStart(2, "0");
  const rounded = `${h}:${m}`;
  return rounded < "01:00" ? "01:00" : rounded;
}

const shellFade = {
  initial: { opacity: 0, y: 24 },
  animate: { opacity: 1, y: 0 },
};

function BackgroundParticles() {
  const particles = useMemo(
    () =>
      Array.from({ length: 18 }, (_, i) => ({
        id: i,
        left: `${5 + ((i * 7.8) % 90)}%`,
        delay: (i % 9) * 0.8,
        duration: 12 + (i % 5) * 3,
        size: 2 + (i % 3),
      })),
    []
  );

  return (
    <div className="pointer-events-none absolute inset-0 overflow-hidden">
      {particles.map((particle) => (
        <motion.span
          key={particle.id}
          className="particle"
          style={{
            left: particle.left,
            width: particle.size,
            height: particle.size,
          }}
          animate={{ y: ["105%", "-10%"], opacity: [0, 0.8, 0] }}
          transition={{
            duration: particle.duration,
            delay: particle.delay,
            repeat: Infinity,
            ease: "linear",
          }}
        />
      ))}
    </div>
  );
}

export default function App() {
  const [date, setDate] = useState("2018-01-01");
  const [time, setTime] = useState(getInitialTime);
  const [nodes, setNodes] = useState([]);

  const [prediction, setPrediction] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [explanation, setExplanation] = useState(null);

  const [viewMode, setViewMode] = useState("prediction");
  const [threshold, setThreshold] = useState(45);

  const [loadingPredict, setLoadingPredict] = useState(false);
  const [loadingExplain, setLoadingExplain] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    const run = async () => {
      const [nodesResult, metricsResult] = await Promise.allSettled([
        fetchNodes(),
        fetchMetrics(),
      ]);

      if (nodesResult.status === "fulfilled") {
        setNodes(nodesResult.value);
      } else {
        const e = nodesResult.reason;
        setError(e?.response?.data?.detail || e?.message || "Failed to load node positions");
      }

      if (metricsResult.status === "fulfilled") {
        setMetrics(metricsResult.value || null);
      } else if (nodesResult.status === "fulfilled") {
        const e = metricsResult.reason;
        setError(e?.response?.data?.detail || e?.message || "Failed to load evaluation metrics");
      }
    };
    run();
  }, []);

  const predictionValues = useMemo(() => prediction?.values || [], [prediction]);
  const topEdges = useMemo(() => explanation?.spatial_explanation?.top_edges || [], [explanation]);

  const handlePredict = async () => {
    setLoadingPredict(true);
    setError("");
    try {
      const res = await postPredict({ date, time, horizon_steps: 1 });
      setPrediction(res.prediction || null);
      setMetrics(res.metrics || null);
      setViewMode("prediction");
    } catch (e) {
      setError(e?.response?.data?.detail || e.message || "Prediction failed");
    } finally {
      setLoadingPredict(false);
    }
  };

  const handleExplain = async () => {
    setLoadingExplain(true);
    setError("");
    setViewMode("explanation");
    try {
      const res = await postExplain({ date, time });
      setExplanation(res || null);
      if (res?.prediction) setPrediction(res.prediction);
      if (res?.metrics) setMetrics(res.metrics);
    } catch (e) {
      setError(e?.response?.data?.detail || e.message || "Explanation failed");
    } finally {
      setLoadingExplain(false);
    }
  };

  return (
    <main className="dashboard-bg relative min-h-screen overflow-hidden p-4 md:p-6">
      <BackgroundParticles />
      <div className="relative z-10 mx-auto flex max-w-[1900px] flex-col gap-4">
        <motion.header
          {...shellFade}
          transition={{ duration: 0.45, ease: "easeOut" }}
          className="glass-card glow-border flex flex-col gap-3 rounded-3xl p-5 md:p-7"
        >
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="space-y-2">
              <h1 className="text-balance text-2xl font-black tracking-tight text-white md:text-4xl">
                Explainable Traffic Prediction
              </h1>
            </div>
            <Badge className="status-badge px-3 py-1.5 text-xs md:text-sm">
              <span className="mr-2 h-2 w-2 rounded-full bg-emerald-400 shadow-[0_0_14px_#10B981]" />
              Model Active
            </Badge>
          </div>

          <div className="flex flex-wrap items-center gap-2 text-[11px] text-slate-300 md:text-xs">
            <Badge variant="secondary" className="pill-badge"><Cpu className="h-3.5 w-3.5" />Edge AI Pipeline</Badge>
            <Badge variant="secondary" className="pill-badge"><BrainCircuit className="h-3.5 w-3.5" />Explainability Engine</Badge>
            <Badge variant="secondary" className="pill-badge"><Radar className="h-3.5 w-3.5" />Citywide Telemetry</Badge>
            <Badge variant="secondary" className="pill-badge"><Sparkles className="h-3.5 w-3.5" />Hierarchical Masks</Badge>
          </div>
        </motion.header>

        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              className="glass-card rounded-2xl border border-red-500/40 bg-red-500/10 px-4 py-2 text-sm text-red-200"
            >
              <div className="flex items-center gap-2"><Activity className="h-4 w-4" />{error}</div>
            </motion.div>
          )}
        </AnimatePresence>

        <section className="grid grid-cols-1 gap-4 xl:grid-cols-[340px_minmax(0,1fr)_430px] 2xl:gap-5">
          <motion.div {...shellFade} transition={{ delay: 0.08, duration: 0.4 }}>
            <ControlPanel
              date={date}
              time={time}
              onDateChange={setDate}
              onTimeChange={setTime}
              onPredict={handlePredict}
              onExplain={handleExplain}
              loadingPredict={loadingPredict}
              loadingExplain={loadingExplain}
            />
          </motion.div>

          <motion.div {...shellFade} transition={{ delay: 0.16, duration: 0.4 }} className="min-h-[520px]">
            <FuturisticMap
              nodes={nodes}
              predictionValues={predictionValues}
              topEdges={topEdges}
              viewMode={viewMode}
              onModeChange={setViewMode}
              threshold={threshold}
              onThresholdChange={setThreshold}
              loadingExplain={loadingExplain}
            />
          </motion.div>

          <motion.div {...shellFade} transition={{ delay: 0.24, duration: 0.4 }}>
            <AnalyticsPanel
              prediction={prediction}
              metrics={metrics}
              explanation={explanation}
              viewMode={viewMode}
              loadingExplain={loadingExplain}
              selectedDate={date}
              selectedTime={time}
            />
          </motion.div>
        </section>
      </div>
    </main>
  );
}
