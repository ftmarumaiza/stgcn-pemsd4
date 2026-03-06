import React from "react";
import { motion } from "framer-motion";
import { CalendarDays, Clock3, Sparkles, Wand2 } from "lucide-react";
import { Button } from "./ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";

export function ControlPanel({
  date,
  time,
  onDateChange,
  onTimeChange,
  onPredict,
  onExplain,
  loadingPredict,
  loadingExplain,
}) {
  const isRunning = loadingPredict || loadingExplain;

  return (
    <Card className="glass-card glow-hover h-full min-h-[520px] rounded-3xl">
      <CardHeader>
        <CardTitle className="text-xl font-bold text-white">AI Control Panel</CardTitle>
        <p className="text-sm text-slate-300">
          Set timestamp context and trigger model execution pipelines.
        </p>
      </CardHeader>

      <CardContent className="space-y-5">
        <label className="space-y-2 text-sm text-slate-200">
          <span className="inline-flex items-center gap-2 font-medium"><CalendarDays className="h-4 w-4 text-cyan-300" />Date Picker</span>
          <input
            type="date"
            value={date}
            onChange={(e) => onDateChange(e.target.value)}
            className="input-future"
          />
        </label>

        <label className="space-y-2 text-sm text-slate-200">
          <span className="inline-flex items-center gap-2 font-medium"><Clock3 className="h-4 w-4 text-cyan-300" />Time Picker</span>
          <input
            type="time"
            step="300"
            value={time}
            onChange={(e) => onTimeChange(e.target.value)}
            className="input-future"
          />
        </label>

        <div className="space-y-3 pt-2">
          <motion.div animate={isRunning ? { scale: [1, 1.01, 1] } : { scale: 1 }} transition={{ repeat: Infinity, duration: 1.4 }}>
            <Button
              onClick={onPredict}
              disabled={isRunning}
              className="btn-gradient-cyan relative w-full py-6 text-base font-bold"
            >
              {loadingPredict ? <span className="loader" /> : <Sparkles className="h-4 w-4" />}
              {loadingPredict ? "Running Prediction..." : "Predict"}
            </Button>
          </motion.div>

          <motion.div animate={isRunning ? { scale: [1, 1.01, 1] } : { scale: 1 }} transition={{ repeat: Infinity, duration: 1.6 }}>
            <Button
              onClick={onExplain}
              disabled={isRunning}
              className="btn-gradient-purple relative w-full py-6 text-base font-bold"
            >
              {loadingExplain ? <span className="loader" /> : <Wand2 className="h-4 w-4" />}
              {loadingExplain ? "Computing Masks..." : "Explain"}
            </Button>
          </motion.div>
        </div>
      </CardContent>
    </Card>
  );
}
