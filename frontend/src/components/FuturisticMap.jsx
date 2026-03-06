import React, { memo, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { CircleMarker, MapContainer, Polyline, TileLayer, Tooltip } from "react-leaflet";
import { Info } from "lucide-react";
import { Slider } from "./ui/slider";
import "leaflet/dist/leaflet.css";

const MAP_CENTER = [36.7783, -119.4179];

function nodeColor(value, minV, maxV) {
  if (value == null || Number.isNaN(value)) return "#334155";
  const norm = (value - minV) / Math.max(maxV - minV, 1e-5);
  if (norm < 0.33) return "#10B981";
  if (norm < 0.66) return "#F59E0B";
  return "#EF4444";
}

function FuturisticMapComponent({
  nodes,
  predictionValues,
  topEdges,
  viewMode,
  onModeChange,
  threshold,
  onThresholdChange,
  loadingExplain,
}) {
  const minV = predictionValues?.length ? Math.min(...predictionValues) : 0;
  const maxV = predictionValues?.length ? Math.max(...predictionValues) : 1;
  const nodeById = useMemo(() => new Map(nodes.map((n) => [n.id, n])), [nodes]);

  const filteredEdges = useMemo(() => {
    const maxImportance = Math.max(1e-6, ...(topEdges || []).map((e) => Number(e.importance || 0)));
    return (topEdges || []).filter((edge) => (Number(edge.importance || 0) / maxImportance) * 100 >= threshold);
  }, [topEdges, threshold]);

  return (
    <section className="glass-card glow-border relative h-full min-h-[520px] overflow-hidden rounded-3xl p-3">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3 px-1">
        <p className="text-sm font-medium text-slate-300">Interactive Graph Map</p>
        <div className="mode-switch">
          <button
            onClick={() => onModeChange("prediction")}
            className={viewMode === "prediction" ? "active" : ""}
          >
            Prediction
          </button>
          <button
            onClick={() => onModeChange("explanation")}
            className={viewMode === "explanation" ? "active" : ""}
          >
            Explanation
          </button>
        </div>
      </div>

      <div className="map-frame relative h-[430px] overflow-hidden rounded-2xl border border-cyan-300/20">
        <MapContainer center={MAP_CENTER} zoom={7} scrollWheelZoom className="h-full w-full" preferCanvas>
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          />

          {nodes.map((node) => {
            const pred = predictionValues?.[node.id];
            const color = nodeColor(pred, minV, maxV);
            const isCongested = color === "#EF4444";
            return (
              <React.Fragment key={node.id}>
                <CircleMarker
                  center={[node.lat, node.lon]}
                  radius={isCongested ? 5.8 : 4.5}
                  pathOptions={{
                    color,
                    fillColor: color,
                    fillOpacity: loadingExplain && viewMode === "explanation" ? 0.2 : 0.9,
                    weight: 1,
                    className: isCongested && viewMode === "prediction" ? "node-ripple" : "",
                  }}
                >
                  <Tooltip>
                    Node {node.id}
                    <br />
                    Flow: {pred != null ? pred.toFixed(2) : "-"}
                  </Tooltip>
                </CircleMarker>

                {viewMode === "prediction" && isCongested && (
                  <CircleMarker
                    center={[node.lat, node.lon]}
                    radius={9}
                    pathOptions={{ color: "#EF4444", fillOpacity: 0, weight: 1.2, className: "pulse-ring" }}
                  />
                )}
              </React.Fragment>
            );
          })}

          <AnimatePresence>
            {viewMode === "explanation" &&
              filteredEdges.map((edge, idx) => {
                const from = nodeById.get(edge.from);
                const to = nodeById.get(edge.to);
                if (!from || !to) return null;
                const importance = Number(edge.importance || 0);
                const hot = importance > 0.75;
                return (
                  <Polyline
                    key={`${edge.from}-${edge.to}-${idx}`}
                    positions={[
                      [from.lat, from.lon],
                      [to.lat, to.lon],
                    ]}
                    pathOptions={{
                      color: hot ? "#EF4444" : "#FB7185",
                      weight: 2 + Math.min(importance * 6, 6),
                      opacity: 0.9,
                      className: hot ? "edge-pulse" : "",
                    }}
                  />
                );
              })}
          </AnimatePresence>
        </MapContainer>

        <div className="legend-float">
          <p className="mb-1 inline-flex items-center gap-1 text-xs font-semibold text-slate-200"><Info className="h-3.5 w-3.5" />Legend</p>
          <span><i style={{ background: "#10B981" }} />Green = Low</span>
          <span><i style={{ background: "#F59E0B" }} />Orange = Medium</span>
          <span><i style={{ background: "#EF4444" }} />Red = High Importance</span>
        </div>

        <AnimatePresence>
          {loadingExplain && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute inset-0 z-[999] flex items-center justify-center bg-slate-950/55 backdrop-blur-[1px]"
            >
              <div className="inline-flex items-center gap-3 rounded-full border border-cyan-300/35 bg-slate-900/70 px-4 py-2 text-sm text-cyan-200">
                <span className="loader" />Building hierarchical masks...
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <div className="mt-4 rounded-2xl border border-cyan-300/20 bg-slate-900/35 p-3">
        <div className="mb-2 flex items-center justify-between text-xs text-slate-300">
          <span>Mask Threshold</span>
          <span>{threshold}%</span>
        </div>
        <Slider min={0} max={100} step={1} value={threshold} onChange={onThresholdChange} />
      </div>
    </section>
  );
}

export const FuturisticMap = memo(FuturisticMapComponent);
