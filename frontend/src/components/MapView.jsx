import React, { memo, useMemo } from "react";
import { MapContainer, TileLayer, CircleMarker, Polyline, Tooltip } from "react-leaflet";
import "leaflet/dist/leaflet.css";

const MAP_CENTER = [36.8, -119.4];

function colorForPrediction(value, minV, maxV) {
  if (value == null || Number.isNaN(value)) return "#64748b";
  const norm = (value - minV) / Math.max(maxV - minV, 1e-6);
  if (norm < 0.33) return "#14b8a6";
  if (norm < 0.66) return "#f59e0b";
  return "#ef4444";
}

function MapView({ nodes, predictionValues, topEdges, viewMode }) {
  const minV = predictionValues?.length ? Math.min(...predictionValues) : 0;
  const maxV = predictionValues?.length ? Math.max(...predictionValues) : 1;

  const nodeById = useMemo(() => new Map(nodes.map((n) => [n.id, n])), [nodes]);

  return (
    <section className="relative h-full overflow-hidden rounded-2xl bg-slate-100 shadow-panel">
      <MapContainer center={MAP_CENTER} zoom={6} scrollWheelZoom className="h-full w-full" preferCanvas>
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {nodes.map((node) => {
          const pred = predictionValues?.[node.id];
          const color = colorForPrediction(pred, minV, maxV);
          return (
            <CircleMarker
              key={node.id}
              center={[node.lat, node.lon]}
              radius={4.5}
              pathOptions={{ color, fillColor: color, fillOpacity: 0.85, weight: 1 }}
            >
              <Tooltip>
                Node {node.id}
                <br />
                Flow: {pred != null ? pred.toFixed(2) : "-"}
              </Tooltip>
            </CircleMarker>
          );
        })}

        {viewMode === "explanation" &&
          (topEdges || []).map((edge, idx) => {
            const from = nodeById.get(edge.from);
            const to = nodeById.get(edge.to);
            if (!from || !to) return null;
            return (
              <Polyline
                key={`${edge.from}-${edge.to}-${idx}`}
                positions={[
                  [from.lat, from.lon],
                  [to.lat, to.lon],
                ]}
                pathOptions={{ color: "#0ea5e9", weight: 2 + Math.min(edge.importance, 3), opacity: 0.8 }}
              />
            );
          })}
      </MapContainer>

      <div className="pointer-events-none absolute left-3 top-3 rounded-lg bg-white/90 px-3 py-2 text-xs shadow">
        {viewMode === "prediction" ? "Prediction View" : "Explanation View"}
      </div>
    </section>
  );
}

export default memo(MapView);
