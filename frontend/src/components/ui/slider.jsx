import React from "react";

export function Slider({ min = 0, max = 100, step = 1, value = 0, onChange }) {
  return (
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={(e) => onChange(Number(e.target.value))}
      className="slider-future"
    />
  );
}
