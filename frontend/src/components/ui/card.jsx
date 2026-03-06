import React from "react";

function mergeClassNames(...values) {
  return values.filter(Boolean).join(" ");
}

export function Card({ className = "", ...props }) {
  return <section className={mergeClassNames("border border-cyan-200/20 bg-slate-900/40", className)} {...props} />;
}

export function CardHeader({ className = "", ...props }) {
  return <div className={mergeClassNames("space-y-1 p-4 pb-2", className)} {...props} />;
}

export function CardTitle({ className = "", ...props }) {
  return <h3 className={mergeClassNames("font-semibold tracking-tight", className)} {...props} />;
}

export function CardContent({ className = "", ...props }) {
  return <div className={mergeClassNames("p-4 pt-2", className)} {...props} />;
}
