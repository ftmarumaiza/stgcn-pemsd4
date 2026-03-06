import React from "react";

function mergeClassNames(...values) {
  return values.filter(Boolean).join(" ");
}

export function Badge({ className = "", variant = "default", ...props }) {
  const base = "inline-flex items-center gap-1 rounded-full border px-2.5 py-1 text-xs font-medium";
  const variants = {
    default: "border-cyan-300/35 bg-cyan-400/10 text-cyan-100",
    secondary: "border-slate-300/20 bg-slate-700/35 text-slate-200",
  };
  return <span className={mergeClassNames(base, variants[variant], className)} {...props} />;
}
