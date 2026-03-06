import React from "react";

function mergeClassNames(...values) {
  return values.filter(Boolean).join(" ");
}

export function Button({ className = "", children, ...props }) {
  return (
    <button
      className={mergeClassNames(
        "inline-flex items-center justify-center gap-2 rounded-xl px-4 py-2 text-sm text-white transition duration-300 disabled:cursor-not-allowed disabled:opacity-60",
        className
      )}
      {...props}
    >
      {children}
    </button>
  );
}
