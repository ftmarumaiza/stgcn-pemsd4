/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        base: {
          bg: "#0B1120",
          card: "rgba(30, 41, 59, 0.6)",
          glow: "#22D3EE",
          primary: "#38BDF8",
          success: "#10B981",
          danger: "#EF4444",
        },
      },
    },
  },
  plugins: [],
};
