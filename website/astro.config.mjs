import { defineConfig } from "astro/config";
import tailwindcss from "@tailwindcss/vite";

// Served from GitHub Pages at https://masoudjafaripour.github.io/Vibe_CADing/
export default defineConfig({
  site: "https://masoudjafaripour.github.io",
  base: "/Vibe_CADing",
  vite: {
    plugins: [tailwindcss()],
    // The live SDF viewer imports the repo's .glsl files directly (?raw)
    server: { fs: { allow: [".."] } },
    // three.js (~550 KB) is a lazy chunk, loaded only when the B-rep viewer scrolls into view
    build: { chunkSizeWarningLimit: 700 },
  },
});
