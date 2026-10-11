# Vibe-CADing website

Project page for the repo, built with Astro + Tailwind CSS and served from
GitHub Pages at <https://masoudjafaripour.github.io/Vibe_CADing/>.

## Develop

Needs Node 22.12 or newer.

```bash
cd website
npm install
npm run dev       # http://localhost:4321/Vibe_CADing/
npm run build     # static site in dist/
npm run preview   # serve dist/ locally
```

`npm run dev` and `npm run build` first run `scripts/sync-assets.mjs`, which
copies the repo's own renders (SDF GIFs, CAD previews, the PDF report) into
`public/media/`. That folder is generated and git-ignored, so the site always
shows the current results.

## How the 3D viewers work

- **Live SDF:** `src/scripts/sdf-viewer.ts` imports the harness's
  `sdf_lib.glsl` and `examples/*.glsl` directly and compiles them in a WebGL2
  port of the Python sphere tracer (`src/shaders/viewer.frag.glsl`). New
  example shapes appear once they're added to `SHAPES`.
- **B-rep plate:** `public/models/plate.json` is tessellated from
  `src/CAD_Harness/cadquery/plate.step` (faces + B-rep edges), shown with
  three.js. Regenerate it after changing the STEP file:

  ```bash
  pip install cadquery
  python website/scripts/step_to_mesh.py src/CAD_Harness/cadquery/plate.step website/public/models/plate.json
  ```

## Content

Text, gallery captions, recorded runs and the module status table live in
`src/data/site.ts`. Keep it factual: only list results the repo actually contains.

## Deploy

`.github/workflows/website.yml` builds and deploys on every push to `main`
that touches the site or its source assets (or when you run the workflow by
hand). One-time setup: **Settings → Pages → Source: GitHub Actions**.
