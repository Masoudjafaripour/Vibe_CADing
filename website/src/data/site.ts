// All content here mirrors the repository's READMEs and recorded runs.
// Keep it factual: no capability or result that the repo doesn't show.

export const base = import.meta.env.BASE_URL.replace(/\/$/, "");
export const asset = (path: string) => `${base}/${path}`;

export const repoUrl = "https://github.com/Masoudjafaripour/Vibe_CADing";
export const treeUrl = (path: string) => `${repoUrl}/tree/main/${path}`;
export const blobUrl = (path: string) => `${repoUrl}/blob/main/${path}`;

export const author = { name: "Masoud Jafaripour", url: "https://github.com/Masoudjafaripour" };

export const bibtex = `@misc{vibecading2025,
  author       = {Jafaripour, Masoud},
  title        = {Vibe-CADing: Conditional CAD Generation and Retrieval
                  for a Text-to-CAD Design Pipeline},
  year         = {2025},
  howpublished = {\\url{https://github.com/Masoudjafaripour/Vibe_CADing}},
}`;

export type Kind = "brep" | "sdf";

export interface GalleryItem {
  title: string;
  image: string;
  kind: Kind;
  method: string;
  author: string; // which model (or a person) wrote the geometry code
  prompt?: string;
  note: string;
  source: string;
}

export const gallery: GalleryItem[] = [
  {
    title: "Dragon",
    image: "media/dragon_gpt4o.gif",
    kind: "sdf",
    method: "GLSL signed distance function",
    author: "GPT-4o",
    prompt: "A stylized dragon made of smooth geometric primitives → “Make the wings larger”",
    note: "Generate and refine each passed on the first attempt. 1.8 KB of GLSL. Crude: floating legs, heart-shaped wings.",
    source: "src/CAD_Harness/sdf/examples/dragon_gpt4o.glsl",
  },
  {
    title: "Mounting plate",
    image: "media/plate_preview.png",
    kind: "brep",
    method: "CadQuery → B-rep → STEP",
    author: "Qwen3-8B",
    prompt: "a 40x20x5mm mounting plate with two 4mm holes 15mm either side of center and 2mm filleted corners",
    note: "Box → face select → hole pattern → edge select → fillet. Also generated through the build123d harness.",
    source: "src/CAD_Harness/cadquery/plate.step",
  },
  {
    title: "Snowman",
    image: "media/snowman.gif",
    kind: "sdf",
    method: "GLSL signed distance function",
    author: "Hand-written reference",
    note: "Three smoothly blended spheres plus coal, carrot, sticks and hat, with per-part colors.",
    source: "src/CAD_Harness/sdf/examples/snowman.glsl",
  },
  {
    title: "Rubber duck",
    image: "media/duck.gif",
    kind: "sdf",
    method: "GLSL signed distance function",
    author: "Hand-written reference",
    note: "An organic shape: ellipsoids and spheres joined with smooth unions.",
    source: "src/CAD_Harness/sdf/examples/duck.glsl",
  },
  {
    title: "Cube",
    image: "media/cube.svg",
    kind: "brep",
    method: "CadQuery → B-rep → STEP (SVG projection)",
    author: "Qwen3-8B",
    prompt: "a cube of size 5",
    note: "Hidden-line SVG export of the generated STEP file.",
    source: "src/CAD_Harness/cadquery/cube.step",
  },
  {
    title: "Extruded square",
    image: "media/brep_cube.png",
    kind: "brep",
    method: "pythonOCC sketch → wire → face → prism",
    author: "Hand-written example",
    note: "The B-rep building blocks used by the LLM → JSON → pythonOCC pipeline, exported to STEP and IGES.",
    source: "src/B-rep/example_B_rep.py",
  },
];

export interface Run {
  harness: string;
  kind: Kind;
  model: string;
  task: string;
  outcome: "pass" | "fail";
  detail: string;
}

// Individual, qualitative runs recorded in the repo. Not a benchmark.
export const runs: Run[] = [
  { harness: "CadQuery", kind: "brep", model: "Qwen3-8B", task: "a cube of size 5", outcome: "pass", detail: "STEP exported" },
  { harness: "CadQuery", kind: "brep", model: "Qwen3-8B", task: "mounting plate, 2 holes, filleted corners", outcome: "pass", detail: "STEP exported, render above" },
  { harness: "build123d", kind: "brep", model: "Qwen3-8B", task: "same mounting plate (builder API)", outcome: "pass", detail: "STEP exported" },
  { harness: "SDF", kind: "sdf", model: "GPT-4o", task: "stylized dragon", outcome: "pass", detail: "1st attempt; crude anatomy" },
  { harness: "SDF", kind: "sdf", model: "GPT-4o", task: "refine: make the wings larger", outcome: "pass", detail: "1st attempt; wings visibly larger" },
  { harness: "SDF", kind: "sdf", model: "Qwen3-8B", task: "stylized dragon", outcome: "fail", detail: "failed all 3 attempts (last: compile error, missing argument)" },
  { harness: "SDF", kind: "sdf", model: "Qwen2.5-Coder-7B", task: "stylized dragon", outcome: "fail", detail: "called undefined functions, omitted sdf()" },
  { harness: "SDF", kind: "sdf", model: "Qwen3-8B", task: "snowman of three blended spheres", outcome: "fail", detail: "failed all 3 attempts (last: shape off-frame, caught by the framing check)" },
];

export type Status = "working" | "prototype" | "stub";

export const modules: { name: string; status: Status; path: string; what: string }[] = [
  { name: "CAD-code harness (CadQuery, build123d)", status: "working", path: "src/CAD_Harness", what: "LLM writes CAD code → subprocess → STEP; tracebacks fed back for retry" },
  { name: "SDF harness (GLSL)", status: "working", path: "src/CAD_Harness/sdf", what: "LLM writes sdf(p) → headless GPU sphere tracer → PNG/GIF; repair and refine" },
  { name: "LLM → JSON → B-rep", status: "working", path: "src/B-rep", what: "Qwen2.5-3B emits a JSON spec; pythonOCC builds the solid" },
  { name: "Text → image → SVG", status: "working", path: "src/pipeline", what: "Stable Diffusion + ControlNet (Canny), then contour tracing to SVG" },
  { name: "PixelCNN 2D CAD grids", status: "working", path: "src/AR", what: "Masked-convolution AR model over 2D token grids (dummy data)" },
  { name: "CLIP retrieval", status: "working", path: "src/CLIP", what: "CLIP ViT-B/32 ranks rendered meshes by text similarity" },
  { name: "Multi-agent loop", status: "prototype", path: "src/Agentic", what: "LangGraph Planner → Retriever → Generator → Critic (outdated APIs, not runnable)" },
  { name: "Streamlit demo", status: "prototype", path: "src/vibe-cading-demo", what: "Voxel viewer UI; sample data not included" },
  { name: "Voxel diffusion / CVAE", status: "stub", path: "src/models", what: "Planned 3D U-Net DDPM and conditional VAE" },
  { name: "B-rep retrieval (RAG)", status: "stub", path: "src/B-rep", what: "Planned topology/geometry embeddings for retrieval" },
  { name: "Validator / retrainer, web app", status: "stub", path: "src/pipeline", what: "Planned manufacturability checks and UI/API" },
];

// ---- Added with the CAD-viewport redesign (numbers from the module READMEs / tests) ----

export const printFixes = [
  { shape: "Chain", before: "10 mm", beforeLabel: "middle link floats above the plate", after: "3 / 3", afterLabel: "links touch the plate", cause: "Links tilted ±45°: still interlocked, and every link now rests on the plate." },
  { shape: "Gyroid", before: "19", beforeLabel: "loose fragments", after: "1", afterLabel: "connected lattice", cause: "A thin sheet became a channel network that thins out before the sphere's surface." },
  { shape: "Rook", before: "23 mm²", beforeLabel: "bed contact on a 40 mm base", after: "1,436 mm²", afterLabel: "flat foot on the plate", cause: "The body's rounded end and smooth blend bulged 1 mm below the foot." },
  { shape: "Vase", before: "clipped", beforeLabel: "wall leaks below the floor", after: "watertight", afterLabel: "closed, inside the box", cause: "The outer wall had no bottom cut." },
];

export const printChecks = [
  ["Floating parts", "Lowest point of every separate shell vs. the build plate", "critical"],
  ["Thin walls", "Exact distance transform + morphological opening, by volume", "critical"],
  ["Watertight · size", "Edge-manifold check; bounding box vs. a 220 × 220 × 250 mm bed", "critical"],
  ["Overhangs", "Faces tilted more than 45° from vertical, off the plate", "supports"],
  ["Bed contact", "Flat area on the plate vs. the part's footprint", "warning"],
];

export const speedups = [
  { step: "Foot locking (two-bone IK)", py: "352 ms", cpp: "1.0 ms", x: 341, area: "motion" },
  { step: "Marching cubes, 128³", py: "3,693 ms", cpp: "16 ms", x: 228, area: "geometry" },
  { step: "SDF sampling, 128³", py: "3,336 ms", cpp: "38 ms", x: 88, area: "geometry" },
  { step: "Stance sliding metric", py: "19 ms", cpp: "0.4 ms", x: 43, area: "motion" },
  { step: "Forward kinematics", py: "299 ms", cpp: "15 ms", x: 19, area: "motion" },
];

export const engineModules = [
  { name: "vibecad_geom", items: ["Marching cubes (SDF → mesh)", "Printability: EDT wall thickness, overhangs, floating parts", "QEM simplification · Taubin smoothing", "STL read / write · glTF export"] },
  { name: "vibecad_motion", items: ["Forward kinematics", "Foot contact + two-bone IK foot locking", "Jerk / sliding metrics", "BVH export (Blender, Maya, Unity)"] },
];

export const newModules: { name: string; status: Status; path: string; what: string }[] = [
  { name: "C++ geometry engine", status: "working", path: "src/geometry_engine", what: "Marching cubes, printability analysis, QEM simplification, STL / glTF export (pybind11)" },
  { name: "C++ motion engine", status: "working", path: "src/geometry_engine", what: "FK, IK foot locking, motion metrics, BVH export; matches NumPy to 1e-12" },
  { name: "Text → motion (AR, DDPM)", status: "working", path: "src/animation", what: "18-joint skeleton animation from text; procedural, AR Transformer and diffusion" },
  { name: "MLflow tracking", status: "working", path: "src/CAD_Harness/sdf", what: "Every LLM attempt, prompt, error and output logged with --track" },
];
