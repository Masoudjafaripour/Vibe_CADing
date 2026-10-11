// Copies existing renders from the repo into public/media so the site
// always shows the repo's current results. Runs before dev and build.
import { copyFileSync, mkdirSync, existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const repo = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const out = resolve(repo, "website/public/media");

const files = {
  "dragon_gpt4o.gif": "src/CAD_Harness/sdf/assets/dragon_gpt4o.gif",
  "snowman.gif": "src/CAD_Harness/sdf/assets/snowman.gif",
  "snowman.png": "src/CAD_Harness/sdf/assets/snowman.png",
  "duck.gif": "src/CAD_Harness/sdf/assets/duck.gif",
  "duck.png": "src/CAD_Harness/sdf/assets/duck.png",
  "plate_preview.png": "src/CAD_Harness/cadquery/plate_preview.png",
  "cube.svg": "src/CAD_Harness/cadquery/cube.svg",
  "brep_cube.png": "src/B-rep/results/B-rep_cub2step_igs.png",
  "vibe_cading_report.pdf":
    "doc/Vibe_CADing__Conditional_CAD_Generation_and_Retrieval_for_a_Text_to_CAD_Design_Pipeline.pdf",
};

mkdirSync(out, { recursive: true });
let missing = 0;
for (const [name, src] of Object.entries(files)) {
  const from = join(repo, src);
  if (!existsSync(from)) {
    console.error(`sync-assets: missing ${src}`);
    missing++;
    continue;
  }
  copyFileSync(from, join(out, name));
}
if (missing) process.exit(1);
console.log(`sync-assets: copied ${Object.keys(files).length} files to public/media`);
