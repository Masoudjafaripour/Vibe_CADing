// Live text-to-CAD in the browser:
//   prompt -> in-browser LLM (WebLLM, WebGPU) picks a part family and fills its numbers (JSON, constrained by the schema)
//          -> C++ engine expands the family to CSG parts with real proportions
//          -> C++ engine (WebAssembly) builds, meshes and checks it for 3D printing
//          -> problems go back to the LLM (up to MAX_REPAIRS times) -> 3D view + STL
// Same prompt, schema and loop as src/live_cad/eval_models.py, which chose the model.
import prompt from "../../../src/live_cad/prompt.json";

export const MAX_REPAIRS = 2;
// WebLLM builds these models with a 4096-token context: ~2400 for the prompt, so answers are capped
// and a repair sees only the latest attempt and its feedback.
export const MAX_TOKENS = 700;
type Msg = { role: "system" | "user" | "assistant"; content: string };

export interface Engine {
  build(spec: string, resolution: number, minWall: number, targetTriangles: number): BuildResult;
  stl(): Uint8Array;
  specGuide(): string;
}
export interface BuildResult {
  ok: boolean; name: string; verdict: string; feedback: string;
  errors: string[]; warnings: string[]; notes: string[];
  expanded: string;  // a family answer, expanded to parts by the C++ engine ("" for custom parts)
  positions?: Float32Array; indices?: Uint32Array; colors?: Float32Array;
  triangles: number; watertight: boolean; shells: number; genus: number; volume: number;
  size: number[]; overhang: number; thin: number; ms: { sample: number; mesh: number; check: number };
}

let enginePromise: Promise<Engine> | null = null;
export function loadGeometryEngine(base: string): Promise<Engine> {
  enginePromise ??= import(/* @vite-ignore */ `${base}wasm/vibecad.mjs`).then((m) => m.default() as Promise<Engine>);
  return enginePromise;
}

export function messagesFor(engine: Engine, request: string): Msg[] {
  const msgs: Msg[] = [{ role: "system", content: prompt.system.replace("{{SPEC_GUIDE}}", engine.specGuide()) }];
  for (const ex of prompt.examples) {
    msgs.push({ role: "user", content: ex.prompt }, { role: "assistant", content: JSON.stringify(ex.spec) });
  }
  msgs.push({ role: "user", content: request });
  return msgs;
}

// WebLLM's attention kernels bind 10 storage buffers per shader; some browsers and GPUs allow fewer.
const NEEDED_STORAGE_BUFFERS = 10;

type Adapter = { features: Set<string>; limits: { maxStorageBuffersPerShaderStage: number } };
export function webgpuStatus(): Promise<{ ok: boolean; f16: boolean; reason?: string }> {
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter(o?: object): Promise<Adapter | null> } }).gpu;
  if (!gpu) return Promise.resolve({ ok: false, f16: false, reason: "This browser has no WebGPU (try desktop Chrome or Edge)." });
  return gpu.requestAdapter({ powerPreference: "high-performance" }).then((a) => {
    if (!a) return { ok: false, f16: false, reason: "No WebGPU adapter was found." };
    const buffers = a.limits.maxStorageBuffersPerShaderStage;
    if (buffers < NEEDED_STORAGE_BUFFERS) {
      return { ok: false, f16: false, reason: `This browser's WebGPU allows ${buffers} storage buffers per shader; ` +
        `the in-browser models need ${NEEDED_STORAGE_BUFFERS} (desktop Chrome or Edge usually works).` };
    }
    return { ok: true, f16: a.features.has("shader-f16") };
  }).catch(() => ({ ok: false, f16: false, reason: "WebGPU could not be started." }));
}

// ---- LLM (WebLLM in a worker) ----
type ChatEngine = {
  chat: { completions: { create(req: object): Promise<AsyncIterable<{ choices: { delta: { content?: string } }[] }>> } };
};
let llm: ChatEngine | null = null;
let llmModel = "";

// Browser tests set window.__vibecadTestLLM to stand in for the model (inference needs a real GPU).
type TestLLM = (messages: Msg[]) => Promise<string>;
const testLLM = (): TestLLM | undefined => (globalThis as { __vibecadTestLLM?: TestLLM }).__vibecadTestLLM;

export async function loadModel(modelId: string, onProgress: (fraction: number, text: string) => void) {
  if (testLLM()) return onProgress(1, "test model");
  if (llm && llmModel === modelId) return;
  const { CreateWebWorkerMLCEngine } = await import("@mlc-ai/web-llm");
  const worker = new Worker(new URL("./llm-worker.ts", import.meta.url), { type: "module" });
  llm = (await CreateWebWorkerMLCEngine(worker, modelId, {
    initProgressCallback: (r: { progress: number; text: string }) => onProgress(r.progress, r.text),
  })) as unknown as ChatEngine;
  llmModel = modelId;
}

async function complete(messages: Msg[], onToken: (text: string) => void): Promise<string> {
  const test = testLLM();
  if (test) return test(messages).then((t) => (onToken(t), t));
  if (!llm) throw new Error("model not loaded");
  const stream = await llm.chat.completions.create({
    messages,
    stream: true,
    temperature: 0,
    max_tokens: MAX_TOKENS,
    response_format: { type: "json_object", schema: JSON.stringify(prompt.schema) },
    extra_body: { enable_thinking: false },  // Qwen3: answer directly
  });
  let text = "";
  for await (const chunk of stream) {
    text += chunk.choices[0]?.delta?.content ?? "";
    onToken(text);
  }
  return text;
}

export interface Attempt { spec: string; result: BuildResult; seconds: number }

// The repair loop: generate, build, and feed problems back until the part is printable.
export async function generatePart(engine: Engine, request: string, hooks: {
  onAttempt(n: number): void;
  onToken(text: string): void;
  onResult(a: Attempt, n: number): void;
}, resolution = 112): Promise<Attempt[]> {
  const attempts: Attempt[] = [];
  const base = messagesFor(engine, request);
  let messages = base;
  for (let n = 1; n <= MAX_REPAIRS + 1; n++) {
    hooks.onAttempt(n);
    const t0 = performance.now();
    const spec = await complete(messages, hooks.onToken);
    const result = engine.build(spec, resolution, 1.0, 60000);
    const a = { spec, result, seconds: (performance.now() - t0) / 1000 };
    attempts.push(a);
    hooks.onResult(a, n);
    if (!result.feedback) break;
    messages = [...base, { role: "assistant", content: spec }, { role: "user", content: result.feedback }];
  }
  return attempts;
}
