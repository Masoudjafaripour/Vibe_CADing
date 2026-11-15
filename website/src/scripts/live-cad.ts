// Live text-to-CAD in the browser:
//   prompt -> in-browser LLM (WebLLM, WebGPU) picks a part family and fills its numbers (JSON, constrained by the schema)
//          -> C++ engine expands the family to CSG parts with real proportions
//          -> C++ engine (WebAssembly) builds, meshes and checks it for 3D printing
//          -> problems go back to the LLM (up to MAX_REPAIRS times) -> 3D view + STL
// Same prompt, schema and loop as src/live_cad/eval_models.py, which chose the model.
import prompt from "../../../src/live_cad/prompt.json";
import assemblyPrompt from "../../../src/live_cad/assembly_prompt.json";

export const MAX_REPAIRS = 2;
// WebLLM builds these models with a 4096-token context: ~2900 for the prompt, so answers are capped
// and a repair sees only the latest attempt and its feedback.
export const MAX_TOKENS = 500;
type Msg = { role: "system" | "user" | "assistant"; content: string };

export interface Engine {
  build(spec: string, resolution: number, minWall: number, targetTriangles: number): BuildResult;
  stl(): Uint8Array;
  specGuide(): string;
  assemble(parts: string, mates: string, resolution: number): AssemblyResult;
  assemblyStl(): Uint8Array;
  describeParts(parts: string): string;
  assemblyGuide(): string;
  partPorts(spec: string): string[];
}
export interface AssemblyResult {
  ok: boolean; name: string; feedback: string; errors: string[]; notes: string[];
  positions?: Float32Array; indices?: Uint32Array; colors?: Float32Array;
  placed: { id: string; of: string; name: string }[]; overlaps: number; triangles: number; ms: number;
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
  unload(): Promise<void>;
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

// Frees the model's GPU memory. The weights stay in the browser cache: loading again needs no download.
export async function unloadModel() {
  const engine = llm;
  llm = null;
  llmModel = "";
  if (engine) await engine.unload();
}
export const modelLoaded = () => llm !== null;

async function complete(messages: Msg[], onToken: (text: string) => void, schema: object = prompt.schema): Promise<string> {
  const test = testLLM();
  if (test) return test(messages).then((t) => (onToken(t), t));
  if (!llm) throw new Error("model not loaded");
  const stream = await llm.chat.completions.create({
    messages,
    stream: true,
    temperature: 0,
    max_tokens: MAX_TOKENS,
    response_format: { type: "json_object", schema: JSON.stringify(schema) },
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

// ---- assembly: the same model writes mates between the parts made above ----
// Same prompt, schema and loop as src/live_cad/eval_assembly.py.
export type PartSpecs = Record<string, string>;  // id -> spec JSON

function assemblyUser(engine: Engine, parts: PartSpecs, instruction: string) {
  return assemblyPrompt.user.replace("{{PARTS}}", engine.describeParts(JSON.stringify(parts))).replace("{{INSTRUCTION}}", instruction);
}

export function assemblyMessages(engine: Engine, parts: PartSpecs, instruction: string): Msg[] {
  const msgs: Msg[] = [{ role: "system", content: assemblyPrompt.system.replace("{{ASSEMBLY_GUIDE}}", engine.assemblyGuide()) }];
  for (const ex of assemblyPrompt.examples) {
    msgs.push({ role: "user", content: assemblyUser(engine, ex.parts, ex.instruction) },
              { role: "assistant", content: JSON.stringify(ex.mates) });
  }
  msgs.push({ role: "user", content: assemblyUser(engine, parts, instruction) });
  return msgs;
}

// Only the real part ids and port names can be written (constrained decoding).
export function assemblySchema(engine: Engine, parts: PartSpecs) {
  const ids = Object.keys(parts).sort();
  const ports = [...new Set(Object.values(parts).flatMap((s) => engine.partPorts(s)))].sort();
  const mate = {
    type: "object",
    properties: {
      part: { type: "string", enum: ids }, port: { type: "string", enum: ports },
      to: { type: "string", enum: ids }, to_port: { type: "string", enum: ports },
      offset: { type: "number" }, angle: { type: "number" }, each: { type: "boolean" },
    },
    required: ["part", "port", "to", "to_port"], additionalProperties: false,
  };
  return {
    type: "object",
    properties: { name: { type: "string" }, mates: { type: "array", minItems: 1, maxItems: 12, items: mate } },
    required: ["name", "mates"], additionalProperties: false,
  };
}

export interface AssemblyAttempt { mates: string; result: AssemblyResult; seconds: number }

export async function generateAssembly(engine: Engine, parts: PartSpecs, instruction: string, hooks: {
  onAttempt(n: number): void;
  onToken(text: string): void;
  onResult(a: AssemblyAttempt, n: number): void;
}, resolution = 80): Promise<AssemblyAttempt[]> {
  const attempts: AssemblyAttempt[] = [];
  const base = assemblyMessages(engine, parts, instruction);
  const schema = assemblySchema(engine, parts);
  let messages = base;
  for (let n = 1; n <= MAX_REPAIRS + 1; n++) {
    hooks.onAttempt(n);
    const t0 = performance.now();
    const mates = await complete(messages, hooks.onToken, schema);
    const result = engine.assemble(JSON.stringify(parts), mates, resolution);
    const a = { mates, result, seconds: (performance.now() - t0) / 1000 };
    attempts.push(a);
    hooks.onResult(a, n);
    if (!result.feedback) break;
    messages = [...base, { role: "assistant", content: mates }, { role: "user", content: result.feedback }];
  }
  return attempts;
}
