// v2 ("fast") live text-to-CAD. Same idea as ../live-cad.ts (v1, unchanged), tuned for less waiting
// and less GPU memory; design and measurements in src/live_cad_fast/README.md.
//
//   request ─► C++ select_families (keywords)          ─► short prompt (~0.9k tokens, not ~2.9k)
//           ─► WebLLM, context 2048 (smaller KV cache)  ─► {"name", "family", "params"}
//           ─► C++ engine in a Web Worker (page never freezes) ─► viewer (paused while the model runs)
import { CreateWebWorkerMLCEngine, hasModelInCache } from "@mlc-ai/web-llm";
import fast from "../../../../src/live_cad_fast/prompt_fast.json";
import v1 from "../../../../src/live_cad/prompt.json";
import type { BuildResult } from "../live-cad";

export const MAX_REPAIRS = 2;
export const CONTEXT = 2048;
const MAX_TOKENS = 300;  // family answers are ~30 to 80 tokens
type Msg = { role: "system" | "user" | "assistant"; content: string };

// ---- the C++ engine in a worker ----
let worker: Worker | null = null;
let seq = 0;
const pending = new Map<number, { resolve(v: unknown): void; reject(e: Error): void }>();
function call<T>(base: string, op: string, ...args: unknown[]): Promise<T> {
  if (!worker) {
    worker = new Worker(new URL("./geometry-worker.ts", import.meta.url), { type: "module" });
    worker.onmessage = (e) => {
      const p = pending.get(e.data.id);
      pending.delete(e.data.id);
      if (e.data.ok) p?.resolve(e.data.value);
      else p?.reject(new Error(e.data.error));
    };
  }
  const id = ++seq;
  return new Promise<T>((resolve, reject) => {
    pending.set(id, { resolve: resolve as (v: unknown) => void, reject });
    worker!.postMessage({ id, op, args, base });
  });
}
export const buildInWorker = (base: string, spec: string) => call<BuildResult>(base, "build", spec, 112, 1.0, 60000);
export const stlFromWorker = (base: string) => call<Uint8Array>(base, "stl");

// ---- the short, per-request prompt ----
// The v1 schema's branch for each offered family. "name" stays (keep_name): writing it before the
// numbers helps the model place them (held-out 10/12 with it, 9/12 without; see the README).
const keepName = (fast as { keep_name?: boolean }).keep_name ?? true;
const branches = (families: string[]) => (v1.schema.anyOf as { properties: Record<string, unknown> & { family: { enum: string[] } }; required: string[] }[])
  .filter((b) => families.includes(b.properties.family.enum[0]))
  .map((b) => keepName ? b : ({ ...b, properties: Object.fromEntries(Object.entries(b.properties).filter(([k]) => k !== "name")),
                                required: b.required.filter((r) => r !== "name") }));

export async function fastMessages(base: string, request: string, always = ""): Promise<{ messages: Msg[]; families: string[] }> {
  const picked = await call<string[]>(base, "selectFamilies", request, fast.k, always);
  const shown = [...new Set([...picked, ...fast.examples.flatMap((e) => e.families)])];
  const guide = await call<string>(base, "familiesGuide", shown.join(","));
  const messages: Msg[] = [{ role: "system", content: fast.system.replace("{{GUIDE}}", guide) }];
  for (const ex of fast.examples) messages.push({ role: "user", content: ex.prompt }, { role: "assistant", content: JSON.stringify(ex.spec) });
  messages.push({ role: "user", content: request });
  return { messages, families: shown };
}

// ---- the model: its own engine with a smaller context window ----
type ChatEngine = {
  chat: { completions: { create(req: object): Promise<AsyncIterable<{ choices: { delta: { content?: string } }[]; usage?: Usage }>> } };
  unload(): Promise<void>;
};
export type Usage = { prompt_tokens: number; completion_tokens: number; extra?: { time_to_first_token_s?: number; prefill_tokens_per_s?: number; decode_tokens_per_s?: number } };
let llm: ChatEngine | null = null, llmModel = "", loading: Promise<void> | null = null;
type TestLLM = (messages: Msg[]) => Promise<string>;
const testLLM = (): TestLLM | undefined => (globalThis as { __vibecadTestLLM?: TestLLM }).__vibecadTestLLM;

const gpuBusy = (busy: boolean) => window.dispatchEvent(new CustomEvent("vibecad-gpu-busy", { detail: busy }));

export async function loadFastModel(modelId: string, onProgress: (f: number, t: string) => void) {
  if (testLLM()) return onProgress(1, "test model");
  if (llm && llmModel === modelId) return;
  if (loading) await loading;
  if (llm && llmModel === modelId) return;
  if (llm) await unloadFastModel();
  loading = (async () => {
    const w = new Worker(new URL("../llm-worker.ts", import.meta.url), { type: "module" });
    llm = (await CreateWebWorkerMLCEngine(w, modelId, { initProgressCallback: (r: { progress: number; text: string }) => onProgress(r.progress, r.text) },
                                          { context_window_size: CONTEXT })) as unknown as ChatEngine;
    llmModel = modelId;
  })();
  try { await loading; } finally { loading = null; }
}
export async function unloadFastModel() {
  const e = llm;
  llm = null, llmModel = "";
  if (e) await e.unload();
}
export const fastModelLoaded = () => llm !== null;

// Warm-up on intent: load the model while the visitor types, but only if it is already in the
// browser cache (never start a 1 to 2 GB download that nobody asked for).
export async function warmUp(modelId: string, onProgress: (f: number, t: string) => void): Promise<boolean> {
  if (testLLM() || (llm && llmModel === modelId)) return true;
  if (!(await hasModelInCache(modelId).catch(() => false))) return false;
  await loadFastModel(modelId, onProgress);
  return true;
}

async function complete(messages: Msg[], schema: object, onToken: (t: string) => void): Promise<{ text: string; usage?: Usage }> {
  const test = testLLM();
  if (test) return test(messages).then((t) => (onToken(t), { text: t }));
  if (!llm) throw new Error("model not loaded");
  gpuBusy(true);
  try {
    const stream = await llm.chat.completions.create({
      messages, stream: true, stream_options: { include_usage: true }, temperature: 0, max_tokens: MAX_TOKENS,
      response_format: { type: "json_object", schema: JSON.stringify(schema) }, extra_body: { enable_thinking: false },
    });
    let text = "", usage: Usage | undefined;
    for await (const chunk of stream) {
      text += chunk.choices[0]?.delta?.content ?? "";
      if (chunk.usage) usage = chunk.usage;
      onToken(text);
    }
    return { text, usage };
  } finally {
    gpuBusy(false);
  }
}

export interface FastAttempt { spec: string; result: BuildResult; seconds: number; usage?: Usage; families: string[] }
export interface FastEdit { prompt: string; spec: string }

// generate -> build (worker) -> feed problems back, as v1; `edit` changes an existing part.
export async function generatePartFast(base: string, request: string, hooks: {
  onAttempt(n: number): void;
  onToken(text: string): void;
  onResult(a: FastAttempt, n: number): void;
}, edit?: FastEdit): Promise<FastAttempt[]> {
  let first: { messages: Msg[]; families: string[] };
  if (edit) {
    const current = JSON.parse(edit.spec);
    if (!keepName) delete current.name;
    const f = await fastMessages(base, edit.prompt, current.family ?? "");
    first = { families: f.families, messages: [...f.messages, { role: "assistant", content: JSON.stringify(current) },
      { role: "user", content: fast.edit.replace("{{INSTRUCTION}}", request) }] };
  } else {
    first = await fastMessages(base, request);
  }
  const schema = { anyOf: branches(first.families) };
  const attempts: FastAttempt[] = [];
  let messages = first.messages;
  for (let n = 1; n <= MAX_REPAIRS + 1; n++) {
    hooks.onAttempt(n);
    const t0 = performance.now();
    const { text, usage } = await complete(messages, schema, hooks.onToken);
    const result = await buildInWorker(base, text);
    const a = { spec: text, result, seconds: (performance.now() - t0) / 1000, usage, families: first.families };
    attempts.push(a);
    hooks.onResult(a, n);
    if (!result.feedback) break;
    messages = [...first.messages, { role: "assistant", content: text }, { role: "user", content: result.feedback }];
  }
  return attempts;
}
