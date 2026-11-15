// Runs the C++ text-to-motion models (compiled to WebAssembly) off the main thread.
// Message in:  {base, method: "ar" | "diffusion", text, seed, steps}
// Message out: {type: "progress", text} | {type: "result", ok, bvh?, ms?, used?, travel?, error?, words?}
type Motion = {
  loadModel(name: string, bytes: Uint8Array): { ok: boolean; error?: string; words: string[] };
  generate(name: string, text: string, seed: number, steps: number):
    { ok: boolean; error?: string; bvh: string; ms: number; used: string[]; travel: number };
};

let module: Promise<Motion> | null = null;
const loaded = new Map<string, string[]>();  // method -> vocabulary

self.onmessage = async (e: MessageEvent) => {
  const { base, method, text, seed, steps } = e.data;
  const post = (m: object) => (self as unknown as Worker).postMessage(m);
  try {
    module ??= import(/* @vite-ignore */ `${base}wasm/motion.mjs`).then((m) => m.default() as Promise<Motion>);
    const motion = await module;
    if (!loaded.has(method)) {
      post({ type: "progress", text: `Downloading the ${method === "ar" ? "AR Transformer" : "diffusion"} weights…` });
      const res = await fetch(`${base}motion/${method}.vcw`);
      if (!res.ok) throw new Error(`weights: HTTP ${res.status}`);
      const info = motion.loadModel(method, new Uint8Array(await res.arrayBuffer()));
      if (!info.ok) throw new Error(info.error);
      loaded.set(method, info.words);
    }
    post({ type: "progress", text: "Generating…" });
    const r = motion.generate(method, text, seed, steps);
    post({ type: "result", ...r, words: loaded.get(method) });
  } catch (err) {
    post({ type: "result", ok: false, error: err instanceof Error ? err.message : String(err) });
  }
};
