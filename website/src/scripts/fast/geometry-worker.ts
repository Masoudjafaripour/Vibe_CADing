// v2: the C++ geometry engine (WebAssembly) in a worker, so a 0.3 to 0.5 s build never freezes the page.
// Messages in: {id, op, args}; out: {id, ok, value | error}. Mesh arrays are transferred, not copied.
type Engine = Record<string, (...args: unknown[]) => unknown>;
let engine: Promise<Engine> | null = null;

self.onmessage = async (e: MessageEvent) => {
  const { id, op, args, base } = e.data;
  const reply = (m: object, transfer: Transferable[] = []) => (self as unknown as Worker).postMessage({ id, ...m }, transfer);
  try {
    engine ??= import(/* @vite-ignore */ `${base}wasm/vibecad.mjs`).then((m) => m.default() as Promise<Engine>);
    const value = (await engine)[op](...args) as Record<string, unknown>;
    const transfer = ArrayBuffer.isView(value) ? [value.buffer as ArrayBuffer]
      : value && typeof value === "object"
        ? Object.values(value).filter((v): v is ArrayBufferView => ArrayBuffer.isView(v)).map((v) => v.buffer as ArrayBuffer)
        : [];
    reply({ ok: true, value }, transfer);
  } catch (err) {
    reply({ ok: false, error: err instanceof Error ? err.message : String(err) });
  }
};
