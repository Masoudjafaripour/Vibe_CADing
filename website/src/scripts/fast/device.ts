// v2: pick the model for this device. A 4B model needs ~3.4 GB of GPU memory, which integrated
// and small GPUs don't have to spare; they get the 1.7B model (~2 GB). A heuristic: the visitor
// can still choose either in the model menu.
type GpuInfo = { vendor?: string; architecture?: string; description?: string; isFallbackAdapter?: boolean };
type Adapter = { info?: GpuInfo; limits: { maxBufferSize: number; maxStorageBufferBindingSize: number } };

export async function suggestModel(big: string, small: string): Promise<{ id: string; reason: string }> {
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter(o?: object): Promise<Adapter | null> } }).gpu;
  const adapter = await gpu?.requestAdapter({ powerPreference: "high-performance" }).catch(() => null);
  if (!adapter) return { id: small, reason: "no WebGPU adapter details" };
  const info = adapter.info ?? {};
  const name = `${info.vendor ?? ""} ${info.architecture ?? ""} ${info.description ?? ""}`.toLowerCase();
  const memoryGB = (navigator as Navigator & { deviceMemory?: number }).deviceMemory;  // Chrome, capped at 8
  const integrated = /intel|integrated|uhd|iris|mali|adreno|powervr|swiftshader/.test(name) || info.isFallbackAdapter;
  const smallBuffers = adapter.limits.maxStorageBufferBindingSize < 1 << 30;  // < 1 GiB per buffer
  if (integrated) return { id: small, reason: `integrated GPU (${info.vendor || "unknown"})` };
  if (memoryGB !== undefined && memoryGB <= 4) return { id: small, reason: `${memoryGB} GB device memory` };
  if (smallBuffers) return { id: small, reason: "GPU buffer limit under 1 GB" };
  return { id: big, reason: `dedicated GPU (${info.vendor || "unknown"})` };
}
