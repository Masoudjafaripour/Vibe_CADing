// "Your parts": every good part and assembly, saved in this browser (localStorage).
// Nothing leaves the device. Export / import moves the library to another browser as a JSON file.
export interface LibraryItem {
  id: string;
  kind: "part" | "assembly";
  name: string;
  prompt: string;                         // what was asked
  spec: string;                           // part spec, or the assembly's mates
  parts?: Record<string, { name: string; spec: string }>;  // assembly only: the parts it uses
  created: number;
}

const KEY = "vibecad-library-v1";
const MAX = 60;

export function loadLibrary(): LibraryItem[] {
  try {
    const items = JSON.parse(localStorage.getItem(KEY) ?? "[]");
    return Array.isArray(items) ? items.filter((i) => i && typeof i.spec === "string" && (i.kind === "part" || i.kind === "assembly")) : [];
  } catch {
    return [];
  }
}

function save(items: LibraryItem[]) {
  try {
    localStorage.setItem(KEY, JSON.stringify(items.slice(0, MAX)));
  } catch {
    // storage full or blocked: the library just isn't kept
  }
}

const same = (a: Pick<LibraryItem, "kind" | "spec" | "parts">, b: Pick<LibraryItem, "kind" | "spec" | "parts">) =>
  a.kind === b.kind && a.spec === b.spec && JSON.stringify(a.parts ?? null) === JSON.stringify(b.parts ?? null);

// Newest first; an identical item moves to the top instead of being added twice.
export function addToLibrary(item: Omit<LibraryItem, "id" | "created">): LibraryItem {
  const items = loadLibrary().filter((i) => !same(i, item));
  const saved = { ...item, id: Math.random().toString(36).slice(2, 10), created: Date.now() };
  save([saved, ...items]);
  return saved;
}

export function removeFromLibrary(id: string) {
  save(loadLibrary().filter((i) => i.id !== id));
}

export function clearLibrary() {
  save([]);
}

export function exportLibrary(): Blob {
  return new Blob([JSON.stringify({ vibecad_library: 1, items: loadLibrary() }, null, 2)], { type: "application/json" });
}

// Adds the items of an exported file; returns how many were new.
export function importLibrary(text: string): number {
  const data = JSON.parse(text);
  const incoming: LibraryItem[] = Array.isArray(data?.items) ? data.items : [];
  const items = loadLibrary();
  let added = 0;
  for (const i of incoming) {
    if (!i || typeof i.spec !== "string" || (i.kind !== "part" && i.kind !== "assembly") || items.some((j) => same(i, j))) continue;
    items.push({ ...i, id: Math.random().toString(36).slice(2, 10), name: String(i.name ?? "part").slice(0, 80) });
    added++;
  }
  save(items.sort((a, b) => b.created - a.created));
  return added;
}
