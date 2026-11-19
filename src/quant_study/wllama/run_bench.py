import json, sys
from playwright.sync_api import sync_playwright
CH = "/home/jafaripo/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome"
model, threads, cases = sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "v2,v1"
with sync_playwright() as p:
    b = p.chromium.launch(executable_path=CH)
    pg = b.new_page()
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto(f"http://127.0.0.1:8099/bench.html?model={model}&threads={threads}&cases={cases}")
    pg.wait_for_function("window.__result || false", timeout=1_800_000, polling=2000)
    r = pg.evaluate("window.__result")
    print(json.dumps({"model": model, "threads": r["threads"], "multithread": r["multithread"], "load_s": round(r["load_s"], 1),
                      **{k: {"prompt_tokens": v["prompt_tokens"], "total_s": round(v["total_s"], 1), "text": v["text"][:90],
                             "pp_tok_s": round((v.get("timings") or {}).get("prompt_per_second", 0), 1),
                             "tg_tok_s": round((v.get("timings") or {}).get("predicted_per_second", 0), 1)} for k, v in r["runs"].items()}}), flush=True)
    print("errors", errs[:2])
    b.close()
