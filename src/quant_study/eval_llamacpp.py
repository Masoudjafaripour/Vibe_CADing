"""Quantization study: how much accuracy, GPU memory and speed each llama.cpp quantization costs on
the live-CAD task. Same prompt, JSON schema (llama.cpp turns it into a grammar), repair loop and
checks as src/live_cad/eval_models.py, so the numbers compare directly with the WebLLM q4f16 results.

    python src/quant_study/eval_llamacpp.py --gpu 0 --models /path/Qwen3-4B-Q4_K_M.gguf ...

Needs llama.cpp's llama-server (built with CUDA): LLAMA_BIN=~/.local/src/llama.cpp/build/bin
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

import vibecad_geom as g

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "live_cad"))
import eval_models as ev  # noqa: E402

BIN = os.path.expanduser(os.environ.get("LLAMA_BIN", "~/.local/src/llama.cpp/build/bin"))


def post(port, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def gpu_used_mib(gpu):
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(gpu)],
                         capture_output=True, text=True).stdout
    return int(out.strip())


class Server:
    def __init__(self, model, gpu, port, ctx=4096):
        self.port, self.gpu = port, gpu
        before = gpu_used_mib(gpu)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
        self.log = open(model + f".gpu{gpu}.log", "w")
        self.proc = subprocess.Popen([f"{BIN}/llama-server", "-m", model, "-ngl", "99", "-c", str(ctx), "--port", str(port),
                                      "-np", "1", "--no-webui", "--jinja"], env=env, stdout=self.log, stderr=subprocess.STDOUT)
        for _ in range(300):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                    if json.loads(r.read()).get("status") == "ok":
                        break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("llama-server did not start")
        time.sleep(2)
        self.vram_mib = gpu_used_mib(gpu) - before  # weights + KV cache (ctx) + compute buffers

    def chat(self, msgs, schema):
        r = post(self.port, {"messages": msgs, "temperature": 0, "max_tokens": ev.MAX_TOKENS, "cache_prompt": False,
                             "response_format": {"type": "json_schema", "json_schema": {"name": "spec", "schema": schema}},
                             "chat_template_kwargs": {"enable_thinking": False}})
        return r["choices"][0]["message"]["content"], r.get("timings", {})

    def stop(self):
        self.proc.terminate()
        self.proc.wait(30)
        self.log.close()


def run(model, gpu, port, out_dir):
    name = os.path.basename(model).replace(".gguf", "")
    srv = Server(model, gpu, port)
    schema = ev.PROMPT["schema"]
    rows, timings = [], []
    try:
        for prompt in ev.PROMPT["suggestions"]:
            base = msgs = ev.messages_for(prompt)
            for attempt in range(1, ev.MAX_REPAIRS + 2):
                spec, t = srv.chat(msgs, schema)
                timings.append(t)
                r = g.csg_build(spec, 112)
                if not r["feedback"]:
                    break
                msgs = base + [{"role": "assistant", "content": spec}, {"role": "user", "content": r["feedback"]}]
            rows.append({"prompt": prompt, "spec": spec, "attempts": attempt,
                         "adheres": bool(r["ok"] and ev.ADHERENCE[prompt](ev._parts(r, spec), r))})
        held = []
        for prompt, (family, want) in ev.HELD_OUT.items():
            spec, t = srv.chat(ev.messages_for(prompt), schema)
            timings.append(t)
            r = g.csg_build(spec, 64)
            held.append({"prompt": prompt, "spec": spec, "pass": bool(r["ok"] and ev.held_out_pass(spec, family, want))})
        recorded = json.load(open(os.path.join(HERE, "..", "live_cad", "recorded.json")))["answers"]
        edits = []
        for prompt, instruction, family, want in ev.EDITS:
            spec, t = srv.chat(ev.edit_messages(prompt, recorded[prompt], instruction), schema)
            r = g.csg_build(spec, 64)
            edits.append({"edit": instruction, "spec": spec, "pass": bool(r["ok"] and ev.held_out_pass(spec, family, want))})
    finally:
        srv.stop()
    pp = [t["prompt_per_second"] for t in timings if t.get("prompt_n", 0) > 1000]
    tg = [t["predicted_per_second"] for t in timings if t.get("predicted_n", 0) > 10]
    summary = {"model": name, "file_gb": round(os.path.getsize(model) / 1e9, 2), "vram_gb": round(srv.vram_mib / 1024, 2),
               "suggestions": f"{sum(r['adheres'] for r in rows)}/{len(rows)}", "held_out": f"{sum(h['pass'] for h in held)}/{len(held)}",
               "edits": f"{sum(e['pass'] for e in edits)}/{len(edits)}",
               "prompt_tok_s": round(sum(pp) / len(pp)) if pp else None, "gen_tok_s": round(sum(tg) / len(tg)) if tg else None}
    with open(os.path.join(out_dir, name + ".json"), "w") as f:
        json.dump({"summary": summary, "suggestions": rows, "held_out": held, "edits": edits}, f, indent=2)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for m in a.models:
        run(m, a.gpu, a.port, a.out)
