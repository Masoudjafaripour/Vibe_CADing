"""v2 (fast) vs v1: the same prompts, checks and repair loop as src/live_cad/eval_models.py,
with the short per-request prompt and schema. Reports accuracy and prompt length.

    EVAL_Q4=1 python src/live_cad_fast/eval_fast.py Qwen/Qwen3-4B Qwen/Qwen3-1.7B
"""
import json
import os
import sys
import time

import torch
import vibecad_geom as g

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "live_cad"))
import eval_models as v1  # noqa: E402
from fast_prompt import edit_messages, messages, schema  # noqa: E402


def prompt_tokens(model, msgs):
    kw = {"enable_thinking": False} if "Qwen3" in model.name else {}
    return len(model.tok(model.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw))["input_ids"])


def run(model_name, out_dir):
    m = v1.Model(model_name)
    grammars = {}

    def grammar(families):
        key = tuple(sorted(families))
        if key not in grammars:
            grammars[key] = m.compiler.compile_json_schema(json.dumps(schema(families)))
        return grammars[key]

    rows, held, edits, tokens_v1, tokens_v2 = [], [], [], [], []
    for prompt in v1.PROMPT["suggestions"]:
        base, fams = messages(prompt)
        tokens_v2.append(prompt_tokens(m, base)), tokens_v1.append(prompt_tokens(m, v1.messages_for(prompt)))
        msgs, secs = base, 0.0
        for attempt in range(1, v1.MAX_REPAIRS + 2):
            spec, n, dt = m.generate(msgs, grammar(fams))
            secs += dt
            r = g.csg_build(spec, 112)
            if not r["feedback"]:
                break
            msgs = base + [{"role": "assistant", "content": spec}, {"role": "user", "content": r["feedback"]}]
        rows.append({"prompt": prompt, "families": fams, "spec": spec, "attempts": attempt, "seconds": round(secs, 1),
                     "verdict": r["verdict"], "adheres": bool(r["ok"] and v1.ADHERENCE[prompt](v1._parts(r, spec), r)),
                     "first_try": attempt == 1 and not r["feedback"]})
        print(f"  {prompt[:44]:44s} {r['verdict']:24s} adheres {rows[-1]['adheres']}  families {fams[:4]}", flush=True)
    for prompt, (family, want) in v1.HELD_OUT.items():
        base, fams = messages(prompt)
        spec, n, dt = m.generate(base, grammar(fams))
        r = g.csg_build(spec, 64)
        held.append({"prompt": prompt, "spec": spec, "pass": bool(r["ok"] and v1.held_out_pass(spec, family, want))})
    recorded = json.load(open(os.path.join(HERE, "..", "live_cad", "recorded.json")))["answers"]
    for prompt, instruction, family, want in v1.EDITS:
        msgs, fams = edit_messages(prompt, recorded[prompt], instruction)
        spec, n, dt = m.generate(msgs, grammar(fams))
        r = g.csg_build(spec, 64)
        edits.append({"prompt": prompt, "edit": instruction, "spec": spec, "pass": bool(r["ok"] and v1.held_out_pass(spec, family, want))})
    del m
    torch.cuda.empty_cache()
    q4 = "-q4" if v1.Q4 else ""
    summary = {"model": model_name + q4, "suggestions": f"{sum(r['adheres'] for r in rows)}/{len(rows)}",
               "first_try": f"{sum(r['first_try'] for r in rows)}/{len(rows)}",
               "held_out": f"{sum(h['pass'] for h in held)}/{len(held)}", "edits": f"{sum(e['pass'] for e in edits)}/{len(edits)}",
               "prompt_tokens_v1": round(sum(tokens_v1) / len(tokens_v1)), "prompt_tokens_v2": round(sum(tokens_v2) / len(tokens_v2))}
    with open(os.path.join(out_dir, "fast__" + model_name.replace("/", "__") + q4 + ".json"), "w") as f:
        json.dump({"summary": summary, "suggestions": rows, "held_out": held, "edits": edits}, f, indent=2)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    out = os.environ.get("EVAL_OUT", os.path.join(HERE, "eval_results"))
    os.makedirs(out, exist_ok=True)
    for name in sys.argv[1:]:
        print(f"== {name}")
        run(name, out)
