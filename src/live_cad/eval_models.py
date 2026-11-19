"""Which open model should run in the browser? Same prompt, same JSON schema
(XGrammar, the constraint engine inside WebLLM), same repair loop as the
website, scored by the C++ engine (vibecad_geom.csg_build).

    python src/live_cad/eval_models.py Qwen/Qwen2.5-Coder-1.5B-Instruct Qwen/Qwen3-1.7B ...

Runs the Hugging Face bf16 weights on a GPU; EVAL_Q4=1 first rounds them the way
WebLLM's q4f16_1 builds do (4-bit, groups of 32), which is what the browser runs.
"""
import json
import os
import sys
import time

import torch
import vibecad_geom as g
import xgrammar as xgr
from transformers import AutoModelForCausalLM, AutoTokenizer
from xgrammar.contrib.hf import LogitsProcessor

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "prompt.json")) as f:
    PROMPT = json.load(f)
SYSTEM = PROMPT["system"].replace("{{SPEC_GUIDE}}", g.spec_guide())
MAX_REPAIRS = 2
Q4 = os.environ.get("EVAL_Q4") == "1"  # simulate the browser's 4-bit weights
Q3 = os.environ.get("EVAL_Q3") == "1"  # simulate MLC's q3f16_1 (3-bit, groups of 40)
MAX_TOKENS = 500  # WebLLM's builds have a 4096-token context; the prompt takes ~2900


def messages_for(prompt):
    msgs = [{"role": "system", "content": SYSTEM}]
    for ex in PROMPT["examples"]:
        msgs += [{"role": "user", "content": ex["prompt"]}, {"role": "assistant", "content": json.dumps(ex["spec"])}]
    return msgs + [{"role": "user", "content": prompt}]


def _parts(r, spec=""):
    """The parts the engine actually built (a family is expanded to parts in C++)."""
    try:
        return json.loads(r["expanded"] or spec[spec.index("{"):])["parts"]
    except Exception:
        return []


def _rep(parts, op, n_min, n_max=99):
    return any(p.get("op", "add") == op and n_min <= int(p.get("repeat", 1)) <= n_max for p in parts)


def _count(parts, op, n):
    """n copies of one feature: a repeat of n, or n separate parts of the same shape and size."""
    if _rep(parts, op, n, n):
        return True
    groups = {}
    for p in parts:
        if p.get("op", "add") == op and int(p.get("repeat", 1)) == 1:
            k = (p["shape"], tuple(p["size"]))
            groups[k] = groups.get(k, 0) + 1
    return n in groups.values()


def _has(parts, shape, op=None):
    return any(p["shape"] == shape and (op is None or p.get("op", "add") == op) for p in parts)


# Does the part look like what was asked? Checked on the built parts and the C++ mesh stats,
# with the exact counts from the prompt.
ADHERENCE = {
    "A pipe flange with six bolt holes": lambda ps, r: _count(ps, "cut", 6) and r["genus"] in (6, 7),
    "A V-belt pulley with a center bore": lambda ps, r: r["genus"] >= 1 and (_has(ps, "torus", "cut") or _has(ps, "cone")),
    "A knurled thumb knob with 12 grip ridges": lambda ps, r: _count(ps, "cut", 12) or _count(ps, "add", 12),
    "A wall hook with two screw holes": lambda ps, r: r["genus"] == 2 and r["size_mm"][1] > 15,
    "A desk cable organizer with three round slots": lambda ps, r: _count(ps, "cut", 3) and r["shells"] == 1,
    "A hand wheel with five spokes": lambda ps, r: _count(ps, "add", 5) and r["genus"] in (5, 6) and _has(ps, "torus"),
    "An M12 hex nut": lambda ps, r: r["genus"] == 1 and any(
        p["shape"] == "prism" and round(p["size"][-1]) == 6 and 16 <= p["size"][0] <= 20 for p in ps)
        and any(p["shape"] == "cylinder" and p.get("op") == "cut" and 11 <= p["size"][0] <= 13 for p in ps),
    "A five-lobed star knob with a hex socket": lambda ps, r: _count(ps, "add", 5) and _has(ps, "prism", "cut"),
    "A spur gear with 24 teeth and an 8 mm bore": lambda ps, r: _count(ps, "add", 24) and r["genus"] == 1 and any(
        p["shape"] == "cylinder" and p.get("op") == "cut" and abs(p["size"][0] - 8) < 0.5 for p in ps),
    "An L bracket with three screw holes per leg": lambda ps, r: r["genus"] == 6 and r["size_mm"][1] > 15,
    "An M8 x 30 hex bolt": lambda ps, r: any(p["shape"] == "thread" and abs(p["size"][0] - 8) < 0.5 and abs(p["size"][2] - 1.25) < 0.01
                                             for p in ps) and 34 <= r["size_mm"][2] <= 37,
    "A 608 skate bearing": lambda ps, r: r["genus"] == 1 and abs(r["size_mm"][0] - 22) < 1 and abs(r["size_mm"][2] - 7) < 0.6,
}


def _family(spec):
    try:
        return json.loads(spec[spec.index("{"):]).get("family", "")
    except Exception:
        return ""


# Held-out prompts (not shown on the website): other wording and numbers. A pass needs the
# right family and the stated numbers in params.
HELD_OUT = {
    "Flange, 100 mm across, eight bolt holes, 40 mm bore": ("flange", {"count": 8, "diameter": 100, "bore": 40}),
    "A gear with 32 teeth": ("spur_gear", {"teeth": 32}),
    "M8 nut": ("hex_nut", {"thread": 8}),
    "A seven-lobed knob for a bolt head": ("star_knob", {"lobes": 7}),
    "Coat hook, 3 screw holes": ("wall_hook", {"holes": 3}),
    "Holder that keeps 5 charging cables on my desk": ("cable_organizer", {"slots": 5}),
    "A grippy round knob, 40 mm wide, 20 ridges": ("knurled_knob", {"ridges": 20, "diameter": 40}),
    "Valve wheel, 3 spokes, 150 mm": ("hand_wheel", {"spokes": 3, "diameter": 150}),
    "A 12 mm bore pulley, 80 mm diameter": ("pulley", {"bore": 12, "diameter": 80}),
    "Corner bracket, 60 mm legs, one hole each": ("l_bracket", {"holes": 1, "base_length": 60, "upright_length": 60}),
    "M6 screw, 20 long": ("bolt", {"thread": 6, "length": 20}),
    "Axle 12 mm thick and 100 mm long with a flat": ("shaft", {"diameter": 12, "length": 100}),
}


def held_out_pass(spec, family, want):
    try:
        got = json.loads(spec[spec.index("{"):])
    except Exception:
        return False
    params = got.get("params", {})
    return got.get("family") == family and all(abs(params.get(k, -1) - v) < 1e-6 for k, v in want.items())


@torch.no_grad()
def fake_q4f16_1(model, group=32, bits=4):
    """Round weights the way WebLLM's MLC group quantization stores them: symmetric integers in
    [-(2^(bits-1) - 1), 2^(bits-1) - 1] in groups along the input dimension, one fp16 scale per group
    (linear layers and embeddings). q4f16_1: bits 4, groups of 32. q3f16_1: bits 3, groups of 40."""
    m = 2 ** (bits - 1) - 1
    for mod in model.modules():
        if isinstance(mod, (torch.nn.Linear, torch.nn.Embedding)):
            rows, k = mod.weight.shape
            pad = (-k) % group  # the last group of a row is zero-padded, as MLC does
            w = torch.nn.functional.pad(mod.weight.data.float(), (0, pad)).reshape(-1, group)
            scale = (w.abs().amax(1, keepdim=True) / m).half().float().clamp_min(1e-8)
            q = ((w / scale).round().clamp(-m, m) * scale).reshape(rows, k + pad)[:, :k]
            mod.weight.data = q.to(mod.weight.dtype)


# Edits: a part from the recorded answers, then a change. A pass needs the change and nothing else changed
# (the family and the other given numbers stay).
EDITS = [
    ("A pipe flange with six bolt holes", "Make it eight holes", "flange", {"count": 8}),
    ("A pipe flange with six bolt holes", "Remove the centre hole", "flange", {"count": 6, "bore": 0}),
    ("An M12 hex nut", "Make it M16", "hex_nut", {"thread": 16}),
    ("A spur gear with 24 teeth and an 8 mm bore", "Make it twice as thick", "spur_gear", {"teeth": 24, "thickness": 16}),
    ("A wall hook with two screw holes", "Add one more screw hole", "wall_hook", {"holes": 3}),
    ("A knurled thumb knob with 12 grip ridges", "Remove the hole underneath", "knurled_knob", {"ridges": 12, "bore": 0}),
    ("A hand wheel with five spokes", "Make it bigger, 150 mm across", "hand_wheel", {"spokes": 5, "diameter": 150}),
    ("An L bracket with three screw holes per leg", "Change it to one hole per leg", "l_bracket", {"holes": 1}),
]


def edit_messages(prompt, spec, instruction):
    return messages_for(prompt) + [{"role": "assistant", "content": spec},
                                   {"role": "user", "content": PROMPT["edit"].replace("{{INSTRUCTION}}", instruction)}]


class Model:
    def __init__(self, name):
        self.name = name
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to("cuda").eval()
        if Q3:
            fake_q4f16_1(self.model, group=40, bits=3)
        elif Q4:
            fake_q4f16_1(self.model)
        info = xgr.TokenizerInfo.from_huggingface(self.tok, vocab_size=self.model.config.get_text_config().vocab_size)
        self.compiler = xgr.GrammarCompiler(info)
        self.grammar = self.compiler.compile_json_schema(json.dumps(PROMPT["schema"]))

    def generate(self, msgs, grammar=None):
        kw = {"enable_thinking": False} if "Qwen3" in self.name else {}
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
        ids = self.tok(text, return_tensors="pt").to("cuda")
        t0 = time.perf_counter()
        out = self.model.generate(**ids, max_new_tokens=MAX_TOKENS, do_sample=False,
                                  logits_processor=[LogitsProcessor(grammar or self.grammar)])
        new = out[0][ids["input_ids"].shape[1]:]
        return self.tok.decode(new, skip_special_tokens=True), len(new), time.perf_counter() - t0


def run(model_name, out_dir):
    m = Model(model_name)
    rows = []
    for prompt in PROMPT["suggestions"]:
        base = msgs = messages_for(prompt)
        attempts, tokens, secs = 0, 0, 0.0
        for attempt in range(1, MAX_REPAIRS + 2):
            spec, n, dt = m.generate(msgs)
            attempts, tokens, secs = attempt, tokens + n, secs + dt
            r = g.csg_build(spec, 112)
            if not r["feedback"]:
                break
            msgs = base + [{"role": "assistant", "content": spec}, {"role": "user", "content": r["feedback"]}]
        row = {"model": model_name, "prompt": prompt, "spec": spec, "attempts": attempts, "tokens": tokens,
               "seconds": round(secs, 1), "ok": r["ok"], "verdict": r["verdict"], "genus": r["genus"],
               "shells": r["shells"], "first_try": attempts == 1 and not r["feedback"],
               "family": _family(spec), "adheres": bool(r["ok"] and ADHERENCE[prompt](_parts(r, spec), r))}
        rows.append(row)
        print(f"  {prompt[:44]:44s} {row['verdict']:24s} attempts {attempts}  adheres {row['adheres']}", flush=True)
    held = []
    for prompt, (family, want) in HELD_OUT.items():
        spec, n, dt = m.generate(messages_for(prompt))
        r = g.csg_build(spec, 64)
        held.append({"prompt": prompt, "spec": spec, "seconds": round(dt, 1),
                     "pass": bool(r["ok"] and held_out_pass(spec, family, want))})
    recorded = json.load(open(os.path.join(HERE, "recorded.json")))["answers"]
    edits = []
    for prompt, instruction, family, want in EDITS:
        spec, n, dt = m.generate(edit_messages(prompt, recorded[prompt], instruction))
        r = g.csg_build(spec, 64)
        edits.append({"prompt": prompt, "edit": instruction, "spec": spec, "seconds": round(dt, 1),
                      "pass": bool(r["ok"] and held_out_pass(spec, family, want))})
    del m
    torch.cuda.empty_cache()
    with open(os.path.join(out_dir, model_name.replace("/", "__") + ("-q3" if Q3 else "-q4" if Q4 else "") + ".json"), "w") as f:
        json.dump({"suggestions": rows, "held_out": held, "edits": edits}, f, indent=2)
    good = sum(r["ok"] and r["verdict"] != "not printable" for r in rows)
    print(f"{model_name}{' (q3)' if Q3 else ' (q4)' if Q4 else ''}: {good}/{len(rows)} valid & printable, {sum(r['adheres'] for r in rows)}/{len(rows)} "
          f"look like the prompt, {sum(r['first_try'] for r in rows)}/{len(rows)} first try; "
          f"held-out {sum(h['pass'] for h in held)}/{len(held)} exact; edits {sum(e['pass'] for e in edits)}/{len(edits)}\n", flush=True)


if __name__ == "__main__":
    out = os.environ.get("EVAL_OUT", os.path.join(HERE, "eval_results"))
    os.makedirs(out, exist_ok=True)
    for name in sys.argv[1:]:
        print(f"== {name}")
        run(name, out)
