"""Can the in-browser model assemble parts from an instruction? Same prompt, schema
(built from the parts' ids and ports, as on the website), repair loop and C++ engine
(vibecad_geom.csg_assemble) as website/src/scripts/live-cad.ts.

    EVAL_Q4=1 python src/live_cad/eval_assembly.py Qwen/Qwen3-4B Qwen/Qwen3-1.7B
"""
import json
import os
import sys

import torch
import vibecad_geom as g

from eval_models import MAX_REPAIRS, Model

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "assembly_prompt.json")) as f:
    PROMPT = json.load(f)
SYSTEM = PROMPT["system"].replace("{{ASSEMBLY_GUIDE}}", g.assembly_guide())


def schema_for(parts):
    """The JSON schema for mates between these parts: only real ids and port names."""
    ids = sorted(parts)
    ports = sorted({p for spec in parts.values() for p in g.part_ports(spec)})
    mate = {"type": "object", "properties": {
        "part": {"type": "string", "enum": ids}, "port": {"type": "string", "enum": ports},
        "to": {"type": "string", "enum": ids}, "to_port": {"type": "string", "enum": ports},
        "offset": {"type": "number"}, "angle": {"type": "number"}, "each": {"type": "boolean"},
        "skip": {"type": "array", "items": {"type": "integer"}}},
        "required": ["part", "port", "to", "to_port"], "additionalProperties": False}
    return {"type": "object", "properties": {"name": {"type": "string"},
            "mates": {"type": "array", "minItems": 1, "maxItems": 12, "items": mate}},
            "required": ["name", "mates"], "additionalProperties": False}


def user_message(parts, instruction):
    return PROMPT["user"].replace("{{PARTS}}", g.describe_parts(json.dumps(parts))).replace("{{INSTRUCTION}}", instruction)


def messages_for(parts, instruction):
    msgs = [{"role": "system", "content": SYSTEM}]
    for ex in PROMPT["examples"]:
        msgs += [{"role": "user", "content": user_message(ex["parts"], ex["instruction"])},
                 {"role": "assistant", "content": json.dumps(ex["mates"])}]
    return msgs + [{"role": "user", "content": user_message(parts, instruction)}]


def edit_messages(parts, current, instruction):
    msgs = messages_for(parts, instruction)[:-1]
    user = (PROMPT["edit_user"].replace("{{PARTS}}", g.describe_parts(json.dumps(parts)))
            .replace("{{CURRENT}}", current).replace("{{INSTRUCTION}}", instruction))
    return msgs + [{"role": "user", "content": user}]


def P(**kw):
    return {k: json.dumps(v) for k, v in kw.items()}


def mates_of(text):
    try:
        return json.loads(text[text.index("{"):])["mates"]
    except Exception:
        return []


def has(ms, part, to, to_port=None, port=None, each=None):
    return any(m["part"] == part and m["to"] == to and (to_port is None or m["to_port"].startswith(to_port))
               and (port is None or m["port"] == port) and (each is None or bool(m.get("each")) == each) for m in ms)


FLANGE = {"name": "flange", "family": "flange", "params": {"count": 6, "hole_diameter": 9}}
BOLT8 = {"name": "M8 bolt", "family": "bolt", "params": {"thread": 8, "length": 25}}
NUT8 = {"name": "M8 nut", "family": "hex_nut", "params": {"thread": 8}}
SHAFT8 = {"name": "shaft", "family": "shaft", "params": {"diameter": 8, "length": 70}}
WHEEL = {"name": "hand wheel", "family": "hand_wheel", "params": {"bore": 8.4}}

# (parts, instruction, check(mates, result)); every check also needs a valid assembly without collisions.
SCENARIOS = [
    (P(p1=FLANGE, p2=BOLT8, p3=NUT8), "Put a bolt through every hole of the flange and a nut under each one",
     lambda ms, r: len(r["placed"]) == 13 and has(ms, "p2", "p1", "hole_", "under_head", True) and has(ms, "p3", "p1", "hole_", each=True)),
    (P(p1=SHAFT8, p2=WHEEL), "Slide the hand wheel 20 mm onto the top of the shaft",
     lambda ms, r: any(m["part"] == "p2" and m["to"] == "p1" and m["to_port"] == "top_end" and -25 <= m.get("offset", 0) <= -15 for m in ms)),
    (P(p1=SHAFT8, p2={"name": "608 bearing", "family": "bearing"}, p3={"name": "pulley", "family": "pulley", "params": {"bore": 8.4}}),
     "Put the bearing on the bottom end of the shaft and the pulley on the top end",
     lambda ms, r: has(ms, "p2", "p1", "bottom_end") and has(ms, "p3", "p1", "top_end")),
    (P(p1={"name": "L-bracket", "family": "l_bracket", "params": {"holes": 2, "hole_diameter": 5.5}},
       p2={"name": "M5 bolt", "family": "bolt", "params": {"thread": 5, "length": 12}}),
     "Fasten the bracket with a bolt in each hole of its base",
     lambda ms, r: len(r["placed"]) == 3 and has(ms, "p2", "p1", "base_hole_", each=True)),
    (P(p1=FLANGE, p2=SHAFT8, p3=WHEEL), "Stand the shaft in the flange's centre hole and fit the wheel on top of the shaft",
     lambda ms, r: has(ms, "p2", "p1", "bore_") and has(ms, "p3", "p2", "top_end")),
    (P(p1=BOLT8, p2={"name": "washer", "family": "washer", "params": {"outer": 16, "inner": 8.4, "thickness": 1.6}}, p3=NUT8),
     "Put the washer under the head of the bolt and the nut below the washer",
     # washer moved under the head, or the bolt set on the washer: both are right
     lambda ms, r: len(r["placed"]) == 3 and any({m["part"], m["to"]} == {"p1", "p2"} and "under_head" in (m["port"], m["to_port"]) for m in ms)),
    (P(p1={"name": "standoff", "family": "standoff", "params": {"across_flats": 13, "length": 20, "hole_diameter": 8.4}}, p2=NUT8),
     "Stack the nut on top of the standoff",
     lambda ms, r: has(ms, "p2", "p1", "bore_top") or has(ms, "p2", "p1", "top")),
    (P(p1={"name": "gear", "family": "spur_gear", "params": {"teeth": 24, "bore": 8.4}}, p2=SHAFT8),
     "The gear goes on the shaft, flush with its top end",
     lambda ms, r: any(m["part"] == "p1" and m["to"] == "p2" and m["to_port"] == "top_end" for m in ms)),
]


DEMO = {p["id"]: json.dumps(p["spec"]) for p in PROMPT["demo"]["parts"]}
BOLTED = json.dumps({"name": "bolted flange", "mates": [
    {"part": "p2", "port": "under_head", "to": "p1", "to_port": "hole_1_top", "each": True},
    {"part": "p3", "port": "bore_top", "to": "p1", "to_port": "hole_1_bottom", "each": True}]})
WHEEL_ON_SHAFT = json.dumps({"name": "wheel on shaft", "mates": [
    {"part": "p4", "port": "bottom_end", "to": "p1", "to_port": "bore_top", "offset": -8},
    {"part": "p5", "port": "bore_bottom", "to": "p4", "to_port": "top_end", "offset": -20}]})
BOLTS_ONLY = json.dumps({"name": "bolts", "mates": [
    {"part": "p2", "port": "under_head", "to": "p1", "to_port": "hole_1_top", "each": True}]})
# (current assembly, instruction, check(mates, result)) on the demo parts
EDITS = [
    (BOLTED, "Remove the nuts", lambda ms, r: len(r["placed"]) == 7 and not any(m["part"] == "p3" for m in ms)),
    (BOLTED, "Leave out the bolt and the nut in hole 3",
     lambda ms, r: len(r["placed"]) == 11 and all(3 in m.get("skip", []) for m in ms if m.get("each"))),
    (WHEEL_ON_SHAFT, "Move the wheel 10 mm lower",
     lambda ms, r: any(m["part"] == "p5" and abs(m.get("offset", 0) + 30) < 0.5 for m in ms) and len(r["placed"]) == 3),
    (BOLTS_ONLY, "Add a nut under each bolt", lambda ms, r: len(r["placed"]) == 13),
    (WHEEL_ON_SHAFT, "Remove the wheel", lambda ms, r: len(r["placed"]) == 2 and not any(m["part"] == "p5" for m in ms)),
]


def run(model_name, out_dir):
    m = Model(model_name)
    rows = []
    for parts, instruction, check in SCENARIOS:
        grammar = m.compiler.compile_json_schema(json.dumps(schema_for(parts)))
        base = msgs = messages_for(parts, instruction)
        secs = 0.0
        for attempt in range(1, MAX_REPAIRS + 2):
            text, n, dt = m.generate(msgs, grammar)
            secs += dt
            r = g.csg_assemble(json.dumps(parts), text, 48)
            if not r["feedback"]:
                break
            msgs = base + [{"role": "assistant", "content": text}, {"role": "user", "content": r["feedback"]}]
        ms = mates_of(text)
        ok = bool(r["ok"] and not r["overlaps"] and check(ms, r))
        rows.append({"instruction": instruction, "answer": text, "attempts": attempt, "seconds": round(secs, 1),
                     "valid": r["ok"], "collisions": len(r["overlaps"]), "pass": ok, "first_try": ok and attempt == 1,
                     "feedback": r["feedback"]})
        print(f"  {instruction[:60]:60s} pass {ok}  attempts {attempt}", flush=True)
    grammar = m.compiler.compile_json_schema(json.dumps(schema_for(DEMO)))
    edits = []
    for current, instruction, check in EDITS:
        text, n, dt = m.generate(edit_messages(DEMO, current, instruction), grammar)
        r = g.csg_assemble(json.dumps(DEMO), text, 48)
        ok = bool(r["ok"] and not r["overlaps"] and check(mates_of(text), r))
        edits.append({"edit": instruction, "answer": text, "pass": ok, "feedback": r["feedback"]})
        print(f"  edit: {instruction:50s} pass {ok}", flush=True)
    del m
    torch.cuda.empty_cache()
    q4 = os.environ.get("EVAL_Q4") == "1"
    with open(os.path.join(out_dir, "assembly__" + model_name.replace("/", "__") + ("-q4" if q4 else "") + ".json"), "w") as f:
        json.dump({"instructions": rows, "edits": edits}, f, indent=2)
    print(f"{model_name}{' (q4)' if q4 else ''}: {sum(r['pass'] for r in rows)}/{len(rows)} assemblies as instructed, "
          f"{sum(r['first_try'] for r in rows)}/{len(rows)} first try; edits {sum(e['pass'] for e in edits)}/{len(edits)}\n", flush=True)


if __name__ == "__main__":
    out = os.environ.get("EVAL_OUT", os.path.join(HERE, "eval_results"))
    os.makedirs(out, exist_ok=True)
    for name in sys.argv[1:]:
        print(f"== {name}")
        run(name, out)
