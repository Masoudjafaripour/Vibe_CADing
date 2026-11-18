"""The v2 (fast) prompt, shared by eval_fast.py; website/src/scripts/fast/live-cad-fast.ts builds the same messages."""
import json
import os

import vibecad_geom as g

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "prompt_fast.json")) as f:
    FAST = json.load(f)
with open(os.path.join(HERE, "..", "live_cad", "prompt.json")) as f:
    V1 = json.load(f)


def branches(families):
    """The v1 schema's branch for each family. With keep_name false, without "name" (C++ names the part)."""
    out = []
    for b in V1["schema"]["anyOf"]:
        fam = b["properties"]["family"]["enum"][0]
        if fam in families:
            if FAST.get("keep_name", True):
                out.append(b)
                continue
            props = {k: v for k, v in b["properties"].items() if k != "name"}
            out.append({**b, "properties": props, "required": [r for r in b["required"] if r != "name"]})
    return out


def schema(families):
    return {"anyOf": branches(families)}


def system(families):
    return FAST["system"].replace("{{GUIDE}}", g.families_guide(families))


def messages(request, always=""):
    """-> (messages, families). The examples are shown with their own families' guide lines."""
    families = g.select_families(request, FAST["k"], always)
    shown = list(dict.fromkeys(families + [f for ex in FAST["examples"] for f in ex["families"]]))
    msgs = [{"role": "system", "content": system(shown)}]
    for ex in FAST["examples"]:
        msgs += [{"role": "user", "content": ex["prompt"]}, {"role": "assistant", "content": json.dumps(ex["spec"])}]
    return msgs + [{"role": "user", "content": request}], shown


def edit_messages(prompt, spec, instruction):
    family = json.loads(spec).get("family", "")
    if not FAST.get("keep_name", True):
        spec = json.dumps({k: v for k, v in json.loads(spec).items() if k != "name"})
    msgs, families = messages(prompt, always=family)
    return msgs + [{"role": "assistant", "content": spec},
                   {"role": "user", "content": FAST["edit"].replace("{{INSTRUCTION}}", instruction)}], families
