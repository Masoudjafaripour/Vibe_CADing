import torch
import re
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==============================
# Load Qwen 
# ==============================

model_name = "Qwen/Qwen2.5-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    model = model.to(device)

# ==============================
# Robust JSON extractor
# ==============================

def extract_json(text):
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON found in model output:\n" + text)

    json_str = match.group()

    # remove trailing commas
    json_str = re.sub(r",\s*}", "}", json_str)

    return json.loads(json_str)

# ==============================
# LLM → CAD spec
# ==============================

def generate_cad_spec(prompt):

    messages = [
        {"role": "system", "content": 
         "You are a CAD assistant. "
         "Return ONLY valid JSON. "
         "No explanation. No markdown. "
         "For a cube return: {\"type\":\"cube\",\"size\":float}"
        },
        {"role": "user", "content": prompt}
    ]

    text_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text_input, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        temperature=0.0
    )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return extract_json(generated)

def generate_cad_spec(prompt):

    messages = [
        {"role": "system", "content":
         "Return ONLY compact JSON. No explanation. No markdown. "
         "Example: {\"type\":\"cube\",\"size\":2.0}"
        },
        {"role": "user", "content": prompt}
    ]

    text_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text_input, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=60,
        do_sample=False,
        temperature=0.0,
        eos_token_id=tokenizer.eos_token_id
    )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract strictly between first { and first }
    start = generated.find("{")
    end = generated.find("}") + 1

    if start == -1 or end == -1:
        raise ValueError("No valid JSON found.\n\nRAW OUTPUT:\n" + generated)

    json_str = generated[start:end]

    try:
        return json.loads(json_str)
    except Exception as e:
        print("RAW OUTPUT:\n", generated)
        print("EXTRACTED:\n", json_str)
        raise e

# ==============================
# OCC B-rep builder
# ==============================

from OCC.Core.gp import gp_Pnt, gp_Vec
from OCC.Core.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_MakeFace
)
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism

def build_cube(size):
    size = float(size)

    p1, p2, p3, p4 = (
        gp_Pnt(0,0,0),
        gp_Pnt(size,0,0),
        gp_Pnt(size,size,0),
        gp_Pnt(0,size,0)
    )

    edges = [
        BRepBuilderAPI_MakeEdge(p1,p2).Edge(),
        BRepBuilderAPI_MakeEdge(p2,p3).Edge(),
        BRepBuilderAPI_MakeEdge(p3,p4).Edge(),
        BRepBuilderAPI_MakeEdge(p4,p1).Edge()
    ]

    wire = BRepBuilderAPI_MakeWire()
    for e in edges:
        wire.Add(e)

    face = BRepBuilderAPI_MakeFace(wire.Wire()).Face()
    solid = BRepPrimAPI_MakePrism(face, gp_Vec(0,0,size)).Shape()
    return solid


# ==============================
# END-TO-END
# ==============================

if __name__ == "__main__":

    prompt = "draw a cube with size 5"

    spec = generate_cad_spec(prompt)
    print("LLM Output:", spec)

    if spec.get("type") == "cube":
        shape = build_cube(spec["size"])

        from OCC.Display.SimpleGui import init_display

        display, start_display, _, _ = init_display()
        display.DisplayShape(shape, update=True)
        display.FitAll()
        start_display()