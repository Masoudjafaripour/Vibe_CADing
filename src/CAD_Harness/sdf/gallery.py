"""Text-to-CAD gallery: every prompt in gallery/*.glsl goes through the full pipeline

    prompt -> LLM -> GLSL -> validate / repair -> render + turntable -> STL (C++, checked)

and the results are combined into a captioned contact sheet and an animated grid.

    python gallery.py                                   # replay the recorded LLM answers in gallery/
    python gallery.py --backend openai --model gpt-4o   # ask a live LLM the same prompts
    python gallery.py --publish                         # also copy the sheet + GIF into assets/

Each gallery/<name>.glsl starts with  // Prompt: "..."  and is the recorded LLM
answer to that prompt; replay feeds it back through run_harness exactly as a
live answer would be (same checks, same repair loop).
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from harness import RESULTS_DIR, run_harness
from llm import BACKENDS, get_llm

HERE = os.path.dirname(os.path.abspath(__file__))
GALLERY_DIR = os.path.join(HERE, "gallery")
ASSETS_DIR = os.path.join(HERE, "assets")
ORDER = ["gear", "rook", "mug", "spring", "gyroid", "rocket", "vase", "chain"]


def load_prompts():
    items = []
    names = sorted(f[:-5] for f in os.listdir(GALLERY_DIR) if f.endswith(".glsl"))
    for name in sorted(names, key=lambda n: ORDER.index(n) if n in ORDER else len(ORDER)):
        with open(os.path.join(GALLERY_DIR, name + ".glsl")) as f:
            code = f.read()
        m = re.search(r'^// Prompt: "(.+)"', code, re.M)
        author = re.search(r"^// LLM: (.+)$", code, re.M)
        items.append({"name": name, "prompt": m.group(1) if m else name, "recorded": code,
                      "author": author.group(1).split(",")[0] if author else "recorded"})
    return items


def font(size):
    try:
        import matplotlib.font_manager as fm
        return ImageFont.truetype(fm.findfont("DejaVu Sans"), size)
    except Exception:
        return ImageFont.load_default()


def stats_line(s):
    shells = f"{s['components']} shell" + ("s" if s["components"] != 1 else "")
    w, d, h = s["size"]
    return (f"{s['triangles'] / 1000:.0f}k triangles · {'watertight' if s['watertight'] else 'NOT watertight'} · "
            f"{shells}\ngenus {s['genus']} · {w:.0f} × {d:.0f} × {h:.0f} mm")


def contact_sheet(rows, path, tile=360, cols=4):
    """Render + prompt + mesh stats for every shape."""
    cap_h = 118
    sheet = Image.new("RGB", (cols * tile, -(-len(rows) // cols) * (tile + cap_h)), "white")
    draw = ImageDraw.Draw(sheet)
    f_title, f_body = font(17), font(12)
    for i, r in enumerate(rows):
        x, y = (i % cols) * tile, (i // cols) * (tile + cap_h)
        sheet.paste(Image.open(r["png"]).convert("RGB").resize((tile, tile), Image.LANCZOS), (x, y))
        draw.text((x + 10, y + tile + 6), r["name"].capitalize(), fill=(20, 20, 25), font=f_title)
        prompt = textwrap.fill(f"“{r['prompt']}”", 52)
        draw.multiline_text((x + 10, y + tile + 28), prompt, fill=(70, 72, 80), font=f_body, spacing=2)
        draw.multiline_text((x + 10, y + tile + cap_h - 36), stats_line(r["stats"]) if r.get("stats") else "no STL",
                            fill=(30, 120, 70), font=f_body, spacing=3)
    sheet.save(path)


def animated_grid(rows, path, tile=220, cols=4, frames=36):
    """All turntables in one looping GIF."""
    gifs = [Image.open(r["gif"]) for r in rows]
    label_font = font(13)
    out = []
    for k in range(frames):
        frame = Image.new("RGB", (cols * tile, -(-len(rows) // cols) * tile), "white")
        draw = ImageDraw.Draw(frame)
        for i, (g, r) in enumerate(zip(gifs, rows)):
            g.seek(k % g.n_frames)
            x, y = (i % cols) * tile, (i // cols) * tile
            frame.paste(g.convert("RGB").resize((tile, tile), Image.LANCZOS), (x, y))
            draw.text((x + 8, y + 6), r["name"], fill=(25, 25, 30), font=label_font)
        out.append(frame.convert("P", palette=Image.ADAPTIVE))
    out[0].save(path, save_all=True, append_images=out[1:], duration=70, loop=0, optimize=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=("replay",) + BACKENDS, default="replay")
    parser.add_argument("--model", help="model for --backend openai / qwen")
    parser.add_argument("--only", nargs="*", help="subset of gallery names")
    parser.add_argument("--res", type=int, default=160, help="STL grid cells per axis (default 160)")
    parser.add_argument("--out-dir", help="default: results/gallery/<backend>")
    parser.add_argument("--publish", action="store_true", help="copy gallery.png / gallery.gif into assets/")
    args = parser.parse_args()

    items = [it for it in load_prompts() if not args.only or it["name"] in args.only]
    out_dir = args.out_dir or os.path.join(RESULTS_DIR, "gallery", args.backend)
    os.makedirs(out_dir, exist_ok=True)
    live_llm = None if args.backend == "replay" else get_llm(args.backend, args.model)

    rows = []
    for it in items:
        llm_fn = live_llm or (lambda prompt, code=it["recorded"]: f"```glsl\n{code}\n```")
        glsl = os.path.join(out_dir, it["name"] + ".glsl")
        result = run_harness(it["prompt"], llm_fn, glsl, repairs=2, gif=True, size=512)
        row = {**{k: it[k] for k in ("name", "prompt")}, "ok": result.success, "attempts": result.attempts,
               "png": result.image_path, "gif": result.gif_path,
               "author": it["author"] if args.backend == "replay" else (args.model or args.backend)}
        if result.success:
            stl, js = os.path.splitext(glsl)[0] + ".stl", os.path.splitext(glsl)[0] + ".mesh.json"
            proc = subprocess.run([sys.executable, os.path.join(HERE, "to_stl.py"), glsl, stl,
                                   "--res", str(args.res), "--json", js], capture_output=True, text=True, timeout=300)
            if proc.returncode == 0:
                with open(js) as f:
                    row["stats"] = json.load(f)
            else:
                print(f"  STL failed for {it['name']}: {proc.stderr.strip()[:200]}", file=sys.stderr)
        status = stats_line(row["stats"]) if row.get("stats") else (result.error or "no STL")[:120]
        print(f"{it['name']:8s} {'ok ' if result.success else 'FAIL'} (attempt {result.attempts})  "
              + status.replace("\n", " · "))
        rows.append(row)

    good = [r for r in rows if r["ok"]]
    if not good:
        sys.exit("nothing to show")
    sheet, anim = os.path.join(out_dir, "gallery.png"), os.path.join(out_dir, "gallery.gif")
    contact_sheet(good, sheet)
    animated_grid(good, anim)
    with open(os.path.join(out_dir, "gallery.json"), "w") as f:
        json.dump([{k: v for k, v in r.items() if k not in ("png", "gif")} for r in rows], f, indent=2)
    print(f"\n{len(good)}/{len(rows)} shapes passed. Wrote {sheet}, {anim}")
    if args.publish:
        for src in (sheet, anim):
            shutil.copyfile(src, os.path.join(ASSETS_DIR, os.path.basename(src)))
        print(f"Published to {ASSETS_DIR}")


if __name__ == "__main__":
    main()
