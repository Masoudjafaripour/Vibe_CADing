"""LLM-to-SDF demo CLI. Writes <name>.glsl, <name>.png and <name>.json.

    python demo.py --example snowman --gif                 # no LLM needed; --gif adds a turntable
    python demo.py --render results/dragon.glsl            # re-render any shader
    python demo.py --prompt "a stylized dragon"            # text -> GLSL
    python demo.py --refine results/dragon.glsl --instruction "make the wings larger"
    python demo.py --prompt "a stylized dragon" --track    # log the run to MLflow (see tracking.py)
"""
import argparse
import contextlib
import json
import os
import re
import shutil
import sys

from harness import RESULTS_DIR, RenderEnvironmentError, refine_harness, render_shader, run_harness
from llm import BACKENDS, default_backend, get_llm

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES_DIR = os.path.join(HERE, "examples")
EXAMPLES = sorted(f[:-5] for f in os.listdir(EXAMPLES_DIR) if f.endswith(".glsl"))
STOPWORDS = {"a", "an", "the", "of", "with", "made", "and", "from", "in", "on"}


def slugify(text: str, max_words: int = 4) -> str:
    words = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]
    return "_".join(words[:max_words]) or "shape"


def next_version(path: str, out_dir: str) -> str:
    """results/dragon.glsl -> dragon_v2, then dragon_v3, ..."""
    stem = re.sub(r"_v\d+$", "", os.path.splitext(os.path.basename(path))[0])
    n = 2
    while os.path.exists(os.path.join(out_dir, f"{stem}_v{n}.glsl")):
        n += 1
    return f"{stem}_v{n}"


def start_tracking(args, mode: str, name: str):
    """Returns a tracking.Tracker when --track is set, else None."""
    if not args.track:
        return None
    from tracking import Tracker  # imports mlflow; keep it optional
    return Tracker(args, mode, name)


def write_metadata(glsl_path: str, **meta):
    with open(os.path.splitext(glsl_path)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--example", choices=EXAMPLES, help="render a bundled example (no LLM)")
    mode.add_argument("--render", metavar="GLSL", help="render an existing .glsl file (no LLM)")
    mode.add_argument("--prompt", help="generate a new shape from text")
    mode.add_argument("--refine", metavar="GLSL", help="edit an existing .glsl with --instruction")
    parser.add_argument("--instruction", help="edit to apply with --refine")
    parser.add_argument("--name", help="output file stem (default: derived from the input)")
    parser.add_argument("--out-dir", default=RESULTS_DIR, help="default: results/ next to this script")
    parser.add_argument("--backend", choices=BACKENDS, default=None,
                        help=f"LLM backend (default: openai if OPENAI_API_KEY/OPENAI_BASE_URL is set, else qwen; now {default_backend()})")
    parser.add_argument("--model", help="model name for the backend")
    parser.add_argument("--retries", type=int, default=2, choices=range(0, 4),
                        help="LLM repair attempts after a failed render, 0-3 (default 2)")
    parser.add_argument("--size", type=int, default=768, help="pixels per view (default 768)")
    parser.add_argument("--views", type=int, choices=(1, 4), default=1, help="1 hero view or a 2x2 sheet")
    parser.add_argument("--aa", type=int, default=2, help="supersampling factor (default 2)")
    parser.add_argument("--no-shadows", action="store_true")
    parser.add_argument("--gif", action="store_true", help="also save a 360-degree turntable <name>.gif")
    parser.add_argument("--gif-frames", type=int, default=36, help="frames in the GIF (default 36)")
    parser.add_argument("--timeout", type=float, default=60, help="render timeout in seconds")
    parser.add_argument("--track", action="store_true", help="log this run to MLflow (needs: pip install mlflow)")
    args = parser.parse_args()

    if args.refine and not args.instruction:
        parser.error("--refine needs --instruction")
    os.makedirs(args.out_dir, exist_ok=True)
    render_kwargs = dict(size=args.size, views=args.views, aa=args.aa, shadows=not args.no_shadows,
                         timeout=args.timeout, gif_frames=args.gif_frames)

    try:
        if args.example or args.render:
            src = os.path.join(EXAMPLES_DIR, args.example + ".glsl") if args.example else args.render
            name = args.name or os.path.splitext(os.path.basename(src))[0]
            glsl_path = os.path.join(args.out_dir, name + ".glsl")
            png_path = os.path.join(args.out_dir, name + ".png")
            with open(src) as f:
                code = f.read()
            gif_path = os.path.join(args.out_dir, name + ".gif") if args.gif else None
            tracker = start_tracking(args, "example" if args.example else "render", name)
            ok, error = render_shader(code, png_path, gif_path=gif_path, **render_kwargs)
            if not ok:
                if tracker:
                    tracker.finish(False, error=error, code=code)
                print(f"Render failed:\n{error}", file=sys.stderr)
                return 1
            if os.path.abspath(src) != os.path.abspath(glsl_path):
                shutil.copyfile(src, glsl_path)
            if tracker:
                tracker.finish(True, files=(glsl_path, png_path, gif_path), code=code)
            print(f"Saved {png_path}" + (f" and {gif_path}" if gif_path else ""))
            return 0

        backend = args.backend or default_backend()
        try:
            llm_fn = get_llm(backend, args.model)
        except ValueError as e:
            parser.error(str(e))
        if args.prompt:
            name = args.name or slugify(args.prompt)
            glsl_path = os.path.join(args.out_dir, name + ".glsl")
            tracker = start_tracking(args, "generate", name)
            if tracker:
                tracker.log_llm(backend, args.model)
                llm_fn = tracker.wrap_llm(llm_fn, backend, args.model)
            with tracker.root_span({"prompt": args.prompt}) if tracker else contextlib.nullcontext():
                result = run_harness(args.prompt, llm_fn, glsl_path, args.retries, args.gif, **render_kwargs)
            meta = {"mode": "generate", "prompt": args.prompt}
        else:
            with open(args.refine) as f:
                source_code = f.read()
            name = args.name or next_version(args.refine, args.out_dir)
            glsl_path = os.path.join(args.out_dir, name + ".glsl")
            tracker = start_tracking(args, "refine", name)
            if tracker:
                tracker.log_llm(backend, args.model)
                llm_fn = tracker.wrap_llm(llm_fn, backend, args.model)
            with tracker.root_span({"source": args.refine, "instruction": args.instruction}) if tracker \
                    else contextlib.nullcontext():
                result = refine_harness(source_code, args.instruction, llm_fn, glsl_path, args.retries, args.gif,
                                        **render_kwargs)
            meta = {"mode": "refine", "source": args.refine, "instruction": args.instruction}
    except RenderEnvironmentError as e:
        print(f"Cannot render on this machine: {e}", file=sys.stderr)
        return 3

    meta.update(backend=backend, model=args.model, success=result.success,
                attempts=result.attempts, error=result.error)
    if result.success:
        write_metadata(glsl_path, **meta)
        if tracker:
            tracker.finish(True, result.attempts, files=(result.output_path, result.image_path, result.gif_path),
                           code=result.code, history=result.history)
        saved = ", ".join(p for p in (result.output_path, result.image_path, result.gif_path) if p)
        print(f"Saved {saved} (attempt {result.attempts}/{args.retries + 1})")
        return 0

    failed_path = os.path.join(args.out_dir, name + ".failed.glsl")
    with open(failed_path, "w") as f:
        f.write(result.code + "\n")
    write_metadata(failed_path, **meta)
    if tracker:
        tracker.finish(False, result.attempts, result.error, files=(failed_path,), code=result.code,
                       history=result.history)
    print(f"Failed after {result.attempts} attempts; last code in {failed_path}\n{result.error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
