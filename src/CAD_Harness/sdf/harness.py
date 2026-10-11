"""SDF generation harness: LLM -> GLSL signed distance function -> render -> retry on error.

Same loop shape as ../cadquery/harness.py, but the geometry is an
implicit surface written as GLSL (cf. Keenan Crane's LLM-written GLSL
dragon) instead of a B-rep. `llm_fn` is any callable(prompt: str) -> str;
see qwen_llm.py for a Qwen3-8B-backed implementation.
"""
import os
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
RENDER_SCRIPT = os.path.join(HERE, "render.py")
RESULTS_DIR = os.path.join(HERE, "results")

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a 3D modeling assistant. Write GLSL (version 330) code that
    defines the function `float sdf(vec3 p)`, a signed distance function
    for the requested shape: negative inside, positive outside, zero on
    the surface. Y is up. The whole shape must fit inside the box
    [-1, 1]^3, centered at the origin.

    Build the shape from SDF primitives (sphere, box, capsule, torus,
    cylinder) combined with union (min), intersection (max), subtraction
    (max(a, -b)), smooth union, and transforms applied to p. Define any
    helper functions above sdf(). Do not write main(), uniforms, or a
    #version line. Output ONLY a single ```glsl code block, no explanation.
""")


@dataclass
class HarnessResult:
    success: bool
    code: str
    attempts: int
    error: str = ""
    output_path: str = ""
    preview_path: str = ""


def build_prompt(instruction: str, error: str = None) -> str:
    if error is None:
        return f"{SYSTEM_PROMPT}\n\nTask: {instruction}"
    return (
        f"{SYSTEM_PROMPT}\n\nTask: {instruction}\n\n"
        f"Your previous code failed with this error:\n{error}\n\n"
        f"Fix the code and return the full corrected GLSL."
    )


def extract_code(text: str) -> str:
    parts = text.split("```")
    # Fenced blocks land at odd indices: [text, code, text, code, ...]
    for part in parts[1::2]:
        part = part.strip()
        if part.startswith("glsl"):
            part = part[len("glsl"):].strip()
        if part:
            return part
    return text.strip()


def strip_version(code: str) -> str:
    """The renderer supplies its own #version line; a second one won't compile."""
    return "\n".join(l for l in code.splitlines() if not l.strip().startswith("#version"))


def execute_sdf(code: str, preview_path: str, timeout: int = 60):
    """Compiles and renders the GLSL in a subprocess (a bad shader can hang
    or crash the GL driver). Returns (success, error_message).
    """
    with tempfile.NamedTemporaryFile("w", suffix=".glsl", delete=False) as f:
        f.write(code)
        shader_path = f.name

    try:
        proc = subprocess.run(
            [sys.executable, RENDER_SCRIPT, shader_path, preview_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, f"Rendering timed out after {timeout}s (sdf() is too slow)"
    finally:
        os.unlink(shader_path)

    if proc.returncode != 0:
        return False, proc.stderr[-2000:]
    return True, ""


def run_harness(instruction: str, llm_fn, output_path: str = "out.glsl", max_retries: int = 3) -> HarnessResult:
    # Bare filenames go to results/; explicit paths are used as given
    if not os.path.dirname(output_path):
        output_path = os.path.join(RESULTS_DIR, output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    preview_path = os.path.splitext(output_path)[0] + "_preview.png"
    error = None
    code = ""
    for attempt in range(1, max_retries + 1):
        prompt = build_prompt(instruction, error)
        raw = llm_fn(prompt)
        code = strip_version(extract_code(raw))
        ok, error = execute_sdf(code, preview_path)
        if ok:
            with open(output_path, "w") as f:
                f.write(code)
            return HarnessResult(True, code, attempt, "", output_path, preview_path)
    return HarnessResult(False, code, max_retries, error)
