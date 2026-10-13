"""SDF generation harness: LLM -> GLSL SDF -> validate by rendering -> repair / refine.

Generation and rendering are separate: this module only builds prompts,
extracts code, and runs render.py in a subprocess with a timeout (the
GLSL is untrusted LLM output; a bad shader can hang or crash the GL
driver). `llm_fn` is any callable(prompt: str) -> str; see llm.py.
"""
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
RENDER_SCRIPT = os.path.join(HERE, "render.py")
RESULTS_DIR = os.path.join(HERE, "results")
MAX_CODE_CHARS = 64_000


def _lib_summary() -> str:
    """Comment + signature lines of sdf_lib.glsl, for the prompt."""
    with open(os.path.join(HERE, "sdf_lib.glsl")) as f:
        body = f.read().split("\n\n", 1)[1]  # drop the file header comment
    lines = []
    for line in body.splitlines():
        if line.startswith("//"):
            lines.append(line)
        elif re.match(r"^(float|vec\d|mat\d)\s+\w+\(", line):
            lines.append(line.split("{")[0].strip())
    return "\n".join(lines)


SYSTEM_PROMPT = textwrap.dedent("""\
    You are a procedural 3D modeling assistant. Write GLSL 330 code that
    defines `float sdf(vec3 p)`: a signed distance function for the
    requested shape (negative inside, zero on the surface, positive
    outside). Y is up and the camera looks from +Z. The ground is at
    y = -1: rest the shape on it. The whole shape must fit inside the
    box [-1, 1]^3. You may also define `vec3 sdf_color(vec3 p)`
    returning an RGB color in [0, 1] for the surface at p.

    These functions already exist. Call them; do not redefine them:
    {lib}

    Build the shape from these primitives. Transform p before passing it
    in (translate: p - center; rotate: rotX/rotY/rotZ; mirror: abs(p.x))
    and blend parts with opSmoothUnion. Put helper functions above
    sdf(). Do not write main(), uniforms, textures or a #version line.
    Output ONLY one ```glsl code block, no explanation.
""").replace("{lib}", textwrap.indent(_lib_summary(), "    ").strip())


@dataclass
class HarnessResult:
    success: bool
    code: str
    attempts: int
    error: str = ""
    output_path: str = ""
    image_path: str = ""
    gif_path: str = ""
    # one {"attempt", "prompt", "response", "error"} dict per LLM call
    history: list = field(default_factory=list)


class RenderEnvironmentError(RuntimeError):
    """No usable OpenGL here; the LLM can't fix that, so don't retry."""


def _failure_section(previous_code: str, error: str) -> str:
    return (
        f"\n\nYour previous attempt:\n```glsl\n{previous_code}\n```\n"
        f"failed with this error:\n{error}\n\n"
        f"Fix it and return the full corrected code."
    )


def build_prompt(instruction: str, error: str = None, previous_code: str = None) -> str:
    prompt = f"{SYSTEM_PROMPT}\nTask: {instruction}"
    if error is not None:
        prompt += _failure_section(previous_code or "", error)
    return prompt


def build_refine_prompt(code: str, instruction: str, error: str = None, previous_code: str = None) -> str:
    prompt = (
        f"{SYSTEM_PROMPT}\nHere is the current code for a shape:\n```glsl\n{code}\n```\n\n"
        f"Change it as follows: {instruction}\n"
        f"Keep everything else the same and return the full updated code."
    )
    if error is not None:
        prompt += _failure_section(previous_code or "", error)
    return prompt


def extract_code(text: str) -> str:
    parts = text.split("```")
    # Fenced blocks land at odd indices: [text, code, text, code, ...]
    for part in parts[1::2]:
        part = part.strip()
        for tag in ("glsl", "c", "cpp"):
            if part.startswith(tag + "\n"):
                part = part[len(tag):].strip()
                break
        if part:
            return part
    return text.strip()


def strip_version(code: str) -> str:
    """The renderer supplies its own #version line; a second one won't compile."""
    return "\n".join(l for l in code.splitlines() if not l.strip().startswith("#version"))


def render_shader(code: str, image_path: str, size: int = 768, views: int = 1, aa: int = 2,
                  shadows: bool = True, timeout: float = 60, gif_path: str = None, gif_frames: int = 36):
    """Compiles and renders `code` in a subprocess (plus a turntable GIF if
    `gif_path` is given). Returns (success, error).

    Raises RenderEnvironmentError if this machine has no usable OpenGL.
    """
    if len(code) > MAX_CODE_CHARS:
        return False, f"Code is {len(code)} characters; the limit is {MAX_CODE_CHARS}."

    with tempfile.TemporaryDirectory() as tmp:
        shader_path = os.path.join(tmp, "shape.glsl")
        with open(shader_path, "w") as f:
            f.write(code)
        cmd = [sys.executable, RENDER_SCRIPT, shader_path, os.path.abspath(image_path),
               "--size", str(size), "--views", str(views), "--aa", str(aa)]
        if not shadows:
            cmd.append("--no-shadows")
        if gif_path:
            cmd += ["--gif", os.path.abspath(gif_path), "--gif-frames", str(gif_frames)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=tmp)
        except subprocess.TimeoutExpired:
            return False, f"Rendering timed out after {timeout}s (sdf() is too slow or never returns)."

    if proc.returncode == 3:
        raise RenderEnvironmentError(proc.stderr.strip())
    if proc.returncode != 0:
        return False, proc.stderr.strip()[-2000:]
    return True, ""


def resolve_output(output_path: str) -> str:
    """Bare filenames go to results/; explicit paths are used as given."""
    if not os.path.dirname(output_path):
        output_path = os.path.join(RESULTS_DIR, output_path)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    return output_path


def _generate(make_prompt, llm_fn, output_path: str, repairs: int, gif: bool, render_kwargs) -> HarnessResult:
    output_path = resolve_output(output_path)
    stem = os.path.splitext(output_path)[0]
    image_path = stem + ".png"
    gif_path = stem + ".gif" if gif else None
    error = None
    code = ""
    history = []
    for attempt in range(1, repairs + 2):
        prompt = make_prompt(error, code)
        raw = llm_fn(prompt)
        code = strip_version(extract_code(raw))
        ok, error = render_shader(code, image_path, gif_path=gif_path, **render_kwargs)
        history.append({"attempt": attempt, "prompt": prompt, "response": raw, "error": error})
        if ok:
            with open(output_path, "w") as f:
                f.write(code + "\n")
            return HarnessResult(True, code, attempt, "", output_path, image_path, gif_path or "", history)
    return HarnessResult(False, code, repairs + 1, error, history=history)


def run_harness(instruction: str, llm_fn, output_path: str = "out.glsl", repairs: int = 2,
                gif: bool = False, **render_kwargs) -> HarnessResult:
    """Text -> GLSL. On a compile / render error the LLM gets the error
    and its own code back, up to `repairs` more times."""
    return _generate(
        lambda error, prev: build_prompt(instruction, error, prev),
        llm_fn, output_path, repairs, gif, render_kwargs,
    )


def refine_harness(code: str, instruction: str, llm_fn, output_path: str, repairs: int = 2,
                   gif: bool = False, **render_kwargs) -> HarnessResult:
    """Existing GLSL + edit instruction -> new GLSL, with the same repair loop."""
    return _generate(
        lambda error, prev: build_refine_prompt(code, instruction, error, prev),
        llm_fn, output_path, repairs, gif, render_kwargs,
    )
