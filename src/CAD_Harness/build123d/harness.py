"""CAD generation harness: LLM -> build123d code -> execute -> retry.

Same generate/execute/retry loop as ../cadquery/harness.py, but targets
build123d (the kernel used by earthtojake/text-to-cad's `cadgen` skill,
see ../../text-to-cad) instead of CadQuery. Self-contained so this
folder can live in its own venv (build123d and cadquery's OCP builds
conflict when installed together). `llm_fn` is any
callable(prompt: str) -> str.
"""
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a CAD assistant. Write Python code using the build123d library
    that builds the requested part and assigns the final result to a
    variable named `result` (a build123d Part/Solid/Compound). Use the
    builder API (e.g. `with BuildPart() as bp: ...; result = bp.part`).
    Output ONLY a single python code block, no explanation.
""")


@dataclass
class HarnessResult:
    success: bool
    code: str
    attempts: int
    error: str = ""
    output_path: str = ""


def extract_code(text: str) -> str:
    parts = text.split("```")
    # Fenced blocks land at odd indices: [text, code, text, code, ...]
    for part in parts[1::2]:
        part = part.strip()
        if part.startswith("python"):
            part = part[len("python"):].strip()
        if part:
            return part
    return text.strip()


def build_prompt(instruction: str, error: str = None) -> str:
    if error is None:
        return f"{SYSTEM_PROMPT}\n\nTask: {instruction}"
    return (
        f"{SYSTEM_PROMPT}\n\nTask: {instruction}\n\n"
        f"Your previous code raised this error:\n{error}\n\n"
        f"Fix the code and return the full corrected script."
    )


def execute_build123d(code: str, output_path: str, timeout: int = 30):
    """Runs generated code in a subprocess and exports the result to STEP.

    Returns (success, error_message).
    """
    script = code
    if "export_step" not in script:
        script += (
            f"\n\nfrom build123d import export_step\n"
            f"export_step(result, {output_path!r})\n"
        )

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(script)
        script_path = f.name

    try:
        proc = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, f"Execution timed out after {timeout}s"

    if proc.returncode != 0:
        return False, proc.stderr[-2000:]
    return True, ""


def run_harness(instruction: str, llm_fn, output_path: str = "out.step", max_retries: int = 3) -> HarnessResult:
    error = None
    code = ""
    for attempt in range(1, max_retries + 1):
        prompt = build_prompt(instruction, error)
        raw = llm_fn(prompt)
        code = extract_code(raw)
        ok, error = execute_build123d(code, output_path)
        if ok:
            return HarnessResult(True, code, attempt, "", output_path)
    return HarnessResult(False, code, max_retries, error)
