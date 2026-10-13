"""Optional MLflow tracking for demo.py (enabled with --track).

Each demo.py call becomes one MLflow run in the "sdf-harness" experiment:
  params    mode, backend, model, prompt / instruction, repair budget, render settings,
            system_prompt_version (hash of harness.SYSTEM_PROMPT)
  metrics   success, attempts, duration_s, code_chars
  tags      failure_kind (compile_error, off_frame, not_visible, ...)
  artifacts the .glsl / .png / .gif, attempts.json (every prompt, response
            and error), system_prompt.txt
  traces    one trace per run: a root span with one LLM span per attempt

Storage defaults to <repo>/mlruns (git-ignored); set MLFLOW_TRACKING_URI to
use a server instead. View with:
    mlflow ui --backend-store-uri sqlite:///<repo>/mlruns/mlflow.db --port 5000
"""
import contextlib
import hashlib
import json
import os
import time

import mlflow
from mlflow.entities import SpanType

from harness import SYSTEM_PROMPT

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
MLRUNS_DIR = os.path.join(REPO_ROOT, "mlruns")
EXPERIMENT = "sdf-harness"
SYSTEM_PROMPT_VERSION = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:10]


def tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{MLRUNS_DIR}/mlflow.db"


def failure_kind(error: str) -> str:
    """Buckets a harness error message so failures can be counted in the UI."""
    if not error:
        return "none"
    checks = [
        ("GLSL compile error", "compile_error"),
        ("must define the function", "missing_sdf"),
        ("not visible from any view", "not_visible"),
        ("fills the whole view", "inside_out"),
        ("cut off by the edge", "off_frame"),
        ("timed out", "timeout"),
        ("the limit is", "too_long"),
    ]
    return next((kind for text, kind in checks if text in error), "other")


class Tracker:
    """Wraps one demo.py invocation in an MLflow run."""

    def __init__(self, args, mode: str, run_name: str):
        os.makedirs(MLRUNS_DIR, exist_ok=True)
        mlflow.set_tracking_uri(tracking_uri())
        if mlflow.get_experiment_by_name(EXPERIMENT) is None:
            mlflow.create_experiment(EXPERIMENT, artifact_location=f"file://{MLRUNS_DIR}/artifacts")
        mlflow.set_experiment(EXPERIMENT)

        self.mode = mode
        self.start = time.time()
        self.run = mlflow.start_run(run_name=run_name)
        params = {
            "mode": mode,
            "repairs": args.retries,
            "size": args.size,
            "views": args.views,
            "aa": args.aa,
            "shadows": not args.no_shadows,
            "gif": args.gif,
            "system_prompt_version": SYSTEM_PROMPT_VERSION,
        }
        for key in ("prompt", "instruction", "refine", "example", "render"):
            if getattr(args, key, None):
                params[key] = getattr(args, key)[:500]  # MLflow caps param values
        mlflow.log_params(params)

    def log_llm(self, backend: str, model: str):
        mlflow.log_params({"backend": backend, "model": model or "default"})

    def wrap_llm(self, llm_fn, backend: str, model: str):
        """Records every LLM call as an LLM span in the run's trace."""
        def traced(prompt: str) -> str:
            with mlflow.start_span(name="llm_call", span_type=SpanType.LLM) as span:
                span.set_inputs({"prompt": prompt})
                span.set_attributes({"backend": backend, "model": model or "default"})
                response = llm_fn(prompt)
                span.set_outputs({"response": response})
                return response
        return traced

    @contextlib.contextmanager
    def root_span(self, inputs: dict):
        """Groups all attempts of one generate / refine call under one trace."""
        with mlflow.start_span(name=f"sdf_{self.mode}", span_type=SpanType.CHAIN) as span:
            span.set_inputs(inputs)
            yield span

    def finish(self, success: bool, attempts: int = 1, error: str = "", files=(), code: str = "",
               history=None):
        mlflow.log_metrics({
            "success": int(success),
            "attempts": attempts,
            "duration_s": round(time.time() - self.start, 2),
            "code_chars": len(code),
        })
        mlflow.set_tag("failure_kind", failure_kind(error))
        if error:
            mlflow.log_text(error, "error.txt")
        for path in files:
            if path and os.path.exists(path):
                mlflow.log_artifact(path)
        if history:
            mlflow.log_text(json.dumps(history, indent=2), "attempts.json")
        if self.mode in ("generate", "refine"):
            mlflow.log_text(SYSTEM_PROMPT, "system_prompt.txt")
        mlflow.end_run(status="FINISHED" if success else "FAILED")
