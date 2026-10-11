"""Tests for the SDF demo. Uses fake LLMs and a stub HTTP server, so no
model or API key is needed. Rendering tests are skipped if moderngl or
an OpenGL 3.3 context isn't available.
"""
import http.server
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest

import numpy as np
from PIL import Image

from harness import (
    SYSTEM_PROMPT, build_prompt, build_refine_prompt, extract_code,
    refine_harness, render_shader, run_harness, strip_version,
)
from llm import openai_llm

HERE = os.path.dirname(os.path.abspath(__file__))

HAS_GL = False
try:
    from render import create_context

    create_context().release()
    HAS_GL = True
except Exception:
    pass

SPHERE = "float sdf(vec3 p) { return sdSphere(p, 0.5); }"
BIG_SPHERE = "float sdf(vec3 p) { return sdSphere(p, 0.8); }"
TYPO = "float sdf(vec3 p) {\n    return lenght(p) - 0.5;\n}"
SMALL = dict(size=64, aa=1)


def fenced(code):
    return f"Sure:\n```glsl\n{code}\n```\nEnjoy."


def image_size(path):
    with Image.open(path) as img:
        return img.size


def load(path):
    with Image.open(path) as img:
        return np.asarray(img, dtype=float)


def scripted_llm(*responses):
    """Fake llm_fn returning `responses` in order; records prompts."""
    calls = []

    def llm_fn(prompt):
        calls.append(prompt)
        return responses[len(calls) - 1]
    return llm_fn, calls


class TestPrompts(unittest.TestCase):
    def test_system_prompt_lists_library(self):
        for name in ("sdSphere", "sdCapsule", "opSmoothUnion", "rotY"):
            self.assertIn(name, SYSTEM_PROMPT)

    def test_build_prompt(self):
        self.assertIn("Task: a snowman", build_prompt("a snowman"))
        self.assertNotIn("previous attempt", build_prompt("a snowman"))

    def test_repair_prompt_includes_code_and_error(self):
        prompt = build_prompt("a snowman", error="undefined variable", previous_code=TYPO)
        self.assertIn("undefined variable", prompt)
        self.assertIn("lenght", prompt)

    def test_refine_prompt(self):
        prompt = build_refine_prompt(SPHERE, "make it bigger")
        self.assertIn(SPHERE, prompt)
        self.assertIn("make it bigger", prompt)

    def test_extract_code(self):
        self.assertEqual(extract_code(fenced(SPHERE)), SPHERE)
        self.assertEqual(extract_code(f"```c\n{SPHERE}\n```"), SPHERE)
        self.assertEqual(extract_code(SPHERE), SPHERE)

    def test_strip_version(self):
        self.assertEqual(strip_version(f"#version 330\n{SPHERE}"), SPHERE)


class TestOpenAIBackend(unittest.TestCase):
    def test_request_and_response(self):
        seen = {}

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                seen["path"] = self.path
                seen["auth"] = self.headers.get("Authorization")
                seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                reply = json.dumps({"choices": [{"message": {"content": fenced(SPHERE)}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(reply)

            def log_message(self, *args):
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            llm_fn = openai_llm("test-model", f"http://127.0.0.1:{server.server_port}/v1", "secret")
            reply = llm_fn("a sphere")
        finally:
            server.shutdown()
        self.assertEqual(extract_code(reply), SPHERE)
        self.assertEqual(seen["path"], "/v1/chat/completions")
        self.assertEqual(seen["auth"], "Bearer secret")
        self.assertEqual(seen["body"]["model"], "test-model")
        self.assertEqual(seen["body"]["messages"][0]["content"], "a sphere")


@unittest.skipUnless(HAS_GL, "moderngl / OpenGL 3.3 not available")
class TestRendering(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def path(self, name):
        return os.path.join(self.tmp.name, name)

    def test_examples_render(self):
        for name in ("snowman", "duck"):
            with open(os.path.join(HERE, "examples", name + ".glsl")) as f:
                ok, error = render_shader(f.read(), self.path(name + ".png"), **SMALL)
            self.assertTrue(ok, error)
            self.assertEqual(image_size(self.path(name + ".png")), (64, 64))

    def test_four_view_sheet(self):
        ok, error = render_shader(SPHERE, self.path("s.png"), size=64, views=4, aa=1)
        self.assertTrue(ok, error)
        self.assertEqual(image_size(self.path("s.png")), (128, 128))

    def test_compile_error_reports_user_line(self):
        ok, error = render_shader(TYPO, self.path("t.png"), **SMALL)
        self.assertFalse(ok)
        self.assertIn("compile error", error)
        self.assertIn("lenght", error)
        self.assertRegex(error, r"\(2\)|:2\b")  # line 2 of the user's code

    def test_compile_error_quotes_offending_line(self):
        ok, error = render_shader(TYPO, self.path("t.png"), **SMALL)
        self.assertIn("line 2: return lenght(p) - 0.5;", error)

    def test_off_center_shape_rejected(self):
        ok, error = render_shader("float sdf(vec3 p) { return sdSphere(p - vec3(0.0, 0.0, 1.5), 0.5); }",
                                  self.path("x.png"), **SMALL)
        self.assertFalse(ok)
        self.assertIn("cut off", error)

    def test_missing_sdf_rejected(self):
        ok, error = render_shader("float shape(vec3 p) { return length(p) - 0.5; }", self.path("x.png"), **SMALL)
        self.assertFalse(ok)
        self.assertIn("must define", error)

    def test_turntable_gif(self):
        ok, error = render_shader(SPHERE, self.path("s.png"), gif_path=self.path("s.gif"),
                                  gif_frames=6, **SMALL)
        self.assertTrue(ok, error)
        with Image.open(self.path("s.gif")) as gif:
            self.assertEqual(gif.n_frames, 6)
            self.assertEqual(gif.size, (384, 384))

    def test_shadows_change_the_image(self):
        render_shader(BIG_SPHERE, self.path("on.png"), **SMALL)
        render_shader(BIG_SPHERE, self.path("off.png"), shadows=False, **SMALL)
        on = load(self.path("on.png"))
        off = load(self.path("off.png"))
        self.assertLess(on.mean(), off.mean())

    def test_invisible_shape_rejected(self):
        ok, error = render_shader("float sdf(vec3 p) { return sdSphere(p - vec3(50.0), 0.1); }",
                                  self.path("x.png"), **SMALL)
        self.assertFalse(ok)
        self.assertIn("not visible", error)

    def test_inverted_sign_rejected(self):
        ok, error = render_shader("float sdf(vec3 p) { return 0.5 - length(p); }",
                                  self.path("x.png"), **SMALL)
        self.assertFalse(ok)
        self.assertIn("fills the whole view", error)

    def test_timeout(self):
        ok, error = render_shader(SPHERE, self.path("x.png"), timeout=0.001, **SMALL)
        self.assertFalse(ok)
        self.assertIn("timed out", error)

    def test_code_size_limit(self):
        ok, error = render_shader("//" + "x" * 70_000 + "\n" + SPHERE, self.path("x.png"), **SMALL)
        self.assertFalse(ok)
        self.assertIn("limit", error)

    def test_generate_first_try(self):
        llm_fn, calls = scripted_llm(fenced(SPHERE))
        result = run_harness("a ball", llm_fn, self.path("ball.glsl"), **SMALL)
        self.assertTrue(result.success)
        self.assertEqual(result.attempts, 1)
        self.assertTrue(os.path.exists(self.path("ball.glsl")))
        self.assertTrue(os.path.exists(self.path("ball.png")))

    def test_repair_feeds_back_error(self):
        llm_fn, calls = scripted_llm(fenced(TYPO), fenced(SPHERE))
        result = run_harness("a ball", llm_fn, self.path("ball.glsl"), repairs=2, **SMALL)
        self.assertTrue(result.success)
        self.assertEqual(result.attempts, 2)
        self.assertIn("compile error", calls[1])
        self.assertIn("lenght", calls[1])

    def test_repairs_exhausted(self):
        llm_fn, calls = scripted_llm(fenced(TYPO), fenced(TYPO))
        result = run_harness("a ball", llm_fn, self.path("ball.glsl"), repairs=1, **SMALL)
        self.assertFalse(result.success)
        self.assertEqual(len(calls), 2)
        self.assertFalse(os.path.exists(self.path("ball.glsl")))

    def test_refine(self):
        llm_fn, calls = scripted_llm(fenced(BIG_SPHERE))
        result = refine_harness(SPHERE, "make it bigger", llm_fn, self.path("ball_v2.glsl"), **SMALL)
        self.assertTrue(result.success)
        self.assertIn(SPHERE, calls[0])
        with open(self.path("ball_v2.glsl")) as f:
            self.assertIn("0.8", f.read())

    def test_demo_cli_example(self):
        proc = subprocess.run(
            [sys.executable, os.path.join(HERE, "demo.py"), "--example", "snowman",
             "--out-dir", self.tmp.name, "--size", "64", "--aa", "1"],
            capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue(os.path.exists(self.path("snowman.png")))
        self.assertTrue(os.path.exists(self.path("snowman.glsl")))


if __name__ == "__main__":
    unittest.main()
