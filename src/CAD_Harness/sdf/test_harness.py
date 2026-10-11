"""Tests for the SDF harness. Uses a fake LLM, so no model download is
needed. Rendering tests are skipped if moderngl or an EGL context
isn't available.
"""
import os
import tempfile
import unittest

from harness import build_prompt, extract_code, run_harness, strip_version

HAS_GL = False
try:
    import moderngl

    moderngl.create_standalone_context(backend="egl").release()
    HAS_GL = True
except Exception:
    pass

SPHERE = "float sdf(vec3 p) { return length(p) - 0.5; }"


class TestPromptHelpers(unittest.TestCase):
    def test_build_prompt_no_error(self):
        prompt = build_prompt("a sphere")
        self.assertIn("a sphere", prompt)
        self.assertNotIn("previous code failed", prompt)

    def test_build_prompt_with_error(self):
        prompt = build_prompt("a sphere", error="undeclared identifier")
        self.assertIn("undeclared identifier", prompt)
        self.assertIn("Fix the code", prompt)

    def test_extract_code_from_fenced_block(self):
        text = f"Here you go:\n```glsl\n{SPHERE}\n```"
        self.assertEqual(extract_code(text), SPHERE)

    def test_extract_code_plain_text(self):
        self.assertEqual(extract_code(SPHERE), SPHERE)

    def test_strip_version(self):
        self.assertEqual(strip_version(f"#version 330\n{SPHERE}"), SPHERE)


@unittest.skipUnless(HAS_GL, "moderngl / EGL not available")
class TestRunHarness(unittest.TestCase):
    def _run(self, responses, max_retries=3):
        calls = []

        def fake_llm(prompt):
            calls.append(prompt)
            return responses[len(calls) - 1]

        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "shape.glsl")
            result = run_harness("a shape", fake_llm, output_path=out, max_retries=max_retries)
            files = (os.path.exists(out), os.path.exists(result.preview_path or "/nonexistent"))
        return result, calls, files

    def test_sphere_renders_first_try(self):
        result, calls, (glsl_exists, png_exists) = self._run([f"```glsl\n{SPHERE}\n```"])
        self.assertTrue(result.success)
        self.assertEqual(result.attempts, 1)
        self.assertTrue(glsl_exists and png_exists)

    def test_compile_error_is_fed_back(self):
        broken = "```glsl\nfloat sdf(vec3 p) { return lenght(p) - 0.5; }\n```"
        result, calls, _ = self._run([broken, f"```glsl\n{SPHERE}\n```"])
        self.assertTrue(result.success)
        self.assertEqual(result.attempts, 2)
        self.assertIn("compile error", calls[1])

    def test_invisible_shape_is_rejected(self):
        tiny = "```glsl\nfloat sdf(vec3 p) { return length(p - vec3(50.0)) - 0.1; }\n```"
        result, calls, _ = self._run([tiny], max_retries=1)
        self.assertFalse(result.success)
        self.assertIn("not visible", result.error)

    def test_inverted_sign_is_rejected(self):
        inverted = "```glsl\nfloat sdf(vec3 p) { return 0.5 - length(p); }\n```"
        result, calls, _ = self._run([inverted], max_retries=1)
        self.assertFalse(result.success)
        self.assertIn("fills the whole view", result.error)


if __name__ == "__main__":
    unittest.main()
