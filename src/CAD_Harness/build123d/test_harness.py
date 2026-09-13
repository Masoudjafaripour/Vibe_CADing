"""Tests for harness.py (build123d variant). Mirrors ../cadquery/test_harness.py:
a fake LLM so no model download is needed, and the end-to-end execution
test skips if build123d isn't installed.
"""
import os
import tempfile
import unittest

from harness import build_prompt, run_harness

build123d = None
try:
    import build123d  # noqa: F401
except ImportError:
    pass


class TestPromptHelpers(unittest.TestCase):
    def test_build_prompt_no_error(self):
        prompt = build_prompt("a cube of size 5")
        self.assertIn("a cube of size 5", prompt)
        self.assertIn("build123d", prompt)

    def test_build_prompt_with_error(self):
        prompt = build_prompt("a cube of size 5", error="NameError: x")
        self.assertIn("NameError: x", prompt)
        self.assertIn("Fix the code", prompt)


class TestRunHarnessRetryLogic(unittest.TestCase):
    def test_retries_until_llm_fixes_the_error(self):
        calls = []

        def fake_llm(prompt):
            calls.append(prompt)
            if len(calls) == 1:
                return "```python\nraise ValueError('boom')\n```"
            # Contains "export_step" (as a comment) so the harness doesn't
            # append its own export call, which would need real build123d.
            return "```python\nresult = None  # export_step placeholder\n```"

        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "out.step")
            result = run_harness(
                "a dummy part", fake_llm, output_path=out_path, max_retries=3
            )

        self.assertEqual(len(calls), 2)
        self.assertIn("boom", calls[1])
        self.assertTrue(result.success)
        self.assertEqual(result.attempts, 2)

    def test_gives_up_after_max_retries(self):
        def always_broken(prompt):
            return "```python\nraise RuntimeError('nope')\n```"

        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "out.step")
            result = run_harness(
                "a dummy part", always_broken, output_path=out_path, max_retries=2
            )

        self.assertFalse(result.success)
        self.assertEqual(result.attempts, 2)
        self.assertIn("nope", result.error)


@unittest.skipIf(build123d is None, "build123d not installed")
class TestRunHarnessWithBuild123d(unittest.TestCase):
    def test_generates_a_box_step_file(self):
        def fake_llm(prompt):
            return (
                "```python\n"
                "from build123d import BuildPart, Box\n"
                "with BuildPart() as bp:\n"
                "    Box(1, 1, 1)\n"
                "result = bp.part\n"
                "```"
            )

        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "out.step")
            result = run_harness("a 1x1x1 cube", fake_llm, output_path=out_path)

            self.assertTrue(result.success)
            self.assertEqual(result.attempts, 1)
            self.assertTrue(os.path.exists(out_path))


if __name__ == "__main__":
    unittest.main()
