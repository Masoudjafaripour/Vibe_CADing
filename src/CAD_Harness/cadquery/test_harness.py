"""Tests for the CAD harness. Uses a fake LLM, so no model download is
needed. The end-to-end execution test is skipped if cadquery isn't
installed.
"""
import os
import tempfile
import unittest

from harness import build_prompt, extract_code, run_harness

cadquery = None
try:
    import cadquery  # noqa: F401
except ImportError:
    pass


class TestPromptHelpers(unittest.TestCase):
    def test_build_prompt_no_error(self):
        prompt = build_prompt("a cube of size 5")
        self.assertIn("a cube of size 5", prompt)
        self.assertNotIn("previous code raised", prompt)

    def test_build_prompt_with_error(self):
        prompt = build_prompt("a cube of size 5", error="NameError: x")
        self.assertIn("NameError: x", prompt)
        self.assertIn("Fix the code", prompt)

    def test_extract_code_from_fenced_block(self):
        text = "Here you go:\n```python\nresult = 1\n```"
        self.assertEqual(extract_code(text), "result = 1")

    def test_extract_code_plain_text(self):
        text = "result = 1"
        self.assertEqual(extract_code(text), "result = 1")


class TestRunHarnessRetryLogic(unittest.TestCase):
    def test_retries_until_llm_fixes_the_error(self):
        """First LLM response is broken; second is a working stub that
        doesn't need cadquery, so this exercises the retry loop even
        when cadquery isn't installed.
        """
        calls = []

        def fake_llm(prompt):
            calls.append(prompt)
            if len(calls) == 1:
                return "```python\nraise ValueError('boom')\n```"
            # Contains the string "exportStep" (as a comment) so the harness
            # doesn't append its own `result.val().exportStep(...)` call,
            # which would need a real cadquery object.
            return "```python\nresult = None  # exportStep placeholder\n```"

        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "out.step")
            result = run_harness(
                "a dummy part", fake_llm, output_path=out_path, max_retries=3
            )

        self.assertEqual(len(calls), 2)
        self.assertIn("boom", calls[1])  # error was fed back into the retry prompt
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


@unittest.skipIf(cadquery is None, "cadquery not installed")
class TestRunHarnessWithCadquery(unittest.TestCase):
    def test_generates_a_box_step_file(self):
        def fake_llm(prompt):
            return (
                "```python\n"
                "import cadquery as cq\n"
                "result = cq.Workplane('XY').box(1, 1, 1)\n"
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
