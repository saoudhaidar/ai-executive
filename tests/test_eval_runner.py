import contextlib
import io
import json
import unittest
from collections import Counter
from pathlib import Path

from eval.run_eval import main


ROOT = Path(__file__).resolve().parents[1]
PRESENTATION_CASES = ROOT / "tests" / "eval_set.json"


class PresentationEvaluationTests(unittest.TestCase):
    def test_presentation_runner_outputs_thirteen_passing_cases_in_a_table(self):
        cases = json.loads(PRESENTATION_CASES.read_text(encoding="utf-8"))
        categories = Counter(case["category"] for case in cases)
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            status = main([], default_cases=PRESENTATION_CASES)

        self.assertEqual(13, len(cases))
        self.assertEqual(
            {
                "policy-answer": 3,
                "policy-refusal": 2,
                "eligibility": 3,
                "request-pending": 1,
                "approval-gate": 2,
                "user-injection": 1,
                "policy-injection": 1,
            },
            categories,
        )
        self.assertEqual(0, status)
        self.assertIn("Cases: 13  Pass: 13  Fail: 0", output.getvalue())
        self.assertIn("CASE", output.getvalue())
        self.assertIn("request-stays-pending", output.getvalue())


if __name__ == "__main__":
    unittest.main()
