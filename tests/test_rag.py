import tempfile
import unittest
from pathlib import Path

from governed_assistant.rag import PolicyIndex


class PolicyIngestionTests(unittest.TestCase):
    def test_ingestion_preserves_punctuation_numbers_negation_and_citation_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.md"
            original = (
                "# Exact Policy\n"
                "## Fees & timing\n"
                "Applicants pay $75. The fee is non-refundable; it does not apply twice.\n"
            )
            path.write_text(original, encoding="utf-8")

            index = PolicyIndex(directory)
            source_text = " ".join(chunk.text for chunk in index.chunks)

        self.assertEqual(2, len(index.chunks))
        self.assertTrue(all(chunk.title == "Exact Policy" for chunk in index.chunks))
        self.assertTrue(all(chunk.section == "Fees & timing" for chunk in index.chunks))
        self.assertIn("$75.", source_text)
        self.assertIn("does not apply twice.", source_text)

    def test_empty_metadata_fails_fast_with_document_location(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "malformed.md"
            path.write_text("#\n## Eligibility\nA valid sentence.\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, r"malformed\.md:1"):
                PolicyIndex(directory)

    def test_missing_policy_directory_fails_fast(self):
        with self.assertRaisesRegex(ValueError, "does not exist"):
            PolicyIndex("/tmp/definitely-missing-governed-policy-directory")


if __name__ == "__main__":
    unittest.main()
