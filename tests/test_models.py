import json
import unittest

from governed_assistant.models import LocalExtractiveProvider, OllamaProvider


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class OllamaProviderTests(unittest.TestCase):
    passages = [
        {
            "document": "Rental Assistance",
            "section": "Required documents",
            "text": "Applicants must provide proof of identity.",
        }
    ]

    def test_valid_response_requires_and_returns_grounded_citation(self):
        def opener(request, timeout):
            return _Response(
                {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    "Applicants must provide proof of identity. "
                                    "[Rental Assistance §Required documents]"
                                )
                            }
                        }
                    ]
                }
            )

        provider = OllamaProvider(
            endpoint="http://localhost:11434/v1/chat/completions",
            timeout_seconds=1,
            fallback=LocalExtractiveProvider(),
            simple_model="qwen3.5:4b",
            complex_model="qwen3.5:9b",
            opener=opener,
        )
        result = provider.generate("qwen3.5:4b", "What documents are required?", self.passages, 100)

        self.assertEqual("ollama", result.provider_name)
        self.assertEqual((0,), result.citation_indices)

    def test_ungrounded_citation_falls_back_to_deterministic_provider(self):
        def opener(request, timeout):
            return _Response(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "Answer [Other Policy §Unknown]"
                            }
                        }
                    ]
                }
            )

        provider = OllamaProvider(
            endpoint="http://localhost:11434/v1/chat/completions",
            timeout_seconds=1,
            fallback=LocalExtractiveProvider(),
            opener=opener,
        )
        result = provider.generate("qwen3.5:4b", "What documents are required?", self.passages, 100)

        self.assertEqual("local-extractive", result.provider_name)
        self.assertIn("proof of identity", result.answer.lower())

    def test_unreachable_ollama_falls_back_without_network_requirement(self):
        def opener(request, timeout):
            raise TimeoutError("offline")

        provider = OllamaProvider(
            endpoint="http://localhost:11434/v1/chat/completions",
            timeout_seconds=1,
            fallback=LocalExtractiveProvider(),
            opener=opener,
        )
        result = provider.generate("qwen3.5:4b", "What documents are required?", self.passages, 100)

        self.assertEqual("local-extractive", result.provider_name)
        self.assertTrue(result.citation_indices)


if __name__ == "__main__":
    unittest.main()
