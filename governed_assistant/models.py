import json
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from typing import Protocol


MODEL_PROVIDER = "local-extractive"
MODEL_PRICES = {
    "local-small": (0.00015, 0.0006),
    "local-reasoning": (0.0006, 0.0024),
}
_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "for", "from",
    "how", "i", "in", "is", "it", "me", "of", "on", "or", "the", "this", "to",
    "what", "when", "where", "which", "who", "why", "with", "you",
}
_ANSWER_EXPANSIONS = {
    "document": {"proof", "provide", "form"},
    "documents": {"proof", "provide", "form"},
    "required": {"must", "provide"},
    "requirements": {"must", "provide"},
    "age": {"child", "under"},
    "child": {"age", "under"},
    "childcare": {"child"},
}


@dataclass(frozen=True)
class ModelResult:
    answer: str
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    latency_ms: int
    citation_indices: tuple[int, ...]
    cost_basis: str = "simulated local profile rates"
    provider_name: str = ""


class ModelProvider(Protocol):
    name: str

    def supports(self, model: str) -> bool:
        ...

    def generate(self, model, question, passages, max_output_tokens) -> ModelResult:
        ...


class OllamaError(Exception):
    """Raised when a local Ollama response cannot be safely used."""


class OllamaProvider:
    """OpenAI-compatible Ollama adapter with deterministic offline fallback."""

    name = "ollama"

    def __init__(
        self,
        endpoint,
        timeout_seconds,
        fallback,
        guardian_model="",
        simple_model="qwen3.5:4b",
        complex_model="qwen3.5:9b",
        opener=None,
    ):
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self.fallback = fallback
        self.guardian_model = guardian_model
        self.models = {simple_model, complex_model}
        self._opener = opener or urllib.request.urlopen

    def supports(self, model):
        return model in self.models

    def generate(self, model, question, passages, max_output_tokens):
        if self.guardian_model and is_prompt_injection(question):
            raise OllamaError("prompt injection blocked")
        try:
            if self.guardian_model:
                self._check_guardian(question)
            result = self._generate_remote(model, question, passages, max_output_tokens)
            return ModelResult(
                answer=result.answer,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                estimated_cost_usd=result.estimated_cost_usd,
                latency_ms=result.latency_ms,
                citation_indices=result.citation_indices,
                cost_basis=result.cost_basis,
                provider_name=self.name,
            )
        except (OllamaError, OSError, urllib.error.URLError, TimeoutError, ValueError):
            fallback_model = (
                "local-reasoning" if model not in {"local-small"} else "local-small"
            )
            result = self.fallback.generate(
                fallback_model, question, passages, max_output_tokens
            )
            return ModelResult(
                answer=result.answer,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                estimated_cost_usd=result.estimated_cost_usd,
                latency_ms=result.latency_ms,
                citation_indices=result.citation_indices,
                cost_basis=f"deterministic fallback: {result.cost_basis}",
                provider_name=self.fallback.name,
            )

    def _check_guardian(self, question):
        payload = self._request_json(
            self.guardian_model,
            [{"role": "user", "content": question}],
            32,
        )
        content = _response_content(payload).lower()
        if any(marker in content for marker in ("unsafe", "block", "injection")):
            raise OllamaError("guardian blocked prompt")

    def _generate_remote(self, model, question, passages, max_output_tokens):
        started = time.monotonic()
        context = "\n".join(
            f"[{p['document']} §{p['section']}] {p['text']}" for p in passages
        )
        payload = self._request_json(
            model,
            [
                {
                    "role": "system",
                    "content": (
                        "Answer only from the supplied policy passages. Preserve "
                        "negation and numbers. End every factual answer with citations "
                        "in the exact form [Document §Section]."
                    ),
                },
                {"role": "user", "content": f"Question: {question}\nPassages:\n{context}"},
            ],
            max_output_tokens,
        )
        answer = _response_content(payload).strip()
        citation_indices = _validated_citations(answer, passages)
        if not citation_indices:
            raise OllamaError("response had no grounded citations")
        input_tokens = _token_count(question + " " + context)
        output_tokens = _token_count(answer)
        input_rate, output_rate = (
            (0.00015, 0.0006) if model == "qwen3.5:4b" else (0.0006, 0.0024)
        )
        return ModelResult(
            answer=answer,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=(input_tokens * input_rate + output_tokens * output_rate) / 1_000,
            latency_ms=max(0, round((time.monotonic() - started) * 1_000)),
            citation_indices=tuple(citation_indices),
            cost_basis="simulated local Ollama profile rates",
        )

    def _request_json(self, model, messages, max_tokens):
        body = json.dumps(
            {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0}
        ).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._opener(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise OllamaError("invalid Ollama JSON response") from error
        if not isinstance(payload, dict):
            raise OllamaError("invalid Ollama response shape")
        return payload

class LocalExtractiveProvider:
    name = MODEL_PROVIDER

    def supports(self, model):
        return model in MODEL_PRICES

    def generate(self, model, question, passages, max_output_tokens):
        started = time.monotonic()
        question_term_counts = _term_counts(question)
        question_terms = set(question_term_counts)
        candidates = []
        for passage_rank, passage in enumerate(passages):
            context_terms = _terms(
                passage.get("document", "") + " " + passage.get("section", "")
            )
            for sentence in re.split(r"(?<=[.!?])\s+", passage["text"]):
                clean_sentence = sentence.strip()
                if not clean_sentence:
                    continue
                overlap = question_terms & _terms(clean_sentence)
                context_overlap = question_terms & context_terms
                if overlap or context_overlap:
                    score = (
                        sum(question_term_counts[term] for term in overlap) * 3
                        + len(context_overlap)
                    )
                    if _is_duration_question(question) and _has_duration(clean_sentence):
                        score += 6
                    if "age" in question_terms and _has_age_rule(clean_sentence):
                        score += 6
                    candidates.append((score, -passage_rank, clean_sentence, passage_rank))

        candidates.sort(key=lambda item: (-item[0], -item[1]))
        sentence_limit = 3 if model == "local-reasoning" else 2
        sentences = []
        seen = set()
        citation_indices = []
        for _, _, sentence, passage_rank in candidates:
            if sentence not in seen:
                sentences.append(sentence)
                seen.add(sentence)
                citation_indices.append(passage_rank)
            if len(sentences) == sentence_limit:
                break

        if not sentences:
            sentences = [passages[0]["text"].strip()]
            citation_indices = [0]
        answer = " ".join(sentences)
        words = answer.split()
        if len(words) > max_output_tokens:
            answer = " ".join(words[:max_output_tokens]).rstrip(" ,;:") + "…"
        input_tokens = _token_count(question + " " + " ".join(p["text"] for p in passages))
        output_tokens = _token_count(answer)
        input_rate, output_rate = MODEL_PRICES[model]
        estimated_cost_usd = (
            input_tokens * input_rate + output_tokens * output_rate
        ) / 1_000
        return ModelResult(
            answer=answer,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=estimated_cost_usd,
            latency_ms=max(0, round((time.monotonic() - started) * 1_000)),
            citation_indices=tuple(citation_indices),
            cost_basis="simulated local profile rates",
            provider_name=self.name,
        )


class ModelRouter:
    def __init__(self, provider: ModelProvider, simple_model, complex_model):
        unknown = {
            model
            for model in {simple_model, complex_model}
            if not provider.supports(model)
        }
        if unknown:
            raise ValueError(f"Provider does not support model profile(s): {', '.join(sorted(unknown))}")
        self.provider = provider
        self.simple_model = simple_model
        self.complex_model = complex_model

    def select(self, question, passages=()):
        terms = _terms(question)
        complex_markers = {
            "compare", "comparison", "exception", "explain", "difference",
            "eligibility", "eligible", "synthesis",
        }
        source_documents = {passage["document"] for passage in passages}
        if len(source_documents) > 1 or len(terms) > 18 or terms & complex_markers:
            return self.complex_model
        return self.simple_model


def _terms(text):
    return set(_term_counts(text))


def _term_counts(text):
    counts = Counter()
    for term in re.findall(r"[a-z0-9]+", text.lower()):
        normalized = _normalize(term)
        if normalized not in _STOP_WORDS:
            counts[normalized] += 1
            for expansion in _ANSWER_EXPANSIONS.get(normalized, set()):
                counts[expansion] += 1
    return counts


def _normalize(term):
    if len(term) > 5 and term.endswith("ies"):
        return term[:-3] + "y"
    if len(term) > 4 and term.endswith("ed"):
        return term[:-2]
    if len(term) > 4 and term.endswith("ing"):
        return term[:-3]
    if len(term) > 3 and term.endswith("s"):
        return term[:-1]
    return term


def _is_duration_question(question):
    return bool(re.search(r"\b(how long|timeline|processing time|turnaround)\b", question.lower()))


def _has_duration(sentence):
    return bool(
        re.search(r"\b\d+\s+(?:business\s+)?(?:days?|weeks?|months?)\b", sentence.lower())
    )


def _has_age_rule(sentence):
    return bool(
        re.search(r"\b(?:under|over|aged?)\s+\d+\b", sentence.lower())
    )


def _token_count(text):
    return len(text.split())


def _response_content(payload):
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise OllamaError("Ollama response did not contain choices")
    message = choices[0].get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise OllamaError("Ollama response did not contain message content")
    return content


def _validated_citations(answer, passages):
    references = {
        (passage["document"], passage["section"]): index
        for index, passage in enumerate(passages)
    }
    indices = []
    for document, section in re.findall(r"\[([^\[\]]+?)\s+§([^\]]+)\]", answer):
        index = references.get((document.strip(), section.strip()))
        if index is None:
            raise OllamaError("Ollama response contained an ungrounded citation")
        if index not in indices:
            indices.append(index)
    return indices
