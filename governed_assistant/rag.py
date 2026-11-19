import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_INJECTION_RE = re.compile(
    r"\b(ignore|disregard)\b.{0,60}\b(instructions?|rules?|polic(?:y|ies))\b"
    r"|\b(reveal|disclose|print|expose)\b.{0,40}\b(system prompt|confidential|secret|credentials?)\b"
    r"|\b(bypass|skip|disable)\b.{0,30}\b(approval|security|review)\b"
    r"|\b(auto[- ]?approve|approve automatically|execute automatically)\b.{0,60}\b(request|application|action)\b"
    r"|\b(execute|submit|create)\b.{0,40}\b(without|skipping)\b.{0,20}\b(approval|review)\b"
    r"|\b(act as|you are now)\b.{0,30}\b(system|developer|administrator)\b",
    re.IGNORECASE,
)
_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "for", "from",
    "how", "i", "in", "is", "it", "me", "of", "on", "or", "the", "this", "to",
    "current", "we", "what", "when", "where", "which", "who", "why", "with", "you",
}
_TERM_EXPANSIONS = {
    "age": {"child", "under"},
    "child": {"age", "under"},
    "childcare": {"child"},
}


def is_prompt_injection(text):
    return bool(_INJECTION_RE.search(text))


@dataclass(frozen=True)
class PolicyChunk:
    document: str
    title: str
    section: str
    text: str


class PolicyIndex:
    def __init__(self, policy_directory):
        self.policy_directory = Path(policy_directory)
        self.filtered_instruction_count = 0
        self.chunks = self._load_chunks()
        if not self.chunks:
            raise ValueError(f"No usable policy content found in {self.policy_directory}")
        documents = [
            f"{chunk.title} {chunk.section} {chunk.text}" for chunk in self.chunks
        ]
        document_frequency = Counter(
            term for text in documents for term in set(_expanded_terms(text))
        )
        document_count = len(documents)
        self._inverse_document_frequency = {
            term: math.log((document_count + 1) / (frequency + 1)) + 1
            for term, frequency in document_frequency.items()
        }
        self._vectors = [
            self._embed(text, self._inverse_document_frequency) for text in documents
        ]
        self._content_vectors = [
            self._embed(chunk.text, self._inverse_document_frequency)
            for chunk in self.chunks
        ]

    def retrieve(self, query, limit=5):
        query_vector = self._embed(query, self._inverse_document_frequency)
        query_terms = set(_expanded_terms(query))
        mentioned_documents = {
            chunk.document
            for chunk in self.chunks
            if set(_expanded_terms(chunk.title)).issubset(query_terms)
        }
        ranked = []
        for chunk, vector, content_vector in zip(
            self.chunks, self._vectors, self._content_vectors
        ):
            if mentioned_documents and chunk.document not in mentioned_documents:
                continue
            score = 0.4 * _cosine(query_vector, vector) + 0.6 * _cosine(
                query_vector, content_vector
            )
            if score >= 0.12:
                ranked.append((score, chunk))
        ranked.sort(key=lambda item: (-item[0], item[1].document, item[1].section))
        return [
            {
                "document": chunk.title,
                "filename": chunk.document,
                "section": chunk.section,
                "text": chunk.text,
                "score": round(score, 4),
            }
            for score, chunk in ranked[:limit]
        ]

    def _load_chunks(self):
        if not self.policy_directory.exists():
            raise ValueError(f"Policy directory does not exist: {self.policy_directory}")
        if not self.policy_directory.is_dir():
            raise ValueError(f"Policy path is not a directory: {self.policy_directory}")
        chunks = []
        for path in sorted(self.policy_directory.glob("*.md")):
            if not path.is_file():
                raise ValueError(f"Policy path is not a regular file: {path}")
            title = path.stem.replace("-", " ").title()
            section = "Overview"
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeError) as error:
                raise ValueError(f"Unable to read policy document: {path.name}") from error
            for line_number, line in enumerate(lines, start=1):
                if line.startswith("# "):
                    title = _metadata_value(line[2:], path, line_number, "title")
                elif line.startswith("## "):
                    section = _metadata_value(line[3:], path, line_number, "section")
                elif line.startswith("#"):
                    raise ValueError(
                        f"Malformed policy heading in {path.name}:{line_number}"
                    )
                elif line.strip() and not line.startswith("#"):
                    for sentence in re.split(r"(?<=[.!?])\s+", line.strip()):
                        cleaned = sentence.strip()
                        if cleaned and _INJECTION_RE.search(cleaned):
                            self.filtered_instruction_count += 1
                        elif cleaned:
                            chunks.append(
                                PolicyChunk(path.name, title, section, cleaned)
                            )
        if not chunks:
            raise ValueError(f"No usable policy content found in {self.policy_directory}")
        return chunks

    @staticmethod
    def _embed(text, inverse_document_frequency):
        terms = Counter(_expanded_terms(text))
        vector = {}
        for token, count in terms.items():
            # Stable, dependency-free feature hashing makes this index reproducible.
            bucket = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16) % 4096
            term_frequency = 1.0 + math.log(count)
            weight = term_frequency * inverse_document_frequency.get(token, 1.0)
            vector[bucket] = vector.get(bucket, 0.0) + weight
        return vector


def _expanded_terms(text):
    terms = []
    for token in _TOKEN_RE.findall(text.lower()):
        normalized = _normalize(token)
        if len(normalized) > 1 and normalized not in _STOP_WORDS:
            terms.append(normalized)
            terms.extend(_TERM_EXPANSIONS.get(normalized, ()))
    return terms


def _normalize(token):
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def _metadata_value(value, path, line_number, field):
    normalized = " ".join(value.split())
    if not normalized:
        raise ValueError(f"Empty policy {field} in {path.name}:{line_number}")
    return normalized


def _cosine(left, right):
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(key, 0.0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0
