import math
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    model_mode: str = "deterministic"
    simple_model: str = "local-small"
    complex_model: str = "local-reasoning"
    ollama_endpoint: str = "http://localhost:11434/v1/chat/completions"
    ollama_timeout_seconds: float = 8.0
    ollama_guardian_model: str = ""
    ollama_simple_model: str = "qwen3.5:4b"
    ollama_complex_model: str = "qwen3.5:9b"
    tool_planner: str = "rules"
    max_input_tokens: int = 2_000
    max_output_tokens: int = 300
    max_conversation_turns: int = 20
    max_session_cost_usd: float = 0.05
    max_tool_attempts: int = 8
    request_expiry_seconds: int = 86_400

    def __post_init__(self):
        for name in (
            "max_input_tokens",
            "max_output_tokens",
            "max_conversation_turns",
            "max_tool_attempts",
            "request_expiry_seconds",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            type(self.max_session_cost_usd) not in (int, float)
            or not math.isfinite(self.max_session_cost_usd)
            or self.max_session_cost_usd <= 0
        ):
            raise ValueError("max_session_cost_usd must be a positive finite number")
        if self.tool_planner not in {"rules", "ollama"}:
            raise ValueError("tool_planner must be rules or ollama")
        if self.model_mode not in {"deterministic", "ollama"}:
            raise ValueError("model_mode must be deterministic or ollama")
        if (
            type(self.ollama_timeout_seconds) not in (int, float)
            or not math.isfinite(self.ollama_timeout_seconds)
            or self.ollama_timeout_seconds <= 0
        ):
            raise ValueError("ollama_timeout_seconds must be a positive finite number")
        for name in ("simple_model", "complex_model"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")

    @classmethod
    def from_env(cls):
        return cls(
            model_mode=os.getenv("ASSISTANT_MODEL_MODE", "deterministic").lower(),
            simple_model=os.getenv("ASSISTANT_SIMPLE_MODEL", "local-small"),
            complex_model=os.getenv("ASSISTANT_COMPLEX_MODEL", "local-reasoning"),
            ollama_endpoint=os.getenv(
                "ASSISTANT_OLLAMA_ENDPOINT",
                "http://localhost:11434/v1/chat/completions",
            ),
            ollama_timeout_seconds=_env_float("ASSISTANT_OLLAMA_TIMEOUT_SECONDS", 8.0),
            ollama_guardian_model=os.getenv("ASSISTANT_OLLAMA_GUARDIAN_MODEL", ""),
            ollama_simple_model=os.getenv("ASSISTANT_OLLAMA_SIMPLE_MODEL", "qwen3.5:4b"),
            ollama_complex_model=os.getenv("ASSISTANT_OLLAMA_COMPLEX_MODEL", "qwen3.5:9b"),
            tool_planner=os.getenv("ASSISTANT_TOOL_PLANNER", "rules").lower(),
            max_input_tokens=_env_int("ASSISTANT_MAX_INPUT_TOKENS", 2_000),
            max_output_tokens=_env_int("ASSISTANT_MAX_OUTPUT_TOKENS", 300),
            max_conversation_turns=_env_int("ASSISTANT_MAX_CONVERSATION_TURNS", 20),
            max_session_cost_usd=_env_float("ASSISTANT_MAX_SESSION_COST_USD", 0.05),
            max_tool_attempts=_env_int("ASSISTANT_MAX_TOOL_ATTEMPTS", 8),
            request_expiry_seconds=_env_int("ASSISTANT_REQUEST_EXPIRY_SECONDS", 86_400),
        )


def _env_int(name, default):
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _env_float(name, default):
    value = float(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value
