"""Governed tool layer: planners propose tool calls, application code disposes.

A planner (deterministic rules or a local function-calling model) may only
propose calls from ``TOOL_REGISTRY``. Every proposal is validated against the
registered argument schema before dispatch. Approval and execution of service
requests are deliberately absent from the registry, so no planner can reach
them; they stay behind authenticated human actions in the workflow API.
"""

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field


SERVICE_ALIASES = {
    "rental": "RENTAL_ASSISTANCE",
    "rent": "RENTAL_ASSISTANCE",
    "rental assistance": "RENTAL_ASSISTANCE",
    "rental_assistance": "RENTAL_ASSISTANCE",
}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    side_effect: str  # "read" or "write"
    requires_human_approval: bool
    parameters: dict = field(default_factory=dict)

    def as_openai_tool(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": dict(self.parameters),
                    "required": sorted(self.parameters),
                    "additionalProperties": False,
                },
            },
        }


_CUSTOMER = {
    "type": "string",
    "description": "Synthetic customer identifier in C-#### format.",
    "pattern": r"^C-\d{4}$",
}
_SERVICE = {
    "type": "string",
    "description": "Service code.",
    "enum": ["RENTAL_ASSISTANCE"],
}

TOOL_REGISTRY = {
    "check_eligibility": ToolSpec(
        name="check_eligibility",
        description="Read-only eligibility check against the back-end system.",
        side_effect="read",
        requires_human_approval=False,
        parameters={"customer_id": _CUSTOMER, "service_code": _SERVICE},
    ),
    "prepare_service_request": ToolSpec(
        name="prepare_service_request",
        description=(
            "Draft a service request for human review. Never submits it; a "
            "different reviewer must approve before execution."
        ),
        side_effect="write",
        requires_human_approval=True,
        parameters={"customer_id": _CUSTOMER, "service_code": _SERVICE},
    ),
}


class ToolValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ToolProposal:
    name: str
    arguments: dict
    planner: str


def validate_proposal(proposal, question):
    """Reject unknown tools, unexpected arguments, and values not grounded in the request."""
    spec = TOOL_REGISTRY.get(proposal.name)
    if spec is None:
        raise ToolValidationError(f"Tool '{proposal.name}' is not available to the assistant.")
    if not isinstance(proposal.arguments, dict):
        raise ToolValidationError("Tool arguments must be an object.")
    unexpected = set(proposal.arguments) - set(spec.parameters)
    missing = set(spec.parameters) - set(proposal.arguments)
    if unexpected:
        raise ToolValidationError(f"Unexpected tool arguments: {', '.join(sorted(unexpected))}.")
    if missing:
        raise ToolValidationError(f"Missing tool arguments: {', '.join(sorted(missing))}.")
    for key, rule in spec.parameters.items():
        value = proposal.arguments[key]
        if not isinstance(value, str):
            raise ToolValidationError(f"{key} must be a string.")
        if "pattern" in rule and not re.fullmatch(rule["pattern"], value):
            raise ToolValidationError(f"{key} has an invalid format.")
        if "enum" in rule and value not in rule["enum"]:
            raise ToolValidationError(f"{key} is not a supported value.")
    # A planner must not invent or substitute a customer: the identifier has to
    # appear verbatim in what the staff member typed.
    if proposal.arguments["customer_id"] not in question.upper():
        raise ToolValidationError("customer_id must be stated in the request.")
    return spec


_CUSTOMER_RE = re.compile(r"\bC-\d{4}\b", re.IGNORECASE)
_REQUEST_INTENT_RE = re.compile(
    r"\b(raise|submit|create|open|file|prepare|start|lodge|draft)\b.{0,60}"
    r"\b(request|application|case|claim)\b",
    re.IGNORECASE,
)
_ELIGIBILITY_INTENT_RE = re.compile(
    r"\b(eligib\w*|qualif\w*|entitled)\b", re.IGNORECASE
)


def _service_code(question):
    lowered = question.lower()
    for alias, code in sorted(SERVICE_ALIASES.items(), key=lambda item: -len(item[0])):
        if re.search(rf"\b{re.escape(alias)}\b", lowered):
            return code
    return None


class RuleBasedToolPlanner:
    """Deterministic intent detection; the offline default and the model fallback."""

    name = "rules"

    def plan(self, question):
        customer = _CUSTOMER_RE.search(question)
        if not customer:
            return None
        if _REQUEST_INTENT_RE.search(question):
            tool = "prepare_service_request"
        elif _ELIGIBILITY_INTENT_RE.search(question):
            tool = "check_eligibility"
        else:
            return None
        return ToolProposal(
            name=tool,
            arguments={
                "customer_id": customer.group(0).upper(),
                "service_code": _service_code(question) or "RENTAL_ASSISTANCE",
            },
            planner=self.name,
        )


class OllamaToolPlanner:
    """Local function-calling planner; any failure or invalid output falls back to rules."""

    name = "ollama"

    def __init__(self, endpoint, model, timeout_seconds, fallback=None, opener=None):
        self.endpoint = endpoint
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.fallback = fallback or RuleBasedToolPlanner()
        self._opener = opener or urllib.request.urlopen

    def plan(self, question):
        try:
            proposal = self._plan_remote(question)
            if proposal is None:
                # Small models under-call tools; the rules remain a safety net.
                return self.fallback.plan(question)
            validate_proposal(proposal, question)
            return proposal
        except (ToolValidationError, OSError, urllib.error.URLError, TimeoutError,
                ValueError, KeyError, TypeError):
            return self.fallback.plan(question)

    def _plan_remote(self, question):
        body = json.dumps(
            {
                "model": self.model,
                "temperature": 0,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Decide whether the staff request needs a tool. Call a tool "
                            "only when the request names a customer ID and clearly asks "
                            "to check eligibility or to prepare a service request. "
                            "Otherwise reply without a tool call."
                        ),
                    },
                    {"role": "user", "content": question},
                ],
                "tools": [spec.as_openai_tool() for spec in TOOL_REGISTRY.values()],
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self._opener(request, timeout=self.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        message = payload["choices"][0]["message"]
        calls = message.get("tool_calls") or []
        if not calls:
            return None
        if len(calls) != 1:
            raise ToolValidationError("Only one tool call per turn is allowed.")
        function = calls[0]["function"]
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if isinstance(arguments, dict) and isinstance(arguments.get("customer_id"), str):
            arguments = {**arguments, "customer_id": arguments["customer_id"].upper()}
        return ToolProposal(name=function["name"], arguments=arguments, planner=self.name)
