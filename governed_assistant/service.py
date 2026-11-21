import hashlib
import json
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .backends import (
    SYNTHETIC_CUSTOMER_IDS,
    BackendUnavailable,
    MockEligibilityAPI,
    MockServiceRequestAPI,
)
from .config import Settings
from .models import LocalExtractiveProvider, ModelRouter, OllamaProvider, PromptBlocked
from .rag import PolicyIndex, is_prompt_injection
from .tools import (
    OllamaToolPlanner,
    RuleBasedToolPlanner,
    ToolValidationError,
    validate_proposal,
)


SERVICE_CODES = {"RENTAL_ASSISTANCE"}
REQUEST_STATES = {
    "DRAFT",
    "PENDING_APPROVAL",
    "APPROVED",
    "REJECTED",
    "EXPIRED",
    "EXECUTING",
    "EXECUTED",
    "FAILED",
}
AUDIT_ROLES = {"staff", "reviewer", "system"}
AUDIT_DETAIL_FIELDS = {
    "policy_answered": {
        "model",
        "provider",
        "retrieved_references",
        "input_tokens",
        "output_tokens",
        "estimated_cost_usd",
        "latency_ms",
        "cost_basis",
    },
    "prompt_injection_blocked": {"guardrail", "outcome"},
    "policy_instruction_filtered": {"filtered_count"},
    "eligibility_checked": {
        "service_code",
        "eligible",
        "outcome",
        "decision_source",
    },
    "request_drafted": {"service_code", "state"},
    "request_prepared": {"service_code", "state"},
    "request_approved": {"state"},
    "request_rejected": {"state"},
    "request_expired": set(),
    "request_execution_started": set(),
    "request_execution_failed": {"reason"},
    "request_executed": {"external_status"},
    "tool_call_planned": {"tool", "planner", "side_effect", "outcome"},
    "session_budget_blocked": {
        "estimated_cost_usd",
        "session_budget_usd",
        "cost_basis",
    },
}


AUDIT_GENESIS_HASH = "0" * 64


def _audit_entry_hash(prev_hash, timestamp, event_type, actor_role, request_id, details_json):
    """Hash an audit row together with its predecessor to make edits detectable."""
    canonical = json.dumps(
        [prev_hash, timestamp, event_type, actor_role, request_id, details_json],
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class AssistantError(Exception):
    pass


class ValidationError(AssistantError):
    pass


class NotFound(AssistantError):
    pass


class InvalidTransition(AssistantError):
    pass


class ApprovalRequired(InvalidTransition):
    pass


class LimitExceeded(AssistantError):
    pass


class ServiceUnavailable(AssistantError):
    pass


class AssistantService:
    def __init__(
        self,
        policy_directory,
        database_path,
        settings=None,
        eligibility_api=None,
        request_api=None,
        model_provider=None,
        clock=None,
        tool_planner=None,
    ):
        self.settings = settings if settings is not None else Settings.from_env()
        self.index = PolicyIndex(policy_directory)
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.eligibility_api = (
            eligibility_api if eligibility_api is not None else MockEligibilityAPI()
        )
        self.request_api = (
            request_api if request_api is not None else MockServiceRequestAPI()
        )
        self.clock = clock if clock is not None else (lambda: datetime.now(timezone.utc))
        provider = model_provider
        simple_model = self.settings.simple_model
        complex_model = self.settings.complex_model
        if provider is None and self.settings.model_mode == "ollama":
            provider = OllamaProvider(
                endpoint=self.settings.ollama_endpoint,
                timeout_seconds=self.settings.ollama_timeout_seconds,
                guardian_model=self.settings.ollama_guardian_model,
                simple_model=self.settings.ollama_simple_model,
                complex_model=self.settings.ollama_complex_model,
                fallback=LocalExtractiveProvider(),
            )
            simple_model = self.settings.ollama_simple_model
            complex_model = self.settings.ollama_complex_model
        self.router = ModelRouter(
            provider if provider is not None else LocalExtractiveProvider(),
            simple_model,
            complex_model,
        )
        if tool_planner is not None:
            self.tool_planner = tool_planner
        elif self.settings.tool_planner == "ollama":
            self.tool_planner = OllamaToolPlanner(
                endpoint=self.settings.ollama_endpoint,
                model=self.settings.ollama_simple_model,
                timeout_seconds=self.settings.ollama_timeout_seconds,
            )
        else:
            self.tool_planner = RuleBasedToolPlanner()
        self._initialize_database()
        if self.index.filtered_instruction_count:
            self._audit(
                "policy_instruction_filtered",
                "system",
                filtered_count=self.index.filtered_instruction_count,
            )

    def ask(self, question, conversation_id, actor_id):
        question = _required_text(question, "question", max_length=4_000)
        conversation_id = _identifier(conversation_id, "conversation_id")
        actor_id = _identifier(actor_id, "actor_id")
        input_tokens = _token_count(question)
        if input_tokens > self.settings.max_input_tokens:
            raise LimitExceeded("The question exceeds the configured input limit.")

        if is_prompt_injection(question):
            return self._blocked_response("prompt_injection", input_tokens)

        proposal = self.tool_planner.plan(question)
        if proposal is not None:
            return self._run_tool(proposal, question, conversation_id, actor_id, input_tokens)

        started = time.monotonic()
        passages = self.index.retrieve(question)
        context_tokens = _token_count(
            question + " " + " ".join(passage["text"] for passage in passages)
        )
        if context_tokens > self.settings.max_input_tokens:
            raise LimitExceeded("The retrieved context exceeds the configured input limit.")
        selected_model = None
        cited_passages = []
        if passages:
            selected_model = self.router.select(question, passages)
            if (
                self.settings.model_mode == "ollama"
                and self._session_cost(conversation_id) >= self.settings.max_session_cost_usd
            ):
                result = LocalExtractiveProvider().generate(
                    "local-reasoning" if selected_model == self.settings.complex_model else "local-small",
                    question,
                    passages,
                    self.settings.max_output_tokens,
                )
            else:
                try:
                    result = self.router.provider.generate(
                        selected_model,
                        question,
                        passages,
                        self.settings.max_output_tokens,
                    )
                except PromptBlocked as blocked:
                    return self._blocked_response(blocked.guardrail, input_tokens)
            answer = result.answer
            usage = {
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "estimated_cost_usd": result.estimated_cost_usd,
                "latency_ms": result.latency_ms,
                "cost_basis": result.cost_basis,
            }
            if (
                self.settings.model_mode == "ollama"
                and self._session_cost(conversation_id) + usage["estimated_cost_usd"]
                > self.settings.max_session_cost_usd
            ):
                fallback_model = (
                    "local-reasoning"
                    if selected_model == self.settings.complex_model
                    else "local-small"
                )
                result = LocalExtractiveProvider().generate(
                    fallback_model,
                    question,
                    passages,
                    self.settings.max_output_tokens,
                )
                usage = {
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "estimated_cost_usd": result.estimated_cost_usd,
                    "latency_ms": result.latency_ms,
                    "cost_basis": f"deterministic budget fallback: {result.cost_basis}",
                }
                answer = result.answer
            provider = result.provider_name or self.router.provider.name
            cited_passages = [
                passages[index] for index in result.citation_indices
            ]
        else:
            answer = "I could not find that information in the available approved policies."
            usage = {
                "input_tokens": input_tokens,
                "output_tokens": _token_count(answer),
                "estimated_cost_usd": 0.0,
                "latency_ms": round((time.monotonic() - started) * 1_000),
                "cost_basis": "no model call",
            }
            provider = None

        try:
            self._record_model_usage(
                conversation_id,
                "staff",
                selected_model,
                provider,
                usage,
                cited_passages,
            )
        except LimitExceeded as error:
            if "model-cost limit" in str(error):
                self._audit(
                    "session_budget_blocked",
                    "system",
                    estimated_cost_usd=usage["estimated_cost_usd"],
                    session_budget_usd=self.settings.max_session_cost_usd,
                    cost_basis=usage["cost_basis"],
                )
            raise
        return {
            "answer": answer,
            "citations": [
                {
                    "document": passage["document"],
                    "section": passage["section"],
                    "excerpt": passage["text"],
                }
                for passage in cited_passages
            ],
            "model": selected_model,
            "provider": provider,
            "usage": usage,
        }

    def _run_tool(self, proposal, question, conversation_id, actor_id, input_tokens):
        """Validate a planner proposal, then dispatch it through the governed service methods."""
        try:
            spec = validate_proposal(proposal, question)
        except ToolValidationError as error:
            self._audit(
                "tool_call_planned",
                "system",
                tool=proposal.name,
                planner=proposal.planner,
                outcome="rejected_invalid_proposal",
            )
            return self._tool_response(
                f"I could not run that action: {error}", proposal, None, None, input_tokens
            )

        arguments = proposal.arguments
        eligibility = self.verify_eligibility(
            arguments["customer_id"],
            arguments["service_code"],
            actor_id,
            session_id=conversation_id,
        )
        request = None
        if spec.name == "check_eligibility":
            outcome = eligibility["outcome"].lower()
            answer = (
                f"Back-end eligibility result for {arguments['customer_id']}: "
                f"{eligibility['outcome']}. {eligibility['reason']} "
                "This is a system decision, not policy guidance."
            )
        elif eligibility["outcome"] != "ELIGIBLE":
            # Business precondition: never queue a request the back end would refuse.
            outcome = "not_drafted_ineligible"
            answer = (
                f"I did not draft a request for {arguments['customer_id']}: the eligibility "
                f"check returned {eligibility['outcome']}. {eligibility['reason']}"
            )
        else:
            request = self.prepare_request(
                arguments["customer_id"],
                arguments["service_code"],
                actor_id,
                session_id=conversation_id,
            )
            outcome = "drafted_pending_approval"
            answer = (
                f"Eligibility confirmed. I drafted request {request['id'][:8]} for "
                f"{arguments['customer_id']}; it is waiting for a different reviewer to "
                "approve it. Nothing has been submitted."
            )
        self._audit(
            "tool_call_planned",
            "system",
            tool=spec.name,
            planner=proposal.planner,
            side_effect=spec.side_effect,
            outcome=outcome,
        )
        return self._tool_response(answer, proposal, eligibility, request, input_tokens, spec)

    def _tool_response(self, answer, proposal, eligibility, request, input_tokens, spec=None):
        return {
            "answer": answer,
            "citations": [],
            "model": None,
            "provider": None,
            "tool_call": {
                "name": proposal.name,
                "planner": proposal.planner,
                "arguments": proposal.arguments,
                "side_effect": spec.side_effect if spec else None,
                "requires_human_approval": spec.requires_human_approval if spec else None,
                "eligibility": eligibility,
                "request": request,
            },
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": _token_count(answer),
                "estimated_cost_usd": 0.0,
                "latency_ms": 0,
                "cost_basis": f"tool planner: {proposal.planner}",
            },
        }

    def _blocked_response(self, guardrail, input_tokens):
        self._audit(
            "prompt_injection_blocked",
            "staff",
            guardrail=guardrail,
            outcome="blocked",
        )
        return {
            "answer": "I can't follow instructions that attempt to override the assistant's safety rules.",
            "citations": [],
            "model": None,
            "provider": None,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": 0,
                "estimated_cost_usd": 0.0,
                "latency_ms": 0,
                "cost_basis": "no model call",
            },
        }

    def verify_eligibility(self, customer_id, service_code, actor_id, session_id=None):
        customer_id = _customer_id(customer_id)
        service_code = _service_code(service_code)
        actor_id = _identifier(actor_id, "actor_id")
        self._reserve_tool_attempt(session_id or actor_id)
        result = self.eligibility_api.verify(customer_id, service_code)
        _validate_eligibility_result(result)
        self._audit(
            "eligibility_checked",
            "staff",
            service_code=service_code,
            eligible=result["eligible"],
            outcome=result["outcome"],
            decision_source=result["decision_source"],
        )
        return {
            **result,
            "customer_id": customer_id,
            "service_code": service_code,
            "decision_type": "back-end decision; not a policy-only assessment",
        }

    def prepare_request(self, customer_id, service_code, actor_id, session_id=None):
        customer_id = _customer_id(customer_id)
        service_code = _service_code(service_code)
        actor_id = _identifier(actor_id, "actor_id")
        if customer_id not in SYNTHETIC_CUSTOMER_IDS:
            raise ValidationError("Unknown synthetic customer ID.")
        self._reserve_tool_attempt(session_id or actor_id)
        request_id = str(uuid.uuid4())
        now = self._now()
        expires_at = now + timedelta(seconds=self.settings.request_expiry_seconds)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO service_requests
                    (id, customer_id, service_code, created_by, state, created_at, updated_at, expires_at)
                VALUES (?, ?, ?, ?, 'DRAFT', ?, ?, ?)
                """,
                (
                    request_id,
                    customer_id,
                    service_code,
                    actor_id,
                    now.isoformat(),
                    now.isoformat(),
                    expires_at.isoformat(),
                ),
            )
            self._insert_audit(
                connection,
                "request_drafted",
                "staff",
                request_id=request_id,
                service_code=service_code,
                state="DRAFT",
            )
            connection.execute(
                "UPDATE service_requests SET state = 'PENDING_APPROVAL' WHERE id = ? AND state = 'DRAFT'",
                (request_id,),
            )
            self._insert_audit(
                connection,
                "request_prepared",
                "staff",
                request_id=request_id,
                service_code=service_code,
                state="PENDING_APPROVAL",
            )
        return self.get_request(request_id)

    def approve_request(self, request_id, reviewer_id):
        reviewer_id = _identifier(reviewer_id, "reviewer_id")
        return self._review_request(request_id, reviewer_id, approve=True)

    def reject_request(self, request_id, reviewer_id):
        reviewer_id = _identifier(reviewer_id, "reviewer_id")
        return self._review_request(request_id, reviewer_id, approve=False)

    def execute_request(self, request_id, actor_id=None):
        request_id = _request_id(request_id)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM service_requests WHERE id = ?", (request_id,)
            ).fetchone()
        if row is None:
            raise NotFound("Service request was not found.")
        actor_id = _identifier(actor_id or row["created_by"], "actor_id")
        self._reserve_tool_attempt(actor_id)

        now = self._now()
        expired = False
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM service_requests WHERE id = ?", (request_id,)
            ).fetchone()
            if row["state"] != "APPROVED":
                raise ApprovalRequired("Only an approved request can be executed.")
            if _parse_time(row["expires_at"]) <= now:
                connection.execute(
                    "UPDATE service_requests SET state = 'EXPIRED', updated_at = ? WHERE id = ?",
                    (now.isoformat(), request_id),
                )
                self._insert_audit(
                    connection, "request_expired", "staff", request_id=request_id
                )
                expired = True
            else:
                connection.execute(
                    "UPDATE service_requests SET state = 'EXECUTING', updated_at = ? WHERE id = ?",
                    (now.isoformat(), request_id),
                )
                self._insert_audit(
                    connection,
                    "request_execution_started",
                    "staff",
                    request_id=request_id,
                )
        if expired:
            raise InvalidTransition("The approval expired before execution.")

        try:
            result = self.request_api.create(
                idempotency_key=request_id,
                customer_id=row["customer_id"],
                service_code=row["service_code"],
            )
        except BackendUnavailable as error:
            self._set_request_state(
                request_id,
                "FAILED",
                "request_execution_failed",
                "staff",
                reason="mock service unavailable",
            )
            raise ServiceUnavailable(str(error)) from error
        except Exception as error:
            # Never leave a request in EXECUTING when an adapter violates its contract.
            self._set_request_state(
                request_id,
                "FAILED",
                "request_execution_failed",
                "staff",
                reason="mock service adapter failure",
            )
            raise ServiceUnavailable("The request service failed unexpectedly.") from error

        with self._connect() as connection:
            connection.execute(
                """
                UPDATE service_requests
                SET state = 'EXECUTED', external_reference = ?, external_status = ?, updated_at = ?
                WHERE id = ? AND state = 'EXECUTING'
                """,
                (
                    result["external_reference"],
                    result["external_status"],
                    self._now().isoformat(),
                    request_id,
                ),
            )
            self._insert_audit(
                connection,
                "request_executed",
                "staff",
                request_id=request_id,
                external_status=result["external_status"],
            )
        return self.get_request(request_id)

    def get_request(self, request_id):
        request_id = _request_id(request_id)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM service_requests WHERE id = ?", (request_id,)
            ).fetchone()
        if row is None:
            raise NotFound("Service request was not found.")
        return dict(row)

    def list_requests(self, state="PENDING_APPROVAL", limit=100):
        if state is not None and state not in REQUEST_STATES:
            raise ValidationError("Unsupported request state.")
        if type(limit) is not int or not 1 <= limit <= 500:
            raise ValidationError("limit must be an integer from 1 to 500.")
        with self._connect() as connection:
            if state is None:
                rows = connection.execute(
                    "SELECT * FROM service_requests ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT * FROM service_requests
                    WHERE state = ? ORDER BY created_at DESC LIMIT ?
                    """,
                    (state, limit),
                ).fetchall()
        return [dict(row) for row in rows]

    def audit_events(self, limit=100):
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, timestamp, event_type, actor_role, request_id, details_json
                FROM audit_events ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        events = []
        for row in rows:
            event = dict(row)
            event["details"] = json.loads(event.pop("details_json"))
            events.append(event)
        return events

    def _review_request(self, request_id, reviewer_id, approve):
        request_id = _request_id(request_id)
        now = self._now()
        target_state = "APPROVED" if approve else "REJECTED"
        event_type = "request_approved" if approve else "request_rejected"
        expired = False
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM service_requests WHERE id = ?", (request_id,)
            ).fetchone()
            if row is None:
                raise NotFound("Service request was not found.")
            if row["state"] != "PENDING_APPROVAL":
                raise InvalidTransition("Only a pending request can be reviewed.")
            if row["created_by"] == reviewer_id:
                raise InvalidTransition("The requester cannot approve or reject their own request.")
            if _parse_time(row["expires_at"]) <= now:
                connection.execute(
                    "UPDATE service_requests SET state = 'EXPIRED', updated_at = ? WHERE id = ?",
                    (now.isoformat(), request_id),
                )
                self._insert_audit(
                    connection, "request_expired", "reviewer", request_id=request_id
                )
                expired = True
            else:
                connection.execute(
                    """
                    UPDATE service_requests
                    SET state = ?, reviewer_id = ?, updated_at = ?
                    WHERE id = ? AND state = 'PENDING_APPROVAL'
                    """,
                    (target_state, reviewer_id, now.isoformat(), request_id),
                )
                self._insert_audit(
                    connection,
                    event_type,
                    "reviewer",
                    request_id=request_id,
                    state=target_state,
                )
        if expired:
            raise InvalidTransition("The request expired before it could be reviewed.")
        return self.get_request(request_id)

    def _reserve_tool_attempt(self, session_id):
        session_id = _identifier(session_id, "session_id")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "INSERT OR IGNORE INTO assistant_sessions (id) VALUES (?)", (session_id,)
            )
            row = connection.execute(
                "SELECT tool_attempts FROM assistant_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if row["tool_attempts"] >= self.settings.max_tool_attempts:
                raise LimitExceeded("The session tool invocation limit has been reached.")
            connection.execute(
                "UPDATE assistant_sessions SET tool_attempts = tool_attempts + 1 WHERE id = ?",
                (session_id,),
            )

    def _record_model_usage(self, session_id, actor_role, model, provider, usage, passages):
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "INSERT OR IGNORE INTO assistant_sessions (id) VALUES (?)", (session_id,)
            )
            row = connection.execute(
                "SELECT turn_count, estimated_cost_usd FROM assistant_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if row["turn_count"] >= self.settings.max_conversation_turns:
                raise LimitExceeded("The conversation turn limit has been reached.")
            total_cost = row["estimated_cost_usd"] + usage["estimated_cost_usd"]
            if total_cost > self.settings.max_session_cost_usd:
                raise LimitExceeded("The session model-cost limit has been reached.")
            connection.execute(
                """
                UPDATE assistant_sessions
                SET turn_count = turn_count + 1, estimated_cost_usd = ?
                WHERE id = ?
                """,
                (total_cost, session_id),
            )
            self._insert_audit(
                connection,
                "policy_answered",
                actor_role,
                model=model,
                provider=provider,
                retrieved_references=[
                    {"document": p["document"], "section": p["section"]}
                    for p in passages
                ],
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                estimated_cost_usd=usage["estimated_cost_usd"],
                latency_ms=usage["latency_ms"],
                cost_basis=usage["cost_basis"],
            )

    def _session_cost(self, session_id):
        with self._connect() as connection:
            row = connection.execute(
                "SELECT estimated_cost_usd FROM assistant_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        return row["estimated_cost_usd"] if row else 0.0

    def _set_request_state(self, request_id, state, event_type, actor_role, **details):
        with self._connect() as connection:
            connection.execute(
                "UPDATE service_requests SET state = ?, updated_at = ? WHERE id = ?",
                (state, self._now().isoformat(), request_id),
            )
            self._insert_audit(
                connection,
                event_type,
                actor_role,
                request_id=request_id,
                **details,
            )

    def _audit(self, event_type, actor_role, **details):
        with self._connect() as connection:
            self._insert_audit(connection, event_type, actor_role, **details)

    def _insert_audit(self, connection, event_type, actor_role, **details):
        request_id = details.pop("request_id", None)
        safe_role = actor_role if actor_role in AUDIT_ROLES else "system"
        safe_details = _sanitize_audit_details(event_type, details)
        if not connection.in_transaction:
            # Serialise chain extension so concurrent writers cannot fork the chain.
            connection.execute("BEGIN IMMEDIATE")
        previous = connection.execute(
            "SELECT entry_hash FROM audit_events ORDER BY id DESC LIMIT 1"
        ).fetchone()
        prev_hash = previous["entry_hash"] if previous else AUDIT_GENESIS_HASH
        timestamp = self._now().isoformat()
        details_json = json.dumps(safe_details, separators=(",", ":"), sort_keys=True)
        entry_hash = _audit_entry_hash(
            prev_hash, timestamp, event_type, safe_role, request_id, details_json
        )
        connection.execute(
            """
            INSERT INTO audit_events
                (timestamp, event_type, actor_role, request_id, details_json, prev_hash, entry_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (timestamp, event_type, safe_role, request_id, details_json, prev_hash, entry_hash),
        )

    def verify_audit_chain(self):
        """Recompute every audit hash; report the first row that was altered or removed."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, timestamp, event_type, actor_role, request_id, details_json,
                       prev_hash, entry_hash
                FROM audit_events ORDER BY id
                """
            ).fetchall()
        expected_prev = AUDIT_GENESIS_HASH
        for row in rows:
            recomputed = _audit_entry_hash(
                expected_prev,
                row["timestamp"],
                row["event_type"],
                row["actor_role"],
                row["request_id"],
                row["details_json"],
            )
            if row["prev_hash"] != expected_prev or row["entry_hash"] != recomputed:
                return {
                    "valid": False,
                    "checked": len(rows),
                    "first_invalid_id": row["id"],
                    "head_hash": None,
                }
            expected_prev = row["entry_hash"]
        return {
            "valid": True,
            "checked": len(rows),
            "first_invalid_id": None,
            "head_hash": expected_prev if rows else None,
        }

    def _initialize_database(self):
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS service_requests (
                    id TEXT PRIMARY KEY,
                    customer_id TEXT NOT NULL,
                    service_code TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    reviewer_id TEXT,
                    state TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    external_reference TEXT,
                    external_status TEXT
                );
                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    actor_role TEXT NOT NULL,
                    request_id TEXT,
                    details_json TEXT NOT NULL,
                    prev_hash TEXT,
                    entry_hash TEXT
                );
                CREATE TABLE IF NOT EXISTS assistant_sessions (
                    id TEXT PRIMARY KEY,
                    turn_count INTEGER NOT NULL DEFAULT 0,
                    tool_attempts INTEGER NOT NULL DEFAULT 0,
                    estimated_cost_usd REAL NOT NULL DEFAULT 0
                );
                """
            )
            columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(audit_events)")
            }
            if "actor_id" in columns:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute("ALTER TABLE audit_events RENAME TO audit_events_legacy")
                connection.execute(
                    """
                    CREATE TABLE audit_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        actor_role TEXT NOT NULL,
                        request_id TEXT,
                        details_json TEXT NOT NULL
                    )
                    """
                )
                legacy_rows = connection.execute(
                    """
                    SELECT id, timestamp, event_type, request_id, details_json
                    FROM audit_events_legacy ORDER BY id
                    """
                ).fetchall()
                for row in legacy_rows:
                    try:
                        legacy_details = json.loads(row["details_json"])
                    except (TypeError, json.JSONDecodeError):
                        legacy_details = {}
                    safe_details = _sanitize_audit_details(
                        row["event_type"],
                        legacy_details if isinstance(legacy_details, dict) else {},
                    )
                    connection.execute(
                        """
                        INSERT INTO audit_events
                            (id, timestamp, event_type, actor_role, request_id, details_json)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            row["id"],
                            row["timestamp"],
                            row["event_type"],
                            _legacy_audit_role(row["event_type"]),
                            row["request_id"],
                            json.dumps(safe_details, separators=(",", ":"), sort_keys=True),
                        ),
                    )
                connection.execute("DROP TABLE audit_events_legacy")
            elif "actor_role" not in columns:
                raise sqlite3.DatabaseError("Unsupported audit_events schema.")
        self._ensure_audit_chain()

    def _ensure_audit_chain(self):
        """Add hash-chain columns to older databases and backfill existing rows."""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(audit_events)")
            }
            for column in ("prev_hash", "entry_hash"):
                if column not in columns:
                    connection.execute(f"ALTER TABLE audit_events ADD COLUMN {column} TEXT")
            unchained = connection.execute(
                "SELECT COUNT(*) AS n FROM audit_events WHERE entry_hash IS NULL"
            ).fetchone()["n"]
            if not unchained:
                return
            prev_hash = AUDIT_GENESIS_HASH
            for row in connection.execute(
                """
                SELECT id, timestamp, event_type, actor_role, request_id, details_json
                FROM audit_events ORDER BY id
                """
            ).fetchall():
                entry_hash = _audit_entry_hash(
                    prev_hash,
                    row["timestamp"],
                    row["event_type"],
                    row["actor_role"],
                    row["request_id"],
                    row["details_json"],
                )
                connection.execute(
                    "UPDATE audit_events SET prev_hash = ?, entry_hash = ? WHERE id = ?",
                    (prev_hash, entry_hash, row["id"]),
                )
                prev_hash = entry_hash

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.database_path, timeout=5)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _now(self):
        current = self.clock()
        return current if current.tzinfo else current.replace(tzinfo=timezone.utc)


def _validate_eligibility_result(result):
    allowed_outcomes = {"ELIGIBLE", "INELIGIBLE", "NOT_FOUND", "UNVERIFIABLE"}
    if not isinstance(result, dict):
        raise ServiceUnavailable("The eligibility service returned an invalid response.")
    eligible = result.get("eligible")
    outcome = result.get("outcome")
    reason = result.get("reason")
    source = result.get("decision_source")
    expected_eligibility = {
        "ELIGIBLE": True,
        "INELIGIBLE": False,
        "NOT_FOUND": None,
        "UNVERIFIABLE": None,
    }
    if (
        outcome not in allowed_outcomes
        or eligible is not expected_eligibility.get(outcome)
        or not isinstance(reason, str)
        or not reason.strip()
        or len(reason) > 500
        or not isinstance(source, str)
        or not source.strip()
        or len(source) > 80
    ):
        raise ServiceUnavailable("The eligibility service returned an invalid response.")


def _sanitize_audit_details(event_type, details):
    allowed_fields = AUDIT_DETAIL_FIELDS.get(event_type, set())
    return {
        key: value
        for key, value in details.items()
        if key in allowed_fields
    }


def _legacy_audit_role(event_type):
    if event_type in {"request_approved", "request_rejected"}:
        return "reviewer"
    if event_type == "policy_instruction_filtered":
        return "system"
    return "staff"


def _required_text(value, name, max_length):
    if not isinstance(value, str) or not value.strip() or len(value) > max_length:
        raise ValidationError(f"{name} must be a non-empty string of at most {max_length} characters.")
    return value.strip()


def _identifier(value, name):
    text = _required_text(value, name, 80)
    if not re.fullmatch(r"[A-Za-z0-9._-]+", text):
        raise ValidationError(f"{name} contains invalid characters.")
    return text


def _customer_id(value):
    text = _required_text(value, "customer_id", 20)
    if not re.fullmatch(r"C-\d{4}", text):
        raise ValidationError("customer_id must use the synthetic C-#### format.")
    return text


def _service_code(value):
    text = _required_text(value, "service_code", 50).upper()
    if text not in SERVICE_CODES:
        raise ValidationError("Unsupported service code.")
    return text


def _request_id(value):
    text = _required_text(value, "request_id", 36)
    try:
        return str(uuid.UUID(text))
    except ValueError as error:
        raise ValidationError("request_id must be a UUID.") from error


def _parse_time(value):
    return datetime.fromisoformat(value)


def _token_count(text):
    return len(text.split())
