import json
import logging
import os
import re
import signal
import subprocess
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import parse_qs, urlparse

from .service import (
    ApprovalRequired,
    AssistantService,
    InvalidTransition,
    LimitExceeded,
    NotFound,
    ServiceUnavailable,
    ValidationError,
)


LOGGER = logging.getLogger("governed_assistant.api")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_ROOT = os.path.join(ROOT, "web")
MAX_BODY_BYTES = 32_768
DEFAULT_PID_PATH = os.path.join(ROOT, ".data", "assistant.pid")
_REQUEST_ROUTE = re.compile(r"^/api/requests/([0-9a-fA-F-]{36})(?:/(approve|reject|execute))?$")
_SAFE_ROUTES = {
    "/",
    "/app.js",
    "/styles.css",
    "/api/health",
    "/api/audit",
    "/api/audit/verify",
    "/api/requests",
    "/api/chat",
    "/api/eligibility",
}
_MIME_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
}


def _safe_route_label(path):
    """Return a route template suitable for logs, never a caller-supplied path."""
    if path in _SAFE_ROUTES:
        return path
    match = _REQUEST_ROUTE.fullmatch(path)
    if match:
        action = match.group(2)
        route = "/api/requests/{request_id}"
        return f"{route}/{action}" if action else route
    return "<unmatched>"


def _process_command(pid):
    """Return a process command line, or an empty string when it cannot be inspected."""
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip()


def _listening_process_ids(port):
    """Return PIDs listening on a TCP port when the platform exposes lsof."""
    try:
        result = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    process_ids = []
    for value in result.stdout.splitlines():
        try:
            process_ids.append(int(value.strip()))
        except ValueError:
            continue
    return tuple(process_ids)


def _terminate_process(pid):
    """Terminate a verified assistant process and wait for it to leave the process table."""
    LOGGER.warning(
        "lifecycle operation=server_recovery phase=started reason=previous_process pid=%d",
        pid,
    )
    try:
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            except PermissionError:
                LOGGER.warning(
                    "lifecycle operation=server_recovery phase=failed reason=permission"
                )
                return
            time.sleep(0.05)
        else:
            os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError:
        LOGGER.exception(
            "lifecycle operation=server_recovery phase=failed reason=termination"
        )
        return
    LOGGER.info(
        "lifecycle operation=server_recovery phase=completed previous_pid=%d",
        pid,
    )


def _stop_previous_server(pid_path):
    """Stop only a prior process recorded by this application."""
    try:
        with open(pid_path, encoding="ascii") as file:
            pid = int(file.read().strip())
    except (OSError, ValueError):
        return

    if pid == os.getpid():
        return
    command = _process_command(pid)
    if not command:
        return
    if "governed_assistant" not in command:
        LOGGER.warning(
            "lifecycle operation=server_recovery phase=skipped reason=pid_not_owned"
        )
        return
    _terminate_process(pid)


def _stop_owned_processes_on_port(port):
    """Recover an older assistant process that was started before PID tracking existed."""
    for pid in _listening_process_ids(port):
        if pid == os.getpid():
            continue
        command = _process_command(pid)
        if "governed_assistant" in command:
            _terminate_process(pid)
        elif command:
            LOGGER.warning(
                "lifecycle operation=server_recovery phase=skipped reason=port_owner_not_owned"
            )


def _write_pid(pid_path):
    os.makedirs(os.path.dirname(pid_path) or ".", exist_ok=True)
    with open(pid_path, "w", encoding="ascii") as file:
        file.write(f"{os.getpid()}\n")


def _remove_pid(pid_path):
    try:
        with open(pid_path, encoding="ascii") as file:
            recorded_pid = int(file.read().strip())
    except (OSError, ValueError):
        return
    if recorded_pid == os.getpid():
        try:
            os.remove(pid_path)
        except FileNotFoundError:
            pass


def _handle_shutdown_signal(signum, frame):
    """Convert process termination into the normal cleanup path."""
    raise KeyboardInterrupt


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """Serve the local API with request threads that do not block shutdown."""

    daemon_threads = True
    allow_reuse_address = True


def make_server(service, host="127.0.0.1", port=8000):
    """Build the demo server around an injected service instance."""

    class RequestHandler(BaseHTTPRequestHandler):
        server_version = "GovernedAssistant/1.0"

        def do_GET(self):
            """Serve static pages and read-only resources, logging only safe route metadata."""
            parsed_path = urlparse(self.path)
            route = parsed_path.path
            safe_route = _safe_route_label(route)
            LOGGER.info(
                "lifecycle operation=api_get phase=started route=%s",
                safe_route,
            )
            if route == "/":
                self._send_file("index.html")
                return
            if route in {"/app.js", "/styles.css"}:
                self._send_file(route.lstrip("/"))
                return
            if route == "/api/health":
                self._send_json(200, {"status": "ok"})
                return
            if route == "/api/audit":
                try:
                    events = service.audit_events()
                    LOGGER.info(
                        "lifecycle operation=audit_list phase=completed event_count=%d",
                        len(events),
                    )
                    self._send_json(200, {"events": events})
                except Exception:
                    LOGGER.exception("Audit history retrieval failed.")
                    self._send_json(500, {"error": "Audit history is temporarily unavailable."})
                return
            if route == "/api/audit/verify":
                try:
                    verification = service.verify_audit_chain()
                    LOGGER.info(
                        "lifecycle operation=audit_verify phase=completed valid=%s checked=%d",
                        verification["valid"],
                        verification["checked"],
                    )
                    self._send_json(200, verification)
                except Exception:
                    LOGGER.exception("Audit chain verification failed.")
                    self._send_json(500, {"error": "Audit verification is temporarily unavailable."})
                return
            if route == "/api/requests":
                query = parse_qs(parsed_path.query, keep_blank_values=True)
                state_values = query.get("state", ["PENDING_APPROVAL"])
                if len(state_values) != 1:
                    LOGGER.warning(
                        "lifecycle operation=request_list phase=failed reason=multiple_states"
                    )
                    self._send_json(400, {"error": "Only one request state may be selected."})
                    return
                state = None if state_values[0] == "all" else state_values[0]
                try:
                    requests = service.list_requests(state=state)
                    LOGGER.info(
                        "lifecycle operation=request_list phase=completed state=%s count=%d",
                        state or "all",
                        len(requests),
                    )
                    self._send_json(200, {"requests": requests})
                except ValidationError as error:
                    LOGGER.warning(
                        "lifecycle operation=request_list phase=failed reason=invalid_state"
                    )
                    self._send_json(400, {"error": str(error)})
                return
            match = _REQUEST_ROUTE.fullmatch(route)
            if match and match.group(2) is None:
                try:
                    request = service.get_request(match.group(1))
                    LOGGER.info(
                        "lifecycle operation=request_read phase=completed state=%s",
                        request["state"],
                    )
                    self._send_json(200, request)
                except NotFound as error:
                    LOGGER.info(
                        "lifecycle operation=request_read phase=failed reason=not_found"
                    )
                    self._send_json(404, {"error": str(error)})
                except ValidationError as error:
                    LOGGER.warning(
                        "lifecycle operation=request_read phase=failed reason=invalid_id"
                    )
                    self._send_json(400, {"error": str(error)})
                return
            self._send_json(404, {"error": "Not found."})

        def do_POST(self):
            """Dispatch policy, tool, and request-transition calls without logging payloads."""
            route = urlparse(self.path).path
            safe_route = _safe_route_label(route)
            LOGGER.info(
                "lifecycle operation=api_post phase=started route=%s",
                safe_route,
            )
            try:
                payload = self._read_json()
                if route == "/api/chat":
                    LOGGER.info("lifecycle operation=chat phase=started")
                    result = service.ask(
                        payload.get("question"),
                        payload.get("conversation_id"),
                        payload.get("actor_id"),
                    )
                    usage = result["usage"]
                    LOGGER.info(
                        "lifecycle operation=chat phase=completed model=%s "
                        "citation_count=%d latency_ms=%d simulated_cost_usd=%.8f",
                        result["model"] or "none",
                        len(result["citations"]),
                        usage["latency_ms"],
                        usage["estimated_cost_usd"],
                    )
                    self._send_json(200, result)
                    return
                if route == "/api/eligibility":
                    LOGGER.info("lifecycle operation=eligibility_check phase=started")
                    result = service.verify_eligibility(
                        payload.get("customer_id"),
                        payload.get("service_code"),
                        payload.get("actor_id"),
                        payload.get("conversation_id"),
                    )
                    LOGGER.info("lifecycle operation=eligibility_check phase=completed")
                    self._send_json(200, result)
                    return
                if route == "/api/requests":
                    LOGGER.info("lifecycle operation=request_prepare phase=started")
                    result = service.prepare_request(
                        payload.get("customer_id"),
                        payload.get("service_code"),
                        payload.get("actor_id"),
                        payload.get("conversation_id"),
                    )
                    LOGGER.info(
                        "lifecycle operation=request_prepare phase=completed state=%s",
                        result["state"],
                    )
                    self._send_json(201, result)
                    return

                match = _REQUEST_ROUTE.fullmatch(route)
                if not match or match.group(2) is None:
                    self._send_json(404, {"error": "Not found."})
                    return
                request_id, action = match.groups()
                if action == "approve":
                    LOGGER.info(
                        "lifecycle operation=request_review phase=started action=approve"
                    )
                    result = service.approve_request(request_id, payload.get("reviewer_id"))
                elif action == "reject":
                    LOGGER.info(
                        "lifecycle operation=request_review phase=started action=reject"
                    )
                    result = service.reject_request(request_id, payload.get("reviewer_id"))
                else:
                    LOGGER.info("lifecycle operation=request_execute phase=started")
                    result = service.execute_request(
                        request_id, payload.get("actor_id")
                    )
                operation = "request_execute" if action == "execute" else "request_review"
                LOGGER.info(
                    "lifecycle operation=%s phase=completed action=%s state=%s",
                    operation,
                    action,
                    result["state"],
                )
                self._send_json(200, result)
            except (json.JSONDecodeError, UnicodeDecodeError):
                LOGGER.warning(
                    "lifecycle operation=api_post phase=failed route=%s "
                    "status=400 reason=invalid_json",
                    safe_route,
                )
                self._send_json(400, {"error": "Request body must be valid JSON."})
            except ValidationError as error:
                LOGGER.warning(
                    "lifecycle operation=api_post phase=failed route=%s "
                    "status=400 reason=validation",
                    safe_route,
                )
                self._send_json(400, {"error": str(error)})
            except NotFound as error:
                LOGGER.info(
                    "lifecycle operation=api_post phase=failed route=%s "
                    "status=404 reason=not_found",
                    safe_route,
                )
                self._send_json(404, {"error": str(error)})
            except LimitExceeded as error:
                LOGGER.info(
                    "lifecycle operation=api_post phase=failed route=%s "
                    "status=429 reason=limit_exceeded",
                    safe_route,
                )
                self._send_json(429, {"error": str(error)})
            except (ApprovalRequired, InvalidTransition) as error:
                LOGGER.info(
                    "lifecycle operation=api_post phase=blocked route=%s "
                    "status=409 reason=invalid_transition",
                    safe_route,
                )
                self._send_json(409, {"error": str(error)})
            except ServiceUnavailable as error:
                LOGGER.warning(
                    "lifecycle operation=api_post phase=failed route=%s "
                    "status=503 reason=service_unavailable",
                    safe_route,
                )
                self._send_json(503, {"error": str(error)})
            except Exception:
                LOGGER.exception(
                    "Unhandled API request failure route=%s",
                    safe_route,
                )
                self._send_json(500, {"error": "The request could not be completed."})

        def _read_json(self):
            """Read one bounded JSON object; do not log its contents."""
            raw_length = self.headers.get("Content-Length", "")
            try:
                length = int(raw_length)
            except ValueError as error:
                raise ValidationError("A valid Content-Length header is required.") from error
            if length <= 0 or length > MAX_BODY_BYTES:
                raise ValidationError("Request body must be between 1 and 32768 bytes.")
            body = self.rfile.read(length)
            payload = json.loads(body)
            if not isinstance(payload, dict):
                raise ValidationError("Request body must be a JSON object.")
            return payload

        def _send_file(self, name):
            path = os.path.join(WEB_ROOT, name)
            try:
                with open(path, "rb") as file:
                    content = file.read()
            except OSError:
                LOGGER.exception("Static application file could not be read.")
                self._send_json(500, {"error": "The application UI is unavailable."})
                return
            extension = os.path.splitext(name)[1]
            self.send_response(200)
            self._set_security_headers()
            self.send_header("Content-Type", _MIME_TYPES[extension])
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def _send_json(self, status, value):
            content = json.dumps(value, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
            self.send_response(status)
            self._set_security_headers()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def _set_security_headers(self):
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            self.send_header("Cache-Control", "no-store")

        def log_request(self, code="-", size="-"):
            """Log access metadata using a route template, omitting IDs and query strings."""
            route = _safe_route_label(urlparse(self.path).path)
            LOGGER.info(
                "http_request method=%s route=%s status=%s response_bytes=%s",
                self.command,
                route,
                code,
                size,
            )

        def log_message(self, format_string, *args):
            # BaseHTTPRequestHandler diagnostics can include raw request lines or values.
            LOGGER.debug("http_server_diagnostic format=%s", format_string)

    return ThreadingHTTPServer((host, port), RequestHandler)


def run():
    """Configure logging, construct the service, and serve until interrupted."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    database_path = os.getenv("ASSISTANT_DATABASE_PATH", os.path.join(ROOT, ".data", "assistant.sqlite3"))
    service = AssistantService(
        policy_directory=os.path.join(ROOT, "data", "policies"),
        database_path=database_path,
    )
    host = os.getenv("ASSISTANT_HOST", "127.0.0.1")
    port = int(os.getenv("ASSISTANT_PORT", "8000"))
    pid_path = os.getenv("ASSISTANT_PID_PATH", DEFAULT_PID_PATH)
    _stop_previous_server(pid_path)
    _stop_owned_processes_on_port(port)
    server = make_server(service, host, port)
    _write_pid(pid_path)
    LOGGER.info("Assistant demo listening on http://%s:%s", host, port)
    previous_sigterm_handler = signal.signal(signal.SIGTERM, _handle_shutdown_signal)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Shutting down assistant demo.")
    finally:
        server.server_close()
        _remove_pid(pid_path)
        signal.signal(signal.SIGTERM, previous_sigterm_handler)
