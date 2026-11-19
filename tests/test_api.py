import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

from governed_assistant.api import (
    _stop_owned_processes_on_port,
    _stop_previous_server,
    make_server,
)
from governed_assistant.service import AssistantService


ROOT = Path(__file__).resolve().parents[1]


class AssistantApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.service = AssistantService(
            policy_directory=ROOT / "data" / "policies",
            database_path=Path(self.temp_dir.name) / "assistant.sqlite3",
        )
        self.server = make_server(self.service, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        self.addCleanup(self._stop_server)
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def _stop_server(self):
        self.server.shutdown()
        self.thread.join(timeout=3)
        self.server.server_close()

    def _post(self, path, payload):
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            response = urllib.request.urlopen(request, timeout=3)
        except urllib.error.HTTPError as error:
            with error:
                return error.code, json.loads(error.read())
        with response:
            return response.status, json.loads(response.read())

    def test_policy_endpoint_returns_citation_and_local_provider(self):
        status, result = self._post(
            "/api/chat",
            {
                "question": "What documents are required for rental assistance?",
                "conversation_id": "http-test",
                "actor_id": "staff-test",
            },
        )

        self.assertEqual(200, status)
        self.assertTrue(result["citations"])
        self.assertEqual("local-extractive", result["provider"])

    def test_chat_lifecycle_log_omits_prompt_and_caller_identifiers(self):
        question = "What documents are required for rental assistance?"
        actor_id = "private-staff-identifier"
        conversation_id = "private-conversation-identifier"

        with self.assertLogs("governed_assistant.api", level="INFO") as captured:
            status, result = self._post(
                "/api/chat",
                {
                    "question": question,
                    "conversation_id": conversation_id,
                    "actor_id": actor_id,
                },
            )

        logs = "\n".join(captured.output)
        self.assertEqual(200, status)
        self.assertIn("operation=chat phase=completed", logs)
        self.assertIn("model=local-small", logs)
        self.assertIn("http_request method=POST route=/api/chat status=200", logs)
        for sensitive_value in (question, actor_id, conversation_id):
            self.assertNotIn(sensitive_value, logs)

    def test_request_lifecycle_logs_states_without_customer_or_request_ids(self):
        actor_id = "private-requester-identifier"
        reviewer_id = "private-reviewer-identifier"

        with self.assertLogs("governed_assistant.api", level="INFO") as captured:
            status, request = self._post(
                "/api/requests",
                {
                    "customer_id": "C-1001",
                    "service_code": "RENTAL_ASSISTANCE",
                    "actor_id": actor_id,
                    "conversation_id": "private-request-conversation",
                },
            )
            self.assertEqual(201, status)
            request_id = request["id"]
            status, approved = self._post(
                f"/api/requests/{request_id}/approve",
                {"reviewer_id": reviewer_id},
            )
            self.assertEqual(200, status)
            self.assertEqual("APPROVED", approved["state"])
            status, executed = self._post(
                f"/api/requests/{request_id}/execute",
                {"actor_id": actor_id},
            )

        logs = "\n".join(captured.output)
        self.assertEqual(200, status)
        self.assertEqual("EXECUTED", executed["state"])
        self.assertIn("operation=request_prepare phase=completed state=PENDING_APPROVAL", logs)
        self.assertIn("operation=request_review phase=completed action=approve state=APPROVED", logs)
        self.assertIn("operation=request_execute phase=started", logs)
        self.assertIn("operation=request_execute phase=completed action=execute state=EXECUTED", logs)
        for sensitive_value in (
            "C-1001",
            actor_id,
            reviewer_id,
            "private-request-conversation",
            request_id,
        ):
            self.assertNotIn(sensitive_value, logs)

    def test_http_execution_endpoint_rejects_pending_request(self):
        request = self.service.prepare_request(
            customer_id="C-1001",
            service_code="RENTAL_ASSISTANCE",
            actor_id="staff-test",
        )

        with self.assertLogs("governed_assistant.api", level="INFO") as captured:
            status, result = self._post(
                f"/api/requests/{request['id']}/execute",
                {"actor_id": "staff-test"},
            )

        self.assertEqual(409, status)
        self.assertIn("approved", result["error"].lower())
        self.assertEqual(0, self.service.request_api.call_count)
        logs = "\n".join(captured.output)
        self.assertIn("operation=request_execute phase=started", logs)
        self.assertIn(
            "operation=api_post phase=blocked route=/api/requests/{request_id}/execute",
            logs,
        )
        self.assertNotIn(request["id"], logs)

    def test_request_queue_endpoint_lists_pending_requests(self):
        created = self.service.prepare_request(
            customer_id="C-1001",
            service_code="RENTAL_ASSISTANCE",
            actor_id="staff-test",
        )
        with urllib.request.urlopen(
            self.base_url + "/api/requests?state=PENDING_APPROVAL", timeout=3
        ) as response:
            result = json.loads(response.read())

        self.assertEqual(200, response.status)
        self.assertEqual([created["id"]], [request["id"] for request in result["requests"]])

    def test_ui_is_served_with_restrictive_security_headers(self):
        with urllib.request.urlopen(self.base_url + "/", timeout=3) as response:
            content = response.read().decode("utf-8")
            self.assertEqual(200, response.status)
            self.assertIn("Content-Security-Policy", response.headers)
            self.assertIn("Service Assistant", content)

    def test_invalid_utf8_json_is_a_client_error(self):
        request = urllib.request.Request(
            self.base_url + "/api/chat",
            data=b"\xff",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with self.assertLogs("governed_assistant.api", level="WARNING") as captured:
            with self.assertRaises(urllib.error.HTTPError) as raised:
                urllib.request.urlopen(request, timeout=3)

        with raised.exception as response:
            self.assertEqual(400, response.code)
            self.assertIn("valid JSON", json.loads(response.read())["error"])
        self.assertIn(
            "operation=api_post phase=failed route=/api/chat status=400 reason=invalid_json",
            "\n".join(captured.output),
        )

    def test_server_recovery_stops_only_its_recorded_process(self):
        pid_path = Path(self.temp_dir.name) / "assistant.pid"
        pid_path.write_text("43210\n", encoding="ascii")

        with patch("governed_assistant.api._process_command", return_value="python -m governed_assistant"), \
             patch("governed_assistant.api.os.kill", side_effect=[None, ProcessLookupError]):
            _stop_previous_server(str(pid_path))

        self.assertTrue(pid_path.exists())

    def test_server_recovery_does_not_stop_unowned_process(self):
        pid_path = Path(self.temp_dir.name) / "assistant.pid"
        pid_path.write_text("43210\n", encoding="ascii")

        with patch("governed_assistant.api._process_command", return_value="python unrelated_server.py"), \
             patch("governed_assistant.api.os.kill") as kill:
            _stop_previous_server(str(pid_path))

        kill.assert_not_called()

    def test_server_recovery_finds_owned_process_without_pid_file(self):
        with patch("governed_assistant.api._listening_process_ids", return_value=(43210,)), \
             patch("governed_assistant.api._process_command", return_value="python -m governed_assistant"), \
             patch("governed_assistant.api.os.kill", side_effect=[None, ProcessLookupError]):
            _stop_owned_processes_on_port(8000)


if __name__ == "__main__":
    unittest.main()
