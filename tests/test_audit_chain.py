import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from governed_assistant.models import LocalExtractiveProvider, PromptBlocked
from governed_assistant.service import AssistantService


ROOT = Path(__file__).resolve().parents[1]
POLICIES = ROOT / "data" / "policies"


class AuditChainTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.database_path = Path(self.temp_dir.name) / "assistant.sqlite3"
        self.service = AssistantService(POLICIES, self.database_path)

    def _workflow(self):
        request = self.service.prepare_request("C-1001", "RENTAL_ASSISTANCE", "staff-1")
        self.service.approve_request(request["id"], "reviewer-1")
        self.service.execute_request(request["id"])
        self.service.ask("What documents are required for rental assistance?", "c-1", "staff-1")

    def test_untampered_chain_verifies(self):
        self._workflow()
        verification = self.service.verify_audit_chain()
        self.assertTrue(verification["valid"])
        self.assertGreaterEqual(verification["checked"], 5)
        self.assertEqual(64, len(verification["head_hash"]))

    def test_edited_event_is_detected(self):
        self._workflow()
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.execute(
                "UPDATE audit_events SET event_type = 'request_rejected' "
                "WHERE event_type = 'request_approved'"
            )
            connection.commit()
            tampered_id = connection.execute(
                "SELECT id FROM audit_events WHERE event_type = 'request_rejected'"
            ).fetchone()[0]
        verification = self.service.verify_audit_chain()
        self.assertFalse(verification["valid"])
        self.assertEqual(tampered_id, verification["first_invalid_id"])

    def test_deleted_event_is_detected(self):
        self._workflow()
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.execute("DELETE FROM audit_events WHERE event_type = 'request_approved'")
            connection.commit()
        self.assertFalse(self.service.verify_audit_chain()["valid"])

    def test_pre_chain_database_is_backfilled_on_startup(self):
        legacy_path = Path(self.temp_dir.name) / "legacy.sqlite3"
        with closing(sqlite3.connect(legacy_path)) as connection:
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
            connection.execute(
                "INSERT INTO audit_events (timestamp, event_type, actor_role, request_id, details_json) "
                "VALUES ('2026-01-01T00:00:00+00:00', 'request_approved', 'reviewer', NULL, '{}')"
            )
            connection.commit()
        service = AssistantService(POLICIES, legacy_path)
        service.ask("What documents are required for rental assistance?", "c-1", "staff-1")
        verification = service.verify_audit_chain()
        self.assertTrue(verification["valid"])
        self.assertEqual(2, verification["checked"])


class GuardianRefusalTests(unittest.TestCase):
    def test_provider_guardrail_block_is_refused_and_audited(self):
        class BlockingProvider(LocalExtractiveProvider):
            def generate(self, *args, **kwargs):
                raise PromptBlocked("guardian_model")

        with tempfile.TemporaryDirectory() as directory:
            service = AssistantService(
                POLICIES,
                Path(directory) / "assistant.sqlite3",
                model_provider=BlockingProvider(),
            )
            result = service.ask("What documents are required for rental assistance?", "c", "s")
            self.assertIn("can't follow", result["answer"])
            self.assertEqual([], result["citations"])
            blocked = [
                event for event in service.audit_events()
                if event["event_type"] == "prompt_injection_blocked"
            ]
            self.assertEqual("guardian_model", blocked[0]["details"]["guardrail"])


if __name__ == "__main__":
    unittest.main()
