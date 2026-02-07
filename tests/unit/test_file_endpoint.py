"""Unit test for /ai/file endpoint."""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

import app.app as app_module


class _FakeGraph:
    def invoke(self, state):
        return {
            **state,
            "file_json": {
                "file_name": "ltb_form_a1.pdf",
                "form_title": state.get("form_title"),
                "fields": {"unit_address": "123 Main St"},
                "missing_fields": ["landlord_name"],
                "assumptions": ["Landlord mailing address not provided."],
            },
        }


class TestFileEndpoint(unittest.TestCase):
    def test_file_endpoint_returns_json(self) -> None:
        original_builder = app_module.build_file_graph
        app_module.build_file_graph = lambda: _FakeGraph()
        try:
            client = TestClient(app_module.app)
            payload = {
                "session_id": "test-1",
                "form_title": "LTB Form A1",
                "chat_history": [
                    {"role": "user", "content": "I want to file LTB Form A1."}
                ],
            }
            response = client.post("/ai/file", json=payload)
            self.assertEqual(response.status_code, 200)
            body = response.json()
            print("\n=== /ai/file response ===")
            print(body)
            self.assertEqual(body["status"], "success")
            self.assertEqual(body["file_name"], "ltb_form_a1.pdf")
            self.assertIn("file_json", body)
        finally:
            app_module.build_file_graph = original_builder


if __name__ == "__main__":
    unittest.main()
