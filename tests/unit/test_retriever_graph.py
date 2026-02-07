"""Integration-style unit test for retriever graph execution."""

from __future__ import annotations

import os
import re
import unittest

from src.rag.retriever import build_retriever_graph


class TestRetrieverGraph(unittest.TestCase):
    def test_retriever_graph_runs(self) -> None:
        _validate_index_envs(self)
        query = "How do I apply to evict a tenant for non-payment of rent in Ontario?"

        compiled = build_retriever_graph()
        result = compiled.invoke(
            {
                "query": query,
                "form": None,
                "chat_history": None,
                "intent": "describing_situation",
            }
        )

        retrieval_results = result.get("retrieval_results", {}) or {}
        self.assertTrue(retrieval_results, "No retrieval results returned.")

        # Ensure each index result has a matches attribute or behaves like a match list.
        for name, res in retrieval_results.items():
            matches = getattr(res, "matches", None)
            self.assertIsNotNone(matches, f"Missing matches for index {name}")

        if os.getenv("DEBUG_RETRIEVER_TEST", "").strip():
            print("\n=== RETRIEVER GRAPH ===")
            print("indexes:", list(retrieval_results.keys()))
            for name, res in retrieval_results.items():
                matches = getattr(res, "matches", []) or []
                print(name, "matches", len(matches))
                if matches:
                    meta = getattr(matches[0], "metadata", {}) or {}
                    text = meta.get("text") or ""
                    print(name, "first_text_preview", text[:160].replace("\n", " "))

    def test_retriever_graph_six_indexes_for_form_questions(self) -> None:
        _validate_index_envs(self)
        query = "What is the filing fee and where do I submit this form?"

        compiled = build_retriever_graph()
        result = compiled.invoke(
            {
                "query": query,
                "form": "LTB Form A1",
                "chat_history": None,
                "intent": "asking_about_form",
            }
        )

        retrieval_results = result.get("retrieval_results", {}) or {}
        self.assertTrue(retrieval_results, "No retrieval results returned.")

        expected = {
            "sppa_act",
            "ltb_rules",
            "ltb_practice_directions",
            "ltb_guidelines",
        }
        self.assertEqual(set(retrieval_results.keys()), expected)


if __name__ == "__main__":
    unittest.main()


def _validate_index_envs(test_case: unittest.TestCase) -> None:
    keys = [
        "INDEX_RTA_ACT",
        "INDEX_SPPA_ACT",
        "INDEX_LTB_RULES",
        "INDEX_LTB_PRACTICE_DIRECTIONS",
        "INDEX_LTB_GUIDELINES",
        "INDEX_LTB_FORMS",
        "INDEX_FORM_INSTRUCTIONS",
    ]
    invalid: list[str] = []
    missing: list[str] = []
    name_re = re.compile(r"^[a-z0-9-]+$")
    for key in keys:
        value = os.getenv(key, "").strip()
        if not value:
            missing.append(key)
            continue
        if not name_re.match(value):
            invalid.append(f"{key}={value}")

    if missing:
        test_case.fail(f"Missing index env vars: {', '.join(missing)}")
    if invalid:
        test_case.fail(
            "Invalid Pinecone index names (must be lowercase letters, numbers, or '-'): "
            + ", ".join(invalid)
        )
