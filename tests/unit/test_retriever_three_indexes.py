"""Integration-style test for retriever graph against configured indexes."""

from __future__ import annotations

import os
import unittest

from src.rag.retriever import build_retriever_graph


class TestRetrieverThreeIndexes(unittest.TestCase):
    def test_retriever_graph_uses_configured_indexes(self) -> None:
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

        configured = {
            "rta_act": os.getenv("INDEX_RTA_ACT", "").strip(),
            "sppa_act": os.getenv("INDEX_SPPA_ACT", "").strip(),
            "ltb_rules": os.getenv("INDEX_LTB_RULES", "").strip(),
            "ltb_practice_directions": os.getenv("INDEX_LTB_PRACTICE_DIRECTIONS", "").strip(),
            "ltb_guidelines": os.getenv("INDEX_LTB_GUIDELINES", "").strip(),
            "ltb_forms": os.getenv("INDEX_LTB_FORMS", "").strip(),
            "form_instructions": os.getenv("INDEX_FORM_INSTRUCTIONS", "").strip(),
        }

        expected = {k for k, v in configured.items() if v}
        self.assertEqual(set(retrieval_results.keys()), expected)

        for name, res in retrieval_results.items():
            matches = getattr(res, "matches", None)
            self.assertIsNotNone(matches, f"Missing matches for index {name}")


if __name__ == "__main__":
    unittest.main()
