"""Integration-style test for querying LTB rules index."""

from __future__ import annotations

import os
import unittest

from src.rag.embedder import embed_query
from src.core.pinecone_client import get_pinecone_index


class TestRetrieverQueryLtbRules(unittest.TestCase):
    def test_query_ltb_rules(self) -> None:
        index_name = os.getenv("INDEX_LTB_RULES", "").strip()
        if not index_name:
            self.fail("INDEX_LTB_RULES not set")

        query = "LTB rules about eviction for non-payment of rent"
        vector = embed_query(query)
        index = get_pinecone_index(index_name, vector_dim=len(vector))
        result = index.query(vector=vector, top_k=1, include_metadata=True)

        matches = getattr(result, "matches", []) or []
        self.assertGreater(len(matches), 0, "No matches returned from Pinecone")

        print("\n=== LTB RULES QUERY ===")
        for i, match in enumerate(matches, 1):
                meta = getattr(match, "metadata", {}) or {}
                text = (meta.get("text") or "").replace("\n", " ")
                print(f"{i}. score=", getattr(match, "score", None))
                print("   text:", text[:300])
            


if __name__ == "__main__":
    unittest.main()
