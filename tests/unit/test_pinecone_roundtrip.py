"""Integration-style unit test for Pinecone roundtrip with embeddings."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

from src.core.pinecone_client import get_pinecone_index
from src.processing.pdf_to_text import convert_pdf_to_text
from src.rag.embedder import embed_query, embed_texts


class TestPineconeRoundtrip(unittest.TestCase):
    def test_embed_upsert_query_roundtrip(self) -> None:
        test_index = os.getenv("PINECONE_TEST_INDEX", "test-index").strip()
        cloud = os.getenv("PINECONE_CLOUD", "aws").strip()
        region = os.getenv("PINECONE_REGION", "us-east-1").strip()

        if not test_index:
            self.skipTest("PINECONE_TEST_INDEX not set")

        pdf_path = Path("/Users/abtinzandi/Downloads/LTB _ Form A1 Instructions.pdf")
        if not pdf_path.exists():
            self.skipTest(f"PDF not found at {pdf_path}")

        text = convert_pdf_to_text(pdf_path, max_pages=1).strip()
        if not text:
            self.fail("PDF text extraction returned empty text")

        vectors = embed_texts([text[:3000]])
        self.assertEqual(len(vectors), 1)
        vector = vectors[0]
        self.assertGreater(len(vector), 0)

        index = get_pinecone_index(
            test_index,
            vector_dim=len(vector),
            similarity_option="cosine",
            cloud=cloud,
            region=region,
        )

        vector_id = "test_a1_instructions_1"
        
        index.upsert(
            [
                {
                    "id": vector_id,
                    "values": vector,
                    "metadata": {
                        "source": "A1 Instructions",
                        "text": text[:2000],
                    },
                }
            ]
        )

        query_vector = embed_query("How to complete Form A1 instructions?")
        result = index.query(vector=query_vector, top_k=1, include_metadata=True)

        matches = getattr(result, "matches", []) or []
        self.assertGreater(len(matches), 0, "No matches returned from Pinecone")

        top = matches[0]
        meta = getattr(top, "metadata", {}) or {}
        retrieved_text = meta.get("text", "")

        self.assertTrue(retrieved_text)
        self.assertIn("Instructions", retrieved_text)

        print("\n=== PINECONE ROUNDTRIP ===")
        print(f"Top score: {getattr(top, 'score', None)}")
        print(f"Retrieved text (first 300 chars): {retrieved_text[:300]}")


if __name__ == "__main__":
    unittest.main()
