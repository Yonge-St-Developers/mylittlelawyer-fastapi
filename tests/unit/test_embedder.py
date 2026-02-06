"""Unit test for Gemini embedder using the A1 instructions PDF."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

from src.processing.pdf_to_text import convert_pdf_to_text
from src.rag.embedder import embed_texts


class TestGeminiEmbedder(unittest.TestCase):
    def test_embedder_on_a1_instructions(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        model_name = os.getenv("GEMINI_EMBEDDING_MODEL", "").strip()
        if not api_key or not model_name:
            self.skipTest("GEMINI_API_KEY or GEMINI_EMBEDDING_MODEL not set")

        pdf_path = Path("/Users/abtinzandi/Downloads/LTB _ Form A1 Instructions.pdf")
        if not pdf_path.exists():
            self.skipTest(f"PDF not found at {pdf_path}")

        text = convert_pdf_to_text(pdf_path, max_pages=1)
        self.assertTrue(text.strip())

        vectors = embed_texts([text[:3000]])
        self.assertEqual(len(vectors), 1)
        self.assertGreater(len(vectors[0]), 0)

        print("\n=== EMBEDDING OUTPUT ===")
        print(f"Vector length: {len(vectors[0])}")
        print(f"First 10 values: {vectors[0][:10]}")


if __name__ == "__main__":
    unittest.main()
