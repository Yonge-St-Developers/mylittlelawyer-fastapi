"""Unit test for chunking with the LTB A1 instructions PDF."""

from __future__ import annotations

import unittest
from pathlib import Path

from src.processing.chunking import chunk_pdf


class TestChunkingA1Instructions(unittest.TestCase):
    def test_chunk_pdf_ltb_a1_instructions(self) -> None:
        pdf_path = Path("/Users/abtinzandi/Downloads/LTB _ Form A1 Instructions.pdf")
        if not pdf_path.exists():
            self.skipTest(f"PDF not found at {pdf_path}")

        chunks = chunk_pdf(pdf_path)

        self.assertGreater(len(chunks), 0, "Expected at least one chunk")
        for chunk in chunks:
            self.assertIn("id", chunk)
            self.assertIn("text", chunk)
            self.assertTrue(chunk["text"].strip())

        output_path = Path("data/cache/a1_instructions_chunks.txt")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as f:
            f.write("=== CHUNKS (A1 INSTRUCTIONS) ===\n")
            for chunk in chunks:
                section = chunk.get("section", "").strip() or "(no section)"
                text = chunk.get("text", "")
                f.write(f"\n--- {chunk['id']} | {section} ---\n{text}\n")


if __name__ == "__main__":
    unittest.main()
