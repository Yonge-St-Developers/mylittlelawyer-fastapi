"""Unit test for local PDF indexer with debug output."""

from __future__ import annotations

import unittest

from src.consultant.local_indexer import index_local_pdfs


class TestConsultantLocalIndexer(unittest.TestCase):
    def test_index_local_pdfs(self) -> None:
        result = index_local_pdfs()
        print("\nLocal indexer result:", result)
        self.assertIn("indexed", result)


if __name__ == "__main__":
    unittest.main()
