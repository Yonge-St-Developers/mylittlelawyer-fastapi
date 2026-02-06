"""Integration-style unit test for indexing a single Drive folder."""

from __future__ import annotations

import os
import unittest

from src.rag.indexer import index_all_from_env

class TestIndexerSingle(unittest.TestCase):
    def test_index_rta_act(self) -> None:
        index_name = os.getenv("INDEX_RTA_ACT", "").strip()
        drive_url = os.getenv("INDEX_RTA_ACT_DRIVE_URL", "").strip()
        if not index_name or not drive_url:
            self.fail("Missing INDEX_RTA_ACT or INDEX_RTA_ACT_DRIVE_URL in environment.")

        results = index_all_from_env(index_keys=["rta_act"])
        self.assertIn("rta_act", results)
        self.assertFalse(results["rta_act"].get("skipped", False))


if __name__ == "__main__":
    unittest.main()
