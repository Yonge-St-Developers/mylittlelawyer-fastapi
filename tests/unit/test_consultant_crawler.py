"""Unit test for the Playwright CanLII crawler (with debug prints)."""

from __future__ import annotations

import unittest

from src.consultant.crawler import crawl_canlii_case_pages


class TestConsultantCrawler(unittest.TestCase):
    def test_crawl_canlii_pages(self) -> None:
        pages = crawl_canlii_case_pages(max_depth=3, max_pages=6)
        print(f"\nCrawled pages: {len(pages)}")
        for i, page in enumerate(pages, start=1):
            source = page.get("source", "")
            html = page.get("html", "")
            print(f"[{i}] {source}")
            print(f"    html_len={len(html)}")

        self.assertGreater(len(pages), 0, "Expected to crawl at least one page")


if __name__ == "__main__":
    unittest.main()
