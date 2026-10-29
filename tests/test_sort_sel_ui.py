"""Sort board must not rebuild the whole queue on Select all."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SortSelUiTests(unittest.TestCase):
    def test_select_all_patches_selection_in_place(self):
        html = (ROOT / "static" / "index.html").read_text()
        addon = (ROOT / "faceid-addon" / "static" / "index.html").read_text()
        self.assertEqual(html, addon)
        self.assertIn("function syncSelUI()", html)
        # Select all / click / clear must not innerHTML the board.
        self.assertIn("t.dataset.selcard", html)
        block = html.split("if(t.dataset.selcard)")[1].split("if(t.dataset.expand)")[0]
        self.assertIn("syncSelUI()", block)
        self.assertNotIn("renderSort()", block)
        self.assertIn("OPEN_MAX=24", html)


if __name__ == "__main__":
    unittest.main()
