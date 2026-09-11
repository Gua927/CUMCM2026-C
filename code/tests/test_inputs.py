from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "src"))

from cumcm_c.data import load_attachment1, load_attachment2, load_attachment3, load_attachment4


class InputStructureTests(unittest.TestCase):
    def test_attachment1(self) -> None:
        self.assertEqual(len(load_attachment1()), 144)

    def test_attachment2(self) -> None:
        sheets = load_attachment2()
        self.assertEqual(set(sheets), {"小区负载", "光伏发电实际功率"})
        self.assertTrue(all(frame.shape == (365, 145) for frame in sheets.values()))

    def test_attachment3(self) -> None:
        self.assertEqual(load_attachment3().shape, (1460, 26))

    def test_attachment4(self) -> None:
        self.assertEqual(load_attachment4().shape, (365, 145))


if __name__ == "__main__":
    unittest.main()
