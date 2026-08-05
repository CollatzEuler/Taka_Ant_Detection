from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


SCRIPT = Path(__file__).parents[2] / "Preprocessing" / "exclusion_select.py"
SPEC = importlib.util.spec_from_file_location("exclusion_select", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ExclusionMaskTests(unittest.TestCase):
    def test_blocks_and_restores_mask_pixels(self) -> None:
        mask = np.full((20, 30), 255, dtype=np.uint8)
        MODULE.block_rectangle(mask, 5, 4, 10, 8)
        self.assertTrue(np.all(mask[4:12, 5:15] == 0))
        MODULE.paint_line(mask, (7, 7), (7, 7), 3, True)
        self.assertEqual(int(mask[7, 7]), 255)

    def test_loads_binary_mask_from_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mask = np.full((10, 12), 255, dtype=np.uint8)
            mask[:, :3] = 0
            cv2.imwrite(str(root / "mask.png"), mask)
            (root / "mask.json").write_text('{"mask_path": "mask.png"}', encoding="utf-8")
            loaded = MODULE.load_exclusion_mask(root / "mask.json", 12, 10)
        self.assertEqual(int(np.count_nonzero(loaded == 0)), 30)


if __name__ == "__main__":
    unittest.main()
