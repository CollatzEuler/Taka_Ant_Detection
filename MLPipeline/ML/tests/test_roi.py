from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from roi_ant_tracker.roi import load_rois


class RoiTests(unittest.TestCase):
    def test_loads_multiple_rectangles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rois.json"
            path.write_text(
                json.dumps(
                    {
                        "rectangles": [
                            {"id": "left", "x": 10, "y": 20, "width": 30, "height": 40},
                            {"id": "right", "x": 60, "y": 20, "width": 25, "height": 35},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            rois = load_rois(path, frame_width=100, frame_height=100)
        self.assertEqual([roi.roi_id for roi in rois], ["left", "right"])
        self.assertEqual(rois[0].area, 1200)
        self.assertTrue(rois[0].contains((10, 20)))
        self.assertFalse(rois[0].contains((40, 60)))
        self.assertEqual(rois[0].padded_bounds(15, 100, 100), (0, 5, 55, 75))

    def test_accepts_legacy_crop_json_as_single_roi(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "crop.json"
            path.write_text(
                json.dumps({"crop_x": 1, "crop_y": 2, "crop_width": 3, "crop_height": 4}),
                encoding="utf-8",
            )
            rois = load_rois(path, frame_width=10, frame_height=10)
        self.assertEqual(len(rois), 1)
        self.assertEqual(rois[0].roi_id, "roi_001")
        self.assertEqual((rois[0].x, rois[0].y, rois[0].width, rois[0].height), (1, 2, 3, 4))

    def test_rejects_out_of_frame_roi(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(
                json.dumps({"rectangles": [{"id": "bad", "x": 8, "y": 8, "width": 5, "height": 5}]}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_rois(path, frame_width=10, frame_height=10)


if __name__ == "__main__":
    unittest.main()
