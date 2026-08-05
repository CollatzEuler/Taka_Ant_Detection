from __future__ import annotations

import unittest

import numpy as np

from roi_ant_tracker.inference import even_video_dimensions, finalize_detections, fit_overlay_frame


class DetectionFinalizationTests(unittest.TestCase):
    def test_suppresses_close_duplicate_after_nms(self) -> None:
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        detections = finalize_detections(
            frame=frame,
            boxes=[[10, 10, 20, 20], [14, 10, 24, 20]],
            scores=[0.95, 0.80],
            roi_ids=["roi", "roi"],
            nms_threshold=0.95,
            max_detections=10,
            duplicate_center_distance=5.0,
            duplicate_overlap_threshold=0.0,
        )

        self.assertEqual(len(detections), 1)
        self.assertAlmostEqual(detections[0].score, 0.95)

    def test_keeps_nearby_distinct_detections_when_center_gate_is_off(self) -> None:
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        detections = finalize_detections(
            frame=frame,
            boxes=[[10, 10, 20, 20], [22, 10, 32, 20]],
            scores=[0.95, 0.80],
            roi_ids=["roi", "roi"],
            nms_threshold=0.95,
            max_detections=10,
            duplicate_center_distance=0.0,
            duplicate_overlap_threshold=0.80,
        )

        self.assertEqual(len(detections), 2)

    def test_keeps_overlapping_detections_from_different_rois(self) -> None:
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        detections = finalize_detections(
            frame=frame,
            boxes=[[10, 10, 20, 20], [10, 10, 20, 20]],
            scores=[0.95, 0.90],
            roi_ids=["left", "right"],
            nms_threshold=0.5,
            max_detections=10,
        )

        self.assertEqual(len(detections), 2)

    def test_pads_odd_roi_overlay_dimensions_for_video_writer(self) -> None:
        self.assertEqual(even_video_dimensions(282, 265), (282, 266))
        frame = np.ones((265, 282, 3), dtype=np.uint8)
        padded = fit_overlay_frame(frame, 282, 266)

        self.assertEqual(padded.shape, (266, 282, 3))
        self.assertTrue(np.all(padded[:265] == 1))
        self.assertTrue(np.all(padded[265] == 0))


if __name__ == "__main__":
    unittest.main()
