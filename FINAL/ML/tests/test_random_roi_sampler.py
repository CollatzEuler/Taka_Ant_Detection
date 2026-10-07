from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


SCRIPT = Path(__file__).parents[2] / "Annotation" / "sample_random_roi_frames.py"
SPEC = importlib.util.spec_from_file_location("sample_random_roi_frames", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class RandomRoiSamplerTests(unittest.TestCase):
    def test_roi_size_stays_within_reference_jitter(self) -> None:
        rng = np.random.default_rng(2)
        sizes = [MODULE.random_roi_size([(300, 270)], 0.08, 1000, 1000, rng) for _ in range(100)]
        self.assertTrue(all(276 <= width <= 324 for width, _height in sizes))
        self.assertTrue(all(248 <= height <= 292 for _width, height in sizes))

    def test_integral_mask_rejects_blocked_fraction(self) -> None:
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[:, :50] = 1
        integral = cv2.integral(mask)
        self.assertEqual(MODULE.rectangle_sum(integral, 0, 0, 50, 100), 5000)
        self.assertEqual(MODULE.rectangle_sum(integral, 50, 0, 50, 100), 0)

    def test_stratified_frames_cover_the_requested_window(self) -> None:
        frames = MODULE.stratified_random_frames(100, 1100, 10, np.random.default_rng(3))
        self.assertEqual(len(frames), 10)
        self.assertTrue(all(100 <= frame < 1100 for frame in frames))
        self.assertLess(frames[0], 200)
        self.assertGreaterEqual(frames[-1], 1000)

    def test_quality_scores_prefer_motion(self) -> None:
        feature = np.zeros((8,), dtype=np.float32)
        candidates = [
            MODULE.PatchCandidate(1, "random", 0, 0.0, 0, 0, 10, 10, feature, 0.0, 1.0, 1.0, 1.0, 0.0, "random"),
            MODULE.PatchCandidate(2, "random", 1, 1.0, 0, 0, 10, 10, feature, 1.0, 1.0, 1.0, 1.0, 0.0, "motion"),
        ]
        MODULE.assign_quality_scores(candidates)
        self.assertGreater(candidates[1].quality_score, candidates[0].quality_score)

    def test_localized_motion_points_find_changed_region(self) -> None:
        current = np.zeros((80, 80), dtype=np.uint8)
        future = current.copy()
        future[32:48, 48:64] = 255
        allowed = np.full_like(current, 255)
        points = MODULE.localized_motion_points(current, future, allowed, downsample=4)

        self.assertGreater(len(points), 0)
        self.assertTrue(np.any((points[:, 0] >= 48) & (points[:, 1] >= 32)))


if __name__ == "__main__":
    unittest.main()
