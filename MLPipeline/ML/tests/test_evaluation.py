from __future__ import annotations

import unittest

import numpy as np

from roi_ant_tracker.evaluation import (
    PredictionRecord,
    average_precision,
    best_operating_point,
    match_image,
    rank_predictions,
    rank_predictions_by_center,
)


class EvaluationTests(unittest.TestCase):
    def test_matching_counts_duplicate_prediction_as_false_positive(self) -> None:
        predictions = [
            PredictionRecord(1, 0.9, (0, 0, 10, 10)),
            PredictionRecord(1, 0.8, (0, 0, 10, 10)),
        ]
        counts = match_image(predictions, [(0, 0, 10, 10)], 0.5, 0.5)

        self.assertEqual(counts["true_positives"], 1)
        self.assertEqual(counts["false_positives"], 1)
        self.assertEqual(counts["false_negatives"], 0)

    def test_ap_and_best_threshold_reward_the_clean_prefix(self) -> None:
        predictions = [
            PredictionRecord(1, 0.9, (0, 0, 10, 10)),
            PredictionRecord(1, 0.8, (20, 20, 30, 30)),
        ]
        scores, flags = rank_predictions(predictions, {1: [(0, 0, 10, 10)]}, 0.5)
        best = best_operating_point(scores, flags, 1)

        self.assertTrue(np.isclose(average_precision(scores, flags, 1), 1.0))
        self.assertTrue(np.isclose(best["score_threshold"], 0.9))
        self.assertTrue(np.isclose(best["f1"], 1.0))

    def test_center_matching_accepts_localization_with_low_iou(self) -> None:
        predictions = [PredictionRecord(1, 0.9, (4, 4, 14, 14))]
        scores, flags = rank_predictions_by_center(predictions, {1: [(0, 0, 10, 10)]}, 6.0)

        self.assertEqual(scores.tolist(), [0.9])
        self.assertEqual(flags.tolist(), [1])


if __name__ == "__main__":
    unittest.main()
