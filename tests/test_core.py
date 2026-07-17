"""Core regression tests for data loading and model inference."""

from __future__ import annotations

import unittest

import pandas as pd

from src.data_loader import get_feature_columns, load_secom_dataset
from src.model_training import load_model_artifact
from src.prediction import (
    high_risk_records,
    predict_faults,
    prepare_prediction_records,
)


class DataAndPredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_secom_dataset()
        cls.feature_columns = get_feature_columns(cls.data)
        cls.artifact = load_model_artifact()

    def test_dataset_shape_and_target_mapping(self):
        self.assertEqual(len(self.data), 1567)
        self.assertEqual(len(self.feature_columns), 590)
        self.assertEqual(int(self.data["target"].sum()), 104)
        self.assertEqual(set(self.data["target"].unique()), {0, 1})

    def test_saved_artifact_can_score_records(self):
        self.assertIsNotNone(self.artifact)
        sample = self.data.loc[:2, self.feature_columns]
        predictions = predict_faults(self.artifact, sample, threshold=0.50)
        self.assertEqual(len(predictions), 3)
        self.assertTrue(predictions["fault_probability"].between(0, 1).all())
        self.assertTrue(set(predictions["predicted_target"]).issubset({0, 1}))

    def test_high_risk_records_are_sorted(self):
        ranked = high_risk_records(
            self.artifact,
            self.data,
            self.feature_columns,
            top_n=10,
            threshold=0.50,
        )
        self.assertEqual(len(ranked), 10)
        self.assertTrue(ranked["fault_probability"].is_monotonic_decreasing)

    def test_input_validation_rejects_missing_columns(self):
        incomplete = pd.DataFrame({self.feature_columns[0]: [1.0]})
        with self.assertRaises(ValueError):
            prepare_prediction_records(incomplete, self.feature_columns)


if __name__ == "__main__":
    unittest.main()
