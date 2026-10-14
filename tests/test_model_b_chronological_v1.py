"""Deterministic guards for the committed, single-configuration Model B run."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.model0_chronological_v1 import FOLDS, _metric, fold_indices
from src.model_b_chronological_v1 import (
    EXPECTED_TRAIN, EXPECTED_VALIDATION, MODEL_PARAMS, OUTPUT_DIR,
    PREDICTION_COLUMNS, assert_development_only, make_pipeline,
    probability_distribution,
)


class ModelBChronologicalTests(unittest.TestCase):
    def test_exact_frozen_constructor_and_pipeline(self) -> None:
        pipeline = make_pipeline()  # Construct only; do not fit a model.
        self.assertEqual(list(pipeline.named_steps), ["impute", "scale", "model"])
        imputer = pipeline.named_steps["impute"]
        self.assertIsInstance(imputer, SimpleImputer)
        self.assertEqual(imputer.strategy, "median")
        self.assertFalse(imputer.add_indicator)
        self.assertTrue(imputer.keep_empty_features)
        self.assertIsInstance(pipeline.named_steps["scale"], StandardScaler)
        params = pipeline.named_steps["model"].get_params(deep=False)
        self.assertEqual({key: params[key] for key in MODEL_PARAMS}, MODEL_PARAMS)
        self.assertIsNone(params["monotonic_cst"])
        self.assertIsNone(params["interaction_cst"])
        self.assertFalse(params["warm_start"])
        self.assertFalse(params["early_stopping"])
        self.assertIsNone(params["class_weight"])

    def test_folds_and_output_firewall(self) -> None:
        self.assertEqual(FOLDS, (
            ("2026-03-27", "2026-04-23"),
            ("2026-04-24", "2026-05-21"),
            ("2026-05-22", "2026-06-18"),
        ))
        self.assertEqual(EXPECTED_TRAIN, (153824, 179491, 205742))
        self.assertEqual(EXPECTED_VALIDATION, (25667, 26251, 25842))
        self.assertEqual(OUTPUT_DIR.name, "model_b_chronological_v1")
        self.assertNotIn("holdout", str(OUTPUT_DIR).lower())
        self.assertEqual(PREDICTION_COLUMNS[-1], "p_model_b")
        self.assertEqual(len(PREDICTION_COLUMNS), 14)

    def test_development_only_guard(self) -> None:
        base = pd.DataFrame({
            "session_date": ["2026-06-18"],
            "decision_timestamp": [pd.Timestamp("2026-06-17 22:01Z")],
            "target_up_15m": [1],
        })
        assert_development_only(base)
        for date in ("2026-06-19", "2026-09-11"):
            with self.assertRaises(ValueError):
                assert_development_only(base.assign(session_date=date))
        with self.assertRaises(ValueError):
            assert_development_only(pd.concat([base, base], ignore_index=True))
        with self.assertRaises(ValueError):
            assert_development_only(base.assign(target_up_15m=2))

    def test_exact_future_reference_purge(self) -> None:
        days = pd.date_range("2026-03-27", periods=20)
        boundary = pd.Timestamp("2026-03-26 22:01Z")
        sample = pd.DataFrame({
            "session_date": ["2026-03-26", "2026-03-26", *days.strftime("%Y-%m-%d")],
            "decision_timestamp": [boundary - pd.Timedelta(minutes=20),
                                   boundary - pd.Timedelta(minutes=15),
                                   *[boundary + pd.Timedelta(days=n) for n in range(20)]],
            "future_reference_timestamp": [boundary - pd.Timedelta(seconds=1),
                                           boundary,
                                           *[boundary + pd.Timedelta(days=n, minutes=15)
                                             for n in range(20)]],
        })
        train, val, purged, _ = fold_indices(sample, "2026-03-27", "2026-04-15")
        self.assertEqual(train.tolist(), [0])
        self.assertEqual(purged.tolist(), [1])
        self.assertEqual(len(val), 20)

    def test_training_only_preprocessor_semantics(self) -> None:
        # No classifier fit: this isolates imputer/scaler fitting semantics.
        train = np.array([[1., np.nan], [3., 4.], [5., 6.]])
        validation = np.array([[1000., 1000.]])
        pipeline = make_pipeline()
        imputer = pipeline.named_steps["impute"]
        scaler = pipeline.named_steps["scale"]
        transformed_train = imputer.fit_transform(train)
        scaler.fit(transformed_train)
        _ = scaler.transform(imputer.transform(validation))
        np.testing.assert_allclose(imputer.statistics_, [3., 5.])
        np.testing.assert_allclose(scaler.mean_, [3., 5.])

    def test_probability_distribution_and_metric(self) -> None:
        p = np.array([.1, .2, .4, .8])
        summary = probability_distribution(p)
        self.assertEqual(summary["minimum"], .1)
        self.assertEqual(summary["maximum"], .8)
        self.assertAlmostEqual(summary["median"], .3)
        self.assertAlmostEqual(summary["p01"], np.quantile(p, .01, method="linear"))
        y = np.array([0, 0, 1, 1], dtype=np.int8)
        metric = _metric(y, p)
        self.assertAlmostEqual(metric["directional_accuracy_at_0_5"], .75)
        self.assertGreaterEqual(metric["brier_score"], 0)
        self.assertGreaterEqual(metric["roc_auc"], 0)


if __name__ == "__main__":
    unittest.main()
