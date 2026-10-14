"""Deterministic split and baseline tests before development evaluation."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.model0_chronological_v1 import fold_indices, momentum_probabilities


class ChronologicalDevelopmentTests(unittest.TestCase):
    def test_exact_future_reference_purge(self) -> None:
        validation_days = pd.date_range("2026-03-27", periods=20, freq="D")
        earliest = pd.Timestamp("2026-03-26 22:01:00Z")
        sample = pd.DataFrame({
            "session_date": [
                pd.Timestamp("2026-03-25").date(),
                pd.Timestamp("2026-03-26").date(),
                pd.Timestamp("2026-03-26").date(),
                *[day.date() for day in validation_days],
            ],
            "decision_timestamp": [
                earliest - pd.Timedelta(days=2),
                earliest - pd.Timedelta(minutes=20),
                earliest - pd.Timedelta(minutes=15),
                *[earliest + pd.Timedelta(days=index) for index in range(20)],
            ],
            "future_reference_timestamp": [
                earliest - pd.Timedelta(days=2) + pd.Timedelta(minutes=15),
                earliest - pd.Timedelta(minutes=4, seconds=59),
                earliest,
                *[earliest + pd.Timedelta(days=index, minutes=15)
                  for index in range(20)],
            ],
        })
        train, validation, purged, audit = fold_indices(
            sample, "2026-03-27", "2026-04-15"
        )
        self.assertEqual(train.tolist(), [0, 1])
        self.assertEqual(purged.tolist(), [2])
        self.assertEqual(len(validation), 20)
        self.assertEqual(audit["purged_rows"], 1)
        self.assertLess(
            sample.loc[train, "future_reference_timestamp"].max(),
            sample.loc[validation, "decision_timestamp"].min(),
        )

    def test_frozen_momentum_states_and_laplace_smoothing(self) -> None:
        train_return = np.array([0.01, 0.02, -0.01, -0.02, 0.0, np.nan])
        train_label = np.array([1, 0, 0, 0, 1, 1], dtype=np.int8)
        validation_return = np.array([0.01, -0.01, 0.0, np.nan])
        p, base, states = momentum_probabilities(
            train_return, train_label, validation_return
        )
        self.assertAlmostEqual(base, 4 / 8)
        np.testing.assert_allclose(p, [2 / 4, 1 / 4, 4 / 8, 4 / 8])
        self.assertEqual(states["positive"]["training_rows"], 2)
        self.assertEqual(states["negative"]["training_up"], 0)


if __name__ == "__main__":
    unittest.main()
