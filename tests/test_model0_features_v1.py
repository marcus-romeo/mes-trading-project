"""Frozen Model 0 boundary tests; no target data or production output writes."""

import copy
import math
import tempfile
import unittest
from collections import defaultdict, deque
from datetime import date
from pathlib import Path

import pandas as pd

from src.model0_features_v1 import (
    MINUTE_DIR, _prior_summary, _session_times, build_full_history_model0,
    compute_session_features, load_contract,
)


CONTRACT = load_contract()


def history():
    return defaultdict(lambda: deque(maxlen=20))


def source(day: str) -> pd.DataFrame:
    return pd.read_parquet(Path(MINUTE_DIR) / f"MES_1m_{day}.parquet")


def at(frame: pd.DataFrame, timestamp: str) -> pd.Series:
    match = frame.loc[frame["decision_timestamp"].eq(pd.Timestamp(timestamp))]
    if len(match) != 1:
        raise AssertionError(f"Expected exactly one decision at {timestamp}")
    return match.iloc[0]


def synthetic(start: str, periods: int, day: str, prices=None, instruments=None) -> pd.DataFrame:
    times = pd.date_range(start, periods=periods, freq="min", tz="UTC")
    prices = list(prices) if prices is not None else [100.0 + i * 0.25 for i in range(periods)]
    instruments = list(instruments) if instruments is not None else [1] * periods
    return pd.DataFrame({
        "decision_timestamp": times,
        "minute_start": times - pd.Timedelta(minutes=1),
        "session_date": [date.fromisoformat(day)] * periods,
        "instrument_id": instruments,
        "contract": ["SYN1" if instrument == 1 else "SYN2" for instrument in instruments],
        "high": prices,
        "low": prices,
        "close": prices,
        "price_volume_sum": [price * 10 for price in prices],
        "total_volume": [10] * periods,
        "trade_count": [2] * periods,
        "size_squared_sum": [50] * periods,
        "max_trade_size": [5] * periods,
        "active_second_count": [1] * periods,
        "last_trade_timestamp": times - pd.Timedelta(nanoseconds=1),
    })


class Model0DeterministicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ordinary, _ = compute_session_features(source("2026-02-03"), CONTRACT, history(), None)

    def test_ordinary_liquid_session_and_causal_source(self):
        row = at(self.ordinary, "2026-02-03 15:01:00+00:00")
        self.assertTrue(pd.notna(row["return_60m"]))
        self.assertTrue(pd.notna(row["range_15m"]))
        self.assertTrue(pd.notna(row["us_opening_range_position"]))
        minute = source("2026-02-03")
        self.assertTrue((minute["last_trade_timestamp"] < minute["decision_timestamp"]).all())

    def test_early_session_insufficient_history(self):
        row = at(self.ordinary, "2026-02-02 23:10:00+00:00")
        self.assertTrue(pd.notna(row["return_5m"]))
        for name in ("return_15m", "return_30m", "return_60m", "realized_volatility_60m"):
            self.assertTrue(pd.isna(row[name]), name)

    def test_first_exact_60_minute_path(self):
        self.assertTrue(pd.isna(at(self.ordinary, "2026-02-03 00:00:00+00:00")["return_60m"]))
        self.assertTrue(pd.notna(at(self.ordinary, "2026-02-03 00:01:00+00:00")["return_60m"]))

    def test_each_roll_and_first_new_contract_observation(self):
        for day in ("2025-12-17", "2026-03-18", "2026-06-17"):
            with self.subTest(day=day):
                output, _ = compute_session_features(source(day), CONTRACT, history(), None)
                first = at(output, f"{day} 00:01:00+00:00")
                later = at(output, f"{day} 01:01:00+00:00")
                self.assertTrue(pd.isna(first["return_1m"]))
                self.assertTrue(pd.isna(first["range_5m"]))
                self.assertTrue(pd.isna(first["activity_acceleration_5m"]))
                self.assertTrue(pd.notna(first["session_range_position"]))
                self.assertTrue(pd.notna(later["return_60m"]))
                self.assertTrue(pd.notna(at(output, f"{day} 00:06:00+00:00")["activity_acceleration_5m"]))

    def test_november_outage_masks_paths_and_session_state(self):
        output, _ = compute_session_features(source("2025-11-28"), CONTRACT, history(), None)
        before = at(output, "2025-11-28 02:45:00+00:00")
        after = at(output, "2025-11-28 13:31:00+00:00")
        self.assertTrue(pd.notna(before["session_range_position"]))
        for name in ("return_1m", "range_5m", "session_range_position",
                     "session_vwap_distance", "session_high_distance", "session_low_distance"):
            self.assertTrue(pd.isna(after[name]), name)

    def test_december_24_two_gaps(self):
        output, _ = compute_session_features(source("2025-12-24"), CONTRACT, history(), None)
        for timestamp in ("2025-12-24 07:53:00+00:00", "2025-12-24 07:57:00+00:00"):
            row = at(output, timestamp)
            self.assertTrue(pd.isna(row["return_1m"]))
            self.assertTrue(pd.isna(row["range_5m"]))
            self.assertTrue(pd.isna(row["session_range_position"]))

    def test_december_30_gap(self):
        output, _ = compute_session_features(source("2025-12-30"), CONTRACT, history(), None)
        row = at(output, "2025-12-30 06:47:00+00:00")
        self.assertTrue(pd.isna(row["return_1m"]))
        self.assertTrue(pd.isna(row["range_5m"]))
        self.assertTrue(pd.isna(row["session_range_position"]))

    def test_scheduled_early_close(self):
        output, _ = compute_session_features(source("2025-11-27"), CONTRACT, history(), None)
        row = at(output, "2025-11-27 18:00:00+00:00")
        self.assertTrue(pd.notna(row["session_range_position"]))
        self.assertTrue(pd.notna(row["return_60m"]))
        self.assertEqual(len(output), 1140)

    def test_opening_range_release(self):
        before = at(self.ordinary, "2026-02-03 14:59:00+00:00")
        at_open = at(self.ordinary, "2026-02-03 15:00:00+00:00")
        after = at(self.ordinary, "2026-02-03 15:01:00+00:00")
        self.assertTrue(pd.isna(before["us_opening_range_position"]))
        self.assertTrue(pd.notna(at_open["us_opening_range_position"]))
        self.assertTrue(pd.notna(after["us_opening_range_position"]))

    def test_invalid_immediately_prior_session(self):
        for prior_day, current_day in (
            ("2025-11-28", "2025-12-01"),
            ("2025-12-17", "2025-12-18"),
        ):
            with self.subTest(current_day=current_day):
                prior = _prior_summary(source(prior_day), CONTRACT)
                self.assertFalse(prior["comparable"])
                output, _ = compute_session_features(source(current_day), CONTRACT, history(), prior)
                row = at(output, f"{current_day} 15:00:00+00:00")
                for name in ("prior_session_high_distance", "prior_session_low_distance",
                             "prior_session_close_distance"):
                    self.assertTrue(pd.isna(row[name]))

    def test_19_versus_20_previous_same_q_observations(self):
        frame = source("2025-11-04")
        h19 = history()
        h19[960].extend([(1.0, 2.0)] * 19)
        output19, _ = compute_session_features(frame, CONTRACT, h19, None)
        row19 = at(output19, "2025-11-04 15:00:00+00:00")
        self.assertTrue(pd.isna(row19["relative_volume_q"]))
        self.assertTrue(pd.isna(row19["relative_trade_count_q"]))
        h20 = history()
        h20[960].extend([(1.0, 2.0)] * 20)
        output20, _ = compute_session_features(frame, CONTRACT, h20, None)
        row20 = at(output20, "2025-11-04 15:00:00+00:00")
        self.assertTrue(pd.notna(row20["relative_volume_q"]))
        self.assertTrue(pd.notna(row20["relative_trade_count_q"]))

    def test_first_source_partial_session_and_full_window_coverage(self):
        output, _ = compute_session_features(source("2025-10-07"), CONTRACT, history(), None)
        first = at(output, "2025-10-07 00:01:00+00:00")
        self.assertTrue(pd.isna(first["range_5m"]))
        self.assertTrue(pd.isna(first["activity_acceleration_5m"]))
        self.assertTrue(output["session_range_position"].isna().all())

    def test_sparse_covered_window_is_distinct_from_unavailable_minute(self):
        frame = synthetic("2026-02-02 23:01:00+00:00", 25, "2026-02-03")
        missing_minute = pd.Timestamp("2026-02-02 23:20:00+00:00")
        frame = frame.loc[frame["minute_start"].ne(missing_minute)].reset_index(drop=True)
        observed, _ = compute_session_features(frame, CONTRACT, history(), None)
        row = at(observed, "2026-02-02 23:23:00+00:00")
        self.assertTrue(pd.notna(row["range_5m"]))
        self.assertTrue(pd.isna(row["return_5m"]))
        unavailable_contract = copy.deepcopy(CONTRACT)
        unavailable_contract["unavailable_intervals_utc"].append({
            "session_date": "2026-02-03",
            "start_inclusive": "2026-02-02T23:20:00Z",
            "end_exclusive": "2026-02-02T23:21:00Z",
            "classification": "synthetic_unavailable",
        })
        unavailable, _ = compute_session_features(frame, unavailable_contract, history(), None)
        self.assertTrue(pd.isna(at(unavailable, "2026-02-02 23:23:00+00:00")["range_5m"]))

    def test_opening_range_roll_and_unclipped_breakout(self):
        frame = synthetic("2026-02-03 14:31:00+00:00", 32, "2026-02-03",
                          prices=[100.0] + [101.0] * 29 + [101.0, 102.0])
        output, _ = compute_session_features(frame, CONTRACT, history(), None)
        self.assertEqual(float(at(output, "2026-02-03 15:02:00+00:00")["us_opening_range_position"]), 2.0)
        rolled = frame.copy()
        rolled.loc[rolled["decision_timestamp"] >= pd.Timestamp("2026-02-03 15:01:00+00:00"),
                   "instrument_id"] = 2
        after_roll, _ = compute_session_features(rolled, CONTRACT, history(), None)
        self.assertTrue(pd.isna(at(after_roll, "2026-02-03 15:02:00+00:00")["us_opening_range_position"]))

    def test_zero_and_near_zero_denominators(self):
        frame = synthetic("2026-02-02 23:01:00+00:00", 70, "2026-02-03",
                          prices=[100.0] * 70)
        output, _ = compute_session_features(frame, CONTRACT, history(), None)
        row = output.iloc[-1]
        self.assertEqual(float(row["trend_efficiency_5m"]), 0.0)
        self.assertEqual(float(row["session_range_position"]), 0.5)
        self.assertEqual(float(row["realized_volatility_60m"]), 0.0)
        self.assertTrue(pd.isna(row["volatility_ratio_15m_60m"]))
        self.assertTrue(pd.isna(row["session_vwap_distance"]))
        tiny = frame.copy()
        tiny["close"] = [100.0 + (1e-12 if i % 2 else 0.0) for i in range(70)]
        tiny["high"] = tiny["close"]
        tiny["low"] = tiny["close"]
        tiny["price_volume_sum"] = tiny["close"] * tiny["total_volume"]
        output_tiny, _ = compute_session_features(tiny, CONTRACT, history(), None)
        self.assertTrue(pd.isna(output_tiny.iloc[-1]["session_vwap_distance"]))

    def test_dst_aware_session_open(self):
        open_ns, _, _, _ = _session_times(date(2026, 3, 9))
        self.assertEqual(pd.Timestamp(open_ns, unit="ns", tz="UTC"),
                         pd.Timestamp("2026-03-08 22:00:00+00:00"))

    def test_writer_refuses_populated_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "model0_v1"
            destination.mkdir()
            (destination / "sentinel.txt").write_text("preserve")
            with self.assertRaises(FileExistsError):
                build_full_history_model0(output_dir=destination)
            self.assertEqual((destination / "sentinel.txt").read_text(), "preserve")


if __name__ == "__main__":
    unittest.main(verbosity=2)
