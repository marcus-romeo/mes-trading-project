"""Independent source calculations checked against in-memory Model 0 output."""

from __future__ import annotations

import math
import unittest
from collections import defaultdict, deque

import pandas as pd

from src.model0_audit_v1 import (
    MINUTE_DIR,
    SELECTED,
    _boundary,
    _expected_selected,
    _gaps,
    _overlap,
)
from src.model0_features_v1 import compute_session_features, load_contract


class IndependentPreflightTest(unittest.TestCase):
    def test_selected_historical_values_before_writing(self) -> None:
        contract = load_contract()
        producer_history = defaultdict(lambda: deque(maxlen=20))
        independent_history: dict[int, list[tuple[float, float]]] = defaultdict(list)
        prior_producer = None
        prior_independent = None
        checked = 0
        for path in sorted(MINUTE_DIR.glob("MES_1m_*.parquet")):
            source = pd.read_parquet(path)
            day = source["session_date"].iloc[0]
            day_string = day.isoformat()
            session_open = _boundary(day, 17, 0, previous_day=True)
            session_close = _boundary(day, 16, 0)
            gaps = _gaps(contract, day)
            coverage_start = pd.Timestamp(contract["source_coverage_utc"]["start_inclusive"])
            coverage_end = pd.Timestamp(contract["source_coverage_utc"]["end_exclusive"])
            if day_string in SELECTED:
                actual, _ = compute_session_features(
                    source, contract, producer_history, prior_producer
                )
                for timestamp in SELECTED[day_string]:
                    boundary = pd.Timestamp(timestamp)
                    expected = _expected_selected(
                        source, prior_independent, independent_history,
                        contract, boundary,
                    )
                    row = actual.loc[actual["decision_timestamp"].eq(boundary)].iloc[0]
                    for name, value in expected.items():
                        with self.subTest(timestamp=timestamp, feature=name):
                            observed = row[name]
                            if value is None:
                                self.assertTrue(pd.isna(observed))
                            else:
                                self.assertTrue(pd.notna(observed))
                                self.assertTrue(math.isclose(
                                    float(observed), float(value),
                                    rel_tol=1e-10, abs_tol=1e-10,
                                ))
                            checked += 1
            # The independent baseline deliberately has its own update path.
            for row in source.itertuples(index=False):
                start, end = row.minute_start, row.decision_timestamp
                if (
                    coverage_start <= start and end <= coverage_end
                    and not _overlap(start, end, gaps)
                ):
                    q = int((end - session_open).total_seconds() / 60)
                    activity = (math.log1p(int(row.total_volume)),
                                math.log1p(int(row.trade_count)))
                    independent_history[q].append(activity)
                    independent_history[q] = independent_history[q][-20:]
                    if day_string not in SELECTED:
                        producer_history[q].append(activity)
            prior_producer = {
                "session_date": day,
                "last_decision": source["decision_timestamp"].iloc[-1],
                "comparable": (
                    coverage_start <= session_open
                    and session_close <= coverage_end
                    and not gaps
                    and source["instrument_id"].nunique() == 1
                ),
                "instrument_id": (
                    int(source["instrument_id"].iloc[0])
                    if source["instrument_id"].nunique() == 1 else None
                ),
                "high": float(source["high"].max()),
                "low": float(source["low"].min()),
                "close": float(source["close"].iloc[-1]),
            }
            prior_independent = source
            if day_string == "2026-06-17":
                break
        self.assertEqual(checked, sum(map(len, SELECTED.values())) * 36)


if __name__ == "__main__":
    unittest.main()
