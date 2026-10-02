"""Independent read-only audit of the frozen Model 0 feature contract.

The auditor reads the approved minute source directly. It does not import or
call the production feature builder, and it never reads target labels.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "MODEL0_FEATURE_CONTRACT_V1.json"
MINUTE_DIR = ROOT / "data/processed/1m/full_history_v1"
FEATURE_DIR = ROOT / "data/processed/features/model0_v1"
ONE_MINUTE = pd.Timedelta(minutes=1)
SELECTED = {
    "2025-10-07": ["2025-10-07 00:01:00+00:00"],
    "2025-11-03": ["2025-11-03 15:00:00+00:00"],
    "2025-11-04": ["2025-11-04 15:00:00+00:00"],
    "2025-11-27": ["2025-11-27 18:00:00+00:00"],
    "2025-11-28": ["2025-11-28 02:45:00+00:00", "2025-11-28 13:31:00+00:00"],
    "2025-12-01": ["2025-12-01 15:00:00+00:00"],
    "2025-12-17": ["2025-12-17 00:01:00+00:00", "2025-12-17 01:01:00+00:00"],
    "2025-12-18": ["2025-12-18 15:00:00+00:00"],
    "2025-12-24": ["2025-12-24 07:53:00+00:00", "2025-12-24 07:57:00+00:00"],
    "2025-12-30": ["2025-12-30 06:47:00+00:00"],
    "2026-02-03": [
        "2026-02-02 23:10:00+00:00", "2026-02-03 00:00:00+00:00",
        "2026-02-03 00:01:00+00:00", "2026-02-03 14:59:00+00:00",
        "2026-02-03 15:00:00+00:00", "2026-02-03 15:01:00+00:00",
    ],
    "2026-03-18": ["2026-03-18 00:01:00+00:00", "2026-03-18 01:01:00+00:00"],
    "2026-06-17": ["2026-06-17 00:01:00+00:00", "2026-06-17 01:01:00+00:00"],
}


def _boundary(day: date, hour: int, minute: int, previous_day: bool = False) -> pd.Timestamp:
    wall = pd.Timestamp(day) - pd.Timedelta(days=1 if previous_day else 0)
    return (wall + pd.Timedelta(hours=hour, minutes=minute)).tz_localize(
        "America/Chicago"
    ).tz_convert("UTC")


def _gaps(contract: dict, day: date) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    return [
        (pd.Timestamp(item["start_inclusive"]), pd.Timestamp(item["end_exclusive"]))
        for item in contract["unavailable_intervals_utc"]
        if item["session_date"] == day.isoformat()
    ]


def _overlap(start: pd.Timestamp, end: pd.Timestamp, gaps) -> bool:
    return any(start < stop and begin < end for begin, stop in gaps)


def _rolls(frame: pd.DataFrame) -> list[pd.Timestamp]:
    ids = frame["instrument_id"].to_numpy()
    return [
        frame["minute_start"].iloc[index]
        for index in range(1, len(frame)) if ids[index] != ids[index - 1]
    ]


def _covered(
    start: pd.Timestamp, end: pd.Timestamp, open_time: pd.Timestamp,
    close_time: pd.Timestamp, contract: dict, gaps, rolls,
) -> bool:
    acquired = contract["source_coverage_utc"]
    return (
        pd.Timestamp(acquired["start_inclusive"]) <= start < end <= pd.Timestamp(acquired["end_exclusive"])
        and open_time <= start < end <= close_time
        and not _overlap(start, end, gaps)
        and not any(start < roll < end for roll in rolls)
    )


def _valid_prior(prior: pd.DataFrame | None, contract: dict, current_id: int, t: pd.Timestamp):
    if prior is None or prior["decision_timestamp"].iloc[-1] >= t:
        return None
    day = prior["session_date"].iloc[0]
    acquired = contract["source_coverage_utc"]
    if (
        pd.Timestamp(acquired["start_inclusive"]) > _boundary(day, 17, 0, previous_day=True)
        or pd.Timestamp(acquired["end_exclusive"]) < _boundary(day, 16, 0)
        or _gaps(contract, day)
        or prior["instrument_id"].nunique() != 1
        or int(prior["instrument_id"].iloc[0]) != current_id
    ):
        return None
    levels = (
        float(prior["high"].max()),
        float(prior["low"].min()),
        float(prior["close"].iloc[-1]),
    )
    return levels if all(value > 0 for value in levels) else None


def _expected_selected(
    source: pd.DataFrame,
    prior: pd.DataFrame | None,
    history: dict[int, list[tuple[float, float]]],
    contract: dict,
    boundary: pd.Timestamp,
) -> dict[str, float | int | None]:
    """Directly recompute all 36 fields for one decision from elapsed keys."""
    match = source.loc[source["decision_timestamp"].eq(boundary)]
    if len(match) != 1:
        raise ValueError(f"Selected decision absent from minute source: {boundary}")
    row = match.iloc[0]
    day = row["session_date"]
    candidate = int(row["instrument_id"])
    session_open = _boundary(day, 17, 0, previous_day=True)
    session_close = _boundary(day, 16, 0)
    or_start = _boundary(day, 8, 30)
    or_end = _boundary(day, 9, 0)
    gaps = _gaps(contract, day)
    rolls = _rolls(source)
    keyed = source.set_index("decision_timestamp", drop=False)
    current_close = float(row["close"])
    values: dict[str, float | int | None] = {
        item["name"]: None for item in contract["columns"] if item["role"] == "predictor"
    }

    path_returns = {}
    for horizon in (1, 5, 15, 30, 60):
        keys = [boundary - offset * ONE_MINUTE for offset in range(horizon, -1, -1)]
        if not all(key in keyed.index for key in keys):
            continue
        segment = keyed.loc[keys]
        if segment["session_date"].nunique() != 1 or not segment["instrument_id"].eq(candidate).all():
            continue
        prices = segment["close"].to_numpy(dtype=float)
        returns = np.log(prices[1:] / prices[:-1])
        path_returns[horizon] = returns
        values[f"return_{horizon}m"] = math.log(prices[-1] / prices[0])
    for horizon in (5, 15):
        if horizon in path_returns:
            travelled = float(np.abs(path_returns[horizon]).sum())
            values[f"trend_efficiency_{horizon}m"] = (
                float(path_returns[horizon].sum()) / travelled if travelled else 0.0
            )
    scale60 = None
    if 15 in path_returns:
        values["realized_volatility_15m"] = math.sqrt(float(np.mean(np.square(path_returns[15]))))
    if 60 in path_returns:
        squares = float(np.sum(np.square(path_returns[60])))
        values["realized_volatility_60m"] = math.sqrt(squares / 60)
        scale60 = math.sqrt(squares)
        if values["realized_volatility_60m"] > 1e-12:
            values["volatility_ratio_15m_60m"] = (
                values["realized_volatility_15m"] / values["realized_volatility_60m"]
            )
    for horizon in (5, 15):
        start = boundary - horizon * ONE_MINUTE
        if not _covered(start, boundary, session_open, session_close, contract, gaps, rolls):
            continue
        window = source.loc[
            source["minute_start"].ge(start) & source["minute_start"].lt(boundary)
        ]
        if not window.empty and window["instrument_id"].eq(candidate).all():
            values[f"range_{horizon}m"] = math.log(
                float(window["high"].max()) / float(window["low"].min())
            )

    volume = float(row["total_volume"])
    trades = float(row["trade_count"])
    size_mean = volume / trades
    values["log_volume_1m"] = math.log1p(volume)
    values["log_trade_count_1m"] = math.log1p(trades)
    values["log_average_trade_size_1m"] = math.log(size_mean)
    values["trade_size_dispersion_1m"] = math.sqrt(max(
        float(row["size_squared_sum"]) / trades - size_mean ** 2, 0.0
    )) / size_mean
    values["log_max_trade_size_1m"] = math.log1p(float(row["max_trade_size"]))
    values["active_second_fraction_1m"] = float(row["active_second_count"]) / 60
    q = int((boundary - session_open).total_seconds() / 60)
    older = history.get(q, [])
    if len(older) == 20:
        values["relative_volume_q"] = math.log1p(volume) - float(np.median([x[0] for x in older]))
        values["relative_trade_count_q"] = math.log1p(trades) - float(np.median([x[1] for x in older]))
    previous_start = boundary - 6 * ONE_MINUTE
    previous_end = boundary - ONE_MINUTE
    if (
        _covered(previous_start, previous_end, session_open, session_close, contract, gaps, rolls)
        and not any(previous_start < roll <= previous_end for roll in rolls)
    ):
        previous = source.loc[
            source["minute_start"].ge(previous_start)
            & source["minute_start"].lt(previous_end)
        ]
        if previous["instrument_id"].eq(candidate).all():
            values["activity_acceleration_5m"] = (
                math.log1p(volume) - math.log1p(float(previous["total_volume"].sum()) / 5)
            )

    acquired_start = pd.Timestamp(contract["source_coverage_utc"]["start_inclusive"])
    dirty = acquired_start > session_open or any(begin < boundary for begin, _ in gaps)
    if not dirty:
        current = source.loc[source["decision_timestamp"].le(boundary)]
        prior_rolls = [roll for roll in rolls if roll < boundary]
        if prior_rolls:
            current = current.loc[current["minute_start"].ge(prior_rolls[-1])]
        if not current["instrument_id"].eq(candidate).all():
            raise ValueError("Independent session state contains more than one contract.")
        session_high = float(current["high"].max())
        session_low = float(current["low"].min())
        values["session_range_position"] = (
            (current_close - session_low) / (session_high - session_low)
            if session_high != session_low else 0.5
        )
        if scale60 is not None and scale60 > 1e-12:
            session_vwap = float(current["price_volume_sum"].sum()) / float(current["total_volume"].sum())
            values["session_vwap_distance"] = math.log(current_close / session_vwap) / scale60
            values["session_high_distance"] = math.log(current_close / session_high) / scale60
            values["session_low_distance"] = math.log(current_close / session_low) / scale60
    prior_levels = _valid_prior(prior, contract, candidate, boundary)
    if prior_levels is not None and scale60 is not None and scale60 > 1e-12:
        for level, number in zip(("high", "low", "close"), prior_levels):
            values[f"prior_session_{level}_distance"] = math.log(current_close / number) / scale60

    opening = source.loc[
        source["minute_start"].ge(or_start) & source["minute_start"].lt(or_end)
    ]
    exact_slots = (
        len(opening) == 30
        and opening["minute_start"].tolist() == [or_start + k * ONE_MINUTE for k in range(30)]
    )
    if (
        boundary >= or_end and exact_slots
        and opening["instrument_id"].eq(candidate).all()
        and _covered(or_start, or_end, session_open, session_close, contract, gaps, rolls)
    ):
        opening_high = float(opening["high"].max())
        opening_low = float(opening["low"].min())
        values["us_opening_range_position"] = (
            (current_close - opening_low) / (opening_high - opening_low)
            if opening_high != opening_low else 0.5
        )
    local = boundary.tz_convert("America/Chicago")
    wall_minute = local.hour * 60 + local.minute
    values["time_of_day_sin"] = math.sin(2 * math.pi * q / 1380)
    values["time_of_day_cos"] = math.cos(2 * math.pi * q / 1380)
    values["minutes_since_cme_open"] = q
    values["asia_flag"] = int(wall_minute >= 17 * 60 or wall_minute < 2 * 60)
    values["europe_flag"] = int(2 * 60 <= wall_minute < 8 * 60 + 30)
    values["us_flag"] = int(8 * 60 + 30 <= wall_minute < 16 * 60)
    values["us_rth_flag"] = int(8 * 60 + 30 <= wall_minute < 15 * 60)
    return values


def audit_full_history(
    minute_dir: Path = MINUTE_DIR,
    feature_dir: Path = FEATURE_DIR,
    contract_path: Path = CONTRACT_PATH,
) -> dict:
    """Audit all identities and validity masks, plus independent selected values."""
    minute_dir, feature_dir = Path(minute_dir), Path(feature_dir)
    contract = json.loads(Path(contract_path).read_text())
    manifest = json.loads((feature_dir / "run_manifest.json").read_text())
    types = {
        "timestamp[ns, tz=UTC]": pa.timestamp("ns", tz="UTC"), "date32": pa.date32(),
        "uint32": pa.uint32(), "large_string": pa.large_string(), "float64": pa.float64(),
        "uint16": pa.uint16(), "uint8": pa.uint8(),
    }
    schema = pa.schema([
        pa.field(c["name"], types[c["arrow_type"]], nullable=c["nullable"])
        for c in contract["columns"]
    ])
    names = [c["name"] for c in contract["columns"]]
    predictors = names[5:]
    if (
        manifest["schema_version"] != contract["output_schema_version"]
        or manifest["predictor_columns"] != predictors
        or manifest.get("target_data_used") is not False
        or manifest.get("predictive_model_trained") is not False
    ):
        raise ValueError("Model 0 run manifest conflicts with frozen feature contract.")
    minute_files = {p.stem.removeprefix("MES_1m_"): p for p in minute_dir.glob("MES_1m_*.parquet")}
    feature_files = {p.stem.removeprefix("MES_model0_v1_"): p for p in feature_dir.glob("MES_model0_v1_*.parquet")}
    if not minute_files or minute_files.keys() != feature_files.keys():
        raise ValueError("Model 0 and source session sets differ.")
    session_audits = {item["session_date"]: item for item in manifest["session_audit"]}
    if session_audits.keys() != minute_files.keys():
        raise ValueError("Model 0 manifest session set differs from source.")
    aggregated_nulls = Counter()
    distributions: dict[str, list[np.ndarray]] = defaultdict(list)
    prior = None
    history: dict[int, list[tuple[float, float]]] = defaultdict(list)
    total = 0
    previous_last = None
    selected_values = 0
    mask_checks = Counter()
    for day_string in sorted(minute_files):
        source = pd.read_parquet(minute_files[day_string])
        path = feature_files[day_string]
        if pq.ParquetFile(path).schema_arrow.remove_metadata() != schema:
            raise ValueError(f"{day_string}: Arrow schema differs from frozen contract")
        output = pd.read_parquet(path)
        if list(output.columns) != names or len(output) != len(source):
            raise ValueError(f"{day_string}: Model 0 columns or row count differ")
        identity = names[:5]
        for name in identity:
            if not source[name].equals(output[name]):
                raise ValueError(f"{day_string}: {name} differs from frozen minute source")
        t = source["decision_timestamp"]
        if not t.is_monotonic_increasing or t.duplicated().any():
            raise ValueError(f"{day_string}: source decisions not ordered and unique")
        if previous_last is not None and t.iloc[0] <= previous_last:
            raise ValueError(f"{day_string}: global decisions duplicate or reorder")
        previous_last = t.iloc[-1]
        if not (source["last_trade_timestamp"] < t).all():
            raise ValueError(f"{day_string}: source event at/after decision boundary")
        if not (source["minute_start"] + ONE_MINUTE == t).all():
            raise ValueError(f"{day_string}: source minute timing differs")
        observed_nulls = {name: int(output[name].isna().sum()) for name in predictors}
        if observed_nulls != session_audits[day_string]["null_counts"]:
            raise ValueError(f"{day_string}: per-session null manifest differs")
        aggregated_nulls.update(observed_nulls)
        for name in predictors:
            nonnull = output[name].dropna().to_numpy(dtype=float)
            if not np.isfinite(nonnull).all():
                raise ValueError(f"{day_string}: {name} contains nonfinite non-null values")
            distributions[name].append(nonnull)
        for name in ("asia_flag", "europe_flag", "us_flag", "us_rth_flag"):
            if not output[name].isin([0, 1]).all():
                raise ValueError(f"{day_string}: invalid regime flag {name}")
        if not output["active_second_fraction_1m"].dropna().between(0, 1).all():
            raise ValueError(f"{day_string}: impossible active-second fraction")
        if not output["session_range_position"].dropna().between(-1e-12, 1 + 1e-12).all():
            raise ValueError(f"{day_string}: impossible session-range position")
        for name in ("trend_efficiency_5m", "trend_efficiency_15m"):
            if not output[name].dropna().between(-1 - 1e-12, 1 + 1e-12).all():
                raise ValueError(f"{day_string}: impossible trend efficiency")
        for name in ("range_5m", "range_15m", "realized_volatility_15m", "realized_volatility_60m"):
            if not output[name].dropna().ge(-1e-12).all():
                raise ValueError(f"{day_string}: impossible negative {name}")

        day = source["session_date"].iloc[0]
        session_open = _boundary(day, 17, 0, previous_day=True)
        session_close = _boundary(day, 16, 0)
        gaps = _gaps(contract, day)
        rolls = _rolls(source)
        ids = source["instrument_id"].to_numpy()
        decision_ns = t.array.asi8
        minute_start_ns = source["minute_start"].array.asi8
        nonnull = {name: output[name].notna().to_numpy() for name in predictors}
        prior_ok = {
            int(instrument): _valid_prior(prior, contract, int(instrument), t.iloc[0]) is not None
            for instrument in np.unique(ids)
        }
        or_start = _boundary(day, 8, 30)
        or_end = _boundary(day, 9, 0)
        opening = source.loc[
            source["minute_start"].ge(or_start) & source["minute_start"].lt(or_end)
        ]
        opening_id = int(opening["instrument_id"].iloc[0]) if len(opening) else None
        opening_base_valid = (
            len(opening) == 30
            and opening["minute_start"].tolist() ==
            [or_start + k * ONE_MINUTE for k in range(30)]
            and opening["instrument_id"].eq(opening_id).all()
            and _covered(or_start, or_end, session_open, session_close, contract, gaps, rolls)
        )
        source_prefix_dirty = (
            pd.Timestamp(contract["source_coverage_utc"]["start_inclusive"]) > session_open
        )
        run = 1
        for index, boundary in enumerate(t):
            if index:
                run = run + 1 if (
                    boundary - t.iloc[index - 1] == ONE_MINUTE
                    and ids[index] == ids[index - 1]
                ) else 1
            for horizon in (1, 5, 15, 30, 60):
                should_exist = run >= horizon + 1
                if bool(nonnull[f"return_{horizon}m"][index]) != should_exist:
                    raise ValueError(f"{boundary}: G_{horizon} validity differs")
                mask_checks["exact_price_paths"] += 1
            for horizon in (5, 15):
                start = boundary - horizon * ONE_MINUTE
                covered = _covered(start, boundary, session_open, session_close, contract, gaps, rolls)
                lo = int(np.searchsorted(decision_ns, start.value, side="right"))
                should_exist = covered and lo <= index and bool(np.all(ids[lo:index + 1] == ids[index]))
                if bool(nonnull[f"range_{horizon}m"][index]) != should_exist:
                    raise ValueError(f"{boundary}: W_{horizon} validity differs")
                mask_checks["elapsed_range_windows"] += 1
            preceding_start = boundary - 6 * ONE_MINUTE
            preceding_end = boundary - ONE_MINUTE
            lo = int(np.searchsorted(decision_ns, preceding_start.value, side="right"))
            hi = int(np.searchsorted(decision_ns, preceding_end.value, side="right"))
            acceleration_valid = (
                _covered(preceding_start, preceding_end, session_open, session_close,
                         contract, gaps, rolls)
                and not any(preceding_start < roll <= preceding_end for roll in rolls)
                and bool(np.all(ids[lo:hi] == ids[index]))
            )
            if bool(nonnull["activity_acceleration_5m"][index]) != acceleration_valid:
                raise ValueError(f"{boundary}: activity acceleration validity differs")
            mask_checks["activity_windows"] += 1
            session_dirty = (
                source_prefix_dirty or any(begin < boundary for begin, _ in gaps)
            )
            state_present = nonnull["session_range_position"][index]
            if state_present == session_dirty:
                raise ValueError(f"{boundary}: session cumulative state validity differs")
            mask_checks["session_cumulative_state"] += 1
            q = int((boundary - session_open).total_seconds() / 60)
            baseline_present = len(history[q]) == 20
            for name in ("relative_volume_q", "relative_trade_count_q"):
                if bool(nonnull[name][index]) != baseline_present:
                    raise ValueError(f"{boundary}: causal 20-session baseline membership differs")
            mask_checks["activity_baselines"] += 1
            if not prior_ok[int(ids[index])]:
                for name in ("prior_session_high_distance", "prior_session_low_distance",
                             "prior_session_close_distance"):
                    if nonnull[name][index]:
                        raise ValueError(f"{boundary}: invalid prior session supplied {name}")
            mask_checks["prior_session_admission"] += 1
            opening_valid = (
                boundary >= or_end and opening_base_valid and ids[index] == opening_id
            )
            if bool(nonnull["us_opening_range_position"][index]) != opening_valid:
                raise ValueError(f"{boundary}: opening-range admission differs")
            mask_checks["opening_range_admission"] += 1
        for timestamp in SELECTED.get(day_string, []):
            boundary = pd.Timestamp(timestamp)
            expected = _expected_selected(source, prior, history, contract, boundary)
            row = output.loc[output["decision_timestamp"].eq(boundary)].iloc[0]
            for name, value in expected.items():
                actual = row[name]
                if value is None:
                    if pd.notna(actual):
                        raise ValueError(f"{boundary}: {name} expected null")
                elif pd.isna(actual) or not math.isclose(
                    float(actual), float(value), rel_tol=1e-10, abs_tol=1e-10
                ):
                    raise ValueError(f"{boundary}: {name} differs from independent source calculation")
                selected_values += 1
        for row in source.itertuples(index=False):
            start = row.minute_start
            end = row.decision_timestamp
            if (
                pd.Timestamp(contract["source_coverage_utc"]["start_inclusive"]) <= start
                and end <= pd.Timestamp(contract["source_coverage_utc"]["end_exclusive"])
                and not _overlap(start, end, gaps)
            ):
                minute_q = int((end - session_open).total_seconds() / 60)
                history[minute_q].append((
                    math.log1p(int(row.total_volume)), math.log1p(int(row.trade_count))
                ))
                history[minute_q] = history[minute_q][-20:]
        prior = source
        total += len(output)
    if total != contract["expected_rows"] or total != manifest["rows"]:
        raise ValueError("Full-history Model 0 row total differs from contract/manifest")
    if len(feature_files) != manifest["session_files"] or aggregated_nulls != manifest["null_counts"]:
        raise ValueError("Model 0 file count or aggregate nulls differ from manifest")
    summary = {}
    for name in predictors:
        values = np.concatenate(distributions[name])
        summary[name] = {
            "non_null": int(len(values)),
            "null": int(aggregated_nulls[name]),
            "null_pct": float(100 * aggregated_nulls[name] / total),
            "finite": int(np.isfinite(values).sum()),
            "min": float(np.min(values)) if len(values) else None,
            "median": float(np.median(values)) if len(values) else None,
            "max": float(np.max(values)) if len(values) else None,
        }
    return {
        "passed": True,
        "session_files": len(feature_files),
        "rows": total,
        "predictors": len(predictors),
        "selected_numeric_checks": selected_values,
        "validity_mask_checks": dict(mask_checks),
        "feature_distributions": summary,
    }


if __name__ == "__main__":
    result = audit_full_history()
    print(json.dumps(result, indent=2))
