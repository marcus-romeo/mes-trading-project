"""Build the frozen 36-predictor Model 0 layer from approved completed minutes.

This module never reads target data. Its JSON contract fixes the output schema,
source interval, and unavailable-gap mask before feature values are calculated.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.feature_engineering_v1 import ONE_MINUTE_ARROW_SCHEMA, ONE_MINUTE_SCHEMA_VERSION


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "MODEL0_FEATURE_CONTRACT_V1.json"
MINUTE_DIR = ROOT / "data/processed/1m/full_history_v1"
OUTPUT_DIR = ROOT / "data/processed/features/model0_v1"
MINUTE_NS = 60_000_000_000
CHICAGO = "America/Chicago"
FLOAT_FEATURES = [
    "return_1m", "return_5m", "return_15m", "return_30m", "return_60m",
    "range_5m", "range_15m", "trend_efficiency_5m", "trend_efficiency_15m",
    "realized_volatility_15m", "realized_volatility_60m",
    "volatility_ratio_15m_60m", "log_volume_1m", "log_trade_count_1m",
    "log_average_trade_size_1m", "trade_size_dispersion_1m",
    "log_max_trade_size_1m", "active_second_fraction_1m",
    "relative_volume_q", "relative_trade_count_q", "activity_acceleration_5m",
    "session_vwap_distance", "session_range_position", "session_high_distance",
    "session_low_distance", "prior_session_high_distance",
    "prior_session_low_distance", "prior_session_close_distance",
    "us_opening_range_position",
]
TIME_FLOATS = ["time_of_day_sin", "time_of_day_cos"]
FLAGS = ["asia_flag", "europe_flag", "us_flag", "us_rth_flag"]
REQUIRED_SOURCE = {
    "decision_timestamp", "minute_start", "session_date", "instrument_id",
    "contract", "high", "low", "close", "price_volume_sum", "total_volume",
    "trade_count", "size_squared_sum", "max_trade_size",
    "active_second_count", "last_trade_timestamp",
}


def load_contract(path: Path = CONTRACT_PATH) -> dict:
    contract = json.loads(Path(path).read_text())
    columns = contract["columns"]
    if (
        contract["source_one_minute_schema_version"] != ONE_MINUTE_SCHEMA_VERSION
        or len(columns) != 41
        or [item["name"] for item in columns[:5]]
        != ["decision_timestamp", "minute_start", "session_date", "instrument_id", "contract"]
        or [item["name"] for item in columns[5:]]
        != FLOAT_FEATURES + TIME_FLOATS + ["minutes_since_cme_open"] + FLAGS
        or contract["expected_rows"] != 329337
        or len(contract["unavailable_intervals_utc"]) != 4
    ):
        raise ValueError("Model 0 JSON contract differs from the frozen V1 agreement.")
    return contract


def arrow_schema(contract: dict) -> pa.Schema:
    types = {
        "timestamp[ns, tz=UTC]": pa.timestamp("ns", tz="UTC"),
        "date32": pa.date32(),
        "uint32": pa.uint32(),
        "large_string": pa.large_string(),
        "float64": pa.float64(),
        "uint16": pa.uint16(),
        "uint8": pa.uint8(),
    }
    return pa.schema([
        pa.field(item["name"], types[item["arrow_type"]], nullable=item["nullable"])
        for item in contract["columns"]
    ])


def _ns(value: str | pd.Timestamp) -> int:
    return int(pd.Timestamp(value).value)


def _session_times(session_date: date) -> tuple[int, int, int, int]:
    day = pd.Timestamp(session_date)
    # Localize each wall-clock boundary independently; elapsed UTC addition
    # would shift the nominal 17:00 open across a daylight-saving transition.
    open_ct = (day - pd.Timedelta(days=1) + pd.Timedelta(hours=17)).tz_localize(CHICAGO)
    close_ct = (day + pd.Timedelta(hours=16)).tz_localize(CHICAGO)
    or_start_ct = (day + pd.Timedelta(hours=8, minutes=30)).tz_localize(CHICAGO)
    or_end_ct = (day + pd.Timedelta(hours=9)).tz_localize(CHICAGO)
    return tuple(int(t.tz_convert("UTC").value) for t in (
        open_ct, close_ct, or_start_ct, or_end_ct
    ))


def _gaps(contract: dict, session_date: date) -> list[tuple[int, int]]:
    return [
        (_ns(item["start_inclusive"]), _ns(item["end_exclusive"]))
        for item in contract["unavailable_intervals_utc"]
        if item["session_date"] == session_date.isoformat()
    ]


def _overlaps_gap(start: int, end: int, gaps: list[tuple[int, int]]) -> bool:
    return any(start < gap_end and gap_start < end for gap_start, gap_end in gaps)


def _interval_valid(
    start: int,
    end: int,
    session_open: int,
    session_close: int,
    source_start: int,
    source_end: int,
    gaps: list[tuple[int, int]],
    rolls: list[int],
) -> bool:
    return (
        source_start <= start < end <= source_end
        and session_open <= start < end <= session_close
        and not _overlaps_gap(start, end, gaps)
        and not any(start < roll < end for roll in rolls)
    )


def _same_contract_window(ids: np.ndarray, lo: int, hi: int, candidate: int) -> bool:
    return bool(np.all(ids[lo:hi] == candidate))


def _level_distance(close: float, level: float, scale: float) -> float:
    return math.log(close / level) / scale


def _prepare_minute_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_SOURCE - set(frame.columns)
    if missing:
        raise ValueError(f"Minute source lacks fields: {sorted(missing)}")
    frame = frame.reset_index(drop=True).copy()
    for column in ("decision_timestamp", "minute_start", "last_trade_timestamp"):
        frame[column] = pd.to_datetime(frame[column], utc=True).astype("datetime64[ns, UTC]")
    if frame.empty:
        raise ValueError("Minute session is empty.")
    if frame["decision_timestamp"].isna().any() or frame["decision_timestamp"].duplicated().any():
        raise ValueError("Minute decision timestamps must be present and unique.")
    if not frame["decision_timestamp"].is_monotonic_increasing:
        raise ValueError("Minute decisions are not ordered.")
    if not (frame["minute_start"] + pd.Timedelta(minutes=1) == frame["decision_timestamp"]).all():
        raise ValueError("Minute start/decision boundary differs from [T-1m,T).")
    if not (frame["last_trade_timestamp"] < frame["decision_timestamp"]).all():
        raise ValueError("A frozen minute has an event at or after its decision boundary.")
    if frame["session_date"].nunique() != 1:
        raise ValueError("Expected exactly one CME session.")
    if not (
        frame["close"].gt(0).all()
        and frame["high"].gt(0).all()
        and frame["low"].gt(0).all()
        and frame["total_volume"].gt(0).all()
        and frame["trade_count"].gt(0).all()
    ):
        raise ValueError("An observed minute has invalid price, volume, or trade count.")
    return frame


def _prior_summary(frame: pd.DataFrame, contract: dict) -> dict:
    day = frame["session_date"].iloc[0]
    session_open, session_close, _, _ = _session_times(day)
    start = _ns(contract["source_coverage_utc"]["start_inclusive"])
    end = _ns(contract["source_coverage_utc"]["end_exclusive"])
    gaps = _gaps(contract, day)
    ids = frame["instrument_id"].unique()
    clean = (
        start <= session_open
        and session_close <= end
        and not gaps
        and len(ids) == 1
    )
    return {
        "session_date": day,
        "last_decision": frame["decision_timestamp"].iloc[-1],
        "comparable": clean,
        "instrument_id": int(ids[0]) if len(ids) == 1 else None,
        "high": float(frame["high"].max()),
        "low": float(frame["low"].min()),
        "close": float(frame["close"].iloc[-1]),
    }


def compute_session_features(
    source: pd.DataFrame,
    contract: dict,
    activity_history: dict[int, deque[tuple[float, float]]],
    prior_session: dict | None,
) -> tuple[pd.DataFrame, dict]:
    """Construct one session without reading targets or mutating earlier history."""
    frame = _prepare_minute_frame(source)
    day = frame["session_date"].iloc[0]
    session_open, session_close, or_start, or_end = _session_times(day)
    source_start = _ns(contract["source_coverage_utc"]["start_inclusive"])
    source_end = _ns(contract["source_coverage_utc"]["end_exclusive"])
    gaps = _gaps(contract, day)
    t = frame["decision_timestamp"].array.asi8
    starts = frame["minute_start"].array.asi8
    ids = frame["instrument_id"].to_numpy(dtype=np.uint32)
    close = frame["close"].to_numpy(dtype=np.float64)
    high = frame["high"].to_numpy(dtype=np.float64)
    low = frame["low"].to_numpy(dtype=np.float64)
    volume = frame["total_volume"].to_numpy(dtype=np.float64)
    trades = frame["trade_count"].to_numpy(dtype=np.float64)
    squared = frame["size_squared_sum"].to_numpy(dtype=np.float64)
    max_size = frame["max_trade_size"].to_numpy(dtype=np.float64)
    active = frame["active_second_count"].to_numpy(dtype=np.float64)
    pv = frame["price_volume_sum"].to_numpy(dtype=np.float64)
    n = len(frame)
    roll_times = [int(starts[i]) for i in range(1, n) if ids[i] != ids[i - 1]]
    if any(t[i] <= t[i - 1] for i in range(1, n)):
        raise ValueError("Minute source decision order is invalid.")
    q = (t - session_open) // MINUTE_NS
    if np.any(q < 0) or np.any(q > 65535) or np.any((t - session_open) % MINUTE_NS):
        raise ValueError("Decision boundary is not an integer minute of its CME session.")
    local = frame["decision_timestamp"].dt.tz_convert(CHICAGO)
    local_minute = local.dt.hour.to_numpy() * 60 + local.dt.minute.to_numpy()
    market = {name: np.full(n, np.nan, dtype=np.float64) for name in FLOAT_FEATURES}
    time_sin = np.sin(2 * math.pi * q / 1380)
    time_cos = np.cos(2 * math.pi * q / 1380)
    regime = {
        "asia_flag": ((local_minute >= 17 * 60) | (local_minute < 2 * 60)).astype(np.uint8),
        "europe_flag": ((local_minute >= 2 * 60) & (local_minute < 8 * 60 + 30)).astype(np.uint8),
        "us_flag": ((local_minute >= 8 * 60 + 30) & (local_minute < 16 * 60)).astype(np.uint8),
        "us_rth_flag": ((local_minute >= 8 * 60 + 30) & (local_minute < 15 * 60)).astype(np.uint8),
    }
    # Historical same-q observations are frozen for this entire session.
    baseline = {
        minute_q: (float(np.median([pair[0] for pair in values])),
                   float(np.median([pair[1] for pair in values])))
        for minute_q, values in activity_history.items() if len(values) == 20
    }
    returns = np.full(n, np.nan, dtype=np.float64)
    run_start = 0
    cumulative_pv = 0.0
    cumulative_volume = 0.0
    session_high = -math.inf
    session_low = math.inf
    range_rows: list[int] = []
    opening_high = -math.inf
    opening_low = math.inf
    opening_ids: set[int] = set()
    session_prefix_unavailable = session_open < source_start
    first_gap_start = gaps[0][0] if gaps else None
    stats = {
        "source_prefix_unavailable_rows": 0,
        "post_gap_session_state_null_rows": 0,
        "contract_change_count": len(roll_times),
        "g60_valid_rows": 0,
        "w5_valid_rows": 0,
        "w15_valid_rows": 0,
        "baseline_20_available_rows": 0,
        "opening_range_valid_rows": 0,
    }

    for i in range(n):
        decision = int(t[i])
        candidate = int(ids[i])
        current_start = int(starts[i])
        if i and (t[i] - t[i - 1] != MINUTE_NS or ids[i] != ids[i - 1]):
            run_start = i
        if i > run_start:
            returns[i] = math.log(close[i] / close[i - 1])
        if i == 0 or ids[i] != ids[i - 1]:
            cumulative_pv = 0.0
            cumulative_volume = 0.0
            session_high = -math.inf
            session_low = math.inf
        cumulative_pv += pv[i]
        cumulative_volume += volume[i]
        session_high = max(session_high, high[i])
        session_low = min(session_low, low[i])
        if or_start <= current_start < or_end:
            range_rows.append(i)
            opening_high = max(opening_high, high[i])
            opening_low = min(opening_low, low[i])
            opening_ids.add(candidate)

        market["log_volume_1m"][i] = math.log1p(volume[i])
        market["log_trade_count_1m"][i] = math.log1p(trades[i])
        mean_size = volume[i] / trades[i]
        market["log_average_trade_size_1m"][i] = math.log(mean_size)
        variance = max(squared[i] / trades[i] - mean_size * mean_size, 0.0)
        market["trade_size_dispersion_1m"][i] = math.sqrt(variance) / mean_size
        market["log_max_trade_size_1m"][i] = math.log1p(max_size[i])
        market["active_second_fraction_1m"][i] = active[i] / 60.0
        if int(q[i]) in baseline:
            bv, bn = baseline[int(q[i])]
            market["relative_volume_q"][i] = math.log1p(volume[i]) - bv
            market["relative_trade_count_q"][i] = math.log1p(trades[i]) - bn
            stats["baseline_20_available_rows"] += 1

        path_returns: dict[int, np.ndarray] = {}
        for horizon in (1, 5, 15, 30, 60):
            if i - run_start < horizon:
                continue
            if t[i] - t[i - horizon] != horizon * MINUTE_NS:
                raise ValueError("An apparent G_h path uses non-elapsed row positions.")
            segment = returns[i - horizon + 1:i + 1]
            if not np.isfinite(segment).all():
                raise ValueError("An exact G_h path has invalid log returns.")
            path_returns[horizon] = segment
            market[f"return_{horizon}m"][i] = math.log(close[i] / close[i - horizon])
        for horizon in (5, 15):
            if horizon in path_returns:
                segment = path_returns[horizon]
                travel = float(np.abs(segment).sum())
                market[f"trend_efficiency_{horizon}m"][i] = (
                    float(segment.sum()) / travel if travel != 0 else 0.0
                )
        sigma15 = sigma60 = scale60 = None
        if 15 in path_returns:
            sigma15 = math.sqrt(float(np.square(path_returns[15]).mean()))
            market["realized_volatility_15m"][i] = sigma15
        if 60 in path_returns:
            sigma60 = math.sqrt(float(np.square(path_returns[60]).mean()))
            scale60 = math.sqrt(float(np.square(path_returns[60]).sum()))
            market["realized_volatility_60m"][i] = sigma60
            stats["g60_valid_rows"] += 1
        if sigma15 is not None and sigma60 is not None and sigma60 > 1e-12:
            market["volatility_ratio_15m_60m"][i] = sigma15 / sigma60

        for horizon in (5, 15):
            start = decision - horizon * MINUTE_NS
            if not _interval_valid(
                start, decision, session_open, session_close,
                source_start, source_end, gaps, roll_times,
            ):
                continue
            lo = int(np.searchsorted(t, start, side="right"))
            if not _same_contract_window(ids, lo, i + 1, candidate):
                continue
            name = f"range_{horizon}m"
            market[name][i] = math.log(float(np.max(high[lo:i + 1])) / float(np.min(low[lo:i + 1])))
            stats[f"w{horizon}_valid_rows"] += 1

        prior_start = decision - 6 * MINUTE_NS
        prior_end = decision - MINUTE_NS
        if _interval_valid(
            prior_start, prior_end, session_open, session_close,
            source_start, source_end, gaps, roll_times,
        ) and not any(prior_start < roll <= current_start for roll in roll_times):
            lo = int(np.searchsorted(t, prior_start, side="right"))
            hi = int(np.searchsorted(t, prior_end, side="right"))
            if _same_contract_window(ids, lo, hi, candidate):
                previous_mean = float(np.sum(volume[lo:hi])) / 5.0
                market["activity_acceleration_5m"][i] = (
                    math.log1p(volume[i]) - math.log1p(previous_mean)
                )

        session_state_valid = not session_prefix_unavailable and (
            first_gap_start is None or decision <= first_gap_start
        )
        if session_prefix_unavailable:
            stats["source_prefix_unavailable_rows"] += 1
        elif not session_state_valid:
            stats["post_gap_session_state_null_rows"] += 1
        if session_state_valid:
            market["session_range_position"][i] = (
                (close[i] - session_low) / (session_high - session_low)
                if session_high != session_low else 0.5
            )
            if scale60 is not None and scale60 > 1e-12:
                market["session_vwap_distance"][i] = _level_distance(
                    close[i], cumulative_pv / cumulative_volume, scale60
                )
                market["session_high_distance"][i] = _level_distance(
                    close[i], session_high, scale60
                )
                market["session_low_distance"][i] = _level_distance(
                    close[i], session_low, scale60
                )
        if (
            prior_session is not None
            and prior_session["comparable"]
            and prior_session["last_decision"] < frame["decision_timestamp"].iloc[i]
            and prior_session["instrument_id"] == candidate
            and scale60 is not None
            and scale60 > 1e-12
        ):
            for level in ("high", "low", "close"):
                market[f"prior_session_{level}_distance"][i] = _level_distance(
                    close[i], prior_session[level], scale60
                )

        if (
            decision >= or_end
            and len(range_rows) == 30
            and len(opening_ids) == 1
            and candidate in opening_ids
            and all(starts[index] == or_start + offset * MINUTE_NS
                    for offset, index in enumerate(range_rows))
            and _interval_valid(
                or_start, or_end, session_open, session_close,
                source_start, source_end, gaps, roll_times,
            )
        ):
            market["us_opening_range_position"][i] = (
                (close[i] - opening_low) / (opening_high - opening_low)
                if opening_high != opening_low else 0.5
            )
            stats["opening_range_valid_rows"] += 1

    output = frame.loc[:, [
        "decision_timestamp", "minute_start", "session_date", "instrument_id", "contract"
    ]].copy()
    for name in FLOAT_FEATURES:
        output[name] = pd.array(market[name], dtype="Float64")
    output["time_of_day_sin"] = np.asarray(time_sin, dtype=np.float64)
    output["time_of_day_cos"] = np.asarray(time_cos, dtype=np.float64)
    output["minutes_since_cme_open"] = np.asarray(q, dtype=np.uint16)
    for name in FLAGS:
        output[name] = regime[name]
    output["contract"] = output["contract"].astype("string")
    output = output[[item["name"] for item in contract["columns"]]]
    if output.isna().iloc[:, 0:5].any().any():
        raise ValueError("Model 0 identity field is missing.")

    # Update the causal activity history only after every current-session row
    # has been calculated; current and future rows never enter its own baseline.
    for i in range(n):
        start = int(starts[i])
        if (
            source_start <= start and int(t[i]) <= source_end
            and not _overlaps_gap(start, int(t[i]), gaps)
        ):
            key = int(q[i])
            activity_history[key].append((math.log1p(volume[i]), math.log1p(trades[i])))
    return output, stats


def build_full_history_model0(
    minute_dir: Path = MINUTE_DIR,
    output_dir: Path = OUTPUT_DIR,
    contract_path: Path = CONTRACT_PATH,
) -> dict:
    """Write one versioned feature file per session; refuse populated outputs."""
    minute_dir = Path(minute_dir)
    output_dir = Path(output_dir)
    contract = load_contract(contract_path)
    schema = arrow_schema(contract)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite populated Model 0 directory: {output_dir}")
    source_manifest = json.loads((minute_dir / "run_manifest.json").read_text())
    if source_manifest["schema_version"] != contract["source_one_minute_schema_version"]:
        raise ValueError("One-minute source schema differs from Model 0 contract.")
    files = sorted(minute_dir.glob("MES_1m_*.parquet"))
    if len(files) != source_manifest["sessions"]:
        raise ValueError("Minute session count differs from its frozen manifest.")
    output_dir.mkdir(parents=True, exist_ok=True)
    history: dict[int, deque[tuple[float, float]]] = defaultdict(lambda: deque(maxlen=20))
    prior = None
    audits = []
    aggregate_nulls = {name: 0 for name in FLOAT_FEATURES + TIME_FLOATS + ["minutes_since_cme_open"] + FLAGS}
    total_rows = 0
    previous_last = None
    for number, path in enumerate(files, 1):
        if pq.ParquetFile(path).schema_arrow.remove_metadata() != ONE_MINUTE_ARROW_SCHEMA:
            raise ValueError(f"Frozen minute Arrow schema differs: {path.name}")
        source = _prepare_minute_frame(pd.read_parquet(path))
        day = source["session_date"].iloc[0]
        if path.stem != f"MES_1m_{day}":
            raise ValueError(f"Minute filename/session mismatch: {path.name}")
        if previous_last is not None and source["decision_timestamp"].iloc[0] <= previous_last:
            raise ValueError("Minute decisions are not globally ordered and unique.")
        previous_last = source["decision_timestamp"].iloc[-1]
        features, validity = compute_session_features(source, contract, history, prior)
        output_path = output_dir / f"MES_model0_v1_{day}.parquet"
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite Model 0 session: {output_path}")
        table = pa.Table.from_pandas(features, schema=schema, preserve_index=False, safe=True)
        pq.write_table(table, output_path, compression="zstd")
        if pq.ParquetFile(output_path).schema_arrow.remove_metadata() != schema:
            raise ValueError(f"Persisted Model 0 schema changed: {output_path.name}")
        nulls = {name: int(features[name].isna().sum()) for name in aggregate_nulls}
        for name, count in nulls.items():
            aggregate_nulls[name] += count
        audits.append({
            "session_date": day.isoformat(),
            "source_file": path.name,
            "output_file": output_path.name,
            "rows": len(features),
            "first_decision_timestamp": str(features["decision_timestamp"].iloc[0]),
            "last_decision_timestamp": str(features["decision_timestamp"].iloc[-1]),
            "instrument_ids": sorted(int(x) for x in source["instrument_id"].unique()),
            "null_counts": nulls,
            "validity": validity,
        })
        total_rows += len(features)
        prior = _prior_summary(source, contract)
        print(f"[{number}/{len(files)}] {day}: {len(features)} Model 0 rows", flush=True)
    if total_rows != contract["expected_rows"] or total_rows != source_manifest["one_minute_rows"]:
        raise ValueError("Model 0 output row total differs from frozen row universe.")
    manifest = {
        "schema_version": contract["output_schema_version"],
        "contract_path": CONTRACT_PATH.relative_to(ROOT).as_posix(),
        "contract_version": contract["contract_version"],
        "source_one_minute_path": minute_dir.relative_to(ROOT).as_posix(),
        "source_one_minute_schema_version": source_manifest["schema_version"],
        "target_data_used": False,
        "predictive_model_trained": False,
        "session_files": len(audits),
        "rows": total_rows,
        "predictor_columns": [item["name"] for item in contract["columns"] if item["role"] == "predictor"],
        "null_counts": aggregate_nulls,
        "source_coverage_utc": contract["source_coverage_utc"],
        "unavailable_intervals_utc": contract["unavailable_intervals_utc"],
        "session_audit": audits,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
