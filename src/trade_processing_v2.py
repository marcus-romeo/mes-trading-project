"""Event-time MES trade processing helpers.

This module is the authoritative implementation for the second version of the
one-second MES foundation layer.  It deliberately keeps the proven v1 design:
read bounded chunks, retain the final incomplete second, carry market state,
aggregate complete seconds, and buffer one CME session at a time.

The important correction is time policy.  Market bars are keyed by Databento
``ts_event`` (exchange event time), not the DataFrame index (``ts_recv``).
Receive time remains useful for delivery-latency research, but it is not the
market clock used by this historical dataset.
"""

from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from databento import DBNStore


CHICAGO_TIMEZONE = "America/Chicago"
SCHEMA_VERSION = "mes_1s_v2_event_time_33"
TIMESTAMP_POLICY = "Databento ts_event (exchange event time), stored in UTC"

# This mapping was verified in notebook 03 for this immutable MES.v.0 source.
CONTRACT_MAP = {
    42004164: "MESZ5",
    42003800: "MESH6",
    42005163: "MESM6",
    42003239: "MESU6",
}

# These are Databento calendar dates reported as degraded during the original
# raw-data validation.  The manifest retains the source dates as provenance;
# it conservatively marks a session when either of its two CT calendar dates
# appears here.
DEGRADED_CALENDAR_DATES = {
    "2025-11-28": "degraded",
    "2026-01-31": "degraded",
    "2026-03-15": "degraded",
    "2026-03-16": "degraded",
    "2026-03-21": "degraded",
    "2026-04-10": "degraded",
    "2026-05-24": "degraded",
    "2026-07-30": "degraded",
    "2026-08-29": "degraded",
}

ONE_SECOND_COLUMNS = [
    "timestamp_second",
    "session_date",
    "instrument_id",
    "contract",
    "open",
    "high",
    "low",
    "close",
    "price_volume_sum",
    "price_squared_volume_sum",
    "total_volume",
    "trade_count",
    "max_trade_size",
    "size_squared_sum",
    "uptick_count",
    "downtick_count",
    "same_price_count",
    "unique_price_levels",
    "inferred_buy_volume",
    "inferred_sell_volume",
    "inferred_unknown_volume",
    "inferred_buy_trade_count",
    "inferred_sell_trade_count",
    "inferred_unknown_trade_count",
    "inferred_delta",
    "native_buy_volume",
    "native_sell_volume",
    "native_unknown_volume",
    "native_delta",
    "first_trade_timestamp",
    "last_trade_timestamp",
    "max_volume_at_price",
    "contract_change",
]

# Counts and non-negative volumes have distinct, stable physical types.  Delta
# is signed because buy minus sell may be negative.  Volume is widened to
# uint64 before squaring or aggregation; this prevents uint32 wraparound.
ONE_SECOND_ARROW_SCHEMA = pa.schema([
    pa.field("timestamp_second", pa.timestamp("ns", tz="UTC")),
    pa.field("session_date", pa.date32()),
    pa.field("instrument_id", pa.uint32()),
    pa.field("contract", pa.string()),
    pa.field("open", pa.float64()),
    pa.field("high", pa.float64()),
    pa.field("low", pa.float64()),
    pa.field("close", pa.float64()),
    pa.field("price_volume_sum", pa.float64()),
    pa.field("price_squared_volume_sum", pa.float64()),
    pa.field("total_volume", pa.uint64()),
    pa.field("trade_count", pa.uint32()),
    pa.field("max_trade_size", pa.uint32()),
    pa.field("size_squared_sum", pa.uint64()),
    pa.field("uptick_count", pa.uint32()),
    pa.field("downtick_count", pa.uint32()),
    pa.field("same_price_count", pa.uint32()),
    pa.field("unique_price_levels", pa.uint32()),
    pa.field("inferred_buy_volume", pa.uint64()),
    pa.field("inferred_sell_volume", pa.uint64()),
    pa.field("inferred_unknown_volume", pa.uint64()),
    pa.field("inferred_buy_trade_count", pa.uint32()),
    pa.field("inferred_sell_trade_count", pa.uint32()),
    pa.field("inferred_unknown_trade_count", pa.uint32()),
    pa.field("inferred_delta", pa.int64()),
    pa.field("native_buy_volume", pa.uint64()),
    pa.field("native_sell_volume", pa.uint64()),
    pa.field("native_unknown_volume", pa.uint64()),
    pa.field("native_delta", pa.int64()),
    pa.field("first_trade_timestamp", pa.timestamp("ns", tz="UTC")),
    pa.field("last_trade_timestamp", pa.timestamp("ns", tz="UTC")),
    pa.field("max_volume_at_price", pa.uint64()),
    pa.field("contract_change", pa.bool_()),
])

UINT64_COLUMNS = [
    "total_volume", "size_squared_sum", "inferred_buy_volume",
    "inferred_sell_volume", "inferred_unknown_volume", "native_buy_volume",
    "native_sell_volume", "native_unknown_volume", "max_volume_at_price",
]
UINT32_COLUMNS = [
    "instrument_id", "trade_count", "max_trade_size", "uptick_count",
    "downtick_count", "same_price_count", "unique_price_levels",
    "inferred_buy_trade_count", "inferred_sell_trade_count",
    "inferred_unknown_trade_count",
]
FLOAT_COLUMNS = [
    "open", "high", "low", "close", "price_volume_sum",
    "price_squared_volume_sum",
]


def event_time_session_date(timestamp: pd.Series) -> pd.Series:
    """Return the CME trade date for UTC exchange event timestamps.

    MES sessions are labelled by their CT end date.  A trade at or after
    17:00 CT therefore belongs to the following calendar date.  Zone-aware
    conversion, rather than a fixed UTC offset, makes this DST-safe.
    """
    chicago = pd.to_datetime(timestamp, utc=True).dt.tz_convert(CHICAGO_TIMEZONE)
    return (chicago + pd.Timedelta(hours=7)).dt.date


def assert_event_time_order(
    trades: pd.DataFrame,
    previous_event_timestamp: pd.Timestamp | None = None,
) -> pd.Timestamp | None:
    """Fail fast unless a raw chunk continues nondecreasing event-time order.

    Incomplete-second carryover and session flushing are correct only for an
    ordered trade stream.  Equal timestamps are allowed; their DBN order is
    retained as the deterministic order for tick-rule and OHLC calculations.
    """
    if trades.empty:
        return previous_event_timestamp
    event_time = pd.to_datetime(trades["ts_event"], utc=True)
    if event_time.isna().any() or not event_time.is_monotonic_increasing:
        raise ValueError("Raw chunk is not ordered by non-null ts_event.")
    first_event = event_time.iloc[0]
    if previous_event_timestamp is not None and first_event < previous_event_timestamp:
        raise ValueError("ts_event moved backward across a raw chunk boundary.")
    return event_time.iloc[-1]


def add_trade_context(trades: pd.DataFrame) -> pd.DataFrame:
    """Add canonical event-time, UTC-second, CME-date, and contract fields."""
    # Databento's ts_recv index is not guaranteed unique.  Processing depends
    # on positional first/last rows, so make that position explicit and avoid
    # accidental label-based updates affecting duplicate receive timestamps.
    trades = trades.copy().reset_index(drop=True)
    # Do not use trades.index: Databento names that index ts_recv.
    trades["timestamp"] = pd.to_datetime(trades["ts_event"], utc=True)
    trades["timestamp_second"] = trades["timestamp"].dt.floor("s")
    trades["session_date"] = event_time_session_date(trades["timestamp"])
    trades["contract"] = trades["instrument_id"].map(CONTRACT_MAP)
    if trades["contract"].isna().any():
        unknown = trades.loc[trades["contract"].isna(), "instrument_id"].unique()
        raise ValueError(f"Unmapped instrument IDs found: {unknown}")
    return trades


def apply_tick_rule_v1(prices, instrument_ids, previous_price=None,
                       previous_direction=0, previous_instrument_id=None):
    """Apply the live-compatible tick rule while preserving chunk state."""
    prices = np.asarray(prices, dtype="float64")
    instrument_ids = np.asarray(instrument_ids)
    directions = np.zeros(len(prices), dtype="int8")
    prev_price, prev_direction, prev_instrument = (
        previous_price, previous_direction, previous_instrument_id
    )
    for index, (price, instrument_id) in enumerate(zip(prices, instrument_ids)):
        if prev_instrument is not None and instrument_id != prev_instrument:
            prev_price, prev_direction = None, 0
        if prev_price is None:
            direction = 0
        elif price > prev_price:
            direction = 1
        elif price < prev_price:
            direction = -1
        else:
            direction = prev_direction
        directions[index] = direction
        prev_price, prev_direction, prev_instrument = price, direction, instrument_id
    return directions, prev_price, prev_direction, prev_instrument


def split_complete_seconds(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold back the final event-time second until the next raw chunk arrives."""
    if trades.empty:
        return trades.copy(), trades.copy()
    final_second = trades["timestamp_second"].iloc[-1]
    pending = trades.loc[trades["timestamp_second"] == final_second].copy()
    return trades.loc[trades["timestamp_second"] != final_second].copy(), pending


def prepare_complete_trades(trades: pd.DataFrame, previous_price=None,
                            previous_direction=0, previous_instrument_id=None):
    """Prepare complete seconds while preserving tick and price-change state."""
    if trades.empty:
        return trades.copy(), previous_price, previous_direction, previous_instrument_id
    trades = trades.copy()
    incoming_price, incoming_instrument = previous_price, previous_instrument_id
    directions, previous_price, previous_direction, previous_instrument_id = apply_tick_rule_v1(
        trades["price"].to_numpy(), trades["instrument_id"].to_numpy(),
        previous_price, previous_direction, previous_instrument_id,
    )
    trades["inferred_direction"] = directions
    trades["timestamp_utc"] = trades["timestamp"]

    sizes = trades["size"].astype("uint64")
    trades["price_volume"] = trades["price"].astype("float64") * sizes
    trades["price_squared_volume"] = (trades["price"].astype("float64") ** 2) * sizes
    trades["size_squared"] = sizes * sizes

    price_change = trades["price"].diff()
    comparable = price_change.notna()
    contract_changed = trades["instrument_id"].ne(trades["instrument_id"].shift())
    first = trades.index[0]
    first_instrument = trades.loc[first, "instrument_id"]
    if incoming_price is not None and incoming_instrument == first_instrument:
        price_change.loc[first] = trades.loc[first, "price"] - incoming_price
        comparable.loc[first] = True
    else:
        # A new series has no valid prior price.  It is intentionally not a
        # same-price observation, and it does not create a false roll return.
        price_change.loc[first] = np.nan
        comparable.loc[first] = False
    contract_changed.loc[first] = False
    price_change.loc[contract_changed] = np.nan
    comparable.loc[contract_changed] = False
    trades["uptick"] = (comparable & (price_change > 0)).astype("uint32")
    trades["downtick"] = (comparable & (price_change < 0)).astype("uint32")
    trades["same_price"] = (comparable & (price_change == 0)).astype("uint32")

    trades["inferred_buy_volume"] = sizes * (directions == 1)
    trades["inferred_sell_volume"] = sizes * (directions == -1)
    trades["inferred_unknown_volume"] = sizes * (directions == 0)
    trades["inferred_buy_trade"] = (directions == 1).astype("uint32")
    trades["inferred_sell_trade"] = (directions == -1).astype("uint32")
    trades["inferred_unknown_trade"] = (directions == 0).astype("uint32")
    trades["inferred_signed_volume"] = sizes.astype("int64") * directions.astype("int64")

    native_direction = trades["side"].astype(str).map({"B": 1, "A": -1}).fillna(0).astype("int8")
    trades["native_buy_volume"] = sizes * (native_direction == 1)
    trades["native_sell_volume"] = sizes * (native_direction == -1)
    trades["native_unknown_volume"] = sizes * (native_direction == 0)
    trades["native_signed_volume"] = sizes.astype("int64") * native_direction.astype("int64")
    return trades, previous_price, previous_direction, previous_instrument_id


def aggregate_to_one_second(trades: pd.DataFrame) -> pd.DataFrame:
    """Aggregate prepared, complete event-time seconds into 32 base columns."""
    if trades.empty:
        return pd.DataFrame()
    keys = ["timestamp_second", "session_date", "instrument_id", "contract"]
    second_data = trades.groupby(keys, as_index=False).agg(
        open=("price", "first"), high=("price", "max"), low=("price", "min"), close=("price", "last"),
        price_volume_sum=("price_volume", "sum"), price_squared_volume_sum=("price_squared_volume", "sum"),
        total_volume=("size", "sum"), trade_count=("size", "size"), max_trade_size=("size", "max"),
        size_squared_sum=("size_squared", "sum"), uptick_count=("uptick", "sum"),
        downtick_count=("downtick", "sum"), same_price_count=("same_price", "sum"),
        unique_price_levels=("price", "nunique"), inferred_buy_volume=("inferred_buy_volume", "sum"),
        inferred_sell_volume=("inferred_sell_volume", "sum"), inferred_unknown_volume=("inferred_unknown_volume", "sum"),
        inferred_buy_trade_count=("inferred_buy_trade", "sum"), inferred_sell_trade_count=("inferred_sell_trade", "sum"),
        inferred_unknown_trade_count=("inferred_unknown_trade", "sum"), inferred_delta=("inferred_signed_volume", "sum"),
        native_buy_volume=("native_buy_volume", "sum"), native_sell_volume=("native_sell_volume", "sum"),
        native_unknown_volume=("native_unknown_volume", "sum"), native_delta=("native_signed_volume", "sum"),
        first_trade_timestamp=("timestamp_utc", "first"), last_trade_timestamp=("timestamp_utc", "last"),
    )
    volume_at_price = trades.groupby(keys + ["price"], as_index=False)["size"].sum()
    max_at_price = volume_at_price.groupby(keys, as_index=False)["size"].max().rename(columns={"size": "max_volume_at_price"})
    return second_data.merge(max_at_price, on=keys, how="left")


def add_contract_change_flag(second_data: pd.DataFrame, previous_instrument_id=None):
    """Flag the first aggregated second after a contract change across batches."""
    flagged = second_data.copy()
    if flagged.empty:
        return flagged, previous_instrument_id
    flagged["contract_change"] = flagged["instrument_id"].ne(flagged["instrument_id"].shift())
    flagged.loc[flagged.index[0], "contract_change"] = (
        False if previous_instrument_id is None else flagged.loc[flagged.index[0], "instrument_id"] != previous_instrument_id
    )
    return flagged, flagged["instrument_id"].iloc[-1]


def enforce_one_second_schema(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the permanent 33-column DataFrame with deliberate stable dtypes."""
    if frame.empty:
        return frame
    if set(frame.columns) != set(ONE_SECOND_COLUMNS):
        raise ValueError("One-second frame does not contain the permanent 33-column schema.")
    frame = frame.loc[:, ONE_SECOND_COLUMNS].copy()
    frame["timestamp_second"] = pd.to_datetime(frame["timestamp_second"], utc=True)
    frame["first_trade_timestamp"] = pd.to_datetime(frame["first_trade_timestamp"], utc=True)
    frame["last_trade_timestamp"] = pd.to_datetime(frame["last_trade_timestamp"], utc=True)
    frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.date
    frame["contract"] = frame["contract"].astype("string")
    for column in FLOAT_COLUMNS:
        frame[column] = frame[column].astype("float64")
    for column in UINT64_COLUMNS:
        frame[column] = frame[column].astype("uint64")
    for column in UINT32_COLUMNS:
        frame[column] = frame[column].astype("uint32")
    frame["inferred_delta"] = frame["inferred_delta"].astype("int64")
    frame["native_delta"] = frame["native_delta"].astype("int64")
    frame["contract_change"] = frame["contract_change"].astype(bool)
    return frame


def validate_one_second_session(session_data: pd.DataFrame) -> pd.DataFrame:
    """Validate a finished session before writing it permanently."""
    if session_data.empty:
        raise ValueError("Cannot write an empty session.")
    session_data = enforce_one_second_schema(session_data).sort_values(["timestamp_second", "instrument_id"]).reset_index(drop=True)
    if session_data["session_date"].nunique() != 1:
        raise ValueError("Session output contains more than one session_date.")
    if session_data.duplicated(["timestamp_second", "instrument_id"]).any():
        raise ValueError("Duplicate 1-second key detected.")
    if not session_data["timestamp_second"].is_monotonic_increasing:
        raise ValueError("Session timestamps are not chronological.")
    if not ((session_data["high"] >= session_data["open"]) & (session_data["high"] >= session_data["close"]) & (session_data["low"] <= session_data["open"]) & (session_data["low"] <= session_data["close"])).all():
        raise ValueError("OHLC validation failed.")
    if not ((session_data["total_volume"] > 0) & (session_data["trade_count"] > 0)).all():
        raise ValueError("Non-positive volume or trade count.")
    signed = lambda column: session_data[column].astype("int64")
    if not (signed("inferred_buy_volume") + signed("inferred_sell_volume") + signed("inferred_unknown_volume") == signed("total_volume")).all():
        raise ValueError("Inferred volume does not reconcile.")
    if not (signed("native_buy_volume") + signed("native_sell_volume") + signed("native_unknown_volume") == signed("total_volume")).all():
        raise ValueError("Native volume does not reconcile.")
    if not (signed("inferred_buy_volume") - signed("inferred_sell_volume") == signed("inferred_delta")).all():
        raise ValueError("Inferred delta does not reconcile.")
    if not (signed("native_buy_volume") - signed("native_sell_volume") == signed("native_delta")).all():
        raise ValueError("Native delta does not reconcile.")
    if not (signed("inferred_buy_trade_count") + signed("inferred_sell_trade_count") + signed("inferred_unknown_trade_count") == signed("trade_count")).all():
        raise ValueError("Inferred trade counts do not reconcile.")
    movement_count = signed("uptick_count") + signed("downtick_count") + signed("same_price_count")
    if not (movement_count <= signed("trade_count")).all():
        raise ValueError("Price-movement counts exceed trade count.")
    if not (session_data["max_volume_at_price"] <= session_data["total_volume"]).all():
        raise ValueError("max_volume_at_price exceeds total_volume.")
    if not (session_data["first_trade_timestamp"] <= session_data["last_trade_timestamp"]).all():
        raise ValueError("Trade timestamps are not ordered.")
    if not (session_data["first_trade_timestamp"].dt.floor("s") == session_data["timestamp_second"]).all():
        raise ValueError("First trade is outside its event-time second.")
    if not (session_data["last_trade_timestamp"].dt.floor("s") == session_data["timestamp_second"]).all():
        raise ValueError("Last trade is outside its event-time second.")
    if not (session_data["session_date"] == event_time_session_date(session_data["timestamp_second"])).all():
        raise ValueError("CME session-date assignment is inconsistent.")
    if not (session_data["contract"] == session_data["instrument_id"].map(CONTRACT_MAP)).all():
        raise ValueError("Contract mapping is inconsistent.")
    return session_data


def _session_manifest_record(session_data: pd.DataFrame) -> dict[str, Any]:
    session_date = str(session_data["session_date"].iloc[0])
    instruments = []
    for (instrument_id, contract), part in session_data.groupby(["instrument_id", "contract"], sort=True):
        instruments.append({"instrument_id": int(instrument_id), "contract": str(contract), "rows": int(len(part)), "trade_count": int(part["trade_count"].sum()), "volume": int(part["total_volume"].sum())})
    calendar_dates = [(date.fromisoformat(session_date) - timedelta(days=1)).isoformat(), session_date]
    degraded = [day for day in calendar_dates if day in DEGRADED_CALENDAR_DATES]
    changes = session_data.loc[session_data["contract_change"], "timestamp_second"].astype(str).tolist()
    return {"session_date": session_date, "row_count": int(len(session_data)), "trade_count": int(session_data["trade_count"].sum()), "volume": int(session_data["total_volume"].sum()), "instruments": instruments, "contract_change_count": len(changes), "contract_change_timestamps": changes, "is_complete_session": None, "degraded_status": "degraded" if degraded else "available", "degraded_calendar_dates": degraded}


def write_session_parquet(session_data: pd.DataFrame, output_dir: Path) -> tuple[Path, dict[str, Any]]:
    """Validate, write with the fixed Arrow schema, reload, and return metadata."""
    session_data = validate_one_second_session(session_data)
    session_date = session_data["session_date"].iloc[0]
    output_path = output_dir / f"MES_1s_{session_date}.parquet"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing session: {output_path}")
    table = pa.Table.from_pandas(session_data, schema=ONE_SECOND_ARROW_SCHEMA, preserve_index=False, safe=True)
    pq.write_table(table, output_path, compression="zstd")
    reloaded_schema = pq.ParquetFile(output_path).schema_arrow.remove_metadata()
    if reloaded_schema != ONE_SECOND_ARROW_SCHEMA:
        raise ValueError("Parquet writer did not preserve the permanent Arrow schema.")
    return output_path, _session_manifest_record(session_data)


def _session_bounds_utc(session_date_value: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the nominal regular CME window used for source-coverage metadata.

    ``is_complete_session`` means the raw source reaches both ends of this
    normal 17:00--16:00 Chicago window.  It does not assert that CME operated
    normal hours: holidays, early closes, and exchange-quality conditions need
    separate downstream treatment.
    """
    end_date = date.fromisoformat(session_date_value)
    start = pd.Timestamp(datetime.combine(end_date - timedelta(days=1), time(17)), tz=CHICAGO_TIMEZONE).tz_convert("UTC")
    end = pd.Timestamp(datetime.combine(end_date, time(16)), tz=CHICAGO_TIMEZONE).tz_convert("UTC")
    return start, end


def write_run_manifests(output_dir: Path, session_records: list[dict[str, Any]], raw_path: Path,
                        raw_trades: int, raw_volume: int, first_event: pd.Timestamp,
                        last_event: pd.Timestamp) -> None:
    """Write compact JSON manifests after all sessions have been completed."""
    # Keep provenance portable when the immutable raw file lives inside this
    # repository. A caller may deliberately provide an external source path,
    # in which case preserving that supplied path is more informative.
    try:
        raw_source = raw_path.resolve().relative_to(Path(__file__).resolve().parents[1]).as_posix()
    except ValueError:
        raw_source = str(raw_path)
    for record in session_records:
        start, end = _session_bounds_utc(record["session_date"])
        # This is deliberately a raw-source coverage check, not a claim that
        # the exchange had a normal full session on a holiday or early close.
        record["is_complete_session"] = bool(first_event <= start and last_event >= end)
    run_manifest = {
        "schema_version": SCHEMA_VERSION,
        "timestamp_policy": TIMESTAMP_POLICY,
        "session_timezone": CHICAGO_TIMEZONE,
        "tick_rule": "tick_rule_v1; state resets when instrument_id changes",
        "raw_source": raw_source,
        "raw_trades": raw_trades,
        "raw_volume": raw_volume,
        "first_event_timestamp": str(first_event),
        "last_event_timestamp": str(last_event),
        "session_files": len(session_records),
        "one_second_rows": sum(record["row_count"] for record in session_records),
        "known_degraded_calendar_dates": DEGRADED_CALENDAR_DATES,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n")
    (output_dir / "session_manifest.json").write_text(json.dumps(session_records, indent=2) + "\n")


def process_full_history_event_time(raw_path, output_dir, chunk_size=500_000):
    """Build a new event-time dataset.  Call explicitly; this module never runs it on import."""
    raw_path, output_dir = Path(raw_path), Path(output_dir)
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    pending = pd.DataFrame()
    previous_price = None
    previous_direction = 0
    previous_trade_instrument = None
    previous_second_instrument = None
    previous_event = None
    active_date = None
    active_parts: list[pd.DataFrame] = []
    records: list[dict[str, Any]] = []
    raw_trades = raw_volume = second_rows = 0
    first_event = last_event = None

    def flush_active_session():
        nonlocal active_date, active_parts, second_rows
        if active_date is None:
            return
        frame = pd.concat(active_parts, ignore_index=True)
        _, record = write_session_parquet(frame, output_dir)
        records.append(record)
        second_rows += len(frame)
        active_date, active_parts = None, []

    for raw_chunk in DBNStore.from_file(raw_path).to_df(count=chunk_size):
        previous_event = assert_event_time_order(raw_chunk, previous_event)
        event_time = pd.to_datetime(raw_chunk["ts_event"], utc=True)
        first_event = event_time.iloc[0] if first_event is None else first_event
        last_event = event_time.iloc[-1]
        raw_trades += len(raw_chunk)
        raw_volume += int(raw_chunk["size"].astype("uint64").sum())
        contextual = add_trade_context(raw_chunk)
        if not pending.empty:
            contextual = pd.concat([pending, contextual], ignore_index=True)
        complete, pending = split_complete_seconds(contextual)
        if complete.empty:
            continue
        prepared, previous_price, previous_direction, previous_trade_instrument = prepare_complete_trades(
            complete, previous_price, previous_direction, previous_trade_instrument
        )
        seconds = aggregate_to_one_second(prepared).sort_values(["timestamp_second", "instrument_id"]).reset_index(drop=True)
        seconds, previous_second_instrument = add_contract_change_flag(seconds, previous_second_instrument)
        seconds = enforce_one_second_schema(seconds)
        for session_date, part in seconds.groupby("session_date", sort=True):
            if active_date is None:
                active_date = session_date
            elif session_date != active_date:
                flush_active_session()
                active_date = session_date
            active_parts.append(part.copy())
    if not pending.empty:
        prepared, previous_price, previous_direction, previous_trade_instrument = prepare_complete_trades(
            pending, previous_price, previous_direction, previous_trade_instrument
        )
        seconds = aggregate_to_one_second(prepared).sort_values(["timestamp_second", "instrument_id"]).reset_index(drop=True)
        seconds, previous_second_instrument = add_contract_change_flag(seconds, previous_second_instrument)
        seconds = enforce_one_second_schema(seconds)
        for session_date, part in seconds.groupby("session_date", sort=True):
            if active_date is None:
                active_date = session_date
            elif session_date != active_date:
                flush_active_session()
                active_date = session_date
            active_parts.append(part.copy())
    flush_active_session()
    write_run_manifests(output_dir, records, raw_path, raw_trades, raw_volume, first_event, last_event)
    return {"raw_trades": raw_trades, "raw_volume": raw_volume, "second_rows": second_rows, "sessions_written": len(records), "output_dir": str(output_dir)}


def scan_raw_event_time_reference(raw_path, chunk_size=500_000) -> dict[str, Any]:
    """Memory-safe independent raw audit for event-time count, trades, and volume."""
    previous_event = None
    pending = pd.DataFrame()
    raw_trades = raw_volume = second_rows = 0
    first_event = last_event = None
    for chunk in DBNStore.from_file(raw_path).to_df(count=chunk_size):
        previous_event = assert_event_time_order(chunk, previous_event)
        event = pd.to_datetime(chunk["ts_event"], utc=True)
        first_event = event.iloc[0] if first_event is None else first_event
        last_event = event.iloc[-1]
        raw_trades += len(chunk)
        raw_volume += int(chunk["size"].astype("uint64").sum())
        keys = pd.DataFrame({"timestamp_second": event.dt.floor("s"), "instrument_id": chunk["instrument_id"].to_numpy()})
        if not pending.empty:
            keys = pd.concat([pending, keys], ignore_index=True)
        final_second = keys["timestamp_second"].iloc[-1]
        complete = keys.loc[keys["timestamp_second"] != final_second]
        pending = keys.loc[keys["timestamp_second"] == final_second].copy()
        second_rows += len(complete.drop_duplicates(["timestamp_second", "instrument_id"]))
    second_rows += len(pending.drop_duplicates(["timestamp_second", "instrument_id"]))
    return {"raw_trades": raw_trades, "raw_volume": raw_volume, "event_time_second_rows": second_rows, "first_event_timestamp": str(first_event), "last_event_timestamp": str(last_event)}


def audit_event_time_dataset(output_dir, raw_path, chunk_size=500_000) -> dict[str, Any]:
    """Independently audit a completed v2 dataset without loading it all at once."""
    output_dir, raw_path = Path(output_dir), Path(raw_path)
    files = sorted(output_dir.glob("MES_1s_*.parquet"))
    if not files:
        raise FileNotFoundError("No session Parquet files found.")
    run_manifest = json.loads((output_dir / "run_manifest.json").read_text())
    sessions = json.loads((output_dir / "session_manifest.json").read_text())
    if run_manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Run manifest schema version is not the v2 schema.")
    if run_manifest.get("timestamp_policy") != TIMESTAMP_POLICY:
        raise ValueError("Run manifest does not document the event-time policy.")
    schema = ONE_SECOND_ARROW_SCHEMA
    rows = trades = volume = duplicates = 0
    prior_instrument = None
    prior_timestamp = None
    observed_changes = []
    session_records = {record["session_date"]: record for record in sessions}
    if len(session_records) != len(sessions):
        raise ValueError("Session manifest has duplicate session-date records.")
    for path in files:
        if pq.ParquetFile(path).schema_arrow.remove_metadata() != schema:
            raise ValueError(f"Schema mismatch: {path.name}")
        frame = pd.read_parquet(path)
        validate_one_second_session(frame)
        file_session_date = path.stem.removeprefix("MES_1s_")
        stored_session_date = str(frame["session_date"].iloc[0])
        if file_session_date != stored_session_date or stored_session_date not in session_records:
            raise ValueError(f"Filename/session manifest mismatch: {path.name}")
        if prior_timestamp is not None and frame["timestamp_second"].iloc[0] < prior_timestamp:
            raise ValueError("Session files are not globally chronological.")
        expected_roll = frame["instrument_id"].ne(frame["instrument_id"].shift())
        expected_roll.iloc[0] = (
            False if prior_instrument is None else frame["instrument_id"].iloc[0] != prior_instrument
        )
        if not frame["contract_change"].equals(expected_roll.astype(bool)):
            raise ValueError(f"Contract-change flag is inconsistent: {path.name}")
        record = session_records[stored_session_date]
        # Rebuild the session-level facts from the stored Parquet file instead
        # of trusting the manifest that accompanied it.  These fields drive
        # later session selection, so a globally reconciled dataset is not
        # sufficient if an individual manifest record is wrong.
        observed_record = _session_manifest_record(frame)
        for field in [
            "row_count",
            "trade_count",
            "volume",
            "instruments",
            "contract_change_count",
            "contract_change_timestamps",
        ]:
            if record.get(field) != observed_record[field]:
                raise ValueError(f"Session manifest {field} mismatch: {path.name}")
        rows += len(frame)
        trades += int(frame["trade_count"].astype("uint64").sum())
        volume += int(frame["total_volume"].astype("uint64").sum())
        duplicates += int(frame.duplicated(["timestamp_second", "instrument_id"]).sum())
        for _, row in frame.loc[frame["contract_change"], ["timestamp_second", "instrument_id"]].iterrows():
            observed_changes.append((str(row["timestamp_second"]), int(row["instrument_id"])))
        prior_instrument = frame["instrument_id"].iloc[-1]
        prior_timestamp = frame["timestamp_second"].iloc[-1]
    raw_reference = scan_raw_event_time_reference(raw_path, chunk_size)
    if trades != raw_reference["raw_trades"]:
        raise ValueError("Output trade count does not reconcile to raw data.")
    if volume != raw_reference["raw_volume"]:
        raise ValueError("Output volume does not reconcile to raw data.")
    if rows != raw_reference["event_time_second_rows"]:
        raise ValueError("Output event-time row count does not reconcile to raw data.")
    if len(files) != len(sessions) or duplicates:
        raise ValueError("File/session manifest count or duplicate-key audit failed.")
    for field, observed in {
        "raw_trades": raw_reference["raw_trades"],
        "raw_volume": raw_reference["raw_volume"],
        "one_second_rows": rows,
        "session_files": len(files),
        "first_event_timestamp": raw_reference["first_event_timestamp"],
        "last_event_timestamp": raw_reference["last_event_timestamp"],
        "schema_version": SCHEMA_VERSION,
        "timestamp_policy": TIMESTAMP_POLICY,
    }.items():
        if run_manifest.get(field) != observed:
            raise ValueError(f"Run manifest {field} mismatch.")
    for record in sessions:
        start, end = _session_bounds_utc(record["session_date"])
        expected_complete = bool(
            pd.Timestamp(raw_reference["first_event_timestamp"]) <= start
            and pd.Timestamp(raw_reference["last_event_timestamp"]) >= end
        )
        if record["is_complete_session"] != expected_complete:
            raise ValueError(f"Session completeness manifest mismatch: {record['session_date']}")
        calendar_dates = [
            (date.fromisoformat(record["session_date"]) - timedelta(days=1)).isoformat(),
            record["session_date"],
        ]
        expected_degraded = [day for day in calendar_dates if day in DEGRADED_CALENDAR_DATES]
        expected_status = "degraded" if expected_degraded else "available"
        if record["degraded_status"] != expected_status or record["degraded_calendar_dates"] != expected_degraded:
            raise ValueError(f"Degraded-date manifest mismatch: {record['session_date']}")
    return {"files": len(files), "sessions": len(sessions), "rows": rows, "trade_count": trades, "volume": volume, "event_time_second_rows": raw_reference["event_time_second_rows"], "contract_changes": observed_changes, "partial_sessions": [record["session_date"] for record in sessions if not record["is_complete_session"]], "degraded_sessions": [record["session_date"] for record in sessions if record["degraded_status"] != "available"], "schema_version": run_manifest["schema_version"], "passed": True}
