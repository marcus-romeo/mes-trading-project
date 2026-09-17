"""Reusable feature-engineering utilities for the MES trading project.

Version 1 converts the validated V2 one-second event-time foundation into a
leakage-safe one-minute foundation. It is a durable market-data layer, not yet
the final model feature matrix.
"""

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ONE_MINUTE_SCHEMA_VERSION = "mes_1m_v1_from_v2_event_time_33"

# This ordered schema is part of the permanent one-minute data contract.
# Writing through it prevents a future pandas/PyArrow environment from silently
# changing physical Parquet types while preserving the same-looking values.
ONE_MINUTE_COLUMNS = [
    "decision_timestamp",
    "minute_start",
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
    "active_second_count",
    "uptick_count",
    "downtick_count",
    "same_price_count",
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
    "contract_change",
]

ONE_MINUTE_ARROW_SCHEMA = pa.schema([
    pa.field("decision_timestamp", pa.timestamp("ns", tz="UTC")),
    pa.field("minute_start", pa.timestamp("ns", tz="UTC")),
    pa.field("session_date", pa.date32()),
    pa.field("instrument_id", pa.uint32()),
    pa.field("contract", pa.large_string()),
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
    pa.field("active_second_count", pa.int64()),
    pa.field("uptick_count", pa.uint32()),
    pa.field("downtick_count", pa.uint32()),
    pa.field("same_price_count", pa.uint32()),
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
    pa.field("contract_change", pa.bool_()),
])

ONE_MINUTE_FLOAT_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "price_volume_sum",
    "price_squared_volume_sum",
]
ONE_MINUTE_UINT64_COLUMNS = [
    "total_volume",
    "size_squared_sum",
    "inferred_buy_volume",
    "inferred_sell_volume",
    "inferred_unknown_volume",
    "native_buy_volume",
    "native_sell_volume",
    "native_unknown_volume",
]
ONE_MINUTE_UINT32_COLUMNS = [
    "instrument_id",
    "trade_count",
    "max_trade_size",
    "uptick_count",
    "downtick_count",
    "same_price_count",
    "inferred_buy_trade_count",
    "inferred_sell_trade_count",
    "inferred_unknown_trade_count",
]


def enforce_one_minute_schema(frame):
    """Return the permanent ordered 33-column minute frame with stable dtypes."""
    if set(frame.columns) != set(ONE_MINUTE_COLUMNS):
        missing = sorted(set(ONE_MINUTE_COLUMNS) - set(frame.columns))
        unexpected = sorted(set(frame.columns) - set(ONE_MINUTE_COLUMNS))
        raise ValueError(
            "One-minute frame does not match the permanent schema. "
            f"Missing={missing}; unexpected={unexpected}."
        )

    frame = frame.loc[:, ONE_MINUTE_COLUMNS].copy()
    for column in [
        "decision_timestamp",
        "minute_start",
        "first_trade_timestamp",
        "last_trade_timestamp",
    ]:
        frame[column] = pd.to_datetime(frame[column], utc=True)
    frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.date
    frame["contract"] = frame["contract"].astype("string")
    for column in ONE_MINUTE_FLOAT_COLUMNS:
        frame[column] = frame[column].astype("float64")
    for column in ONE_MINUTE_UINT64_COLUMNS:
        frame[column] = frame[column].astype("uint64")
    for column in ONE_MINUTE_UINT32_COLUMNS:
        frame[column] = frame[column].astype("uint32")
    frame["active_second_count"] = frame["active_second_count"].astype("int64")
    frame["inferred_delta"] = frame["inferred_delta"].astype("int64")
    frame["native_delta"] = frame["native_delta"].astype("int64")
    frame["contract_change"] = frame["contract_change"].astype(bool)
    return frame


def aggregate_seconds_to_minutes(seconds):
    """Aggregate one validated V2 session from traded seconds to completed minutes.

    A row at ``decision_timestamp`` 10:31 contains only events from the
    completed interval [10:30:00, 10:31:00). It is therefore an event-time
    eligibility boundary: future live operation must additionally wait until
    the final relevant IBKR event has actually been received and processed.

    A minute may contain only one futures contract. Later rolling price
    features must restart after every contract change because Databento prices
    are unadjusted across rolls.
    """
    df = seconds.copy()
    for column in [
        "timestamp_second",
        "first_trade_timestamp",
        "last_trade_timestamp",
    ]:
        df[column] = pd.to_datetime(df[column], utc=True)

    # First/last rules depend on chronological event-time order.
    df = df.sort_values("timestamp_second").reset_index(drop=True)
    df["minute_start"] = df["timestamp_second"].dt.floor("min")

    contracts_per_minute = df.groupby("minute_start")["instrument_id"].nunique()
    if (contracts_per_minute > 1).any():
        bad_minutes = contracts_per_minute[contracts_per_minute > 1].index.tolist()
        raise ValueError(f"Mixed-contract minute detected: {bad_minutes[:5]}")

    # V2 has rows only for seconds containing trades. active_second_count is
    # consequently the number of traded seconds, not an imputed 60-second bar.
    minute = (
        df.groupby("minute_start", sort=True)
        .agg(
            session_date=("session_date", "first"),
            instrument_id=("instrument_id", "first"),
            contract=("contract", "first"),
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            price_volume_sum=("price_volume_sum", "sum"),
            price_squared_volume_sum=("price_squared_volume_sum", "sum"),
            total_volume=("total_volume", "sum"),
            trade_count=("trade_count", "sum"),
            max_trade_size=("max_trade_size", "max"),
            size_squared_sum=("size_squared_sum", "sum"),
            active_second_count=("timestamp_second", "count"),
            uptick_count=("uptick_count", "sum"),
            downtick_count=("downtick_count", "sum"),
            same_price_count=("same_price_count", "sum"),
            inferred_buy_volume=("inferred_buy_volume", "sum"),
            inferred_sell_volume=("inferred_sell_volume", "sum"),
            inferred_unknown_volume=("inferred_unknown_volume", "sum"),
            inferred_buy_trade_count=("inferred_buy_trade_count", "sum"),
            inferred_sell_trade_count=("inferred_sell_trade_count", "sum"),
            inferred_unknown_trade_count=("inferred_unknown_trade_count", "sum"),
            inferred_delta=("inferred_delta", "sum"),
            # Native fields are historical Databento benchmarks only. Future
            # production features use inferred fields reproducible from IBKR.
            native_buy_volume=("native_buy_volume", "sum"),
            native_sell_volume=("native_sell_volume", "sum"),
            native_unknown_volume=("native_unknown_volume", "sum"),
            native_delta=("native_delta", "sum"),
            first_trade_timestamp=("first_trade_timestamp", "min"),
            last_trade_timestamp=("last_trade_timestamp", "max"),
            contract_change=("contract_change", "max"),
        )
        .reset_index()
    )
    minute["decision_timestamp"] = minute["minute_start"] + pd.Timedelta(minutes=1)
    minute = enforce_one_minute_schema(minute)

    # Do not manufacture approximations for two V2 price-level fields.
    # Summing unique_price_levels double-counts recurring prices, while taking
    # a maximum of max_volume_at_price misses volume accumulated at one price
    # across multiple seconds. V2 remains available if that research is needed.
    if int(minute["total_volume"].sum()) != int(df["total_volume"].sum()):
        raise ValueError("One-minute volume does not reconcile to source seconds.")
    if int(minute["trade_count"].sum()) != int(df["trade_count"].sum()):
        raise ValueError("One-minute trade count does not reconcile to source seconds.")
    if not (minute["last_trade_timestamp"] < minute["decision_timestamp"]).all():
        raise ValueError(
            "A minute contains a trade at or after its decision timestamp."
        )

    return minute


def write_one_minute_session(input_path, output_dir):
    """Convert one validated V2 session into one permanently typed Parquet file.

    The source is read only, existing output is never overwritten, and the
    saved file is reloaded before the writer reports success.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input session file does not exist: {input_path}")

    seconds = pd.read_parquet(input_path)
    if seconds.empty:
        raise ValueError(f"Input session contains no rows: {input_path}")
    minutes = aggregate_seconds_to_minutes(seconds)

    session_dates = minutes["session_date"].astype(str).unique()
    if len(session_dates) != 1:
        raise ValueError(f"Expected one session_date, found: {session_dates.tolist()}")
    session_date = session_dates[0]
    output_path = output_dir / f"MES_1m_{session_date}.parquet"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {output_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(
        minutes,
        schema=ONE_MINUTE_ARROW_SCHEMA,
        preserve_index=False,
        safe=True,
    )
    pq.write_table(table, output_path)

    persisted_schema = pq.ParquetFile(output_path).schema_arrow.remove_metadata()
    if persisted_schema != ONE_MINUTE_ARROW_SCHEMA:
        raise ValueError(
            "Parquet writer did not preserve the permanent one-minute schema."
        )
    reloaded = pd.read_parquet(output_path)
    pd.testing.assert_frame_equal(reloaded, minutes, check_exact=True)
    if int(reloaded["total_volume"].sum()) != int(seconds["total_volume"].sum()):
        raise ValueError(
            "Persisted one-minute volume does not reconcile to source seconds."
        )
    if int(reloaded["trade_count"].sum()) != int(seconds["trade_count"].sum()):
        raise ValueError(
            "Persisted one-minute trade count does not reconcile to source seconds."
        )

    return {
        "session_date": session_date,
        "input_file": input_path.name,
        "output_file": output_path.name,
        "second_rows": len(seconds),
        "minute_rows": len(reloaded),
        "volume": int(reloaded["total_volume"].sum()),
        "trades": int(reloaded["trade_count"].sum()),
        "output_path": str(output_path),
    }


def build_full_history_minutes(input_dir, output_dir):
    """Build the complete one-minute foundation one validated V2 session at a time.

    This historical build is deliberately conservative: it requires an empty
    destination, refuses individual overwrites, and does not alter source V2.
    Partial and degraded-session eligibility remains documented in V2 manifests
    for downstream modeling rather than being inferred by this transformation.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    session_files = sorted(input_dir.glob("MES_1s_*.parquet"))
    if not session_files:
        raise FileNotFoundError(f"No V2 session Parquet files found in: {input_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to use non-empty output directory: {output_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    session_results = []
    for file_number, input_path in enumerate(session_files, start=1):
        result = write_one_minute_session(input_path, output_dir)
        session_results.append(result)
        print(
            f"[{file_number}/{len(session_files)}] "
            f"{result['session_date']} -> {result['minute_rows']} minute rows"
        )

    results_df = pd.DataFrame(session_results)
    if len(results_df) != len(session_files):
        raise RuntimeError("Not every discovered V2 session produced one minute file.")

    return results_df, {
        "sessions_written": len(results_df),
        "source_second_rows": int(results_df["second_rows"].sum()),
        "minute_rows": int(results_df["minute_rows"].sum()),
        "volume": int(results_df["volume"].sum()),
        "trades": int(results_df["trades"].sum()),
        "output_dir": str(output_dir),
    }
