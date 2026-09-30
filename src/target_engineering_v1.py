"""In-memory construction and validation for the frozen 15-minute MES target.

This module reads the approved event-time foundations only. It does not write
target data, create model features, or inspect predictive performance.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MINUTE_DIR = PROJECT_ROOT / "data" / "processed" / "1m" / "full_history_v1"
DEFAULT_SECOND_DIR = PROJECT_ROOT / "data" / "processed" / "1s" / "full_history_v2_event_time"
DEFAULT_TARGET_DIR = PROJECT_ROOT / "data" / "processed" / "targets" / "target_v1"
TARGET_HORIZON = pd.Timedelta(minutes=15)
REFERENCE_DELAY_LIMIT = pd.Timedelta(seconds=10)
TARGET_SCHEMA_VERSION = "mes_target_v1_event_time_15m_10s"

ELIGIBLE = "ELIGIBLE"
NO_FUTURE_BOUNDARY = "NO_FUTURE_BOUNDARY"
CROSSES_SESSION_BOUNDARY = "CROSSES_SESSION_BOUNDARY"
CROSSES_CONTRACT_ROLL = "CROSSES_CONTRACT_ROLL"
CROSSES_INTERNAL_GAP = "CROSSES_INTERNAL_GAP"
NO_START_REFERENCE_WITHIN_10S = "NO_START_REFERENCE_WITHIN_10S"
NO_FUTURE_REFERENCE_WITHIN_10S = "NO_FUTURE_REFERENCE_WITHIN_10S"
EQUAL_REFERENCE_PRICE = "EQUAL_REFERENCE_PRICE"

TARGET_COLUMNS = [
    "decision_timestamp",
    "session_date",
    "instrument_id",
    "contract",
    "start_reference_timestamp",
    "start_reference_delay_seconds",
    "start_reference_price",
    "future_boundary_timestamp",
    "future_reference_timestamp",
    "future_reference_delay_seconds",
    "future_reference_price",
    "forward_price_change",
    "forward_log_return",
    "target_up_15m",
    "target_eligible",
    "eligibility_reason",
]

TARGET_ARROW_SCHEMA = pa.schema([
    pa.field("decision_timestamp", pa.timestamp("ns", tz="UTC")),
    pa.field("session_date", pa.date32()),
    pa.field("instrument_id", pa.uint32()),
    pa.field("contract", pa.large_string()),
    pa.field("start_reference_timestamp", pa.timestamp("ns", tz="UTC")),
    pa.field("start_reference_delay_seconds", pa.float64()),
    pa.field("start_reference_price", pa.float64()),
    pa.field("future_boundary_timestamp", pa.timestamp("ns", tz="UTC")),
    pa.field("future_reference_timestamp", pa.timestamp("ns", tz="UTC")),
    pa.field("future_reference_delay_seconds", pa.float64()),
    pa.field("future_reference_price", pa.float64()),
    pa.field("forward_price_change", pa.float64()),
    pa.field("forward_log_return", pa.float64()),
    pa.field("target_up_15m", pa.int8()),
    pa.field("target_eligible", pa.bool_()),
    pa.field("eligibility_reason", pa.large_string()),
])


def load_one_minute_decision_rows(session_date, input_dir: Path = DEFAULT_MINUTE_DIR) -> pd.DataFrame:
    """Load one frozen one-minute session without altering it."""
    path = Path(input_dir) / f"MES_1m_{session_date}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"One-minute session file not found: {path}")
    return _prepare_minute_rows(pd.read_parquet(path))


def load_v2_one_second_session(session_date, input_dir: Path = DEFAULT_SECOND_DIR) -> pd.DataFrame:
    """Load the matching frozen V2 event-time seconds without altering them."""
    path = Path(input_dir) / f"MES_1s_{session_date}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"V2 one-second session file not found: {path}")
    return _prepare_second_rows(pd.read_parquet(path))


def first_actual_trade_within_10_seconds(seconds: pd.DataFrame, boundary) -> dict | None:
    """Return the first V2 trade after a boundary when delay is in [0, 10s)."""
    prepared = _prepare_second_rows(seconds)
    return _reference_from_sorted_seconds(prepared, pd.to_datetime(boundary, utc=True))


def evaluate_target_window(
    minute_rows: pd.DataFrame,
    decision_timestamp,
    boundary_lookup: dict[pd.Timestamp, pd.Series] | None = None,
) -> tuple[str, pd.Timestamp]:
    """Apply elapsed-time, session, roll, and gap rules before price lookup.

    Failure priority is fixed: missing future boundary, session boundary,
    contract roll, then missing internal minute. Reference-price failures are
    evaluated later only after this structural window is valid.
    """
    if boundary_lookup is None:
        boundary_lookup = _boundary_lookup(minute_rows)

    decision_timestamp = pd.to_datetime(decision_timestamp, utc=True)
    start_row = boundary_lookup.get(decision_timestamp)
    if start_row is None:
        raise ValueError("Candidate decision timestamp is absent from minute rows.")

    future_boundary = decision_timestamp + TARGET_HORIZON
    expected = [
        decision_timestamp + pd.Timedelta(minutes=offset)
        for offset in range(TARGET_HORIZON.components.minutes + 1)
    ]
    path = [boundary_lookup.get(timestamp) for timestamp in expected]

    if path[-1] is None:
        return NO_FUTURE_BOUNDARY, future_boundary

    start_session = start_row["session_date"]
    start_instrument = start_row["instrument_id"]
    present_rows = [row for row in path if row is not None]
    if any(row["session_date"] != start_session for row in present_rows):
        return CROSSES_SESSION_BOUNDARY, future_boundary
    if any(row["instrument_id"] != start_instrument for row in present_rows):
        return CROSSES_CONTRACT_ROLL, future_boundary
    if any(row is None for row in path[1:-1]):
        return CROSSES_INTERNAL_GAP, future_boundary
    return ELIGIBLE, future_boundary


def construct_15m_target_for_session(
    minute_rows: pd.DataFrame,
    seconds: pd.DataFrame,
) -> pd.DataFrame:
    """Construct frozen 15-minute target diagnostics for one session in memory."""
    minutes = _prepare_minute_rows(minute_rows)
    prepared_seconds = _prepare_second_rows(seconds)
    boundary_lookup = _boundary_lookup(minutes)
    reference_lookup = _reference_lookup_for_boundaries(
        prepared_seconds,
        set(boundary_lookup) | {timestamp + TARGET_HORIZON for timestamp in boundary_lookup},
    )
    records = []

    for row in minutes.itertuples(index=False):
        decision_timestamp = pd.to_datetime(row.decision_timestamp, utc=True)
        reason, future_boundary = evaluate_target_window(
            minutes,
            decision_timestamp,
            boundary_lookup,
        )
        record = _empty_target_record(row, future_boundary, reason)

        if reason == ELIGIBLE:
            start_reference = reference_lookup[decision_timestamp]
            future_reference = reference_lookup[future_boundary]
            _store_reference(record, "start", start_reference)
            _store_reference(record, "future", future_reference)

            if start_reference is None:
                record["eligibility_reason"] = NO_START_REFERENCE_WITHIN_10S
            elif future_reference is None:
                record["eligibility_reason"] = NO_FUTURE_REFERENCE_WITHIN_10S
            else:
                price_change = future_reference["price"] - start_reference["price"]
                record["forward_price_change"] = float(price_change)
                record["forward_log_return"] = float(
                    np.log(future_reference["price"] / start_reference["price"])
                )
                if price_change > 0:
                    record["target_up_15m"] = 1
                    record["target_eligible"] = True
                elif price_change < 0:
                    record["target_up_15m"] = 0
                    record["target_eligible"] = True
                else:
                    record["eligibility_reason"] = EQUAL_REFERENCE_PRICE

        records.append(record)

    result = _enforce_target_schema(pd.DataFrame.from_records(records, columns=TARGET_COLUMNS))
    validate_one_session_target_output(result)
    return result


def validate_one_session_target_output(target_rows: pd.DataFrame) -> None:
    """Check target timing, outcome, and reason-code invariants for one session."""
    if list(target_rows.columns) != TARGET_COLUMNS:
        raise ValueError("Target output columns do not match the fixed V1 contract.")
    if target_rows["decision_timestamp"].duplicated().any():
        raise ValueError("Target output has duplicate decision timestamps.")
    if not (
        target_rows["future_boundary_timestamp"]
        == target_rows["decision_timestamp"] + TARGET_HORIZON
    ).all():
        raise ValueError("Future boundary is not exactly 15 elapsed minutes later.")

    eligible = target_rows["target_eligible"]
    ineligible = ~eligible
    if not target_rows.loc[eligible, "eligibility_reason"].eq(ELIGIBLE).all():
        raise ValueError("Eligible target has a non-eligible reason code.")
    if not target_rows.loc[ineligible, "eligibility_reason"].ne(ELIGIBLE).all():
        raise ValueError("Ineligible target has an eligible reason code.")
    if not target_rows.loc[eligible, "target_up_15m"].isin([0, 1]).all():
        raise ValueError("Eligible target is not binary.")
    if target_rows.loc[ineligible, "target_up_15m"].notna().any():
        raise ValueError("Ineligible target has a binary outcome.")
    if not target_rows.loc[
        eligible & target_rows["target_up_15m"].eq(1), "forward_price_change"
    ].gt(0).all():
        raise ValueError("UP target does not have a positive forward price change.")
    if not target_rows.loc[
        eligible & target_rows["target_up_15m"].eq(0), "forward_price_change"
    ].lt(0).all():
        raise ValueError("DOWN target does not have a negative forward price change.")

    for prefix in ["start", "future"]:
        delay = target_rows.loc[eligible, f"{prefix}_reference_delay_seconds"]
        if not ((delay >= 0) & (delay < REFERENCE_DELAY_LIMIT.total_seconds())).all():
            raise ValueError(f"Eligible {prefix} reference delay violates [0, 10s).")
        if target_rows.loc[eligible, f"{prefix}_reference_price"].isna().any():
            raise ValueError(f"Eligible target lacks a {prefix} reference price.")

    equal = target_rows["eligibility_reason"].eq(EQUAL_REFERENCE_PRICE)
    if equal.any() and not (
        target_rows.loc[equal, "forward_price_change"].eq(0).all()
    ):
        raise ValueError("Equal-price exclusion does not have zero price change.")


def summarize_target_session(target_rows: pd.DataFrame) -> dict:
    """Return compact target-construction diagnostics for a representative session."""
    validate_one_session_target_output(target_rows)
    eligible = target_rows.loc[target_rows["target_eligible"]]
    reasons = {
        str(reason): int(count)
        for reason, count in target_rows.loc[
            ~target_rows["target_eligible"], "eligibility_reason"
        ].value_counts().sort_index().items()
    }
    return {
        "candidate_boundaries": int(len(target_rows)),
        "eligible_targets": int(len(eligible)),
        "ineligible_by_reason": reasons,
        "up_count": int((eligible["target_up_15m"] == 1).sum()),
        "down_count": int((eligible["target_up_15m"] == 0).sum()),
        "equal_price_exclusions": int(
            target_rows["eligibility_reason"].eq(EQUAL_REFERENCE_PRICE).sum()
        ),
        "start_delay_seconds": _delay_summary(eligible["start_reference_delay_seconds"]),
        "future_delay_seconds": _delay_summary(eligible["future_reference_delay_seconds"]),
        "earliest_eligible_t": (
            eligible["decision_timestamp"].min() if not eligible.empty else pd.NaT
        ),
        "latest_eligible_t": (
            eligible["decision_timestamp"].max() if not eligible.empty else pd.NaT
        ),
    }


def run_deterministic_target_tests() -> dict[str, str]:
    """Run compact boundary, outcome, roll, and partial-source checks in memory."""
    boundary = pd.Timestamp("2026-06-22 14:00:00+00:00")
    exact = _synthetic_seconds([(boundary, 100.0)])
    near_limit = _synthetic_seconds([
        (boundary + pd.Timedelta(seconds=9, nanoseconds=999_999_999), 100.0),
    ])
    at_limit = _synthetic_seconds([(boundary + pd.Timedelta(seconds=10), 100.0)])
    empty = _synthetic_seconds([])

    assert first_actual_trade_within_10_seconds(exact, boundary)["price"] == 100.0
    assert first_actual_trade_within_10_seconds(near_limit, boundary) is not None
    assert first_actual_trade_within_10_seconds(at_limit, boundary) is None
    assert first_actual_trade_within_10_seconds(empty, boundary) is None

    minutes = _synthetic_minutes(boundary, 16)
    higher = construct_15m_target_for_session(
        minutes,
        _synthetic_seconds([(boundary, 100.0), (boundary + TARGET_HORIZON, 101.0)]),
    ).iloc[0]
    lower = construct_15m_target_for_session(
        minutes,
        _synthetic_seconds([(boundary, 101.0), (boundary + TARGET_HORIZON, 100.0)]),
    ).iloc[0]
    equal = construct_15m_target_for_session(
        minutes,
        _synthetic_seconds([(boundary, 100.0), (boundary + TARGET_HORIZON, 100.0)]),
    ).iloc[0]
    assert higher["target_up_15m"] == 1 and higher["target_eligible"]
    assert lower["target_up_15m"] == 0 and lower["target_eligible"]
    assert (
        not equal["target_eligible"]
        and equal["eligibility_reason"] == EQUAL_REFERENCE_PRICE
    )

    missing_internal = minutes.drop(index=minutes.index[8]).reset_index(drop=True)
    missing_result = construct_15m_target_for_session(
        missing_internal,
        _synthetic_seconds([(boundary, 100.0), (boundary + TARGET_HORIZON, 101.0)]),
    ).iloc[0]
    assert missing_result["eligibility_reason"] == CROSSES_INTERNAL_GAP

    session_crossing = minutes.copy()
    session_crossing.loc[session_crossing.index[-1], "session_date"] = pd.Timestamp(
        "2026-06-23"
    ).date()
    session_result = construct_15m_target_for_session(
        session_crossing,
        _synthetic_seconds([(boundary, 100.0), (boundary + TARGET_HORIZON, 101.0)]),
    ).iloc[0]
    assert session_result["eligibility_reason"] == CROSSES_SESSION_BOUNDARY

    roll_crossing = minutes.copy()
    roll_crossing.loc[roll_crossing.index[8], ["instrument_id", "contract"]] = [
        2,
        "MESU6",
    ]
    roll_result = construct_15m_target_for_session(
        roll_crossing,
        _synthetic_seconds([(boundary, 100.0), (boundary + TARGET_HORIZON, 101.0)]),
    ).iloc[0]
    assert roll_result["eligibility_reason"] == CROSSES_CONTRACT_ROLL

    early_close_boundary = pd.Timestamp("2025-11-27 17:00:00+00:00")
    early_close_minutes = _synthetic_minutes(
        early_close_boundary,
        16,
        session_date="2025-11-27",
    )
    early_close_result = construct_15m_target_for_session(
        early_close_minutes,
        _synthetic_seconds([
            (early_close_boundary, 100.0),
            (early_close_boundary + TARGET_HORIZON, 101.0),
        ]),
    ).iloc[0]
    assert early_close_result["target_eligible"]

    source_partial_boundary = pd.Timestamp("2025-10-07 03:00:00+00:00")
    source_partial_minutes = _synthetic_minutes(
        source_partial_boundary,
        16,
        session_date="2025-10-07",
    )
    source_partial_result = construct_15m_target_for_session(
        source_partial_minutes,
        _synthetic_seconds([
            (source_partial_boundary, 100.0),
            (source_partial_boundary + TARGET_HORIZON, 101.0),
        ]),
    ).iloc[0]
    assert source_partial_result["target_eligible"]

    return {
        "reference_exact_boundary": "PASS",
        "reference_9_999_seconds": "PASS",
        "reference_10_seconds_rejected": "PASS",
        "reference_absent_rejected": "PASS",
        "higher_lower_equal_outcomes": "PASS",
        "missing_internal_minute": "PASS",
        "session_boundary": "PASS",
        "contract_roll": "PASS",
        "early_close_before_close": "PASS",
        "source_partial_self_contained": "PASS",
    }


def write_one_target_session(target_rows: pd.DataFrame, output_dir: Path) -> Path:
    """Write one validated target session and verify its fixed Parquet schema."""
    validate_one_session_target_output(target_rows)
    session_dates = target_rows["session_date"].astype(str).unique()
    if len(session_dates) != 1:
        raise ValueError(f"Target output has multiple session dates: {session_dates.tolist()}")

    output_path = Path(output_dir) / f"MES_target_v1_{session_dates[0]}.parquet"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing target file: {output_path}")

    table = pa.Table.from_pandas(
        target_rows,
        schema=TARGET_ARROW_SCHEMA,
        preserve_index=False,
        safe=True,
    )
    pq.write_table(table, output_path, compression="zstd")
    persisted_schema = pq.ParquetFile(output_path).schema_arrow.remove_metadata()
    if persisted_schema != TARGET_ARROW_SCHEMA:
        raise ValueError(f"Target schema changed during write: {output_path.name}")
    return output_path


def validate_target_session_against_sources(
    target_rows: pd.DataFrame,
    minute_rows: pd.DataFrame,
    seconds: pd.DataFrame,
) -> dict:
    """Independently reconcile one written target session to frozen inputs.

    This check keeps target outcomes separate from model features. It rebuilds
    the allowed endpoint references from V2 seconds and verifies every output
    reason against the elapsed-time decision path.
    """
    targets = _enforce_target_schema(target_rows)
    minutes = _prepare_minute_rows(minute_rows)
    prepared_seconds = _prepare_second_rows(seconds)
    validate_one_session_target_output(targets)

    if len(targets) != len(minutes):
        raise ValueError("Target row count does not equal one-minute candidate count.")
    source_identity = minutes.loc[:, [
        "decision_timestamp", "session_date", "instrument_id", "contract"
    ]].reset_index(drop=True)
    target_identity = targets.loc[:, [
        "decision_timestamp", "session_date", "instrument_id", "contract"
    ]].reset_index(drop=True)
    if not target_identity.equals(source_identity):
        raise ValueError("Target decision rows do not match the one-minute source.")
    if targets["decision_timestamp"].duplicated().any():
        raise ValueError("Target session has duplicate decision timestamps.")

    boundary_lookup = _boundary_lookup(minutes)
    boundaries = set(boundary_lookup) | {
        timestamp + TARGET_HORIZON for timestamp in boundary_lookup
    }
    reference_lookup = _reference_lookup_for_boundaries(prepared_seconds, boundaries)
    stages = {
        "candidate_rows": int(len(targets)),
        "structurally_valid_paths": 0,
        "reference_pairs_within_10s": 0,
        "equal_price_count": 0,
        "binary_eligible_rows": 0,
    }

    for row in targets.itertuples(index=False):
        decision_timestamp = pd.to_datetime(row.decision_timestamp, utc=True)
        structural_reason, future_boundary = evaluate_target_window(
            minutes,
            decision_timestamp,
            boundary_lookup,
        )
        if row.future_boundary_timestamp != future_boundary:
            raise ValueError("Target future boundary is not the expected elapsed timestamp.")

        expected_reason = structural_reason
        start_reference = future_reference = None
        if structural_reason == ELIGIBLE:
            stages["structurally_valid_paths"] += 1
            start_reference = reference_lookup[decision_timestamp]
            future_reference = reference_lookup[future_boundary]
            if start_reference is None:
                expected_reason = NO_START_REFERENCE_WITHIN_10S
            elif future_reference is None:
                expected_reason = NO_FUTURE_REFERENCE_WITHIN_10S
            else:
                stages["reference_pairs_within_10s"] += 1
                price_change = future_reference["price"] - start_reference["price"]
                if price_change == 0:
                    expected_reason = EQUAL_REFERENCE_PRICE
                    stages["equal_price_count"] += 1
                else:
                    expected_reason = ELIGIBLE
                    stages["binary_eligible_rows"] += 1

        if row.eligibility_reason != expected_reason:
            raise ValueError("Stored eligibility reason does not match source evidence.")
        _validate_reference_fields(row, "start", start_reference)
        _validate_reference_fields(row, "future", future_reference)

        if expected_reason == ELIGIBLE:
            expected_change = future_reference["price"] - start_reference["price"]
            expected_target = 1 if expected_change > 0 else 0
            if not bool(row.target_eligible) or int(row.target_up_15m) != expected_target:
                raise ValueError("Stored binary target does not match reference prices.")
            _validate_outcome_fields(row, expected_change, start_reference, future_reference)
        elif expected_reason == EQUAL_REFERENCE_PRICE:
            if bool(row.target_eligible) or not pd.isna(row.target_up_15m):
                raise ValueError("Equal reference price received a binary target.")
            _validate_outcome_fields(row, 0.0, start_reference, future_reference)
        elif bool(row.target_eligible) or not pd.isna(row.target_up_15m):
            raise ValueError("Ineligible target received a binary label.")
        elif not (
            pd.isna(row.forward_price_change) and pd.isna(row.forward_log_return)
        ):
            raise ValueError("Target without both references has an outcome value.")

    reason_counts = {
        str(reason): int(count)
        for reason, count in targets["eligibility_reason"].value_counts().sort_index().items()
    }
    stages["up_count"] = int((targets["target_up_15m"] == 1).sum())
    stages["down_count"] = int((targets["target_up_15m"] == 0).sum())
    stages["eligibility_reason_counts"] = reason_counts
    return stages


def build_full_history_target_v1(
    minute_dir: Path = DEFAULT_MINUTE_DIR,
    second_dir: Path = DEFAULT_SECOND_DIR,
    output_dir: Path = DEFAULT_TARGET_DIR,
) -> dict:
    """Build Target V1 once from frozen session files, with read-back audits.

    The destination must not already exist. Each session is loaded, written,
    and reconciled independently to keep memory bounded and prevent silent
    replacement of a completed target layer.
    """
    minute_dir = Path(minute_dir)
    second_dir = Path(second_dir)
    output_dir = Path(output_dir)
    if not minute_dir.exists() or not second_dir.exists():
        raise FileNotFoundError("Frozen one-minute or V2 one-second input directory is absent.")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to use existing target directory: {output_dir}")

    minute_paths = sorted(minute_dir.glob("MES_1m_*.parquet"))
    second_paths = sorted(second_dir.glob("MES_1s_*.parquet"))
    if not minute_paths or not second_paths:
        raise FileNotFoundError("Frozen target inputs contain no session Parquet files.")
    minute_by_session = {
        path.stem.removeprefix("MES_1m_"): path for path in minute_paths
    }
    second_by_session = {
        path.stem.removeprefix("MES_1s_"): path for path in second_paths
    }
    if set(minute_by_session) != set(second_by_session):
        raise ValueError("Frozen one-minute and V2 source session sets do not match.")

    output_dir.mkdir(parents=True, exist_ok=False)
    session_audits = []
    try:
        for file_number, session_date in enumerate(sorted(minute_by_session), start=1):
            minute_rows = load_one_minute_decision_rows(session_date, minute_dir)
            seconds = load_v2_one_second_session(session_date, second_dir)
            target_rows = construct_15m_target_for_session(minute_rows, seconds)
            output_path = write_one_target_session(target_rows, output_dir)
            reloaded = pd.read_parquet(output_path)
            persisted_schema = pq.ParquetFile(output_path).schema_arrow.remove_metadata()
            if persisted_schema != TARGET_ARROW_SCHEMA:
                raise ValueError(f"Target schema mismatch after read-back: {output_path.name}")
            audit = validate_target_session_against_sources(reloaded, minute_rows, seconds)
            audit["session_date"] = session_date
            session_audits.append(audit)
            print(
                f"[{file_number}/{len(minute_by_session)}] {session_date} "
                f"-> {audit['binary_eligible_rows']} binary targets"
            )
    except Exception:
        # A partial directory is preserved for inspection and is never reused.
        raise

    manifest = _build_run_manifest(
        minute_dir,
        second_dir,
        output_dir,
        session_audits,
    )
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def audit_full_history_target_v1(
    minute_dir: Path = DEFAULT_MINUTE_DIR,
    second_dir: Path = DEFAULT_SECOND_DIR,
    output_dir: Path = DEFAULT_TARGET_DIR,
) -> dict:
    """Read every Target V1 file and reconcile it to its frozen source session."""
    minute_dir = Path(minute_dir)
    second_dir = Path(second_dir)
    output_dir = Path(output_dir)
    manifest_path = output_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Target run manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != TARGET_SCHEMA_VERSION:
        raise ValueError("Target manifest schema version does not match V1.")

    target_paths = sorted(output_dir.glob("MES_target_v1_*.parquet"))
    if len(target_paths) != manifest.get("session_files"):
        raise ValueError("Target file count does not match the run manifest.")
    audits = []
    prior_timestamp = None
    for target_path in target_paths:
        session_date = target_path.stem.removeprefix("MES_target_v1_")
        target_rows = pd.read_parquet(target_path)
        if pq.ParquetFile(target_path).schema_arrow.remove_metadata() != TARGET_ARROW_SCHEMA:
            raise ValueError(f"Target schema mismatch: {target_path.name}")
        minute_rows = load_one_minute_decision_rows(session_date, minute_dir)
        seconds = load_v2_one_second_session(session_date, second_dir)
        if prior_timestamp is not None and target_rows["decision_timestamp"].iloc[0] <= prior_timestamp:
            raise ValueError("Target decision timestamps are not globally unique and ordered.")
        prior_timestamp = target_rows["decision_timestamp"].iloc[-1]
        audit = validate_target_session_against_sources(target_rows, minute_rows, seconds)
        audit["session_date"] = session_date
        audits.append(audit)

    observed = _aggregate_session_audits(audits)
    for field in [
        "session_files",
        "total_candidate_rows",
        "structurally_valid_paths",
        "reference_pairs_within_10s",
        "equal_price_count",
        "binary_eligible_rows",
        "up_count",
        "down_count",
        "eligibility_reason_counts",
    ]:
        if manifest.get(field) != observed[field]:
            raise ValueError(f"Target run manifest {field} does not reconcile.")
    return {**observed, "passed": True}


def _validate_reference_fields(row, prefix: str, expected: dict | None) -> None:
    timestamp = getattr(row, f"{prefix}_reference_timestamp")
    delay = getattr(row, f"{prefix}_reference_delay_seconds")
    price = getattr(row, f"{prefix}_reference_price")
    if expected is None:
        if not (pd.isna(timestamp) and pd.isna(delay) and pd.isna(price)):
            raise ValueError(f"Target stores an unavailable {prefix} reference.")
        return
    if timestamp != expected["timestamp"]:
        raise ValueError(f"Stored {prefix} reference timestamp is incorrect.")
    if not _same_float(delay, expected["delay_seconds"]):
        raise ValueError(f"Stored {prefix} reference delay is incorrect.")
    if not _same_float(price, expected["price"]):
        raise ValueError(f"Stored {prefix} reference price is incorrect.")


def _validate_outcome_fields(
    row,
    expected_change: float,
    start_reference: dict,
    future_reference: dict,
) -> None:
    expected_log_return = np.log(future_reference["price"] / start_reference["price"])
    if not _same_float(row.forward_price_change, expected_change):
        raise ValueError("Stored forward price change is incorrect.")
    if not _same_float(row.forward_log_return, expected_log_return):
        raise ValueError("Stored forward log return is incorrect.")
    if not np.isfinite(row.forward_log_return):
        raise ValueError("Stored forward log return is not finite.")


def _same_float(observed, expected: float) -> bool:
    return not pd.isna(observed) and float(observed) == float(expected)


def _aggregate_session_audits(session_audits: list[dict]) -> dict:
    if not session_audits:
        raise ValueError("Target audit has no session records.")
    reason_counts: dict[str, int] = {}
    for audit in session_audits:
        for reason, count in audit["eligibility_reason_counts"].items():
            reason_counts[reason] = reason_counts.get(reason, 0) + int(count)
    return {
        "session_files": len(session_audits),
        "total_candidate_rows": sum(audit["candidate_rows"] for audit in session_audits),
        "structurally_valid_paths": sum(
            audit["structurally_valid_paths"] for audit in session_audits
        ),
        "reference_pairs_within_10s": sum(
            audit["reference_pairs_within_10s"] for audit in session_audits
        ),
        "equal_price_count": sum(audit["equal_price_count"] for audit in session_audits),
        "binary_eligible_rows": sum(
            audit["binary_eligible_rows"] for audit in session_audits
        ),
        "up_count": sum(audit["up_count"] for audit in session_audits),
        "down_count": sum(audit["down_count"] for audit in session_audits),
        "eligibility_reason_counts": dict(sorted(reason_counts.items())),
    }


def _build_run_manifest(
    minute_dir: Path,
    second_dir: Path,
    output_dir: Path,
    session_audits: list[dict],
) -> dict:
    aggregate = _aggregate_session_audits(session_audits)
    minute_manifest = json.loads((minute_dir / "run_manifest.json").read_text())
    second_manifest = json.loads((second_dir / "run_manifest.json").read_text())
    ordered_audits = sorted(session_audits, key=lambda audit: audit["session_date"])
    target_paths = sorted(output_dir.glob("MES_target_v1_*.parquet"))
    if len(target_paths) != aggregate["session_files"]:
        raise ValueError("Target file count does not match completed session audits.")

    first_timestamp = pd.read_parquet(target_paths[0], columns=["decision_timestamp"])[
        "decision_timestamp"
    ].min()
    last_timestamp = pd.read_parquet(target_paths[-1], columns=["decision_timestamp"])[
        "decision_timestamp"
    ].max()
    roll_sessions = [
        audit["session_date"]
        for audit in ordered_audits
        if audit["eligibility_reason_counts"].get(CROSSES_CONTRACT_ROLL, 0) > 0
    ]
    return {
        "schema_version": TARGET_SCHEMA_VERSION,
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "source_one_minute": {
            "schema_version": minute_manifest["schema_version"],
            "path": _project_relative_path(minute_dir),
            "timestamp_policy": minute_manifest["timestamp_policy"],
            "decision_timestamp_convention": minute_manifest[
                "decision_timestamp_convention"
            ],
        },
        "source_one_second_v2": {
            "schema_version": second_manifest["schema_version"],
            "path": _project_relative_path(second_dir),
            "timestamp_policy": second_manifest["timestamp_policy"],
        },
        "prediction_horizon_minutes": 15,
        "reference_price_rule": (
            "first actual V2 event-time trade at or after each boundary; "
            "0 <= delay < 10 seconds; use the one-second row open"
        ),
        "equal_price_policy": "equal endpoint prices have no binary target",
        "continuity_policy": (
            "require every elapsed one-minute boundary from T through T+15m; "
            "do not bridge session boundaries, closures, gaps, or absent minutes"
        ),
        "contract_roll_policy": (
            "require one instrument_id throughout T through T+15m; "
            "targets crossing a roll are ineligible"
        ),
        "source_boundary_policy": (
            "2025-10-07 and 2026-09-11 remain; normal observation-level "
            "eligibility applies without synthetic coverage"
        ),
        "known_gap_outage_policy": (
            "the exact elapsed-minute continuity rule blocks the November 28, "
            "2025 outage and unresolved December 24/30 gaps"
        ),
        "session_files": aggregate["session_files"],
        "total_candidate_rows": aggregate["total_candidate_rows"],
        "structurally_valid_paths": aggregate["structurally_valid_paths"],
        "reference_pairs_within_10s": aggregate["reference_pairs_within_10s"],
        "equal_price_count": aggregate["equal_price_count"],
        "binary_eligible_rows": aggregate["binary_eligible_rows"],
        "up_count": aggregate["up_count"],
        "down_count": aggregate["down_count"],
        "eligibility_reason_counts": aggregate["eligibility_reason_counts"],
        "first_decision_timestamp": str(first_timestamp),
        "last_decision_timestamp": str(last_timestamp),
        "contract_roll_sessions": roll_sessions,
        "contract_roll_count": int(minute_manifest["contract_change_count"]),
        "source_boundary_sessions": minute_manifest[
            "partial_source_boundary_sessions"
        ],
        "target_outcome_fields": [
            "start_reference_timestamp",
            "start_reference_delay_seconds",
            "start_reference_price",
            "future_boundary_timestamp",
            "future_reference_timestamp",
            "future_reference_delay_seconds",
            "future_reference_price",
            "forward_price_change",
            "forward_log_return",
            "target_up_15m",
            "target_eligible",
            "eligibility_reason",
        ],
        "predictive_features_included": False,
        "target_outcomes_must_not_enter_feature_matrix": True,
        "session_audit": ordered_audits,
    }


def _project_relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def _prepare_minute_rows(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"decision_timestamp", "session_date", "instrument_id", "contract"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Minute rows lack required target fields: {sorted(missing)}")
    prepared = frame.copy()
    prepared["decision_timestamp"] = pd.to_datetime(
        prepared["decision_timestamp"],
        utc=True,
    )
    prepared["session_date"] = pd.to_datetime(prepared["session_date"]).dt.date
    prepared["instrument_id"] = prepared["instrument_id"].astype("uint32")
    prepared["contract"] = prepared["contract"].astype("string")
    prepared = prepared.sort_values("decision_timestamp").reset_index(drop=True)
    if prepared["decision_timestamp"].duplicated().any():
        raise ValueError("Minute rows contain duplicate decision timestamps.")
    return prepared


def _prepare_second_rows(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp_second", "first_trade_timestamp", "open"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"V2 seconds lack required target fields: {sorted(missing)}")
    prepared = frame.copy()
    prepared["timestamp_second"] = pd.to_datetime(prepared["timestamp_second"], utc=True)
    prepared["first_trade_timestamp"] = pd.to_datetime(
        prepared["first_trade_timestamp"],
        utc=True,
    )
    prepared["open"] = prepared["open"].astype("float64")
    return prepared.sort_values("first_trade_timestamp").reset_index(drop=True)


def _boundary_lookup(minute_rows: pd.DataFrame) -> dict[pd.Timestamp, pd.Series]:
    return {
        pd.to_datetime(row["decision_timestamp"], utc=True): row
        for _, row in minute_rows.iterrows()
    }


def _reference_lookup_for_boundaries(
    seconds: pd.DataFrame,
    boundaries: set[pd.Timestamp],
) -> dict[pd.Timestamp, dict | None]:
    """Find reference trades once for all candidate boundaries in a session."""
    first_trades = pd.DatetimeIndex(seconds["first_trade_timestamp"])
    return {
        pd.to_datetime(boundary, utc=True): _reference_from_sorted_seconds(
            seconds,
            pd.to_datetime(boundary, utc=True),
            first_trades,
        )
        for boundary in boundaries
    }


def _reference_from_sorted_seconds(
    seconds: pd.DataFrame,
    boundary: pd.Timestamp,
    first_trades: pd.DatetimeIndex | None = None,
) -> dict | None:
    if first_trades is None:
        first_trades = pd.DatetimeIndex(seconds["first_trade_timestamp"])
    position = first_trades.searchsorted(boundary, side="left")
    if position == len(seconds):
        return None

    row = seconds.iloc[position]
    timestamp = pd.to_datetime(row["first_trade_timestamp"], utc=True)
    delay_seconds = (timestamp - boundary).total_seconds()
    if not 0 <= delay_seconds < REFERENCE_DELAY_LIMIT.total_seconds():
        return None
    return {
        "timestamp": timestamp,
        "delay_seconds": float(delay_seconds),
        "price": float(row["open"]),
    }


def _empty_target_record(row, future_boundary, reason: str) -> dict:
    return {
        "decision_timestamp": pd.to_datetime(row.decision_timestamp, utc=True),
        "session_date": row.session_date,
        "instrument_id": row.instrument_id,
        "contract": row.contract,
        "start_reference_timestamp": pd.NaT,
        "start_reference_delay_seconds": np.nan,
        "start_reference_price": np.nan,
        "future_boundary_timestamp": future_boundary,
        "future_reference_timestamp": pd.NaT,
        "future_reference_delay_seconds": np.nan,
        "future_reference_price": np.nan,
        "forward_price_change": np.nan,
        "forward_log_return": np.nan,
        "target_up_15m": pd.NA,
        "target_eligible": False,
        "eligibility_reason": reason,
    }


def _store_reference(record: dict, prefix: str, reference: dict | None) -> None:
    if reference is None:
        return
    record[f"{prefix}_reference_timestamp"] = reference["timestamp"]
    record[f"{prefix}_reference_delay_seconds"] = reference["delay_seconds"]
    record[f"{prefix}_reference_price"] = reference["price"]


def _enforce_target_schema(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.loc[:, TARGET_COLUMNS].copy()
    for column in [
        "decision_timestamp",
        "start_reference_timestamp",
        "future_boundary_timestamp",
        "future_reference_timestamp",
    ]:
        result[column] = pd.to_datetime(result[column], utc=True)
    result["session_date"] = pd.to_datetime(result["session_date"]).dt.date
    result["instrument_id"] = result["instrument_id"].astype("uint32")
    result["contract"] = result["contract"].astype("string")
    for column in [
        "start_reference_delay_seconds",
        "start_reference_price",
        "future_reference_delay_seconds",
        "future_reference_price",
        "forward_price_change",
        "forward_log_return",
    ]:
        result[column] = result[column].astype("float64")
    result["target_up_15m"] = result["target_up_15m"].astype("Int8")
    result["target_eligible"] = result["target_eligible"].astype(bool)
    result["eligibility_reason"] = result["eligibility_reason"].astype("string")
    return result


def _delay_summary(delay: pd.Series) -> dict:
    if delay.empty:
        return {"count": 0, "median": None, "max": None}
    return {
        "count": int(delay.notna().sum()),
        "median": float(delay.median()),
        "max": float(delay.max()),
    }


def _synthetic_minutes(
    start,
    periods: int,
    session_date: str = "2026-06-22",
    instrument_id: int = 1,
    contract: str = "MESM6",
) -> pd.DataFrame:
    timestamps = pd.date_range(start, periods=periods, freq="min", tz="UTC")
    return pd.DataFrame({
        "decision_timestamp": timestamps,
        "session_date": [pd.Timestamp(session_date).date()] * periods,
        "instrument_id": [instrument_id] * periods,
        "contract": [contract] * periods,
    })


def _synthetic_seconds(entries: list[tuple[pd.Timestamp, float]]) -> pd.DataFrame:
    timestamps = pd.to_datetime([entry[0] for entry in entries], utc=True)
    return pd.DataFrame({
        "timestamp_second": timestamps.floor("s"),
        "first_trade_timestamp": timestamps,
        "open": [entry[1] for entry in entries],
    })
