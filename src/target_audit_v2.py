"""Independent, read-only reconciliation of Target V2 to frozen source files.

This audit deliberately does not call the Target V2 builder's window or
reference-lookup helpers. It walks elapsed minute keys and uses a separate
integer-nanosecond search of the V2 trade-bearing seconds.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.target_engineering_v2 import TARGET_ARROW_SCHEMA, TARGET_SCHEMA_VERSION


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MINUTE_DIR = PROJECT_ROOT / "data/processed/1m/full_history_v1"
SECOND_DIR = PROJECT_ROOT / "data/processed/1s/full_history_v2_event_time"
TARGET_DIR = PROJECT_ROOT / "data/processed/targets/target_v2"
MINUTE_NS = 60_000_000_000
HORIZON_NS = 15 * MINUTE_NS
DELAY_LIMIT_NS = 10_000_000_000


def _actual_reference(seconds: pd.DataFrame, times_ns: np.ndarray, boundary_ns: int):
    position = int(np.searchsorted(times_ns, boundary_ns, side="left"))
    if position == len(times_ns):
        return None
    delay_ns = int(times_ns[position]) - boundary_ns
    if delay_ns < 0 or delay_ns >= DELAY_LIMIT_NS:
        return None
    row = seconds.iloc[position]
    return (
        pd.Timestamp(int(times_ns[position]), unit="ns", tz="UTC"),
        float(row["open"]),
        int(row["instrument_id"]),
        delay_ns / 1_000_000_000,
    )


def _check_reference(row, side: str, actual) -> None:
    timestamp = getattr(row, f"{side}_reference_timestamp")
    price = getattr(row, f"{side}_reference_price")
    instrument_id = getattr(row, f"{side}_reference_instrument_id")
    delay = getattr(row, f"{side}_reference_delay_seconds")
    if actual is None:
        if not all(pd.isna(value) for value in (timestamp, price, instrument_id, delay)):
            raise ValueError(f"{row.decision_timestamp}: stored unavailable {side} reference")
        return
    actual_timestamp, actual_price, actual_instrument, actual_delay = actual
    if timestamp != actual_timestamp or float(price) != actual_price:
        raise ValueError(f"{row.decision_timestamp}: {side} timestamp or price differs from V2")
    if pd.isna(instrument_id) or int(instrument_id) != actual_instrument:
        raise ValueError(f"{row.decision_timestamp}: {side} instrument differs from V2")
    # V1/V2 store pandas Timedelta.total_seconds(), which truncates ns to us.
    # Use ns for eligibility and allow only that representation difference here.
    if pd.isna(delay) or not np.isclose(float(delay), actual_delay, atol=1e-6, rtol=0):
        raise ValueError(f"{row.decision_timestamp}: {side} delay differs from V2")
    if not 0 <= actual_delay < 10:
        raise ValueError(f"{row.decision_timestamp}: {side} violates strict delay")


def audit_target_v2_session(session_date: str, minute_dir=MINUTE_DIR, second_dir=SECOND_DIR, target_dir=TARGET_DIR) -> dict:
    """Rebuild each reason independently and compare every stored outcome field."""
    minute_path = Path(minute_dir) / f"MES_1m_{session_date}.parquet"
    second_path = Path(second_dir) / f"MES_1s_{session_date}.parquet"
    target_path = Path(target_dir) / f"MES_target_v2_{session_date}.parquet"
    if pq.ParquetFile(target_path).schema_arrow.remove_metadata() != TARGET_ARROW_SCHEMA:
        raise ValueError(f"{session_date}: wrong Target V2 Arrow schema")

    minutes = pd.read_parquet(minute_path, columns=[
        "decision_timestamp", "session_date", "instrument_id", "contract", "last_trade_timestamp",
    ])
    seconds = pd.read_parquet(second_path, columns=[
        "first_trade_timestamp", "instrument_id", "open",
    ]).sort_values("first_trade_timestamp", kind="stable").reset_index(drop=True)
    targets = pd.read_parquet(target_path)
    identity = ["decision_timestamp", "session_date", "instrument_id", "contract"]
    if len(minutes) != len(targets) or not minutes[identity].equals(targets[identity]):
        raise ValueError(f"{session_date}: target candidates differ from minute source")
    if minutes["decision_timestamp"].duplicated().any():
        raise ValueError(f"{session_date}: duplicate decision timestamp")
    if not (minutes["last_trade_timestamp"] < minutes["decision_timestamp"]).all():
        raise ValueError(f"{session_date}: minute uses an event at or after decision time")
    if not (targets["future_boundary_timestamp"] == targets["decision_timestamp"] + pd.Timedelta(minutes=15)).all():
        raise ValueError(f"{session_date}: future boundary differs from T+15m")

    minute_keys = {
        int(row.decision_timestamp.value): (row.session_date, int(row.instrument_id))
        for row in minutes.itertuples(index=False)
    }
    trade_times_ns = pd.DatetimeIndex(seconds["first_trade_timestamp"]).asi8
    if len(trade_times_ns) and not np.all(trade_times_ns[1:] >= trade_times_ns[:-1]):
        raise ValueError(f"{session_date}: seconds not ordered by first trade")

    counts = Counter()
    stages = Counter()
    mismatched_eligible_references = 0
    for row in targets.itertuples(index=False):
        t_ns = int(row.decision_timestamp.value)
        future_ns = t_ns + HORIZON_NS
        candidate_id = int(row.instrument_id)
        path = [minute_keys.get(t_ns + offset * MINUTE_NS) for offset in range(16)]
        if path[0] is None:
            raise ValueError(f"{row.decision_timestamp}: candidate minute absent")
        start_reference = future_reference = None
        if path[-1] is None:
            reason = "NO_FUTURE_BOUNDARY"
        elif any(item is not None and item[0] != row.session_date for item in path):
            reason = "CROSSES_SESSION_BOUNDARY"
        elif any(item is not None and item[1] != candidate_id for item in path):
            reason = "CROSSES_CONTRACT_ROLL"
        elif any(item is None for item in path[1:-1]):
            reason = "CROSSES_INTERNAL_GAP"
        else:
            stages["structurally_valid_paths"] += 1
            start_reference = _actual_reference(seconds, trade_times_ns, t_ns)
            future_reference = _actual_reference(seconds, trade_times_ns, future_ns)
            if start_reference is not None and future_reference is not None:
                stages["reference_pairs_within_10s"] += 1
            if start_reference is None:
                reason = "NO_START_REFERENCE_WITHIN_10S"
            elif start_reference[2] != candidate_id:
                reason = "START_REFERENCE_CONTRACT_MISMATCH"
            elif future_reference is None:
                reason = "NO_FUTURE_REFERENCE_WITHIN_10S"
            elif future_reference[2] != candidate_id:
                reason = "FUTURE_REFERENCE_CONTRACT_MISMATCH"
            elif future_reference[1] == start_reference[1]:
                reason = "EQUAL_REFERENCE_PRICE"
                stages["equal_price_count"] += 1
            else:
                reason = "ELIGIBLE"
                stages["binary_eligible_rows"] += 1
        _check_reference(row, "start", start_reference)
        _check_reference(row, "future", future_reference)
        if row.eligibility_reason != reason or bool(row.target_eligible) != (reason == "ELIGIBLE"):
            raise ValueError(f"{row.decision_timestamp}: reason or eligibility differs from sources")
        if reason == "ELIGIBLE":
            if start_reference[2] != candidate_id or future_reference[2] != candidate_id:
                mismatched_eligible_references += 1
            change = future_reference[1] - start_reference[1]
            label = 1 if change > 0 else 0
            if pd.isna(row.target_up_15m) or int(row.target_up_15m) != label:
                raise ValueError(f"{row.decision_timestamp}: wrong binary label")
            stages["up_count" if label else "down_count"] += 1
        elif reason == "EQUAL_REFERENCE_PRICE":
            change = 0.0
            if not pd.isna(row.target_up_15m):
                raise ValueError(f"{row.decision_timestamp}: equal price received a label")
        else:
            change = None
            if not pd.isna(row.target_up_15m):
                raise ValueError(f"{row.decision_timestamp}: ineligible row received a label")
        if change is None:
            if not pd.isna(row.forward_price_change) or not pd.isna(row.forward_log_return):
                raise ValueError(f"{row.decision_timestamp}: unavailable forward outcome stored")
        else:
            expected_return = np.log(future_reference[1] / start_reference[1])
            if float(row.forward_price_change) != change or not np.isclose(
                float(row.forward_log_return), expected_return, atol=1e-12, rtol=0
            ):
                raise ValueError(f"{row.decision_timestamp}: forward outcome differs from references")
        counts[reason] += 1
    return {
        "session_date": session_date,
        "candidate_rows": len(targets),
        **{field: int(stages[field]) for field in [
            "structurally_valid_paths", "reference_pairs_within_10s", "equal_price_count",
            "binary_eligible_rows", "up_count", "down_count",
        ]},
        "eligibility_reason_counts": dict(sorted(counts.items())),
        "eligible_reference_contract_mismatches": mismatched_eligible_references,
        "first_decision_timestamp": str(targets["decision_timestamp"].iloc[0]),
        "last_decision_timestamp": str(targets["decision_timestamp"].iloc[-1]),
    }


def audit_full_history_target_v2(minute_dir=MINUTE_DIR, second_dir=SECOND_DIR, target_dir=TARGET_DIR) -> dict:
    """Verify all source/target session sets, per-session audits, and run totals."""
    target_dir = Path(target_dir)
    manifest = json.loads((target_dir / "run_manifest.json").read_text())
    if manifest["schema_version"] != TARGET_SCHEMA_VERSION:
        raise ValueError("Target V2 manifest schema version differs from code")
    minute_manifest = json.loads((Path(minute_dir) / "run_manifest.json").read_text())
    second_manifest = json.loads((Path(second_dir) / "run_manifest.json").read_text())
    if manifest["source_one_minute"]["schema_version"] != minute_manifest["schema_version"]:
        raise ValueError("Target V2 minute lineage schema differs")
    if manifest["source_one_second_v2"]["schema_version"] != second_manifest["schema_version"]:
        raise ValueError("Target V2 second lineage schema differs")
    if manifest["source_one_minute"]["timestamp_policy"] != minute_manifest["timestamp_policy"]:
        raise ValueError("Target V2 minute timestamp policy differs")
    if manifest["source_one_second_v2"]["timestamp_policy"] != second_manifest["timestamp_policy"]:
        raise ValueError("Target V2 second timestamp policy differs")
    if manifest.get("predictive_features_included") is not False or manifest.get(
        "target_outcomes_must_not_enter_feature_matrix"
    ) is not True:
        raise ValueError("Target V2 manifest lacks explicit outcome/feature separation")
    if not {"start_reference_instrument_id", "future_reference_instrument_id"}.issubset(
        manifest.get("target_outcome_fields", [])
    ):
        raise ValueError("Target V2 manifest omits reference IDs from excluded outcome fields")
    source_dates = {p.stem.removeprefix("MES_1m_") for p in Path(minute_dir).glob("MES_1m_*.parquet")}
    second_dates = {p.stem.removeprefix("MES_1s_") for p in Path(second_dir).glob("MES_1s_*.parquet")}
    target_dates = {p.stem.removeprefix("MES_target_v2_") for p in target_dir.glob("MES_target_v2_*.parquet")}
    if not source_dates or source_dates != second_dates or source_dates != target_dates:
        raise ValueError("Target V2, minute, and V2-second session sets differ")
    expected_sessions = {record["session_date"]: record for record in manifest["session_audit"]}
    if len(expected_sessions) != len(source_dates):
        raise ValueError("Target V2 manifest session audit set differs")
    totals = Counter()
    reasons = Counter()
    previous_last = None
    for date in sorted(target_dates):
        observed = audit_target_v2_session(date, minute_dir, second_dir, target_dir)
        if previous_last is not None and pd.Timestamp(observed["first_decision_timestamp"]) <= previous_last:
            raise ValueError(f"{date}: target decisions not globally ordered and unique")
        previous_last = pd.Timestamp(observed["last_decision_timestamp"])
        for key in [
            "candidate_rows", "structurally_valid_paths", "reference_pairs_within_10s",
            "equal_price_count", "binary_eligible_rows", "up_count", "down_count",
            "eligibility_reason_counts",
        ]:
            if observed[key] != expected_sessions[date][key]:
                raise ValueError(f"{date}: manifest audit {key} differs from independent result")
        totals.update({key: observed[key] for key in [
            "candidate_rows", "structurally_valid_paths", "reference_pairs_within_10s",
            "equal_price_count", "binary_eligible_rows", "up_count", "down_count",
            "eligible_reference_contract_mismatches",
        ]})
        reasons.update(observed["eligibility_reason_counts"])
    mapping = {
        "candidate_rows": "total_candidate_rows",
        "structurally_valid_paths": "structurally_valid_paths",
        "reference_pairs_within_10s": "reference_pairs_within_10s",
        "equal_price_count": "equal_price_count",
        "binary_eligible_rows": "binary_eligible_rows",
        "up_count": "up_count",
        "down_count": "down_count",
    }
    for actual_key, manifest_key in mapping.items():
        if totals[actual_key] != manifest[manifest_key]:
            raise ValueError(f"Target V2 manifest {manifest_key} differs from independent total")
    if dict(sorted(reasons.items())) != manifest["eligibility_reason_counts"]:
        raise ValueError("Target V2 manifest reason counts differ from independent total")
    if totals["eligible_reference_contract_mismatches"]:
        raise ValueError("Eligible Target V2 rows cross instrument_id")
    return {
        "passed": True,
        "session_files": len(target_dates),
        **dict(totals),
        "eligibility_reason_counts": dict(sorted(reasons.items())),
    }


def audit_target_v1_v2_changes(
    v1_dir=PROJECT_ROOT / "data/processed/targets/target_v1",
    v2_dir=TARGET_DIR,
) -> dict:
    """Independently enumerate changed V1 fields; reject unrelated V2 drift."""
    v1_dir, v2_dir = Path(v1_dir), Path(v2_dir)
    old_files = {p.stem.removeprefix("MES_target_v1_"): p for p in v1_dir.glob("MES_target_v1_*.parquet")}
    new_files = {p.stem.removeprefix("MES_target_v2_"): p for p in v2_dir.glob("MES_target_v2_*.parquet")}
    if not old_files or old_files.keys() != new_files.keys():
        raise ValueError("Target V1/V2 session file sets differ")
    mutable_fields = [
        "forward_price_change", "forward_log_return", "target_up_15m",
        "target_eligible", "eligibility_reason",
    ]
    changed = []
    for date in sorted(old_files):
        old = pd.read_parquet(old_files[date])
        new = pd.read_parquet(new_files[date])
        if len(old) != len(new):
            raise ValueError(f"{date}: V1/V2 candidate count differs")
        for field in old.columns:
            if field not in mutable_fields and not old[field].equals(new[field]):
                raise ValueError(f"{date}: unrelated field {field} changed")
        changed_mask = pd.Series(False, index=old.index)
        for field in mutable_fields:
            equal = old[field].eq(new[field]).fillna(False) | (old[field].isna() & new[field].isna())
            changed_mask |= ~equal
        for index in old.index[changed_mask]:
            old_row, new_row = old.iloc[index], new.iloc[index]
            changed.append({
                "session_date": date,
                "decision_timestamp": str(old_row["decision_timestamp"]),
                "old_reason": str(old_row["eligibility_reason"]),
                "new_reason": str(new_row["eligibility_reason"]),
                "old_eligible": bool(old_row["target_eligible"]),
                "new_eligible": bool(new_row["target_eligible"]),
                "old_label": None if pd.isna(old_row["target_up_15m"]) else int(old_row["target_up_15m"]),
                "new_label": None if pd.isna(new_row["target_up_15m"]) else int(new_row["target_up_15m"]),
            })
    manifest = json.loads((v2_dir / "run_manifest.json").read_text())
    recorded = manifest["target_v1_comparison"]["changed_rows"]
    if len(changed) != manifest["target_v1_comparison"]["changed_row_count"]:
        raise ValueError("Target V1/V2 changed-row count differs from manifest")
    for actual, stored in zip(changed, recorded):
        if any(actual[field] != stored[field] for field in actual):
            raise ValueError("Target V1/V2 changed-row detail differs from manifest")
    return {"passed": True, "session_files": len(old_files), "changed_rows": changed}
