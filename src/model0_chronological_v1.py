"""Chronological Model 0 development using only pre-holdout Target V2 labels.

The final 60-session holdout is checked for identity by reading keys alone;
its labels, references, and predictive performance are never loaded here.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "MODEL0_FEATURE_CONTRACT_V1.json"
FEATURE_DIR = ROOT / "data/processed/features/model0_v1"
TARGET_DIR = ROOT / "data/processed/targets/target_v2"
OUTPUT_DIR = ROOT / "data/processed/development/model0_chronological_v1"
HOLDOUT_START = "2026-06-19"
HOLDOUT_END = "2026-09-10"
FINAL_PARTIAL = "2026-09-11"
FOLDS = (
    ("2026-03-27", "2026-04-23"),
    ("2026-04-24", "2026-05-21"),
    ("2026-05-22", "2026-06-18"),
)
IDENTITY = ["decision_timestamp", "session_date", "instrument_id", "contract"]
TARGET_FIELDS = IDENTITY + [
    "target_eligible", "target_up_15m", "future_boundary_timestamp",
    "future_reference_timestamp",
]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _paths(directory: Path, prefix: str) -> dict[str, Path]:
    return {
        path.stem.removeprefix(prefix): path
        for path in directory.glob(f"{prefix}*.parquet")
    }


def _verify_identity(features: pd.DataFrame, target: pd.DataFrame, day: str) -> None:
    if len(features) != len(target):
        raise ValueError(f"{day}: feature/target candidate counts differ")
    for frame in (features, target):
        if frame["decision_timestamp"].duplicated().any():
            raise ValueError(f"{day}: duplicate decision timestamps")
        if not frame["decision_timestamp"].is_monotonic_increasing:
            raise ValueError(f"{day}: decision timestamps are not ordered")
    for column in IDENTITY:
        if not features[column].equals(target[column]):
            raise ValueError(f"{day}: {column} differs between feature and target")


def load_development_sample(
    feature_dir: Path = FEATURE_DIR, target_dir: Path = TARGET_DIR,
) -> tuple[pd.DataFrame, dict, list[str]]:
    """Audit all identities, but load labels/references only before holdout."""
    feature_dir, target_dir = Path(feature_dir), Path(target_dir)
    contract = _read_json(CONTRACT)
    feature_manifest = _read_json(feature_dir / "run_manifest.json")
    target_manifest = _read_json(target_dir / "run_manifest.json")
    predictors = [item["name"] for item in contract["columns"] if item["role"] == "predictor"]
    if len(predictors) != 36 or feature_manifest["predictor_columns"] != predictors:
        raise ValueError("Predictor order differs from frozen Model 0 contract")
    feature_paths = _paths(feature_dir, "MES_model0_v1_")
    target_paths = _paths(target_dir, "MES_target_v2_")
    if feature_paths.keys() != target_paths.keys() or len(feature_paths) != 242:
        raise ValueError("Feature/Target V2 session sets differ")
    target_audits = {entry["session_date"]: entry for entry in target_manifest["session_audit"]}
    if target_audits.keys() != feature_paths.keys():
        raise ValueError("Target V2 manifest session set differs from files")
    pieces = []
    candidate_total = feature_total = holdout_candidates = holdout_eligible = 0
    final_partial_candidates = final_partial_eligible = 0
    development_candidates = development_eligible = 0
    previous_last = None
    for day in sorted(feature_paths):
        feature_keys = pd.read_parquet(feature_paths[day], columns=IDENTITY)
        target_keys = pd.read_parquet(target_paths[day], columns=IDENTITY)
        _verify_identity(feature_keys, target_keys, day)
        if previous_last is not None and feature_keys["decision_timestamp"].iloc[0] <= previous_last:
            raise ValueError("Feature/target decisions duplicate across sessions")
        previous_last = feature_keys["decision_timestamp"].iloc[-1]
        count = len(feature_keys)
        if count != target_audits[day]["candidate_rows"]:
            raise ValueError(f"{day}: Target V2 candidate manifest differs")
        feature_total += count
        candidate_total += count
        if HOLDOUT_START <= day <= HOLDOUT_END:
            # Only frozen aggregate eligibility counts are read here. No
            # holdout label or reference column is opened.
            holdout_candidates += count
            holdout_eligible += target_audits[day]["binary_eligible_rows"]
            continue
        if day == FINAL_PARTIAL:
            final_partial_candidates += count
            final_partial_eligible += target_audits[day]["binary_eligible_rows"]
            continue
        if day >= HOLDOUT_START:
            raise ValueError(f"Unexpected session after development: {day}")
        feature = pd.read_parquet(feature_paths[day])
        target = pd.read_parquet(target_paths[day], columns=TARGET_FIELDS)
        _verify_identity(feature[IDENTITY], target[IDENTITY], day)
        joined = feature.merge(target, on=IDENTITY, how="inner", validate="one_to_one", sort=False)
        if len(joined) != count or not joined["decision_timestamp"].equals(feature["decision_timestamp"]):
            raise ValueError(f"{day}: join lost/reordered candidate rows")
        valid = joined["target_eligible"] & joined["target_up_15m"].notna()
        if int(valid.sum()) != target_audits[day]["binary_eligible_rows"]:
            raise ValueError(f"{day}: eligible join differs from Target V2 manifest")
        if not joined.loc[valid, "target_up_15m"].isin([0, 1]).all():
            raise ValueError(f"{day}: eligible label is not binary")
        eligible = joined.loc[valid, [
            *[item["name"] for item in contract["columns"]],
            "target_up_15m", "future_boundary_timestamp", "future_reference_timestamp",
        ]].copy()
        eligible["target_up_15m"] = eligible["target_up_15m"].astype("int8")
        if eligible["future_reference_timestamp"].isna().any():
            raise ValueError(f"{day}: eligible target lacks actual future reference")
        delay = eligible["future_reference_timestamp"] - eligible["future_boundary_timestamp"]
        if not (delay.ge(pd.Timedelta(0)) & delay.lt(pd.Timedelta(seconds=10))).all():
            raise ValueError(f"{day}: eligible future reference violates strict <10s")
        if not (
            eligible["future_boundary_timestamp"]
            == eligible["decision_timestamp"] + pd.Timedelta(minutes=15)
        ).all():
            raise ValueError(f"{day}: future boundary is not T+15m")
        pieces.append(eligible)
        development_candidates += count
        development_eligible += len(eligible)
    sample = pd.concat(pieces, ignore_index=True)
    if sample["decision_timestamp"].duplicated().any() or not sample["decision_timestamp"].is_monotonic_increasing:
        raise ValueError("Development sample decisions are not unique and chronological")
    counts = {
        "feature_rows": feature_total,
        "target_v2_candidate_rows": candidate_total,
        "target_v2_binary_eligible_rows": target_manifest["binary_eligible_rows"],
        "development_candidate_rows": development_candidates,
        "development_modeling_rows": development_eligible,
        "holdout_candidate_rows": holdout_candidates,
        "holdout_eligible_rows_from_manifest_only": holdout_eligible,
        "final_partial_candidate_rows": final_partial_candidates,
        "final_partial_eligible_rows_from_manifest_only": final_partial_eligible,
    }
    if (
        feature_total != contract["expected_rows"]
        or candidate_total != target_manifest["total_candidate_rows"]
        or development_eligible + holdout_eligible + final_partial_eligible
        != target_manifest["binary_eligible_rows"]
        or development_eligible != len(sample)
    ):
        raise ValueError("Global feature/target/modeling row reconciliation failed")
    return sample, counts, predictors


def fold_indices(sample: pd.DataFrame, start: str, end: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    dates = sample["session_date"].astype(str)
    raw_train = np.flatnonzero(dates.lt(start).to_numpy())
    validation = np.flatnonzero(dates.between(start, end).to_numpy())
    if not len(raw_train) or not len(validation):
        raise ValueError(f"Empty chronological fold: {start} through {end}")
    validation_days = dates.iloc[validation].unique()
    if len(validation_days) != 20:
        raise ValueError(f"Expected 20 validation sessions, got {len(validation_days)}")
    boundary = sample["decision_timestamp"].iloc[validation[0]]
    references = sample["future_reference_timestamp"].iloc[raw_train]
    purge_mask = references.ge(boundary).to_numpy()
    purged = raw_train[purge_mask]
    train = raw_train[~purge_mask]
    if not len(train) or sample["future_reference_timestamp"].iloc[train].max() >= boundary:
        raise ValueError("Retained training target reaches validation")
    if sample["decision_timestamp"].iloc[train].max() >= boundary:
        raise ValueError("Training decision reaches validation")
    result = {
        "validation_start_session": start,
        "validation_end_session": end,
        "validation_sessions": len(validation_days),
        "raw_training_rows": len(raw_train),
        "purged_rows": len(purged),
        "final_training_rows": len(train),
        "validation_rows": len(validation),
        "latest_retained_training_decision": str(sample["decision_timestamp"].iloc[train[-1]]),
        "latest_retained_training_future_reference": str(sample["future_reference_timestamp"].iloc[train].max()),
        "earliest_validation_decision": str(boundary),
        "earliest_validation_session": str(validation_days[0]),
        "latest_validation_session": str(validation_days[-1]),
    }
    return train, validation, purged, result


def momentum_probabilities(train_return: np.ndarray, train_label: np.ndarray,
                           validation_return: np.ndarray) -> tuple[np.ndarray, float, dict]:
    """Frozen three-state Baseline 1, fitted on purged training labels only."""
    base = (int(train_label.sum()) + 1) / (len(train_label) + 2)
    probabilities = {}
    for name, mask in (
        ("positive", np.isfinite(train_return) & (train_return > 0)),
        ("negative", np.isfinite(train_return) & (train_return < 0)),
    ):
        n = int(mask.sum())
        probabilities[name] = {
            "training_rows": n,
            "training_up": int(train_label[mask].sum()),
            "p_up": (int(train_label[mask].sum()) + 1) / (n + 2),
        }
    probabilities["neutral_or_unavailable"] = {
        "training_rows": int((~np.isfinite(train_return) | (train_return == 0)).sum()),
        "p_up": base,
    }
    output = np.full(len(validation_return), base, dtype=np.float64)
    output[np.isfinite(validation_return) & (validation_return > 0)] = probabilities["positive"]["p_up"]
    output[np.isfinite(validation_return) & (validation_return < 0)] = probabilities["negative"]["p_up"]
    return output, base, probabilities


def _metric(y: np.ndarray, p: np.ndarray) -> dict:
    if not (np.isfinite(p).all() and ((0 < p) & (p < 1)).all()):
        raise ValueError("Predicted probabilities must be finite and strictly between 0 and 1")
    return {
        "rows": int(len(y)),
        "up": int(y.sum()),
        "down": int(len(y) - y.sum()),
        "up_rate": float(y.mean()),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, p)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "directional_accuracy_at_0_5": float(np.mean((p >= 0.5) == y)),
    }


def _calibration(y: np.ndarray, p: np.ndarray) -> list[dict]:
    # Fixed ten equal-width bins are descriptive; they fit no parameter.
    bins = np.minimum((p * 10).astype(int), 9)
    return [
        {
            "lower_inclusive": index / 10,
            "upper_exclusive_except_final": (index + 1) / 10,
            "rows": int(mask.sum()),
            "mean_predicted_up": float(p[mask].mean()) if mask.any() else None,
            "observed_up_rate": float(y[mask].mean()) if mask.any() else None,
        }
        for index in range(10)
        for mask in [bins == index]
    ]


def _stability(predictions: pd.DataFrame) -> dict:
    groups = {}
    months = predictions["session_date"].astype(str).str.slice(0, 7)
    for month in sorted(months.unique()):
        groups[f"month:{month}"] = predictions.loc[months.eq(month)]
    for name, column in (
        ("Asia", "asia_flag"), ("Europe", "europe_flag"),
        ("U.S.", "us_flag"), ("U.S. RTH", "us_rth_flag"),
    ):
        groups[f"regime:{name}"] = predictions.loc[predictions[column].eq(1)]
    return {
        name: {model: _metric(frame["target_up_15m"].to_numpy(dtype=np.int8),
                              frame[column].to_numpy(dtype=np.float64))
               for model, column in (
                   ("baseline_0", "p_base"), ("baseline_1", "p_momentum"),
                   ("logistic", "p_logistic"),
               )}
        for name, frame in groups.items() if len(frame)
    }


def run_development(output_dir: Path = OUTPUT_DIR) -> dict:
    """Fit only chronological development folds; write no holdout prediction."""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing populated development output: {output_dir}")
    sample, counts, predictors = load_development_sample()
    config = {
        "imputation": "training-fold median, all 36 columns, no missing indicators",
        "scaling": "StandardScaler fitted on each purged training fold after imputation",
        "model": "LogisticRegression",
        "penalty": "L2",
        "l1_ratio": 0.0,
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 1000,
        "class_weight": None,
        "feature_selection": False,
        "probability_threshold": 0.5,
        "calibration": "ten fixed equal-width validation-probability bins",
        "target_v2_joined_into_feature_builder": False,
        "holdout_labels_loaded": False,
        "holdout_predictions_generated": False,
    }
    metrics, coefficients, fold_audits, predictions, null_reports = [], [], [], [], []
    for number, (start, end) in enumerate(FOLDS, 1):
        train, validation, purged, audit = fold_indices(sample, start, end)
        x_train = sample.loc[train, predictors].to_numpy(dtype=np.float64, na_value=np.nan)
        x_validation = sample.loc[validation, predictors].to_numpy(dtype=np.float64, na_value=np.nan)
        y_train = sample.loc[train, "target_up_15m"].to_numpy(dtype=np.int8)
        y_validation = sample.loc[validation, "target_up_15m"].to_numpy(dtype=np.int8)
        if len(np.unique(y_train)) != 2 or len(np.unique(y_validation)) != 2:
            raise ValueError(f"Fold {number} lacks both target classes")
        p_momentum, p_base, momentum_state = momentum_probabilities(
            x_train[:, predictors.index("return_15m")], y_train,
            x_validation[:, predictors.index("return_15m")],
        )
        pipeline = Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=False,
                                     keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=1.0, l1_ratio=0.0,
                                         solver="lbfgs", max_iter=1000)),
        ])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            pipeline.fit(x_train, y_train)
        convergence = [str(item.message) for item in caught
                       if issubclass(item.category, ConvergenceWarning)]
        if convergence:
            raise RuntimeError(f"Fold {number} logistic regression did not converge: {convergence}")
        p_logistic = pipeline.predict_proba(x_validation)[:, list(pipeline.classes_).index(1)]
        p_base_array = np.full(len(validation), p_base, dtype=np.float64)
        fold_metric = {
            "fold": number,
            "training_up": int(y_train.sum()),
            "training_down": int(len(y_train) - y_train.sum()),
            "training_up_rate": float(y_train.mean()),
            "baseline_0": _metric(y_validation, p_base_array),
            "baseline_1": _metric(y_validation, p_momentum),
            "logistic": _metric(y_validation, p_logistic),
            "calibration": {
                "baseline_0": _calibration(y_validation, p_base_array),
                "baseline_1": _calibration(y_validation, p_momentum),
                "logistic": _calibration(y_validation, p_logistic),
            },
        }
        for model in ("baseline_1", "logistic"):
            fold_metric[model]["log_loss_improvement_vs_base"] = (
                fold_metric["baseline_0"]["log_loss"] - fold_metric[model]["log_loss"]
            )
        metrics.append(fold_metric)
        audit["purged_decisions"] = [str(x) for x in sample.loc[purged, "decision_timestamp"]]
        audit["training_target_max_strictly_before_validation"] = True
        audit["momentum_training_probabilities"] = momentum_state
        fold_audits.append(audit)
        missing = np.isnan(x_train).sum(axis=0)
        constant = pipeline.named_steps["scale"].var_ == 0
        null_reports.append({
            "fold": number,
            "training_rows": len(train),
            "training_null_counts": dict(zip(predictors, map(int, missing))),
            "all_null_training_predictors": [name for name, count in zip(predictors, missing)
                                              if count == len(train)],
            "constant_after_imputation": [name for name, flag in zip(predictors, constant)
                                          if flag],
            "imputer_training_medians": dict(zip(
                predictors, [None if math.isnan(value) else float(value)
                             for value in pipeline.named_steps["impute"].statistics_],
            )),
        })
        coefficients.append({
            "fold": number,
            "intercept_standardized": float(pipeline.named_steps["model"].intercept_[0]),
            "standardized_coefficients": dict(zip(
                predictors, map(float, pipeline.named_steps["model"].coef_[0]),
            )),
            "iterations": int(pipeline.named_steps["model"].n_iter_[0]),
            "convergence_warnings": convergence,
        })
        frame = sample.loc[validation, [*IDENTITY,
            "asia_flag", "europe_flag", "us_flag", "us_rth_flag", "target_up_15m"]].copy()
        frame.insert(0, "fold", number)
        frame["p_base"] = p_base_array
        frame["p_momentum"] = p_momentum
        frame["p_logistic"] = p_logistic
        predictions.append(frame)
    development_predictions = pd.concat(predictions, ignore_index=True)
    stability = _stability(development_predictions)
    manifest = {
        "version": "mes_model0_chronological_development_v1",
        "feature_schema_version": _read_json(FEATURE_DIR / "run_manifest.json")["schema_version"],
        "target_schema_version": _read_json(TARGET_DIR / "run_manifest.json")["schema_version"],
        "predictor_columns": predictors,
        "identity_keys": IDENTITY,
        "counts": counts,
        "holdout_session_start": HOLDOUT_START,
        "holdout_session_end": HOLDOUT_END,
        "holdout_outcomes_loaded": False,
        "final_source_partial_session_excluded": FINAL_PARTIAL,
        "folds": fold_audits,
        "configuration": config,
        "development_predictions_rows": len(development_predictions),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(output_dir / "development_eligible_sample.parquet", index=False)
    development_predictions.to_parquet(output_dir / "validation_predictions.parquet", index=False)
    for name, value in (
        ("run_manifest.json", manifest), ("fold_metrics.json", metrics),
        ("logistic_coefficients.json", coefficients),
        ("training_nulls.json", null_reports), ("development_stability.json", stability),
    ):
        (output_dir / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    return manifest


if __name__ == "__main__":
    result = run_development()
    print(json.dumps({"counts": result["counts"], "folds": result["folds"]}, indent=2))
