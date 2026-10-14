"""Read-only split, probability, and leakage audit for Model 0 development.

Only saved pre-holdout development sample and validation predictions are read.
This auditor never opens holdout target labels or produces holdout predictions.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "MODEL0_FEATURE_CONTRACT_V1.json"
OUTPUT_DIR = ROOT / "data/processed/development/model0_chronological_v1"
IDENTITY = ["decision_timestamp", "session_date", "instrument_id", "contract"]
FOLDS = (
    ("2026-03-27", "2026-04-23"),
    ("2026-04-24", "2026-05-21"),
    ("2026-05-22", "2026-06-18"),
)


def audit_development(output_dir: Path = OUTPUT_DIR) -> dict:
    output_dir = Path(output_dir)
    contract = json.loads(CONTRACT.read_text())
    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    metrics = json.loads((output_dir / "fold_metrics.json").read_text())
    nulls = json.loads((output_dir / "training_nulls.json").read_text())
    coefficients = json.loads((output_dir / "logistic_coefficients.json").read_text())
    sample = pd.read_parquet(output_dir / "development_eligible_sample.parquet")
    predictions = pd.read_parquet(output_dir / "validation_predictions.parquet")
    predictors = [item["name"] for item in contract["columns"] if item["role"] == "predictor"]
    sample_columns = [item["name"] for item in contract["columns"]] + [
        "target_up_15m", "future_boundary_timestamp", "future_reference_timestamp",
    ]
    prediction_columns = ["fold", *IDENTITY, "asia_flag", "europe_flag", "us_flag",
                          "us_rth_flag", "target_up_15m", "p_base", "p_momentum", "p_logistic"]
    if (
        len(predictors) != 36 or manifest["predictor_columns"] != predictors
        or list(sample.columns) != sample_columns
        or list(predictions.columns) != prediction_columns
        or len(sample) != manifest["counts"]["development_modeling_rows"]
        or len(predictions) != manifest["development_predictions_rows"]
        or manifest["holdout_outcomes_loaded"] is not False
        or manifest["configuration"]["holdout_predictions_generated"] is not False
        or any("target" in name or "reference" in name or "eligible" in name
               for name in predictors)
    ):
        raise ValueError("Development artifact schema or firewall differs from contract")
    if (
        sample["session_date"].astype(str).ge("2026-06-19").any()
        or predictions["session_date"].astype(str).ge("2026-06-19").any()
        or sample["decision_timestamp"].duplicated().any()
        or predictions["decision_timestamp"].duplicated().any()
        or not sample["decision_timestamp"].is_monotonic_increasing
        or not predictions["decision_timestamp"].is_monotonic_increasing
        or not sample["target_up_15m"].isin([0, 1]).all()
    ):
        raise ValueError("Holdout, duplicate, ordering, or label error in saved sample")
    dates = sample["session_date"].astype(str)
    checked_rows = checked_probabilities = checked_medians = 0
    for number, ((start, end), fold, saved_metric, saved_null, saved_coefficient) in enumerate(
        zip(FOLDS, manifest["folds"], metrics, nulls, coefficients, strict=True), 1
    ):
        raw_training = sample.loc[dates.lt(start)]
        validation = sample.loc[dates.between(start, end)]
        saved_validation = predictions.loc[predictions["fold"].eq(number)]
        if (
            len(validation) != len(saved_validation)
            or len(validation["session_date"].unique()) != 20
            or not validation[IDENTITY].reset_index(drop=True).equals(
                saved_validation[IDENTITY].reset_index(drop=True)
            )
            or not validation["target_up_15m"].reset_index(drop=True).equals(
                saved_validation["target_up_15m"].reset_index(drop=True)
            )
        ):
            raise ValueError(f"Fold {number}: saved validation identities or labels differ")
        boundary = validation["decision_timestamp"].min()
        purged = raw_training.loc[raw_training["future_reference_timestamp"].ge(boundary)]
        training = raw_training.loc[raw_training["future_reference_timestamp"].lt(boundary)]
        if (
            len(raw_training) != fold["raw_training_rows"]
            or len(purged) != fold["purged_rows"]
            or len(training) != fold["final_training_rows"]
            or len(validation) != fold["validation_rows"]
            or training["future_reference_timestamp"].max() >= boundary
            or training["decision_timestamp"].max() >= boundary
            or str(training["future_reference_timestamp"].max())
            != fold["latest_retained_training_future_reference"]
            or str(boundary) != fold["earliest_validation_decision"]
        ):
            raise ValueError(f"Fold {number}: actual-reference purge audit differs")
        y_training = training["target_up_15m"].to_numpy(dtype=np.int8)
        y_validation = validation["target_up_15m"].to_numpy(dtype=np.int8)
        p_base = (int(y_training.sum()) + 1) / (len(y_training) + 2)
        state_training = training["return_15m"].to_numpy(dtype=np.float64, na_value=np.nan)
        state_validation = validation["return_15m"].to_numpy(dtype=np.float64, na_value=np.nan)
        p_expected = np.full(len(validation), p_base)
        for condition in ("positive", "negative"):
            train_mask = np.isfinite(state_training) & (
                state_training > 0 if condition == "positive" else state_training < 0
            )
            validation_mask = np.isfinite(state_validation) & (
                state_validation > 0 if condition == "positive" else state_validation < 0
            )
            p_state = (int(y_training[train_mask].sum()) + 1) / (int(train_mask.sum()) + 2)
            if not math.isclose(p_state, fold["momentum_training_probabilities"][condition]["p_up"],
                                abs_tol=1e-14):
                raise ValueError(f"Fold {number}: momentum training probability differs")
            p_expected[validation_mask] = p_state
        if not (
            np.allclose(saved_validation["p_base"], p_base, rtol=0, atol=1e-14)
            and np.allclose(saved_validation["p_momentum"], p_expected, rtol=0, atol=1e-14)
        ):
            raise ValueError(f"Fold {number}: baseline probabilities use other information")
        checked_probabilities += 2 * len(validation)
        for name in predictors:
            observed_nulls = int(training[name].isna().sum())
            if observed_nulls != saved_null["training_null_counts"][name]:
                raise ValueError(f"Fold {number}: training missingness differs for {name}")
            median = training[name].median(skipna=True)
            saved_median = saved_null["imputer_training_medians"][name]
            if pd.isna(median):
                if saved_median is not None:
                    raise ValueError(f"Fold {number}: all-null median differs for {name}")
            elif not math.isclose(float(median), float(saved_median), rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError(f"Fold {number}: train-only imputation median differs for {name}")
            checked_medians += 1
        if saved_coefficient["convergence_warnings"] or not all(
            math.isfinite(value) for value in saved_coefficient["standardized_coefficients"].values()
        ):
            raise ValueError(f"Fold {number}: logistic coefficient/convergence problem")
        for model, column in (
            ("baseline_0", "p_base"), ("baseline_1", "p_momentum"),
            ("logistic", "p_logistic"),
        ):
            probability = saved_validation[column].to_numpy(dtype=float)
            if not (np.isfinite(probability).all() and ((probability > 0) & (probability < 1)).all()):
                raise ValueError(f"Fold {number}: {model} has invalid probability")
            expected = {
                "log_loss": float(log_loss(y_validation, probability, labels=[0, 1])),
                "brier_score": float(brier_score_loss(y_validation, probability)),
                "roc_auc": float(roc_auc_score(y_validation, probability)),
                "directional_accuracy_at_0_5": float(np.mean((probability >= 0.5) == y_validation)),
            }
            for metric, value in expected.items():
                if not math.isclose(value, saved_metric[model][metric], abs_tol=1e-12):
                    raise ValueError(f"Fold {number}: saved {model} {metric} differs")
        checked_rows += len(validation)
    return {
        "passed": True,
        "development_sample_rows": len(sample),
        "validation_rows": checked_rows,
        "baseline_probabilities_checked": checked_probabilities,
        "training_medians_checked": checked_medians,
        "holdout_labels_loaded": False,
        "holdout_predictions_present": False,
        "split_and_metric_reconciliation": True,
    }


if __name__ == "__main__":
    print(json.dumps(audit_development(), indent=2))
