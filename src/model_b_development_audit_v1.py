"""Independent read-only Model B artifact audit; development observations only.

Replays saved fitted pipelines on validation rows without calling fit. Pipeline
pickles are local trusted artifacts, verified by SHA-256 before loading.
"""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.verify_artifact_provenance import sha256_file, verify_artifacts


ROOT = Path(__file__).resolve().parents[1]
A_DIR = ROOT / "data/processed/development/model0_chronological_v1"
B_DIR = ROOT / "data/processed/development/model_b_chronological_v1"
A_PROVENANCE = ROOT / "MODEL0_DEVELOPMENT_PROVENANCE_V1.json"
B_PROVENANCE = ROOT / "MODEL_B_DEVELOPMENT_PROVENANCE_V1.json"
FOLDS = (("2026-03-27", "2026-04-23"),
         ("2026-04-24", "2026-05-21"),
         ("2026-05-22", "2026-06-18"))
EXPECTED_TRAIN = (153824, 179491, 205742)
EXPECTED_VALID = (25667, 26251, 25842)
IDENTITY = ["decision_timestamp", "session_date", "instrument_id", "contract"]
PROBABILITIES = {"baseline_0": "p_base", "baseline_1": "p_momentum",
                 "logistic": "p_logistic", "model_b": "p_model_b"}
EXPLICIT_PARAMS = {
    "loss": "log_loss", "learning_rate": .05, "max_iter": 200,
    "max_leaf_nodes": 7, "max_depth": None, "min_samples_leaf": 500,
    "l2_regularization": 1.0, "max_bins": 255, "early_stopping": False,
    "class_weight": None, "random_state": 42, "max_features": 1.0,
    "categorical_features": None,
}


def _json(path: Path):
    return json.loads(path.read_text())


def _close(a, b, tol=1e-11):
    if not math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol):
        raise ValueError(f"Numeric audit difference: {a} != {b}")


def _metric(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "rows": len(y), "up": int(y.sum()), "down": int(len(y) - y.sum()),
        "up_rate": float(y.mean()),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "directional_accuracy_at_0_5": float(((p >= .5) == y).mean()),
    }


def _check_metric(saved: dict, actual: dict) -> None:
    for key, value in actual.items():
        _close(saved[key], value)


def _calibration(y: np.ndarray, p: np.ndarray) -> list[dict]:
    bins = np.minimum(np.floor(p * 10).astype(int), 9)
    result = []
    for index in range(10):
        mask = bins == index
        result.append({
            "lower_inclusive": index / 10,
            "upper_exclusive_except_final": (index + 1) / 10,
            "rows": int(mask.sum()),
            "mean_predicted_up": float(p[mask].mean()) if mask.any() else None,
            "observed_up_rate": float(y[mask].mean()) if mask.any() else None,
        })
    return result


def audit_development(output_dir: Path = B_DIR) -> dict:
    output_dir = Path(output_dir)
    if sklearn.__version__ != "1.9.1":
        raise ValueError("Auditor requires frozen sklearn 1.9.1")
    verify_artifacts(A_PROVENANCE)
    a_record = _json(A_PROVENANCE)
    if sha256_file(A_DIR / "run_manifest.json") != a_record["run_manifest_sha256"]:
        raise ValueError("Model A manifest SHA differs")
    if B_PROVENANCE.is_file():
        verify_artifacts(B_PROVENANCE)
        b_record = _json(B_PROVENANCE)
        if sha256_file(output_dir / "run_manifest.json") != b_record["run_manifest_sha256"]:
            raise ValueError("Model B manifest SHA differs")
    contract = _json(ROOT / "MODEL0_FEATURE_CONTRACT_V1.json")
    predictors = [item["name"] for item in contract["columns"] if item["role"] == "predictor"]
    a_manifest = _json(A_DIR / "run_manifest.json")
    b_manifest = _json(output_dir / "run_manifest.json")
    metrics = _json(output_dir / "fold_metrics.json")
    distributions = _json(output_dir / "probability_distribution.json")
    preprocessing = _json(output_dir / "training_preprocessing.json")
    stability = _json(output_dir / "development_stability.json")
    a_metrics = _json(A_DIR / "fold_metrics.json")
    sample = pd.read_parquet(A_DIR / "development_eligible_sample.parquet")
    a_predictions = pd.read_parquet(A_DIR / "validation_predictions.parquet")
    predictions = pd.read_parquet(output_dir / "validation_predictions.parquet")
    expected_columns = [item["name"] for item in contract["columns"]] + [
        "target_up_15m", "future_boundary_timestamp", "future_reference_timestamp"]
    if (len(predictors) != 36 or a_manifest["predictor_columns"] != predictors
            or b_manifest["predictor_columns"] != predictors
            or b_manifest["model_parameters_explicit"] != EXPLICIT_PARAMS
            or b_manifest["pre_result_spec_commit"] != "0ef1c34af947f3a563c1e9a3652dbed086e3233d"
            or b_manifest["sklearn_version"] != sklearn.__version__
            or list(sample.columns) != expected_columns
            or list(predictions.columns) != [*a_predictions.columns, "p_model_b"]
            or len(sample) != 231584 or len(predictions) != 77760
            or len(metrics) != 3 or len(distributions) != 3 or len(preprocessing) != 3
            or len(b_manifest["folds"]) != 3
            or b_manifest["holdout_outcomes_loaded"] is not False
            or b_manifest["holdout_predictions_generated"] is not False):
        raise ValueError("Schema, sample, frozen configuration, or holdout firewall differs")
    if (not predictions[a_predictions.columns].equals(a_predictions)
            or sample["session_date"].astype(str).ge("2026-06-19").any()
            or predictions["session_date"].astype(str).ge("2026-06-19").any()
            or sample["decision_timestamp"].duplicated().any()
            or predictions["decision_timestamp"].duplicated().any()
            or not sample["decision_timestamp"].is_monotonic_increasing
            or not predictions["decision_timestamp"].is_monotonic_increasing):
        raise ValueError("Model A validation universe or development-only firewall differs")
    checked_probabilities = checked_medians = 0
    for number, (start, end) in enumerate(FOLDS, 1):
        dates = sample["session_date"].astype(str)
        validation = sample.loc[dates.between(start, end)]
        raw_training = sample.loc[dates.lt(start)]
        boundary = validation["decision_timestamp"].min()
        purged = raw_training.loc[raw_training["future_reference_timestamp"].ge(boundary)]
        training = raw_training.loc[raw_training["future_reference_timestamp"].lt(boundary)]
        fold_pred = predictions.loc[predictions["fold"].eq(number)]
        fold = b_manifest["folds"][number - 1]
        if (len(training) != EXPECTED_TRAIN[number - 1]
                or len(validation) != EXPECTED_VALID[number - 1]
                or len(purged) != 0
                or len(validation["session_date"].unique()) != 20
                or len(fold_pred) != len(validation)
                or not fold_pred[IDENTITY].reset_index(drop=True).equals(validation[IDENTITY].reset_index(drop=True))
                or not fold_pred["target_up_15m"].reset_index(drop=True).equals(validation["target_up_15m"].reset_index(drop=True))
                or str(training["future_reference_timestamp"].max()) != fold["latest_retained_training_future_reference"]
                or str(boundary) != fold["earliest_validation_decision"]
                or fold["final_training_rows"] != len(training)
                or fold["validation_rows"] != len(validation)
                or fold["purged_rows"] != len(purged)):
            raise ValueError(f"Fold {number} boundary/purge/validation differs")
        if not fold_pred.index.equals(a_predictions.loc[a_predictions["fold"].eq(number)].index):
            raise ValueError(f"Fold {number} Model A validation indices differ")
        y = validation["target_up_15m"].to_numpy(dtype=np.int8)
        x_train = training[predictors].to_numpy(dtype=np.float64, na_value=np.nan)
        x_val = validation[predictors].to_numpy(dtype=np.float64, na_value=np.nan)
        # Load only our own SHA-checked local pipeline; never call fit here.
        model_path = output_dir / f"fitted_pipeline_fold_{number}.pkl"
        with model_path.open("rb") as source:
            pipeline = pickle.load(source)
        if (not isinstance(pipeline, Pipeline)
                or list(pipeline.named_steps) != ["impute", "scale", "model"]
                or not isinstance(pipeline.named_steps["impute"], SimpleImputer)
                or not isinstance(pipeline.named_steps["scale"], StandardScaler)
                or not isinstance(pipeline.named_steps["model"], HistGradientBoostingClassifier)):
            raise ValueError(f"Fold {number} fitted pipeline differs")
        imputer = pipeline.named_steps["impute"]
        scaler = pipeline.named_steps["scale"]
        model = pipeline.named_steps["model"]
        params = model.get_params(deep=False)
        if (any(params[k] != v for k, v in EXPLICIT_PARAMS.items())
                or params["monotonic_cst"] is not None
                or params["interaction_cst"] is not None
                or params["warm_start"] is not False
                or imputer.strategy != "median" or imputer.add_indicator
                or not imputer.keep_empty_features or model.do_early_stopping_
                or model.n_iter_ != 200 or model._bin_mapper.subsample != 200000
                or model._bin_mapper.random_state != model._random_seed
                or list(pipeline.classes_) != [0, 1]
                or model.n_features_in_ != 36 or model.is_categorical_ is not None
                or np.any(model._bin_mapper.is_categorical_)
                or imputer.n_features_in_ != 36 or scaler.n_features_in_ != 36
                or int(scaler.n_samples_seen_) != len(training)
                or int(model._predictors[0][0].nodes["count"][0]) != len(training)):
            raise ValueError(f"Fold {number} fitted parameters/boosting rows differ")
        summary = preprocessing[number - 1]
        for index, name in enumerate(predictors):
            median = float(np.nanmedian(x_train[:, index]))
            _close(median, imputer.statistics_[index])
            _close(median, summary["imputer_training_medians"][name])
            if summary["training_null_counts"][name] != int(np.isnan(x_train[:, index]).sum()):
                raise ValueError(f"Fold {number} {name} missingness differs")
            checked_medians += 1
        x_imputed = np.where(np.isnan(x_train), imputer.statistics_, x_train)
        for key, actual in (("scaler_training_means", x_imputed.mean(axis=0)),
                            ("scaler_training_variances", x_imputed.var(axis=0))):
            saved = np.array([summary[key][name] for name in predictors])
            np.testing.assert_allclose(saved, actual, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(scaler.mean_, x_imputed.mean(axis=0), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(scaler.var_, x_imputed.var(axis=0), rtol=1e-10, atol=1e-10)
        expected_scale = np.where(scaler.var_ == 0, 1, np.sqrt(scaler.var_))
        np.testing.assert_allclose(scaler.scale_, expected_scale, rtol=1e-10, atol=1e-10)
        if (summary["training_rows"] != len(training)
                or summary["boosting_training_rows_first_tree_root"] != len(training)
                or summary["histogram_bin_threshold_subsample_cap"] != 200000
                or summary["bin_threshold_sampling_expected"] != (len(training) > 200000)):
            raise ValueError(f"Fold {number} training/bins summary differs")
        replay = pipeline.predict_proba(x_val)[:, 1]
        p_b = fold_pred["p_model_b"].to_numpy(dtype=np.float64)
        if not (np.isfinite(p_b).all() and ((p_b >= 0) & (p_b <= 1)).all()):
            raise ValueError(f"Fold {number} invalid Model B probabilities")
        np.testing.assert_allclose(p_b, replay, rtol=0, atol=1e-14)
        checked_probabilities += len(p_b)
        for name, column in PROBABILITIES.items():
            actual = _metric(y, fold_pred[column].to_numpy(dtype=np.float64))
            _check_metric(metrics[number - 1][name], actual)
            if name != "model_b":
                _check_metric(a_metrics[number - 1][name], actual)
        b_loss = metrics[number - 1]["model_b"]["log_loss"]
        _close(metrics[number - 1]["model_b"]["log_loss_improvement_vs_base"],
               metrics[number - 1]["baseline_0"]["log_loss"] - b_loss)
        _close(metrics[number - 1]["model_b"]["log_loss_improvement_vs_logistic"],
               metrics[number - 1]["logistic"]["log_loss"] - b_loss)
        for observed, saved in zip(_calibration(y, p_b), metrics[number - 1]["calibration"]["model_b"], strict=True):
            for key, value in observed.items():
                if value is None:
                    if saved[key] is not None:
                        raise ValueError("Empty calibration bin differs")
                else:
                    _close(value, saved[key])
        q = np.quantile(p_b, [0, .01, .05, .25, .5, .75, .95, .99, 1], method="linear")
        for key, value in zip(("minimum", "p01", "p05", "p25", "median", "p75", "p95", "p99", "maximum"), q):
            _close(distributions[number - 1][key], value)
    months = predictions["session_date"].astype(str).str[:7]
    groups = {f"month:{month}": predictions.loc[months.eq(month)] for month in sorted(months.unique())}
    for label, flag in (("Asia", "asia_flag"), ("Europe", "europe_flag"),
                        ("U.S.", "us_flag"), ("U.S. RTH", "us_rth_flag")):
        groups[f"regime:{label}"] = predictions.loc[predictions[flag].eq(1)]
    if set(stability) != {key for key, frame in groups.items() if len(frame)}:
        raise ValueError("Month/regime stability groups differ")
    for key, frame in groups.items():
        if len(frame):
            y = frame["target_up_15m"].to_numpy(dtype=np.int8)
            for name, column in PROBABILITIES.items():
                _check_metric(stability[key][name], _metric(y, frame[column].to_numpy(dtype=np.float64)))
    return {
        "passed": True, "development_sample_rows": len(sample),
        "validation_rows": len(predictions), "replayed_model_b_probabilities": checked_probabilities,
        "training_medians_checked": checked_medians,
        "training_only_scaler_checks": 3, "folds_purged": [0, 0, 0],
        "model_a_validation_universe_identical": True,
        "holdout_outcomes_loaded": False, "holdout_predictions_generated": False,
    }


if __name__ == "__main__":
    print(json.dumps(audit_development(), indent=2))
