"""Single frozen Model B run on saved, pre-holdout Model A development rows."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.model0_chronological_v1 import FOLDS, IDENTITY, _calibration, _metric, fold_indices
from src.verify_artifact_provenance import sha256_file, verify_artifacts


ROOT = Path(__file__).resolve().parents[1]
MODEL_A_DIR = ROOT / "data/processed/development/model0_chronological_v1"
OUTPUT_DIR = ROOT / "data/processed/development/model_b_chronological_v1"
MODEL_A_PROVENANCE = ROOT / "MODEL0_DEVELOPMENT_PROVENANCE_V1.json"
SPEC_COMMIT = "0ef1c34af947f3a563c1e9a3652dbed086e3233d"
EXPECTED_TRAIN = (153824, 179491, 205742)
EXPECTED_VALIDATION = (25667, 26251, 25842)
HOLDOUT_START = "2026-06-19"
MODEL_PARAMS = {
    "loss": "log_loss",
    "learning_rate": 0.05,
    "max_iter": 200,
    "max_leaf_nodes": 7,
    "max_depth": None,
    "min_samples_leaf": 500,
    "l2_regularization": 1.0,
    "max_bins": 255,
    "early_stopping": False,
    "class_weight": None,
    "random_state": 42,
    "max_features": 1.0,
    "categorical_features": None,
}
PREDICTION_COLUMNS = [
    "fold", *IDENTITY, "asia_flag", "europe_flag", "us_flag", "us_rth_flag",
    "target_up_15m", "p_base", "p_momentum", "p_logistic", "p_model_b",
]
PROBABILITY_COLUMNS = {
    "baseline_0": "p_base", "baseline_1": "p_momentum",
    "logistic": "p_logistic", "model_b": "p_model_b",
}


def read_json(path: Path):
    return json.loads(path.read_text())


def frozen_predictors() -> list[str]:
    contract = read_json(ROOT / "MODEL0_FEATURE_CONTRACT_V1.json")
    predictors = [item["name"] for item in contract["columns"] if item["role"] == "predictor"]
    if len(predictors) != 36 or any(
        "target" in x or "reference" in x or "eligible" in x for x in predictors
    ):
        raise ValueError("Frozen Model 0 predictor contract differs")
    return predictors


def make_pipeline() -> Pipeline:
    if sklearn.__version__ != "1.9.1":
        raise RuntimeError(f"Frozen sklearn version is 1.9.1; got {sklearn.__version__}")
    model = HistGradientBoostingClassifier(**MODEL_PARAMS)
    params = model.get_params(deep=False)
    if any(params[key] != value for key, value in MODEL_PARAMS.items()):
        raise RuntimeError("Estimator differs from frozen Model B specification")
    if (params["monotonic_cst"], params["interaction_cst"], params["warm_start"]) != (None, None, False):
        raise RuntimeError("Unexpected estimator constraint/warm-start default")
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=False, keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("model", model),
    ])


def assert_development_only(frame: pd.DataFrame) -> None:
    if frame["session_date"].astype(str).ge(HOLDOUT_START).any():
        raise ValueError("Development input contains holdout/partial-session rows")
    if frame["decision_timestamp"].duplicated().any() or not frame["decision_timestamp"].is_monotonic_increasing:
        raise ValueError("Development decisions must be unique and ordered")
    if not frame["target_up_15m"].isin([0, 1]).all():
        raise ValueError("Development input has a nonbinary target")


def validate_inputs(sample: pd.DataFrame, model_a: pd.DataFrame, predictors: list[str]) -> None:
    contract = read_json(ROOT / "MODEL0_FEATURE_CONTRACT_V1.json")
    a_manifest = read_json(MODEL_A_DIR / "run_manifest.json")
    sample_cols = [item["name"] for item in contract["columns"]] + [
        "target_up_15m", "future_boundary_timestamp", "future_reference_timestamp",
    ]
    if list(sample.columns) != sample_cols or list(model_a.columns) != PREDICTION_COLUMNS[:-1]:
        raise ValueError("Saved development schemas differ from frozen inputs")
    if a_manifest["predictor_columns"] != predictors or a_manifest["development_predictions_rows"] != 77760:
        raise ValueError("Model A predictor order or validation count differs")
    if len(sample) != 231584 or len(model_a) != 77760:
        raise ValueError("Development row counts differ")
    for frame in (sample, model_a):
        assert_development_only(frame)
    if not (sample["future_reference_timestamp"] > sample["decision_timestamp"]).all():
        raise ValueError("Invalid development outcome timing")
    if sorted(model_a["fold"].unique().tolist()) != [1, 2, 3]:
        raise ValueError("Unexpected Model A folds")


def probability_distribution(p: np.ndarray) -> dict:
    q = np.quantile(p, [0, .01, .05, .25, .5, .75, .95, .99, 1], method="linear")
    return dict(zip(("minimum", "p01", "p05", "p25", "median", "p75", "p95", "p99", "maximum"), map(float, q)))


def stability(predictions: pd.DataFrame) -> dict:
    groups = {}
    months = predictions["session_date"].astype(str).str.slice(0, 7)
    for month in sorted(months.unique()):
        groups[f"month:{month}"] = predictions.loc[months.eq(month)]
    for label, flag in (("Asia", "asia_flag"), ("Europe", "europe_flag"),
                        ("U.S.", "us_flag"), ("U.S. RTH", "us_rth_flag")):
        groups[f"regime:{label}"] = predictions.loc[predictions[flag].eq(1)]
    return {
        key: {name: _metric(frame["target_up_15m"].to_numpy(dtype=np.int8),
                            frame[column].to_numpy(dtype=np.float64))
              for name, column in PROBABILITY_COLUMNS.items()}
        for key, frame in groups.items() if len(frame)
    }


def run_development(output_dir: Path = OUTPUT_DIR) -> dict:
    """Fit exactly one specified estimator per development fold; no source/holdout reads."""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing populated output directory: {output_dir}")
    verify_artifacts(MODEL_A_PROVENANCE)
    provenance = read_json(MODEL_A_PROVENANCE)
    if sha256_file(MODEL_A_DIR / "run_manifest.json") != provenance["run_manifest_sha256"]:
        raise ValueError("Model A run manifest differs from its provenance record")
    sample = pd.read_parquet(MODEL_A_DIR / "development_eligible_sample.parquet")
    model_a = pd.read_parquet(MODEL_A_DIR / "validation_predictions.parquet")
    predictors = frozen_predictors()
    validate_inputs(sample, model_a, predictors)
    a_manifest = read_json(MODEL_A_DIR / "run_manifest.json")
    a_metrics = read_json(MODEL_A_DIR / "fold_metrics.json")
    fold_records, metrics, distributions, preprocessing, frames = [], [], [], [], []
    output_dir.mkdir(parents=True, exist_ok=True)
    for number, (start, end) in enumerate(FOLDS, 1):
        train, validation, purged, audit = fold_indices(sample, start, end)
        if (len(train), len(validation), len(purged)) != (
            EXPECTED_TRAIN[number - 1], EXPECTED_VALIDATION[number - 1], 0,
        ):
            raise ValueError(f"Fold {number} count/purge differs from pre-result specification")
        old_fold = a_manifest["folds"][number - 1]
        if any(audit[key] != old_fold[key] for key in audit):
            raise ValueError(f"Fold {number} boundary differs from Model A")
        a_validation = model_a.loc[model_a["fold"].eq(number)].reset_index(drop=True)
        s_validation = sample.iloc[validation].reset_index(drop=True)
        same_columns = [*IDENTITY, "asia_flag", "europe_flag", "us_flag", "us_rth_flag", "target_up_15m"]
        if not a_validation[same_columns].equals(s_validation[same_columns]):
            raise ValueError(f"Fold {number} validation rows differ from Model A")
        x_train = sample.iloc[train][predictors].to_numpy(dtype=np.float64, na_value=np.nan)
        x_val = s_validation[predictors].to_numpy(dtype=np.float64, na_value=np.nan)
        y_train = sample.iloc[train]["target_up_15m"].to_numpy(dtype=np.int8)
        y_val = s_validation["target_up_15m"].to_numpy(dtype=np.int8)
        if set(np.unique(y_train)) != {0, 1} or set(np.unique(y_val)) != {0, 1}:
            raise ValueError(f"Fold {number} requires both classes")
        pipeline = make_pipeline()
        pipeline.fit(x_train, y_train)  # Fit imputer, scaler, then all-row boosting, training only.
        model = pipeline.named_steps["model"]
        if model.do_early_stopping_ or model.n_iter_ != 200 or list(pipeline.classes_) != [0, 1]:
            raise RuntimeError(f"Fold {number} estimator did not run frozen 200-iteration design")
        fitted_tree_rows = int(model._predictors[0][0].nodes["count"][0])
        if fitted_tree_rows != len(train) or model._bin_mapper.subsample != 200000:
            raise RuntimeError(f"Fold {number} did not fit all training rows with approved binning")
        p = pipeline.predict_proba(x_val)[:, 1]
        if not np.isfinite(p).all() or not ((p > 0) & (p < 1)).all():
            raise ValueError(f"Fold {number} has invalid probabilities")
        frame = a_validation.copy()
        frame["p_model_b"] = p
        frames.append(frame[PREDICTION_COLUMNS])
        fold_metric = {
            "fold": number,
            "training_up": int(y_train.sum()),
            "training_down": int(len(y_train) - y_train.sum()),
            "training_up_rate": float(y_train.mean()),
            **{name: _metric(y_val, frame[column].to_numpy(dtype=np.float64))
               for name, column in PROBABILITY_COLUMNS.items()},
            "calibration": {"model_b": _calibration(y_val, p)},
        }
        for name in ("baseline_0", "baseline_1", "logistic"):
            for key, value in a_metrics[number - 1][name].items():
                if key in fold_metric[name] and not np.isclose(fold_metric[name][key], value, rtol=0, atol=1e-12):
                    raise ValueError(f"Fold {number} Model A {name} metric changed: {key}")
        fold_metric["model_b"]["log_loss_improvement_vs_base"] = (
            fold_metric["baseline_0"]["log_loss"] - fold_metric["model_b"]["log_loss"]
        )
        fold_metric["model_b"]["log_loss_improvement_vs_logistic"] = (
            fold_metric["logistic"]["log_loss"] - fold_metric["model_b"]["log_loss"]
        )
        metrics.append(fold_metric)
        distributions.append({"fold": number, **probability_distribution(p)})
        imputer, scaler = pipeline.named_steps["impute"], pipeline.named_steps["scale"]
        preprocessing.append({
            "fold": number, "training_rows": len(train),
            "training_null_counts": dict(zip(predictors, map(int, np.isnan(x_train).sum(axis=0)))),
            "imputer_training_medians": dict(zip(predictors, map(float, imputer.statistics_))),
            "scaler_training_means": dict(zip(predictors, map(float, scaler.mean_))),
            "scaler_training_variances": dict(zip(predictors, map(float, scaler.var_))),
            "scaler_training_scales": dict(zip(predictors, map(float, scaler.scale_))),
            "boosting_iterations": int(model.n_iter_),
            "boosting_training_rows_first_tree_root": fitted_tree_rows,
            "histogram_bin_threshold_subsample_cap": 200000,
            "bin_threshold_sampling_expected": len(train) > 200000,
        })
        audit["purged_decisions"] = [str(x) for x in sample.iloc[purged]["decision_timestamp"]]
        fold_records.append(audit)
        with (output_dir / f"fitted_pipeline_fold_{number}.pkl").open("wb") as sink:
            pickle.dump(pipeline, sink, protocol=pickle.HIGHEST_PROTOCOL)
    predictions = pd.concat(frames, ignore_index=True)
    manifest = {
        "version": "mes_model_b_chronological_development_v1",
        "specification": "MODEL_B_SPEC_V1.md", "pre_result_spec_commit": SPEC_COMMIT,
        "sklearn_version": sklearn.__version__, "predictor_columns": predictors,
        "identity_keys": IDENTITY, "model_parameters_explicit": MODEL_PARAMS,
        "preprocessing": "per-fold training-only median imputation then StandardScaler",
        "sample_rows": len(sample), "validation_prediction_rows": len(predictions),
        "folds": fold_records, "holdout_start": HOLDOUT_START,
        "holdout_outcomes_loaded": False, "holdout_predictions_generated": False,
        "input_model_a_provenance": "MODEL0_DEVELOPMENT_PROVENANCE_V1.json",
    }
    predictions.to_parquet(output_dir / "validation_predictions.parquet", index=False)
    for name, value in (("run_manifest", manifest), ("fold_metrics", metrics),
                        ("probability_distribution", distributions),
                        ("development_stability", stability(predictions)),
                        ("training_preprocessing", preprocessing)):
        (output_dir / f"{name}.json").write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    return manifest


if __name__ == "__main__":
    result = run_development()
    print(json.dumps({"sample_rows": result["sample_rows"],
                      "validation_prediction_rows": result["validation_prediction_rows"],
                      "folds": result["folds"]}, indent=2))
