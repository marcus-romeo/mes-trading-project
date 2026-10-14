# Model B Development Specification V1

## Status and research question

This specification is frozen and committed **before the first Model B fit,
prediction, or development result**. Model B asks whether conservative
nonlinear interactions among the already-frozen 36 Model 0 predictors contain
predictive information that Model A's linear logistic regression did not
capture. The comparison is one predefined nonlinear experiment, not an
algorithm search, feature-selection exercise, response to Model A coefficients,
or trading-rule optimization.

Research Protocol V1 precommitted to one conservative tree-model comparison
but did not specify its exact configuration. This document makes that choice
before observing any Model B result. The completed Model A milestone is commit
`0672969be05fa628ed79503cb5bf816d5aac884b`; its settings and results
were committed together, as documented in the README. No Model B result is
used to choose any setting below.

## Frozen data and validation

- Use exactly the 36 ordered columns with `role == "predictor"` in
  `MODEL0_FEATURE_CONTRACT_V1.json`. Do not add, remove, reorder, or select
  predictors. Target outcomes, target eligibility reasons, future references,
  and identifiers are never predictor inputs.
- Use the same eligible development observations and three expanding
  chronological folds as Model A. The existing development sample is
  `data/processed/development/model0_chronological_v1/development_eligible_sample.parquet`,
  whose SHA-256 is recorded in `MODEL0_DEVELOPMENT_PROVENANCE_V1.json`.
- Validate on these inclusive `session_date` blocks, in order:

  | Fold | Validation sessions | Expected retained training rows | Expected validation rows |
  | --- | --- | ---: | ---: |
  | 1 | 2026-03-27 through 2026-04-23 | 153,824 | 25,667 |
  | 2 | 2026-04-24 through 2026-05-21 | 179,491 | 26,251 |
  | 3 | 2026-05-22 through 2026-06-18 | 205,742 | 25,842 |

- Train on all earlier development sessions. At each boundary, purge every
  training label whose actual `future_reference_timestamp` is at or after the
  earliest validation `decision_timestamp`. The saved Model A sample yields
  zero purge removals at all three boundaries; apply the rule rather than
  assuming zero. No shuffled split or change to the observation universe.
- The final historical holdout is 2026-06-19 through 2026-09-10. The partial
  2026-09-11 session is outside it. Do not fit preprocessing or the model on
  holdout data, generate holdout probabilities, inspect holdout probability
  tails, or calculate holdout metrics or P&L during development.

## Frozen preprocessing and estimator

Fit the following pipeline independently on each **purged training fold**,
then transform and predict its validation fold:

```python
Pipeline([
    ("impute", SimpleImputer(
        strategy="median", add_indicator=False, keep_empty_features=True,
    )),
    ("scale", StandardScaler()),
    ("model", HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=7,
        max_depth=None,
        min_samples_leaf=500,
        l2_regularization=1.0,
        max_bins=255,
        early_stopping=False,
        class_weight=None,
        random_state=42,
        max_features=1.0,
        categorical_features=None,
    )),
])
```

Use the installed `scikit-learn==1.9.1`, pinned with the other dependencies in
`requirements.txt`. Constructor inspection in that version confirms that all
listed parameters exist and accept these values. `max_features=1.0` permits
all predictors at every split; `categorical_features=None` treats all 36 as
numeric. Do not pass `sample_weight`; each eligible training row has unit
weight. `monotonic_cst=None`, `interaction_cst=None`, and `warm_start=False`
remain their verified constructor defaults: no monotonic or interaction
constraints and no warm-start continuation. Do not add other constructor
arguments merely because the API exposes them.

With `early_stopping=False`, sklearn does not create an internal validation
split or stop according to validation scores. Its `scoring`,
`validation_fraction`, `n_iter_no_change`, and `tol` defaults are therefore
irrelevant here and are left untouched. The fixed maximum is 200 boosting
iterations. Use direct `predict_proba` for class 1; do not fit a secondary
Platt, isotonic, temperature, or other calibration model.

Median imputation and scaling are fitted on training only, exactly as for
Model A. No missing-indicator columns or target-aware preprocessing are
allowed. Median imputation prevents native missing-value branching from
becoming an additional information channel. Scaling is retained even though
trees do not require it, so the comparison changes primarily the model family.

## Approved sklearn 1.9.1 histogram-binning behavior

The installed sklearn source constructs an internal `_BinMapper` with
`subsample=int(2e5)`, or **200,000 rows**. If a fold has more than 200,000
training rows, the mapper draws 200,000 row indices **with replacement** to
estimate continuous-feature quantile bin thresholds. This applies to Fold 3
(205,742 rows), but not Folds 1 and 2 (153,824 and 179,491 rows). Some
training observations may therefore not determine Fold 3's bin thresholds;
the draw may also contain repeated observations. The classifier passes a seed
derived from `random_state=42` to the mapper, making this implementation step
reproducible under the pinned environment and fixed input order.

**This does not reduce the rows used to fit the boosting model itself. All
eligible fold-training observations remain part of model fitting.** With
`early_stopping=False`, sklearn retains the full training fold for boosting,
bins every training row, and grows trees using the full binned training
matrix. No row subsampling is specified for the boosting fit. The internal
bin-threshold sample is expressly approved for this V1 comparison. Do not
modify sklearn private source, subclass the mapper, or monkey-patch it.

## Pre-result capacity rationale and reporting

Histogram gradient boosting can capture nonlinear thresholds and interactions,
produces binary probabilities, optimizes log loss, handles this development
sample size, and is already in the audited dependency stack. The small 0.05
learning rate, seven-leaf limit per tree, 500-observation minimum leaf, L2
regularization of 1.0, and fixed 200 iterations constrain capacity in the
presence of heavily overlapping 15-minute targets. No early-stopping search,
feature subsampling search, or hyperparameter search is permitted.

Compare against the already saved Baseline 0, Baseline 1, and Model A
predictions on identical validation observations. Primary metric: log loss.
Secondary diagnostics: Brier score, ROC-AUC, fixed 0.5 directional accuracy,
the same ten fixed equal-width calibration bins as Model A, month stability,
and the same broad Asia, Europe, U.S., and U.S. RTH regime summaries. Use the
same metric definitions as `src/model0_chronological_v1.py`. Do not treat
overlapping minute observations as independent evidence for significance.

For each validation fold, report the **outcome-unconditioned** distribution
of Model B class-1 probabilities: minimum, 1st, 5th, 25th, 50th, 75th, 95th,
99th percentiles, and maximum. Use `numpy.quantile(..., method="linear")`
for the percentiles. This describes probability dispersion only; do not
compare outcomes or P&L within probability tails, select confidence cutoffs,
or infer a trading rule from this report.

Run exactly this one configuration. No grid, random, or Bayesian search; no
alternate tree family, leaf count, minimum leaf size, learning rate, iteration
count, L2 strength, preprocessing, or random seed. Preserve a weak Model B
result if that is what the frozen comparison produces. A favorable development
comparison does not by itself authorize opening the holdout; the broader
Model 0 versus separately specified Model 1 research program remains.

The ultimate objective is a realistically executable, net-profitable MES
edge. Model B V1 tests predictive structure only. Do not optimize entry or
abstention thresholds, long/short or time-of-day filters, stops, profit
targets, position sizing, commission/slippage assumptions, or P&L curves.
Economic translation, if justified by stable predictive information, requires
its own separately frozen pre-result protocol.
