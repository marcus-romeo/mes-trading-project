# MES Futures Trading Research

This repository is a research and learning project for building a technically
sound, explainable pipeline for Micro E-mini S&P 500 futures (`MES`) data. It
does not contain a trading strategy or a trained model yet. The event-time
one-second and derived one-minute historical foundations are complete and
frozen. Target V1 is preserved as provenance; corrected Target V2 is the
approved outcome layer. The Model 0 feature layer has been built and
independently audited; no predictive model has been trained or evaluated.

## Objective

The project converts immutable Databento trade data into reusable event-time
one-second and one-minute market datasets, then will progress through feature
engineering, chronological model validation, and live-compatible research. The
guiding priorities are correctness first, followed by simplicity, clarity, and
reproducibility.

## Data and time policy

- **Source:** Databento `GLBX.MDP3` `trades`, downloaded as the continuous
  `MES.v.0` series.
- **Raw source:** `data/mes_trades_2025-10-07_to_2026-09-11.dbn`. It is an
  immutable source artifact and must never be overwritten by a notebook run.
- **Coverage:** 99,319,450 trades from 2025-10-07 through 2026-09-10 UTC.
- **Market clock:** version 2 uses Databento `ts_event` (exchange event time),
  stored in UTC. Databento's DataFrame index is `ts_recv` (receive time) and
  is not used to construct market bars.
- **CME session date:** MES sessions use `America/Chicago`; trades at or after
  17:00 CT belong to the following session date. Timezone conversion is
  daylight-saving-aware.

## Architecture

```text
immutable Databento DBN trades
        |
        |  event-time chunks; carry state and incomplete final second
        v
validated V2 event-time 1-second Parquet sessions + JSON manifests
        |
        |  one session at a time; completed-minute aggregation
        v
frozen V1 one-minute foundation + JSON run manifest
        |                              |
        |  15-minute paths; V2 trades  |  frozen causal formulas
        v                              v
corrected Target V2              validated Model 0 features
(Target V1 preserved)            (no target data used)
                                       |
                                       v
                              future Model 1 features
```

Target V2 is joined with a feature layer only later for controlled
chronological validation. Trading research follows that evaluation.

The 1-second processor deliberately retains the validated chunk architecture:
it carries tick-rule and price-change state between chunks, holds the final
incomplete second until the next chunk, resets state at a contract change, and
writes one validated Parquet file per CME session.

## Historical-to-live compatibility

`tick_rule_v1` estimates trade direction from the sequence of traded prices:
an uptick is buyer-directed, a downtick is seller-directed, and an unchanged
price inherits the most recent non-zero direction. It resets when the futures
contract changes. This is intentionally usable with a planned live trade feed.

Databento's native aggressor side is retained only as a historical benchmark.
It must not become a production feature because the planned live feed does not
provide an equivalent exchange-native label.

## Notebook guide

- `01_environment_test.ipynb` — confirms the isolated Python environment.
- `02_databento_cost_estimate.ipynb` — archived cost-estimation and
  acquisition record. Run All makes no remote requests.
- `03_raw_data_validation.ipynb` — validates raw coverage, contract mapping,
  and Databento quality conditions.
- `04_trade_processing.ipynb` — technical record for the V2 design decisions
  and validation results; the executable implementation is in source.
- `05_feature_engineering.ipynb` — technical record for the frozen one-minute
  derivation, its read-only full-history audit, and the transition to model
  feature creation.
- `06_target_eligibility.ipynb` — read-only Target V2 boundary and manifest
  checks, including the Target V1-to-V2 change record.

The executable implementations are `src/trade_processing_v2.py`,
`src/feature_engineering_v1.py`, `src/target_engineering_v2.py`, and
`src/model0_features_v1.py`.
`src/target_audit_v2.py` provides a separate read-only source audit. The
read-only `src/model0_audit_v1.py` audits the feature layer against the minute
source, independently recalculating selected values. The preserved
`src/target_engineering_v1.py` records the historical Target V1
implementation. Keeping production logic in source modules prevents notebook
prototypes and validation paths from drifting.

Recommended review order:

1. This README
2. `03_raw_data_validation.ipynb`
3. `src/trade_processing_v2.py`
4. `05_feature_engineering.ipynb` and `src/feature_engineering_v1.py`
5. `06_target_eligibility.ipynb` and the Target V2 source/audit modules

The pre-model research design is recorded in
[Research Protocol V1](RESEARCH_PROTOCOL_V1.md).

The frozen feature definitions are recorded in
[Feature Formula Specification V1](FEATURE_FORMULAS_V1.md).

## Dataset versions and status

- `data/processed/1s/full_history_v1/` is preserved for provenance. It was
  built with receive-time (`ts_recv`) indexing and is **not approved for
  modeling or feature engineering**.
- `data/processed/1s/full_history_v2_event_time/` is the approved historical
  research foundation. Its event-time production run completed successfully:
  99,319,450 raw trades, 298,546,252 raw volume, 13,165,975 one-second rows,
  and 242 session files.
- The independent post-run audit passed. V2 uses the fixed 33-column Parquet
  schema and records its provenance in `run_manifest.json` and
  `session_manifest.json`.
- `data/processed/1m/full_history_v1/` is the frozen version 1 **minute
  derivation** from the approved V2 event-time foundation: 242 sessions,
  329,337 minute rows, 298,546,252 volume, and 99,319,450 trades. Its
  `run_manifest.json` records the timing contract, schema, totals, and rolls.
  The directory name does **not** refer to the invalid receive-time 1-second
  V1 dataset above.
- `data/processed/targets/target_v1/` is the preserved historical Target V1.
  Its headline counts reconcile, but three eligible rows compare prices across
  rolls exactly at future reference boundaries. Do not use it for modeling.
- `data/processed/targets/target_v2/` is the corrected target from the same
  approved 1m and V2 1s sources. It checks the actual instrument_id of both
  endpoint trades and stores those IDs as target-side audit fields. Its writer
  refuses an existing destination; its manifest records every changed row and
  old/new aggregate counts. V2 has 242 files, 329,337 candidates, 318,444
  timing-qualified endpoint pairs, 9,245 equal-price exclusions, and 309,196
  eligible labels (156,957 UP; 152,239 DOWN). Exactly three former UP labels
  are now `FUTURE_REFERENCE_CONTRACT_MISMATCH`; no other shared target field
  changed outside those rows. Target-side prices, delays, IDs, labels, and
  eligibility diagnostics must never enter the predictive feature matrix.
- `data/processed/features/model0_v1/` contains the 242-session, 329,337-row
  Model 0 layer derived solely from the frozen minute foundation. Its 41-column
  Arrow schema contains five identity fields and exactly 36 predictors, with
  unavailable feature values stored as nulls. `run_manifest.json` records
  source lineage, per-session validity summaries, and per-feature null counts;
  `independent_audit.json` records the separate full-history audit and
  feature-only distribution checks. The Parquet files are ignored by Git and
  must be transferred separately when reproducing this built state.

Each minute row is timestamped at the decision boundary immediately after its
source minute: a row at `10:31:00` contains only the completed
`10:30:00`–`10:30:59.xxx` interval. No contributing trade occurs at or after
the decision timestamp. This is an event-time eligibility boundary; a future
live IBKR system must additionally wait until the relevant final event has been
received and processed before acting.

The first and last raw-covered CME sessions are partial. In the manifest,
`is_complete_session` is source-boundary metadata: it records whether the
acquired source range reaches the nominal CME session endpoints. It does not
assess exchange-calendar coverage, ordinary sparse trading, or vendor quality
within a session. The manifest also conservatively records sessions touched by
Databento degraded-date notices. Downstream training and test splits must
choose their session eligibility rules deliberately.

Databento prices are unadjusted across continuous-contract rolls. The minute
foundation preserves instrument identity and `contract_change`; returns,
momentum, volatility, level distances, and other rolling price features must
restart at every new contract. Native Databento aggressor-side fields remain
historical benchmark fields only. Production model inputs must use the
live-compatible inferred order-flow fields when order flow is included.

The V2 `unique_price_levels` and `max_volume_at_price` fields are intentionally
not included in the minute foundation. Their exact minute-level equivalents
cannot be reconstructed from one-second summaries without misleading
approximations. V2 remains available if later research needs them; a true
minute/session volume-at-price layer would require a separate deliberate design.

## Validation philosophy

Before data is written, each session is checked for one CME date, chronological
ordering, unique `(timestamp_second, instrument_id)` keys, OHLC consistency,
volume and delta reconciliation, event-time consistency, contract mapping, and
stable types. The completed memory-bounded audit rescanned the raw DBN and
output files to reconcile trade count, volume, active event-time-second count,
and run/session-manifest facts independently.

## Important limitations

- A 1-second row describes the completed interval `[t, t + 1 second)` and is
  only available after that second closes. Future features and models must use
  an explicit decision-time convention to avoid look-ahead bias.
- Continuous-contract roll boundaries require downstream features to segment
  by contract or otherwise mask roll-dependent returns and rolling windows.
- Degraded Databento dates are metadata warnings, not automatic proof that all
  trades are unusable; their treatment must be chosen explicitly in later
  research.

## Model 0 verification and next step

The feature implementation uses `FEATURE_FORMULAS_V1.md` and the ordered
schema/source-validity mask in `MODEL0_FEATURE_CONTRACT_V1.json`. It builds one
row per approved minute independently of Target V2 eligibility or outcomes.
The deterministic suite can be run with
`python -m unittest discover -s tests -v`. The production writer is
`src.model0_features_v1.build_full_history_model0`; it refuses to overwrite a
populated output directory. The separate read-only audit can be rerun with
`python -m src.model0_audit_v1` after the ignored feature Parquets are present.

Review and commit this feature-engineering milestone before beginning
chronological modeling. No target was joined, no predictive model was trained,
and no holdout performance was inspected during feature construction.

## Reproducing the current data state

Git contains source code, notebooks, research specifications, and JSON run
manifests. The purchased raw DBN, generated Parquet sessions, `.env`, and
`.venv` are deliberately ignored. A clean clone therefore cannot read the
historical target or Model 0 feature files without a separately transferred
artifact set or a rebuild from the purchased DBN. The required local paths
are listed in the dataset section above; the raw DBN is the only input to the
original event-time build. A rebuild must use `ts_event`, verify the DBN
metadata and contract mapping, and run the source-to-output audits before
trusting the result. Do not run production writers against already populated
frozen paths.

`requirements.txt` records the current Python dependencies; the audited local
environment used Python 3.12.10. Manifests record row counts and lineage, not
cryptographic hashes. See `ARTIFACT_PROVENANCE_V1.json` for separately recorded
SHA-256 hashes of the currently held raw, frozen, and corrected target files.
The `.env` contents and any Databento credentials are not part of Git or that
provenance record. After transferring the ignored artifacts, run
`python -m src.verify_artifact_provenance` from the repository root to check
every recorded file without modifying it.
