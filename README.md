# MES Futures Trading Research

This repository is a research and learning project for building a technically
sound, explainable pipeline for Micro E-mini S&P 500 futures (`MES`) data. It
does not contain a trading strategy or a trained model yet. The event-time
one-second and derived one-minute historical foundations are complete and
frozen; the next stage is actual model-feature creation.

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
        |
        v
future live-compatible model feature creation
        |
        v
future chronological model validation and trading research
```

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
- `02_databento_cost_estimate.ipynb` — records historical-data decisions and
  contains guarded download examples. Existing raw targets hard-stop rather
  than silently re-download or overwrite data.
- `03_raw_data_validation.ipynb` — validates raw coverage, contract mapping,
  and Databento quality conditions.
- `04_trade_processing.ipynb` — technical report for the 1-second pipeline,
  its historical prototypes, authoritative v2 implementation, and completed
  production/audit record.
- `05_feature_engineering.ipynb` — technical record for the frozen one-minute
  derivation, its read-only full-history audit, and the transition to model
  feature creation.

The authoritative implementations are `src/trade_processing_v2.py` and
`src/feature_engineering_v1.py`. Keeping production logic in source modules
prevents notebook prototypes and validation paths from drifting.

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

Each minute row is timestamped at the decision boundary immediately after its
source minute: a row at `10:31:00` contains only the completed
`10:30:00`–`10:30:59.xxx` interval. No contributing trade occurs at or after
the decision timestamp. This is an event-time eligibility boundary; a future
live IBKR system must additionally wait until the relevant final event has been
received and processed before acting.

The first and last raw-covered CME sessions are partial. In the manifest,
`is_complete_session` means the raw source spans the nominal CME 17:00–16:00
Chicago session window; it does not mean a holiday or early-close session had
ordinary trading hours. The manifest also conservatively records sessions
touched by Databento degraded-date notices. Downstream training and test splits
must choose their session eligibility rules deliberately.

Databento prices are unadjusted across continuous-contract rolls. The minute
foundation preserves instrument identity and `contract_change`; returns,
momentum, volatility, level distances, and other rolling price features must
restart at every new contract. Native Databento aggressor-side fields remain
historical benchmark fields only. Production model inputs must use the
live-compatible inferred order-flow fields.

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

## Next step

Create model features from the frozen `1m/full_history_v1` foundation, using
explicit decision-time rules and deliberate treatment of partial, degraded,
holiday, and contract-roll sessions. Do not use the receive-time
`1s/full_history_v1` dataset for modeling.
