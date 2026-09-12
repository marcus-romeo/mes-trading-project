# MES Futures Trading Research

This repository is a research and learning project for building a technically
sound, explainable pipeline for Micro E-mini S&P 500 futures (`MES`) data. It
does not contain a trading strategy or a trained model yet. The completed
historical foundation is ready for the next feature-engineering stage.

## Objective

The project converts immutable Databento trade data into a reusable 1-second
market dataset, then will progress through feature engineering, chronological
model validation, and live-compatible research. The guiding priorities are
correctness first, followed by simplicity, clarity, and reproducibility.

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
versioned 1-second Parquet sessions + JSON run/session manifests
        |
        v
future live-compatible feature engineering
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

The authoritative implementation is `src/trade_processing_v2.py`. Keeping the
logic there prevents the prototype and production implementations from drifting
while the notebooks remain readable explanations and test runners.

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

The first and last raw-covered CME sessions are partial. In the manifest,
`is_complete_session` means the raw source spans the nominal CME 17:00–16:00
Chicago session window; it does not mean a holiday or early-close session had
ordinary trading hours. The manifest also conservatively records sessions
touched by Databento degraded-date notices. Downstream training and test splits
must choose their session eligibility rules deliberately.

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

Build feature engineering from `full_history_v2_event_time`, using explicit
decision-time rules and deliberate treatment of partial, degraded, holiday,
and contract-roll sessions. Do not use `full_history_v1` for modeling.
