# MES Research Protocol V1

## Status and scope

Research Protocol V1 was defined before predictive model results were
inspected. Its purpose is to reduce leakage, hindsight-driven feature changes,
multiple-testing bias, and accidental overfitting.

Feature Set V1 is conceptually frozen for the first experiment. Later research
ideas require a separately versioned protocol or feature set rather than a
silent addition after results are seen.

The exact pre-implementation definitions are recorded in
[Feature Formula Specification V1](FEATURE_FORMULAS_V1.md).

The dated pre-model admission clarification in that specification defines
relative-activity source rows, comparable prior sessions, and opening-range
validity before any Model 0 feature values or predictive results exist.

This document specifies research design only. It does not create features,
targets, datasets, models, or trading rules.

## 1. Research question

Can information available before a one-minute MES decision boundary predict
whether MES will be trading higher or lower approximately 15 minutes later?

The initial task is binary directional classification. The model outputs
P(UP), not only a hard UP or DOWN prediction.

## 2. Decision cadence

One prediction is generated at every eligible one-minute boundary T. For
example:

- 10:30 has an outcome boundary at 10:45.
- 10:31 has an outcome boundary at 10:46.
- 10:32 has an outcome boundary at 10:47.

Overlapping 15-minute prediction windows are intentional.

## 3. Information cutoff

For a prediction at boundary T, model inputs may contain only information
available strictly before T. The most recent completed one-minute bar is
eligible when it summarizes the interval ending at T and contains no event at
or after T. No later event may enter the feature vector.

## 4. Reference-price rule

The original exact-first-second rule was tested before modeling on the frozen
V2 event-time foundation:

- 329,337 observed trading minutes;
- 74.51% exact-first-second coverage;
- 60.17% exact-first-second 15-minute pair retention;
- 97.87% pair retention when both endpoints use a first trade within 10
  seconds.

Exact-first-second coverage was 61.34% in Asia, 70.82% in Europe, 93.92% in
the U.S. bucket, and 96.60% during U.S. RTH. The shortfall is primarily a
valid overnight liquidity pattern, not a reason to synthesize prices.

Protocol V1 therefore uses:

- **Start reference:** the first actual V2 trade at or after T, with delay
  less than 10 seconds.
- **Future reference:** the first actual V2 trade at or after T plus 15
  minutes, with delay less than 10 seconds.

The reference price is the first trade price represented by that event-time
second. If either endpoint lacks a qualifying trade, the target observation is
excluded. This timing rule was selected from availability analysis before any
predictive result was examined.

## 5. Target

For valid endpoint prices:

- future price greater than start price: UP = 1;
- future price less than start price: DOWN = 0;
- equal prices: exclude the observation from the binary target.

The equal-price policy is fixed before model results are inspected.

## 6. Hard target boundaries

A target is invalid if its T through T plus 15-minute interval:

- crosses a futures contract roll;
- crosses a session boundary;
- crosses a known scheduled market closure;
- crosses the confirmed November 28, 2025 CME outage;
- crosses an unresolved isolated source gap;
- lacks the required actual timestamps; or
- violates the less-than-10-second endpoint rule.

Prices are never synthesized, and closed periods are never filled with
artificial zero-volume bars.

## 7. Market-calendar policy

Legitimate CME behavior remains in the research sample, including normal
sessions, holidays, scheduled early closes, the normal maintenance closure,
legitimate shortened sessions, the November 28, 2025 CME outage, and
Databento-degraded sessions unless a specific observation is unusable.

A short session is not bad data merely because it has fewer trading minutes.
The frozen sparse one-minute representation remains valid. The first and last
source-boundary sessions require observation-level handling because the
purchased range begins and ends inside those CME sessions.

No entire session is automatically excluded solely because it is a holiday,
early close, or Databento-warning session.

## 8. Market coverage and regimes

Protocol V1 uses the full eligible CME Globex session. Training coverage and
intended prediction coverage should match. All timestamps use timezone-aware
America/Chicago conversion.

| Regime | Chicago time |
|---|---|
| Asia | 17:00–02:00 |
| Europe | 02:00–08:30 |
| U.S. | 08:30–16:00 |
| U.S. RTH | 08:30–15:00 |

Asia, Europe, and U.S. are the main non-overlapping research buckets. U.S.
RTH is an additional subset flag. These labels are research regimes, not
claims about the literal opening time of every underlying global exchange.

V1 begins with one global model. Separate Asia, Europe, and U.S. models are
outside V1.

## 9. Lookback and missing-feature policy

All rolling features use actual timestamps. Fifteen rows never automatically
mean 15 minutes.

A rolling feature may not silently cross closures, unresolved gaps, session
boundaries when continuous elapsed trading is required, or contract rolls. If
a longer lookback is unavailable after a reopen or boundary, the otherwise
valid prediction observation remains and that feature is unavailable. Historic
observations are not invented solely to populate a lookback.

Any learned missing-value treatment is fitted on training data only.

## 10. Contract-roll policy

Continuous-contract prices remain unadjusted. No price-derived target or
rolling feature may cross instrument_id.

At a roll:

- price-based rolling state resets;
- VWAP and price-level logic remain contract-safe;
- targets crossing the roll are invalid; and
- prior-session levels are unavailable without a comparable same-contract
  prior period.

Target V2 additionally checks the actual start and future reference trades'
instrument_id, including when a roll occurs exactly at a reference boundary.
It preserves Target V1 as provenance and retains the original horizon,
strict less-than-10-second timing, and equal-price policy.

Contract basis change is not interpreted as market price movement.

## 11. Feature Set V1: Model 0

Model 0 asks whether conventional price, volatility, participation, structure,
and time context predict 15-minute direction. The following conceptual feature
set is frozen.

| Family | Features |
|---|---|
| Momentum | 1-, 5-, 15-, 30-, and 60-minute returns |
| Price behavior | 5- and 15-minute range; 5- and 15-minute trend efficiency |
| Volatility | 15- and 60-minute realized volatility; short-versus-long volatility ratio |
| Participation | 1-minute volume, trade count, average trade size, trade-size dispersion, maximum trade size, active-second count |
| Relative activity | Relative volume by time of day, relative trade count by time of day, activity acceleration |
| Current structure | Distance from session VWAP, session-range position, distance from current session high and low |
| Previous-session structure | Distance from prior-session high, low, and close |
| Opening structure | Position relative to the completed U.S. 30-minute opening range |
| Time and regime | Time-of-day representation, minutes since CME session open, Asia, Europe, U.S., and U.S. RTH flags |

## 12. Model 0 implementation principles

- Returns use a scale-independent representation such as log returns.
- Relative-volume and relative-trade-count baselines use past information
  only.
- Market-structure distances are scale-aware or volatility-aware.
- Opening-range information becomes available only after that range completes.
- Historical baselines may not use future sessions.
- Every production feature must be reproducible from the intended live data
  path.

Exact numerical formulas are implemented and tested before predictive modeling
while preserving this frozen feature universe.

## 13. Feature Set V1: Model 1

Model 1 contains all Model 0 features plus live-reproducible trade and tape
information:

- 1-minute inferred delta divided by volume;
- 5- and 15-minute cumulative inferred delta;
- inferred buy/sell trade-count imbalance;
- uptick/downtick imbalance;
- delta acceleration;
- 5- and 15-minute price-versus-delta divergence;
- price movement per volume;
- price movement per absolute inferred delta; and
- effort-versus-result.

The test is whether inferred trade and tape information adds stable
out-of-sample information beyond conventional price, volatility, volume,
structure, and time context.

## 14. Native Databento aggressor policy

Native Databento aggressor side and delta are diagnostic only. They may
validate the live-reproducible inferred trade-direction method, but may not
enter Model 0 or Model 1, influence production feature selection, or become a
production predictor.

The planned IBKR live trade feed has no equivalent native aggressor-side
field. Historical training features must remain reproducible live.

## 15. Features excluded from V1

The initial experiment excludes:

- RSI, MACD, stochastic oscillators, large moving-average collections,
  Bollinger Band variants, Fibonacci levels, candlestick labels, and large
  technical-indicator libraries;
- hand-labeled bullish or bearish setups and arbitrary return-horizon
  collections;
- true bid/ask spread, historical quote imbalance, resting liquidity,
  cancellations, replenishment, and L2 depth;
- nanosecond ordering features, approximate volume-at-price features, and
  native Databento aggressor features.

Any later addition requires technical justification and a versioned protocol
before its result is inspected.

## 16. Preprocessing and leakage policy

Anything that learns from data is fitted on training data only. This includes
scaling, imputation, learned baselines, learned thresholds, preprocessing
parameters, and model parameters.

Validation and final-holdout data do not influence fitted preprocessing. Purely
causal features are permitted when they use only information available before
each prediction timestamp.

## 17. Development and final holdout

The newest 60 source-covered CME sessions before the final truncated
source-boundary session are the untouched historical holdout:

- **Start session_date:** 2026-06-19
- **End session_date:** 2026-09-10
- **Session count:** 60

The final source-boundary session, 2026-09-11, is not part of this holdout.
Holiday and early-close sessions remain. Databento-warning sessions are not
automatically removed. Development and holdout use the same observation-level
validity rules.

Everything before the holdout is development history.

## 18. Development validation

Development uses three expanding chronological folds. The final 60 development
sessions form three consecutive 20-session validation blocks:

1. Train on all earlier development sessions; validate on the next 20.
2. Expand training through the first validation block; validate on the next
   20.
3. Expand training through the second validation block; validate on the next
   20.

At each train/validation boundary, purge training labels whose 15-minute
outcome reaches into validation. Shuffled cross-validation is not permitted.

After development selection, refit the selected frozen candidate on all
development history before the one-time final-holdout evaluation.

## 19. Baselines and initial models

Before flexible models:

- **Baseline 0:** historical/base-rate UP probability.
- **Baseline 1:** simple recent-momentum directional baseline.
- **Model A:** regularized logistic regression.
- **Model B:** one conservative nonlinear tree-based model.

Large hyperparameter searches are outside V1. The small tree-model
configuration is frozen before final evaluation.

## 20. Primary evaluation

Probability quality is primary:

- log loss versus the base-rate predictor.

Also report Brier score, calibration, ROC-AUC, directional accuracy, and
improvement over the momentum baseline. Report stability fold by fold, month
by month, and across Asia, Europe, U.S., and U.S. RTH.

Fifteen-minute targets overlap. One-minute predictions are not treated as
statistically independent; later uncertainty estimates use sensible
time/session blocks.

## 21. Model 0 versus Model 1

The central V1 comparison is:

- **Model 0:** conventional market information.
- **Model 1:** the identical foundation plus inferred trade and tape
  information.

The question is whether Model 1 adds stable out-of-sample information beyond
Model 0. Model 0 is not changed after Model 1 results are inspected merely to
improve the comparison.

## 22. Final-holdout rule

Inspect the final 60-session holdout once, after development decisions are
frozen. After those results are seen, do not alter target, features,
eligibility, preprocessing, thresholds, model family, or hyperparameters and
then retest the same period as untouched.

A subsequent change creates a new research version and requires new forward
evidence.

## 23. No trading-strategy optimization yet

V1 tests predictive information first. It does not optimize entries,
probability thresholds, stops, profit targets, position sizing, maximum daily
trades, time-of-day exclusions, or trading P&L.

Economic evaluation follows only evidence of stable directional prediction and
accounts for commission, bid/ask effects, slippage, turnover, latency, and
actual live or paper execution behavior.

## 24. Live forward stage

If a historical candidate survives the untouched holdout, freeze it and
collect approximately 30–60 trading days of IBKR paper/live-feed predictions.
At minimum, log:

- prediction timestamp;
- exchange/event timestamp where available;
- local receipt timestamp;
- model version;
- feature version;
- probability;
- predicted direction;
- actual outcome; and
- simulated execution and P&L information when strategy testing begins.

Historical event-time eligibility is not automatically identical to live
receipt-time availability. That difference is measured during the IBKR stage.

## 25. Change control

Once predictive experiments begin, Feature Set V1 is frozen. A disappointing
result does not justify silently adding indicators or changing target rules.

Every meaningful change is documented before its result is inspected and is
versioned separately, for example Research Protocol V2, Feature Set V2, or
Model Specification V2. Git history preserves those decisions.

## 26. Pre-model clarification and version control

On 2026-10-01, before Model 0 feature calculation or model results, the
project recorded operational session-admission rules in Feature Formula
Specification V1. These clarify existing formulas without adding features or
using target eligibility to select historical activity observations. Target V2
was created separately because independent audit found three Target V1 labels
whose future reference trades belonged to the next contract. Target V1 and
all earlier foundations remain preserved; V2 supplies the corrected outcome
layer for subsequent research. Neither change follows predictive results.

A final pre-implementation clarification in the same feature specification
freezes full elapsed-window source coverage, current-session cumulative-state
validity after source truncation or internal gaps, and same-contract activity
acceleration. [Model 0 Feature Contract V1](MODEL0_FEATURE_CONTRACT_V1.json)
is the sole machine-readable definition of the four known unavailable
intervals and the ordered 41-column output schema: five identity columns and
36 Model 0 predictors. The feature row universe is every observed approved
one-minute decision boundary, independent of Target V2 eligibility. These
decisions precede Model 0 feature calculation and predictive modeling.

## 27. Current project stage

**Completed**

- Raw data acquisition
- Raw validation
- V2 event-time one-second foundation
- V1 one-minute foundation
- Repository and notebook cleanup
- Session/gap investigation
- First-second timing study
- Research Protocol V1 design
- Feature Formula Specification V1 and pre-model admission clarification
- Final Model 0 validity rules, gap mask, and output schema contract
- Target V1 construction and audit; preserved for provenance
- Corrected Target V2 construction and independent contract-identity audit

**Next**

1. Implement causal Model 0 features from the frozen one-minute foundation.
2. Independently validate the feature formulas and missingness rules.
3. Implement and validate Model 1 features later.
4. Construct chronological development folds and purge crossing labels.
5. Begin predictive modeling only after the preceding steps.
