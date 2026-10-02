# MES Feature Formula Specification V1

## Status and scope

Feature Formula Specification V1 was frozen before predictive model results
were inspected. It implements the conceptual feature universe frozen in
[Research Protocol V1](RESEARCH_PROTOCOL_V1.md).

Formula changes after predictive results begin require a new version. This
document defines feature formulas only; it does not report predictive
usefulness, create a feature matrix, create targets, or train models.

## Shared timing and source notation

At prediction boundary T:

- m_T is the completed one-minute row summarizing [T-1m, T).
- C_T is close(m_T), the final completed-minute price known before T.
- r_T = log(C_T / C_(T-1m)).
- Exact elapsed-time windows are required; h rows never automatically mean h
  minutes.

G_h(T) is a price-path window. It requires exact completed closes at T,
T-1m, through T-hm; one instrument; and no prohibited boundary.

W_h(T) is an aggregation window containing actual trade-bearing minute data
inside [T-hm, T). It may contain sparse valid trading minutes, but may not
cross a closure, unresolved gap, required-continuity session boundary, or
contract roll.

Source notation:

| Symbol | Frozen one-minute field |
|---|---|
| V | total_volume |
| N | trade_count |
| Q | size_squared_sum |
| PV | price_volume_sum |
| D | inferred_delta |

For h valid one-minute log returns:

    sigma_h = sqrt(mean(r^2))

The local 60-minute price scale is:

    S_60 = sqrt(60) * sigma_60
         = sqrt(sum(r^2))

No missing minute is synthesized. A price-path formula requiring G_h is
missing if a required close is unavailable. W_h formulas aggregate actual
trades but are missing when their elapsed window crosses a prohibited
boundary.

## Model 0 formulas

### Momentum

| Feature | Formula | Source and normalization | Missing and boundary rule | Live parity |
|---|---|---|---|---|
| 1-minute return | R_1 = log(C_T / C_(T-1m)) | Completed-minute close; log return | G_1 required | LIVE SAFE WITH STATE |
| 5-minute return | R_5 = log(C_T / C_(T-5m)) | Completed-minute close; log return | G_5 required | LIVE SAFE WITH STATE |
| 15-minute return | R_15 = log(C_T / C_(T-15m)) | Completed-minute close; log return | G_15 required | LIVE SAFE WITH STATE |
| 30-minute return | R_30 = log(C_T / C_(T-30m)) | Completed-minute close; log return | G_30 required | LIVE SAFE WITH STATE |
| 60-minute return | R_60 = log(C_T / C_(T-60m)) | Completed-minute close; log return | G_60 required | LIVE SAFE WITH STATE |

### Price behavior

| Feature | Formula | Source and normalization | Missing and boundary rule | Live parity |
|---|---|---|---|---|
| 5-minute range | log(max(high in W_5) / min(low in W_5)) | high, low; log range | Valid W_5 required | LIVE SAFE WITH STATE |
| 15-minute range | log(max(high in W_15) / min(low in W_15)) | high, low; log range | Valid W_15 required | LIVE SAFE WITH STATE |
| 5-minute trend efficiency | TE_5 = sum(r) / sum(abs(r)) | G_5 log-return path; bounded [-1, 1] | Use 0 when denominator is exactly zero; otherwise G_5 required | LIVE SAFE WITH STATE |
| 15-minute trend efficiency | TE_15 = sum(r) / sum(abs(r)) | G_15 log-return path; bounded [-1, 1] | Use 0 when denominator is exactly zero; otherwise G_15 required | LIVE SAFE WITH STATE |

Trend efficiency measures directional displacement relative to total traveled
price path. Positive values indicate an efficient upward path, negative values
an efficient downward path, and values near zero a choppy path.

### Volatility

| Feature | Formula | Source and normalization | Missing and boundary rule | Live parity |
|---|---|---|---|---|
| 15-minute realized volatility | sigma_15 = sqrt(mean(r^2)) | G_15 log returns; non-annualized RMS | G_15 required | LIVE SAFE WITH STATE |
| 60-minute realized volatility | sigma_60 = sqrt(mean(r^2)) | G_60 log returns; non-annualized RMS | G_60 required | LIVE SAFE WITH STATE |
| Short/long volatility ratio | sigma_15 / sigma_60 | Dimensionless ratio | Missing if either input missing or sigma_60 <= 1e-12 | LIVE SAFE WITH STATE |

RMS log-return volatility is used rather than de-meaned standard deviation so
a smooth directional move still records price movement. Neither volatility
measure is annualized.

### Participation

| Feature | Formula | Source and normalization | Missing and boundary rule | Live parity |
|---|---|---|---|---|
| 1-minute volume | log1p(V_T) | total_volume | m_T required | LIVE SAFE WITH STATE |
| Trade count | log1p(N_T) | trade_count | m_T required | LIVE SAFE WITH STATE |
| Average trade size | log(V_T / N_T) | total_volume, trade_count | Missing if N_T = 0 | LIVE SAFE WITH STATE |
| Trade-size dispersion | sqrt(max(Q_T/N_T - (V_T/N_T)^2, 0)) / (V_T/N_T) | total_volume, trade_count, size_squared_sum; coefficient of variation | Missing if N_T = 0 or V_T/N_T = 0 | LIVE SAFE WITH STATE |
| Maximum trade size | log1p(max_trade_size_T) | max_trade_size | m_T required | LIVE SAFE WITH STATE |
| Active-second count | active_second_count_T / 60 | active_second_count; ratio in (0, 1] | m_T required; no imputed seconds | LIVE SAFE WITH STATE |

### Relative activity

For minute-of-session q, use a causal baseline from the previous 20 eligible
source-covered sessions with an observed row at q:

    b_V(q) = median(log1p(volume))
    b_N(q) = median(log1p(trade_count))

The median is taken over those 20 prior same-q observations. Current and future
sessions never enter the baseline. The baseline is missing until 20 eligible
prior observations exist.

| Feature | Formula | Source and normalization | Missing and boundary rule | Live parity |
|---|---|---|---|---|
| Relative volume | log1p(V_T) - b_V(q) | Log difference from causal same-time baseline | Current row and b_V(q) required | LIVE SAFE WITH STATE |
| Relative trade count | log1p(N_T) - b_N(q) | Log difference from causal same-time baseline | Current row and b_N(q) required | LIVE SAFE WITH STATE |
| Activity acceleration | log1p(V_T) - log1p(V_[T-6m,T-1m) / 5) | Current volume versus prior five elapsed minutes | Valid preceding five-minute window required | LIVE SAFE WITH STATE |

The previous-five quantity is the sum of actual volume in [T-6m, T-1m)
divided by five. It uses no current or future trade and does not create
zero-volume bars.

### Market structure

Within the active same-contract segment of the current CME session:

    VWAP_T = sum(PV_i) / sum(V_i)

where all contributing minutes end at or before T.

Define a signed, dimensionless level distance:

    distance(level) = log(C_T / level) / S_60

| Feature | Formula | Source and normalization | Missing and boundary rule | Live parity |
|---|---|---|---|---|
| Session VWAP distance | distance(VWAP_T) | close, price_volume_sum, total_volume; normalized by S_60 | Missing if C_T, VWAP_T, or S_60 unavailable; reset at roll | LIVE SAFE WITH STATE |
| Session-range position | (C_T - session_low) / (session_high - session_low) | close, high, low; bounded position | Use 0.5 if high equals low; reset at roll | LIVE SAFE WITH STATE |
| Session-high distance | distance(session_high) | close, high; normalized by S_60 | Missing if S_60 unavailable; reset at roll | LIVE SAFE WITH STATE |
| Session-low distance | distance(session_low) | close, low; normalized by S_60 | Missing if S_60 unavailable; reset at roll | LIVE SAFE WITH STATE |
| Prior-session-high distance | distance(prior_session_high) | close, prior same-contract high; normalized by S_60 | Comparable source-covered same-contract prior session required | LIVE SAFE WITH STATE |
| Prior-session-low distance | distance(prior_session_low) | close, prior same-contract low; normalized by S_60 | Comparable source-covered same-contract prior session required | LIVE SAFE WITH STATE |
| Prior-session-close distance | distance(prior_session_close) | close, prior same-contract close; normalized by S_60 | Comparable source-covered same-contract prior session required | LIVE SAFE WITH STATE |

Price-derived state never crosses instrument_id. A prior-session level is
unavailable when no comparable same-contract prior session exists.

### Opening structure

The U.S. opening range is the actual trade range during 08:30–09:00
America/Chicago:

    OR_high = max(high)
    OR_low = min(low)

over valid opening-range data.

| Feature | Formula | Source and normalization | Missing and boundary rule | Live parity |
|---|---|---|---|---|
| Opening-range position | (C_T - OR_low) / (OR_high - OR_low) | close, opening-range high and low; unclipped position | Available at or after 09:00 CT; use 0.5 if range is zero; missing after an invalid range boundary or contract transition | LIVE SAFE WITH STATE |

### Time and regime

Let q be elapsed minutes from the 17:00 CT CME session open.

| Feature | Formula | Source and normalization | Missing and boundary rule | Live parity |
|---|---|---|---|---|
| Time of day | sin(2*pi*q/1380), cos(2*pi*q/1380) | Cyclical Chicago-time representation | Always available | LIVE SAFE |
| Minutes since CME open | q | Integer session age | Always available | LIVE SAFE |
| Asia flag | 1 when 17:00 <= CT < 02:00 | Binary | Always available | LIVE SAFE |
| Europe flag | 1 when 02:00 <= CT < 08:30 | Binary | Always available | LIVE SAFE |
| U.S. flag | 1 when 08:30 <= CT < 16:00 | Binary | Always available | LIVE SAFE |
| U.S. RTH flag | 1 when 08:30 <= CT < 15:00 | Binary | Always available | LIVE SAFE |

All regime calculations use timezone-aware America/Chicago timestamps.

## Model 1 formulas

Model 1 contains every Model 0 feature plus only live-reproducible
inferred/tick-rule trade information.

| Feature | Formula | Source and normalization | Missing and boundary rule | Live parity |
|---|---|---|---|---|
| 1-minute inferred delta/volume | delta_1 = D_T / V_T | inferred_delta, total_volume; bounded [-1, 1] | Missing if V_T = 0 | LIVE SAFE WITH STATE |
| 5-minute cumulative inferred delta | delta_5 = sum(D in W_5) / sum(V in W_5) | Inferred delta divided by cumulative volume | Valid W_5 and positive cumulative volume required | LIVE SAFE WITH STATE |
| 15-minute cumulative inferred delta | delta_15 = sum(D in W_15) / sum(V in W_15) | Inferred delta divided by cumulative volume | Valid W_15 and positive cumulative volume required | LIVE SAFE WITH STATE |
| Buy/sell trade-count imbalance | (buy_count - sell_count) / (buy_count + sell_count) | inferred_buy_trade_count, inferred_sell_trade_count; bounded [-1, 1] | Missing if no classified buy/sell trades; unknown trades excluded from denominator | LIVE SAFE WITH STATE |
| Uptick/downtick imbalance | (uptick_count - downtick_count) / (uptick_count + downtick_count) | uptick_count, downtick_count; bounded [-1, 1] | Missing if no price-changing trades | LIVE SAFE WITH STATE |
| Delta acceleration | delta_1 - delta_prev5 | delta_prev5 is sum(D in [T-6m,T-1m)) / sum(V in [T-6m,T-1m)) | Valid preceding five-minute window and positive volume required | LIVE SAFE WITH STATE |
| 5-minute price/delta divergence | DIV_5 = -TE_5 * delta_5 | Trend efficiency and normalized inferred flow; bounded [-1, 1] | TE_5 and delta_5 required | LIVE SAFE WITH STATE |
| 15-minute price/delta divergence | DIV_15 = -TE_15 * delta_15 | Trend efficiency and normalized inferred flow; bounded [-1, 1] | TE_15 and delta_15 required | LIVE SAFE WITH STATE |
| Price movement per volume | See final price-impact formulas | Completed close, S_60, volume baseline | S_60 and b_V(q) required | LIVE SAFE WITH STATE |
| Price movement per absolute inferred delta | See final price-impact formulas | Completed close, S_60, inferred delta, volume baseline | S_60 and b_V(q) required | LIVE SAFE WITH STATE |
| delta_price_alignment | delta_1 * (r_T / S_60) | One-minute normalized inferred delta and signed price move | delta_1 and S_60 required | LIVE SAFE WITH STATE |

For divergence:

- positive values indicate price direction and inferred flow oppose;
- negative values indicate price direction and inferred flow agree;
- values near zero indicate one or both are weak.

## Final price-impact formulas

Define normalized one-minute price movement:

    z_T = abs(r_T) / S_60

Define the positive same-time volume scale:

    B_q = exp(b_V(q))

Define relative total activity:

    a_V = (1 + V_T) / B_q

The final price-movement-per-volume formula is:

    price_movement_per_volume = z_T / (1 + a_V)

Define relative directional effort:

    a_D = abs(D_T) / B_q

The final price-movement-per-absolute-inferred-delta formula is:

    price_movement_per_absolute_inferred_delta = z_T / (1 + a_D)

These are bounded inverse-relative-effort scores. A larger normalized price
move with less relative total activity or inferred directional effort produces
a larger value. The outer one-plus term prevents an unstable divide-by-small
effort ratio without an arbitrary epsilon.

The earlier square-root stabilization formulas are not part of V1.

## Delta/price alignment

The former effort-versus-result name is replaced by:

    delta_price_alignment

Formula:

    delta_price_alignment = delta_1 * (r_T / S_60)

Interpretation:

- Positive: inferred net buying aligns with upward movement, or inferred net
  selling aligns with downward movement.
- Negative: inferred buying accompanies downward movement, or inferred selling
  accompanies upward movement.
- Near zero: directional imbalance is small, normalized price movement is
  small, or both.

This feature is not evidence of absorption, true aggressor behavior, or
unobserved market intent.

## Boundary and missingness rules

- Price-path formulas using G_h are missing when required exact completed
  closes are unavailable.
- W_h aggregation formulas use actual valid sparse trade-bearing minutes, but
  never cross prohibited boundaries.
- No synthetic bar is created.
- No price-derived feature crosses instrument_id.
- Price state resets at a contract roll.
- Long-lookback features may remain missing after closures or reopens while an
  otherwise valid prediction row remains.
- Any later missing-value treatment is fitted on training data only.
- Relative-activity baselines use prior historical observations only.
- Every formula using S_60 is missing when S_60 <= 1e-12.
- All target-layer columns, including Target V2 reference prices, timestamps,
  delays, reference instrument IDs, forward changes/returns, eligibility,
  reasons, and labels, are outcome diagnostics and never Model 0 inputs.

## Pre-model operational clarification — 2026-10-01

The formulas above were frozen before any predictive features or model results.
This clarification fixes session admission for their existing Model 0 families;
it adds no feature, changes no numerical formula, and was recorded before
calculating Model 0 feature values.

### Relative-activity baseline admission

- At a decision boundary T, q is the elapsed minute from that session's 17:00
  CT open. Search strictly earlier sessions in reverse chronological order and
  take the 20 most recent sessions with an actual one-minute row at the same q.
  A session without that row is skipped; never synthesize a zero-volume row.
- The row's full source interval [T_q-1m, T_q) must lie inside the purchased
  DBN request interval, and its observed volume/trade count must be valid.
  Source-boundary partial sessions may contribute q values whose full minute
  is covered. `is_complete_session` is not a general admission filter.
- A known outage or unresolved internal gap excludes an absent or affected q,
  but does not discard an observed q elsewhere in the same session. Scheduled
  early closes and holidays contribute only q values actually observed.
  Databento-degraded warnings do not exclude an otherwise valid observed row.
- A roll session may contribute an observed q: this is a participation baseline,
  not a cross-contract price comparison. The current session, target
  eligibility, target labels, and future sessions never determine membership.
  During validation and holdout, the baseline continues to update only from
  activity observed before each T, as it would in a live stateful calculation.
- If fewer than 20 qualifying prior same-q observations exist, both relative
  activity features are missing. Activity acceleration retains its stated
  separate five-elapsed-minute rule and cannot cross a known gap or closure.

### Immediately prior-session levels

- "Prior session" means the immediately preceding observed CME session_date,
  never the most recent convenient earlier session. If it is not comparable,
  all three prior-session distances are missing for the current decision; do
  not search farther back.
- The prior session must not be source-truncated or contain a known internal
  outage or unresolved missing-minute interval. In this frozen history,
  2025-10-07 and 2026-09-11 are source-boundary partial sessions; 2025-11-28,
  2025-12-24, and 2025-12-30 have disqualifying internal discontinuities.
  A vendor warning alone does not disqualify a session.
- A scheduled early close or holiday is comparable when its available session
  data have no such defect. Do not demand a fixed row count or a 16:00 CT last
  trade. Use the maximum observed high, minimum observed low, and final actual
  completed-minute close of that prior session; missing or nonpositive required
  prices make the levels unavailable.
- Every minute used for those prior levels must have one instrument_id, with
  no contract transition inside the prior session, and that ID must equal the
  current decision's ID. A mixed-contract prior session supplies no whole-
  session levels, even for its new-contract segment. No roll-adjusted price is
  substituted. The prior session must have ended before T.

### U.S. 30-minute opening-range validity

- Use only current-session minutes whose Chicago-local minute_start lies in
  [08:30, 09:00). Require all 30 exact elapsed one-minute slots to have actual
  completed rows. This completeness rule is specific to the opening range;
  it does not change the general sparse W_h aggregation rule.
- If a slot is absent, a known outage or unresolved gap intersects the range,
  source coverage is partial within it, or instrument_id changes inside it,
  the opening-range position is missing. Do not fill or shorten the range.
- The range becomes available at T = 09:00 CT, after [08:59, 09:00) completes.
  At T and later, its contract ID must equal the current decision's ID; a later
  roll invalidates the old range for the new contract. The stated zero-range
  value of 0.5 is retained. A missing opening range leaves the otherwise valid
  prediction observation in place with this feature missing.

## Final pre-implementation Model 0 clarification — 2026-10-01

The following decisions were approved after the read-only Model 0 readiness
audit, before any Model 0 feature value or predictive result was calculated.
They clarify validity and output representation without changing the 36
formulas above or the prior-20, prior-session, and opening-range admission
rules. [Model 0 Feature Contract V1](MODEL0_FEATURE_CONTRACT_V1.json) is the
single machine-readable source for the output schema, purchased UTC coverage,
and known unavailable-interval mask. These rules apply to Model 0 only; they
do not rewrite frozen source or Target V2 artifacts.

### Elapsed-window source coverage and sparse observations

- A valid W_h(T) requires the entire requested [T-hm, T) interval to lie
  inside the acquired UTC source interval in the contract. The preceding-five
  activity comparison independently requires its entire [T-6m, T-1m)
  interval to be source-covered. If any part precedes the acquired start or
  reaches its exclusive end, the corresponding feature is null. A single
  observed minute cannot stand in for an unacquired part of the window.
- A window is null if it intersects an unavailable interval in the contract's
  `unavailable_intervals_utc` mask, a prohibited session or scheduled market
  closure, or a contract transition. The gap mask is authoritative for this
  frozen history; do not duplicate its timestamp literals across feature
  functions. Use half-open interval overlap: a window ending exactly when a
  gap begins, or starting exactly when it ends, does not intersect that gap.
  A window starting exactly at a new contract's first minute may be valid if
  all of its minutes use that contract; a transition inside the window is not.
  G_h paths continue to require every exact elapsed close.
- A fully source-covered window may use its actual sparse trade-bearing rows
  where W_h permits this. A verified no-trade minute is not an unavailable
  source interval and contributes no invented row. Do not forward-fill or
  convert unavailable source data into a zero-volume market minute. In this
  historical minute foundation, the four observed within-session absent-minute
  intervals are the four masked unavailable intervals; a future verified
  zero-trade minute is conceptually different.

### Current-session cumulative price state

- Session VWAP, session high, session low, session-range position, and the
  normalized session VWAP/high/low distances require trustworthy source
  coverage from the nominal 17:00 CT session start, or from the current
  contract's start after a roll in an otherwise clean session, through T.
  Ordinary verified no-trade minutes within covered source do not invalidate
  state and do not add synthetic trades.
- When acquisition begins after the nominal session start, all these
  whole-session cumulative features are null for that entire source-truncated
  session, even though individual completed minutes and shorter covered
  windows may remain valid. In this frozen history this applies to the first
  source-boundary session, 2025-10-07. The final source-boundary session starts
  within acquired coverage and follows the same observation-level rule; do not
  reject it merely because acquisition ends before its nominal session close.
- An unavailable internal gap or outage does not invalidate otherwise valid
  cumulative values before the gap. After the first such interval, these
  whole-session cumulative features are null for every later decision in that
  CME session, including after any later contract change. Do not carry state
  through the gap or restart a post-gap segment under a whole-session name.
  A clean next CME session starts fresh. Scheduled early closes and holidays
  do not themselves invalidate observed, source-covered session state.
- A contract change resets cumulative price state to the new instrument in a
  clean session. Never combine old- and new-contract raw prices. A valid new
  contract segment uses only its own completed minutes; prior old-contract
  state does not enter it. The earlier source-truncation and post-gap
  invalidation rules still take precedence.

### Five-minute activity acceleration at a roll

The current completed minute and every observed comparison minute in
[T-6m, T-1m) must have the same instrument_id. The whole comparison interval
must also avoid a contract transition, even if a no-trade minute occurs at the
transition. If the current minute belongs to the new contract and any part of
the required comparison belongs to the old one, activity acceleration is null.
This does not change the separate rule allowing earlier roll sessions to
contribute to the prior-20 same-q participation baseline.

### Row universe and fixed output schema

- Emit exactly one Model 0 feature row per observed approved one-minute
  decision row: 329,337 rows in this frozen foundation. Preserve source row
  identity and order. Do not use Target V2 eligibility, labels, reference
  fields, or outcome diagnostics to construct or filter predictor rows.
- Keep a row when a predictor is unavailable. Nullable numerical predictors
  use pandas `Float64` with `pd.NA` in memory and Arrow null in the stored
  `float64` field. Do not replace nulls with a learned or fixed value during
  deterministic construction; any later learned missing-value treatment is
  fitted on training history only.
- The contract JSON freezes exactly five ordered identity/lineage columns
  followed by 36 ordered Model 0 predictor columns. It fixes each column's
  name, Arrow type, pandas type, and nullability, plus the physical schema
  version `mes_model0_features_v1_from_1m_event_time_41`. Time sine/cosine,
  integer session age, and four regime flags are non-null. No diagnostic or
  Target V2 column is part of this 41-column predictor-layer schema.
- Target V2 may be joined only later when constructing a controlled modeling
  sample. If validity metadata is useful, write it separately as
  non-predictive audit or manifest information. Model 0 features remain based
  only on the approved one-minute foundation and the known calendar/source
  validity contract.

### Predefined deterministic validation cases

These cases specify assertions for the next implementation pass; they are not
calculated feature results. Historical timestamps below are UTC and precede
the designated model holdout. The implementation audit should check every
feature's exact formula and null propagation in addition to these boundaries.

| Case | Decision boundary or fixture | Required assertion |
|---|---|---|
| Ordinary liquid session | 2026-02-03 15:01 | Exact G_h and W_h timestamp membership, current-minute inclusion, completed opening range, and no event at/after T. |
| Early insufficient history | 2026-02-02 23:10 | Short exact paths may exist; G_15/G_30/G_60 and dependent values are null. |
| First 60-minute path | 2026-02-03 00:00 and 00:01 | G_60 is unavailable one minute before, then uses exactly 61 same-contract closes at 00:01. |
| Three roll transitions | 2025-12-17, 2026-03-18, and 2026-06-17 at 00:01; also 2025-12-17 01:01 | No path or price state combines contracts; the first new-contract minute lacks old-contract returns, and a new G_60 first becomes possible after 60 exact minutes. |
| First post-roll activity | Each roll's 00:01 decision and a later fully new-contract comparison | Acceleration is null when its comparison uses old-contract time; it becomes eligible only with a covered, same-contract preceding five-minute interval. |
| Confirmed outage | 2025-11-28 02:45 and 13:31 | Pre-outage cumulative state may exist; post-outage whole-session state is null, and windows/paths do not bridge the masked interval. |
| Isolated December gaps | 2025-12-24 07:53 and 07:57; 2025-12-30 06:47 | Every path/window intersecting the respective mask is null; no missing minute is filled. |
| Scheduled early close | 2025-11-27 18:00 | The last observed boundary remains valid; no fixed session row count or ordinary 16:00 CT close is required. |
| Opening-range release | 2026-02-03 14:59, 15:00, and 15:01 | Unavailable before 09:00 CT, available at 09:00 only after all 30 slots, retained later only for the same contract. |
| Invalid immediately prior session | 2025-12-01 15:00 and 2025-12-18 15:00 | Prior-session distances are null because the immediately prior session has an outage or an internal roll; do not skip backward. |
| Causal activity baseline | 2025-11-03 and 2025-11-04 at 15:00 | At q=960, 19 prior same-q rows leave both relative features null; exactly 20 permit the frozen medians. |
| First source-partial session | 2025-10-07 00:01 | A window reaching before acquisition is null, and whole-session cumulative state stays null for this session. |
| Source-covered sparse versus unavailable | Synthetic two otherwise identical windows, one with a verified zero-trade minute and one intersecting a masked unavailable minute | The former uses observed sparse rows where W_h allows; the latter is null. Neither synthesizes a bar. |
| Roll after opening range | Synthetic same-session transition after 09:00 CT | The old range does not transfer to the new contract. |
| Zero and near-zero denominators | Synthetic flat and near-flat price/size cases | Trend efficiency uses 0 for exactly zero travel; zero session/opening range uses 0.5; sigma_60 and S_60 thresholds propagate null exactly as specified. |
| Opening-range breakout | Synthetic OR_low=100, OR_high=101, C_T=102 | Position is 2, not clipped to 1. |

These tests must not use Target V2 eligibility or outcomes to decide feature
validity. The production feature layer is not built by this clarification.

## Live parity

All non-time V1 features are **LIVE SAFE WITH STATE**. Time and regime
features are **LIVE SAFE**.

Required live state includes:

- completed-minute aggregation;
- trailing price and return history;
- session VWAP, high, and low state;
- prior-session state;
- rolling inferred-flow state;
- causal time-of-day activity baselines;
- tick-rule state; and
- contract identity.

Native Databento aggressor fields remain diagnostic only. They are never Model
0 or Model 1 features.

## Redundancy notes

Potential redundancy remains intentionally visible in V1:

- nested return horizons;
- sigma_15, sigma_60, and their ratio;
- volume, trade count, and average trade size;
- session-range position and high/low distances;
- the 1-, 5-, and 15-minute delta family;
- trend efficiency, delta, and divergence interactions;
- price movement per volume, price movement per absolute inferred delta, and
  delta_price_alignment;
- cyclical time, minutes since open, and regime flags.

All remain in V1. No feature is removed based on predictive results during
initial implementation.

## Explicit exclusions

V1 does not add:

- RSI, MACD, stochastic oscillators, broad moving-average libraries,
  Bollinger variants, Fibonacci levels, candlestick labels, or large
  technical-indicator libraries;
- hand-labeled trading setups or arbitrary extra horizons;
- quote/depth features, resting liquidity, cancellations, replenishment, or
  L2 data;
- native Databento aggressor predictors; or
- approximate volume-at-price features.

## Implementation status

Feature Formula Specification V1 and the dated admission clarification above
are frozen before Model 0 feature calculation and predictive modeling.

Next steps:

Target V2 corrects a roll-boundary endpoint defect in preserved Target V1.
The next step is to implement and independently validate Model 0 features.
Model 1 features, chronological development folds, and predictive training
remain later steps.
