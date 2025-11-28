# Strategy Analysis and Backtest Plan

## AI Trading Assistant - Research Paper

**Author:** Claude (AI Assistant)
**Date:** 2025-11-28
**Version:** 1.0

---

## Executive Summary

This paper analyzes the three primary trading strategies extracted from the reference project: VCP (Volatility Contraction Pattern), Pivot Breakout, and Cup with Handle. These strategies are rooted in the methodologies of Mark Minervini and William O'Neil, focusing on momentum-based breakout trading with strict risk management.

My analysis identifies several areas where the current parameters may be either too restrictive (missing valid setups) or insufficiently selective (generating false signals). The proposed backtest will evaluate these hypotheses and provide data-driven recommendations for optimization.

---

## Part 1: Strategy Deep Dive

### 1.1 VCP (Volatility Contraction Pattern) Strategy

#### Concept Origin

The VCP strategy was popularized by Mark Minervini, a U.S. Investing Champion who achieved a 155% return in the 1997 competition. The core insight is that winning stocks exhibit a specific pattern before major price advances: **volatility contracts** while price consolidates near highs, followed by a breakout on increased volume.

#### Theoretical Foundation

The VCP pattern reflects institutional accumulation behavior:

1. **Initial Decline**: After an advance, institutions begin accumulating, creating natural pullbacks
2. **Contraction Sequence**: Each pullback is shallower than the previous (e.g., 25% → 15% → 10%)
3. **Volume Dry-Up**: Selling pressure exhausts, volume declines significantly
4. **Pivot Point**: Price consolidates in a tight range near the highs
5. **Breakout**: When supply is absorbed, buyers overwhelm sellers, price breaks out on volume

#### Current Implementation Analysis

```
Current Parameters:
- contraction_threshold: 0.20 (20% max range)
- volume_dry_up_ratio: 0.5 (volume must be < 50% of average)
- min_consolidation_weeks: 4
- pivot_breakout_volume_multiplier: 1.5
- rs_rating_min: 90 (top 10% performers)
- distance_from_52w_high_max: 0.25 (within 25% of 52-week high)
```

#### My Observations and Concerns

**1. RS Rating Threshold (90) is Very Strict**

The current requirement of RS ≥ 90 means only the top 10% of stocks qualify. While this aligns with Minervini's emphasis on relative strength, it may be:
- **Too restrictive** in sideways markets where fewer stocks outperform
- **Self-fulfilling but late**: By the time RS reaches 90, significant gains may have occurred

*Hypothesis*: Lowering to RS ≥ 85 could increase trade frequency without significantly degrading win rate.

**2. Contraction Threshold May Be Too Tight**

A 20% maximum range in the consolidation period is quite tight. In my understanding:
- Classic VCP shows **multiple contractions**, each tighter
- The reference implementation checks only the final period, not the contraction sequence
- 20% works in stable markets but may miss valid setups in higher volatility environments

*Hypothesis*: Testing 25% and 30% thresholds could capture valid patterns while maintaining quality.

**3. Volume Dry-Up Requirement**

The 50% volume dry-up ratio is aggressive. This means recent volume must be less than half the 50-day average. While this correctly identifies institutional absorption, it may:
- Filter out stocks in early breakout stages
- Miss patterns in highly liquid stocks where volume doesn't contract as dramatically

*Hypothesis*: Testing 0.6 and 0.7 ratios could improve trade frequency.

**4. Single Contraction Check vs. Sequential Analysis**

The current implementation checks if the last N weeks have < 20% range. True VCP requires **sequential contractions** (e.g., T1 > T2 > T3 where T is the trading range). This is a significant gap that could generate false positives.

*Recommendation*: Enhance the algorithm to detect at least 2 contractions with decreasing magnitude.

---

### 1.2 Pivot Breakout Strategy

#### Concept Origin

The pivot breakout strategy combines O'Neil's CAN SLIM methodology with Minervini's trend template. A "pivot point" is the optimal buy point where a stock breaks out of a proper base formation.

#### Theoretical Foundation

Proper bases form when:
1. Stock advances significantly (100%+ in leaders)
2. Normal profit-taking causes a correction (15-35%)
3. Weak hands sell, strong hands accumulate
4. Stock forms a "shelf" near highs (the pivot)
5. Breakout above pivot signals new leg up

#### Current Implementation Analysis

```
Current Parameters:
- min_base_weeks: 4
- max_base_weeks: 52
- max_base_depth: 0.35 (35% max correction)
- min_base_depth: 0.08 (8% min correction)
- pivot_volume_multiplier: 1.5
- handle_max_depth: 0.12 (12%)
- rs_rating_min: 80
- distance_from_52w_high_max: 0.30
- price_above_50ma: true
- price_above_200ma: true
```

#### My Observations and Concerns

**1. Base Depth Range (8-35%) is Reasonable**

This aligns well with O'Neil's guidelines:
- Bases < 8% are too shallow (not enough reset)
- Bases > 35% suggest structural problems

However, in bull markets, bases are often shallower (10-20%). In bear markets, deeper bases (30-40%) are common.

*Hypothesis*: Consider market regime detection to dynamically adjust acceptable base depth.

**2. RS Threshold (80) is More Permissive Than VCP**

The lower RS requirement (80 vs 90) increases trade universe but may include lower-quality setups. The trade-off is more signals vs higher risk.

*Hypothesis*: Test RS = 85 as a middle ground.

**3. Base Detection Algorithm is Simplified**

The current algorithm:
- Scans backward to find the first valid base depth
- Checks if current price is near the base high (pivot)
- Confirms volume on breakout

What's missing:
- **Base count tracking**: How many bases has the stock formed? (1st base > 2nd > 3rd in reliability)
- **Base pattern recognition**: Is it flat, ascending, or saucer-shaped?
- **Prior advance verification**: Did the stock advance 30%+ before forming this base?

*Recommendation*: Add prior advance requirement (stock should have risen significantly before base formation).

**4. Handle Detection**

The handle detection is present but optional. A proper handle should:
- Form in the upper half of the base
- Have declining volume
- Depth < 12% (implemented)
- Last 1-4 weeks (partially implemented)

The current check could be enhanced to verify handle position within the base.

---

### 1.3 Cup with Handle Strategy

#### Concept Origin

Pioneered by William O'Neil, the Cup with Handle is one of the most reliable chart patterns. O'Neil found that stocks forming this pattern before major advances had a high success rate.

#### Theoretical Foundation

The pattern resembles a teacup:
1. **Left Side**: Price declines 12-33% from highs
2. **Bottom**: U-shaped (not V-shaped) consolidation
3. **Right Side**: Price recovers to near the left side high
4. **Handle**: Small pullback (5-12%) that shakes out weak holders
5. **Breakout**: Price breaks above handle high on volume

The U-shape is critical: it indicates gradual sentiment shift, not panic selling/buying.

#### Current Implementation Analysis

```
Current Parameters:
- min_cup_weeks: 7
- max_cup_weeks: 65
- min_cup_depth: 0.12 (12%)
- max_cup_depth: 0.33 (33%)
- handle_max_depth: 0.12
- u_shape_position: low must be between 30-70% of cup duration
- volume_multiplier: 1.5
```

#### My Observations and Concerns

**1. U-Shape Detection is Simplistic**

Checking that the low is between 30-70% of cup duration is a proxy for U-shape, but doesn't verify:
- Gradual decline (left side)
- Rounded bottom
- Gradual recovery (right side)

V-shaped patterns can still pass this check if the spike low happens to be in the middle.

*Recommendation*: Add symmetry checks (left decline rate ≈ right recovery rate).

**2. Handle Formation Validation**

The current handle check is basic. Missing:
- Volume should decline during handle formation
- Handle should form in upper half of the pattern
- Handle duration should be 1-4 weeks

**3. Duration Range is Very Wide**

7-65 weeks covers a huge range. Consider:
- 7-12 weeks: Short cup (higher risk, may not be fully formed)
- 13-30 weeks: Classic cup (optimal)
- 30-65 weeks: Very long cup (may indicate distribution, not accumulation)

*Hypothesis*: Weight signals by cup duration, preferring 13-30 week patterns.

---

### 1.4 Relative Strength (RS) Calculation

#### TradingView Implementation

The reference project uses the TradingView/IBD methodology:

1. Calculate price performance ratios for 4 periods (63, 126, 189, 252 trading days)
2. Apply weights: 40% recent (63d), 20% each for others
3. Compare to SPY benchmark
4. Map the ratio to a 1-99 percentile scale using calibrated thresholds

#### My Observations

**Strengths:**
- Uses ratio-based comparison (proper relative strength)
- Weighted toward recent performance (correct - recent momentum matters more)
- Calibrated against 6,600 stocks (TradingView data)

**Potential Issues:**
- **Calendar vs Trading Days**: The implementation mixes calendar days (in data fetching) with trading day concepts (in the formula). This could cause slight misalignment.
- **Benchmark Selection**: Using SPY is correct for US stocks, but the reference uses EWH (Hong Kong ETF) for HK stocks and EWJ for Japan. These regional ETFs have different characteristics than broad market indices.
- **Staleness**: RS is calculated point-in-time but may be cached for hours. In fast-moving markets, this could miss opportunities.

---

### 1.5 Risk Management Parameters

```
Current Parameters:
- max_risk_per_trade_pct: 2.0%
- default_stop_loss_pct: 7.0%
- default_target_pct: 20.0%
- reward_risk_ratio_min: 3.0
- trailing_stop_activation_pct: 8.0%
```

#### Analysis

**Stop Loss at 7%:**
- Reasonable for most setups
- May be too tight for volatile stocks (frequent stop-outs)
- May be too loose for tight VCP patterns (should fail quickly if wrong)

*Hypothesis*: Stop loss should be pattern-dependent:
- VCP with tight contraction: 5-6% stop
- Pivot from deep base: 8-10% stop

**Target at 20%:**
- Conservative but achievable
- Minervini advocates for letting winners run
- Fixed targets may cut winners too early

*Hypothesis*: Consider trailing stops or scaled exits instead of fixed targets.

**R:R Ratio of 3.0:**
- With 7% stop and 20% target: R:R = 20/7 = 2.86 (close to 3.0)
- This requires ~26% win rate to break even
- Achievable but demanding

---

## Part 2: Backtest Methodology

### 2.1 Objectives

1. **Validate current parameters**: Measure win rate, profit factor, and R:R ratio
2. **Identify optimization opportunities**: Test parameter variations
3. **Compare strategies**: VCP vs Pivot vs Cup with Handle
4. **Risk parameter optimization**: Find optimal stop/target combinations

### 2.2 Test Universe

I've selected 60 US stocks across sectors:
- **Tech** (20): AAPL, MSFT, NVDA, GOOGL, META, etc.
- **Healthcare** (10): LLY, UNH, JNJ, etc.
- **Industrials** (10): GE, CAT, BA, etc.
- **Financials** (10): JPM, V, MA, etc.
- **Consumer** (10): HD, NKE, SBUX, etc.

This provides diversity while remaining computationally tractable.

### 2.3 Test Period

**Primary Period**: 2023-01-01 to 2025-11-28 (approximately 2 years)

This captures:
- 2023 recovery rally (bull market)
- 2024 consolidation and AI boom
- Various market regimes

**Data Requirement**: 2 years of daily OHLCV data per symbol

### 2.4 Signal Generation Process

For each trading day in the test period:
1. Calculate indicators (MAs, volume averages)
2. Check pattern criteria (VCP, Pivot, or Cup)
3. Verify RS rating threshold
4. Check breakout conditions (price above pivot, volume > threshold)
5. If all criteria met, generate entry signal

### 2.5 Trade Simulation

For each signal:
1. **Entry**: At closing price on signal day
2. **Stop Loss**: Fixed percentage below entry
3. **Target**: Fixed percentage above entry
4. **Exit Conditions**:
   - Stop hit (price touches stop level)
   - Target hit (price touches target level)
   - Time exit (60 days max holding period)

### 2.6 Metrics Calculated

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Win Rate | Wins / Total Closed | % of profitable trades |
| Avg Win | Mean(winning P&L %) | Average gain on winners |
| Avg Loss | Mean(losing P&L %) | Average loss on losers |
| Profit Factor | Total Wins / Total Losses | > 1.5 is good, > 2.0 is excellent |
| Avg R:R Realized | Avg Win / |Avg Loss| | Should exceed theoretical R:R |
| Total Return | Sum of all P&L % | Cumulative performance |
| Max Drawdown | Worst peak-to-trough | Risk measure |
| Avg Days Held | Mean holding period | Trade duration |

### 2.7 Parameter Optimization Tests

#### Phase 1: Current Parameters (Baseline)

Run VCP, Pivot, and Cup strategies with current settings to establish baseline.

#### Phase 2: RS Rating Sensitivity

Test RS thresholds: 70, 75, 80, 85, 90, 95

Expected outcome: Lower RS = more trades, potentially lower win rate

#### Phase 3: VCP-Specific Parameters

| Parameter | Test Values | Rationale |
|-----------|-------------|-----------|
| contraction_threshold | 0.10, 0.15, 0.20, 0.25, 0.30 | Find optimal tightness |
| volume_dry_up_ratio | 0.3, 0.4, 0.5, 0.6, 0.7 | Volume sensitivity |
| min_consolidation_weeks | 3, 4, 5, 6, 8 | Pattern maturity |

#### Phase 4: Pivot-Specific Parameters

| Parameter | Test Values | Rationale |
|-----------|-------------|-----------|
| max_base_depth | 0.25, 0.30, 0.35, 0.40 | Acceptable correction |
| min_base_depth | 0.05, 0.08, 0.10, 0.12 | Minimum reset |
| pivot_volume_multiplier | 1.2, 1.5, 1.8, 2.0 | Breakout conviction |

#### Phase 5: Risk Parameter Optimization

| Stop Loss | Target | Expected R:R |
|-----------|--------|--------------|
| 5% | 15% | 3.0 |
| 5% | 20% | 4.0 |
| 6% | 18% | 3.0 |
| 7% | 20% | 2.86 |
| 7% | 25% | 3.57 |
| 8% | 20% | 2.5 |
| 8% | 25% | 3.13 |
| 10% | 30% | 3.0 |

### 2.8 Expected Outcomes and Hypotheses

#### Hypothesis 1: RS Relaxation

**Expectation**: Lowering RS from 90 to 85 will:
- Increase trade count by 30-50%
- Decrease win rate by 5-10%
- Net effect: Similar or better total return due to volume

#### Hypothesis 2: VCP Contraction Relaxation

**Expectation**: Increasing contraction threshold from 20% to 25% will:
- Increase trade count by 20-40%
- Slightly decrease win rate (5%)
- Improve profit factor due to catching more valid patterns

#### Hypothesis 3: Tighter Stops for VCP

**Expectation**: Using 5-6% stops instead of 7% for tight VCP patterns will:
- Decrease avg loss magnitude
- May increase stop-out frequency
- Net effect: Better R:R realized if pattern is valid

#### Hypothesis 4: Pivot Strategy Outperforms VCP

**Expectation**: Pivot strategy will show:
- Higher trade count (less restrictive)
- Lower win rate
- Similar profit factor
- Better for active traders

---

## Part 3: Limitations and Caveats

### 3.1 Backtest Limitations

1. **No Slippage**: We assume execution at exact prices
2. **No Transaction Costs**: Commissions and fees not included
3. **Survivorship Bias**: Test universe contains currently traded stocks
4. **Look-Ahead Bias**: Using 2-year data that we can see today
5. **Simplified Exit**: Real trading involves discretionary decisions

### 3.2 Data Quality Issues

1. **Adjusted Prices**: Using adjusted close may affect percentage calculations
2. **Volume Data**: Some sources have unreliable volume data
3. **Market Hours**: After-hours moves not captured in daily OHLCV

### 3.3 Market Regime Dependency

The test period (2023-2025) was generally bullish. Results may not generalize to:
- Bear markets
- High volatility regimes
- Sector rotations

---

## Part 4: Recommendations Before Backtesting

### 4.1 Immediate Recommendations

Before running the backtest, I recommend considering:

1. **Add Sequential Contraction Check to VCP**
   - Current: Single period range check
   - Proposed: Verify at least 2 contractions with T1 > T2

2. **Add Prior Advance Requirement to Pivot**
   - Current: No check for prior move
   - Proposed: Stock should have risen 30%+ before base formation

3. **Market Regime Filter (Optional)**
   - If SPY is below 200-day MA, tighten all parameters
   - If VIX > 25, reduce position sizes or pause new entries

### 4.2 Questions for Your Consideration

1. **Trade Frequency vs Quality**: Do you prefer fewer, higher-quality signals or more frequent trading opportunities?

2. **Risk Tolerance**: Is the 7% stop loss acceptable, or would you prefer tighter/looser stops?

3. **Target Approach**: Fixed target (20%) or trailing stop after +8%?

4. **Market Scope**: Should we backtest HK and JP stocks as well, or focus on US only?

---

## Conclusion

The strategies in the reference project are well-founded in proven trading methodologies. However, my analysis suggests several areas for potential improvement:

1. **RS threshold** may be too strict for current market conditions
2. **VCP detection** lacks sequential contraction verification
3. **Pivot detection** would benefit from prior advance confirmation
4. **Risk parameters** could be pattern-dependent rather than fixed

The proposed backtest will provide empirical data to validate or refute these hypotheses. I recommend proceeding with the backtest while keeping these observations in mind for interpretation of results.

---

## Appendix: Key Terminology

| Term | Definition |
|------|------------|
| VCP | Volatility Contraction Pattern - price range narrows over time |
| Pivot | Price level where stock breaks out of base formation |
| RS Rating | Relative Strength - stock performance vs benchmark (0-99) |
| Profit Factor | Gross profits / Gross losses |
| R:R Ratio | Reward to Risk ratio (target / stop distance) |
| Base | Consolidation period after price advance |
| Handle | Small pullback within or after a cup pattern |
| Contraction | Decrease in trading range over time |
| Volume Dry-Up | Significant decrease in trading volume |

---

*End of Research Paper*
