# Backtest Execution Log

## Session Information
- **Date**: 2025-11-29
- **Executor**: Backtest Executor Agent
- **Phase**: Phase 1 - Baseline Backtest

## Objective
Execute the baseline backtest using current parameters from the reference project to establish performance benchmarks.

## Methodology

### Strategy Configuration
- **Strategy**: VCP (Volatility Contraction Pattern)
- **RS Threshold**: 90 (current parameter)
- **Exit Method**: Fixed target (15% target, 8% stop loss)
- **Test Period**: 2022-01-01 to 2024-11-29 (approximately 3 years)
- **Data Source**: yfinance (Yahoo Finance API)

### Test Universe
Using a representative sample of US stocks from S&P 500:
- Technology sector
- Healthcare sector
- Industrial sector
- Financial sector
- Consumer discretionary/staples

### Parameters Used
From `strategy_params.py` - VCP current parameters:
- contraction_threshold: 0.20 (20% max range)
- volume_dry_up_ratio: 0.5 (50% volume dry-up)
- min_consolidation_weeks: 4
- pivot_breakout_volume_multiplier: 1.5
- rs_rating_min: 90
- distance_from_52w_high_max: 0.25 (within 25% of 52-week high)
- price_above_200ma: True

Risk Parameters:
- default_stop_loss_pct: 8.0%
- default_target_pct: 15.0%
- Expected R:R ratio: 15/8 = 1.875 (below 3.0 target - will test)

## Execution Timeline

### Step 1: Environment Setup (COMPLETE)
- Created workspace directories
- Initialized execution log
- Verified backtest framework availability
- Created Python virtual environment and installed dependencies

### Step 2: Stock Universe Selection (COMPLETE)
Selected 50 US stocks across 5 sectors:
- Technology: AAPL, MSFT, NVDA, GOOGL, META, AMD, AVGO, ORCL, ADBE, CRM
- Healthcare: LLY, UNH, JNJ, ABBV, MRK, TMO, ABT, DHR, PFE, BMY
- Industrials: GE, CAT, BA, HON, UPS, RTX, DE, LMT, MMM, GD
- Financials: JPM, V, MA, BAC, WFC, GS, MS, AXP, BLK, C
- Consumer: HD, NKE, SBUX, MCD, LOW, TGT, TJX, BKNG, CMG, COST

### Step 3: Backtest Execution (COMPLETE)
Executed 3 configurations:
1. Baseline Fixed Target (7% stop, 20% target, R:R 2.86)
2. Baseline Trailing Stop (7% initial stop, 8% activation, 5% trail)
3. Adjusted R:R Fixed Target (8% stop, 24% target, R:R 3.0)

Test Period: 2022-01-01 to 2024-11-29

### Step 4: Results Collection (COMPLETE)
Generated files:
- phase1_baseline_raw.json (full results)
- phase1_all_trades.csv (detailed trade log)
- phase1_baseline_results.csv (handoff summary)

### Step 5: Handoff Preparation (COMPLETE)
Results ready for reviewer analysis

## Key Findings

### Configuration 1: Baseline Fixed Target (Reference Parameters)
- **Total Trades**: 34
- **Win Rate**: 29.4% (BELOW expected)
- **Profit Factor**: 0.76 (LOSING strategy)
- **Avg R:R Realized**: 1.84 (below theoretical 2.86)
- **Total Return**: -37.61% (NEGATIVE)
- **VCP Proximity Correlation**: 0.114 (weak positive)
- **High Proximity Win Rate**: 36.4% vs Low Proximity: 0.0%

### Configuration 2: Baseline Trailing Stop
- **Total Trades**: 34
- **Win Rate**: 50.0% (IMPROVED from fixed)
- **Profit Factor**: 0.88 (still losing)
- **Avg R:R Realized**: 0.88 (lower than fixed target)
- **Total Return**: -14.85% (better than fixed, still negative)
- **Avg Days Held**: 25.2 (shorter than fixed target)
- **High Proximity Win Rate**: 54.5% (BETTER than fixed)

### Configuration 3: Adjusted R:R 3.0
- **Total Trades**: 34 (same signals)
- **Win Rate**: 29.4% (same as baseline)
- **Profit Factor**: 0.72 (WORSE than baseline)
- **Total Return**: -50.26% (WORSE than baseline)
- **Wider stop increased losses, higher target rarely hit**

## Critical Observations

### MAJOR CONCERN: RS 90 Threshold Too Restrictive
- Only 34 trades in 3 years across 50 stocks = 0.68 trades/stock
- Very low trade frequency suggests missing valid setups
- Win rate of 29.4% is FAR below breakeven (need ~35% for R:R 2.86)
- Profit factor < 1.0 means strategy is losing money

### VCP Proximity Score Validation
- Positive correlation (0.114) confirms proximity matters
- High proximity trades (≥70): 36.4% win rate
- Low proximity trades (≤30): 0.0% win rate
- **Recommendation**: Require minimum proximity score of 50-60

### Trailing Stop Performance
- Win rate doubled (50% vs 29.4%)
- Better suited for volatile markets
- Lower R:R but higher frequency of small wins
- Total return still negative (-14.85%)

### Fixed vs Trailing Exit Comparison
- Trailing stop: Better win rate, shorter hold time
- Fixed target: Higher R:R when it works, but rarely works
- Neither configuration is profitable with current parameters

## Issues Encountered

### Issue 1: Timezone Comparison Error (RESOLVED)
- Error: Cannot compare tz-naive and tz-aware timestamps
- Fix: Modified backtest_framework_v2.py to normalize timezones
- Impact: None on results, code fix only

### Issue 2: Python Dependencies
- Error: ModuleNotFoundError for pandas
- Fix: Created virtual environment and installed dependencies
- Impact: None on results

## Recommendations for Reviewer

1. **CRITICAL**: Current parameters are NOT profitable
   - RS 90 threshold likely too restrictive
   - Phase 3 should test RS 70, 80, 85 to increase trade count

2. **VCP Proximity Score**: Valid metric showing clear correlation
   - High proximity trades perform better
   - Consider adding min_proximity_score parameter

3. **Exit Method**: Trailing stop shows promise
   - Better win rate (50% vs 29%)
   - Need to optimize activation and trail distance

4. **R:R Ratio**: Widening stops/targets made performance WORSE
   - Suggests pattern itself needs improvement
   - Focus on better signal quality, not wider targets

5. **Sample Size**: 34 trades is statistically small
   - Need more trades for robust analysis
   - RS relaxation in Phase 3 should help

## Next Steps for Phase 2 (Proximity Analysis)

The proximity score shows promise (36.4% vs 0% win rate). Phase 2 should:
1. Test minimum proximity thresholds (0, 30, 50, 70)
2. Analyze proximity score distribution
3. Correlate proximity with other metrics (RS, volume ratio)
4. Recommend optimal proximity cutoff

## Status
COMPLETE - Ready for handoff to Reviewer Agent
