# VCP Entry Method Comparison Results

**Date:** 2025-11-30
**Backtest Period:** 2020-01-01 to 2024-12-31
**Universe:** 66 US Stocks (S&P 500, NASDAQ 100 leaders)

## Executive Summary

This backtest compares four VCP entry methods from Minervini's methodology:
1. **Pivot Breakout** - Traditional entry above the pivot point
2. **Cheat Entry** - Entry in middle third of base on tight range breakout
3. **Low Cheat Entry** - Entry in lower third of base (aggressive)
4. **Handle Entry** - Entry after a small pullback near the pivot

## Results Summary

| Entry Method | Trades | Win Rate | Profit Factor | Total Return | Avg Risk | Avg Days |
|-------------|--------|----------|---------------|--------------|----------|----------|
| **Pivot Only** | 33 | 66.7% | **2.53** | 112.5% | 15.4% | 27.4 |
| **Cheat Only** | 16 | 43.8% | 0.85 | -6.4% | 4.9% | 10.2 |
| **Low Cheat Only** | 15 | 66.7% | 1.92 | 34.2% | 7.6% | 17.7 |
| **Handle Only** | 80 | **70.0%** | 1.58 | 141.6% | 15.4% | 26.0 |
| **All Entries** | 121 | 67.8% | 1.76 | **252.7%** | 13.4% | 23.7 |
| **Cheat + Low Cheat** | 30 | 56.7% | 1.41 | 31.1% | 6.3% | 14.0 |

## Key Findings

### 1. Pivot Breakout Has Highest Profit Factor (2.53)
- Traditional pivot breakout remains the most reliable entry method
- Best risk-adjusted returns despite fewer opportunities
- Higher risk per trade (15.4%) but compensated by strong win/loss ratio

### 2. Handle Entry Provides Most Opportunities (80 trades)
- Highest win rate at 70%
- Second highest total return (141.6%)
- Good balance of frequency and reliability

### 3. Cheat Entry Underperforms (-6.4% total return)
- Lowest win rate (43.8%)
- Only entry method with profit factor below 1.0
- Tight range entries in middle third may trigger too early
- May need stricter criteria or different market conditions

### 4. Low Cheat Shows Promise Despite Limited Sample
- Strong win rate (66.7%) matching pivot breakout
- Lower risk per trade (7.6%) compared to pivot (15.4%)
- Profit factor of 1.92 is second best
- Fewer opportunities (15 trades) limits statistical significance

### 5. Combined Approach Maximizes Total Return
- "All Entries" configuration achieves 252.7% total return
- 121 trades provide better diversification
- Maintains 67.8% win rate and 1.76 profit factor
- Best overall balance of frequency and performance

## Entry Method Characteristics

### Pivot Breakout
- **Position:** Upper third of base
- **Trigger:** Price breaks above pivot with 1.5x volume
- **Risk:** Wider stop (below base low)
- **Best for:** Higher conviction setups with strong volume confirmation

### Cheat Entry
- **Position:** Middle third of base
- **Trigger:** Tight range (<5%) breakout with 1.2x volume
- **Risk:** Tighter stop (below recent swing low)
- **Best for:** Anticipating breakout with reduced risk
- **Caution:** May trigger in weak patterns that fail to break out

### Low Cheat Entry
- **Position:** Lower third of base
- **Trigger:** Tight range or shakeout recovery above 200 MA
- **Risk:** Tight stop below recent low
- **Best for:** Maximum R:R ratio when pattern recovers from shakeout
- **Caution:** Higher failure rate in choppy markets

### Handle Entry
- **Position:** Upper third (after pullback)
- **Trigger:** 3-12% pullback from near-pivot, then recovery
- **Risk:** Below handle low
- **Best for:** Secondary entry after initial pivot approach fails

## Risk Analysis

| Entry Method | Avg Risk | Avg Win | Avg Loss | Win/Loss Ratio |
|-------------|----------|---------|----------|----------------|
| Pivot Only | 15.4% | +8.5% | -6.7% | 1.27 |
| Cheat Only | 4.9% | +5.0% | -4.6% | 1.09 |
| Low Cheat Only | 7.6% | +7.1% | -7.4% | 0.96 |
| Handle Only | 15.4% | +6.9% | -10.2% | 0.68 |
| All Entries | 13.4% | +7.1% | -8.5% | 0.84 |
| Cheat + Low Cheat | 6.3% | +6.3% | -5.8% | 1.09 |

**Observations:**
- Cheat entries have lowest risk (4.9-7.6%) but also lower reward
- Pivot breakout has best win/loss ratio (1.27)
- Handle entries have largest average losses (-10.2%)

## Recommendations

### For Conservative Traders
Use **Pivot Breakout Only**:
- Highest profit factor (2.53)
- Best risk/reward characteristics
- Fewer trades but higher quality

### For Aggressive Traders
Use **All Entries** with priority:
1. Low Cheat (when available in lower third)
2. Cheat (when in middle third)
3. Handle (after pullback)
4. Pivot Breakout (traditional)

### Avoid
- **Cheat Only** strategy - underperforms with 0.85 profit factor
- Cheat entries may need additional filters:
  - Market regime (bull market only)
  - Higher RS threshold
  - Stronger volume confirmation

## Sample Charts

Sample charts for each entry method are available in:
- `results/entry_comparison/pivot_only_charts/`
- `results/entry_comparison/cheat_only_charts/`
- `results/entry_comparison/low_cheat_only_charts/`
- `results/entry_comparison/handle_only_charts/`
- `results/entry_comparison/all_entries_charts/`
- `results/entry_comparison/cheat_and_low_cheat_charts/`

Each folder contains 10 sample charts (5 winners, 5 losers) showing:
- VCP pattern with contractions marked
- Entry type and position in base
- Entry/exit points with stop loss and target
- Prior uptrend percentage
- Overall pattern score

## Technical Implementation

### V5 Detector Entry Logic

```python
# Entry priority (earlier = better R:R)
priority = {
    EntryType.LOW_CHEAT: 1,    # Best R:R, most aggressive
    EntryType.CHEAT: 2,        # Good R:R, less aggressive
    EntryType.HANDLE: 3,       # After pullback
    EntryType.PIVOT_BREAKOUT: 4 # Traditional, highest conviction
}

# Base zone calculation
base_low = min(contraction lows)
base_high = pivot_price
lower_third = base_low + (base_high - base_low) * 0.33
middle_third = base_low + (base_high - base_low) * 0.67
```

### Entry Criteria

**Cheat Entry:**
- Price in middle third of base
- Tight daily range (<5%)
- Bullish candle (close > open)
- Volume > 1.2x 50-day average
- Close in upper 60% of day's range

**Low Cheat Entry:**
- Price in lower third of base
- Tight range OR shakeout recovery
- Price above 200 MA
- Volume > 1.2x average

**Handle Entry:**
- Price near pivot (within 3%)
- Pullback 3-12% from recent high
- Handle duration max 15 days
- Recovery above handle high

## Files Created

- `backtest/vcp_detector_v5.py` - V5 detector with entry type detection
- `backtest/run_backtest_entry_comparison.py` - Comparison backtest script
- `backtest/regenerate_entry_charts.py` - Chart generation utility
- `results/entry_comparison/comparison_summary.csv` - Results summary
- `results/entry_comparison/*_trades.csv` - Trade details for each config

## Next Steps

1. **Refine Cheat Entry Criteria** - Add market regime filter
2. **Test Longer History** - Extend to 2015-2024 for more samples
3. **Add RS Filter** - Require RS > 80 for cheat entries
4. **Volume Threshold Tuning** - Test 1.3x and 1.5x for cheat entries
5. **Market Regime Filter** - Only trade when SPY > 200 MA
