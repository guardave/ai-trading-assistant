# VCP Backtest V3 Results

**Date:** 2025-11-30
**Backtest Period:** 2023-01-01 to present
**Universe:** Top 100 S&P 500 stocks

## Overview

This backtest uses the refined VCP detection algorithm (`vcp_detector.py`) which includes:
- Proper swing high/low detection (5-bar lookback)
- "Extend until recovery" contraction identification
- Progressive tightening enforcement
- Consolidation base rule (rejects staircase patterns)
- Volume dry-up validation

## Test Configurations

| Test | RS Min | Proximity Min | Exit Method | Stop/Target |
|------|--------|---------------|-------------|-------------|
| 1 | 70 | 0 | Trailing Stop | 7% / 8%+5% trail |
| 2 | 70 | 50 | Trailing Stop | 7% / 8%+5% trail |
| 3 | 80 | 0 | Trailing Stop | 7% / 8%+5% trail |
| 4 | 80 | 50 | Trailing Stop | 7% / 8%+5% trail |
| 5 | 70 | 50 | Fixed Target | 7% / 21% |

## Results Summary

### Initial Run (Loose Breakout Criteria)

First run used minimal breakout confirmation (price > pivot + volume > 1.5x avg):

| Config | Trades | Win Rate | Profit Factor | Return |
|--------|--------|----------|---------------|--------|
| RS 70 | 126 | 40.0% | 0.66 | -177.2% |
| RS 70 + Prox 50 | 126 | 40.0% | 0.66 | -177.2% |
| RS 80 | 86 | 39.5% | 0.60 | -143.9% |

**Problem Identified:** Entries were occurring on weak breakout days (e.g., close near day's low), not genuine breakouts.

### Final Run (Strict Breakout Criteria)

Added strict breakout confirmation:
1. Close must be above pivot
2. Volume > 1.5x average
3. Bullish candle (close > open)
4. Close in upper 50% of day's range
5. Previous close was below pivot (fresh breakout)

| Config | Trades | Win Rate | Profit Factor | Return |
|--------|--------|----------|---------------|--------|
| RS 70 | 23 | 47.8% | 0.98 | -1.76% |
| RS 70 + Prox 50 | 23 | 47.8% | 0.98 | -1.76% |
| RS 80 | 13 | 38.5% | 0.82 | -9.96% |
| RS 70 + Fixed | 23 | 30.4% | 0.67 | -37.30% |

## Detailed Analysis

### Exit Reason Distribution (RS 70 Trailing)

| Exit Reason | Count | Avg P&L |
|-------------|-------|---------|
| Stop | 12 | -7.00% |
| Trailing Stop | 11 | +7.48% |

### Monthly Performance

| Month | Trades | Win Rate | Avg P&L |
|-------|--------|----------|---------|
| Dec 2024 | 1 | 100% | +18.59% |
| Jan 2025 | 6 | 50% | -0.82% |
| Feb 2025 | 5 | 40% | -0.49% |
| Mar 2025 | 3 | 0% | -7.00% |
| Jun 2025 | 2 | 50% | -2.05% |
| Jul 2025 | 3 | 100% | +5.75% |
| Aug 2025 | 3 | 33% | -1.73% |

**Observation:** Strong seasonal variation. March 2025 had 0% win rate, July 2025 had 100%.

### Winners vs Losers Characteristics

| Metric | Winners | Losers |
|--------|---------|--------|
| Volume Ratio | 1.97x | 1.98x |
| RS Rating | 80.7 | 82.6 |
| Num Contractions | 2.5 | 2.2 |
| Proximity Score | 98.9 | 95.6 |

**Observation:** No strong differentiator between winners and losers. Both have similar characteristics.

### Top Performing Trades

| Symbol | Entry Date | P&L | Max Gain | Contractions |
|--------|------------|-----|----------|--------------|
| AVGO | 2024-12-11 | +18.59% | +24.84% | 3 |
| GILD | 2025-02-03 | +15.84% | +21.94% | 2 |
| JNJ | 2025-08-04 | +8.81% | +14.54% | 3 |
| NOC | 2025-07-22 | +8.42% | +14.12% | 3 |
| APD | 2025-01-13 | +7.40% | +13.06% | 2 |

## Comparison with V2 Framework

| Metric | V2 (Old Detector) | V3 (Refined Detector) |
|--------|-------------------|----------------------|
| VCP Algorithm | Rolling window | Swing high/low |
| Trades (RS 70, Prox 50) | 62 | 23 |
| Win Rate | ~50% | 47.8% |
| Profit Factor | 1.31 | 0.98 |
| Total Return | +65.26% | -1.76% |

**Key Difference:** The old V2 framework reported profitable results with RS 70 + Proximity >= 50. The new V3 framework with the refined detector is more selective but not generating better results.

## Issues Identified

### 1. Proximity Filter Not Effective
All detected patterns have proximity scores 85-100. The proximity filter (>=50) doesn't filter anything because the refined detector only produces high-quality patterns.

### 2. Market Timing Dominates
Monthly win rates vary from 0% to 100%, suggesting market conditions are more important than pattern quality.

### 3. Fewer Trades Than Expected
Strict breakout criteria reduced trades from 126 to 23. This may be too restrictive.

### 4. Different Detection Approach
The V2 and V3 detectors may be finding fundamentally different patterns, making direct comparison difficult.

## Possible Improvements to Investigate

1. **Relax breakout criteria** - Try upper 40% close instead of 50%
2. **Add market regime filter** - Avoid trading when SPY < 200 MA
3. **Compare pattern overlap** - Check if V2 and V3 find the same patterns
4. **Test longer history** - 2020-2024 instead of 2023+
5. **Adjust trailing stop parameters** - Try 10% activation instead of 8%

## Files Created

- `backtest/run_backtest_v3.py` - New backtest script using refined detector
- `results/v3/*.json` - Summary results
- `results/v3/*.csv` - Trade details

## Conclusion

The refined VCP detector correctly identifies valid VCP patterns with proper contraction sequences. However, the backtest results are not profitable (PF 0.98).

The strict breakout criteria significantly improved results (from -177% to -1.76%) by filtering out weak entry signals. Further optimization of entry timing and market regime filtering may be needed.

**Status:** Algorithm refinement complete. Further optimization required for profitability.
