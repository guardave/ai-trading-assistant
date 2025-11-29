# VCP Detection Algorithm - Lessons Learned

**Date:** 2025-11-29
**Session:** VCP Algorithm Refinement

## Overview

This document captures the lessons learned during the refinement of the VCP (Volatility Contraction Pattern) detection algorithm. The algorithm went through multiple iterations to correctly identify true VCP patterns while rejecting false positives.

## Key Issues Encountered and Solutions

### 1. Invalid Contractions (Price Going Up)

**Problem:** Initial algorithm paired swing highs with subsequent swing lows without validating that the high price was actually higher than the low price. This resulted in "contractions" that showed price going UP (negative contraction percentage).

**Example:** AMD showed C3 going from $186 UP to $224, labeled as -26.5% contraction.

**Solution:** Added validation that `swing_high.price > swing_low.price` before creating a contraction.

---

### 2. Wrong Endpoint Selection (Missing Deeper Lows)

**Problem:** Algorithm paired swing high with the FIRST swing low found, not the DEEPEST low before recovery. This missed the full depth of pullbacks.

**Example:** MSFT C1 ended at $515 instead of the deeper $489 low.

**Solution:** Implemented "extend until recovery" approach:
- A contraction extends from swing high to the DEEPEST swing low
- Only closes when price recovers above the starting high
- Each swing low can only be used once

---

### 3. Progressive Loosening Patterns

**Problem:** Algorithm accepted contractions that were getting WIDER over time, not tighter. This is the opposite of a true VCP.

**Examples:**
- GOOGL: 4.9% → 5.4% → 7.3% (getting looser)
- META: 8.1% → 7.4% → 14.4% (tighter then much looser)

**Solution:** Strict filtering rule: each contraction must have a smaller range percentage than the previous one. No exceptions.

---

### 4. Duplicate/Shared Endpoints

**Problem:** Multiple contractions shared the same swing low endpoint, which is incorrect. Each contraction should have distinct endpoints.

**Example:** GOOGL C1 and C2 both ended at the same low price point.

**Solution:** Track used swing lows and ensure each is only used once. The "extend until recovery" approach naturally handles this by closing contractions only when price makes a new higher high.

---

### 5. Staircase Patterns Misidentified as VCP

**Problem:** The 10% consolidation base threshold was too generous. Patterns where each contraction started at a progressively HIGHER price level (staircase uptrend) were incorrectly identified as VCP.

**Example:** AAPL showed:
- C1 high: $215.78
- C2 high: $234.89 (8.9% higher - passed 10% threshold)

This is just an uptrend with pullbacks, NOT a consolidation base.

**Solution:** Changed the consolidation base rule:
- Later swing highs can be at most **3% ABOVE** the base high (slight overshoot allowed)
- Can be up to 10% BELOW the base high
- Stepping significantly HIGHER indicates uptrend, not consolidation

---

### 6. Image Size API Error

**Problem:** Generated chart images exceeded 2000 pixel dimension limit, causing API errors when trying to display them.

**Cause:** `figsize=(16, 12)` with `dpi=150` created images of 2400x1800 pixels.

**Solution:** Reduced figure sizes to stay under 2000px:
- `visualize_vcp_review.py`: `figsize=(13, 10)` → 1950x1500 pixels
- `visualize_trades_enhanced.py`: `figsize=(13, 12)` → 1950x1800 pixels

---

## Final Algorithm Rules

### Contraction Identification
1. Start at a swing high (5-bar lookback)
2. Find the DEEPEST swing low before price recovers above the high
3. Each swing low can only be used once
4. Contraction only closes when price makes a new higher high

### Sequential Filtering
1. **Progressive Tightening**: Each contraction must be smaller % than previous
2. **Consolidation Base**: Later highs max 3% above base high (rejects staircases)
3. **Time Proximity**: Max 30 trading days gap between contractions
4. **No Downtrend**: Lows cannot decline more than 15%

### Validation
1. Minimum 2 contractions required
2. Final contraction should be < 15-20%
3. Volume dry-up in later contractions (< 1.0x average)

---

## Test Results (8 Symbols)

| Symbol | Status | Reason |
|--------|--------|--------|
| TSLA | ✅ VALID | 4 contractions, all highs around $350 level |
| NVDA | ✅ VALID | 2 contractions, highs around $180 level |
| MSFT | ✅ VALID | 2 contractions, highs around $550 level |
| AMZN | ✅ VALID | 2 contractions, highs around $235 level |
| AAPL | ❌ INVALID | Staircase uptrend (C1 $215 → C2 $235) |
| GOOGL | ❌ INVALID | Contractions getting looser |
| META | ❌ INVALID | Downtrend with massive expansion |
| AMD | ❌ INVALID | Contractions getting looser |

---

## Key Takeaways

1. **Validate assumptions**: Always check that fundamental assumptions hold (high > low for a contraction).

2. **Capture full patterns**: Find the deepest point of a pullback, not just the first dip.

3. **Order matters**: VCP requires progressively TIGHTER contractions - enforce this strictly.

4. **Base vs Staircase**: A true VCP consolidates around a price level. Stepping higher is an uptrend, not consolidation.

5. **Image constraints**: When generating visualizations for API consumption, keep dimensions under 2000px.

6. **Visual validation**: Always review generated charts to verify algorithm correctness - numbers alone can be misleading.

---

## Files Modified

- `backtest/vcp_detector.py` - Core VCP detection algorithm
- `backtest/visualize_vcp_review.py` - VCP chart generation
- `backtest/visualize_trades_enhanced.py` - Trade chart generation (image size fix)

## Output Files

- `backtest/charts/vcp_review/*_vcp_review.png` - Valid VCP patterns
- `backtest/charts/vcp_review/*_debug.png` - Invalid patterns (for review)
