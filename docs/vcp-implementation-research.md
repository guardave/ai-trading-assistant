# VCP Implementation Research Report

**Date:** 2025-11-30 (Updated: 2025-12-01)
**Purpose:** Review how others implement VCP detection algorithms and compare with our V2/V3 implementations
**Status:** Research complete, VCPAlertSystem implemented

## Implementation Status (2025-12-01)

Based on this research, the VCPAlertSystem has been implemented in `src/vcp/` with:
- Swing-based detection (V3 approach)
- Progressive tightening validation
- Base structure validation (no staircase patterns)
- Volume dry-up scoring
- Three-stage alert flow (Contraction → Pre-Alert → Trade)
- 144 unit tests passing

**Note:** Prior uptrend validation is documented as a future enhancement.

## Executive Summary

After reviewing literature, open-source implementations, and community discussions, the key findings are:

1. **Our V2 detector deviates significantly from Minervini's methodology** - uses rolling window range detection instead of true swing high/low contractions
2. **Our V3 detector is closer to the canonical approach** - uses swing point detection with contraction sequences
3. **Industry consensus parameters** are well-documented and should be our baseline
4. **Critical missing element**: Prior uptrend validation (30%+ advance before base)

---

## Mark Minervini's Canonical VCP Definition

### Pattern Structure (from "Trade Like a Stock Market Wizard")

| Element | Specification |
|---------|---------------|
| **Contractions** | 2-6 pullbacks, each smaller than the previous |
| **Typical Depths** | First: 20-25%, Second: 10-15%, Third: 5-8%, Final: 2-5% |
| **Tightening Rule** | Each contraction ~50% of previous (±10%) |
| **Base Duration** | 3-65 weeks typical; minimum 3-4 weeks |
| **Volume Pattern** | Decreasing during contractions, spike on breakout |
| **Final Contraction** | Single-digit percentages ideal (<10%) |

### Prior Uptrend Requirement (CRITICAL)

> "A constructive VCP needs to have a prior uptrend. Period." - [Sakatas Homma](https://sakatas.substack.com/p/what-really-is-a-vcp-pattern)

- Stock must have advanced **30-50%+ in 2-3 months** before base formation
- This filters out weak stocks that merely consolidate sideways

### Trend Template (Pre-Qualification)

From [GitHub VCP Screener](https://github.com/marco-hui-95/vcp_screener.github.io):

1. Current price > 150-day and 200-day moving averages
2. 150-day MA > 200-day MA
3. 200-day MA trending UP for ≥1 month
4. 50-day MA > 150-day MA > 200-day MA
5. Current price > 50-day MA
6. Current price ≥ 30% above 52-week low
7. Current price within 25% of 52-week high
8. **Relative Strength rating ≥ 70**

---

## Industry Implementation Approaches

### Approach 1: Swing Point Detection (Correct Method)

Used by: TradingView indicators, QuantConnect, our V3

**Algorithm:**
```
1. Identify swing highs using N-bar lookback (typically N=5)
2. Identify swing lows using N-bar lookback
3. Pair swing high with deepest low before recovery
4. Calculate contraction depth: (high - low) / low × 100
5. Validate progressive tightening: C2 < C1, C3 < C2, etc.
6. Validate swing highs form flat base (not stepping higher)
```

**Key Parameters:**
| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| Swing lookback | 5 bars | Lower = more sensitive, more noise |
| Min contractions | 2 | MM suggests 2-6 |
| Max first contraction | 30-35% | Deeper = riskier base |
| Final contraction max | 10-15% | Tighter = better |
| Tightening ratio | 0.5-0.7 | Each ~50-70% of previous |
| Max gap between contractions | 30 days | Prevents stale bases |
| Base high tolerance | ≤3% | Rejects staircase uptrends |

### Approach 2: Rolling Range Detection (Simplified, Our V2)

Used by: ThinkorSwim scripts, basic screeners

**Algorithm:**
```
1. Calculate rolling N-day high and low (N=20 typical)
2. Calculate range = (high - low) / low × 100
3. Detect when range contracts by >30%
4. Count number of contraction phases
```

**Problems with this approach:**
- Doesn't identify actual swing points
- Can't verify progressive tightening properly
- Misses the HIGH-to-LOW structure that defines true contractions
- May detect false "contractions" from sideways drift

### Approach 3: ATR-Based Detection

Used by: [VCP-Minervini v2 on TradingView](https://www.tradingview.com/script/q2IGWu2N-VCP-Minervini-v2/)

**Algorithm:**
```
1. Calculate 14-period ATR
2. Track ATR reduction over time
3. Flag when ATR contracts by threshold (e.g., 20%+)
4. Combine with EMA trend filter
```

**Pros:** Simple, captures volatility contraction
**Cons:** Doesn't capture the discrete contraction structure

---

## Parameter Comparison: Our V2 vs V3 vs Industry

| Parameter | Our V2 | Our V3 | Industry Consensus |
|-----------|--------|--------|-------------------|
| **Method** | Rolling window | Swing points | Swing points |
| **Swing lookback** | N/A | 5 bars | 5-10 bars |
| **Min contractions** | Not explicit | 2 | 2 |
| **Contraction validation** | Range decreasing | Sequential tighter | Sequential tighter |
| **Base high check** | None | ≤3% above base | ≤3% above base |
| **Volume dry-up** | Yes | Yes | Yes |
| **Prior uptrend** | **NO** | **NO** | **YES (CRITICAL)** |
| **Trend template** | Basic RS | Basic RS | Full 8 criteria |

---

## Real-World Examples with Specific Parameters

### Example 1: TSLA VCP (from TraderLion)
- **Prior advance:** 600% off lows
- **Contractions:** 25% → 20% → 10%
- **Result:** +90% post-breakout

### Example 2: Royal Caribbean (from TraderLion)
- **Prior advance:** 100% in 9 months
- **Contractions:** 25% → 9% → 3%
- **Risk/Reward:** Entry $173, Stop $167 (3.5% risk), Target achieved $257 (14:1 R:R)

### Example 3: Spotify (from DeepVue)
- **Prior advance:** 4.5x in 1 year
- **Contractions:** 16% → 8% → 3%
- **Result:** Successful breakout

### Example 4: NVDA (from EBC Financial)
- **Base duration:** 6 weeks
- **Contractions:** 15% → 8% → 4%
- **Volume:** Lowest in 2 months at final contraction

---

## Why Our V2 Produces "Better" Results Despite Deviation

1. **Less restrictive filtering** - Catches more patterns, including lower-quality ones
2. **No progressive tightening enforcement** - Accepts any range contraction
3. **No base structure validation** - Accepts staircase uptrends
4. **Quantity over quality** - More trades, some work out

However, this is **not a true VCP strategy** - it's a generic volatility contraction strategy.

---

## Recommended Implementation Improvements

### High Priority

1. **Add Prior Uptrend Validation**
   ```python
   def has_prior_uptrend(df, lookback=60, min_advance=0.30):
       """Check if stock advanced 30%+ in prior 60 days"""
       start_price = df['Close'].iloc[-lookback]
       peak_price = df['Close'].iloc[-lookback:].max()
       return (peak_price - start_price) / start_price >= min_advance
   ```

2. **Implement Full Trend Template**
   - Price > 50 MA > 150 MA > 200 MA
   - 200 MA rising for 1+ month
   - Price ≥ 30% above 52-week low
   - Price within 25% of 52-week high

3. **Enforce Contraction Tightening Ratio**
   - Each contraction should be 50-70% of previous
   - Reject if any contraction is WIDER than previous

### Medium Priority

4. **Refine Base Structure Validation**
   - Later swing highs should NOT exceed first swing high by >3%
   - Lows should show higher-low pattern (not declining)

5. **Time-Based Filters**
   - Minimum base duration: 15-20 trading days
   - Maximum base duration: 65 weeks (per MM)
   - Maximum gap between contractions: 30 days

6. **Volume Pattern Scoring**
   - Volume should decline LEFT to RIGHT through base
   - Final contraction volume should be <70% of 50-day average

### Lower Priority

7. **Breakout Quality Scoring**
   - Volume on breakout day > 150% of 50-day average
   - Close in upper 50% of day's range
   - Gap-up breakouts get bonus points

---

## Quantitative Thresholds Summary (Recommended)

```python
VCP_PARAMS = {
    # Swing detection
    'swing_lookback': 5,  # 5-bar pivot detection

    # Contraction requirements
    'min_contractions': 2,
    'max_contractions': 5,
    'max_first_contraction_pct': 35.0,  # First pullback max 35%
    'max_final_contraction_pct': 15.0,  # Final should be <15%, ideally <10%
    'tightening_ratio': 0.70,  # Each contraction ≤70% of previous

    # Base structure
    'max_high_deviation_pct': 3.0,  # Later highs can't be >3% above first
    'max_days_between_contractions': 30,
    'min_base_duration_days': 15,

    # Prior uptrend
    'min_prior_advance_pct': 30.0,  # Must have 30%+ advance before base
    'prior_advance_lookback_days': 60,

    # Trend template
    'min_rs_rating': 70,
    'require_ma_alignment': True,  # 50 > 150 > 200
    'min_above_52wk_low_pct': 30.0,
    'max_below_52wk_high_pct': 25.0,

    # Volume
    'final_contraction_volume_ratio': 0.70,  # <70% of 50-day avg
    'breakout_volume_ratio': 1.50,  # >150% of 50-day avg
}
```

---

## Sources

1. [TraderLion - Mastering The Volatility Contraction Pattern](https://traderlion.com/technical-analysis/volatility-contraction-pattern/)
2. [TrendSpider - VCP Pattern Explained](https://trendspider.com/learning-center/volatility-contraction-pattern-vcp/)
3. [Sakatas Homma - What Really is a VCP Pattern?](https://sakatas.substack.com/p/what-really-is-a-vcp-pattern)
4. [GitHub - VCP Screener by marco-hui-95](https://github.com/marco-hui-95/vcp_screener.github.io)
5. [TradingView - VCP-Minervini v2](https://www.tradingview.com/script/q2IGWu2N-VCP-Minervini-v2/)
6. [TradingView - Volatility Contraction Pattern Indicator](https://www.tradingview.com/script/J1tqSCqR-Volatility-Contraction-Pattern/)
7. [useThinkScript - ThinkorSwim VCP Implementation](https://usethinkscript.com/threads/thinkorswim-volatility-contraction-pattern-vcp-by-mark-minervini.180/)
8. [QuantConnect - VCP Pattern Forum Discussion](https://www.quantconnect.com/forum/discussion/13951/model-volatility-contraction-cup-and-handle-patterns)
9. [Tikamalma Substack - Python VCP Scanner](https://tikamalma.substack.com/p/understanding-basics-of-vcp-and-creating)
10. [DeepVue - VCP Guide](https://deepvue.com/screener/volatility-contraction-pattern/)
11. Mark Minervini - "Trade Like a Stock Market Wizard"
12. Mark Minervini - "Think & Trade Like a Champion"

---

## Conclusion

Our V3 detector is structurally correct but missing the **prior uptrend validation** which is critical to Minervini's methodology. The V2 detector, while producing more trades, does not implement true VCP detection - it's a generic range contraction scanner.

**Recommended next step:** Implement a V4 detector that adds:
1. Prior uptrend validation (30%+ advance)
2. Full trend template criteria
3. Stricter tightening ratio enforcement (50-70%)
4. Minimum base duration (15+ days)

This should reduce trade count but significantly improve win rate and profit factor by filtering for only the highest-quality VCP setups.

---

## Implementation Complete (2025-12-01)

The VCPAlertSystem has been implemented based on this research:

### Components Implemented

| Component | File | Description |
|-----------|------|-------------|
| Data Models | `src/vcp/models.py` | Alert, AlertChain, VCPPattern, Contraction, SwingPoint |
| Detector | `src/vcp/detector.py` | Swing-based detection with progressive tightening |
| Alert Manager | `src/vcp/alert_manager.py` | State machine, deduplication, TTL |
| Repository | `src/vcp/repository.py` | SQLite and InMemory implementations |
| Notifications | `src/vcp/notifications.py` | Multi-channel notification hub |
| Orchestrator | `src/vcp/alert_system.py` | Main VCPAlertSystem class |
| Static Charts | `src/vcp/chart.py` | Matplotlib PNG generation |
| Interactive Charts | `src/vcp/chart_lightweight.py` | TradingView Lightweight Charts dashboard |

### Test Coverage

- 144 unit tests across 6 test files
- All tests passing as of 2025-12-01

### Live Scan Results (S&P 500, 2025-12-01)

| Metric | Value |
|--------|-------|
| Symbols Scanned | 494 |
| Valid VCP Patterns | 260 (52.6% hit rate) |
| Trade Alerts | 124 |
| Pre-Alerts | 32 |
| Contraction Alerts | 104 |

### Future Enhancements

Based on this research, the following are documented for future implementation:
1. Prior uptrend validation (30%+ advance requirement)
2. Full Minervini trend template (8 criteria)
3. Stricter tightening ratio enforcement (50-70%)
4. Minimum base duration filter (15+ days)
5. Breakout quality scoring
