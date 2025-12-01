# AI Trading Assistant - Project Context

## Project Overview

This project is an AI-powered trading assistant built with reference to https://github.com/nic-tsang02/grok-trading-bot.git, designed with flexibility for:
- **Multi-LLM Support**: Grok, OpenAI, Claude, local models
- **Multi-channel Interfaces**: Telegram, Discord, Web Dashboard, REST API
- **Plugin-based Strategies**: Extensible strategy system
- **Scalable Database**: SQLite for dev, PostgreSQL for production

## Current Phase: VCP Alert System Implementation

### Status (as of 2025-11-30)

**Completed:**
1. Full SOP documentation in `docs/` (7 documents)
2. Strategy extraction from reference project
3. Research paper on VCP, Pivot, Cup with Handle strategies
4. Backtest framework v1 and v2 with enhanced features
5. Jupyter notebook for interactive backtesting with visualizations
6. Refactored all backtest files to `backtest/` folder
7. **VCP Detection Algorithm** - Refined with proper swing high/low methodology
8. **VCP Visualization Tool** - Charts for pattern review
9. **Backtest V3** - New backtest script using refined VCP detector
10. **Comprehensive Analysis** - V2 vs V3 comparison, market filter, extended history, trailing stop optimization
11. **Three-Stage Alert System Backtest** - Contraction → Pre-Alert → Trade alert flow validated
12. **Documentation Updated** - Business requirements, system requirements, system design for VCPAlertSystem

**Next: VCPAlertSystem Implementation**

### Key Files

```
backtest/
├── strategy_params.py          # Extracted strategy parameters
├── backtest_framework.py       # Base backtest engine
├── backtest_framework_v2.py    # Enhanced with proximity analysis (V2 detector)
├── run_backtest_v3.py          # Uses refined VCP detector (V3)
├── run_backtest_dual_alert.py  # Three-stage alert system backtest
├── vcp_detector.py             # Refined VCP pattern detection
├── vcp_detector_v5.py          # V5 detector with entry types
├── comprehensive_analysis.py   # Full comparison analysis script
├── visualize_vcp_review.py     # VCP chart generation
├── visualize_trades_enhanced.py # Trade visualization
└── charts/
    └── vcp_review/             # Generated VCP analysis charts

results/
├── v3/                         # V3 backtest results
├── comprehensive_analysis/     # Full analysis with charts
├── entry_timing/               # EOD vs SOD entry comparison
└── three_stage_alert/          # Three-stage alert backtest results
    ├── contraction_alerts.csv  # All contraction alerts
    ├── pre_alerts.csv          # All pre-alerts
    ├── trades.csv              # All trades with alert linkage
    ├── summary.json            # Statistics summary
    └── charts/                 # Sample trade charts

src/vcp/                        # (TO BE IMPLEMENTED)
├── models.py                   # Alert, AlertChain, VCPPattern
├── detector.py                 # VCPDetector class
├── alert_manager.py            # AlertManager class
├── repository.py               # AlertRepository (SQLite)
├── notifications.py            # NotificationHub + channels
└── alert_system.py             # VCPAlertSystem orchestrator
```

### VCP Detection Algorithm (2025-11-29)

**Algorithm Rules:**
1. **Swing Detection**: 5-bar lookback for swing highs/lows
2. **Contraction Identification**: Swing high → deepest low before recovery
3. **Progressive Tightening**: Each contraction must be smaller % than previous
4. **Consolidation Base**: Later highs max 3% above base (rejects staircases)
5. **Time Proximity**: Max 30 trading days gap between contractions
6. **Volume Dry-up**: Later contractions should show declining volume

**Pattern Test Results (8 symbols):**
- ✅ Valid: TSLA (4 contractions), NVDA, MSFT, AMZN (2 each)
- ❌ Invalid: AAPL (staircase), GOOGL/AMD (loosening), META (downtrend)

### Backtest V3 Results (2025-11-30)

**With strict breakout criteria (RS 70, trailing stop):**
- Trades: 23
- Win Rate: 47.8%
- Profit Factor: 0.98 (near breakeven)
- Total Return: -1.76%

**Key Finding:** Strict breakout criteria dramatically improved results:
- Loose criteria: 126 trades, -177% return
- Strict criteria: 23 trades, -1.76% return

**Breakout Criteria Added:**
1. Close must be above pivot
2. Volume > 1.5x average
3. Bullish candle (close > open)
4. Close in upper 50% of day's range
5. Previous close was below pivot (fresh breakout)

See `docs/backtest-v3-results.md` for full analysis.

### Backtest Configuration

**User Requirements:**
- R:R Ratio >= 3.0 (prioritize quality over quantity)
- Stop Loss: 7-8% acceptable
- Test BOTH fixed target AND trailing stop exits
- US stocks only (via yfinance)
- Test VCP contraction proximity effect (new metric)

### Lessons Learned

1. **Date Validation**: Always verify dates in generated documents match current date
2. **Git Setup**: User git config - email: dawo.dev@idficient.com, name: dawo
3. **Strategy Gaps Identified in Reference**:
   - RS threshold (90) may be too restrictive
   - Single contraction check vs sequential validation
   - Missing prior advance (30%+ in 2-3 months) verification
   - Volume confirmation not enforced at breakout
4. **VCP Algorithm Refinement (2025-11-29)** - See `docs/vcp-algorithm-lessons-learned.md`:
   - Validate high > low for contractions
   - Capture full pullback depth (extend until recovery)
   - Enforce progressive tightening strictly
   - Reject staircase patterns (highs stepping higher)
   - Image dimensions must stay under 2000px for API compatibility
5. **Backtest V3 Findings (2025-11-30)** - See `docs/backtest-v3-results.md`:
   - Breakout entry timing is critical (strict criteria needed)
   - Proximity filter ineffective when all patterns score 85-100
   - Market timing dominates (monthly win rates vary 0-100%)
   - V2 vs V3 detectors may find different patterns

### Git Repository

- Remote: https://github.com/guardave/ai-trading-assistant.git
- User: dawo (dawo.dev@idficient.com)

### Comprehensive Analysis Results (2025-11-30)

**Key Findings:**
1. **V2 detector dramatically outperforms V3**: +672.5% vs +77.9% return
2. **V2 generates 7.3x more trades**: 619 vs 85 trades
3. **V2 works in both bull and bear markets**: 52% win rate in both
4. **V3 struggles in bear markets**: Only 25% win rate
5. **Market filter helps V3 but hurts V2**: V3 improves to PF 1.36, V2 drops to +547%
6. **Extended history**: V2 shows +473% (2020-2024), V3 shows -29.9%
7. **Trailing stop 8%/5% optimal**: Current config beats alternatives

**Recommended Configuration:**
- Detector: V2 (rolling window method)
- RS Threshold: 70
- Exit: Trailing stop (8% activation, 5% trail)
- Market Filter: Not recommended for V2

See `docs/comprehensive-analysis-report.md` for full analysis.

### Three-Stage Alert System Results (2025-11-30)

**Backtest Period:** 2020-2024, 66 symbols

**Stage Statistics:**
| Stage | Sequences | Total Alerts | Avg per Seq | Conversion Rate |
|-------|-----------|--------------|-------------|-----------------|
| Contraction | 357 | 367 | 1.0 | 24.6% to trade |
| Pre-Alert | 147 | 570 | 3.9 | 21.1% to trade |
| Trade | 108 | 108 | - | - |

**Win Rates by Alert Type:**
| Alert Combination | Win Rate |
|-------------------|----------|
| Both alerts (Contr + Pre) | 71.4% |
| Pre-Alert only | 74.2% |
| Contraction only | 68.2% |
| No prior alerts | 70.6% |

**Key Findings:**
- Pre-alerts have highest win rate (74.2%) - extra review time helps
- Average lead time: Contraction→Trade = 8.5 days, Pre-Alert→Trade = 5.7 days
- Profit factor: 1.83, Total return: 250.6%

See `docs/three-stage-alert-backtest-results.md` for full analysis.

## Next Steps: VCPAlertSystem Implementation

### Implementation Plan

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Core Data Models (`src/vcp/models.py`) | Pending |
| 2 | Repository Layer (`src/vcp/repository.py`) | Pending |
| 3 | Alert Manager (`src/vcp/alert_manager.py`) | Pending |
| 4 | VCP Detector Refactor (`src/vcp/detector.py`) | Pending |
| 5 | Notification Hub (`src/vcp/notifications.py`) | Pending |
| 6 | Main Orchestrator (`src/vcp/alert_system.py`) | Pending |
| 7 | Integration & Testing | Pending |

### Design Documents
- Business Requirements: `docs/01-business-requirements.md` (Section 4.8)
- System Requirements: `docs/02-system-requirements.md` (Section 5)
- System Design: `docs/03-system-design.md` (Section 8)
- Test Strategy: `docs/06-test-strategy.md` (Section 12)
- Test Plan: `docs/07-test-plan.md` (Section 10)

### Agent SOPs
- Developer: `agents/developer/SOP.md`
- Tester: `agents/tester/SOP.md`

## Technical Notes

- Data source: yfinance (Yahoo Finance API)
- Target universe: S&P 500, NASDAQ 100 (US stocks)
- Historical period: 2023-present (V3), 2020-2024 (V2)
- Visualization: matplotlib, seaborn (max 2000px dimensions)
- Analysis: pandas, numpy
