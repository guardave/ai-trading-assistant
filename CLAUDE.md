# AI Trading Assistant - Project Context

## Project Overview

This project is an AI-powered trading assistant built with reference to https://github.com/nic-tsang02/grok-trading-bot.git, designed with flexibility for:
- **Multi-LLM Support**: Grok, OpenAI, Claude, local models
- **Multi-channel Interfaces**: Telegram, Discord, Web Dashboard, REST API
- **Plugin-based Strategies**: Extensible strategy system
- **Scalable Database**: SQLite for dev, PostgreSQL for production

## Current Phase: Strategy Backtesting Research

### Status (as of 2025-11-29)

**Completed:**
1. Full SOP documentation in `docs/` (7 documents)
2. Strategy extraction from reference project
3. Research paper on VCP, Pivot, Cup with Handle strategies
4. Backtest framework v1 and v2 with enhanced features
5. Jupyter notebook for interactive backtesting with visualizations
6. Refactored all backtest files to `backtest/` folder
7. **VCP Detection Algorithm** - Refined with proper swing high/low methodology
8. **VCP Visualization Tool** - Charts for pattern review

**In Progress:**
- VCP algorithm validation and testing
- Integration with backtest framework

### Key Files

```
backtest/
├── strategy_params.py          # Extracted strategy parameters
├── backtest_framework.py       # Base backtest engine
├── backtest_framework_v2.py    # Enhanced with proximity analysis
├── run_backtest.py             # CLI runner v1
├── run_backtest_v2.py          # CLI runner v2
├── strategy_research_paper.md  # Strategy analysis document
├── strategy_backtest.ipynb     # Interactive Jupyter notebook
├── vcp_detector.py             # VCP pattern detection (NEW)
├── visualize_vcp_review.py     # VCP chart generation (NEW)
├── visualize_trades_enhanced.py # Trade visualization
└── charts/
    └── vcp_review/             # Generated VCP analysis charts
```

### VCP Detection Algorithm (2025-11-29)

**Algorithm Rules:**
1. **Swing Detection**: 5-bar lookback for swing highs/lows
2. **Contraction Identification**: Swing high → deepest low before recovery
3. **Progressive Tightening**: Each contraction must be smaller % than previous
4. **Consolidation Base**: Later highs max 3% above base (rejects staircases)
5. **Time Proximity**: Max 30 trading days gap between contractions
6. **Volume Dry-up**: Later contractions should show declining volume

**Test Results (8 symbols):**
- ✅ Valid: TSLA (4 contractions), NVDA, MSFT, AMZN (2 each)
- ❌ Invalid: AAPL (staircase), GOOGL/AMD (loosening), META (downtrend)

### Backtest Configuration

**User Requirements:**
- R:R Ratio >= 3.0 (prioritize quality over quantity)
- Stop Loss: 7-8% acceptable
- Test BOTH fixed target AND trailing stop exits
- US stocks only (via yfinance)
- Test VCP contraction proximity effect (new metric)

**VCP Proximity Score (0-100):**
A new metric measuring how close/continuous VCP contractions are:
- Gap penalty: days between contractions
- Sequence validity: decreasing range check
- Volume pattern: dry-up confirmation
- Higher score = tighter, more continuous pattern

### Backtest Phases (in Jupyter notebook)

1. **Phase 1 - Baseline**: Current parameters from reference project
2. **Phase 2 - Proximity Analysis**: VCP proximity score correlation
3. **Phase 3 - RS Relaxation**: Test RS 70, 80, 90 thresholds
4. **Phase 4 - Stop/Target Optimization**: R:R ratio optimization
5. **Phase 5 - Trailing vs Fixed**: Exit method comparison
6. **Phase 6 - Summary**: Final recommendations

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

### Git Repository

- Remote: https://github.com/guardave/ai-trading-assistant.git
- User: dawo (dawo.dev@idficient.com)

## Next Steps

1. Integrate VCP detector into backtest framework
2. Run backtests with refined VCP detection
3. Analyze proximity score correlation with trade outcomes
4. Compare trailing stop vs fixed target results
5. Document findings and optimize strategy parameters

## Technical Notes

- Data source: yfinance (Yahoo Finance API)
- Target universe: S&P 500, NASDAQ 100 (US stocks)
- Historical period: 2020-2024 (adjustable)
- Visualization: matplotlib, seaborn (max 2000px dimensions)
- Analysis: pandas, numpy
