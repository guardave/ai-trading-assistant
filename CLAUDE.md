# AI Trading Assistant - Project Context

## Project Overview

This project is an AI-powered trading assistant built with reference to https://github.com/nic-tsang02/grok-trading-bot.git, designed with flexibility for:
- **Multi-LLM Support**: Grok, OpenAI, Claude, local models
- **Multi-channel Interfaces**: Telegram, Discord, Web Dashboard, REST API
- **Plugin-based Strategies**: Extensible strategy system
- **Scalable Database**: SQLite for dev, PostgreSQL for production

## Current Phase: Strategy Backtesting Research

### Status (as of 2025-11-28)

**Completed:**
1. Full SOP documentation in `docs/` (7 documents)
2. Strategy extraction from reference project
3. Research paper on VCP, Pivot, Cup with Handle strategies
4. Backtest framework v1 and v2 with enhanced features
5. Jupyter notebook for interactive backtesting with visualizations
6. Refactored all backtest files to `backtest/` folder

**Pending (to start 2025-11-29):**
- Execute the Jupyter notebook backtest
- Analyze results and optimize parameters

### Key Files

```
backtest/
├── strategy_params.py          # Extracted strategy parameters
├── backtest_framework.py       # Base backtest engine
├── backtest_framework_v2.py    # Enhanced with proximity analysis
├── run_backtest.py             # CLI runner v1
├── run_backtest_v2.py          # CLI runner v2
├── strategy_research_paper.md  # Strategy analysis document
└── strategy_backtest.ipynb     # Interactive Jupyter notebook
```

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

### Git Repository

- Remote: https://github.com/guardave/ai-trading-assistant.git
- User: dawo (dawo.dev@idficient.com)

## Next Session (2025-11-29)

1. Open `backtest/strategy_backtest.ipynb` in Jupyter
2. Execute cells sequentially to run backtests
3. Review intermediate tables and charts
4. Analyze proximity score correlation
5. Compare trailing stop vs fixed target results
6. Document findings and update strategy parameters

## Technical Notes

- Data source: yfinance (Yahoo Finance API)
- Target universe: S&P 500, NASDAQ 100 (US stocks)
- Historical period: 2020-2024 (adjustable)
- Visualization: matplotlib, seaborn
- Analysis: pandas, numpy
