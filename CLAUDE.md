# AI Trading Assistant - Project Context

## Project Overview

This project is an AI-powered trading assistant built with reference to https://github.com/nic-tsang02/grok-trading-bot.git, designed with flexibility for:
- **Multi-LLM Support**: Grok, OpenAI, Claude, local models
- **Multi-channel Interfaces**: Telegram, Discord, Web Dashboard, REST API
- **Plugin-based Strategies**: Extensible strategy system
- **Scalable Database**: SQLite for dev, PostgreSQL for production

## Current Phase: VCP Alert System - IMPLEMENTED

### Status (as of 2025-12-02)

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
13. **VCPAlertSystem Implementation** - Full implementation with 144 unit tests passing
14. **Static Chart Generation** - Matplotlib-based PNG charts with VCP pattern visualization
15. **Interactive Dashboard** - TradingView Lightweight Charts integration with multi-stock selector
16. **VCP CLI Scanner** - Standalone `vcp-scan` command with watchlist support
17. **Staleness Detection** - Pattern validity checking (time decay, pivot/support violations)
18. **Dashboard Enhancements** - MA indicators, VRVP, volume 50MA, symbol selection & copy

### Key Files

```
src/vcp/                           # VCP ALERT SYSTEM (IMPLEMENTED)
├── __init__.py                    # Package exports
├── models.py                      # Alert, AlertChain, VCPPattern, Contraction, SwingPoint
├── detector.py                    # VCPDetector with swing detection, scoring & staleness
├── alert_manager.py               # AlertManager with state machine & deduplication
├── repository.py                  # SQLite & InMemory alert repositories
├── notifications.py               # NotificationHub with multiple channels
├── alert_system.py                # VCPAlertSystem orchestrator
├── chart.py                       # Matplotlib static chart generation
└── chart_lightweight.py           # TradingView Lightweight Charts dashboard

script/                            # CLI TOOLS
├── __init__.py                    # Package init
└── vcp_scan.py                    # vcp-scan CLI command (S&P 500, watchlist support)

tests/vcp/                         # 144 UNIT TESTS
├── test_models.py                 # 32 tests
├── test_detector.py               # 15 tests
├── test_alert_manager.py          # 27 tests
├── test_repository.py             # 20 tests
├── test_notifications.py          # 26 tests
└── test_alert_system.py           # 24 tests

temp/                              # SCAN OUTPUT (generated)
├── sp500_staleness_test/          # Latest S&P 500 scan with staleness
│   ├── vcp_dashboard.html         # Interactive dashboard
│   └── scan_results.json          # Scan results JSON
└── ...                            # Other test outputs

backtest/
├── strategy_params.py             # Extracted strategy parameters
├── backtest_framework.py          # Base backtest engine
├── backtest_framework_v2.py       # Enhanced with proximity analysis (V2 detector)
├── run_backtest_v3.py             # Uses refined VCP detector (V3)
├── run_backtest_dual_alert.py     # Three-stage alert system backtest
├── vcp_detector.py                # Refined VCP pattern detection
├── vcp_detector_v5.py             # V5 detector with entry types
├── comprehensive_analysis.py      # Full comparison analysis script
├── visualize_vcp_review.py        # VCP chart generation
├── visualize_trades_enhanced.py   # Trade visualization
└── charts/
    └── vcp_review/                # Generated VCP analysis charts

results/
├── v3/                            # V3 backtest results
├── comprehensive_analysis/        # Full analysis with charts
├── entry_timing/                  # EOD vs SOD entry comparison
└── three_stage_alert/             # Three-stage alert backtest results
```

### VCPAlertSystem Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      VCPAlertSystem                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────────┐   │
│  │ VCPDetector │  │ AlertManager │  │   NotificationHub     │   │
│  │             │  │              │  │                       │   │
│  │ - Swing     │  │ - State      │  │ - Console Channel     │   │
│  │   Detection │  │   Machine    │  │ - Log Channel         │   │
│  │ - Contrac-  │  │ - Dedup      │  │ - Webhook Channel     │   │
│  │   tions     │  │ - TTL        │  │ - Callback Channel    │   │
│  │ - Scoring   │  │ - Chains     │  │                       │   │
│  └─────────────┘  └──────────────┘  └───────────────────────┘   │
│         │                │                     │                 │
│         └────────────────┼─────────────────────┘                 │
│                          │                                       │
│                 ┌────────▼────────┐                              │
│                 │ AlertRepository │                              │
│                 │ (SQLite/Memory) │                              │
│                 └─────────────────┘                              │
└─────────────────────────────────────────────────────────────────┘
```

### Three-Stage Alert Flow

1. **Contraction Alert** - VCP pattern detected with 2+ qualified contractions
2. **Pre-Alert** - Price within 3% of pivot, pattern still valid
3. **Trade Alert** - Price breaks above pivot, entry signal triggered

### Usage Examples

**CLI Scanner (vcp-scan command):**
```bash
# Activate venv first
source venv/bin/activate

# Scan S&P 500 (default)
vcp-scan

# Scan specific symbols
vcp-scan --symbols AAPL,NVDA,MSFT,GOOGL

# Scan from watchlist file
vcp-scan -w watchlist.txt

# Custom output directory
vcp-scan -o output/my_scan

# Disable staleness check (legacy mode)
vcp-scan --no-staleness

# Quiet mode
vcp-scan -q
```

**Python API:**
```python
# Basic usage
from src.vcp import VCPAlertSystem, create_system

system = create_system()
alerts = system.process_symbol("AAPL", price_data)

# Custom configuration with staleness detection
from src.vcp import SystemConfig, DetectorConfig, AlertConfig

config = SystemConfig(
    detector_config=DetectorConfig(
        min_contractions=2,
        enable_staleness_check=True,  # Enable staleness detection
        max_days_since_contraction=42,  # 6 weeks
    ),
    alert_config=AlertConfig(min_score_contraction=60.0),
)
system = VCPAlertSystem(config)

# Generate interactive dashboard
from src.vcp import LightweightChartGenerator
generator = LightweightChartGenerator(output_dir="temp")
generator.generate_dashboard(scan_results, filename="dashboard.html")
```

### S&P 500 Scan Results (2025-12-02, with staleness)

| Metric | Value |
|--------|-------|
| Symbols Scanned | 503 |
| VCP Patterns Found | 151 (30% hit rate) |
| Fresh Patterns | 73 |
| Stale Patterns | 78 |
| Trade Alerts | 86 |
| Pre-Alerts | 24 |
| Contraction Alerts | 41 |

### Interactive Dashboard Features

| Feature | Description |
|---------|-------------|
| Multi-stock selector | Filter by alert type (Trade, Pre, Contraction, None) |
| MA Indicators | 10, 20, 50, 63, 150, 200 period moving averages (toggleable) |
| Volume Profile (VRVP) | Visible Range Volume Profile on left side |
| Volume 50MA | Moving average line on volume chart |
| Symbol Selection | Checkbox selection with Select All / Clear |
| Copy Symbols | Copy selected symbols as comma-separated list |
| Keyboard Navigation | Arrow keys to navigate, synced crosshair |
| Staleness Indicators | [STALE] and [!] markers for pattern validity |

### VCP Detection Algorithm

**Algorithm Rules:**
1. **Swing Detection**: 5-bar lookback for swing highs/lows
2. **Contraction Identification**: Swing high → deepest low before recovery
3. **Progressive Tightening**: Each contraction must be smaller % than previous
4. **Consolidation Base**: Later highs max 3% above base (rejects staircases)
5. **Time Proximity**: Max 30 trading days gap between contractions
6. **Volume Dry-up**: Later contractions should show declining volume

**Scoring System:**
- Base score from contraction quality (progressive tightening)
- Volume quality bonus (declining volume in contractions)
- Final contraction tightness bonus (<10% range)
- Scores range 0-100, minimum 60 for alerts

### Backtest Results Summary

**Three-Stage Alert System (2020-2024):**
- Win Rate: 71-74% depending on alert combination
- Profit Factor: 1.83
- Total Return: 250.6%
- Pre-alerts have highest win rate (74.2%)

**V2 vs V3 Detector:**
- V2 (rolling window): +672.5% return, 619 trades
- V3 (swing-based): +77.9% return, 85 trades
- Recommendation: Use V2 for live trading

### Configuration

**Default Settings:**
```python
DetectorConfig(
    swing_lookback=5,
    min_contractions=2,
    max_contraction_range=20.0,
    lookback_days=120,
    # Staleness detection
    enable_staleness_check=True,
    max_days_since_contraction=42,  # 6 weeks
    max_pivot_violations=2,
    max_support_violations=1,
)

AlertConfig(
    min_score_contraction=60.0,
    pre_alert_proximity_pct=3.0,
    dedup_window_days=7,
    alert_ttl_days=30,
)
```

### Staleness Detection

Patterns are checked for staleness based on:
1. **Time Decay** - Patterns older than 6 weeks (42 trading days) are marked stale
2. **Pivot Violations** - Price crossed above pivot then fell back (max 2 allowed)
3. **Support Violations** - Price closed below support (invalidates pattern)

**Staleness Metrics in VCPPattern:**
- `days_since_last_contraction` - Trading days since last contraction
- `pivot_violations` - Count of pivot level violations
- `support_violations` - Count of support breaks
- `is_stale` - Boolean flag
- `freshness_score` - 0-100 score (lower = more stale)
- `staleness_reasons` - List of reasons

Use `--no-staleness` flag or `enable_staleness_check=False` for legacy behavior.

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
6. **VCPAlertSystem Implementation (2025-12-01)**:
   - Protocol-based repository allows easy swapping SQLite/InMemory
   - Observer pattern for notifications enables flexible integrations
   - Deduplication window prevents alert spam
   - Alert chains track full progression for conversion analytics
7. **CLI & Dashboard Enhancements (2025-12-02)**:
   - `vcp-scan` standalone command via pyproject.toml entry point
   - Staleness detection identifies stale/invalidated VCP patterns
   - Dashboard: MA toggles, VRVP, volume 50MA, symbol checkboxes
   - Synced price/volume chart scrolling via logical range
   - Volume scale starts from 0 with B/M/K formatting

### Git Repository

- Remote: https://github.com/guardave/ai-trading-assistant.git
- User: dawo (dawo.dev@idficient.com)

### Design Documents
- Business Requirements: `docs/01-business-requirements.md` (Section 4.8)
- System Requirements: `docs/02-system-requirements.md` (Section 5)
- System Design: `docs/03-system-design.md` (Section 8)
- Test Strategy: `docs/06-test-strategy.md`
- Test Plan: `docs/07-test-plan.md`

## Technical Notes

- Data source: yfinance (Yahoo Finance API)
- Target universe: S&P 500, NASDAQ 100 (US stocks)
- Historical period: 2023-present (V3), 2020-2024 (V2)
- Visualization: matplotlib (static), TradingView Lightweight Charts (interactive)
- Analysis: pandas, numpy
- Testing: pytest (144 tests)
- Chart library: TradingView Lightweight Charts v4.1.0 (35KB, Apache 2.0)
