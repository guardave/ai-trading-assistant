# Configuration and Usage Guide

## AI Trading Assistant

**Version:** 1.1.0
**Date:** 2025-12-01
**Status:** Updated with VCPAlertSystem

---

## 1. Overview

This guide covers the configuration options and usage patterns for the AI Trading Assistant. It is intended for users who want to customize the system behavior and interact with it effectively.

---

## 2. Configuration Files

### 2.1 File Locations

```
config/
├── config.yaml           # Main application configuration
├── watchlist.yaml        # Stock watchlist
├── currency.yaml         # Exchange rates
├── fees/
│   ├── us.yaml          # US market fee structure
│   ├── hk.yaml          # Hong Kong market fees
│   └── jp.yaml          # Japanese market fees
└── strategies/
    ├── vcp.yaml         # VCP scanner settings
    ├── pivot.yaml       # Pivot scanner settings
    └── ai_scanner.yaml  # AI scanner settings
```

### 2.2 Main Configuration (config.yaml)

```yaml
# Application settings
app:
  name: "AI Trading Assistant"
  environment: "development"  # development | staging | production
  log_level: "INFO"           # DEBUG | INFO | WARNING | ERROR
  timezone: "Asia/Hong_Kong"

# User interface channels
interfaces:
  telegram:
    enabled: true
    # Set via environment: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

  discord:
    enabled: false
    # Set via environment: DISCORD_BOT_TOKEN, DISCORD_GUILD_ID

  rest_api:
    enabled: true
    host: "0.0.0.0"
    port: 8000
    # API key set via environment: API_KEY

  web_ui:
    enabled: false
    port: 3000

# AI provider settings
ai:
  default_provider: "grok"      # grok | openai | claude | ollama
  fallback_order:
    - "grok"
    - "openai"
    - "claude"

  settings:
    max_tokens: 1000
    temperature: 0.7
    timeout_seconds: 30

  # Provider-specific settings
  grok:
    model: "grok-beta"
    # API key via environment: GROK_API_KEY

  openai:
    model: "gpt-4-turbo-preview"
    # API key via environment: OPENAI_API_KEY

  claude:
    model: "claude-3-sonnet-20240229"
    # API key via environment: ANTHROPIC_API_KEY

  ollama:
    model: "llama2"
    base_url: "http://localhost:11434"

# Data provider settings
data:
  cache_ttl_seconds: 60

  providers:
    eodhd:
      enabled: true
      priority: 1
      # API key via environment: EODHD_API_KEY

    finnhub:
      enabled: true
      priority: 2
      # API key via environment: FINNHUB_API_KEY

    yahoo:
      enabled: true
      priority: 3
      # No API key required

# Scheduler settings
scheduler:
  market_scan:
    enabled: true
    interval_hours: 8
    run_on_startup: false

  portfolio_monitor:
    enabled: true
    interval_minutes: 60

  daily_standup:
    enabled: true
    time: "09:00"
    days: ["monday", "tuesday", "wednesday", "thursday", "friday"]

# Storage settings
storage:
  type: "sqlite"  # sqlite | postgresql

  sqlite:
    path: "data/trading.db"

  postgresql:
    # Connection via environment: DATABASE_URL
    pool_size: 5
    max_overflow: 10

# Cache settings
cache:
  type: "memory"  # memory | redis

  redis:
    # Connection via environment: REDIS_URL
    prefix: "trading:"

# Portfolio settings
portfolio:
  base_currency: "USD"
  initial_capital: 50000.00

  risk_management:
    max_risk_per_trade_pct: 2.0      # Maximum 2% risk per trade
    max_portfolio_heat_pct: 10.0     # Maximum 10% total risk
    trailing_stop_activation_pct: 8.0 # Move stop to breakeven at +8%

  position_sizing:
    method: "fixed_risk"  # fixed_risk | fixed_amount | kelly
    fixed_amount: 5000    # Used if method is fixed_amount

# Alert settings
alerts:
  cooldown_minutes: 60
  send_daily_summary: true
  summary_time: "18:00"

  channels:
    - telegram
    - discord  # Only if discord.enabled is true
```

### 2.3 Watchlist Configuration (watchlist.yaml)

```yaml
# Stock watchlist
# Supports US, Hong Kong (.HK), and Japanese (.T) stocks

symbols:
  # US Stocks
  - AAPL
  - NVDA
  - MSFT
  - GOOGL
  - META

  # Hong Kong Stocks
  - 0700.HK   # Tencent
  - 0005.HK   # HSBC
  - 9988.HK   # Alibaba

  # Japanese Stocks
  - 7974.T    # Nintendo
  - 6758.T    # Sony

# Groups for organized scanning
groups:
  tech_leaders:
    - AAPL
    - NVDA
    - MSFT

  asia_tech:
    - 0700.HK
    - 9988.HK
    - 7974.T

# Notes (optional, for reference)
notes:
  NVDA: "AI leader, watch for earnings"
  0700.HK: "China tech, regulatory risk"
```

### 2.4 Currency Configuration (currency.yaml)

```yaml
# Currency exchange rates
# Rates are relative to base currency (USD)

base_currency: "USD"

exchange_rates:
  USD: 1.0
  HKD: 7.80      # 1 USD = 7.80 HKD
  JPY: 150.0     # 1 USD = 150 JPY
  EUR: 0.92      # 1 USD = 0.92 EUR
  GBP: 0.79      # 1 USD = 0.79 GBP

# Auto-update settings
auto_update:
  enabled: false
  provider: "exchangerate-api"  # Optional: fetch live rates
  interval_hours: 24
```

### 2.5 Fee Configuration (fees/us.yaml)

```yaml
# US Market Fee Structure
market: "US"
currency: "USD"

commission:
  type: "per_share"  # per_share | percentage | flat
  rate: 0.005        # $0.005 per share
  minimum: 1.00      # Minimum $1.00
  maximum: null      # No maximum

# Regulatory fees
levy:
  sec_fee: 0.0000278   # SEC fee rate
  taf_fee: 0.000166    # TAF fee rate (capped)
  taf_cap: 8.30        # Maximum TAF per trade

# Example calculation:
# Buy 100 shares at $150
# Commission: max($0.50, $1.00) = $1.00
# SEC: $15,000 * 0.0000278 = $0.42
# TAF: 100 * $0.000166 = $0.02
# Total fees: $1.44
```

### 2.6 Strategy Configuration (strategies/vcp.yaml)

```yaml
# VCP (Volatility Contraction Pattern) Scanner Configuration
name: "vcp"
display_name: "Minervini VCP Scanner"
enabled: true
version: "1.0.0"

# Minervini Trend Template Criteria
trend_template:
  price_above_150ma: true
  price_above_200ma: true
  ma_150_above_200: true
  ma_50_above_150: true
  price_above_50ma: true
  ma_50_trending_up_weeks: 8
  price_above_52w_low_pct: 30    # At least 30% above 52-week low
  price_within_52w_high_pct: 25  # Within 25% of 52-week high

# Relative Strength Requirements
relative_strength:
  min_rs_rating: 90              # Minimum RS rating (0-100)
  benchmark: "SPY"               # Benchmark for RS calculation

# VCP Pattern Parameters
pattern:
  min_consolidation_weeks: 4     # Minimum base length
  max_consolidation_weeks: 52    # Maximum base length
  min_contractions: 2            # Minimum number of contractions
  max_base_depth_pct: 35         # Maximum base depth
  contraction_threshold: 0.20    # Each contraction should be 20% tighter
  volume_dry_up_ratio: 0.5       # Volume should decrease to 50% of average

# Breakout Criteria
breakout:
  pivot_buffer_pct: 2            # Enter within 2% of pivot
  min_volume_multiplier: 1.5     # Volume should be 1.5x average on breakout
  intraday_confirmation: true    # Wait for price to hold above pivot

# Risk Management Defaults
defaults:
  stop_loss_pct: 7               # Default stop 7% below entry
  target_pct: 20                 # Default target 20% above entry
  risk_reward_min: 2.0           # Minimum 2:1 reward to risk

# Alert Settings
alerts:
  on_vcp_detected: true          # Alert when VCP pattern forms
  on_breakout: true              # Alert on breakout
  min_confidence: 70             # Minimum confidence score for alerts
```

---

## 3. Environment Variables

### 3.1 Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `TELEGRAM_BOT_TOKEN` | Telegram bot API token | `123456:ABC-DEF...` |
| `TELEGRAM_CHAT_ID` | Your Telegram chat ID | `123456789` |

### 3.2 AI Provider Keys (at least one required)

| Variable | Description |
|----------|-------------|
| `GROK_API_KEY` | xAI Grok API key |
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic Claude API key |

### 3.3 Data Provider Keys (optional but recommended)

| Variable | Description |
|----------|-------------|
| `EODHD_API_KEY` | EODHD API key (paid, recommended) |
| `FINNHUB_API_KEY` | Finnhub API key (free tier available) |

### 3.4 Storage (production)

| Variable | Description | Example |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection URL | `postgresql://user:pass@host:5432/db` |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379/0` |

### 3.5 Example .env File

```bash
# .env - DO NOT COMMIT THIS FILE

# Telegram Bot
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=987654321

# AI Providers (configure at least one)
GROK_API_KEY=xai-your-grok-api-key
OPENAI_API_KEY=sk-your-openai-api-key
ANTHROPIC_API_KEY=sk-ant-your-claude-api-key

# Data Providers
EODHD_API_KEY=your-eodhd-api-key
FINNHUB_API_KEY=your-finnhub-api-key

# Database (production)
# DATABASE_URL=postgresql://postgres:password@localhost:5432/trading

# Cache (production)
# REDIS_URL=redis://localhost:6379/0

# API Authentication
API_KEY=your-secure-api-key-for-rest-api
```

---

## 4. Command Reference

### 4.1 Telegram Commands

#### Portfolio Commands

| Command | Description | Example |
|---------|-------------|---------|
| `/portfolio` | Show all positions with P&L | `/portfolio` |
| `/cash` | Show cash balance | `/cash` |
| `/summary` | Portfolio summary with metrics | `/summary` |
| `/add` | Add new position | `/add AAPL 100 150.00 145.00 165.00` |
| `/close` | Close position | `/close AAPL` or `/close AAPL 155.00` |
| `/update` | Update stop or target | `/update AAPL stop 148.00` |
| `/deposit` | Deposit cash | `/deposit 5000` |
| `/withdraw` | Withdraw cash | `/withdraw 1000` |

#### Scanning Commands

| Command | Description | Example |
|---------|-------------|---------|
| `/scan` | Run market scan | `/scan` |
| `/scan vcp` | Run VCP scanner only | `/scan vcp` |
| `/analyze` | Analyze single stock | `/analyze NVDA` |
| `/rs` | Get RS rating | `/rs AAPL` |
| `/quote` | Get current quote | `/quote NVDA` |

#### Watchlist Commands

| Command | Description | Example |
|---------|-------------|---------|
| `/watchlist` | Show watchlist | `/watchlist` |
| `/watch` | Add to watchlist | `/watch TSLA META` |
| `/unwatch` | Remove from watchlist | `/unwatch TSLA` |

#### Alert Commands

| Command | Description | Example |
|---------|-------------|---------|
| `/alerts` | Show pending alerts | `/alerts` |
| `/ack` | Acknowledge alert | `/ack 123` |

#### System Commands

| Command | Description | Example |
|---------|-------------|---------|
| `/help` | Show help | `/help` |
| `/health` | System health check | `/health` |
| `/status` | Bot status | `/status` |

### 4.2 Natural Language Examples

The AI assistant understands natural language. Here are examples:

**Portfolio Questions:**
- "How is my portfolio doing?"
- "What's my total P&L?"
- "Show me my positions"
- "Which stocks are at a loss?"

**Stock Analysis:**
- "Analyze NVDA for me"
- "Is AAPL a good buy right now?"
- "What's the RS rating for GOOGL?"
- "Check if META has a VCP pattern"

**Trading Actions:**
- "I bought 50 shares of AAPL at 175 with stop at 168"
- "Close my NVDA position at 520"
- "Move my stop on GOOGL to breakeven"
- "Add MSFT to my watchlist"

**Market Questions:**
- "How's the market today?"
- "What's the VIX at?"
- "Is it a good time to buy stocks?"

---

## 5. Usage Scenarios

### 5.1 Daily Routine

**Morning (Market Open):**
1. Receive daily standup message with:
   - Market regime (SPY, VIX)
   - Open positions summary
   - Priority alerts
   - Top 3 watchlist setups

2. Review alerts:
   ```
   /alerts
   ```

3. Check portfolio:
   ```
   /portfolio
   ```

**During Market Hours:**
1. React to breakout alerts
2. Use `/analyze SYMBOL` for quick analysis
3. Add new positions with `/add`
4. Adjust stops when +8% profit reached

**After Market Close:**
1. Review daily summary
2. Update watchlist
3. Run manual scan if needed: `/scan`

### 5.2 Adding a New Position

**Scenario:** You want to buy NVDA at $500 with stop at $475 and target at $575.

**Step 1:** Analyze the stock
```
/analyze NVDA
```
AI will provide VCP analysis, RS rating, and recommendation.

**Step 2:** Calculate position size
The system calculates based on risk:
- Portfolio: $50,000
- Max risk: 2% = $1,000
- Risk per share: $500 - $475 = $25
- Position size: $1,000 / $25 = 40 shares

**Step 3:** Add position
```
/add NVDA 40 500.00 475.00 575.00
```

**Response:**
```
✅ Position Added: NVDA

Shares: 40
Entry: $500.00
Stop: $475.00 (-5.0%)
Target: $575.00 (+15.0%)

Cost: $20,000.00
Risk: $1,000.00 (2.0% of portfolio)
R:R Ratio: 3.0

Cash Remaining: $30,000.00
```

### 5.3 Managing Existing Positions

**Update Stop Loss:**
```
/update NVDA stop 490.00
```

**Move to Breakeven (after +8%):**
```
/update NVDA stop 500.00
```
Or wait for the automatic trailing stop suggestion alert.

**Close Position:**
```
/close NVDA
```
Uses current market price.

Or specify exit price:
```
/close NVDA 550.00
```

### 5.4 Running Custom Scans

**Full Scan:**
```
/scan
```

**VCP-Only Scan:**
```
/scan vcp
```

**Analyze Specific Symbol:**
```
/analyze AAPL
```

**Quick Quote:**
```
/quote AAPL
```

---

## 6. Troubleshooting

### 6.1 Common Issues

**Issue:** Bot not responding

**Solutions:**
1. Check bot token is correct
2. Verify chat ID matches your Telegram
3. Check logs for errors: `docker logs trading-assistant`

---

**Issue:** No price data for symbol

**Solutions:**
1. Check symbol format (US: `AAPL`, HK: `0700.HK`, JP: `7974.T`)
2. Verify data provider API keys
3. Check if market is open
4. Try `/health` to check data provider status

---

**Issue:** AI responses are slow

**Solutions:**
1. Check AI provider status
2. Increase timeout in config
3. Try switching provider: update `ai.default_provider`

---

**Issue:** Incorrect currency conversion

**Solutions:**
1. Update exchange rates in `currency.yaml`
2. Enable auto-update if available
3. Restart application after config change

---

### 6.2 Logs

**View logs (Docker):**
```bash
docker logs -f trading-assistant
```

**Log locations:**
- Application logs: `logs/app.log`
- Error logs: `logs/error.log`

**Enable debug logging:**
```yaml
# config.yaml
app:
  log_level: "DEBUG"
```

### 6.3 Health Check

**Telegram:**
```
/health
```

**REST API:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "components": {
    "database": "healthy",
    "cache": "healthy",
    "ai_provider": "healthy",
    "telegram_bot": "healthy"
  }
}
```

---

## 7. Best Practices

### 7.1 Risk Management

1. **Never risk more than 2% per trade**
   - Configured in `portfolio.risk_management.max_risk_per_trade_pct`

2. **Keep portfolio heat under 10%**
   - Total open risk should not exceed 10% of portfolio

3. **Honor every stop loss**
   - Never move stops lower to avoid losses

4. **Move stops to breakeven at +8%**
   - The system suggests this automatically

### 7.2 Watchlist Management

1. **Keep watchlist focused**
   - 20-50 symbols maximum for quality scanning

2. **Remove weak performers**
   - RS < 85 should be removed

3. **Group symbols by theme**
   - Easier to track sector movements

### 7.3 Alert Management

1. **Acknowledge alerts promptly**
   - Unacknowledged alerts may repeat after cooldown

2. **Review daily summary**
   - Catches any missed alerts

3. **Set appropriate cooldowns**
   - Too short: alert fatigue
   - Too long: missed opportunities

---

## 8. VCPAlertSystem Configuration

### 8.1 Overview

The VCPAlertSystem is a three-stage alert system for VCP pattern detection, implemented in `src/vcp/`. It provides:

- Pattern detection with swing-based analysis
- Three-stage alerts (Contraction → Pre-Alert → Trade)
- Multiple notification channels
- Static PNG and interactive HTML chart generation

### 8.2 Python Configuration

```python
from src.vcp import (
    VCPAlertSystem,
    SystemConfig,
    DetectorConfig,
    AlertConfig,
)

# Full configuration
config = SystemConfig(
    # Database settings
    db_path="data/alerts.db",      # SQLite database path
    use_memory_db=False,           # Use in-memory DB for testing

    # Detector settings
    detector_config=DetectorConfig(
        swing_lookback=5,          # Bars for swing detection
        min_contractions=2,        # Minimum contractions required
        max_contraction_range=20.0, # Max first contraction %
        lookback_days=120,         # Analysis window
    ),

    # Alert settings
    alert_config=AlertConfig(
        min_score_contraction=60.0,   # Min score for contraction alerts
        min_score_pre_alert=60.0,     # Min score for pre-alerts
        min_score_trade=60.0,         # Min score for trade alerts
        pre_alert_proximity_pct=3.0,  # Pre-alert trigger distance
        dedup_window_days=7,          # Deduplication window
        alert_ttl_days=30,            # Alert expiration
    ),

    # Notifications
    enable_console_notifications=True,
    enable_log_notifications=True,
)

system = VCPAlertSystem(config)
```

### 8.3 Quick Start (Factory Function)

```python
from src.vcp import create_system

# Simple configuration
system = create_system(
    db_path="data/alerts.db",
    min_score=60.0,
    pre_alert_proximity=3.0,
    enable_console=True,
)
```

### 8.4 Processing Symbols

```python
import yfinance as yf

# Fetch data
df = yf.download("AAPL", period="6mo")

# Process single symbol
alerts = system.process_symbol("AAPL", df)

# Process multiple symbols
def fetch_data(symbol):
    return yf.download(symbol, period="6mo")

results = system.process_symbols(
    ["AAPL", "NVDA", "MSFT"],
    fetch_data,
    check_entry=True,
)
```

### 8.5 Chart Generation

#### Static PNG Charts (Matplotlib)

```python
# Generate single chart
path = system.generate_chart(
    symbol="AAPL",
    df=df,
    pattern=pattern,
    alerts=alerts,
    output_dir="charts",
)

# Batch generate from scan results
paths = system.generate_charts_for_scan(
    scan_results=results,
    output_dir="charts",
    max_charts=30,
    by_alert_type=True,  # Organize in subdirs
)
```

#### Interactive Dashboard (TradingView Lightweight Charts)

```python
from src.vcp import LightweightChartGenerator

generator = LightweightChartGenerator(output_dir="charts")
dashboard_path = generator.generate_dashboard(
    scan_results=results,
    filename="vcp_dashboard.html",
)
# Open in browser for interactive viewing
```

### 8.6 Notification Channels

```python
# Add webhook notifications
system.add_webhook_handler(
    name="slack",
    url="https://hooks.slack.com/services/xxx",
    headers={"Content-Type": "application/json"},
)

# Add callback handler
def on_alert(alert, message):
    print(f"New alert: {alert.symbol} - {alert.alert_type}")

system.add_callback_handler(
    name="custom",
    callback=on_alert,
)

# Remove channel
system.remove_notification_channel("slack")
```

### 8.7 Scan Script Usage

```bash
# Full S&P 500 scan with all outputs
python temp/run_sp500_scan.py

# Skip static PNG charts (faster)
python temp/run_sp500_scan.py --no-charts

# Skip interactive dashboard
python temp/run_sp500_scan.py --no-dashboard

# Limit charts per category
python temp/run_sp500_scan.py --max-charts 10
```

### 8.8 Output Files

| File | Description |
|------|-------------|
| `temp/sp500_scan_results.json` | Full scan results in JSON |
| `temp/sp500_vcp_dashboard.html` | Interactive Lightweight Charts dashboard |
| `temp/sp500_charts/trade_alerts/` | PNG charts for trade alerts |
| `temp/sp500_charts/pre_alerts/` | PNG charts for pre-alerts |
| `temp/sp500_charts/contraction_alerts/` | PNG charts for contraction alerts |
| `temp/alerts_sp500.db` | SQLite alert database |

### 8.9 Default Parameters Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `swing_lookback` | 5 | Bars for swing high/low detection |
| `min_contractions` | 2 | Minimum contractions for valid pattern |
| `max_contraction_range` | 20.0 | Maximum first contraction percentage |
| `lookback_days` | 120 | Days of price history to analyze |
| `min_score_contraction` | 60.0 | Minimum score for contraction alert |
| `pre_alert_proximity_pct` | 3.0 | Distance to pivot for pre-alert |
| `dedup_window_days` | 7 | Days before re-alerting same symbol |
| `alert_ttl_days` | 30 | Days until alert expires |

---

## 9. Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-11-28 | Claude | Initial draft |
| 1.1.0 | 2025-12-01 | Claude | Added VCPAlertSystem configuration (Section 8) |
