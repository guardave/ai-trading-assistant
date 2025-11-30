# Business Requirement Specification

## AI Trading Assistant

**Version:** 1.0.0
**Date:** 2025-11-28
**Status:** Draft

---

## 1. Executive Summary

The AI Trading Assistant is an intelligent, automated trading support system designed to assist individual traders with market monitoring, pattern recognition, portfolio management, and trade execution guidance. The system leverages multiple AI models for analysis and provides a multi-channel interface for user interaction.

This project is inspired by the [grok-trading-bot](https://github.com/nic-tsang02/grok-trading-bot) but redesigned with a modular, extensible architecture to support flexibility in AI providers, communication channels, trading strategies, and deployment scales.

---

## 2. Business Objectives

### 2.1 Primary Goals

1. **Automated Market Monitoring**
   - Continuous scanning of watchlist stocks for trading opportunities
   - Real-time price alerts and pattern detection
   - Multi-market support (US, Hong Kong, Japan, and extensible to others)

2. **AI-Powered Analysis**
   - Leverage large language models for market analysis and insights
   - Support multiple AI providers (pluggable architecture)
   - Learn from user feedback and trading outcomes

3. **Portfolio Management**
   - Track open positions with real-time P&L calculations
   - Multi-currency support with automatic conversion
   - Risk management and position sizing recommendations

4. **User Accessibility**
   - Multiple communication channels (Telegram, Discord, Web, API)
   - Natural language interaction for commands and queries
   - Mobile-friendly alerts and notifications

### 2.2 Secondary Goals

1. **Extensibility**
   - Plugin system for custom trading strategies
   - Modular data provider architecture
   - Easy integration with new exchanges and brokers

2. **Scalability**
   - Start with zero-cost deployment options (dev/small scale)
   - Scale to production-grade infrastructure when needed
   - Containerized deployment (Docker)

3. **Learning & Improvement**
   - Store trading insights and lessons learned
   - Backtest strategies against historical data
   - Self-improvement through outcome analysis

---

## 3. Target Users

### 3.1 Primary Users

| User Type | Description | Key Needs |
|-----------|-------------|-----------|
| Individual Trader | Active swing/position traders | Pattern alerts, portfolio tracking, AI insights |
| Technical Analyst | Chart pattern specialists | VCP, Cup & Handle, Breakout detection |
| Multi-Market Trader | Trades across US, HK, JP markets | Multi-currency support, timezone handling |

### 3.2 User Personas

**Persona 1: David - The Swing Trader**
- Follows Minervini VCP methodology
- Trades US and Hong Kong stocks
- Needs mobile alerts while at work
- Values discipline and risk management

**Persona 2: Alex - The Part-Time Trader**
- Has limited time for market analysis
- Needs automated scanning and alerts
- Prefers simple natural language commands
- Wants AI to do heavy lifting

---

## 4. Functional Requirements

### 4.1 Market Monitoring (FR-MM)

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-MM-01 | System shall scan watchlist stocks at configurable intervals | Must Have |
| FR-MM-02 | System shall detect VCP (Volatility Contraction Pattern) setups | Must Have |
| FR-MM-03 | System shall detect Pivot Breakout patterns | Must Have |
| FR-MM-04 | System shall detect Cup with Handle patterns | Should Have |
| FR-MM-05 | System shall calculate Relative Strength (RS) ratings | Must Have |
| FR-MM-06 | System shall support US market stocks | Must Have |
| FR-MM-07 | System shall support Hong Kong market stocks (.HK) | Must Have |
| FR-MM-08 | System shall support Japanese market stocks (.T) | Should Have |
| FR-MM-09 | System shall allow custom pattern detection via plugins | Should Have |
| FR-MM-10 | System shall cache market data to minimize API calls | Must Have |

### 4.2 AI Analysis (FR-AI)

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-AI-01 | System shall support multiple LLM providers (Grok, OpenAI, Claude) | Must Have |
| FR-AI-02 | System shall provide conversational AI for market questions | Must Have |
| FR-AI-03 | System shall analyze stocks and provide trade recommendations | Must Have |
| FR-AI-04 | System shall remember conversation context within sessions | Must Have |
| FR-AI-05 | System shall store long-term trading insights persistently | Should Have |
| FR-AI-06 | System shall support function calling for real-time data retrieval | Must Have |
| FR-AI-07 | System shall allow switching between AI providers at runtime | Should Have |
| FR-AI-08 | System shall support local LLM deployment (Ollama, etc.) | Could Have |

### 4.3 Portfolio Management (FR-PM)

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-PM-01 | System shall track open positions with entry/stop/target prices | Must Have |
| FR-PM-02 | System shall calculate real-time P&L for each position | Must Have |
| FR-PM-03 | System shall support multiple currencies (USD, HKD, JPY) | Must Have |
| FR-PM-04 | System shall automatically convert currencies to base currency | Must Have |
| FR-PM-05 | System shall calculate transaction fees per market | Should Have |
| FR-PM-06 | System shall track cash balance and buying power | Must Have |
| FR-PM-07 | System shall generate stop-hit and target-hit alerts | Must Have |
| FR-PM-08 | System shall suggest trailing stop adjustments at +8% profit | Should Have |
| FR-PM-09 | System shall track closed trades and historical performance | Should Have |
| FR-PM-10 | System shall calculate portfolio-level risk metrics | Must Have |

### 4.4 User Interface (FR-UI)

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-UI-01 | System shall provide Telegram bot interface | Must Have |
| FR-UI-02 | System shall provide REST API for programmatic access | Must Have |
| FR-UI-03 | System shall provide Discord bot interface | Should Have |
| FR-UI-04 | System shall provide web dashboard for analytics | Should Have |
| FR-UI-05 | System shall support natural language commands | Must Have |
| FR-UI-06 | System shall support slash commands (/portfolio, /scan, etc.) | Must Have |
| FR-UI-07 | System shall format output with markdown and emojis | Should Have |
| FR-UI-08 | System shall provide inline buttons for quick actions | Should Have |

### 4.5 Strategy Plugin System (FR-SP)

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-SP-01 | System shall define standard interface for strategy plugins | Must Have |
| FR-SP-02 | System shall load strategy plugins dynamically at runtime | Must Have |
| FR-SP-03 | System shall support strategy parameter configuration | Must Have |
| FR-SP-04 | System shall allow enabling/disabling strategies | Must Have |
| FR-SP-05 | System shall support strategy backtesting | Should Have |
| FR-SP-06 | System shall log strategy signals and outcomes | Should Have |

### 4.6 Data Management (FR-DM)

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-DM-01 | System shall support multiple data providers with fallback | Must Have |
| FR-DM-02 | System shall cache price data to reduce API calls | Must Have |
| FR-DM-03 | System shall store historical price data for analysis | Should Have |
| FR-DM-04 | System shall support SQLite for development/small deployments | Must Have |
| FR-DM-05 | System shall support PostgreSQL for production deployments | Should Have |
| FR-DM-06 | System shall support Redis for caching and real-time features | Should Have |
| FR-DM-07 | System shall support TimescaleDB for time-series data | Could Have |

### 4.7 Alerts & Notifications (FR-AN)

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-AN-01 | System shall send breakout alerts when patterns trigger | Must Have |
| FR-AN-02 | System shall send stop-loss hit notifications | Must Have |
| FR-AN-03 | System shall send target-hit notifications | Must Have |
| FR-AN-04 | System shall send daily portfolio summaries | Should Have |
| FR-AN-05 | System shall track alert acknowledgment status | Should Have |
| FR-AN-06 | System shall implement cooldown periods for repeated alerts | Should Have |
| FR-AN-07 | System shall send earnings warnings for positions | Could Have |

### 4.8 Three-Stage VCP Alert System (FR-VCP)

The VCP alert system provides progressive notifications as patterns develop, giving traders early warning and preparation time.

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-VCP-01 | System shall generate Contraction Alerts when a new qualified VCP contraction is detected (minimum 2 contractions, score >= 60) | Must Have |
| FR-VCP-02 | System shall generate Pre-Alerts when price moves within 3% of the pivot price | Must Have |
| FR-VCP-03 | System shall generate Trade Alerts when entry signal triggers (handle break, pivot breakout) | Must Have |
| FR-VCP-04 | System shall track alert chains linking contraction → pre-alert → trade alerts | Must Have |
| FR-VCP-05 | System shall persist all alerts with timestamps for historical analysis | Must Have |
| FR-VCP-06 | System shall prevent duplicate alerts for the same pattern/day | Must Have |
| FR-VCP-07 | System shall expire stale alerts (contraction: 20 days, pre-alert: 5 days) | Must Have |
| FR-VCP-08 | System shall support multiple notification channels per alert type | Should Have |
| FR-VCP-09 | System shall provide alert history queries with filtering | Should Have |
| FR-VCP-10 | System shall calculate conversion statistics (contraction → pre-alert → trade rates) | Should Have |

**Alert Stage Definitions:**

1. **Contraction Alert** (Earliest Signal)
   - Triggered when VCP pattern forms a new qualified contraction
   - Purpose: Build watchlist, begin due diligence
   - Average lead time: 8-10 trading days before trade

2. **Pre-Alert** (Imminent Signal)
   - Triggered when price approaches within 3% of pivot
   - Purpose: Watch closely, prepare position sizing, set alerts
   - Average lead time: 3-7 trading days before trade

3. **Trade Alert** (Action Signal)
   - Triggered when entry conditions met (breakout with volume)
   - Purpose: Execute trade
   - Lead time: Same day (EOD) or next day (SOD)

---

## 5. Non-Functional Requirements

### 5.1 Performance (NFR-P)

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-P-01 | Telegram message response time | < 3 seconds |
| NFR-P-02 | Market scan completion time (50 symbols) | < 2 minutes |
| NFR-P-03 | REST API response time | < 500ms |
| NFR-P-04 | Memory usage (idle) | < 256MB |
| NFR-P-05 | CPU usage (idle) | < 5% |

### 5.2 Reliability (NFR-R)

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-R-01 | System uptime | 99.5% |
| NFR-R-02 | Graceful recovery from API failures | Automatic retry with backoff |
| NFR-R-03 | Data persistence on crash | No data loss |
| NFR-R-04 | Health check endpoint availability | Always available |

### 5.3 Security (NFR-S)

| ID | Requirement | Description |
|----|-------------|-------------|
| NFR-S-01 | API keys stored securely | Environment variables, never in code |
| NFR-S-02 | Telegram chat ID validation | Only respond to authorized users |
| NFR-S-03 | REST API authentication | API key or JWT authentication |
| NFR-S-04 | No sensitive data in logs | Mask API keys and tokens |
| NFR-S-05 | HTTPS for all external APIs | TLS 1.2+ required |

### 5.4 Scalability (NFR-SC)

| ID | Requirement | Description |
|----|-------------|-------------|
| NFR-SC-01 | Support watchlist up to 500 symbols | Without performance degradation |
| NFR-SC-02 | Support portfolio up to 100 positions | Real-time updates |
| NFR-SC-03 | Horizontal scaling capability | Via containerization |

### 5.5 Maintainability (NFR-M)

| ID | Requirement | Description |
|----|-------------|-------------|
| NFR-M-01 | Modular architecture | Loosely coupled components |
| NFR-M-02 | Comprehensive logging | Structured, leveled logging |
| NFR-M-03 | Configuration externalization | YAML/JSON config files |
| NFR-M-04 | Docker containerization | Single command deployment |
| NFR-M-05 | Automated testing | >80% code coverage target |

---

## 6. Constraints

### 6.1 Technical Constraints

1. **Python 3.10+** - Required for asyncio features and type hints
2. **Docker** - Deployment must be containerized
3. **Free Tier APIs** - Initial deployment should work with free API tiers
4. **Rate Limiting** - Must respect API provider rate limits

### 6.2 Business Constraints

1. **Development Timeline** - Phased delivery approach
2. **Budget** - Minimize operational costs for initial deployment
3. **Single Developer** - Architecture must support solo maintenance

### 6.3 Regulatory Constraints

1. **Not Financial Advice** - Clear disclaimers required
2. **No Automated Trading** - Information/alert system only
3. **Data Privacy** - User data must be handled responsibly

---

## 7. Assumptions & Dependencies

### 7.1 Assumptions

1. User has basic understanding of trading concepts
2. User can obtain required API keys (Telegram, LLM provider, data providers)
3. Internet connectivity is stable
4. Market data providers remain available

### 7.2 Dependencies

| Dependency | Type | Risk Level |
|------------|------|------------|
| xAI Grok API | AI Provider | Medium (alternative providers available) |
| OpenAI API | AI Provider | Low (widely available) |
| Anthropic Claude API | AI Provider | Low (widely available) |
| Telegram Bot API | Communication | Low (highly reliable) |
| Yahoo Finance | Data Provider | Medium (rate limiting, reliability) |
| EODHD API | Data Provider | Low (paid, reliable) |
| Finnhub API | Data Provider | Low (free tier available) |

---

## 8. Success Criteria

### 8.1 MVP Success Criteria

1. Successfully detect VCP patterns in US stocks
2. Send Telegram alerts within 30 seconds of pattern detection
3. Track portfolio with accurate P&L calculations
4. Respond to natural language queries via AI
5. Run continuously for 7 days without manual intervention

### 8.2 Full Release Success Criteria

1. Support all three markets (US, HK, JP)
2. All four communication channels operational
3. At least 3 strategy plugins available
4. Web dashboard with analytics
5. 99%+ uptime over 30 days

---

## 9. Glossary

| Term | Definition |
|------|------------|
| VCP | Volatility Contraction Pattern - Mark Minervini's chart pattern |
| RS Rating | Relative Strength - stock performance vs market benchmark |
| P&L | Profit and Loss |
| Pivot | Price level where a stock breaks out of a base pattern |
| Stop Loss | Price level to exit a losing trade |
| Trailing Stop | Dynamic stop loss that follows price upward |
| Portfolio Heat | Total risk as percentage of portfolio value |
| LLM | Large Language Model (AI) |

---

## 10. Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-11-28 | Claude | Initial draft |

---

## Appendix A: Reference Project Analysis

The reference project (grok-trading-bot) provides the following capabilities that inform this specification:

### Features Retained
- Telegram bot interface with command handlers
- VCP and Pivot pattern detection
- Multi-currency portfolio tracking
- AI-powered chat and analysis
- Scheduled scanning jobs
- Alert management with acknowledgment

### Features Enhanced
- **Single AI provider → Multi-LLM support** with pluggable architecture
- **Telegram only → Multi-channel** (Telegram, Discord, Web, API)
- **Hard-coded strategies → Plugin system** for extensibility
- **JSON config only → YAML + hot-reload** configuration
- **Single database → Flexible storage** (SQLite → PostgreSQL → TimescaleDB)

### Features Added
- REST API for programmatic access
- Web dashboard for analytics
- Strategy backtesting framework
- Comprehensive test suite
- Docker Compose deployment
