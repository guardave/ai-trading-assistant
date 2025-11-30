# System Requirement Specification

## AI Trading Assistant

**Version:** 1.0.0
**Date:** 2025-11-28
**Status:** Draft
**Format:** Test-Driven Development (TDD)

---

## 1. Introduction

### 1.1 Purpose

This document defines the system requirements for the AI Trading Assistant in a Test-Driven Development (TDD) format. Each requirement is expressed as a testable specification with clear acceptance criteria.

### 1.2 Scope

The system provides automated market monitoring, AI-powered analysis, portfolio management, and multi-channel user interfaces for individual traders.

### 1.3 Document Conventions

Requirements are organized by module and expressed in the format:
- **GIVEN** (preconditions)
- **WHEN** (action/trigger)
- **THEN** (expected outcome)

Priority levels:
- **P0**: Must have for MVP
- **P1**: Should have for v1.0
- **P2**: Could have for future releases

---

## 2. Core Module Specifications

### 2.1 AI Provider Module (`ai_provider`)

#### 2.1.1 Multi-Provider Support

**SRS-AI-001: Provider Registration** [P0]
```
GIVEN the AI provider module is initialized
WHEN a new provider (Grok, OpenAI, Claude) is registered
THEN the provider should be available for selection
AND the provider should implement the standard AIProvider interface
```

**SRS-AI-002: Provider Switching** [P1]
```
GIVEN multiple AI providers are registered
WHEN the user or system requests a different provider
THEN the active provider should switch without service interruption
AND existing conversation context should be preserved where possible
```

**SRS-AI-003: Provider Fallback** [P1]
```
GIVEN the primary AI provider is configured
WHEN the primary provider fails or times out (>30 seconds)
THEN the system should automatically try the fallback provider
AND log the failover event
```

#### 2.1.2 Chat Interface

**SRS-AI-010: Basic Chat** [P0]
```
GIVEN a valid AI provider is active
WHEN the user sends a text message
THEN the AI should respond within 10 seconds
AND the response should be relevant to the trading context
```

**SRS-AI-011: Conversation History** [P0]
```
GIVEN an active chat session
WHEN the user sends multiple messages
THEN the AI should maintain context from previous messages in the session
AND history should be limited to the last 20 messages to manage token usage
```

**SRS-AI-012: Function Calling** [P0]
```
GIVEN the AI provider supports function calling
WHEN the user asks for real-time data (price, RS rating, etc.)
THEN the AI should invoke the appropriate tool function
AND incorporate the result into its response
```

**SRS-AI-013: System Prompt Loading** [P0]
```
GIVEN the system prompt configuration file exists
WHEN the AI provider initializes
THEN the system prompt should be loaded from config
AND include current portfolio and watchlist data
```

#### 2.1.3 Memory System

**SRS-AI-020: Short-term Memory** [P0]
```
GIVEN a chat session is active
WHEN the session has conversation history
THEN the history should be stored in memory (RAM or Redis)
AND be retrievable for context injection
```

**SRS-AI-021: Long-term Memory Storage** [P1]
```
GIVEN the memory system is configured
WHEN a trading insight is marked for storage
THEN the insight should be persisted to the database
AND be searchable by type, symbol, and date
```

**SRS-AI-022: Semantic Search** [P1]
```
GIVEN trading insights exist in the vector database
WHEN the user asks about past experiences
THEN the system should perform semantic search
AND return the top 5 most relevant insights
```

---

### 2.2 Data Provider Module (`data_provider`)

#### 2.2.1 Price Data

**SRS-DP-001: Current Price Retrieval** [P0]
```
GIVEN a valid stock symbol
WHEN get_current_price(symbol) is called
THEN the current market price should be returned
AND the price should be no older than 15 minutes during market hours
```

**SRS-DP-002: Multi-Provider Fallback** [P0]
```
GIVEN multiple data providers are configured (EODHD, Finnhub, Yahoo)
WHEN the primary provider fails
THEN the system should try the next provider in priority order
AND return data from the first successful provider
```

**SRS-DP-003: Price Caching** [P0]
```
GIVEN a price was recently fetched for a symbol
WHEN the same price is requested within 60 seconds
THEN the cached price should be returned
AND no API call should be made
```

**SRS-DP-004: US Stock Support** [P0]
```
GIVEN a US stock symbol (e.g., "AAPL", "NVDA")
WHEN price data is requested
THEN the price should be returned in USD
AND source should be identified
```

**SRS-DP-005: Hong Kong Stock Support** [P0]
```
GIVEN a Hong Kong stock symbol (e.g., "0700.HK", "5.HK")
WHEN price data is requested
THEN the price should be returned in HKD
AND the symbol should be normalized to 4-digit format (e.g., "0005.HK")
```

**SRS-DP-006: Japanese Stock Support** [P1]
```
GIVEN a Japanese stock symbol (e.g., "7974.T")
WHEN price data is requested
THEN the price should be returned in JPY
AND J-Quants provider should be used when available
```

#### 2.2.2 Historical Data

**SRS-DP-010: Historical OHLCV** [P0]
```
GIVEN a valid stock symbol and date range
WHEN get_historical_data(symbol, start_date, end_date) is called
THEN a DataFrame with Open, High, Low, Close, Volume should be returned
AND data should be sorted by date ascending
```

**SRS-DP-011: Period-based Historical Data** [P0]
```
GIVEN a valid stock symbol and period string
WHEN get_historical_data(symbol, period="6mo") is called
THEN 6 months of historical data should be returned
AND supported periods include: 1mo, 3mo, 6mo, 1y, 2y
```

#### 2.2.3 Quote Data

**SRS-DP-020: Full Quote** [P0]
```
GIVEN a valid stock symbol
WHEN get_quote(symbol) is called
THEN a dictionary should be returned containing:
  - current price
  - open, high, low
  - previous close
  - change and change_percent
  - volume
  - currency
  - source
```

---

### 2.3 Scanner Module (`scanner`)

#### 2.3.1 Pattern Detection Interface

**SRS-SC-001: Scanner Interface** [P0]
```
GIVEN a class implements the ScannerPlugin interface
WHEN the scanner is registered with the system
THEN it should be callable via scan_watchlist()
AND return a list of detected setups
```

**SRS-SC-002: Setup Result Format** [P0]
```
GIVEN a pattern is detected by any scanner
WHEN the setup is returned
THEN it should contain:
  - symbol
  - pattern_type
  - entry_price
  - stop_price
  - target_price
  - confidence_score (0-100)
  - reason (text explanation)
  - timestamp
```

#### 2.3.2 VCP Scanner (Minervini)

**SRS-SC-010: VCP Detection** [P0]
```
GIVEN a stock with 6 months of price history
WHEN the VCP scanner analyzes the stock
THEN it should detect Volatility Contraction Patterns where:
  - Price is within 25% of 52-week high
  - At least 2 contractions with decreasing volatility
  - Volume dry-up during consolidation
  - Price above 50-day and 200-day moving averages
```

**SRS-SC-011: VCP Breakout Trigger** [P0]
```
GIVEN a VCP pattern has been detected
WHEN price breaks above the pivot point on above-average volume
THEN a breakout alert should be generated
AND entry, stop, and target prices should be calculated
```

**SRS-SC-012: RS Rating Calculation** [P0]
```
GIVEN a stock symbol and benchmark (default: SPY)
WHEN calculate_rs_rating(symbol) is called
THEN a rating from 0-100 should be returned
AND the rating should reflect relative performance over 12 months
AND RS >= 90 indicates top 10% performers
```

#### 2.3.3 Pivot Scanner

**SRS-SC-020: Pivot Detection** [P1]
```
GIVEN a stock with proper base formation
WHEN the pivot scanner analyzes the stock
THEN it should detect pivot breakout setups where:
  - Base depth is 15-35%
  - Base length is 4-52 weeks
  - Price is near pivot point (within 5%)
  - Volume is contracting in the base
```

#### 2.3.4 AI Scanner

**SRS-SC-030: AI-Powered Analysis** [P0]
```
GIVEN a stock symbol and current market data
WHEN the AI scanner analyzes the stock
THEN the AI should evaluate VCP, Pivot, and Cup patterns
AND return a structured analysis with entry/stop/target
```

**SRS-SC-031: AI Scanner Rate Limiting** [P0]
```
GIVEN the AI scanner is processing a watchlist
WHEN scanning multiple symbols
THEN there should be a configurable delay between API calls
AND the total scan should not exceed API rate limits
```

---

### 2.4 Portfolio Module (`portfolio`)

#### 2.4.1 Position Management

**SRS-PM-001: Add Position** [P0]
```
GIVEN sufficient cash balance
WHEN add_position(symbol, shares, entry_price, stop_loss, target_price) is called
THEN a new position should be created
AND cash balance should be reduced by (shares * entry_price + fees)
AND invested amount should increase accordingly
```

**SRS-PM-002: Add Position - Insufficient Funds** [P0]
```
GIVEN insufficient cash balance
WHEN add_position() is called with cost exceeding cash
THEN a ValueError should be raised
AND no position should be created
AND cash balance should remain unchanged
```

**SRS-PM-003: Close Position** [P0]
```
GIVEN an existing position
WHEN close_position(symbol, exit_price) is called
THEN the position should be removed
AND cash balance should increase by (shares * exit_price - fees)
AND P&L should be calculated and returned
```

**SRS-PM-004: Update Stop Loss** [P0]
```
GIVEN an existing position
WHEN update_stop(symbol, new_stop) is called
THEN the stop loss should be updated
AND the change should be persisted
```

**SRS-PM-005: Update Target** [P0]
```
GIVEN an existing position
WHEN update_target(symbol, new_target) is called
THEN the target price should be updated
AND the change should be persisted
```

**SRS-PM-006: Duplicate Position Prevention** [P0]
```
GIVEN an existing position for a symbol
WHEN add_position() is called for the same symbol
THEN a ValueError should be raised
AND the existing position should remain unchanged
```

#### 2.4.2 Multi-Currency Support

**SRS-PM-010: Currency Detection** [P0]
```
GIVEN a stock symbol
WHEN the currency is determined
THEN:
  - US stocks (no suffix) → USD
  - .HK stocks → HKD
  - .T stocks → JPY
```

**SRS-PM-011: Currency Conversion** [P0]
```
GIVEN an amount in a foreign currency
WHEN convert_to_base_currency(amount, currency) is called
THEN the amount should be converted to USD using configured exchange rates
```

**SRS-PM-012: Exchange Rate Configuration** [P0]
```
GIVEN a currency configuration file exists
WHEN the portfolio module loads
THEN exchange rates should be loaded from config
AND default rates should be used if config is missing
```

#### 2.4.3 Fee Calculation

**SRS-PM-020: US Market Fees** [P1]
```
GIVEN a US stock transaction
WHEN fees are calculated
THEN commission and SEC/TAF fees should be applied
AND fees should match the configured fee structure
```

**SRS-PM-021: Hong Kong Market Fees** [P1]
```
GIVEN a Hong Kong stock transaction
WHEN fees are calculated
THEN commission, stamp duty, trading fee, and transaction levy should be applied
AND stamp duty should only apply to buy transactions
```

**SRS-PM-022: Japanese Market Fees** [P1]
```
GIVEN a Japanese stock transaction
WHEN fees are calculated
THEN commission and consumption tax should be applied
```

#### 2.4.4 Portfolio Metrics

**SRS-PM-030: Position Metrics** [P0]
```
GIVEN an existing position with current price available
WHEN calculate_position_metrics(symbol) is called
THEN the following should be calculated:
  - Current value (in USD)
  - Unrealized P&L (in USD)
  - P&L percentage
  - Risk (entry - stop) * shares
  - Reward/Risk ratio
  - Distance to stop (%)
  - Distance to target (%)
  - Days held
```

**SRS-PM-031: Portfolio Summary** [P0]
```
GIVEN one or more positions exist
WHEN get_portfolio_summary() is called
THEN the following should be returned:
  - Total positions count
  - Total portfolio value (USD)
  - Total unrealized P&L (USD)
  - Total risk (USD)
  - Portfolio heat (risk / value %)
  - Cash balance
  - Buying power
```

#### 2.4.5 Alert Generation

**SRS-PM-040: Stop Loss Alert** [P0]
```
GIVEN a position where current price <= stop loss
WHEN check_alerts() is called
THEN a "stop_hit" alert should be generated
AND include symbol, current price, stop price, and loss amount
```

**SRS-PM-041: Target Hit Alert** [P0]
```
GIVEN a position where current price >= target price
WHEN check_alerts() is called
THEN a "target_hit" alert should be generated
AND include symbol, current price, target price, and gain amount
```

**SRS-PM-042: Trailing Stop Suggestion** [P1]
```
GIVEN a position with P&L >= +8%
AND stop loss is still below entry price
WHEN check_alerts() is called
THEN a "trailing_stop_suggestion" alert should be generated
AND suggest moving stop to entry price (breakeven)
```

#### 2.4.6 Cash Management

**SRS-PM-050: Deposit Cash** [P0]
```
GIVEN a positive amount
WHEN deposit_cash(amount) is called
THEN cash_balance should increase by amount
AND total_capital should increase by amount
```

**SRS-PM-051: Withdraw Cash** [P0]
```
GIVEN a positive amount <= cash_balance
WHEN withdraw_cash(amount) is called
THEN cash_balance should decrease by amount
AND total_capital should decrease by amount
```

**SRS-PM-052: Withdraw Cash - Insufficient Funds** [P0]
```
GIVEN an amount > cash_balance
WHEN withdraw_cash(amount) is called
THEN a ValueError should be raised
AND balances should remain unchanged
```

---

### 2.5 Bot Interface Module (`bot`)

#### 2.5.1 Telegram Bot

**SRS-BOT-001: Bot Initialization** [P0]
```
GIVEN valid TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID
WHEN the Telegram bot initializes
THEN the bot should connect successfully
AND be ready to receive commands
```

**SRS-BOT-002: Command Handler - /portfolio** [P0]
```
GIVEN the bot is running
WHEN user sends "/portfolio"
THEN the bot should respond with:
  - All positions with current prices
  - P&L for each position
  - Total portfolio value
  - Cash balance
```

**SRS-BOT-003: Command Handler - /scan** [P0]
```
GIVEN the bot is running
WHEN user sends "/scan"
THEN the bot should:
  - Acknowledge the request immediately
  - Run the market scanner
  - Report results when complete
```

**SRS-BOT-004: Command Handler - /add** [P0]
```
GIVEN the bot is running
WHEN user sends "/add AAPL 100 150.00 145.00 165.00"
THEN a position should be added with:
  - Symbol: AAPL
  - Shares: 100
  - Entry: $150.00
  - Stop: $145.00
  - Target: $165.00
AND confirmation should be sent
```

**SRS-BOT-005: Command Handler - /close** [P0]
```
GIVEN an existing position
WHEN user sends "/close AAPL"
THEN the position should be closed
AND P&L should be reported
```

**SRS-BOT-006: Natural Language Processing** [P0]
```
GIVEN the bot is running
WHEN user sends a non-command message
THEN the message should be forwarded to the AI provider
AND the AI response should be sent back to the user
```

**SRS-BOT-007: Chat ID Validation** [P0]
```
GIVEN a message from an unauthorized chat ID
WHEN any command or message is received
THEN the bot should ignore the message
AND optionally log the unauthorized attempt
```

#### 2.5.2 REST API

**SRS-API-001: Health Endpoint** [P0]
```
GIVEN the API server is running
WHEN GET /health is called
THEN HTTP 200 should be returned
AND response should include status, timestamp, and version
```

**SRS-API-002: Portfolio Endpoint** [P0]
```
GIVEN valid API authentication
WHEN GET /api/v1/portfolio is called
THEN the portfolio summary should be returned as JSON
```

**SRS-API-003: Positions Endpoint** [P0]
```
GIVEN valid API authentication
WHEN GET /api/v1/positions is called
THEN all positions should be returned as JSON array
```

**SRS-API-004: Scan Endpoint** [P1]
```
GIVEN valid API authentication
WHEN POST /api/v1/scan is called
THEN a scan should be triggered
AND scan ID should be returned for status polling
```

**SRS-API-005: Authentication** [P0]
```
GIVEN an API request without valid authentication
WHEN any protected endpoint is called
THEN HTTP 401 Unauthorized should be returned
```

#### 2.5.3 Discord Bot

**SRS-DISC-001: Discord Bot Initialization** [P1]
```
GIVEN valid DISCORD_BOT_TOKEN
WHEN the Discord bot initializes
THEN the bot should connect successfully
AND be ready to receive commands
```

**SRS-DISC-002: Discord Command Parity** [P1]
```
GIVEN the Discord bot is running
WHEN any command available in Telegram is used
THEN the same functionality should be available
AND response format should be adapted for Discord
```

---

### 2.6 Strategy Plugin Module (`strategy`)

#### 2.6.1 Plugin Interface

**SRS-STR-001: Plugin Interface Definition** [P0]
```
GIVEN the StrategyPlugin abstract base class
WHEN a new strategy is implemented
THEN it must implement:
  - name: str (property)
  - description: str (property)
  - scan(symbols: List[str]) -> List[Setup]
  - configure(params: Dict) -> None
  - validate_setup(setup: Setup) -> bool
```

**SRS-STR-002: Plugin Discovery** [P0]
```
GIVEN strategy plugins exist in the plugins directory
WHEN the system starts
THEN all valid plugins should be discovered
AND registered with the strategy manager
```

**SRS-STR-003: Plugin Enable/Disable** [P0]
```
GIVEN a registered strategy plugin
WHEN the plugin is disabled via configuration
THEN the plugin should not run during scans
AND should be excluded from scan results
```

#### 2.6.2 Plugin Configuration

**SRS-STR-010: Parameter Configuration** [P0]
```
GIVEN a strategy plugin with configurable parameters
WHEN configure(params) is called
THEN the parameters should be validated
AND applied to the strategy instance
```

**SRS-STR-011: Configuration Persistence** [P1]
```
GIVEN strategy configuration changes
WHEN the configuration is saved
THEN it should persist to the config file
AND be loaded on next startup
```

---

### 2.7 Scheduler Module (`scheduler`)

#### 2.7.1 Scheduled Jobs

**SRS-SCH-001: Market Scan Job** [P0]
```
GIVEN the scheduler is running
WHEN the configured scan interval elapses
THEN the market scanner should run
AND alerts should be generated for detected setups
```

**SRS-SCH-002: Portfolio Monitor Job** [P0]
```
GIVEN the scheduler is running
WHEN the portfolio monitor interval elapses
THEN all positions should be checked for alerts
AND stop/target hit alerts should be sent
```

**SRS-SCH-003: Daily Standup Job** [P1]
```
GIVEN the scheduler is running
WHEN the configured standup time is reached
THEN a daily summary should be generated
AND sent to configured channels
```

**SRS-SCH-004: Job Configuration** [P0]
```
GIVEN job intervals are defined in configuration
WHEN the scheduler starts
THEN jobs should run at configured intervals
AND intervals should be configurable without code changes
```

---

### 2.8 Storage Module (`storage`)

#### 2.8.1 Database Abstraction

**SRS-STO-001: Database Interface** [P0]
```
GIVEN the DatabaseInterface abstract class
WHEN a database backend is implemented
THEN it must implement:
  - connect() -> None
  - disconnect() -> None
  - save_position(position: Dict) -> None
  - get_positions() -> List[Dict]
  - save_account(account: Dict) -> None
  - get_account() -> Dict
```

**SRS-STO-002: SQLite Backend** [P0]
```
GIVEN SQLite is configured as the database backend
WHEN database operations are performed
THEN data should be persisted to the SQLite file
AND be recoverable after restart
```

**SRS-STO-003: PostgreSQL Backend** [P1]
```
GIVEN PostgreSQL is configured as the database backend
WHEN database operations are performed
THEN data should be persisted to PostgreSQL
AND support concurrent connections
```

#### 2.8.2 Caching

**SRS-STO-010: Cache Interface** [P0]
```
GIVEN the CacheInterface abstract class
WHEN a cache backend is implemented
THEN it must implement:
  - get(key: str) -> Optional[Any]
  - set(key: str, value: Any, ttl: int) -> None
  - delete(key: str) -> None
  - exists(key: str) -> bool
```

**SRS-STO-011: In-Memory Cache** [P0]
```
GIVEN in-memory cache is the default
WHEN cache operations are performed
THEN data should be stored in memory
AND respect TTL expiration
```

**SRS-STO-012: Redis Cache** [P1]
```
GIVEN Redis is configured as the cache backend
WHEN cache operations are performed
THEN data should be stored in Redis
AND be shared across instances
```

---

## 3. Integration Specifications

### 3.1 Startup Sequence

**SRS-INT-001: System Startup** [P0]
```
GIVEN all required environment variables are set
WHEN the application starts
THEN components should initialize in order:
  1. Configuration loading
  2. Database connection
  3. Cache initialization
  4. Data providers initialization
  5. AI provider initialization
  6. Strategy plugins loading
  7. Bot interfaces startup
  8. Scheduler startup
AND startup message should be sent to configured channels
```

### 3.2 Graceful Shutdown

**SRS-INT-002: System Shutdown** [P0]
```
GIVEN the application receives a shutdown signal (SIGTERM, SIGINT)
WHEN shutdown is initiated
THEN components should stop in reverse order:
  1. Scheduler stopped
  2. Bot interfaces stopped
  3. Active scans completed or cancelled
  4. Database connections closed
  5. Cache flushed if needed
AND no data should be lost
```

### 3.3 Error Handling

**SRS-INT-010: API Error Recovery** [P0]
```
GIVEN an external API call fails
WHEN the error is caught
THEN the error should be logged with context
AND retry should be attempted with exponential backoff
AND user should be notified if all retries fail
```

**SRS-INT-011: Unhandled Exception** [P0]
```
GIVEN an unhandled exception occurs
WHEN the exception propagates
THEN it should be caught at the top level
AND logged with full stack trace
AND alert should be sent to admin channel
AND system should attempt to continue if possible
```

---

## 4. Configuration Specifications

### 4.1 Environment Variables

**SRS-CFG-001: Required Environment Variables** [P0]
```
GIVEN the application starts
WHEN environment is validated
THEN the following must be present:
  - TELEGRAM_BOT_TOKEN (if Telegram enabled)
  - TELEGRAM_CHAT_ID (if Telegram enabled)
  - At least one AI provider key (GROK_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY)
AND missing required variables should prevent startup
```

### 4.2 Configuration Files

**SRS-CFG-010: Configuration File Format** [P0]
```
GIVEN configuration files exist
WHEN they are loaded
THEN they should be in YAML format
AND support environment variable substitution (${VAR_NAME})
```

**SRS-CFG-011: Configuration Validation** [P0]
```
GIVEN a configuration file is loaded
WHEN validation runs
THEN schema validation should be performed
AND invalid configuration should prevent startup
AND validation errors should be clearly reported
```

---

## 5. Testing Requirements

### 5.1 Unit Test Coverage

**SRS-TST-001: Minimum Coverage** [P0]
```
GIVEN the test suite runs
WHEN coverage is measured
THEN code coverage should be >= 80%
AND critical modules should have >= 90% coverage
```

### 5.2 Integration Tests

**SRS-TST-010: API Integration Tests** [P1]
```
GIVEN external API mocks are configured
WHEN integration tests run
THEN all API interactions should be verified
AND error scenarios should be tested
```

### 5.3 End-to-End Tests

**SRS-TST-020: Bot Command Tests** [P1]
```
GIVEN a test bot instance
WHEN commands are sent
THEN responses should match expected output
AND timing requirements should be met
```

---

## 5. VCP Alert System Specifications

### 5.1 Alert Data Models

**SRS-ALERT-001: Alert Entity** [P0]
```
GIVEN an alert is created in the system
THEN it must contain:
  - id: UUID (unique identifier)
  - symbol: str (stock symbol)
  - alert_type: enum (CONTRACTION | PRE_ALERT | TRADE)
  - state: enum (PENDING | NOTIFIED | CONVERTED | EXPIRED | COMPLETED)
  - created_at: datetime (when alert was generated)
  - updated_at: datetime (last state change)
  - converted_at: datetime (when converted to next stage, nullable)
  - expired_at: datetime (when alert expired, nullable)
  - parent_alert_id: UUID (link to parent alert, nullable)
  - pattern_snapshot: JSON (VCP pattern state at alert time)
  - trigger_price: float (price when alert triggered)
  - pivot_price: float (pivot price at alert time)
  - distance_to_pivot_pct: float (distance from pivot)
  - score: float (pattern score at alert time)
```

**SRS-ALERT-002: Alert Chain** [P0]
```
GIVEN a trade alert exists
WHEN the alert chain is queried
THEN the system should return:
  - The originating contraction alert (if any)
  - All pre-alerts in the sequence
  - The trade alert
  - Total lead time from first alert to trade
```

### 5.2 VCP Detector

**SRS-VCP-001: Pattern Detection** [P0]
```
GIVEN a DataFrame with OHLCV data for a symbol
WHEN analyze(df) is called
THEN it should return Optional[VCPPattern] containing:
  - contractions: List of detected contractions with dates and percentages
  - pivot_price: float
  - overall_score: float (0-100)
  - entry_signals: List of potential entry types (handle, pivot_breakout)
  - is_valid: bool
AND return None if no valid pattern detected
```

**SRS-VCP-002: Contraction Validation** [P0]
```
GIVEN a potential VCP pattern
WHEN contractions are validated
THEN each contraction must have:
  - High price > Low price
  - Progressive tightening (each contraction smaller than previous)
  - Time proximity (max 30 trading days between contractions)
  - No staircase pattern (later highs not stepping higher)
```

### 5.3 Alert Manager

**SRS-AM-001: Contraction Alert Generation** [P0]
```
GIVEN a VCP pattern with num_contractions >= 2
AND pattern overall_score >= 60
AND no existing active contraction alert for this symbol
WHEN check_contraction_alert() is called
THEN a new CONTRACTION alert should be created
AND persisted to the repository
AND subscribers should be notified
```

**SRS-AM-002: Pre-Alert Generation** [P0]
```
GIVEN a valid VCP pattern
AND current price is within 3% of pivot price
AND no existing active pre-alert for this symbol/date
WHEN check_pre_alert() is called
THEN a new PRE_ALERT should be created
AND any active contraction alert should be marked CONVERTED
AND persisted to the repository
AND subscribers should be notified
```

**SRS-AM-003: Trade Alert Generation** [P0]
```
GIVEN a valid VCP pattern with entry signal triggered
AND breakout criteria met (volume, price action)
WHEN check_trade_alert() is called
THEN a new TRADE alert should be created
AND any active pre-alert should be marked CONVERTED
AND persisted to the repository
AND subscribers should be notified
```

**SRS-AM-004: Alert Deduplication** [P0]
```
GIVEN an alert already exists for symbol/type/date
WHEN a new alert would be created with same parameters
THEN no duplicate alert should be created
AND the existing alert should be returned
```

**SRS-AM-005: Alert Expiration** [P0]
```
GIVEN a PENDING contraction alert older than 20 trading days
OR a PENDING pre-alert older than 5 trading days
WHEN expire_stale_alerts() is called
THEN the alert state should change to EXPIRED
AND expired_at timestamp should be set
```

**SRS-AM-006: Alert History Query** [P0]
```
GIVEN alerts exist in the repository
WHEN get_alert_history(symbol, alert_type, start_date, end_date, state) is called
THEN alerts matching all provided filters should be returned
AND results should be sorted by created_at descending
```

**SRS-AM-007: Subscriber Notification** [P0]
```
GIVEN one or more handlers are subscribed via subscribe()
WHEN a new alert is created
THEN all subscribed handlers should be called with the alert
AND notification should be asynchronous (non-blocking)
```

### 5.4 Alert Repository

**SRS-REPO-001: Alert Persistence** [P0]
```
GIVEN a valid Alert object
WHEN save(alert) is called
THEN the alert should be persisted to storage
AND be retrievable via get_by_id(alert_id)
```

**SRS-REPO-002: Alert Update** [P0]
```
GIVEN an existing alert
WHEN update(alert) is called with modified state
THEN the alert should be updated in storage
AND updated_at timestamp should be refreshed
```

**SRS-REPO-003: Query by Symbol** [P0]
```
GIVEN alerts exist for multiple symbols
WHEN query(symbol="NVDA") is called
THEN only alerts for NVDA should be returned
```

**SRS-REPO-004: Query by State** [P0]
```
GIVEN alerts exist in various states
WHEN query(state=PENDING) is called
THEN only PENDING alerts should be returned
```

**SRS-REPO-005: Query by Date Range** [P0]
```
GIVEN alerts exist across different dates
WHEN query(start_date, end_date) is called
THEN only alerts with created_at in range should be returned
```

### 5.5 Notification Hub

**SRS-NOTIFY-001: Channel Registration** [P0]
```
GIVEN a NotificationChannel implementation
WHEN register_channel(channel) is called
THEN the channel should be available for dispatching
```

**SRS-NOTIFY-002: Alert Dispatch** [P0]
```
GIVEN one or more channels are registered
WHEN dispatch(alert) is called
THEN each channel's send(alert) method should be called
AND failures in one channel should not block others
```

**SRS-NOTIFY-003: Channel Filtering** [P0]
```
GIVEN channels have alert_type preferences configured
WHEN dispatch(alert) is called
THEN only channels configured for that alert_type should receive it
```

### 5.6 Alert System Integration

**SRS-SYS-001: Symbol Processing** [P0]
```
GIVEN a symbol and its price DataFrame
WHEN process_symbol(symbol, df) is called on VCPAlertSystem
THEN the detector should analyze for patterns
AND AlertManager should check for all alert conditions
AND any new alerts should trigger notifications
```

**SRS-SYS-002: Scanner Loop** [P1]
```
GIVEN a list of symbols and scan interval
WHEN run_scanner(symbols, interval) is called
THEN each symbol should be processed at the specified interval
AND the loop should handle errors gracefully without stopping
AND scan progress should be logged
```

**SRS-SYS-003: Startup Initialization** [P0]
```
GIVEN the VCPAlertSystem is initialized
WHEN start() is called
THEN:
  1. Repository connection should be established
  2. Stale alerts should be expired
  3. Active alerts should be loaded into memory
  4. Notification channels should be initialized
```

---

## 6. Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-11-28 | Claude | Initial draft |

---

## 7. Appendices

### Appendix A: Interface Definitions

```python
# AI Provider Interface
class AIProvider(ABC):
    @abstractmethod
    async def chat(self, message: str, context: Optional[Dict] = None) -> str:
        """Send message to AI and get response."""
        pass

    @abstractmethod
    async def chat_with_tools(self, message: str, tools: List[Dict]) -> Tuple[str, List[Dict]]:
        """Chat with function calling support."""
        pass

    @abstractmethod
    def clear_history(self) -> None:
        """Clear conversation history."""
        pass

# Scanner Plugin Interface
class ScannerPlugin(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    async def scan(self, symbols: List[str]) -> List[Dict]:
        """Scan symbols and return detected setups."""
        pass

    @abstractmethod
    def configure(self, params: Dict) -> None:
        """Configure scanner parameters."""
        pass

# Database Interface
class DatabaseInterface(ABC):
    @abstractmethod
    def connect(self) -> None:
        pass

    @abstractmethod
    def save_positions(self, positions: List[Dict]) -> None:
        pass

    @abstractmethod
    def get_positions(self) -> List[Dict]:
        pass
```

### Appendix B: Error Codes

| Code | Module | Description |
|------|--------|-------------|
| E1001 | AI | Provider initialization failed |
| E1002 | AI | API call failed |
| E1003 | AI | Rate limit exceeded |
| E2001 | Data | Price fetch failed |
| E2002 | Data | Historical data unavailable |
| E3001 | Portfolio | Insufficient funds |
| E3002 | Portfolio | Position not found |
| E4001 | Bot | Authentication failed |
| E4002 | Bot | Command parse error |
| E5001 | Storage | Database connection failed |
| E5002 | Storage | Data integrity error |
