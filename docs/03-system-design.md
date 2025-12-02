# System Design Specification

## AI Trading Assistant

**Version:** 1.2.0
**Date:** 2025-12-02
**Status:** Active (VCPAlertSystem with Staleness Detection)

---

## 1. Introduction

### 1.1 Purpose

This document describes the technical architecture and design of the AI Trading Assistant system. It provides a blueprint for developers to implement the system according to the requirements specified in the System Requirement Specification.

### 1.2 Scope

This design covers:
- System architecture and component interactions
- Module designs and interfaces
- Data models and storage strategies
- Deployment architecture
- Security considerations

### 1.3 Design Principles

1. **Modularity**: Loosely coupled components with clear interfaces
2. **Extensibility**: Plugin architecture for strategies and providers
3. **Scalability**: Start simple, scale when needed
4. **Testability**: Design for unit and integration testing
5. **Configuration over Code**: Behavior changes via config, not code

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACES                                 │
├─────────────────┬─────────────────┬─────────────────┬──────────────────────┤
│   Telegram Bot  │   Discord Bot   │    Web UI       │      REST API        │
└────────┬────────┴────────┬────────┴────────┬────────┴──────────┬───────────┘
         │                 │                 │                   │
         └─────────────────┴─────────────────┴───────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CORE APPLICATION                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  AI Engine  │  │  Scanner    │  │  Portfolio  │  │  Scheduler  │        │
│  │  Manager    │  │  Manager    │  │  Manager    │  │             │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         │                │                │                │               │
│         └────────────────┴────────────────┴────────────────┘               │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        SERVICE LAYER                                 │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │   │
│  │  │ Alert       │  │ Notification│  │ Analytics   │                  │   │
│  │  │ Service     │  │ Service     │  │ Service     │                  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EXTERNAL INTEGRATIONS                              │
├─────────────────┬─────────────────┬─────────────────┬──────────────────────┤
│   AI Providers  │  Data Providers │    Storage      │      Cache           │
│  ┌───────────┐  │  ┌───────────┐  │  ┌───────────┐  │  ┌───────────┐       │
│  │   Grok    │  │  │  EODHD    │  │  │  SQLite   │  │  │  Memory   │       │
│  │  OpenAI   │  │  │  Finnhub  │  │  │ PostgreSQL│  │  │   Redis   │       │
│  │  Claude   │  │  │  Yahoo    │  │  │TimescaleDB│  │  └───────────┘       │
│  │  Ollama   │  │  │  J-Quants │  │  └───────────┘  │                      │
│  └───────────┘  │  └───────────┘  │                 │                      │
└─────────────────┴─────────────────┴─────────────────┴──────────────────────┘
```

### 2.2 Component Overview

| Component | Responsibility | Key Dependencies |
|-----------|---------------|------------------|
| Telegram Bot | User commands via Telegram | AI Engine, Portfolio |
| Discord Bot | User commands via Discord | AI Engine, Portfolio |
| Web UI | Dashboard and analytics | REST API |
| REST API | Programmatic access | All managers |
| AI Engine Manager | LLM provider abstraction | AI Providers |
| Scanner Manager | Strategy plugin orchestration | Data Providers, AI Engine |
| Portfolio Manager | Position and cash management | Storage |
| Scheduler | Background job execution | All managers |
| Alert Service | Alert generation and tracking | Notification Service |
| Notification Service | Multi-channel messaging | Bot interfaces |
| Analytics Service | Performance metrics | Storage |

---

## 3. Module Design

### 3.1 AI Engine Module

#### 3.1.1 Class Diagram

```
┌──────────────────────────────────────┐
│         AIEngineManager              │
├──────────────────────────────────────┤
│ - providers: Dict[str, AIProvider]   │
│ - active_provider: str               │
│ - fallback_order: List[str]          │
│ - conversation_store: ConversationStore │
├──────────────────────────────────────┤
│ + register_provider(provider)        │
│ + set_active_provider(name)          │
│ + chat(message, context) -> str      │
│ + chat_with_tools(message, tools)    │
│ + get_available_providers() -> List  │
└──────────────────────────────────────┘
                    │
                    │ uses
                    ▼
┌──────────────────────────────────────┐
│       <<interface>> AIProvider       │
├──────────────────────────────────────┤
│ + name: str                          │
│ + is_available: bool                 │
├──────────────────────────────────────┤
│ + initialize(config) -> None         │
│ + chat(message, history) -> str      │
│ + chat_with_tools(message, tools)    │
│ + get_token_usage() -> Dict          │
└──────────────────────────────────────┘
          △                  △
          │                  │
    ┌─────┴─────┐     ┌──────┴──────┐
    │           │     │             │
┌───┴───┐ ┌────┴────┐ ┌─────┴─────┐
│GrokAI │ │OpenAIAPI│ │ClaudeAPI  │
│Provider│ │Provider │ │Provider   │
└───────┘ └─────────┘ └───────────┘
```

#### 3.1.2 Conversation Store

```
┌──────────────────────────────────────┐
│      <<interface>> ConversationStore │
├──────────────────────────────────────┤
│ + store_message(user_id, role, msg)  │
│ + get_history(user_id, limit) -> List│
│ + clear_history(user_id)             │
│ + store_insight(type, data)          │
│ + search_insights(query) -> List     │
└──────────────────────────────────────┘
          △                  △
          │                  │
┌─────────┴──────┐  ┌────────┴────────┐
│MemoryStore     │  │ RedisStore      │
│(development)   │  │ (production)    │
└────────────────┘  └─────────────────┘
```

#### 3.1.3 Tool Definitions

The AI engine supports function calling with these built-in tools:

| Tool | Description | Parameters |
|------|-------------|------------|
| `get_stock_quote` | Get real-time price | symbol: str |
| `get_rs_rating` | Calculate RS rating | symbol: str |
| `get_market_data` | Get SPY/VIX data | (none) |
| `get_historical_data` | Get OHLCV data | symbol, start_date, end_date |
| `detect_vcp_pattern` | Detect VCP pattern | symbol, start_date, end_date |
| `store_trading_insight` | Store insight | type, symbol, description |
| `search_trading_memory` | Search past insights | query, limit |

### 3.2 Data Provider Module

#### 3.2.1 Class Diagram

```
┌──────────────────────────────────────┐
│         DataProviderManager          │
├──────────────────────────────────────┤
│ - providers: List[DataProvider]      │
│ - cache: CacheInterface              │
│ - priority_order: List[str]          │
├──────────────────────────────────────┤
│ + get_current_price(symbol) -> float │
│ + get_quote(symbol) -> Dict          │
│ + get_historical(symbol, period)     │
│ + test_connections() -> Dict         │
└──────────────────────────────────────┘
                    │
                    │ uses
                    ▼
┌──────────────────────────────────────┐
│     <<interface>> DataProvider       │
├──────────────────────────────────────┤
│ + name: str                          │
│ + supported_markets: List[str]       │
├──────────────────────────────────────┤
│ + get_quote(symbol) -> Optional[Dict]│
│ + get_historical(symbol, period)     │
│ + is_available() -> bool             │
└──────────────────────────────────────┘
          △           △           △
          │           │           │
    ┌─────┴───┐ ┌─────┴───┐ ┌─────┴───┐
    │EODHD    │ │Finnhub  │ │Yahoo    │
    │Provider │ │Provider │ │Provider │
    └─────────┘ └─────────┘ └─────────┘
```

#### 3.2.2 Market Symbol Handling

| Market | Symbol Format | Example | Currency |
|--------|--------------|---------|----------|
| US | TICKER | AAPL, NVDA | USD |
| Hong Kong | NNNN.HK | 0700.HK, 0005.HK | HKD |
| Japan | NNNN.T | 7974.T, 6758.T | JPY |

Symbol normalization rules:
- HK: Pad to 4 digits (5.HK → 0005.HK)
- JP: Verify 4 digits
- US: No modification

### 3.3 Scanner Module

#### 3.3.1 Plugin Architecture

```
┌──────────────────────────────────────┐
│          ScannerManager              │
├──────────────────────────────────────┤
│ - plugins: Dict[str, ScannerPlugin]  │
│ - enabled_plugins: Set[str]          │
│ - data_provider: DataProviderManager │
│ - ai_engine: AIEngineManager         │
├──────────────────────────────────────┤
│ + register_plugin(plugin)            │
│ + enable_plugin(name)                │
│ + disable_plugin(name)               │
│ + scan_watchlist(symbols) -> List    │
│ + scan_single(symbol) -> Optional    │
└──────────────────────────────────────┘
                    │
                    │ manages
                    ▼
┌──────────────────────────────────────┐
│      <<interface>> ScannerPlugin     │
├──────────────────────────────────────┤
│ + name: str                          │
│ + description: str                   │
│ + version: str                       │
├──────────────────────────────────────┤
│ + scan(symbols, data_provider) -> List│
│ + configure(params: Dict)            │
│ + get_config_schema() -> Dict        │
└──────────────────────────────────────┘
          △           △           △
          │           │           │
    ┌─────┴───┐ ┌─────┴───┐ ┌─────┴───┐
    │VCP      │ │Pivot    │ │AI       │
    │Scanner  │ │Scanner  │ │Scanner  │
    └─────────┘ └─────────┘ └─────────┘
```

#### 3.3.2 Setup Data Model

```python
@dataclass
class Setup:
    symbol: str
    pattern_type: str
    entry_price: float
    stop_price: float
    target_price: float
    current_price: float
    confidence: int  # 0-100
    reason: str
    scanner_name: str
    timestamp: datetime
    metadata: Dict[str, Any]  # Scanner-specific data

    @property
    def risk_per_share(self) -> float:
        return self.entry_price - self.stop_price

    @property
    def reward_risk_ratio(self) -> float:
        risk = self.entry_price - self.stop_price
        reward = self.target_price - self.entry_price
        return reward / risk if risk > 0 else 0
```

### 3.4 Portfolio Module

#### 3.4.1 Class Diagram

```
┌──────────────────────────────────────┐
│         PortfolioManager             │
├──────────────────────────────────────┤
│ - positions: List[Position]          │
│ - account: Account                   │
│ - storage: StorageInterface          │
│ - data_provider: DataProviderManager │
│ - fee_calculator: FeeCalculator      │
│ - currency_converter: CurrencyConverter│
├──────────────────────────────────────┤
│ + add_position(...)                  │
│ + close_position(symbol, price)      │
│ + update_stop(symbol, new_stop)      │
│ + update_target(symbol, new_target)  │
│ + get_position(symbol) -> Position   │
│ + get_all_positions() -> List        │
│ + get_summary() -> PortfolioSummary  │
│ + check_alerts() -> List[Alert]      │
│ + deposit_cash(amount)               │
│ + withdraw_cash(amount)              │
└──────────────────────────────────────┘
```

#### 3.4.2 Data Models

```python
@dataclass
class Position:
    symbol: str
    shares: int
    entry_date: date
    entry_price: float
    entry_price_usd: float
    stop_loss: float
    target_price: float
    currency: str
    notes: str
    entry_fees: float
    entry_fees_usd: float

@dataclass
class Account:
    total_capital: float
    cash_balance: float
    invested: float

@dataclass
class PortfolioSummary:
    total_positions: int
    total_value: float
    total_risk: float
    total_pnl: float
    total_pnl_pct: float
    portfolio_heat: float
    cash_balance: float
    buying_power: float
    positions: List[PositionMetrics]

@dataclass
class Alert:
    type: str  # stop_hit, target_hit, trailing_stop_suggestion
    symbol: str
    current_price: float
    trigger_price: float
    pnl: float
    pnl_pct: float
    message: str
    timestamp: datetime
```

### 3.5 Bot Interface Module

#### 3.5.1 Command Router Architecture

```
┌──────────────────────────────────────┐
│          BotInterface                │
│        (Abstract Base)               │
├──────────────────────────────────────┤
│ - command_router: CommandRouter      │
│ - ai_engine: AIEngineManager         │
│ - portfolio: PortfolioManager        │
│ - scanner: ScannerManager            │
├──────────────────────────────────────┤
│ + start()                            │
│ + stop()                             │
│ + send_message(channel, message)     │
│ + send_alert(alert: Alert)           │
└──────────────────────────────────────┘
          △           △
          │           │
    ┌─────┴────┐ ┌────┴─────┐
    │Telegram  │ │Discord   │
    │Bot       │ │Bot       │
    └──────────┘ └──────────┘

┌──────────────────────────────────────┐
│          CommandRouter               │
├──────────────────────────────────────┤
│ - handlers: Dict[str, CommandHandler]│
├──────────────────────────────────────┤
│ + register(command, handler)         │
│ + route(command, args, context)      │
│ + get_help() -> str                  │
└──────────────────────────────────────┘
```

#### 3.5.2 Command Mapping

| Command | Handler | Description |
|---------|---------|-------------|
| /start | StartHandler | Welcome message |
| /help | HelpHandler | List commands |
| /portfolio | PortfolioHandler | Show positions |
| /cash | CashHandler | Show cash balance |
| /add | AddPositionHandler | Add position |
| /close | ClosePositionHandler | Close position |
| /update | UpdatePositionHandler | Update stop/target |
| /scan | ScanHandler | Trigger scan |
| /watchlist | WatchlistHandler | Show/manage watchlist |
| /alerts | AlertsHandler | Show pending alerts |
| /health | HealthHandler | System status |
| (text) | ChatHandler | AI conversation |

### 3.6 Storage Module

#### 3.6.1 Storage Abstraction

```
┌──────────────────────────────────────┐
│      <<interface>> StorageInterface  │
├──────────────────────────────────────┤
│ + connect()                          │
│ + disconnect()                       │
│ + save_positions(List[Position])     │
│ + get_positions() -> List[Position]  │
│ + save_account(Account)              │
│ + get_account() -> Account           │
│ + save_trade(Trade)                  │
│ + get_trades(filters) -> List[Trade] │
│ + save_watchlist(List[str])          │
│ + get_watchlist() -> List[str]       │
└──────────────────────────────────────┘
          △           △           △
          │           │           │
    ┌─────┴───┐ ┌─────┴───┐ ┌─────┴───┐
    │SQLite   │ │Postgres │ │Timescale│
    │Storage  │ │Storage  │ │Storage  │
    └─────────┘ └─────────┘ └─────────┘
```

#### 3.6.2 Database Schema

```sql
-- Positions table
CREATE TABLE positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol VARCHAR(20) NOT NULL UNIQUE,
    shares INTEGER NOT NULL,
    entry_date DATE NOT NULL,
    entry_price DECIMAL(12,4) NOT NULL,
    entry_price_usd DECIMAL(12,4) NOT NULL,
    stop_loss DECIMAL(12,4) NOT NULL,
    target_price DECIMAL(12,4) NOT NULL,
    currency VARCHAR(3) NOT NULL DEFAULT 'USD',
    notes TEXT,
    entry_fees DECIMAL(10,4) DEFAULT 0,
    entry_fees_usd DECIMAL(10,4) DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Account table
CREATE TABLE account (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    total_capital DECIMAL(14,2) NOT NULL,
    cash_balance DECIMAL(14,2) NOT NULL,
    invested DECIMAL(14,2) NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Trades history table
CREATE TABLE trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol VARCHAR(20) NOT NULL,
    action VARCHAR(10) NOT NULL,  -- BUY, SELL
    shares INTEGER NOT NULL,
    price DECIMAL(12,4) NOT NULL,
    price_usd DECIMAL(12,4) NOT NULL,
    fees DECIMAL(10,4) DEFAULT 0,
    fees_usd DECIMAL(10,4) DEFAULT 0,
    currency VARCHAR(3) NOT NULL,
    pnl DECIMAL(12,4),
    pnl_pct DECIMAL(8,4),
    trade_date TIMESTAMP NOT NULL,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Watchlist table
CREATE TABLE watchlist (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol VARCHAR(20) NOT NULL UNIQUE,
    added_date DATE NOT NULL,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Alerts table
CREATE TABLE alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol VARCHAR(20) NOT NULL,
    alert_type VARCHAR(30) NOT NULL,
    trigger_price DECIMAL(12,4),
    current_price DECIMAL(12,4),
    message TEXT,
    status VARCHAR(20) DEFAULT 'pending',  -- pending, acknowledged, expired
    acknowledged_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Trading insights table (for AI memory)
CREATE TABLE trading_insights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    insight_type VARCHAR(30) NOT NULL,
    symbol VARCHAR(20),
    description TEXT NOT NULL,
    outcome VARCHAR(50),
    key_learnings TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX idx_positions_symbol ON positions(symbol);
CREATE INDEX idx_trades_symbol ON trades(symbol);
CREATE INDEX idx_trades_date ON trades(trade_date);
CREATE INDEX idx_alerts_symbol_status ON alerts(symbol, status);
CREATE INDEX idx_insights_type ON trading_insights(insight_type);
```

### 3.7 Cache Module

#### 3.7.1 Cache Strategy

| Data Type | TTL | Backend |
|-----------|-----|---------|
| Current Price | 60s | Memory/Redis |
| Historical Data | 1h | Memory/Redis |
| RS Rating | 4h | Memory/Redis |
| AI Response | No cache | - |
| Config | Until change | Memory |

#### 3.7.2 Cache Interface

```python
class CacheInterface(ABC):
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: int = 300) -> None:
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        pass

    @abstractmethod
    def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern. Returns count deleted."""
        pass
```

---

## 4. Configuration Design

### 4.1 Configuration Structure

```
config/
├── config.yaml           # Main configuration
├── strategies/
│   ├── vcp.yaml         # VCP strategy config
│   ├── pivot.yaml       # Pivot strategy config
│   └── ai_scanner.yaml  # AI scanner config
├── providers/
│   ├── ai.yaml          # AI provider settings
│   └── data.yaml        # Data provider settings
├── fees/
│   ├── us.yaml          # US market fees
│   ├── hk.yaml          # HK market fees
│   └── jp.yaml          # JP market fees
└── currency.yaml        # Exchange rates
```

### 4.2 Main Configuration Schema

```yaml
# config/config.yaml
app:
  name: "AI Trading Assistant"
  version: "1.0.0"
  environment: "development"  # development, staging, production
  log_level: "INFO"

interfaces:
  telegram:
    enabled: true
    token: "${TELEGRAM_BOT_TOKEN}"
    chat_id: "${TELEGRAM_CHAT_ID}"

  discord:
    enabled: false
    token: "${DISCORD_BOT_TOKEN}"
    guild_id: "${DISCORD_GUILD_ID}"

  rest_api:
    enabled: true
    host: "0.0.0.0"
    port: 8000
    api_key: "${API_KEY}"

ai:
  default_provider: "grok"
  fallback_order: ["grok", "openai", "claude"]
  max_tokens: 1000
  temperature: 0.7

scheduler:
  market_scan:
    enabled: true
    interval_seconds: 28800  # 8 hours
  portfolio_monitor:
    enabled: true
    interval_seconds: 3600  # 1 hour
  daily_standup:
    enabled: true
    time: "09:00"
    timezone: "Asia/Hong_Kong"

storage:
  type: "sqlite"  # sqlite, postgresql
  sqlite:
    path: "data/trading.db"
  postgresql:
    host: "${POSTGRES_HOST}"
    port: 5432
    database: "${POSTGRES_DB}"
    user: "${POSTGRES_USER}"
    password: "${POSTGRES_PASSWORD}"

cache:
  type: "memory"  # memory, redis
  redis:
    url: "${REDIS_URL}"

portfolio:
  base_currency: "USD"
  max_risk_per_trade_pct: 2.0
  trailing_stop_activation_pct: 8.0
```

### 4.3 Strategy Configuration Schema

```yaml
# config/strategies/vcp.yaml
name: "vcp"
display_name: "Minervini VCP Scanner"
enabled: true
version: "1.0.0"

parameters:
  rs_rating_min: 90
  distance_from_52w_high_max: 0.25
  contraction_threshold: 0.20
  volume_dry_up_ratio: 0.5
  min_consolidation_weeks: 4
  pivot_breakout_volume_multiplier: 1.5

filters:
  price_above_50ma: true
  price_above_200ma: true

alerts:
  send_on_breakout: true
  min_confidence: 70
```

---

## 5. Deployment Architecture

### 5.1 Development Deployment

```
┌─────────────────────────────────────┐
│           Developer Machine         │
├─────────────────────────────────────┤
│  ┌─────────────────────────────┐    │
│  │    Python Application       │    │
│  │    (uvicorn + bot)          │    │
│  └──────────────┬──────────────┘    │
│                 │                   │
│  ┌──────────────┴──────────────┐    │
│  │       SQLite Database        │    │
│  │       (data/trading.db)      │    │
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
```

### 5.2 Production Deployment (Docker)

```
┌──────────────────────────────────────────────────────────────┐
│                    Docker Compose Stack                       │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │   trading   │  │   redis     │  │  postgres   │          │
│  │    app      │  │             │  │             │          │
│  │  (Python)   │  │  (Cache)    │  │ (Database)  │          │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘          │
│         │                │                │                  │
│         └────────────────┴────────────────┘                  │
│                          │                                   │
│                    Internal Network                          │
│                                                              │
└──────────────────────────────────────────────────────────────┘
            │
            │ Ports: 8000 (API), 443 (webhook)
            ▼
      External Access
```

### 5.3 Docker Compose Configuration

```yaml
# docker-compose.yaml
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: trading-assistant
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}
      - TELEGRAM_CHAT_ID=${TELEGRAM_CHAT_ID}
      - GROK_API_KEY=${GROK_API_KEY}
      - DATABASE_URL=postgresql://postgres:${POSTGRES_PASSWORD}@postgres:5432/trading
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - postgres
      - redis
    volumes:
      - ./config:/app/config:ro
      - ./data:/app/data
      - ./logs:/app/logs

  postgres:
    image: postgres:15-alpine
    container_name: trading-postgres
    restart: unless-stopped
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - POSTGRES_DB=trading
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: trading-redis
    restart: unless-stopped
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  postgres_data:
  redis_data:
```

### 5.4 Dockerfile

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

# Create non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Create directories
RUN mkdir -p data logs

# Environment
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run
CMD ["python", "-m", "src.main"]
```

---

## 6. Security Design

### 6.1 Secret Management

| Secret | Storage Method | Access |
|--------|---------------|--------|
| API Keys | Environment variables | Runtime only |
| Database credentials | Environment variables | Runtime only |
| Bot tokens | Environment variables | Runtime only |
| User data | Encrypted at rest (PostgreSQL) | Application only |

### 6.2 Authentication & Authorization

```
┌─────────────────────────────────────────────────┐
│              Authentication Flow                │
├─────────────────────────────────────────────────┤
│                                                 │
│  Telegram:                                      │
│    Request → Check chat_id → Allow/Deny        │
│                                                 │
│  Discord:                                       │
│    Request → Check user roles → Allow/Deny     │
│                                                 │
│  REST API:                                      │
│    Request → Validate API key → Allow/Deny     │
│                                                 │
└─────────────────────────────────────────────────┘
```

### 6.3 Input Validation

All user inputs are validated:
- Symbol format: regex pattern matching
- Numeric values: range and type checking
- Text inputs: length limits and sanitization
- Commands: whitelist-based routing

---

## 7. Error Handling Design

### 7.1 Error Categories

| Category | Handling Strategy | User Notification |
|----------|-------------------|-------------------|
| API Timeout | Retry with backoff | Notify after 3 failures |
| Invalid Input | Reject immediately | Show usage help |
| Data Not Found | Return gracefully | Inform user |
| System Error | Log and alert | Generic error message |
| Rate Limit | Queue and retry | Notify if persistent |

### 7.2 Retry Strategy

```python
RETRY_CONFIG = {
    "max_retries": 3,
    "initial_delay": 1.0,  # seconds
    "backoff_factor": 2.0,
    "max_delay": 30.0,
    "retryable_exceptions": [
        ConnectionError,
        TimeoutError,
        RateLimitError,
    ]
}
```

---

## 8. Monitoring & Observability

### 8.1 Logging Strategy

| Log Level | Usage |
|-----------|-------|
| DEBUG | Detailed diagnostic information |
| INFO | Normal operation events |
| WARNING | Unexpected but handled situations |
| ERROR | Error events, operation continues |
| CRITICAL | System failure, needs attention |

### 8.2 Metrics

| Metric | Type | Description |
|--------|------|-------------|
| api_requests_total | Counter | Total API requests |
| api_request_duration | Histogram | Request latency |
| scan_duration | Histogram | Scan completion time |
| positions_count | Gauge | Current open positions |
| portfolio_value | Gauge | Total portfolio value |
| alerts_sent | Counter | Alerts sent by type |

### 8.3 Health Check Endpoint

```json
GET /health

{
    "status": "healthy",
    "version": "1.0.0",
    "timestamp": "2025-11-28T12:00:00Z",
    "components": {
        "database": "healthy",
        "cache": "healthy",
        "ai_provider": "healthy",
        "telegram_bot": "healthy"
    },
    "uptime_seconds": 86400
}
```

---

## 8. VCP Alert System Design

### 8.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         VCPAlertSystem                              │
│  (Main orchestrator - coordinates detection, alerts, notifications) │
└─────────────────────────────────────────────────────────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          ▼                        ▼                        ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   VCPDetector    │    │   AlertManager   │    │ NotificationHub  │
│                  │    │                  │    │                  │
│ - analyze()      │    │ - check_alerts() │    │ - dispatch()     │
│ - get_pattern()  │───▶│ - persist()      │───▶│ - telegram()     │
│ - get_entries()  │    │ - deduplicate()  │    │ - discord()      │
└──────────────────┘    │ - track_state()  │    │ - webhook()      │
                        │ - get_history()  │    └──────────────────┘
                        └──────────────────┘
                                   │
                                   ▼
                        ┌──────────────────┐
                        │  AlertRepository │
                        │   (Database)     │
                        │                  │
                        │ - save()         │
                        │ - query()        │
                        │ - update()       │
                        └──────────────────┘
```

### 8.2 Component Design

#### 8.2.1 VCPDetector

```
┌──────────────────────────────────────┐
│            VCPDetector               │
├──────────────────────────────────────┤
│ - lookback_days: int = 120           │
│ - min_contractions: int = 2          │
│ - swing_lookback: int = 5            │
│ - max_contraction_gap: int = 30      │
│ - enable_staleness_check: bool = True│
│ - max_days_since_contraction: int=42 │
│ - max_pivot_violations: int = 2      │
│ - max_support_violations: int = 1    │
├──────────────────────────────────────┤
│ + analyze(df: DataFrame) -> Optional[VCPPattern]           │
│ + detect_contractions(df) -> List[Contraction]             │
│ + calculate_score(pattern) -> float                        │
│ + get_entry_signals(pattern, df) -> List[EntrySignal]      │
│ + _check_staleness(df, pattern) -> VCPPattern              │
│ + _count_pivot_violations(df, pivot, start) -> int         │
│ + _count_support_violations(df, support, start) -> int     │
└──────────────────────────────────────┘
```

**Staleness Detection:**
The detector checks for pattern validity after contraction detection:
- **Time Decay**: Patterns older than 42 trading days (6 weeks) are marked stale
- **Pivot Violations**: If price crossed above pivot then fell back (max 2 allowed)
- **Support Violations**: If price closed below support (invalidates pattern)

Staleness metrics are stored in VCPPattern:
- `days_since_last_contraction`, `pivot_violations`, `support_violations`
- `is_stale`, `freshness_score` (0-100), `staleness_reasons`

#### 8.2.2 AlertManager

```
┌──────────────────────────────────────┐
│           AlertManager               │
├──────────────────────────────────────┤
│ - repository: AlertRepository        │
│ - subscribers: List[Callable]        │
│ - config: AlertConfig                │
├──────────────────────────────────────┤
│ + check_and_emit(pattern) -> List[Alert]                   │
│ + check_contraction_alert(pattern) -> Optional[Alert]      │
│ + check_pre_alert(pattern, price) -> Optional[Alert]       │
│ + check_trade_alert(pattern, signal) -> Optional[Alert]    │
│ + mark_converted(alert_id, child_id) -> None               │
│ + expire_stale_alerts() -> int                             │
│ + get_alert_history(...) -> List[Alert]                    │
│ + get_active_alerts(symbol) -> List[Alert]                 │
│ + get_alert_chain(alert_id) -> AlertChain                  │
│ + subscribe(handler: Callable) -> None                     │
│ - _notify(alert: Alert) -> None                            │
│ - _deduplicate(alert: Alert) -> bool                       │
└──────────────────────────────────────┘
```

#### 8.2.3 AlertRepository

```
┌──────────────────────────────────────┐
│    <<interface>> AlertRepository     │
├──────────────────────────────────────┤
│ + save(alert: Alert) -> None                               │
│ + get_by_id(alert_id: str) -> Optional[Alert]              │
│ + update(alert: Alert) -> None                             │
│ + delete(alert_id: str) -> None                            │
│ + query(                                                   │
│     symbol: Optional[str],                                 │
│     alert_type: Optional[AlertType],                       │
│     state: Optional[AlertState],                           │
│     start_date: Optional[datetime],                        │
│     end_date: Optional[datetime],                          │
│     parent_alert_id: Optional[str]                         │
│   ) -> List[Alert]                                         │
│ + get_children(parent_id: str) -> List[Alert]              │
│ + get_expiring(before: datetime) -> List[Alert]            │
└──────────────────────────────────────┘
          △
          │
    ┌─────┴─────┐
    │           │
┌───┴───────┐ ┌─┴───────────┐
│ SQLite    │ │ PostgreSQL  │
│ Repository│ │ Repository  │
└───────────┘ └─────────────┘
```

#### 8.2.4 NotificationHub

```
┌──────────────────────────────────────┐
│          NotificationHub             │
├──────────────────────────────────────┤
│ - channels: List[NotificationChannel]│
├──────────────────────────────────────┤
│ + register_channel(channel) -> None  │
│ + dispatch(alert: Alert) -> None     │
│ + dispatch_async(alert: Alert) -> None │
└──────────────────────────────────────┘
                    │
                    │ uses
                    ▼
┌──────────────────────────────────────┐
│  <<interface>> NotificationChannel   │
├──────────────────────────────────────┤
│ + name: str                          │
│ + alert_types: List[AlertType]       │
├──────────────────────────────────────┤
│ + send(alert: Alert) -> bool         │
│ + format_message(alert) -> str       │
└──────────────────────────────────────┘
          △           △           △
          │           │           │
    ┌─────┴───┐ ┌─────┴───┐ ┌─────┴───┐
    │Telegram │ │Discord  │ │Webhook  │
    │Channel  │ │Channel  │ │Channel  │
    └─────────┘ └─────────┘ └─────────┘
```

### 8.3 Data Models

```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List
import uuid

class AlertType(Enum):
    CONTRACTION = "contraction"
    PRE_ALERT = "pre_alert"
    TRADE = "trade"

class AlertState(Enum):
    PENDING = "pending"
    NOTIFIED = "notified"
    CONVERTED = "converted"
    EXPIRED = "expired"
    COMPLETED = "completed"

@dataclass
class Alert:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    alert_type: AlertType
    state: AlertState = AlertState.PENDING

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    converted_at: Optional[datetime] = None
    expired_at: Optional[datetime] = None

    # Linkage
    parent_alert_id: Optional[str] = None

    # Pattern data at alert time
    trigger_price: float
    pivot_price: float
    distance_to_pivot_pct: float
    score: float
    num_contractions: int
    pattern_snapshot: dict = field(default_factory=dict)

@dataclass
class AlertChain:
    symbol: str
    contraction_alert: Optional[Alert] = None
    pre_alerts: List[Alert] = field(default_factory=list)
    trade_alert: Optional[Alert] = None

    @property
    def total_lead_time_days(self) -> Optional[int]:
        if self.contraction_alert and self.trade_alert:
            delta = self.trade_alert.created_at - self.contraction_alert.created_at
            return delta.days
        return None

@dataclass
class ConversionStats:
    period_start: datetime
    period_end: datetime
    total_contraction_alerts: int
    total_pre_alerts: int
    total_trade_alerts: int
    contraction_to_pre_alert_rate: float
    contraction_to_trade_rate: float
    pre_alert_to_trade_rate: float
    avg_days_contraction_to_trade: float
    avg_days_pre_alert_to_trade: float
```

### 8.4 Database Schema

```sql
-- VCP Alerts table (extends existing alerts table)
CREATE TABLE vcp_alerts (
    id VARCHAR(36) PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    alert_type VARCHAR(20) NOT NULL,  -- contraction, pre_alert, trade
    state VARCHAR(20) NOT NULL DEFAULT 'pending',

    -- Timestamps
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    converted_at TIMESTAMP,
    expired_at TIMESTAMP,

    -- Linkage
    parent_alert_id VARCHAR(36) REFERENCES vcp_alerts(id),

    -- Pattern data
    trigger_price DECIMAL(12,4) NOT NULL,
    pivot_price DECIMAL(12,4) NOT NULL,
    distance_to_pivot_pct DECIMAL(8,4) NOT NULL,
    score DECIMAL(5,2) NOT NULL,
    num_contractions INTEGER NOT NULL,
    pattern_snapshot JSON,

    -- Indexes
    CONSTRAINT valid_alert_type CHECK (alert_type IN ('contraction', 'pre_alert', 'trade')),
    CONSTRAINT valid_state CHECK (state IN ('pending', 'notified', 'converted', 'expired', 'completed'))
);

CREATE INDEX idx_vcp_alerts_symbol ON vcp_alerts(symbol);
CREATE INDEX idx_vcp_alerts_type_state ON vcp_alerts(alert_type, state);
CREATE INDEX idx_vcp_alerts_created ON vcp_alerts(created_at);
CREATE INDEX idx_vcp_alerts_parent ON vcp_alerts(parent_alert_id);
```

### 8.5 Alert State Machine

```
                    ┌─────────────┐
                    │   PENDING   │
                    └──────┬──────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
   │  NOTIFIED   │  │  CONVERTED  │  │   EXPIRED   │
   └──────┬──────┘  └─────────────┘  └─────────────┘
          │
          ▼
   ┌─────────────┐
   │  COMPLETED  │  (Trade alerts only, after trade closes)
   └─────────────┘
```

**State Transitions:**
- PENDING → NOTIFIED: Alert sent to user
- PENDING → CONVERTED: Next-stage alert created (contraction→pre-alert or pre-alert→trade)
- PENDING → EXPIRED: Alert TTL exceeded without conversion
- NOTIFIED → COMPLETED: Trade alert acknowledged and trade closed

### 8.6 Alert Flow Sequence

```
Day 1: Pattern detected with 2 contractions, score=75
       │
       ▼
   ┌───────────────────────────────────────┐
   │ AlertManager.check_contraction_alert()│
   │ → Creates CONTRACTION alert (PENDING) │
   │ → Notifies subscribers                │
   └───────────────────────────────────────┘

Day 5: Price moves within 3% of pivot
       │
       ▼
   ┌───────────────────────────────────────┐
   │ AlertManager.check_pre_alert()        │
   │ → Creates PRE_ALERT (PENDING)         │
   │ → Marks CONTRACTION as CONVERTED      │
   │ → Notifies subscribers                │
   └───────────────────────────────────────┘

Day 8: Entry signal triggers (pivot breakout)
       │
       ▼
   ┌───────────────────────────────────────┐
   │ AlertManager.check_trade_alert()      │
   │ → Creates TRADE alert (PENDING)       │
   │ → Marks PRE_ALERT as CONVERTED        │
   │ → Notifies subscribers                │
   └───────────────────────────────────────┘
```

### 8.7 File Structure

```
src/
└── vcp/
    ├── __init__.py
    ├── models.py          # Alert, AlertChain, AlertType, AlertState, etc.
    ├── detector.py        # VCPDetector class
    ├── alert_manager.py   # AlertManager class
    ├── repository.py      # AlertRepository protocol + SQLite implementation
    ├── notifications.py   # NotificationHub + channel implementations
    ├── alert_system.py    # VCPAlertSystem orchestrator
    ├── chart.py           # Matplotlib static chart generation
    └── chart_lightweight.py  # TradingView Lightweight Charts dashboard

tests/
└── vcp/
    ├── test_models.py
    ├── test_detector.py
    ├── test_alert_manager.py
    ├── test_repository.py
    ├── test_notifications.py
    └── test_alert_system.py
```

### 8.8 Chart Generation Components

#### 8.8.1 Static Chart Generator (Matplotlib)

```
┌──────────────────────────────────────┐
│          ChartGenerator              │
├──────────────────────────────────────┤
│ - output_dir: Path                   │
├──────────────────────────────────────┤
│ + generate_chart(symbol, df,         │
│     pattern, alerts) -> Path         │
│ + generate_batch(results,            │
│     max_per_category) -> Dict        │
└──────────────────────────────────────┘
```

**Features:**
- Candlestick charts with OHLCV data
- Volume bars with color coding
- Contraction zones highlighted (shaded regions)
- Pivot and support price lines
- Alert markers (triangles) with color by type
- Pattern score annotation
- Automatic file organization by alert type

**Output Structure:**
```
charts/
├── trade_alerts/
│   ├── AAPL_vcp.png
│   └── NVDA_vcp.png
├── pre_alerts/
│   └── MSFT_vcp.png
└── contraction_alerts/
    └── GOOGL_vcp.png
```

#### 8.8.2 Interactive Dashboard Generator (TradingView Lightweight Charts)

```
┌──────────────────────────────────────┐
│     LightweightChartGenerator        │
├──────────────────────────────────────┤
│ - output_dir: Path                   │
├──────────────────────────────────────┤
│ + generate_dashboard(scan_results,   │
│     filename) -> Path                │
│ - _prepare_chart_data(result) -> Dict│
│ - _render_template(data) -> str      │
└──────────────────────────────────────┘
```

**Architecture:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    HTML Dashboard File                           │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌───────────────────────────────────────┐ │
│  │   Sidebar       │  │          Chart Area                   │ │
│  │                 │  │  ┌─────────────────────────────────┐  │ │
│  │ [Filter Btns]   │  │  │   TradingView Candlestick       │  │ │
│  │ ○ All           │  │  │   Chart (Lightweight Charts)    │  │ │
│  │ ○ Trade         │  │  │                                 │  │ │
│  │ ○ Pre-Alert     │  │  │   - Series markers for alerts   │  │ │
│  │ ○ Contraction   │  │  │   - Price lines for pivot/stop  │  │ │
│  │                 │  │  │                                 │  │ │
│  │ [Search Box]    │  │  └─────────────────────────────────┘  │ │
│  │                 │  │  ┌─────────────────────────────────┐  │ │
│  │ [Stock List]    │  │  │   Volume Chart                  │  │ │
│  │ > AAPL ★        │  │  └─────────────────────────────────┘  │ │
│  │   NVDA ●        │  │                                       │ │
│  │   MSFT ◆        │  │  ┌─────────────────────────────────┐  │ │
│  │   ...           │  │  │   Pattern Info Panel            │  │ │
│  └─────────────────┘  │  │   Score: 85 | Contractions: 3   │  │ │
│                       │  └─────────────────────────────────┘  │ │
└─────────────────────────────────────────────────────────────────┘
```

**Technology Stack:**
- TradingView Lightweight Charts v4.1.0 (CDN)
- Vanilla JavaScript (no framework dependencies)
- Embedded JSON data (no server required)
- Single HTML file deployment

**Data Format (embedded JSON):**
```javascript
const VCP_DATA = {
    "AAPL": {
        "ohlcv": [
            {"time": "2025-01-02", "open": 150.0, "high": 152.0,
             "low": 149.0, "close": 151.5, "volume": 50000000},
            // ... more bars
        ],
        "pattern": {
            "contractions": [
                {"start_date": "2025-01-10", "end_date": "2025-01-20",
                 "high": 155.0, "low": 148.0, "depth_pct": 4.5},
            ],
            "pivot_price": 156.0,
            "support_price": 148.0,
            "score": 85.0
        },
        "alerts": [
            {"type": "trade", "date": "2025-01-25", "price": 156.5}
        ],
        "alert_type": "trade"  // highest priority alert
    },
    // ... more symbols
};
```

**Features:**
- Instant chart switching (no network requests)
- Filter buttons by alert type
- Real-time search filtering
- Keyboard navigation (↑↓ arrows)
- Responsive layout
- Alert type indicators (★ Trade, ● Pre-Alert, ◆ Contraction)
- Contraction zones visualization
- Pivot/support price lines

### 8.9 Implementation Status

**VCPAlertSystem - COMPLETE ✅**

| Component | Status | Tests |
|-----------|--------|-------|
| models.py | ✅ Complete | 32 tests |
| detector.py | ✅ Complete | 15 tests |
| alert_manager.py | ✅ Complete | 27 tests |
| repository.py | ✅ Complete | 20 tests |
| notifications.py | ✅ Complete | 26 tests |
| alert_system.py | ✅ Complete | 24 tests |
| chart.py | ✅ Complete | - |
| chart_lightweight.py | ✅ Complete | - |

**Total: 144 unit tests passing**

---

## 9. Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-11-28 | Claude | Initial draft |
| 1.1.0 | 2025-12-01 | Claude | VCPAlertSystem implementation complete, added charting components |

---

## 10. Appendices

### Appendix A: Technology Stack

| Layer | Technology | Version | Purpose |
|-------|------------|---------|---------|
| Language | Python | 3.11+ | Core application |
| Web Framework | FastAPI | 0.100+ | REST API |
| Bot Framework | python-telegram-bot | 20.x | Telegram interface |
| Task Scheduler | APScheduler | 3.10+ | Background jobs |
| Database | SQLite/PostgreSQL | 3.x/15+ | Persistence |
| Cache | Redis | 7.x | Caching, sessions |
| AI Client | OpenAI SDK | 1.x | LLM integration |
| Data Analysis | Pandas | 2.x | Data processing |
| HTTP Client | Requests/HTTPX | - | API calls |
| Containerization | Docker | 24+ | Deployment |
| Static Charts | Matplotlib | 3.x | PNG chart generation |
| Interactive Charts | TradingView Lightweight Charts | 4.1.0 | HTML dashboard |
| Market Data | yfinance | - | Yahoo Finance API |

### Appendix B: Project Structure

```
ai-trading-assistant/
├── src/
│   ├── __init__.py
│   ├── main.py                 # Application entry point
│   ├── ai/
│   │   ├── __init__.py
│   │   ├── engine.py           # AI Engine Manager
│   │   ├── providers/
│   │   │   ├── __init__.py
│   │   │   ├── base.py         # AIProvider interface
│   │   │   ├── grok.py
│   │   │   ├── openai.py
│   │   │   └── claude.py
│   │   └── memory/
│   │       ├── __init__.py
│   │       ├── base.py         # ConversationStore interface
│   │       ├── memory_store.py
│   │       └── redis_store.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── manager.py          # DataProviderManager
│   │   └── providers/
│   │       ├── __init__.py
│   │       ├── base.py         # DataProvider interface
│   │       ├── eodhd.py
│   │       ├── finnhub.py
│   │       └── yahoo.py
│   ├── scanner/
│   │   ├── __init__.py
│   │   ├── manager.py          # ScannerManager
│   │   ├── base.py             # ScannerPlugin interface
│   │   └── plugins/
│   │       ├── __init__.py
│   │       ├── vcp.py
│   │       ├── pivot.py
│   │       └── ai_scanner.py
│   ├── portfolio/
│   │   ├── __init__.py
│   │   ├── manager.py          # PortfolioManager
│   │   ├── models.py           # Position, Account, etc.
│   │   ├── fees.py             # FeeCalculator
│   │   └── currency.py         # CurrencyConverter
│   ├── bot/
│   │   ├── __init__.py
│   │   ├── telegram.py         # TelegramBot
│   │   ├── discord.py          # DiscordBot
│   │   ├── commands/
│   │   │   ├── __init__.py
│   │   │   ├── portfolio.py
│   │   │   ├── scan.py
│   │   │   └── ...
│   │   └── router.py           # CommandRouter
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py              # FastAPI app
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── portfolio.py
│   │   │   ├── scan.py
│   │   │   └── health.py
│   │   └── auth.py             # API authentication
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── base.py             # StorageInterface
│   │   ├── sqlite.py
│   │   └── postgresql.py
│   ├── cache/
│   │   ├── __init__.py
│   │   ├── base.py             # CacheInterface
│   │   ├── memory.py
│   │   └── redis.py
│   ├── scheduler/
│   │   ├── __init__.py
│   │   └── jobs.py             # Scheduled jobs
│   └── utils/
│       ├── __init__.py
│       ├── config.py           # Configuration loading
│       ├── logging.py          # Logging setup
│       └── helpers.py          # Utility functions
├── tests/
│   ├── __init__.py
│   ├── conftest.py             # pytest fixtures
│   ├── unit/
│   │   ├── test_ai_engine.py
│   │   ├── test_portfolio.py
│   │   └── ...
│   └── integration/
│       ├── test_telegram_bot.py
│       └── ...
├── config/
│   ├── config.yaml
│   ├── strategies/
│   └── ...
├── data/                       # SQLite and local files
├── logs/                       # Application logs
├── docs/                       # Documentation
├── scripts/                    # Utility scripts
├── docker-compose.yaml
├── Dockerfile
├── requirements.txt
├── pyproject.toml
└── README.md
```
