# Test Plan

## AI Trading Assistant

**Version:** 1.0.0
**Date:** 2025-11-28
**Status:** Draft

---

## 1. Introduction

### 1.1 Purpose

This document provides the detailed test plan for the AI Trading Assistant, specifying the test cases, test data, and execution procedures for each module.

### 1.2 Scope

This plan covers testing for:
- Portfolio Management Module
- Data Provider Module
- Scanner Module
- AI Engine Module
- Bot Interface Module
- Storage Module

### 1.3 References

- System Requirement Specification (02-system-requirements.md)
- System Design Specification (03-system-design.md)
- Test Strategy (06-test-strategy.md)

---

## 2. Test Schedule

### 2.1 Milestones

| Phase | Duration | Start | Activities |
|-------|----------|-------|------------|
| Unit Test Development | 2 weeks | Sprint 1 | Write unit tests alongside code |
| Integration Testing | 1 week | Sprint 2 | Module integration tests |
| System Testing | 1 week | Sprint 3 | End-to-end testing |
| Performance Testing | 3 days | Sprint 3 | Load and benchmark tests |
| UAT | 1 week | Sprint 4 | User acceptance testing |

### 2.2 Entry Criteria

- Code complete for module under test
- Unit tests written and passing
- Code review completed
- No P0/P1 bugs open

### 2.3 Exit Criteria

- All test cases executed
- ≥ 80% code coverage achieved
- No P0/P1 bugs remaining
- Performance targets met

---

## 3. Portfolio Module Test Plan

### 3.1 Unit Tests

#### 3.1.1 Position Management

| Test ID | Test Case | Input | Expected Output | Priority |
|---------|-----------|-------|-----------------|----------|
| PM-001 | Add position with valid data | symbol="AAPL", shares=100, entry=150.00, stop=145.00, target=165.00 | Position created, cash reduced | P0 |
| PM-002 | Add position with insufficient funds | cost > cash_balance | ValueError raised | P0 |
| PM-003 | Add duplicate position | existing symbol | ValueError raised | P1 |
| PM-004 | Close position with market price | symbol="AAPL", exit_price=None | Position removed, cash updated, P&L calculated | P0 |
| PM-005 | Close position with specified price | symbol="AAPL", exit_price=160.00 | Position removed at specified price | P0 |
| PM-006 | Close non-existent position | symbol="INVALID" | ValueError raised | P1 |
| PM-007 | Update stop loss | symbol="AAPL", new_stop=148.00 | Stop updated, persisted | P0 |
| PM-008 | Update target price | symbol="AAPL", new_target=170.00 | Target updated, persisted | P1 |

#### 3.1.2 Multi-Currency Support

| Test ID | Test Case | Input | Expected Output | Priority |
|---------|-----------|-------|-----------------|----------|
| MC-001 | Add HK position | symbol="0700.HK", price in HKD | Position created, USD conversion correct | P0 |
| MC-002 | Add JP position | symbol="7974.T", price in JPY | Position created, USD conversion correct | P1 |
| MC-003 | Currency detection - US | symbol="AAPL" | currency="USD" | P0 |
| MC-004 | Currency detection - HK | symbol="0005.HK" | currency="HKD" | P0 |
| MC-005 | Currency detection - JP | symbol="7974.T" | currency="JPY" | P1 |
| MC-006 | Convert to base currency | amount=780.00, currency="HKD" | ~100.00 USD | P0 |
| MC-007 | Convert from base currency | amount=100.00, to="JPY" | ~15000.00 JPY | P1 |

#### 3.1.3 Fee Calculation

| Test ID | Test Case | Input | Expected Output | Priority |
|---------|-----------|-------|-----------------|----------|
| FE-001 | US market fees | symbol="AAPL", shares=100, price=150.00 | Commission + SEC + TAF | P1 |
| FE-002 | HK market fees (buy) | symbol="0700.HK", shares=100, price=400.00 | Commission + stamp duty + levy | P1 |
| FE-003 | HK market fees (sell) | symbol="0700.HK", shares=100, price=400.00 | Commission + levy (no stamp) | P1 |
| FE-004 | Minimum commission applied | small order | min commission enforced | P2 |

#### 3.1.4 Portfolio Metrics

| Test ID | Test Case | Input | Expected Output | Priority |
|---------|-----------|-------|-----------------|----------|
| MT-001 | Calculate position P&L | entry=100, current=110 | pnl=10%, pnl_usd correct | P0 |
| MT-002 | Calculate position risk | entry=100, stop=95 | risk=5%, risk_usd correct | P0 |
| MT-003 | Portfolio summary - empty | no positions | zeros, cash balance only | P1 |
| MT-004 | Portfolio summary - with positions | multiple positions | totals calculated correctly | P0 |
| MT-005 | Portfolio heat calculation | total_risk / total_value | percentage calculated | P1 |

#### 3.1.5 Alert Generation

| Test ID | Test Case | Input | Expected Output | Priority |
|---------|-----------|-------|-----------------|----------|
| AL-001 | Stop loss hit | current_price <= stop_loss | stop_hit alert generated | P0 |
| AL-002 | Target hit | current_price >= target_price | target_hit alert generated | P0 |
| AL-003 | Trailing stop suggestion | pnl >= 8%, stop < entry | trailing_stop_suggestion alert | P1 |
| AL-004 | No alert when prices normal | stop < current < target | empty alerts list | P1 |

#### 3.1.6 Cash Management

| Test ID | Test Case | Input | Expected Output | Priority |
|---------|-----------|-------|-----------------|----------|
| CA-001 | Deposit cash | amount=5000.00 | cash += 5000, capital += 5000 | P0 |
| CA-002 | Withdraw cash - valid | amount=1000.00, cash=5000.00 | cash -= 1000, capital -= 1000 | P0 |
| CA-003 | Withdraw cash - insufficient | amount=10000.00, cash=5000.00 | ValueError raised | P0 |
| CA-004 | Deposit negative amount | amount=-100.00 | ValueError raised | P1 |

### 3.2 Integration Tests

| Test ID | Test Case | Description | Priority |
|---------|-----------|-------------|----------|
| PI-001 | Full trade workflow | Add → Monitor → Close position | P0 |
| PI-002 | Multi-currency trade | Add HK position, close with P&L | P1 |
| PI-003 | Portfolio persistence | Add position, restart, verify data | P0 |
| PI-004 | Alert generation with real prices | Monitor positions with mocked prices | P1 |

---

## 4. Data Provider Module Test Plan

### 4.1 Unit Tests

#### 4.1.1 Price Retrieval

| Test ID | Test Case | Input | Expected Output | Priority |
|---------|-----------|-------|-----------------|----------|
| DP-001 | Get US stock price | symbol="AAPL" | price > 0, currency="USD" | P0 |
| DP-002 | Get HK stock price | symbol="0700.HK" | price > 0, currency="HKD" | P0 |
| DP-003 | Get JP stock price | symbol="7974.T" | price > 0, currency="JPY" | P1 |
| DP-004 | Invalid symbol | symbol="INVALID123" | None or error | P1 |
| DP-005 | Price caching | same symbol, < 60s | cached value returned | P1 |
| DP-006 | Cache expiry | same symbol, > 60s | fresh value fetched | P2 |

#### 4.1.2 Provider Fallback

| Test ID | Test Case | Input | Expected Output | Priority |
|---------|-----------|-------|-----------------|----------|
| FB-001 | Primary provider success | EODHD available | EODHD used | P0 |
| FB-002 | Primary fails, fallback succeeds | EODHD fails, Finnhub available | Finnhub used | P0 |
| FB-003 | All providers fail | all APIs fail | None returned, logged | P1 |

#### 4.1.3 Historical Data

| Test ID | Test Case | Input | Expected Output | Priority |
|---------|-----------|-------|-----------------|----------|
| HD-001 | Get 6 months history | symbol="AAPL", period="6mo" | DataFrame with OHLCV | P0 |
| HD-002 | Get date range history | symbol, start_date, end_date | DataFrame for date range | P0 |
| HD-003 | Invalid period | period="invalid" | None or error | P2 |

#### 4.1.4 Quote Data

| Test ID | Test Case | Input | Expected Output | Priority |
|---------|-----------|-------|-----------------|----------|
| QT-001 | Full quote | symbol="AAPL" | Dict with price, ohlc, volume, change | P0 |
| QT-002 | Quote includes source | any symbol | source field populated | P1 |

### 4.2 Integration Tests

| Test ID | Test Case | Description | Priority |
|---------|-----------|-------------|----------|
| DI-001 | Multi-provider integration | Test fallback chain with real APIs | P1 |
| DI-002 | Rate limiting handling | Rapid requests handled gracefully | P2 |

---

## 5. Scanner Module Test Plan

### 5.1 Unit Tests

#### 5.1.1 VCP Scanner

| Test ID | Test Case | Input | Expected Output | Priority |
|---------|-----------|-------|-----------------|----------|
| VC-001 | Detect valid VCP | stock with VCP pattern | Setup returned with details | P0 |
| VC-002 | Reject non-VCP | stock without pattern | No setup returned | P0 |
| VC-003 | RS rating >= 90 check | rs=85 | Filtered out | P0 |
| VC-004 | RS rating >= 90 pass | rs=92 | Included | P0 |
| VC-005 | Trend template validation | below 200MA | Filtered out | P1 |
| VC-006 | Volume dry-up detection | volume contracting | VCP score increased | P1 |
| VC-007 | Breakout detection | price > pivot on volume | Breakout alert | P0 |

#### 5.1.2 RS Calculator

| Test ID | Test Case | Input | Expected Output | Priority |
|---------|-----------|-------|-----------------|----------|
| RS-001 | Calculate RS rating | stock outperforming SPY | RS > 50 | P0 |
| RS-002 | Calculate RS rating | stock underperforming SPY | RS < 50 | P0 |
| RS-003 | RS = 90 interpretation | rs=90 | "Top 10%" | P1 |
| RS-004 | Historical RS | date in past | RS at that date | P2 |

#### 5.1.3 Plugin System

| Test ID | Test Case | Input | Expected Output | Priority |
|---------|-----------|-------|-----------------|----------|
| PL-001 | Register plugin | valid plugin | Plugin added to registry | P0 |
| PL-002 | Enable/disable plugin | plugin name | Plugin state changed | P1 |
| PL-003 | Scan with multiple plugins | watchlist | Results from all enabled | P1 |

### 5.2 Integration Tests

| Test ID | Test Case | Description | Priority |
|---------|-----------|-------------|----------|
| SI-001 | Full watchlist scan | Scan 10 symbols with VCP scanner | P0 |
| SI-002 | Scan with real data | Use mocked historical data | P1 |
| SI-003 | AI scanner integration | AI analyzes scanner results | P1 |

---

## 6. AI Engine Module Test Plan

### 6.1 Unit Tests

#### 6.1.1 Provider Management

| Test ID | Test Case | Input | Expected Output | Priority |
|---------|-----------|-------|-----------------|----------|
| AI-001 | Register provider | valid provider | Provider in registry | P0 |
| AI-002 | Set active provider | provider name | Active provider changed | P0 |
| AI-003 | Provider fallback | primary fails | Fallback used | P1 |
| AI-004 | Get available providers | - | List of registered providers | P1 |

#### 6.1.2 Chat Functionality

| Test ID | Test Case | Input | Expected Output | Priority |
|---------|-----------|-------|-----------------|----------|
| CH-001 | Basic chat | "Hello" | Non-empty response | P0 |
| CH-002 | Chat with context | message + context dict | Context-aware response | P0 |
| CH-003 | Conversation history | multiple messages | History maintained | P0 |
| CH-004 | Clear history | - | History cleared | P1 |
| CH-005 | Chat timeout | slow provider | Timeout handled | P1 |

#### 6.1.3 Function Calling

| Test ID | Test Case | Input | Expected Output | Priority |
|---------|-----------|-------|-----------------|----------|
| FC-001 | Tool execution - get_quote | "What's AAPL price?" | Tool called, result in response | P0 |
| FC-002 | Tool execution - get_rs | "RS rating for NVDA" | RS tool called | P0 |
| FC-003 | Multiple tools | complex query | Multiple tools called | P1 |
| FC-004 | Tool error handling | tool fails | Graceful error message | P1 |

#### 6.1.4 Memory System

| Test ID | Test Case | Input | Expected Output | Priority |
|---------|-----------|-------|-----------------|----------|
| MM-001 | Store insight | insight data | Insight persisted | P1 |
| MM-002 | Recall insights | insight type | List of insights | P1 |
| MM-003 | Search memory | query string | Relevant results | P1 |

### 6.2 Integration Tests

| Test ID | Test Case | Description | Priority |
|---------|-----------|-------------|----------|
| AII-001 | Full conversation flow | Multi-turn conversation with tools | P0 |
| AII-002 | Provider switching | Switch provider mid-conversation | P2 |

---

## 7. Bot Interface Module Test Plan

### 7.1 Unit Tests

#### 7.1.1 Command Parsing

| Test ID | Test Case | Input | Expected Output | Priority |
|---------|-----------|-------|-----------------|----------|
| CP-001 | Parse /portfolio | "/portfolio" | portfolio command routed | P0 |
| CP-002 | Parse /add with args | "/add AAPL 100 150 145 165" | Arguments extracted correctly | P0 |
| CP-003 | Parse invalid command | "/invalid" | Error message | P1 |
| CP-004 | Parse malformed args | "/add AAPL abc" | Validation error | P1 |

#### 7.1.2 Command Handlers

| Test ID | Test Case | Input | Expected Output | Priority |
|---------|-----------|-------|-----------------|----------|
| HA-001 | /portfolio handler | - | Portfolio summary message | P0 |
| HA-002 | /add handler | valid args | Position added, confirmation | P0 |
| HA-003 | /close handler | symbol | Position closed, P&L shown | P0 |
| HA-004 | /scan handler | - | Scan triggered, results sent | P0 |
| HA-005 | /watchlist handler | - | Watchlist displayed | P1 |
| HA-006 | /help handler | - | Help message | P1 |

#### 7.1.3 Authentication

| Test ID | Test Case | Input | Expected Output | Priority |
|---------|-----------|-------|-----------------|----------|
| AU-001 | Valid chat ID | authorized chat_id | Command processed | P0 |
| AU-002 | Invalid chat ID | unauthorized chat_id | Command ignored | P0 |

### 7.2 Integration Tests

| Test ID | Test Case | Description | Priority |
|---------|-----------|-------------|----------|
| BI-001 | Full command workflow | Send command → Process → Response | P0 |
| BI-002 | Natural language processing | Message → AI → Response | P0 |
| BI-003 | Alert delivery | Alert generated → Message sent | P1 |

---

## 8. Storage Module Test Plan

### 8.1 Unit Tests

#### 8.1.1 SQLite Storage

| Test ID | Test Case | Input | Expected Output | Priority |
|---------|-----------|-------|-----------------|----------|
| SQ-001 | Save position | position dict | Position in database | P0 |
| SQ-002 | Get positions | - | List of positions | P0 |
| SQ-003 | Update position | symbol, updates | Position updated | P0 |
| SQ-004 | Delete position | symbol | Position removed | P0 |
| SQ-005 | Save account | account dict | Account saved | P0 |
| SQ-006 | Get account | - | Account data | P0 |

#### 8.1.2 Data Integrity

| Test ID | Test Case | Input | Expected Output | Priority |
|---------|-----------|-------|-----------------|----------|
| DI-001 | Concurrent writes | parallel saves | No data corruption | P1 |
| DI-002 | Transaction rollback | failed operation | Previous state preserved | P1 |
| DI-003 | Schema migration | version upgrade | Data preserved, schema updated | P2 |

### 8.2 Integration Tests

| Test ID | Test Case | Description | Priority |
|---------|-----------|-------------|----------|
| STI-001 | Full CRUD cycle | Create → Read → Update → Delete | P0 |
| STI-002 | Persistence across restarts | Save → Restart → Verify | P0 |

---

## 9. REST API Test Plan

### 9.1 Unit Tests

#### 9.1.1 Endpoint Tests

| Test ID | Test Case | Endpoint | Expected Status | Priority |
|---------|-----------|----------|-----------------|----------|
| AP-001 | Health check | GET /health | 200 OK | P0 |
| AP-002 | Get portfolio | GET /api/v1/portfolio | 200 OK | P0 |
| AP-003 | Get positions | GET /api/v1/positions | 200 OK | P0 |
| AP-004 | Trigger scan | POST /api/v1/scan | 202 Accepted | P1 |
| AP-005 | Unauthorized access | GET /api/v1/portfolio (no key) | 401 Unauthorized | P0 |

### 9.2 Integration Tests

| Test ID | Test Case | Description | Priority |
|---------|-----------|-------------|----------|
| API-001 | Full API workflow | Auth → Request → Response | P0 |
| API-002 | Error handling | Invalid requests handled | P1 |

---

## 10. VCP Alert System Test Plan

### 10.1 Unit Tests

#### 10.1.1 Alert Models

| Test ID | Test Case | Input | Expected Output | Priority |
|---------|-----------|-------|-----------------|----------|
| AM-001 | Create Alert with defaults | symbol, alert_type, trigger_price | Alert with UUID, PENDING state | P0 |
| AM-002 | Alert state transition | PENDING → NOTIFIED | state updated, updated_at changed | P0 |
| AM-003 | Invalid state transition | EXPIRED → PENDING | Raises ValueError | P1 |
| AM-004 | AlertChain lead time calculation | chain with contraction + trade | Correct days calculated | P1 |
| AM-005 | Alert serialization | Alert object | JSON-serializable dict | P1 |

#### 10.1.2 VCPDetector

| Test ID | Test Case | Input | Expected Output | Priority |
|---------|-----------|-------|-----------------|----------|
| VD-001 | Detect valid VCP - NVDA | NVDA data 2024-01-08 | VCPPattern with 2+ contractions | P0 |
| VD-002 | Detect valid VCP - TSLA | TSLA data with 4 contractions | VCPPattern with score > 80 | P0 |
| VD-003 | Reject staircase - AAPL | AAPL staircase data | None or is_valid=False | P0 |
| VD-004 | Reject loosening - GOOGL | GOOGL loosening data | None or is_valid=False | P0 |
| VD-005 | Calculate pattern score | Valid pattern | Score 0-100 | P1 |
| VD-006 | Detect entry signals | Pattern near pivot | List of EntrySignal | P1 |
| VD-007 | Handle insufficient data | < 30 days data | None returned | P1 |

#### 10.1.3 AlertManager

| Test ID | Test Case | Input | Expected Output | Priority |
|---------|-----------|-------|-----------------|----------|
| MG-001 | Create contraction alert | Pattern with 2+ contractions, score >= 60 | Alert created and persisted | P0 |
| MG-002 | Create pre-alert | Pattern with price within 3% of pivot | PRE_ALERT created | P0 |
| MG-003 | Create trade alert | Pattern with entry signal | TRADE alert created | P0 |
| MG-004 | Deduplicate alert | Same symbol/type/date | Existing alert returned | P0 |
| MG-005 | Link alert chain | Pre-alert from contraction | parent_alert_id set correctly | P0 |
| MG-006 | Expire stale contraction | Alert > 20 days old | State = EXPIRED | P1 |
| MG-007 | Expire stale pre-alert | Alert > 5 days old | State = EXPIRED | P1 |
| MG-008 | Notify subscribers | New alert created | All handlers called | P0 |
| MG-009 | Get active alerts | symbol="NVDA" | List of PENDING/NOTIFIED alerts | P1 |
| MG-010 | Get alert chain | trade_alert_id | Full AlertChain object | P1 |

#### 10.1.4 AlertRepository

| Test ID | Test Case | Input | Expected Output | Priority |
|---------|-----------|-------|-----------------|----------|
| RP-001 | Save alert | Valid Alert object | Alert persisted, retrievable | P0 |
| RP-002 | Get by ID | Existing alert_id | Alert returned | P0 |
| RP-003 | Get by ID - not found | Non-existent ID | None returned | P1 |
| RP-004 | Update alert | Alert with changed state | Update persisted | P0 |
| RP-005 | Query by symbol | symbol="NVDA" | All NVDA alerts | P0 |
| RP-006 | Query by type | alert_type=CONTRACTION | All contraction alerts | P1 |
| RP-007 | Query by state | state=PENDING | All pending alerts | P1 |
| RP-008 | Query by date range | start, end dates | Alerts in range | P1 |
| RP-009 | Get children | parent_alert_id | Child alerts | P1 |
| RP-010 | Get expiring | before datetime | Alerts past TTL | P1 |

#### 10.1.5 NotificationHub

| Test ID | Test Case | Input | Expected Output | Priority |
|---------|-----------|-------|-----------------|----------|
| NH-001 | Register channel | TelegramChannel | Channel in registry | P0 |
| NH-002 | Dispatch to all channels | Alert | All channels receive alert | P0 |
| NH-003 | Channel failure isolation | One channel throws | Others still receive | P0 |
| NH-004 | Filter by alert type | Channel with type filter | Only matching alerts sent | P1 |
| NH-005 | Format message | Alert | Channel-specific message | P1 |

### 10.2 Integration Tests

| Test ID | Test Case | Description | Priority |
|---------|-----------|-------------|----------|
| VI-001 | Full alert chain flow | Detect → Contraction → Pre-Alert → Trade | P0 |
| VI-002 | Alert persistence recovery | Create alerts, restart, verify | P0 |
| VI-003 | Multi-symbol scan | Scan 10 symbols, verify alerts | P1 |
| VI-004 | Expiration job | Create old alerts, run expiration | P1 |
| VI-005 | Notification delivery | Create alert, verify handlers called | P0 |
| VI-006 | Alert history query | Create many alerts, test filters | P1 |
| VI-007 | Concurrent alert creation | Parallel alert creation | P2 |

### 10.3 Test Data

```python
# VCP Pattern Test Fixtures
VCP_TEST_PATTERNS = {
    "NVDA_2024": {
        "symbol": "NVDA",
        "date": "2024-01-08",
        "expected_contractions": 2,
        "expected_score_min": 90,
        "expected_valid": True
    },
    "AAPL_STAIRCASE": {
        "symbol": "AAPL",
        "date": "2023-05-01",
        "expected_valid": False,
        "rejection_reason": "staircase"
    }
}

# Alert State Transition Matrix
STATE_TRANSITIONS = [
    ("PENDING", "notify", "NOTIFIED", True),
    ("PENDING", "convert", "CONVERTED", True),
    ("PENDING", "expire", "EXPIRED", True),
    ("NOTIFIED", "complete", "COMPLETED", True),
    ("EXPIRED", "notify", None, False),  # Invalid
    ("CONVERTED", "expire", None, False),  # Invalid
]
```

---

## 11. Performance Test Plan

### 11.1 Benchmark Tests

| Test ID | Test Case | Target | Priority |
|---------|-----------|--------|----------|
| PF-001 | API response time | < 500ms | P1 |
| PF-002 | Scan 50 symbols | < 2 minutes | P1 |
| PF-003 | Portfolio calculation | < 100ms | P2 |
| PF-004 | Memory usage (idle) | < 256MB | P2 |

### 11.2 Load Tests

| Test ID | Test Case | Load | Target | Priority |
|---------|-----------|------|--------|----------|
| LD-001 | Concurrent API requests | 10 req/s | No failures | P2 |
| LD-002 | Sustained load | 1 hour | No degradation | P2 |

---

## 12. Test Data

### 12.1 Test Portfolios

```python
# Empty portfolio
EMPTY_PORTFOLIO = {
    "account": {"total_capital": 50000, "cash_balance": 50000, "invested": 0},
    "positions": []
}

# Single position portfolio
SINGLE_POSITION = {
    "account": {"total_capital": 50000, "cash_balance": 35000, "invested": 15000},
    "positions": [{
        "symbol": "AAPL",
        "shares": 100,
        "entry_price": 150.00,
        "stop_loss": 145.00,
        "target_price": 165.00,
        "currency": "USD"
    }]
}

# Multi-currency portfolio
MULTI_CURRENCY = {
    "account": {"total_capital": 50000, "cash_balance": 20000, "invested": 30000},
    "positions": [
        {"symbol": "AAPL", "shares": 50, "entry_price": 150.00, "currency": "USD"},
        {"symbol": "0700.HK", "shares": 100, "entry_price": 400.00, "currency": "HKD"},
        {"symbol": "7974.T", "shares": 10, "entry_price": 8000.00, "currency": "JPY"}
    ]
}
```

### 12.2 Mock API Responses

```python
# Mock price response
MOCK_AAPL_QUOTE = {
    "symbol": "AAPL",
    "current": 155.00,
    "open": 154.00,
    "high": 156.50,
    "low": 153.50,
    "volume": 50000000,
    "change": 2.00,
    "change_pct": 1.31,
    "currency": "USD",
    "source": "mock"
}

# Mock historical data
MOCK_HISTORICAL = pd.DataFrame({
    "Open": [150, 151, 152],
    "High": [152, 153, 154],
    "Low": [149, 150, 151],
    "Close": [151, 152, 153],
    "Volume": [1000000, 1100000, 1200000]
}, index=pd.date_range("2025-01-01", periods=3))
```

---

## 13. Test Execution

### 13.1 Run All Tests

```bash
# Run full test suite
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific module
pytest tests/unit/test_portfolio/

# Run single test
pytest tests/unit/test_portfolio/test_manager.py::test_add_position_success
```

### 13.2 Test Tags

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run only P0 priority tests
pytest -m "p0"

# Skip slow tests
pytest -m "not slow"
```

### 13.3 CI Execution

Tests are automatically run on every push via GitHub Actions (see Test Strategy).

---

## 14. Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-11-28 | Claude | Initial draft |
| 1.1.0 | 2025-11-30 | Claude | Added VCP Alert System test plan (Section 10) |
