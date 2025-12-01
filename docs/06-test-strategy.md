# Test Strategy

## AI Trading Assistant

**Version:** 1.0.0
**Date:** 2025-11-28
**Status:** Draft

---

## 1. Introduction

### 1.1 Purpose

This document defines the overall testing strategy for the AI Trading Assistant project. It establishes the approach, methodologies, and frameworks to be used to ensure software quality throughout the development lifecycle.

### 1.2 Scope

This strategy covers:
- All application modules (AI, Data, Scanner, Portfolio, Bot, Storage)
- Unit, integration, and end-to-end testing
- Performance and security testing
- Test automation and CI/CD integration

### 1.3 Quality Objectives

| Objective | Target |
|-----------|--------|
| Code Coverage | ≥ 80% overall, ≥ 90% for critical modules |
| Bug Escape Rate | < 5% of bugs found in production |
| Test Execution Time | < 5 minutes for unit tests, < 15 minutes for full suite |
| Zero Critical Bugs | No P0/P1 bugs in production releases |

---

## 2. Test Levels

### 2.1 Unit Testing

**Purpose:** Verify individual components work correctly in isolation.

**Scope:**
- Individual functions and methods
- Class behaviors
- Edge cases and error handling

**Approach:**
- Test-Driven Development (TDD) where practical
- Mock external dependencies
- Focus on business logic

**Tools:**
- pytest
- pytest-asyncio (for async code)
- pytest-mock
- pytest-cov (coverage)

**Coverage Targets:**

| Module | Target Coverage |
|--------|-----------------|
| portfolio/ | 95% |
| scanner/ | 90% |
| **vcp/** | 90% |
| ai/ | 85% |
| data/ | 85% |
| storage/ | 90% |
| bot/ | 80% |
| utils/ | 80% |

### 2.2 Integration Testing

**Purpose:** Verify components work correctly together.

**Scope:**
- Module interactions
- Database operations
- External API integrations (mocked)
- End-to-end workflows

**Approach:**
- Use test database (SQLite in-memory)
- Mock external APIs
- Test realistic scenarios

**Tools:**
- pytest
- pytest-asyncio
- httpx (for API testing)
- responses / httpretty (for HTTP mocking)

### 2.3 End-to-End Testing

**Purpose:** Verify the complete system works as expected from user perspective.

**Scope:**
- Telegram bot commands
- REST API endpoints
- Complete user workflows

**Approach:**
- Test against staging environment
- Use test Telegram bot
- Automated smoke tests

**Tools:**
- pytest
- Docker for test environment
- Custom bot testing framework

### 2.4 Performance Testing

**Purpose:** Ensure system meets performance requirements.

**Scope:**
- API response times
- Scan execution time
- Database query performance
- Memory usage

**Approach:**
- Benchmark critical operations
- Load testing for API
- Memory profiling

**Tools:**
- pytest-benchmark
- locust (load testing)
- memory_profiler

### 2.5 Security Testing

**Purpose:** Identify security vulnerabilities.

**Scope:**
- Input validation
- Authentication/Authorization
- Secrets management
- Dependency vulnerabilities

**Approach:**
- Static analysis
- Dependency scanning
- Input fuzzing

**Tools:**
- bandit (static analysis)
- safety (dependency check)
- pip-audit

---

## 3. Test Types

### 3.1 Functional Tests

| Test Type | Description | Priority |
|-----------|-------------|----------|
| Positive Tests | Verify expected behavior | High |
| Negative Tests | Verify error handling | High |
| Boundary Tests | Test edge cases | Medium |
| Equivalence Partitioning | Test representative values | Medium |

### 3.2 Non-Functional Tests

| Test Type | Description | Priority |
|-----------|-------------|----------|
| Performance | Response times, throughput | High |
| Reliability | Error recovery, uptime | High |
| Usability | Command clarity, error messages | Medium |
| Compatibility | Python versions, OS | Low |

---

## 4. Test Environment

### 4.1 Environment Matrix

| Environment | Purpose | Database | External APIs |
|-------------|---------|----------|---------------|
| Local | Development | SQLite in-memory | Mocked |
| CI | Automated tests | SQLite in-memory | Mocked |
| Staging | Integration testing | PostgreSQL | Real (test accounts) |
| Production | Live system | PostgreSQL | Real |

### 4.2 Test Data Strategy

**Fixtures:**
- Pre-defined test portfolios
- Sample watchlists
- Mock market data responses

**Generation:**
- Factory patterns for test objects
- Faker for realistic test data

**Cleanup:**
- Automatic cleanup after each test
- Transaction rollback where applicable

---

## 5. Test Automation

### 5.1 Automation Scope

| Category | Automation Level |
|----------|------------------|
| Unit Tests | 100% automated |
| Integration Tests | 100% automated |
| API Tests | 100% automated |
| Bot Command Tests | 90% automated |
| Performance Tests | 80% automated |
| Security Scans | 100% automated |

### 5.2 CI/CD Integration

**Pipeline Stages:**

```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│  Lint   │───▶│  Unit   │───▶│  Integ  │───▶│ Security│
│         │    │  Tests  │    │  Tests  │    │  Scan   │
└─────────┘    └─────────┘    └─────────┘    └─────────┘
                                                  │
                                                  ▼
                              ┌─────────┐    ┌─────────┐
                              │  Build  │◀───│Coverage │
                              │  Image  │    │ Report  │
                              └─────────┘    └─────────┘
```

**GitHub Actions Workflow:**

```yaml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install ruff black isort
      - name: Run linters
        run: |
          ruff check src/
          black --check src/
          isort --check-only src/

  test:
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt -r requirements-dev.txt
      - name: Run tests
        run: pytest --cov=src --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  security:
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - uses: actions/checkout@v4
      - name: Run security scan
        run: |
          pip install bandit safety
          bandit -r src/
          safety check
```

### 5.3 Test Execution Schedule

| Test Type | Trigger | Frequency |
|-----------|---------|-----------|
| Unit Tests | Every commit | Continuous |
| Integration Tests | Every commit | Continuous |
| Security Scan | Every commit | Continuous |
| Performance Tests | Nightly | Daily |
| Full Regression | Before release | Per release |

---

## 6. Test Organization

### 6.1 Directory Structure

```
tests/
├── conftest.py              # Shared fixtures
├── unit/
│   ├── __init__.py
│   ├── test_portfolio/
│   │   ├── test_manager.py
│   │   ├── test_fees.py
│   │   └── test_currency.py
│   ├── test_scanner/
│   │   ├── test_vcp.py
│   │   └── test_pivot.py
│   ├── test_vcp/                    # VCP Alert System tests
│   │   ├── test_models.py           # Alert, AlertChain dataclasses
│   │   ├── test_detector.py         # VCPDetector pattern detection
│   │   ├── test_alert_manager.py    # AlertManager business logic
│   │   ├── test_repository.py       # AlertRepository persistence
│   │   ├── test_notifications.py    # NotificationHub channels
│   │   └── test_alert_system.py     # VCPAlertSystem orchestration
│   ├── test_ai/
│   │   ├── test_engine.py
│   │   └── test_providers.py
│   ├── test_data/
│   │   └── test_providers.py
│   └── test_storage/
│       └── test_sqlite.py
├── integration/
│   ├── __init__.py
│   ├── test_bot_commands.py
│   ├── test_api_endpoints.py
│   ├── test_scan_workflow.py
│   ├── test_trade_workflow.py
│   └── test_vcp_alert_workflow.py   # Full alert chain integration
├── e2e/
│   ├── __init__.py
│   └── test_full_scenarios.py
├── performance/
│   ├── __init__.py
│   ├── test_scan_performance.py
│   └── test_api_performance.py
└── fixtures/
    ├── portfolio_data.py
    ├── market_data.py
    ├── vcp_pattern_data.py          # VCP pattern test fixtures
    └── mock_responses.py
```

### 6.2 Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Test file | `test_<module>.py` | `test_portfolio.py` |
| Test class | `Test<ClassName>` | `TestPortfolioManager` |
| Test method | `test_<behavior>` | `test_add_position_success` |
| Fixture | `<object>_fixture` | `portfolio_fixture` |

### 6.3 Test Documentation

Each test should include:
```python
def test_add_position_with_insufficient_funds():
    """
    Test that adding a position fails when cash is insufficient.

    Given: Portfolio with $1,000 cash balance
    When: Adding position costing $5,000
    Then: ValueError is raised with appropriate message
          AND cash balance remains unchanged
    """
```

---

## 7. Defect Management

### 7.1 Defect Classification

| Severity | Description | Response Time |
|----------|-------------|---------------|
| P0 - Critical | System unusable, data loss risk | Immediate |
| P1 - High | Major feature broken, workaround exists | 24 hours |
| P2 - Medium | Feature impaired, minor impact | 1 week |
| P3 - Low | Cosmetic, minor annoyance | Next release |

### 7.2 Defect Workflow

```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│   New   │───▶│  Triage │───▶│ In Work │───▶│  Fixed  │
└─────────┘    └─────────┘    └─────────┘    └─────────┘
                    │                             │
                    │                             ▼
                    │                       ┌─────────┐
                    └───────────────────────│ Verified│
                       (Won't Fix)          └─────────┘
```

### 7.3 Bug Report Template

```markdown
## Bug Report

**Summary:** [Brief description]

**Severity:** P0 / P1 / P2 / P3

**Environment:**
- Version: [app version]
- Python: [version]
- OS: [operating system]

**Steps to Reproduce:**
1. [Step 1]
2. [Step 2]
3. [Step 3]

**Expected Behavior:**
[What should happen]

**Actual Behavior:**
[What actually happened]

**Logs/Screenshots:**
[Attach relevant logs or screenshots]
```

---

## 8. Risk-Based Testing

### 8.1 Risk Assessment Matrix

| Component | Impact | Likelihood | Risk Score | Testing Priority |
|-----------|--------|------------|------------|------------------|
| Portfolio Manager | High | Medium | High | P1 |
| Trade Execution | High | Low | Medium | P1 |
| **VCP Alert System** | High | Medium | High | P1 |
| AI Analysis | Medium | Medium | Medium | P2 |
| Data Providers | Medium | High | High | P1 |
| Telegram Bot | Low | Low | Low | P3 |
| REST API | Medium | Low | Low | P3 |

### 8.2 Critical Path Testing

**Critical Paths (Must Test Extensively):**
1. Add Position → Update Position → Close Position
2. Scan Watchlist → Detect Pattern → Send Alert
3. Receive Command → Process → Respond
4. **VCP Alert Flow:** Detect Pattern → Contraction Alert → Pre-Alert → Trade Alert → Notify

### 8.3 Regression Test Selection

For each release, prioritize:
1. All P0/P1 bug fixes
2. New feature tests
3. Critical path tests
4. Integration tests
5. Random sampling of unit tests

---

## 9. Test Metrics

### 9.1 Key Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| Test Coverage | Lines covered / Total lines | ≥ 80% |
| Test Pass Rate | Passed / Total tests | ≥ 98% |
| Defect Density | Defects / KLOC | < 5 |
| Test Execution Time | Total time for test suite | < 15 min |
| Escaped Defects | Prod bugs / Total bugs | < 5% |

### 9.2 Reporting

**Daily:**
- CI pipeline status
- Failed tests count

**Weekly:**
- Coverage trend
- New bugs found
- Bug fix velocity

**Per Release:**
- Full test report
- Coverage report
- Known issues list

---

## 10. Roles and Responsibilities

### 10.1 Developer Responsibilities

- Write unit tests for new code
- Maintain ≥ 80% coverage for owned modules
- Fix failing tests before merging
- Review test code in PRs

### 10.2 Test Review Checklist

Before merging:
- [ ] All tests pass
- [ ] Coverage meets target
- [ ] No flaky tests introduced
- [ ] Test names are descriptive
- [ ] Edge cases covered
- [ ] Negative tests included

---

## 11. Tools and Frameworks

### 11.1 Primary Tools

| Tool | Purpose | Version |
|------|---------|---------|
| pytest | Test framework | 7.x |
| pytest-asyncio | Async testing | 0.21+ |
| pytest-cov | Coverage | 4.x |
| pytest-mock | Mocking | 3.x |
| responses | HTTP mocking | 0.23+ |
| factory_boy | Test data | 3.x |

### 11.2 Supporting Tools

| Tool | Purpose |
|------|---------|
| ruff | Linting |
| black | Code formatting |
| bandit | Security analysis |
| mypy | Type checking |
| tox | Multi-env testing |

### 11.3 Requirements (requirements-dev.txt)

```
# Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0
pytest-mock>=3.11.0
pytest-benchmark>=4.0.0
responses>=0.23.0
factory-boy>=3.3.0
faker>=19.0.0
httpx>=0.24.0

# Code quality
ruff>=0.1.0
black>=23.0.0
isort>=5.12.0
mypy>=1.5.0

# Security
bandit>=1.7.0
safety>=2.3.0
pip-audit>=2.6.0
```

---

## 12. VCP Alert System Test Strategy

### 12.1 Overview

The VCP Alert System is a critical component requiring comprehensive testing due to its:
- **State management** - Alert lifecycle (PENDING → NOTIFIED → CONVERTED → EXPIRED)
- **Persistence requirements** - All alerts must be reliably stored and queryable
- **Notification reliability** - Alerts must reach users without duplication or loss
- **Pattern detection accuracy** - Incorrect alerts impact trading decisions

### 12.2 Test Categories

| Category | Focus Area | Coverage Target |
|----------|------------|-----------------|
| Unit Tests | Individual components in isolation | 90% |
| Integration Tests | Component interactions, DB operations | 85% |
| State Machine Tests | Alert state transitions | 100% |
| Notification Tests | Channel delivery, formatting | 80% |

### 12.3 Component Test Focus

#### VCPDetector
- Pattern detection with known VCP stocks (positive cases)
- Rejection of non-VCP patterns (negative cases)
- Edge cases: staircase patterns, loosening contractions
- Score calculation accuracy

#### AlertManager
- Alert creation with proper deduplication
- State transition correctness
- Alert chain linkage (parent_alert_id)
- Expiration logic
- Subscriber notification

#### AlertRepository
- CRUD operations
- Query filtering (by symbol, type, state, date range)
- Concurrent access handling
- Data integrity on restarts

#### NotificationHub
- Channel registration
- Multi-channel dispatch
- Failure isolation (one channel failure doesn't block others)
- Message formatting per channel

### 12.4 Test Data Requirements

**VCP Pattern Fixtures:**
```python
# Known VCP patterns from backtest (use as positive test cases)
VCP_POSITIVE_CASES = [
    {"symbol": "NVDA", "date": "2024-01-08", "contractions": 2, "score": 96},
    {"symbol": "TSLA", "date": "2023-06-15", "contractions": 4, "score": 85},
]

# Known non-VCP patterns (use as negative test cases)
VCP_NEGATIVE_CASES = [
    {"symbol": "AAPL", "date": "2023-05-01", "reason": "staircase"},
    {"symbol": "GOOGL", "date": "2023-08-15", "reason": "loosening"},
]
```

**Alert Fixtures:**
```python
# Sample alerts for state transition testing
SAMPLE_CONTRACTION_ALERT = Alert(
    symbol="NVDA",
    alert_type=AlertType.CONTRACTION,
    state=AlertState.PENDING,
    trigger_price=48.50,
    pivot_price=52.00,
    score=75,
)
```

### 12.5 State Machine Test Matrix

| Initial State | Trigger | Expected State | Validation |
|---------------|---------|----------------|------------|
| PENDING | notify() | NOTIFIED | updated_at changed |
| PENDING | convert(child_id) | CONVERTED | converted_at set, child linked |
| PENDING | expire() | EXPIRED | expired_at set |
| NOTIFIED | complete() | COMPLETED | (trade alerts only) |
| CONVERTED | any | CONVERTED | No state change allowed |
| EXPIRED | any | EXPIRED | No state change allowed |

### 12.6 Integration Test Scenarios

1. **Full Alert Chain Flow**
   - Detect pattern → Create contraction alert
   - Price approaches pivot → Create pre-alert, convert contraction
   - Entry signal fires → Create trade alert, convert pre-alert
   - Verify all alerts persisted with correct linkage

2. **Alert Expiration**
   - Create contraction alert
   - Advance time 21 days
   - Run expire_stale_alerts()
   - Verify alert state = EXPIRED

3. **Notification Delivery**
   - Subscribe mock handlers
   - Create alert
   - Verify all handlers called with correct alert data

4. **Repository Recovery**
   - Create alerts, close connection
   - Reopen connection
   - Verify all alerts retrievable

### 12.7 Performance Benchmarks

| Operation | Target | Measurement |
|-----------|--------|-------------|
| Pattern analysis (single symbol) | < 100ms | pytest-benchmark |
| Alert creation + persistence | < 50ms | pytest-benchmark |
| Alert query (by symbol) | < 10ms | pytest-benchmark |
| Full scan (100 symbols) | < 60s | Integration test |

### 12.8 Mocking Strategy

| External Dependency | Mock Approach |
|---------------------|---------------|
| Price data (yfinance) | Pre-loaded DataFrame fixtures |
| Database | SQLite in-memory |
| Notification channels | Mock handlers with call tracking |
| Current time | freezegun for time-based tests |

---

## 13. Multi-Agent Collaboration Framework

### 13.1 Overview

For complex testing tasks such as strategy backtesting, a multi-agent collaboration framework is employed to ensure quality and thoroughness. This framework uses specialized agents working in parallel with defined handoff protocols.

### 13.2 Agent Roles

| Agent | Role | Workspace | Responsibility |
|-------|------|-----------|----------------|
| **Supervisor** | Coordinator | N/A | Oversees agents, ensures SOP compliance, identifies blind spots |
| **Executor** | Task Runner | `agents/executor/` | Executes tests, produces results, documents methodology |
| **Reviewer** | Quality Assurance | `agents/reviewer/` | Validates results, checks methodology, flags issues |

### 13.3 Workspace Structure

```
agents/
├── AGENT_PROTOCOL.md           # Collaboration protocol (required reading)
├── executor/
│   ├── status.md               # Current execution status
│   ├── execution_log.md        # Detailed execution log
│   ├── results/                # Raw results (intermediate)
│   └── handoff/                # Files ready for reviewer
└── reviewer/
    ├── status.md               # Current review status
    ├── review_log.md           # Detailed review notes
    ├── issues/                 # Identified issues
    └── approved/               # Approved for final delivery
```

### 13.4 Communication Protocol

**Handoff Mechanism:**
1. Executor completes work → places in `executor/handoff/`
2. Executor updates `executor/status.md` with handoff notice
3. Reviewer monitors status, picks up handoff
4. Reviewer validates → moves to `approved/` OR creates issue in `issues/`
5. Supervisor moves approved work to final destination

**Status File Requirements:**
- Current phase and description
- Timestamp of last update
- Pending handoffs list
- Blocking issues
- Notes for other agents

### 13.5 Quality Gates

**Before Handoff (Executor):**
- [ ] Results are complete and formatted
- [ ] Execution log documents methodology
- [ ] Status file updated with handoff notice

**Before Approval (Reviewer):**
- [ ] Methodology aligns with specifications
- [ ] Results are statistically valid
- [ ] No data quality issues detected
- [ ] Success criteria met

**Before Final Delivery (Supervisor):**
- [ ] Both agents have signed off
- [ ] Results are reproducible
- [ ] Documentation is complete

### 13.6 Issue Resolution

1. **Technical Issues**: Reviewer creates `ISSUE_[phase]_[number].md` in `issues/`
2. **Executor Response**: Must address before proceeding to next phase
3. **Escalation**: Unresolved issues escalate to Supervisor
4. **Documentation**: All issues and resolutions logged for future reference

### 13.7 Cross-Agent Visibility

Agents are permitted and encouraged to:
- Inspect each other's workspace folders
- Read status files for coordination
- Reference execution logs for context
- Review approved deliverables for consistency

### 13.8 Final Deliverables

Only Supervisor-approved work moves to production folders:
- Test results → appropriate `results/` folder
- Documentation → `docs/` folder
- Validated findings → project-specific location

---

## 14. Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-11-28 | Claude | Initial draft |
| 1.1.0 | 2025-11-29 | Claude | Added multi-agent collaboration framework (Section 13) |
| 1.2.0 | 2025-11-30 | Claude | Added VCP Alert System test strategy (Section 12) |
