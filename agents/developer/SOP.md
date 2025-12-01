# Developer Agent SOP

## Role
Implement the VCP Alert System according to specifications and design documents.

## Inputs

| Document | Location | Purpose |
|----------|----------|---------|
| System Requirements | `docs/02-system-requirements.md` Section 5 | TDD specifications to implement |
| System Design | `docs/03-system-design.md` Section 8 | Architecture, data models, DB schema |
| Existing Code | `backtest/vcp_detector.py`, `backtest/run_backtest_dual_alert.py` | Reference implementation |
| Issues/Defects | From Tester Agent | Bugs to fix, improvements needed |

## Outputs

| Artifact | Location | Description |
|----------|----------|-------------|
| Source Code | `src/vcp/` | Implementation files |
| Unit Tests | `tests/vcp/` | Tests for each component |
| Handoff Report | `agents/developer/handoff.md` | Summary for Tester |

## Implementation Phases

### Phase 1: Core Data Models
- File: `src/vcp/models.py`
- Implement: `AlertType`, `AlertStatus`, `Alert`, `AlertChain`, `VCPPattern`
- Tests: `tests/vcp/test_models.py`

### Phase 2: Repository Layer
- File: `src/vcp/repository.py`
- Implement: `AlertRepository` protocol, `SQLiteAlertRepository`
- Tests: `tests/vcp/test_repository.py`

### Phase 3: Alert Manager
- File: `src/vcp/alert_manager.py`
- Implement: `AlertManager` class with state machine
- Tests: `tests/vcp/test_alert_manager.py`

### Phase 4: VCP Detector Refactor
- File: `src/vcp/detector.py`
- Implement: `VCPDetector` class (refactor from backtest version)
- Tests: `tests/vcp/test_detector.py`

### Phase 5: Notification Hub
- File: `src/vcp/notifications.py`
- Implement: `NotificationChannel` protocol, `NotificationHub`, `LogNotificationChannel`
- Tests: `tests/vcp/test_notifications.py`

### Phase 6: Main Orchestrator
- File: `src/vcp/alert_system.py`
- Implement: `VCPAlertSystem` class
- Tests: `tests/vcp/test_alert_system.py`

## Workflow

1. **Read** design docs and requirements before starting each phase
2. **Implement** code following TDD specifications
3. **Write** unit tests for all public methods
4. **Run** tests locally to verify: `pytest tests/vcp/`
5. **Document** any deviations or decisions in handoff report
6. **Commit** after each phase completion

## Handoff Report Template

```markdown
# Developer Handoff - Phase X

## Completed
- [ ] Files created/modified
- [ ] Tests written and passing

## Test Results
- Total tests: X
- Passed: X
- Coverage: X%

## Notes
- Any deviations from design
- Implementation decisions made

## Ready for Testing
- Components: [list]
- Test focus areas: [list]
```

## Quality Checklist

- [ ] All TDD specs from Section 5 implemented
- [ ] Type hints on all functions
- [ ] Docstrings on all public methods
- [ ] No hardcoded values (use config)
- [ ] Error handling for edge cases
- [ ] Unit tests with >90% coverage
