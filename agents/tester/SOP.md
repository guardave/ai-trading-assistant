# Tester Agent SOP

## Role
Validate the VCP Alert System implementation against requirements and test plans.

## Inputs

| Document | Location | Purpose |
|----------|----------|---------|
| System Requirements | `docs/02-system-requirements.md` Section 5 | Validate against TDD specs |
| Test Strategy | `docs/06-test-strategy.md` Section 12 | Testing approach and coverage targets |
| Test Plan | `docs/07-test-plan.md` Section 10 | Specific test cases to execute |
| Source Code | `src/vcp/` | Code under test |
| Developer Handoff | `agents/developer/handoff.md` | Implementation notes |

## Outputs

| Artifact | Location | Description |
|----------|----------|-------------|
| Test Results | `agents/tester/results.md` | Execution summary |
| Issues/Defects | `agents/tester/issues.md` | Bugs found for Developer |
| Coverage Report | `agents/tester/coverage.html` | Code coverage analysis |

## Test Categories

### Unit Tests (32 cases)
From `07-test-plan.md` Section 10.2:
- AM-01 to AM-08: Alert Manager
- VD-01 to VD-08: VCP Detector
- MG-01 to MG-06: Models
- RP-01 to RP-06: Repository
- NH-01 to NH-04: Notification Hub

### Integration Tests (7 cases)
From `07-test-plan.md` Section 10.3:
- VI-01 to VI-07: End-to-end workflows

### State Machine Tests
From `06-test-strategy.md` Section 12.3:
- All valid transitions
- Invalid transition rejection
- Edge cases (double conversion, expired conversion)

## Workflow

1. **Review** developer handoff report
2. **Setup** test environment and fixtures
3. **Execute** unit tests: `pytest tests/vcp/ -v`
4. **Execute** integration tests: `pytest tests/vcp/integration/ -v`
5. **Measure** coverage: `pytest --cov=src/vcp tests/vcp/`
6. **Validate** against requirements (TDD specs)
7. **Document** results and raise issues

## Issue Report Template

```markdown
# Issue: [Title]

## Severity
- [ ] Critical - Blocks functionality
- [ ] Major - Incorrect behavior
- [ ] Minor - Edge case or improvement

## Requirement Reference
- TDD Spec: SRS-XXX-XX

## Description
[What is wrong]

## Steps to Reproduce
1. ...
2. ...

## Expected Behavior
[From requirements]

## Actual Behavior
[What happened]

## Evidence
- Test case: [name]
- Log output: [if applicable]
```

## Test Results Template

```markdown
# Test Results - Phase X

## Summary
| Category | Total | Passed | Failed | Skipped |
|----------|-------|--------|--------|---------|
| Unit | X | X | X | X |
| Integration | X | X | X | X |

## Coverage
- Overall: X%
- Target: 90% unit, 85% integration

## Requirements Validation
| Spec ID | Description | Status |
|---------|-------------|--------|
| SRS-ALERT-01 | ... | Pass/Fail |

## Issues Raised
- [Issue links or references]

## Recommendation
- [ ] Ready for next phase
- [ ] Needs fixes (see issues)
```

## Quality Checklist

- [ ] All test cases from Section 10.2 executed
- [ ] All integration tests from Section 10.3 executed
- [ ] State machine transitions 100% covered
- [ ] Coverage meets targets (90% unit, 85% integration)
- [ ] All issues documented with reproduction steps
- [ ] Requirements traceability verified
