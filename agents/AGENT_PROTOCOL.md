# Multi-Agent Collaboration Protocol

## Overview

This document defines the collaboration protocol for agents working on the AI Trading Assistant backtest system. All agents must adhere to this protocol for effective communication and quality assurance.

## Agent Roles

### 1. Backtest Executor Agent
- **Workspace**: `agents/executor/`
- **Responsibility**: Execute backtests according to defined phases, produce results
- **Outputs**:
  - Raw backtest results (CSV, JSON)
  - Execution logs
  - Status updates
- **Deliverables to `backtest/`**: Final validated results, charts, summary tables

### 2. Results Reviewer Agent
- **Workspace**: `agents/reviewer/`
- **Responsibility**: Review intermediate results, validate methodology, identify issues
- **Outputs**:
  - Review reports
  - Issue flags
  - Recommendations
- **Deliverables to `backtest/`**: Validated findings, quality-assured conclusions

### 3. Supervisor (Claude)
- **Responsibility**: Coordinate agents, ensure SOP compliance, resolve conflicts
- **Actions**:
  - Monitor agent workspaces
  - Identify blind spots in communication
  - Ensure deliverables meet quality standards

## Workspace Structure

```
agents/
├── AGENT_PROTOCOL.md           # This file
├── executor/
│   ├── status.md               # Current execution status
│   ├── execution_log.md        # Detailed execution log
│   ├── results/                # Raw results (intermediate)
│   └── handoff/                # Files ready for reviewer
└── reviewer/
    ├── status.md               # Current review status
    ├── review_log.md           # Detailed review notes
    ├── issues/                 # Identified issues
    └── approved/               # Approved for backtest/ folder
```

## Communication Protocol

### 1. Handoff Mechanism
- Executor places completed work in `executor/handoff/`
- Executor updates `executor/status.md` with handoff notice
- Reviewer monitors `executor/status.md` for new handoffs
- Reviewer moves approved work to `reviewer/approved/`
- Supervisor moves final approved work to `backtest/results/`

### 2. Status File Format
```markdown
# Agent Status

## Current Phase
[Phase name and description]

## Last Update
[Timestamp]

## Pending Handoffs
- [List of files ready for review]

## Blocking Issues
- [Any issues blocking progress]

## Notes for Other Agents
- [Cross-agent communication]
```

### 3. Issue Flagging
- Reviewer creates issue files in `reviewer/issues/`
- Issue filename: `ISSUE_[phase]_[number].md`
- Executor must address issues before proceeding

### 4. Approval Flow
1. Executor completes phase → handoff/
2. Reviewer validates → approved/ or issues/
3. If issues: Executor addresses, re-submits
4. If approved: Supervisor moves to backtest/

## Quality Gates

### Before Handoff (Executor)
- [ ] Results are complete and formatted
- [ ] Execution log documents methodology
- [ ] Status file updated with handoff notice

### Before Approval (Reviewer)
- [ ] Methodology aligns with research paper
- [ ] Results are statistically valid
- [ ] No data quality issues detected
- [ ] R:R ratio meets minimum threshold (3.0)
- [ ] Both exit methods tested (fixed/trailing)

### Before Final Delivery (Supervisor)
- [ ] Both agents have signed off
- [ ] Results are reproducible
- [ ] Documentation is complete
- [ ] Files properly formatted for user review

## Backtest Phases

| Phase | Description | Executor Output | Reviewer Focus |
|-------|-------------|-----------------|----------------|
| 1 | Baseline | baseline_results.csv | Parameter accuracy |
| 2 | Proximity | proximity_analysis.csv | Score correlation validity |
| 3 | RS Relaxation | rs_threshold_results.csv | Trade count vs quality |
| 4 | Stop/Target | rr_optimization.csv | R:R ratio compliance |
| 5 | Trailing vs Fixed | exit_comparison.csv | Fair comparison methodology |
| 6 | Summary | final_recommendations.md | Actionable conclusions |

## Conflict Resolution

1. Technical disagreements: Escalate to Supervisor with evidence
2. Methodology questions: Reference `strategy_research_paper.md`
3. Ambiguous requirements: Supervisor consults user

## File Naming Convention

- Results: `[phase]_[description]_[timestamp].csv`
- Logs: `[agent]_log_[date].md`
- Issues: `ISSUE_[phase]_[number].md`
- Approvals: `APPROVED_[phase]_[timestamp].md`

Timestamp format: `YYYYMMDD_HHMMSS`
