"""
Unit tests for VCP Alert System Repository

Test cases from test plan (07-test-plan.md Section 10.2):
- RP-01: Save and retrieve alert
- RP-02: Update alert state
- RP-03: Query by symbol
- RP-04: Query by alert type
- RP-05: Query by date range
- RP-06: Get children of parent alert
"""

import pytest
import tempfile
import os
from datetime import datetime, timedelta
from pathlib import Path

from src.vcp.models import Alert, AlertType, AlertState
from src.vcp.repository import SQLiteAlertRepository, InMemoryAlertRepository


class TestSQLiteAlertRepository:
    """Tests for SQLiteAlertRepository."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        yield path
        os.unlink(path)

    @pytest.fixture
    def repo(self, temp_db):
        """Create a repository with temporary database."""
        return SQLiteAlertRepository(temp_db)

    @pytest.fixture
    def sample_alert(self):
        """Create a sample alert for testing."""
        return Alert(
            symbol="AAPL",
            alert_type=AlertType.CONTRACTION,
            trigger_price=150.0,
            pivot_price=160.0,
            distance_to_pivot_pct=6.67,
            score=85.0,
            num_contractions=3,
            pattern_snapshot={"test": "data"},
        )

    def test_save_and_retrieve(self, repo, sample_alert):
        """RP-01: Save and retrieve alert."""
        repo.save(sample_alert)
        retrieved = repo.get_by_id(sample_alert.id)

        assert retrieved is not None
        assert retrieved.id == sample_alert.id
        assert retrieved.symbol == sample_alert.symbol
        assert retrieved.alert_type == sample_alert.alert_type
        assert retrieved.trigger_price == sample_alert.trigger_price
        assert retrieved.pattern_snapshot == sample_alert.pattern_snapshot

    def test_get_nonexistent(self, repo):
        """Get returns None for nonexistent ID."""
        result = repo.get_by_id("nonexistent-id")
        assert result is None

    def test_update_alert_state(self, repo, sample_alert):
        """RP-02: Update alert state."""
        repo.save(sample_alert)

        # Update the alert
        sample_alert.mark_notified()
        repo.update(sample_alert)

        # Retrieve and verify
        retrieved = repo.get_by_id(sample_alert.id)
        assert retrieved.state == AlertState.NOTIFIED
        assert retrieved.notified_at is not None

    def test_delete_alert(self, repo, sample_alert):
        """Delete removes alert from repository."""
        repo.save(sample_alert)
        assert repo.get_by_id(sample_alert.id) is not None

        repo.delete(sample_alert.id)
        assert repo.get_by_id(sample_alert.id) is None

    def test_query_by_symbol(self, repo):
        """RP-03: Query by symbol."""
        # Create alerts for different symbols
        alert1 = Alert(
            symbol="AAPL",
            alert_type=AlertType.CONTRACTION,
            trigger_price=150.0, pivot_price=160.0,
            distance_to_pivot_pct=6.67, score=80.0, num_contractions=2,
        )
        alert2 = Alert(
            symbol="NVDA",
            alert_type=AlertType.CONTRACTION,
            trigger_price=450.0, pivot_price=480.0,
            distance_to_pivot_pct=6.67, score=85.0, num_contractions=3,
        )
        alert3 = Alert(
            symbol="AAPL",
            alert_type=AlertType.PRE_ALERT,
            trigger_price=158.0, pivot_price=160.0,
            distance_to_pivot_pct=1.25, score=82.0, num_contractions=2,
        )

        repo.save(alert1)
        repo.save(alert2)
        repo.save(alert3)

        # Query by symbol
        aapl_alerts = repo.query(symbol="AAPL")
        assert len(aapl_alerts) == 2
        assert all(a.symbol == "AAPL" for a in aapl_alerts)

        nvda_alerts = repo.query(symbol="NVDA")
        assert len(nvda_alerts) == 1
        assert nvda_alerts[0].symbol == "NVDA"

    def test_query_by_alert_type(self, repo):
        """RP-04: Query by alert type."""
        alert1 = Alert(
            symbol="AAPL",
            alert_type=AlertType.CONTRACTION,
            trigger_price=150.0, pivot_price=160.0,
            distance_to_pivot_pct=6.67, score=80.0, num_contractions=2,
        )
        alert2 = Alert(
            symbol="AAPL",
            alert_type=AlertType.PRE_ALERT,
            trigger_price=158.0, pivot_price=160.0,
            distance_to_pivot_pct=1.25, score=82.0, num_contractions=2,
        )
        alert3 = Alert(
            symbol="AAPL",
            alert_type=AlertType.TRADE,
            trigger_price=161.0, pivot_price=160.0,
            distance_to_pivot_pct=-0.63, score=85.0, num_contractions=2,
        )

        repo.save(alert1)
        repo.save(alert2)
        repo.save(alert3)

        contraction_alerts = repo.query(alert_type=AlertType.CONTRACTION)
        assert len(contraction_alerts) == 1
        assert contraction_alerts[0].alert_type == AlertType.CONTRACTION

        trade_alerts = repo.query(alert_type=AlertType.TRADE)
        assert len(trade_alerts) == 1
        assert trade_alerts[0].alert_type == AlertType.TRADE

    def test_query_by_state(self, repo):
        """Query by state."""
        alert1 = Alert(
            symbol="AAPL",
            alert_type=AlertType.CONTRACTION,
            trigger_price=150.0, pivot_price=160.0,
            distance_to_pivot_pct=6.67, score=80.0, num_contractions=2,
        )
        alert2 = Alert(
            symbol="NVDA",
            alert_type=AlertType.CONTRACTION,
            trigger_price=450.0, pivot_price=480.0,
            distance_to_pivot_pct=6.67, score=85.0, num_contractions=3,
        )
        alert2.mark_notified()

        repo.save(alert1)
        repo.save(alert2)

        pending = repo.query(state=AlertState.PENDING)
        assert len(pending) == 1
        assert pending[0].symbol == "AAPL"

        notified = repo.query(state=AlertState.NOTIFIED)
        assert len(notified) == 1
        assert notified[0].symbol == "NVDA"

    def test_query_by_date_range(self, repo):
        """RP-05: Query by date range."""
        now = datetime.now()

        alert1 = Alert(
            symbol="AAPL",
            alert_type=AlertType.CONTRACTION,
            trigger_price=150.0, pivot_price=160.0,
            distance_to_pivot_pct=6.67, score=80.0, num_contractions=2,
        )
        alert1.created_at = now - timedelta(days=10)

        alert2 = Alert(
            symbol="NVDA",
            alert_type=AlertType.CONTRACTION,
            trigger_price=450.0, pivot_price=480.0,
            distance_to_pivot_pct=6.67, score=85.0, num_contractions=3,
        )
        alert2.created_at = now - timedelta(days=5)

        alert3 = Alert(
            symbol="TSLA",
            alert_type=AlertType.CONTRACTION,
            trigger_price=200.0, pivot_price=220.0,
            distance_to_pivot_pct=10.0, score=75.0, num_contractions=2,
        )
        alert3.created_at = now

        repo.save(alert1)
        repo.save(alert2)
        repo.save(alert3)

        # Query last 7 days
        recent = repo.query(start_date=now - timedelta(days=7))
        assert len(recent) == 2

        # Query older than 7 days
        older = repo.query(end_date=now - timedelta(days=7))
        assert len(older) == 1
        assert older[0].symbol == "AAPL"

    def test_get_children(self, repo):
        """RP-06: Get children of parent alert."""
        # Create parent
        parent = Alert(
            symbol="AAPL",
            alert_type=AlertType.CONTRACTION,
            trigger_price=150.0, pivot_price=160.0,
            distance_to_pivot_pct=6.67, score=80.0, num_contractions=2,
        )
        repo.save(parent)

        # Create children
        child1 = Alert(
            symbol="AAPL",
            alert_type=AlertType.PRE_ALERT,
            trigger_price=158.0, pivot_price=160.0,
            distance_to_pivot_pct=1.25, score=82.0, num_contractions=2,
            parent_alert_id=parent.id,
        )
        child2 = Alert(
            symbol="AAPL",
            alert_type=AlertType.TRADE,
            trigger_price=161.0, pivot_price=160.0,
            distance_to_pivot_pct=-0.63, score=85.0, num_contractions=2,
            parent_alert_id=parent.id,
        )
        repo.save(child1)
        repo.save(child2)

        # Get children
        children = repo.get_children(parent.id)
        assert len(children) == 2
        assert all(c.parent_alert_id == parent.id for c in children)

    def test_get_expiring(self, repo):
        """Get alerts that should be expired."""
        now = datetime.now()

        # Old alert that should expire
        old_alert = Alert(
            symbol="AAPL",
            alert_type=AlertType.CONTRACTION,
            trigger_price=150.0, pivot_price=160.0,
            distance_to_pivot_pct=6.67, score=80.0, num_contractions=2,
        )
        old_alert.created_at = now - timedelta(days=30)
        repo.save(old_alert)

        # Recent alert that should not expire
        new_alert = Alert(
            symbol="NVDA",
            alert_type=AlertType.CONTRACTION,
            trigger_price=450.0, pivot_price=480.0,
            distance_to_pivot_pct=6.67, score=85.0, num_contractions=3,
        )
        new_alert.created_at = now
        repo.save(new_alert)

        # Get alerts older than 7 days
        expiring = repo.get_expiring(now - timedelta(days=7))
        assert len(expiring) == 1
        assert expiring[0].symbol == "AAPL"

    def test_get_active_by_symbol(self, repo):
        """Get active alerts for a symbol."""
        alert1 = Alert(
            symbol="AAPL",
            alert_type=AlertType.CONTRACTION,
            trigger_price=150.0, pivot_price=160.0,
            distance_to_pivot_pct=6.67, score=80.0, num_contractions=2,
        )
        alert2 = Alert(
            symbol="AAPL",
            alert_type=AlertType.PRE_ALERT,
            trigger_price=158.0, pivot_price=160.0,
            distance_to_pivot_pct=1.25, score=82.0, num_contractions=2,
        )
        alert2.mark_expired()

        repo.save(alert1)
        repo.save(alert2)

        active = repo.get_active_by_symbol("AAPL")
        assert len(active) == 1
        assert active[0].state == AlertState.PENDING

    def test_count_by_state(self, repo):
        """Count alerts by state."""
        alert1 = Alert(
            symbol="AAPL",
            alert_type=AlertType.CONTRACTION,
            trigger_price=150.0, pivot_price=160.0,
            distance_to_pivot_pct=6.67, score=80.0, num_contractions=2,
        )
        alert2 = Alert(
            symbol="NVDA",
            alert_type=AlertType.CONTRACTION,
            trigger_price=450.0, pivot_price=480.0,
            distance_to_pivot_pct=6.67, score=85.0, num_contractions=3,
        )
        alert2.mark_notified()

        repo.save(alert1)
        repo.save(alert2)

        counts = repo.count_by_state()
        assert counts.get("pending", 0) == 1
        assert counts.get("notified", 0) == 1

    def test_clear_all(self, repo, sample_alert):
        """Clear all alerts."""
        repo.save(sample_alert)
        count = repo.clear_all()
        assert count == 1
        assert repo.get_by_id(sample_alert.id) is None

    def test_combined_query(self, repo):
        """Query with multiple criteria."""
        now = datetime.now()

        alert1 = Alert(
            symbol="AAPL",
            alert_type=AlertType.CONTRACTION,
            trigger_price=150.0, pivot_price=160.0,
            distance_to_pivot_pct=6.67, score=80.0, num_contractions=2,
        )
        alert1.created_at = now - timedelta(days=2)

        alert2 = Alert(
            symbol="AAPL",
            alert_type=AlertType.PRE_ALERT,
            trigger_price=158.0, pivot_price=160.0,
            distance_to_pivot_pct=1.25, score=82.0, num_contractions=2,
        )

        repo.save(alert1)
        repo.save(alert2)

        # Query AAPL contractions
        results = repo.query(symbol="AAPL", alert_type=AlertType.CONTRACTION)
        assert len(results) == 1
        assert results[0].alert_type == AlertType.CONTRACTION


class TestInMemoryAlertRepository:
    """Tests for InMemoryAlertRepository."""

    @pytest.fixture
    def repo(self):
        """Create an in-memory repository."""
        return InMemoryAlertRepository()

    @pytest.fixture
    def sample_alert(self):
        """Create a sample alert for testing."""
        return Alert(
            symbol="AAPL",
            alert_type=AlertType.CONTRACTION,
            trigger_price=150.0,
            pivot_price=160.0,
            distance_to_pivot_pct=6.67,
            score=85.0,
            num_contractions=3,
        )

    def test_save_and_retrieve(self, repo, sample_alert):
        """Save and retrieve alert."""
        repo.save(sample_alert)
        retrieved = repo.get_by_id(sample_alert.id)

        assert retrieved is not None
        assert retrieved.id == sample_alert.id

    def test_update(self, repo, sample_alert):
        """Update alert."""
        repo.save(sample_alert)
        sample_alert.mark_notified()
        repo.update(sample_alert)

        retrieved = repo.get_by_id(sample_alert.id)
        assert retrieved.state == AlertState.NOTIFIED

    def test_delete(self, repo, sample_alert):
        """Delete alert."""
        repo.save(sample_alert)
        repo.delete(sample_alert.id)
        assert repo.get_by_id(sample_alert.id) is None

    def test_query_by_symbol(self, repo):
        """Query by symbol."""
        alert1 = Alert(
            symbol="AAPL",
            alert_type=AlertType.CONTRACTION,
            trigger_price=150.0, pivot_price=160.0,
            distance_to_pivot_pct=6.67, score=80.0, num_contractions=2,
        )
        alert2 = Alert(
            symbol="NVDA",
            alert_type=AlertType.CONTRACTION,
            trigger_price=450.0, pivot_price=480.0,
            distance_to_pivot_pct=6.67, score=85.0, num_contractions=3,
        )

        repo.save(alert1)
        repo.save(alert2)

        results = repo.query(symbol="AAPL")
        assert len(results) == 1
        assert results[0].symbol == "AAPL"

    def test_get_children(self, repo):
        """Get children of parent."""
        parent = Alert(
            symbol="AAPL",
            alert_type=AlertType.CONTRACTION,
            trigger_price=150.0, pivot_price=160.0,
            distance_to_pivot_pct=6.67, score=80.0, num_contractions=2,
        )
        child = Alert(
            symbol="AAPL",
            alert_type=AlertType.PRE_ALERT,
            trigger_price=158.0, pivot_price=160.0,
            distance_to_pivot_pct=1.25, score=82.0, num_contractions=2,
            parent_alert_id=parent.id,
        )

        repo.save(parent)
        repo.save(child)

        children = repo.get_children(parent.id)
        assert len(children) == 1
        assert children[0].parent_alert_id == parent.id

    def test_clear_all(self, repo, sample_alert):
        """Clear all alerts."""
        repo.save(sample_alert)
        count = repo.clear_all()
        assert count == 1
        assert repo.get_by_id(sample_alert.id) is None
