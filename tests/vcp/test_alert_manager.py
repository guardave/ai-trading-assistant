"""
Unit tests for VCP Alert System Alert Manager

Test cases from test plan (07-test-plan.md Section 10.2):
- AM-01: Create contraction alert for valid pattern
- AM-02: Reject duplicate contraction alert
- AM-03: Create pre-alert when price within threshold
- AM-04: Create trade alert on entry signal
- AM-05: Mark parent as converted when child created
- AM-06: Expire stale alerts based on TTL
- AM-07: Build alert chain from any alert
- AM-08: Subscribe and receive alert notifications
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from src.vcp.models import (
    Alert,
    AlertChain,
    AlertState,
    AlertType,
    Contraction,
    EntrySignal,
    EntryType,
    SwingPoint,
    VCPPattern,
)
from src.vcp.alert_manager import AlertManager, AlertConfig
from src.vcp.repository import InMemoryAlertRepository


class TestAlertManagerSetup:
    """Tests for AlertManager initialization and configuration."""

    def test_default_config(self):
        """AlertManager uses default config if none provided."""
        repo = InMemoryAlertRepository()
        manager = AlertManager(repo)

        assert manager.config.dedup_window_days == 7
        assert manager.config.pre_alert_proximity_pct == 3.0
        assert manager.config.min_score_contraction == 60.0

    def test_custom_config(self):
        """AlertManager uses provided config."""
        repo = InMemoryAlertRepository()
        config = AlertConfig(
            dedup_window_days=14,
            pre_alert_proximity_pct=5.0,
            min_score_contraction=70.0,
        )
        manager = AlertManager(repo, config)

        assert manager.config.dedup_window_days == 14
        assert manager.config.pre_alert_proximity_pct == 5.0


class TestContractionAlerts:
    """Tests for contraction alert creation."""

    @pytest.fixture
    def manager(self):
        """Create manager with in-memory repository."""
        repo = InMemoryAlertRepository()
        return AlertManager(repo)

    @pytest.fixture
    def valid_pattern(self):
        """Create a valid VCP pattern."""
        contractions = [
            Contraction(
                swing_high=SwingPoint(10, datetime(2024, 1, 10), 100.0, "high"),
                swing_low=SwingPoint(15, datetime(2024, 1, 15), 85.0, "low"),
                range_pct=17.6,
                duration_days=5,
                avg_volume_ratio=1.2
            ),
            Contraction(
                swing_high=SwingPoint(20, datetime(2024, 1, 25), 98.0, "high"),
                swing_low=SwingPoint(25, datetime(2024, 1, 30), 90.0, "low"),
                range_pct=8.9,
                duration_days=5,
                avg_volume_ratio=0.7
            ),
        ]
        return VCPPattern(
            symbol="AAPL",
            contractions=contractions,
            is_valid=True,
            validity_reasons=["Progressive tightening"],
            contraction_quality=85.0,
            volume_quality=75.0,
            proximity_score=80.0,
            pivot_price=100.0,
            support_price=90.0,
        )

    def test_create_contraction_alert_valid_pattern(self, manager, valid_pattern):
        """AM-01: Create contraction alert for valid pattern."""
        alert = manager.check_contraction_alert(valid_pattern, current_price=95.0)

        assert alert is not None
        assert alert.symbol == "AAPL"
        assert alert.alert_type == AlertType.CONTRACTION
        assert alert.state == AlertState.PENDING
        assert alert.pivot_price == 100.0
        assert alert.score == 80.0
        assert alert.num_contractions == 2

    def test_reject_invalid_pattern(self, manager):
        """Reject alert for invalid pattern."""
        pattern = VCPPattern(
            symbol="AAPL",
            contractions=[],
            is_valid=False,
            validity_reasons=["No contractions found"],
            contraction_quality=0.0,
            volume_quality=0.0,
            proximity_score=0.0,
            pivot_price=100.0,
            support_price=90.0,
        )
        alert = manager.check_contraction_alert(pattern, current_price=95.0)
        assert alert is None

    def test_reject_low_score_pattern(self, manager, valid_pattern):
        """Reject alert for pattern below score threshold."""
        valid_pattern.proximity_score = 50.0  # Below default 60.0 threshold
        alert = manager.check_contraction_alert(valid_pattern, current_price=95.0)
        assert alert is None

    def test_reject_insufficient_contractions(self, manager, valid_pattern):
        """Reject alert for pattern with too few contractions."""
        valid_pattern.contractions = valid_pattern.contractions[:1]  # Only 1 contraction
        alert = manager.check_contraction_alert(valid_pattern, current_price=95.0)
        assert alert is None

    def test_reject_duplicate_alert(self, manager, valid_pattern):
        """AM-02: Reject duplicate contraction alert."""
        # Create first alert
        alert1 = manager.check_contraction_alert(valid_pattern, current_price=95.0)
        assert alert1 is not None

        # Try to create duplicate
        alert2 = manager.check_contraction_alert(valid_pattern, current_price=96.0)
        assert alert2 is None

    def test_allow_alert_after_dedup_window(self, manager, valid_pattern):
        """Allow new alert after deduplication window expires."""
        # Create first alert and backdate it
        alert1 = manager.check_contraction_alert(valid_pattern, current_price=95.0)
        alert1.created_at = datetime.now() - timedelta(days=10)  # Beyond 7-day window
        manager.repository.update(alert1)

        # Should allow new alert
        alert2 = manager.check_contraction_alert(valid_pattern, current_price=96.0)
        assert alert2 is not None


class TestPreAlerts:
    """Tests for pre-alert creation."""

    @pytest.fixture
    def manager(self):
        """Create manager with in-memory repository."""
        repo = InMemoryAlertRepository()
        return AlertManager(repo)

    def test_create_pre_alert_within_threshold(self, manager):
        """AM-03: Create pre-alert when price within threshold."""
        # Price is 2% below pivot (within 3% threshold)
        alert = manager.check_pre_alert(
            symbol="NVDA",
            current_price=98.0,
            pivot_price=100.0,
            pattern_score=85.0,
            num_contractions=3,
        )

        assert alert is not None
        assert alert.alert_type == AlertType.PRE_ALERT
        assert alert.symbol == "NVDA"
        assert alert.distance_to_pivot_pct == pytest.approx(2.04, rel=0.1)

    def test_reject_pre_alert_beyond_threshold(self, manager):
        """Reject pre-alert when price beyond threshold."""
        # Price is 5% below pivot (beyond 3% threshold)
        alert = manager.check_pre_alert(
            symbol="NVDA",
            current_price=95.0,
            pivot_price=100.0,
            pattern_score=85.0,
            num_contractions=3,
        )
        assert alert is None

    def test_pre_alert_links_to_parent(self, manager):
        """Pre-alert links to parent contraction alert."""
        # Create parent first
        parent = Alert(
            symbol="AAPL",
            alert_type=AlertType.CONTRACTION,
            trigger_price=90.0,
            pivot_price=100.0,
            distance_to_pivot_pct=11.11,
            score=80.0,
            num_contractions=2,
        )
        manager.repository.save(parent)

        # Create pre-alert with parent
        alert = manager.check_pre_alert(
            symbol="AAPL",
            current_price=98.0,
            pivot_price=100.0,
            pattern_score=82.0,
            num_contractions=2,
            parent_alert_id=parent.id,
        )

        assert alert is not None
        assert alert.parent_alert_id == parent.id


class TestTradeAlerts:
    """Tests for trade alert creation."""

    @pytest.fixture
    def manager(self):
        """Create manager with in-memory repository."""
        repo = InMemoryAlertRepository()
        return AlertManager(repo)

    @pytest.fixture
    def entry_signal(self):
        """Create a sample entry signal."""
        return EntrySignal(
            entry_type=EntryType.PIVOT_BREAKOUT,
            date=datetime(2024, 2, 1),
            price=101.0,
            stop_price=93.0,
            target_price=125.0,
            volume_ratio=1.8,
        )

    def test_create_trade_alert(self, manager, entry_signal):
        """AM-04: Create trade alert on entry signal."""
        alert = manager.check_trade_alert(
            symbol="TSLA",
            entry_signal=entry_signal,
            pivot_price=100.0,
            pattern_score=90.0,
            num_contractions=3,
        )

        assert alert is not None
        assert alert.alert_type == AlertType.TRADE
        assert alert.symbol == "TSLA"
        assert alert.trigger_price == 101.0

    def test_trade_alert_links_to_parent(self, manager, entry_signal):
        """Trade alert links to parent pre-alert."""
        # Create parent pre-alert
        parent = Alert(
            symbol="TSLA",
            alert_type=AlertType.PRE_ALERT,
            trigger_price=99.0,
            pivot_price=100.0,
            distance_to_pivot_pct=1.01,
            score=85.0,
            num_contractions=3,
        )
        manager.repository.save(parent)

        # Create trade alert
        alert = manager.check_trade_alert(
            symbol="TSLA",
            entry_signal=entry_signal,
            pivot_price=100.0,
            pattern_score=90.0,
            num_contractions=3,
            parent_alert_id=parent.id,
        )

        assert alert.parent_alert_id == parent.id


class TestAlertConversion:
    """Tests for alert state conversions."""

    @pytest.fixture
    def manager(self):
        """Create manager with in-memory repository."""
        repo = InMemoryAlertRepository()
        return AlertManager(repo)

    def test_mark_parent_converted(self, manager):
        """AM-05: Mark parent as converted when child created."""
        # Create parent
        parent = Alert(
            symbol="AAPL",
            alert_type=AlertType.CONTRACTION,
            trigger_price=90.0,
            pivot_price=100.0,
            distance_to_pivot_pct=11.11,
            score=80.0,
            num_contractions=2,
        )
        manager.repository.save(parent)

        # Create pre-alert with parent
        manager.check_pre_alert(
            symbol="AAPL",
            current_price=98.0,
            pivot_price=100.0,
            pattern_score=82.0,
            num_contractions=2,
            parent_alert_id=parent.id,
        )

        # Verify parent is converted
        updated_parent = manager.repository.get_by_id(parent.id)
        assert updated_parent.state == AlertState.CONVERTED
        assert updated_parent.converted_at is not None

    def test_mark_notified(self, manager):
        """Mark alert as notified."""
        alert = Alert(
            symbol="AAPL",
            alert_type=AlertType.CONTRACTION,
            trigger_price=90.0,
            pivot_price=100.0,
            distance_to_pivot_pct=11.11,
            score=80.0,
            num_contractions=2,
        )
        manager.repository.save(alert)

        result = manager.mark_notified(alert.id)
        assert result is True

        updated = manager.repository.get_by_id(alert.id)
        assert updated.state == AlertState.NOTIFIED

    def test_mark_completed(self, manager):
        """Mark trade alert as completed."""
        alert = Alert(
            symbol="AAPL",
            alert_type=AlertType.TRADE,
            trigger_price=101.0,
            pivot_price=100.0,
            distance_to_pivot_pct=-1.0,
            score=85.0,
            num_contractions=2,
        )
        alert.mark_notified()
        manager.repository.save(alert)

        result = manager.mark_completed(alert.id)
        assert result is True

        updated = manager.repository.get_by_id(alert.id)
        assert updated.state == AlertState.COMPLETED


class TestAlertExpiration:
    """Tests for alert expiration."""

    @pytest.fixture
    def manager(self):
        """Create manager with in-memory repository."""
        repo = InMemoryAlertRepository()
        config = AlertConfig(
            contraction_ttl_days=30,
            pre_alert_ttl_days=14,
            trade_ttl_days=7,
        )
        return AlertManager(repo, config)

    def test_expire_stale_alerts(self, manager):
        """AM-06: Expire stale alerts based on TTL."""
        now = datetime.now()

        # Old contraction alert (should expire)
        old_alert = Alert(
            symbol="AAPL",
            alert_type=AlertType.CONTRACTION,
            trigger_price=90.0,
            pivot_price=100.0,
            distance_to_pivot_pct=11.11,
            score=80.0,
            num_contractions=2,
        )
        old_alert.created_at = now - timedelta(days=35)
        manager.repository.save(old_alert)

        # Recent alert (should not expire)
        new_alert = Alert(
            symbol="NVDA",
            alert_type=AlertType.CONTRACTION,
            trigger_price=450.0,
            pivot_price=480.0,
            distance_to_pivot_pct=6.67,
            score=85.0,
            num_contractions=3,
        )
        new_alert.created_at = now - timedelta(days=5)
        manager.repository.save(new_alert)

        # Expire stale alerts
        expired_count = manager.expire_stale_alerts()

        assert expired_count == 1
        old = manager.repository.get_by_id(old_alert.id)
        new = manager.repository.get_by_id(new_alert.id)
        assert old.state == AlertState.EXPIRED
        assert new.state == AlertState.PENDING


class TestAlertChains:
    """Tests for alert chain building."""

    @pytest.fixture
    def manager(self):
        """Create manager with in-memory repository."""
        repo = InMemoryAlertRepository()
        return AlertManager(repo)

    def test_build_alert_chain(self, manager):
        """AM-07: Build alert chain from any alert."""
        # Create full chain
        contraction = Alert(
            symbol="AAPL",
            alert_type=AlertType.CONTRACTION,
            trigger_price=90.0,
            pivot_price=100.0,
            distance_to_pivot_pct=11.11,
            score=80.0,
            num_contractions=2,
        )
        manager.repository.save(contraction)

        pre_alert = Alert(
            symbol="AAPL",
            alert_type=AlertType.PRE_ALERT,
            trigger_price=98.0,
            pivot_price=100.0,
            distance_to_pivot_pct=2.04,
            score=82.0,
            num_contractions=2,
            parent_alert_id=contraction.id,
        )
        manager.repository.save(pre_alert)

        trade = Alert(
            symbol="AAPL",
            alert_type=AlertType.TRADE,
            trigger_price=101.0,
            pivot_price=100.0,
            distance_to_pivot_pct=-1.0,
            score=85.0,
            num_contractions=2,
            parent_alert_id=pre_alert.id,
        )
        manager.repository.save(trade)

        # Build chain from trade alert
        chain = manager.get_alert_chain(trade.id)

        assert chain is not None
        assert chain.symbol == "AAPL"
        assert chain.contraction_alert.id == contraction.id
        assert len(chain.pre_alerts) == 1
        assert chain.pre_alerts[0].id == pre_alert.id
        assert chain.trade_alert.id == trade.id

    def test_build_chain_from_middle(self, manager):
        """Build chain from pre-alert in the middle."""
        contraction = Alert(
            symbol="NVDA",
            alert_type=AlertType.CONTRACTION,
            trigger_price=450.0,
            pivot_price=480.0,
            distance_to_pivot_pct=6.67,
            score=80.0,
            num_contractions=2,
        )
        manager.repository.save(contraction)

        pre_alert = Alert(
            symbol="NVDA",
            alert_type=AlertType.PRE_ALERT,
            trigger_price=470.0,
            pivot_price=480.0,
            distance_to_pivot_pct=2.13,
            score=82.0,
            num_contractions=2,
            parent_alert_id=contraction.id,
        )
        manager.repository.save(pre_alert)

        # Build chain from pre-alert
        chain = manager.get_alert_chain(pre_alert.id)

        assert chain.contraction_alert.id == contraction.id
        assert len(chain.pre_alerts) == 1


class TestSubscriptions:
    """Tests for subscriber notifications."""

    @pytest.fixture
    def manager(self):
        """Create manager with in-memory repository."""
        repo = InMemoryAlertRepository()
        return AlertManager(repo)

    @pytest.fixture
    def valid_pattern(self):
        """Create a valid VCP pattern."""
        contractions = [
            Contraction(
                swing_high=SwingPoint(10, datetime(2024, 1, 10), 100.0, "high"),
                swing_low=SwingPoint(15, datetime(2024, 1, 15), 85.0, "low"),
                range_pct=17.6,
                duration_days=5,
                avg_volume_ratio=1.2
            ),
            Contraction(
                swing_high=SwingPoint(20, datetime(2024, 1, 25), 98.0, "high"),
                swing_low=SwingPoint(25, datetime(2024, 1, 30), 90.0, "low"),
                range_pct=8.9,
                duration_days=5,
                avg_volume_ratio=0.7
            ),
        ]
        return VCPPattern(
            symbol="TSLA",
            contractions=contractions,
            is_valid=True,
            validity_reasons=["Progressive tightening"],
            contraction_quality=85.0,
            volume_quality=75.0,
            proximity_score=80.0,
            pivot_price=100.0,
            support_price=90.0,
        )

    def test_subscribe_and_receive(self, manager, valid_pattern):
        """AM-08: Subscribe and receive alert notifications."""
        received_alerts = []

        def handler(alert: Alert):
            received_alerts.append(alert)

        manager.subscribe(handler)
        manager.check_contraction_alert(valid_pattern, current_price=95.0)

        assert len(received_alerts) == 1
        assert received_alerts[0].symbol == "TSLA"

    def test_unsubscribe(self, manager, valid_pattern):
        """Unsubscribed handlers don't receive notifications."""
        received_alerts = []

        def handler(alert: Alert):
            received_alerts.append(alert)

        manager.subscribe(handler)
        manager.unsubscribe(handler)
        manager.check_contraction_alert(valid_pattern, current_price=95.0)

        assert len(received_alerts) == 0

    def test_multiple_subscribers(self, manager, valid_pattern):
        """Multiple subscribers all receive notifications."""
        received1 = []
        received2 = []

        manager.subscribe(lambda a: received1.append(a))
        manager.subscribe(lambda a: received2.append(a))

        manager.check_contraction_alert(valid_pattern, current_price=95.0)

        assert len(received1) == 1
        assert len(received2) == 1

    def test_subscriber_error_doesnt_break_others(self, manager, valid_pattern):
        """Error in one subscriber doesn't break others."""
        received = []

        def bad_handler(alert: Alert):
            raise Exception("Handler error")

        def good_handler(alert: Alert):
            received.append(alert)

        manager.subscribe(bad_handler)
        manager.subscribe(good_handler)

        # Should not raise
        manager.check_contraction_alert(valid_pattern, current_price=95.0)

        # Good handler should still receive
        assert len(received) == 1


class TestCheckAndEmit:
    """Tests for the check_and_emit method."""

    @pytest.fixture
    def manager(self):
        """Create manager with in-memory repository."""
        repo = InMemoryAlertRepository()
        return AlertManager(repo)

    @pytest.fixture
    def valid_pattern(self):
        """Create a valid VCP pattern."""
        contractions = [
            Contraction(
                swing_high=SwingPoint(10, datetime(2024, 1, 10), 100.0, "high"),
                swing_low=SwingPoint(15, datetime(2024, 1, 15), 85.0, "low"),
                range_pct=17.6,
                duration_days=5,
                avg_volume_ratio=1.2
            ),
            Contraction(
                swing_high=SwingPoint(20, datetime(2024, 1, 25), 98.0, "high"),
                swing_low=SwingPoint(25, datetime(2024, 1, 30), 90.0, "low"),
                range_pct=8.9,
                duration_days=5,
                avg_volume_ratio=0.7
            ),
        ]
        return VCPPattern(
            symbol="MSFT",
            contractions=contractions,
            is_valid=True,
            validity_reasons=["Progressive tightening"],
            contraction_quality=85.0,
            volume_quality=75.0,
            proximity_score=80.0,
            pivot_price=100.0,
            support_price=90.0,
        )

    def test_emit_contraction_alert(self, manager, valid_pattern):
        """Emit contraction alert when price far from pivot."""
        alerts = manager.check_and_emit(valid_pattern, current_price=90.0)

        assert len(alerts) == 1
        assert alerts[0].alert_type == AlertType.CONTRACTION

    def test_emit_pre_alert(self, manager, valid_pattern):
        """Emit pre-alert when price within threshold."""
        # First create contraction alert
        manager.check_and_emit(valid_pattern, current_price=90.0)

        # Now check with price near pivot
        alerts = manager.check_and_emit(valid_pattern, current_price=98.0)

        assert len(alerts) == 1
        assert alerts[0].alert_type == AlertType.PRE_ALERT

    def test_emit_trade_alert(self, manager, valid_pattern):
        """Emit trade alert when entry signal provided."""
        entry_signal = EntrySignal(
            entry_type=EntryType.PIVOT_BREAKOUT,
            date=datetime.now(),
            price=101.0,
            stop_price=93.0,
            target_price=125.0,
            volume_ratio=1.8,
        )

        alerts = manager.check_and_emit(
            valid_pattern,
            current_price=101.0,
            entry_signal=entry_signal,
        )

        assert len(alerts) == 1
        assert alerts[0].alert_type == AlertType.TRADE


class TestConversionStats:
    """Tests for conversion statistics."""

    @pytest.fixture
    def manager(self):
        """Create manager with in-memory repository."""
        repo = InMemoryAlertRepository()
        return AlertManager(repo)

    def test_get_conversion_stats(self, manager):
        """Calculate conversion statistics."""
        now = datetime.now()
        start = now - timedelta(days=30)

        # Create alerts
        contraction = Alert(
            symbol="AAPL",
            alert_type=AlertType.CONTRACTION,
            trigger_price=90.0,
            pivot_price=100.0,
            distance_to_pivot_pct=11.11,
            score=80.0,
            num_contractions=2,
        )
        contraction.created_at = now - timedelta(days=10)
        manager.repository.save(contraction)

        pre_alert = Alert(
            symbol="AAPL",
            alert_type=AlertType.PRE_ALERT,
            trigger_price=98.0,
            pivot_price=100.0,
            distance_to_pivot_pct=2.04,
            score=82.0,
            num_contractions=2,
            parent_alert_id=contraction.id,
        )
        pre_alert.created_at = now - timedelta(days=5)
        contraction.mark_converted(pre_alert.id)
        manager.repository.save(pre_alert)
        manager.repository.update(contraction)

        trade = Alert(
            symbol="AAPL",
            alert_type=AlertType.TRADE,
            trigger_price=101.0,
            pivot_price=100.0,
            distance_to_pivot_pct=-1.0,
            score=85.0,
            num_contractions=2,
            parent_alert_id=pre_alert.id,
        )
        trade.created_at = now - timedelta(days=2)
        pre_alert.mark_converted(trade.id)
        manager.repository.save(trade)
        manager.repository.update(pre_alert)

        stats = manager.get_conversion_stats(start, now)

        assert stats.total_contraction_alerts == 1
        assert stats.total_pre_alerts == 1
        assert stats.total_trade_alerts == 1
        assert stats.contraction_to_pre_alert_rate == 1.0
        assert stats.pre_alert_to_trade_rate == 1.0
