"""
Unit tests for VCP Alert System Models

Test cases from test plan (07-test-plan.md Section 10.2):
- MG-01: Alert creation with all fields
- MG-02: Alert state transitions
- MG-03: Alert serialization/deserialization
- MG-04: AlertChain lead time calculation
- MG-05: VCPPattern score calculation
- MG-06: EntrySignal risk/reward calculation
"""

import pytest
from datetime import datetime, timedelta
from src.vcp.models import (
    AlertType,
    AlertState,
    Alert,
    AlertChain,
    ConversionStats,
    VCPPattern,
    Contraction,
    SwingPoint,
    EntrySignal,
    EntryType,
)


class TestSwingPoint:
    """Tests for SwingPoint dataclass."""

    def test_creation(self):
        """SwingPoint can be created with all required fields."""
        sp = SwingPoint(
            index=10,
            date=datetime(2024, 1, 15),
            price=150.0,
            point_type="high"
        )
        assert sp.index == 10
        assert sp.price == 150.0
        assert sp.point_type == "high"

    def test_serialization(self):
        """SwingPoint can be serialized to dict and back."""
        sp = SwingPoint(
            index=10,
            date=datetime(2024, 1, 15, 10, 30),
            price=150.0,
            point_type="high"
        )
        data = sp.to_dict()
        restored = SwingPoint.from_dict(data)

        assert restored.index == sp.index
        assert restored.price == sp.price
        assert restored.point_type == sp.point_type


class TestContraction:
    """Tests for Contraction dataclass."""

    def test_creation(self):
        """Contraction can be created with swing points."""
        swing_high = SwingPoint(10, datetime(2024, 1, 10), 100.0, "high")
        swing_low = SwingPoint(15, datetime(2024, 1, 15), 90.0, "low")

        c = Contraction(
            swing_high=swing_high,
            swing_low=swing_low,
            range_pct=11.11,
            duration_days=5,
            avg_volume_ratio=0.8
        )
        assert c.range_pct == 11.11
        assert c.duration_days == 5

    def test_serialization(self):
        """Contraction can be serialized to dict and back."""
        swing_high = SwingPoint(10, datetime(2024, 1, 10), 100.0, "high")
        swing_low = SwingPoint(15, datetime(2024, 1, 15), 90.0, "low")

        c = Contraction(
            swing_high=swing_high,
            swing_low=swing_low,
            range_pct=11.11,
            duration_days=5,
            avg_volume_ratio=0.8
        )
        data = c.to_dict()
        restored = Contraction.from_dict(data)

        assert restored.range_pct == c.range_pct
        assert restored.swing_high.price == c.swing_high.price


class TestVCPPattern:
    """Tests for VCPPattern dataclass (MG-05)."""

    @pytest.fixture
    def sample_pattern(self):
        """Create a sample VCP pattern for testing."""
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
            validity_reasons=["Progressive tightening", "Volume dry-up"],
            contraction_quality=85.0,
            volume_quality=75.0,
            proximity_score=80.0,
            pivot_price=100.0,
            support_price=90.0,
        )

    def test_creation(self, sample_pattern):
        """VCPPattern can be created with all fields."""
        assert sample_pattern.symbol == "AAPL"
        assert sample_pattern.is_valid is True
        assert sample_pattern.pivot_price == 100.0

    def test_num_contractions(self, sample_pattern):
        """num_contractions property returns correct count."""
        assert sample_pattern.num_contractions == 2

    def test_last_contraction_range_pct(self, sample_pattern):
        """last_contraction_range_pct returns final contraction range."""
        assert sample_pattern.last_contraction_range_pct == 8.9

    def test_serialization(self, sample_pattern):
        """VCPPattern can be serialized to dict and back."""
        data = sample_pattern.to_dict()
        restored = VCPPattern.from_dict(data)

        assert restored.symbol == sample_pattern.symbol
        assert restored.num_contractions == sample_pattern.num_contractions
        assert restored.pivot_price == sample_pattern.pivot_price


class TestEntrySignal:
    """Tests for EntrySignal dataclass (MG-06)."""

    @pytest.fixture
    def sample_signal(self):
        """Create a sample entry signal."""
        return EntrySignal(
            entry_type=EntryType.PIVOT_BREAKOUT,
            date=datetime(2024, 2, 1),
            price=100.0,
            stop_price=92.0,
            target_price=124.0,
            volume_ratio=1.8
        )

    def test_creation(self, sample_signal):
        """EntrySignal can be created with all fields."""
        assert sample_signal.entry_type == EntryType.PIVOT_BREAKOUT
        assert sample_signal.price == 100.0
        assert sample_signal.stop_price == 92.0

    def test_risk_pct(self, sample_signal):
        """risk_pct calculates correctly."""
        # (100 - 92) / 100 * 100 = 8%
        assert sample_signal.risk_pct == 8.0

    def test_reward_risk_ratio(self, sample_signal):
        """reward_risk_ratio calculates correctly."""
        # Risk = 100 - 92 = 8
        # Reward = 124 - 100 = 24
        # R:R = 24 / 8 = 3.0
        assert sample_signal.reward_risk_ratio == 3.0

    def test_serialization(self, sample_signal):
        """EntrySignal can be serialized to dict and back."""
        data = sample_signal.to_dict()
        restored = EntrySignal.from_dict(data)

        assert restored.entry_type == sample_signal.entry_type
        assert restored.price == sample_signal.price
        assert restored.risk_pct == sample_signal.risk_pct


class TestAlert:
    """Tests for Alert dataclass (MG-01, MG-02, MG-03)."""

    @pytest.fixture
    def sample_alert(self):
        """Create a sample alert."""
        return Alert(
            symbol="NVDA",
            alert_type=AlertType.CONTRACTION,
            trigger_price=450.0,
            pivot_price=480.0,
            distance_to_pivot_pct=6.67,
            score=85.0,
            num_contractions=3,
            pattern_snapshot={"contractions": [{"range_pct": 15.0}]}
        )

    def test_creation_with_all_fields(self, sample_alert):
        """MG-01: Alert can be created with all required fields."""
        assert sample_alert.symbol == "NVDA"
        assert sample_alert.alert_type == AlertType.CONTRACTION
        assert sample_alert.state == AlertState.PENDING
        assert sample_alert.trigger_price == 450.0
        assert sample_alert.pivot_price == 480.0
        assert sample_alert.id is not None
        assert sample_alert.created_at is not None

    def test_state_transition_pending_to_notified(self, sample_alert):
        """MG-02: Alert transitions from PENDING to NOTIFIED."""
        assert sample_alert.state == AlertState.PENDING
        sample_alert.mark_notified()
        assert sample_alert.state == AlertState.NOTIFIED
        assert sample_alert.notified_at is not None

    def test_state_transition_pending_to_converted(self, sample_alert):
        """MG-02: Alert transitions from PENDING to CONVERTED."""
        sample_alert.mark_converted("child-123")
        assert sample_alert.state == AlertState.CONVERTED
        assert sample_alert.converted_at is not None

    def test_state_transition_notified_to_converted(self, sample_alert):
        """MG-02: Alert transitions from NOTIFIED to CONVERTED."""
        sample_alert.mark_notified()
        sample_alert.mark_converted("child-123")
        assert sample_alert.state == AlertState.CONVERTED

    def test_state_transition_to_expired(self, sample_alert):
        """MG-02: Alert transitions to EXPIRED."""
        sample_alert.mark_expired()
        assert sample_alert.state == AlertState.EXPIRED
        assert sample_alert.expired_at is not None

    def test_state_transition_to_completed(self):
        """MG-02: Trade alert transitions to COMPLETED."""
        trade_alert = Alert(
            symbol="AAPL",
            alert_type=AlertType.TRADE,
            trigger_price=175.0,
            pivot_price=175.0,
            distance_to_pivot_pct=0.0,
            score=90.0,
            num_contractions=2,
        )
        trade_alert.mark_notified()
        trade_alert.mark_completed()
        assert trade_alert.state == AlertState.COMPLETED

    def test_completed_only_for_trade_alerts(self, sample_alert):
        """MG-02: Only trade alerts can be marked completed."""
        sample_alert.mark_notified()
        sample_alert.mark_completed()
        # Should not change state for non-trade alerts
        assert sample_alert.state == AlertState.NOTIFIED

    def test_serialization(self, sample_alert):
        """MG-03: Alert can be serialized to dict and back."""
        sample_alert.mark_notified()
        data = sample_alert.to_dict()

        assert data["symbol"] == "NVDA"
        assert data["alert_type"] == "contraction"
        assert data["state"] == "notified"

        restored = Alert.from_dict(data)
        assert restored.symbol == sample_alert.symbol
        assert restored.alert_type == sample_alert.alert_type
        assert restored.state == sample_alert.state
        assert restored.id == sample_alert.id

    def test_invalid_state_transitions_ignored(self, sample_alert):
        """MG-02: Invalid state transitions are ignored."""
        sample_alert.mark_expired()
        # Try to mark notified after expired - should not change
        sample_alert.mark_notified()
        assert sample_alert.state == AlertState.EXPIRED


class TestAlertChain:
    """Tests for AlertChain dataclass (MG-04)."""

    @pytest.fixture
    def full_chain(self):
        """Create a full alert chain for testing."""
        chain = AlertChain(symbol="TSLA")

        # Contraction alert - Day 1
        chain.contraction_alert = Alert(
            symbol="TSLA",
            alert_type=AlertType.CONTRACTION,
            trigger_price=200.0,
            pivot_price=220.0,
            distance_to_pivot_pct=10.0,
            score=75.0,
            num_contractions=2,
        )
        chain.contraction_alert.created_at = datetime(2024, 1, 1)

        # Pre-alert - Day 5
        pre_alert = Alert(
            symbol="TSLA",
            alert_type=AlertType.PRE_ALERT,
            trigger_price=215.0,
            pivot_price=220.0,
            distance_to_pivot_pct=2.3,
            score=80.0,
            num_contractions=2,
            parent_alert_id=chain.contraction_alert.id,
        )
        pre_alert.created_at = datetime(2024, 1, 5)
        chain.add_pre_alert(pre_alert)

        # Trade alert - Day 8
        chain.trade_alert = Alert(
            symbol="TSLA",
            alert_type=AlertType.TRADE,
            trigger_price=221.0,
            pivot_price=220.0,
            distance_to_pivot_pct=-0.5,
            score=85.0,
            num_contractions=2,
            parent_alert_id=pre_alert.id,
        )
        chain.trade_alert.created_at = datetime(2024, 1, 8)

        return chain

    def test_total_lead_time_days(self, full_chain):
        """MG-04: Lead time calculated correctly for full chain."""
        # Day 8 - Day 1 = 7 days
        assert full_chain.total_lead_time_days == 7

    def test_pre_alert_lead_time_days(self, full_chain):
        """MG-04: Pre-alert lead time calculated correctly."""
        # Day 8 - Day 5 = 3 days
        assert full_chain.pre_alert_lead_time_days == 3

    def test_has_full_chain(self, full_chain):
        """AlertChain.has_full_chain returns True for complete chain."""
        assert full_chain.has_full_chain is True

    def test_partial_chain(self):
        """AlertChain.has_full_chain returns False for partial chain."""
        chain = AlertChain(symbol="AAPL")
        chain.contraction_alert = Alert(
            symbol="AAPL",
            alert_type=AlertType.CONTRACTION,
            trigger_price=150.0,
            pivot_price=160.0,
            distance_to_pivot_pct=6.67,
            score=70.0,
            num_contractions=2,
        )
        assert chain.has_full_chain is False
        assert chain.total_lead_time_days is None

    def test_add_pre_alert(self):
        """AlertChain.add_pre_alert adds only pre-alerts."""
        chain = AlertChain(symbol="TEST")

        pre_alert = Alert(
            symbol="TEST",
            alert_type=AlertType.PRE_ALERT,
            trigger_price=100.0,
            pivot_price=105.0,
            distance_to_pivot_pct=5.0,
            score=75.0,
            num_contractions=2,
        )
        chain.add_pre_alert(pre_alert)
        assert len(chain.pre_alerts) == 1

        # Try adding non-pre-alert
        contraction_alert = Alert(
            symbol="TEST",
            alert_type=AlertType.CONTRACTION,
            trigger_price=95.0,
            pivot_price=105.0,
            distance_to_pivot_pct=10.5,
            score=70.0,
            num_contractions=2,
        )
        chain.add_pre_alert(contraction_alert)
        # Should not be added
        assert len(chain.pre_alerts) == 1

    def test_serialization(self, full_chain):
        """AlertChain can be serialized to dict."""
        data = full_chain.to_dict()
        assert data["symbol"] == "TSLA"
        assert data["contraction_alert"] is not None
        assert len(data["pre_alerts"]) == 1
        assert data["trade_alert"] is not None


class TestConversionStats:
    """Tests for ConversionStats dataclass."""

    def test_creation(self):
        """ConversionStats can be created with all fields."""
        stats = ConversionStats(
            period_start=datetime(2024, 1, 1),
            period_end=datetime(2024, 3, 31),
            total_contraction_alerts=100,
            total_pre_alerts=50,
            total_trade_alerts=25,
            contraction_to_pre_alert_rate=0.50,
            contraction_to_trade_rate=0.25,
            pre_alert_to_trade_rate=0.50,
            avg_days_contraction_to_trade=8.5,
            avg_days_pre_alert_to_trade=5.0,
        )
        assert stats.total_contraction_alerts == 100
        assert stats.contraction_to_trade_rate == 0.25

    def test_serialization(self):
        """ConversionStats can be serialized to dict."""
        stats = ConversionStats(
            period_start=datetime(2024, 1, 1),
            period_end=datetime(2024, 3, 31),
            total_contraction_alerts=100,
            total_pre_alerts=50,
            total_trade_alerts=25,
            contraction_to_pre_alert_rate=0.50,
            contraction_to_trade_rate=0.25,
            pre_alert_to_trade_rate=0.50,
            avg_days_contraction_to_trade=8.5,
            avg_days_pre_alert_to_trade=5.0,
        )
        data = stats.to_dict()
        assert data["total_trade_alerts"] == 25
        assert data["avg_days_pre_alert_to_trade"] == 5.0


class TestAlertType:
    """Tests for AlertType enum."""

    def test_all_types_exist(self):
        """All three alert types exist."""
        assert AlertType.CONTRACTION.value == "contraction"
        assert AlertType.PRE_ALERT.value == "pre_alert"
        assert AlertType.TRADE.value == "trade"


class TestAlertState:
    """Tests for AlertState enum."""

    def test_all_states_exist(self):
        """All five alert states exist."""
        assert AlertState.PENDING.value == "pending"
        assert AlertState.NOTIFIED.value == "notified"
        assert AlertState.CONVERTED.value == "converted"
        assert AlertState.EXPIRED.value == "expired"
        assert AlertState.COMPLETED.value == "completed"


class TestEntryType:
    """Tests for EntryType enum."""

    def test_all_entry_types_exist(self):
        """All entry types exist."""
        assert EntryType.PIVOT_BREAKOUT.value == "pivot_breakout"
        assert EntryType.HANDLE_BREAK.value == "handle_break"
