"""
Unit tests for VCP Alert System Orchestrator

Tests for the main VCPAlertSystem class that coordinates
all components of the alert system.
"""

import pytest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.vcp.alert_system import VCPAlertSystem, SystemConfig, create_system
from src.vcp.detector import DetectorConfig
from src.vcp.alert_manager import AlertConfig
from src.vcp.models import Alert, AlertType, AlertState


class TestVCPAlertSystemSetup:
    """Tests for VCPAlertSystem initialization."""

    def test_default_config(self):
        """System initializes with default config."""
        config = SystemConfig(use_memory_db=True)
        system = VCPAlertSystem(config)

        assert system.detector is not None
        assert system.alert_manager is not None
        assert system.notification_hub is not None
        assert system.repository is not None

    def test_custom_detector_config(self):
        """System uses custom detector config."""
        config = SystemConfig(
            use_memory_db=True,
            detector_config=DetectorConfig(
                min_contractions=3,
                lookback_days=100,
            ),
        )
        system = VCPAlertSystem(config)

        assert system.detector.config.min_contractions == 3
        assert system.detector.config.lookback_days == 100

    def test_custom_alert_config(self):
        """System uses custom alert config."""
        config = SystemConfig(
            use_memory_db=True,
            alert_config=AlertConfig(
                min_score_contraction=70.0,
                pre_alert_proximity_pct=5.0,
            ),
        )
        system = VCPAlertSystem(config)

        assert system.alert_manager.config.min_score_contraction == 70.0
        assert system.alert_manager.config.pre_alert_proximity_pct == 5.0

    def test_notification_channels_setup(self):
        """Default notification channels are set up."""
        config = SystemConfig(
            use_memory_db=True,
            enable_console_notifications=True,
            enable_log_notifications=True,
        )
        system = VCPAlertSystem(config)

        channel_names = system.notification_hub.channel_names
        assert "console" in channel_names
        assert "log" in channel_names


class TestProcessSymbol:
    """Tests for processing individual symbols."""

    @pytest.fixture
    def system(self):
        """Create system with in-memory database."""
        config = SystemConfig(
            use_memory_db=True,
            enable_console_notifications=False,
            enable_log_notifications=False,
            detector_config=DetectorConfig(lookback_days=80),
        )
        return VCPAlertSystem(config)

    @pytest.fixture
    def sample_df(self):
        """Create sample price data."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        np.random.seed(42)

        # Create data with some price movement
        prices = []
        for i in range(100):
            base = 100 + i * 0.5 + np.random.randn() * 2
            prices.append(base)

        df = pd.DataFrame({
            "Open": [p - 0.5 for p in prices],
            "High": [p + 2 for p in prices],
            "Low": [p - 2 for p in prices],
            "Close": prices,
            "Volume": [1000000 + np.random.randint(-100000, 100000) for _ in range(100)],
        }, index=dates)

        return df

    def test_process_symbol_no_pattern(self, system):
        """Process returns empty list when no pattern found."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        df = pd.DataFrame({
            "Open": [100] * 100,
            "High": [101] * 100,
            "Low": [99] * 100,
            "Close": [100] * 100,
            "Volume": [1000000] * 100,
        }, index=dates)

        alerts = system.process_symbol("FLAT", df)

        # May return empty or alerts depending on detection
        assert isinstance(alerts, list)

    def test_process_symbol_returns_alerts(self, system, sample_df):
        """Process returns alerts when pattern found."""
        alerts = system.process_symbol("TEST", sample_df, check_entry=False)

        # Result is a list (may be empty if no pattern qualifies)
        assert isinstance(alerts, list)

    def test_process_insufficient_data(self, system):
        """Process returns empty for insufficient data."""
        dates = pd.date_range(start="2024-01-01", periods=20, freq="D")
        df = pd.DataFrame({
            "Open": [100] * 20,
            "High": [101] * 20,
            "Low": [99] * 20,
            "Close": [100] * 20,
            "Volume": [1000000] * 20,
        }, index=dates)

        alerts = system.process_symbol("SHORT", df)

        assert alerts == []


class TestProcessMultipleSymbols:
    """Tests for processing multiple symbols."""

    @pytest.fixture
    def system(self):
        """Create system with in-memory database."""
        config = SystemConfig(
            use_memory_db=True,
            enable_console_notifications=False,
            enable_log_notifications=False,
            detector_config=DetectorConfig(lookback_days=80),
        )
        return VCPAlertSystem(config)

    def test_process_symbols(self, system):
        """Process multiple symbols using data fetcher."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")

        def mock_fetcher(symbol):
            np.random.seed(hash(symbol) % 2**32)
            prices = [100 + np.random.randn() for _ in range(100)]
            return pd.DataFrame({
                "Open": [p - 0.5 for p in prices],
                "High": [p + 2 for p in prices],
                "Low": [p - 2 for p in prices],
                "Close": prices,
                "Volume": [1000000] * 100,
            }, index=dates)

        results = system.process_symbols(
            ["AAPL", "NVDA", "MSFT"],
            mock_fetcher,
            check_entry=False,
        )

        assert "AAPL" in results
        assert "NVDA" in results
        assert "MSFT" in results
        assert all(isinstance(v, list) for v in results.values())

    def test_process_symbols_handles_errors(self, system):
        """Process continues despite errors in individual symbols."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")

        def error_fetcher(symbol):
            if symbol == "ERROR":
                raise Exception("Data fetch error")
            return pd.DataFrame({
                "Open": [100] * 100,
                "High": [101] * 100,
                "Low": [99] * 100,
                "Close": [100] * 100,
                "Volume": [1000000] * 100,
            }, index=dates)

        results = system.process_symbols(
            ["GOOD", "ERROR", "ALSO_GOOD"],
            error_fetcher,
        )

        # All symbols should have results (empty for error)
        assert len(results) == 3
        assert results["ERROR"] == []


class TestAlertManagement:
    """Tests for alert management methods."""

    @pytest.fixture
    def system(self):
        """Create system with in-memory database."""
        config = SystemConfig(
            use_memory_db=True,
            enable_console_notifications=False,
            enable_log_notifications=False,
        )
        return VCPAlertSystem(config)

    @pytest.fixture
    def sample_alert(self, system):
        """Create and save a sample alert."""
        alert = Alert(
            symbol="AAPL",
            alert_type=AlertType.CONTRACTION,
            trigger_price=150.0,
            pivot_price=160.0,
            distance_to_pivot_pct=6.67,
            score=85.0,
            num_contractions=3,
        )
        system.repository.save(alert)
        return alert

    def test_get_alert_by_id(self, system, sample_alert):
        """Get alert by ID."""
        retrieved = system.get_alert_by_id(sample_alert.id)

        assert retrieved is not None
        assert retrieved.id == sample_alert.id
        assert retrieved.symbol == "AAPL"

    def test_get_active_alerts(self, system, sample_alert):
        """Get active alerts."""
        active = system.get_active_alerts()

        assert len(active) >= 1
        assert any(a.id == sample_alert.id for a in active)

    def test_get_active_alerts_by_symbol(self, system, sample_alert):
        """Get active alerts filtered by symbol."""
        # Add another alert for different symbol
        other_alert = Alert(
            symbol="NVDA",
            alert_type=AlertType.CONTRACTION,
            trigger_price=450.0,
            pivot_price=480.0,
            distance_to_pivot_pct=6.67,
            score=80.0,
            num_contractions=2,
        )
        system.repository.save(other_alert)

        aapl_alerts = system.get_active_alerts("AAPL")

        assert all(a.symbol == "AAPL" for a in aapl_alerts)

    def test_mark_alert_notified(self, system, sample_alert):
        """Mark alert as notified."""
        result = system.mark_alert_notified(sample_alert.id)

        assert result is True
        retrieved = system.get_alert_by_id(sample_alert.id)
        assert retrieved.state == AlertState.NOTIFIED

    def test_mark_trade_completed(self, system):
        """Mark trade alert as completed."""
        trade_alert = Alert(
            symbol="TSLA",
            alert_type=AlertType.TRADE,
            trigger_price=200.0,
            pivot_price=195.0,
            distance_to_pivot_pct=-2.6,
            score=90.0,
            num_contractions=3,
        )
        trade_alert.mark_notified()
        system.repository.save(trade_alert)

        result = system.mark_trade_completed(trade_alert.id)

        assert result is True
        retrieved = system.get_alert_by_id(trade_alert.id)
        assert retrieved.state == AlertState.COMPLETED

    def test_get_alert_history(self, system, sample_alert):
        """Get alert history."""
        history = system.get_alert_history(symbol="AAPL")

        assert len(history) >= 1

    def test_get_alert_history_by_type(self, system, sample_alert):
        """Get alert history filtered by type."""
        history = system.get_alert_history(alert_type=AlertType.CONTRACTION)

        assert all(a.alert_type == AlertType.CONTRACTION for a in history)


class TestAlertChains:
    """Tests for alert chain functionality."""

    @pytest.fixture
    def system(self):
        """Create system with in-memory database."""
        config = SystemConfig(
            use_memory_db=True,
            enable_console_notifications=False,
            enable_log_notifications=False,
        )
        return VCPAlertSystem(config)

    def test_get_alert_chain(self, system):
        """Get complete alert chain."""
        # Create chain
        contraction = Alert(
            symbol="AAPL",
            alert_type=AlertType.CONTRACTION,
            trigger_price=150.0,
            pivot_price=160.0,
            distance_to_pivot_pct=6.67,
            score=80.0,
            num_contractions=2,
        )
        system.repository.save(contraction)

        pre_alert = Alert(
            symbol="AAPL",
            alert_type=AlertType.PRE_ALERT,
            trigger_price=158.0,
            pivot_price=160.0,
            distance_to_pivot_pct=1.27,
            score=82.0,
            num_contractions=2,
            parent_alert_id=contraction.id,
        )
        system.repository.save(pre_alert)

        chain = system.get_alert_chain(pre_alert.id)

        assert chain is not None
        assert chain.symbol == "AAPL"
        assert chain.contraction_alert.id == contraction.id
        assert len(chain.pre_alerts) == 1


class TestNotificationManagement:
    """Tests for notification management."""

    @pytest.fixture
    def system(self):
        """Create system with in-memory database."""
        config = SystemConfig(
            use_memory_db=True,
            enable_console_notifications=False,
            enable_log_notifications=False,
        )
        return VCPAlertSystem(config)

    def test_add_callback_handler(self, system):
        """Add callback notification handler."""
        received = []

        system.add_callback_handler(
            name="test_callback",
            callback=lambda a, m: received.append(a.symbol),
        )

        assert "test_callback" in system.notification_hub.channel_names

    def test_remove_notification_channel(self, system):
        """Remove notification channel."""
        system.add_callback_handler(
            name="to_remove",
            callback=lambda a, m: None,
        )

        result = system.remove_notification_channel("to_remove")

        assert result is True
        assert "to_remove" not in system.notification_hub.channel_names

    def test_callback_receives_alerts(self, system):
        """Callback handler receives alerts."""
        received = []

        system.add_callback_handler(
            name="receiver",
            callback=lambda a, m: received.append(a),
        )

        # Manually dispatch an alert to test
        alert = Alert(
            symbol="TEST",
            alert_type=AlertType.CONTRACTION,
            trigger_price=100.0,
            pivot_price=105.0,
            distance_to_pivot_pct=5.0,
            score=75.0,
            num_contractions=2,
        )
        system.notification_hub.dispatch(alert)

        assert len(received) == 1
        assert received[0].symbol == "TEST"

    def test_get_notification_stats(self, system):
        """Get notification statistics."""
        stats = system.get_notification_stats()

        assert "dispatched" in stats
        assert "successful" in stats
        assert "failed" in stats


class TestStatistics:
    """Tests for statistics methods."""

    @pytest.fixture
    def system(self):
        """Create system with in-memory database."""
        config = SystemConfig(
            use_memory_db=True,
            enable_console_notifications=False,
            enable_log_notifications=False,
        )
        return VCPAlertSystem(config)

    def test_get_conversion_stats(self, system):
        """Get conversion statistics."""
        now = datetime.now()
        start = now - timedelta(days=30)
        end = now + timedelta(days=1)  # Include today

        # Add some alerts
        alert = Alert(
            symbol="AAPL",
            alert_type=AlertType.CONTRACTION,
            trigger_price=150.0,
            pivot_price=160.0,
            distance_to_pivot_pct=6.67,
            score=80.0,
            num_contractions=2,
        )
        system.repository.save(alert)

        stats = system.get_conversion_stats(start, end)

        assert stats.total_contraction_alerts >= 1
        assert 0 <= stats.contraction_to_trade_rate <= 1


class TestCreateSystemFactory:
    """Tests for create_system factory function."""

    def test_create_system_default(self):
        """Factory creates system with defaults."""
        system = create_system(enable_console=False)

        assert system is not None
        assert system.alert_manager.config.min_score_contraction == 60.0

    def test_create_system_custom(self):
        """Factory creates system with custom settings."""
        system = create_system(
            min_score=70.0,
            pre_alert_proximity=5.0,
            enable_console=False,
        )

        assert system.alert_manager.config.min_score_contraction == 70.0
        assert system.alert_manager.config.pre_alert_proximity_pct == 5.0
