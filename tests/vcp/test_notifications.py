"""
Unit tests for VCP Alert System Notification Hub

Test cases from test plan (07-test-plan.md Section 10.2):
- NH-01: Register notification channel
- NH-02: Dispatch to appropriate channels
- NH-03: Format message correctly
- NH-04: Track notification statistics
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from src.vcp.models import Alert, AlertType, AlertState
from src.vcp.notifications import (
    NotificationHub,
    LogNotificationChannel,
    ConsoleNotificationChannel,
    WebhookNotificationChannel,
    CallbackNotificationChannel,
)


class TestLogNotificationChannel:
    """Tests for LogNotificationChannel."""

    def test_creation(self):
        """Channel can be created with default settings."""
        channel = LogNotificationChannel()

        assert channel.name == "log"
        assert len(channel.alert_types) == 3  # All types

    def test_custom_alert_types(self):
        """Channel can filter specific alert types."""
        channel = LogNotificationChannel(
            alert_types=[AlertType.TRADE, AlertType.PRE_ALERT]
        )

        assert AlertType.TRADE in channel.alert_types
        assert AlertType.PRE_ALERT in channel.alert_types
        assert AlertType.CONTRACTION not in channel.alert_types

    def test_format_message(self):
        """Message formatting includes all key fields."""
        channel = LogNotificationChannel()
        alert = Alert(
            symbol="AAPL",
            alert_type=AlertType.CONTRACTION,
            trigger_price=150.0,
            pivot_price=160.0,
            distance_to_pivot_pct=6.67,
            score=85.0,
            num_contractions=3,
        )

        message = channel.format_message(alert)

        assert "AAPL" in message
        assert "CONTRACTION" in message
        assert "150.00" in message
        assert "160.00" in message
        assert "6.7" in message
        assert "85" in message
        assert "3" in message

    def test_send_returns_true(self):
        """Send returns True on success."""
        channel = LogNotificationChannel()
        alert = Alert(
            symbol="NVDA",
            alert_type=AlertType.TRADE,
            trigger_price=450.0,
            pivot_price=445.0,
            distance_to_pivot_pct=-1.1,
            score=90.0,
            num_contractions=2,
        )

        result = channel.send(alert)
        assert result is True


class TestConsoleNotificationChannel:
    """Tests for ConsoleNotificationChannel."""

    def test_creation(self):
        """Channel can be created."""
        channel = ConsoleNotificationChannel()

        assert channel.name == "console"
        assert channel.use_colors is True

    def test_no_colors(self):
        """Channel can disable colors."""
        channel = ConsoleNotificationChannel(use_colors=False)

        assert channel.use_colors is False

    def test_format_message_with_colors(self):
        """Message includes color codes when enabled."""
        channel = ConsoleNotificationChannel(use_colors=True)
        alert = Alert(
            symbol="MSFT",
            alert_type=AlertType.PRE_ALERT,
            trigger_price=380.0,
            pivot_price=390.0,
            distance_to_pivot_pct=2.6,
            score=80.0,
            num_contractions=2,
        )

        message = channel.format_message(alert)

        # Should contain ANSI color codes
        assert "\033[" in message

    def test_format_message_without_colors(self):
        """Message excludes color codes when disabled."""
        channel = ConsoleNotificationChannel(use_colors=False)
        alert = Alert(
            symbol="MSFT",
            alert_type=AlertType.PRE_ALERT,
            trigger_price=380.0,
            pivot_price=390.0,
            distance_to_pivot_pct=2.6,
            score=80.0,
            num_contractions=2,
        )

        message = channel.format_message(alert)

        # Should NOT contain ANSI color codes
        assert "\033[" not in message

    def test_send_prints_to_stdout(self, capsys):
        """Send prints to stdout."""
        channel = ConsoleNotificationChannel(use_colors=False)
        alert = Alert(
            symbol="GOOGL",
            alert_type=AlertType.CONTRACTION,
            trigger_price=175.0,
            pivot_price=180.0,
            distance_to_pivot_pct=2.9,
            score=75.0,
            num_contractions=2,
        )

        result = channel.send(alert)
        captured = capsys.readouterr()

        assert result is True
        assert "GOOGL" in captured.out
        assert "CONTRACTION" in captured.out


class TestCallbackNotificationChannel:
    """Tests for CallbackNotificationChannel."""

    def test_callback_called(self):
        """Callback function is called on send."""
        received = []

        def my_callback(alert, message):
            received.append((alert, message))

        channel = CallbackNotificationChannel(
            name="test",
            callback=my_callback,
        )

        alert = Alert(
            symbol="TSLA",
            alert_type=AlertType.TRADE,
            trigger_price=200.0,
            pivot_price=195.0,
            distance_to_pivot_pct=-2.6,
            score=88.0,
            num_contractions=3,
        )

        result = channel.send(alert)

        assert result is True
        assert len(received) == 1
        assert received[0][0].symbol == "TSLA"
        assert "TSLA" in received[0][1]

    def test_callback_error_returns_false(self):
        """Returns False if callback raises exception."""
        def bad_callback(alert, message):
            raise Exception("Callback error")

        channel = CallbackNotificationChannel(
            name="bad",
            callback=bad_callback,
        )

        alert = Alert(
            symbol="AAPL",
            alert_type=AlertType.CONTRACTION,
            trigger_price=150.0,
            pivot_price=160.0,
            distance_to_pivot_pct=6.7,
            score=75.0,
            num_contractions=2,
        )

        result = channel.send(alert)
        assert result is False


class TestNotificationHub:
    """Tests for NotificationHub."""

    @pytest.fixture
    def hub(self):
        """Create a notification hub."""
        return NotificationHub()

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
        )

    def test_register_channel(self, hub):
        """NH-01: Register notification channel."""
        channel = LogNotificationChannel(name="test_log")
        hub.register_channel(channel)

        assert "test_log" in hub.channel_names
        assert len(hub.channels) == 1

    def test_register_multiple_channels(self, hub):
        """Multiple channels can be registered."""
        hub.register_channel(LogNotificationChannel(name="log1"))
        hub.register_channel(LogNotificationChannel(name="log2"))
        hub.register_channel(ConsoleNotificationChannel(name="console"))

        assert len(hub.channels) == 3
        assert "log1" in hub.channel_names
        assert "log2" in hub.channel_names
        assert "console" in hub.channel_names

    def test_unregister_channel(self, hub):
        """Channels can be unregistered."""
        hub.register_channel(LogNotificationChannel(name="to_remove"))
        hub.register_channel(LogNotificationChannel(name="to_keep"))

        result = hub.unregister_channel("to_remove")

        assert result is True
        assert "to_remove" not in hub.channel_names
        assert "to_keep" in hub.channel_names

    def test_unregister_nonexistent(self, hub):
        """Unregister returns False for nonexistent channel."""
        result = hub.unregister_channel("nonexistent")
        assert result is False

    def test_dispatch_to_channels(self, hub, sample_alert):
        """NH-02: Dispatch to appropriate channels."""
        received = []

        def callback(alert, message):
            received.append(alert.symbol)

        hub.register_channel(CallbackNotificationChannel(
            name="callback",
            callback=callback,
        ))

        successful = hub.dispatch(sample_alert)

        assert successful == 1
        assert "NVDA" in received

    def test_dispatch_filters_by_type(self, hub, sample_alert):
        """Dispatch only sends to channels that handle the alert type."""
        received = []

        def callback(alert, message):
            received.append(alert.symbol)

        # This channel only handles TRADE alerts
        hub.register_channel(CallbackNotificationChannel(
            name="trade_only",
            callback=callback,
            alert_types=[AlertType.TRADE],
        ))

        # sample_alert is CONTRACTION type
        successful = hub.dispatch(sample_alert)

        assert successful == 0
        assert len(received) == 0

    def test_dispatch_to_matching_type(self, hub):
        """Dispatch sends to channels that match the alert type."""
        received = []

        def callback(alert, message):
            received.append(alert.symbol)

        hub.register_channel(CallbackNotificationChannel(
            name="trade_only",
            callback=callback,
            alert_types=[AlertType.TRADE],
        ))

        trade_alert = Alert(
            symbol="AAPL",
            alert_type=AlertType.TRADE,
            trigger_price=175.0,
            pivot_price=170.0,
            distance_to_pivot_pct=-2.9,
            score=90.0,
            num_contractions=2,
        )

        successful = hub.dispatch(trade_alert)

        assert successful == 1
        assert "AAPL" in received

    def test_format_message(self, hub, sample_alert):
        """NH-03: Format message correctly."""
        channel = LogNotificationChannel()
        message = channel.format_message(sample_alert)

        assert "NVDA" in message
        assert "450.00" in message
        assert "480.00" in message

    def test_track_statistics(self, hub, sample_alert):
        """NH-04: Track notification statistics."""
        received = []

        hub.register_channel(CallbackNotificationChannel(
            name="callback",
            callback=lambda a, m: received.append(a),
        ))

        hub.dispatch(sample_alert)
        hub.dispatch(sample_alert)

        stats = hub.get_stats()

        assert stats["dispatched"] == 2
        assert stats["successful"] == 2
        assert stats["by_type"]["contraction"] == 2
        assert stats["by_channel"]["callback"]["sent"] == 2

    def test_track_failures(self, hub):
        """Statistics track failed notifications."""
        def bad_callback(alert, message):
            raise Exception("Fail")

        hub.register_channel(CallbackNotificationChannel(
            name="bad",
            callback=bad_callback,
        ))

        alert = Alert(
            symbol="TEST",
            alert_type=AlertType.CONTRACTION,
            trigger_price=100.0,
            pivot_price=105.0,
            distance_to_pivot_pct=5.0,
            score=70.0,
            num_contractions=2,
        )

        hub.dispatch(alert)

        stats = hub.get_stats()
        assert stats["failed"] == 1
        assert stats["by_channel"]["bad"]["failed"] == 1

    def test_reset_stats(self, hub, sample_alert):
        """Statistics can be reset."""
        hub.register_channel(CallbackNotificationChannel(
            name="callback",
            callback=lambda a, m: None,
        ))

        hub.dispatch(sample_alert)
        hub.reset_stats()

        stats = hub.get_stats()
        assert stats["dispatched"] == 0
        assert stats["successful"] == 0

    def test_dispatch_batch(self, hub):
        """Dispatch batch sends multiple alerts."""
        received = []

        hub.register_channel(CallbackNotificationChannel(
            name="callback",
            callback=lambda a, m: received.append(a.symbol),
        ))

        alerts = [
            Alert(
                symbol="AAPL",
                alert_type=AlertType.CONTRACTION,
                trigger_price=150.0,
                pivot_price=160.0,
                distance_to_pivot_pct=6.7,
                score=80.0,
                num_contractions=2,
            ),
            Alert(
                symbol="NVDA",
                alert_type=AlertType.PRE_ALERT,
                trigger_price=475.0,
                pivot_price=480.0,
                distance_to_pivot_pct=1.1,
                score=85.0,
                num_contractions=3,
            ),
            Alert(
                symbol="MSFT",
                alert_type=AlertType.TRADE,
                trigger_price=390.0,
                pivot_price=385.0,
                distance_to_pivot_pct=-1.3,
                score=90.0,
                num_contractions=2,
            ),
        ]

        successful = hub.dispatch_batch(alerts)

        assert successful == 3
        assert "AAPL" in received
        assert "NVDA" in received
        assert "MSFT" in received


class TestWebhookNotificationChannel:
    """Tests for WebhookNotificationChannel."""

    def test_creation(self):
        """Channel can be created with webhook URL."""
        channel = WebhookNotificationChannel(
            name="slack",
            webhook_url="https://hooks.slack.com/test",
        )

        assert channel.name == "slack"
        assert channel.webhook_url == "https://hooks.slack.com/test"

    def test_build_payload(self):
        """Payload contains all required fields."""
        channel = WebhookNotificationChannel(
            name="test",
            webhook_url="https://example.com/webhook",
        )

        alert = Alert(
            symbol="TSLA",
            alert_type=AlertType.TRADE,
            trigger_price=200.0,
            pivot_price=195.0,
            distance_to_pivot_pct=-2.6,
            score=88.0,
            num_contractions=3,
        )

        payload = channel._build_payload(alert)

        assert payload["symbol"] == "TSLA"
        assert payload["alert_type"] == "trade"
        assert payload["trigger_price"] == 200.0
        assert payload["pivot_price"] == 195.0
        assert "message" in payload

    @patch("urllib.request.urlopen")
    def test_send_success(self, mock_urlopen):
        """Send returns True on successful webhook."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        channel = WebhookNotificationChannel(
            name="test",
            webhook_url="https://example.com/webhook",
        )

        alert = Alert(
            symbol="TEST",
            alert_type=AlertType.CONTRACTION,
            trigger_price=100.0,
            pivot_price=105.0,
            distance_to_pivot_pct=5.0,
            score=75.0,
            num_contractions=2,
        )

        result = channel.send(alert)

        assert result is True
        mock_urlopen.assert_called_once()
