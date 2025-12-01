"""
VCP Alert System - Notification Hub

Multi-channel notification system for VCP alerts:
- Protocol-based channel abstraction
- Support for Telegram, Discord, Webhook, and Log channels
- Asynchronous dispatch capability
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Protocol

from .models import Alert, AlertType


logger = logging.getLogger(__name__)


class NotificationChannel(Protocol):
    """Protocol for notification channels."""

    @property
    def name(self) -> str:
        """Channel name."""
        ...

    @property
    def alert_types(self) -> List[AlertType]:
        """Alert types this channel handles."""
        ...

    def send(self, alert: Alert) -> bool:
        """Send an alert notification. Returns True if successful."""
        ...

    def format_message(self, alert: Alert) -> str:
        """Format alert into a message string."""
        ...


class BaseNotificationChannel(ABC):
    """Base class for notification channels."""

    def __init__(
        self,
        name: str,
        alert_types: Optional[List[AlertType]] = None,
    ):
        """
        Initialize the channel.

        Args:
            name: Channel name
            alert_types: Alert types to handle (None = all types)
        """
        self._name = name
        self._alert_types = alert_types or list(AlertType)

    @property
    def name(self) -> str:
        """Channel name."""
        return self._name

    @property
    def alert_types(self) -> List[AlertType]:
        """Alert types this channel handles."""
        return self._alert_types

    def format_message(self, alert: Alert) -> str:
        """
        Format alert into a message string.

        Default implementation creates a structured text message.
        Override for channel-specific formatting.
        """
        type_emoji = {
            AlertType.CONTRACTION: "📊",
            AlertType.PRE_ALERT: "⚠️",
            AlertType.TRADE: "🎯",
        }

        emoji = type_emoji.get(alert.alert_type, "📌")
        type_name = alert.alert_type.value.replace("_", " ").title()

        lines = [
            f"{emoji} **VCP {type_name}** - {alert.symbol}",
            "",
            f"**Price:** ${alert.trigger_price:.2f}",
            f"**Pivot:** ${alert.pivot_price:.2f}",
            f"**Distance to Pivot:** {alert.distance_to_pivot_pct:.1f}%",
            f"**Score:** {alert.score:.0f}",
            f"**Contractions:** {alert.num_contractions}",
            "",
            f"*{alert.created_at.strftime('%Y-%m-%d %H:%M:%S')}*",
        ]

        return "\n".join(lines)

    @abstractmethod
    def send(self, alert: Alert) -> bool:
        """Send an alert notification."""
        ...


class LogNotificationChannel(BaseNotificationChannel):
    """
    Notification channel that logs alerts.

    Useful for development, testing, and audit trails.
    """

    def __init__(
        self,
        name: str = "log",
        alert_types: Optional[List[AlertType]] = None,
        log_level: int = logging.INFO,
    ):
        """
        Initialize the log channel.

        Args:
            name: Channel name
            alert_types: Alert types to handle
            log_level: Logging level to use
        """
        super().__init__(name, alert_types)
        self.log_level = log_level
        self.logger = logging.getLogger(f"vcp.notifications.{name}")

    def send(self, alert: Alert) -> bool:
        """Log the alert notification."""
        try:
            message = self.format_message(alert)
            self.logger.log(self.log_level, f"Alert notification:\n{message}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to log alert: {e}")
            return False

    def format_message(self, alert: Alert) -> str:
        """Format for logging (plain text)."""
        lines = [
            f"[{alert.alert_type.value.upper()}] {alert.symbol}",
            f"  Price: ${alert.trigger_price:.2f}",
            f"  Pivot: ${alert.pivot_price:.2f}",
            f"  Distance: {alert.distance_to_pivot_pct:.1f}%",
            f"  Score: {alert.score:.0f}",
            f"  Contractions: {alert.num_contractions}",
            f"  Time: {alert.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        return "\n".join(lines)


class ConsoleNotificationChannel(BaseNotificationChannel):
    """
    Notification channel that prints alerts to console.

    Useful for development and debugging.
    """

    def __init__(
        self,
        name: str = "console",
        alert_types: Optional[List[AlertType]] = None,
        use_colors: bool = True,
    ):
        """
        Initialize the console channel.

        Args:
            name: Channel name
            alert_types: Alert types to handle
            use_colors: Whether to use ANSI colors
        """
        super().__init__(name, alert_types)
        self.use_colors = use_colors

    def send(self, alert: Alert) -> bool:
        """Print the alert to console."""
        try:
            message = self.format_message(alert)
            print(message)
            return True
        except Exception as e:
            logger.error(f"Failed to print alert: {e}")
            return False

    def format_message(self, alert: Alert) -> str:
        """Format for console with optional colors."""
        colors = {
            AlertType.CONTRACTION: "\033[94m",  # Blue
            AlertType.PRE_ALERT: "\033[93m",     # Yellow
            AlertType.TRADE: "\033[92m",         # Green
        }
        reset = "\033[0m"

        color = colors.get(alert.alert_type, "") if self.use_colors else ""
        end = reset if self.use_colors else ""

        type_name = alert.alert_type.value.replace("_", " ").upper()

        lines = [
            f"{color}{'='*50}{end}",
            f"{color}VCP {type_name}: {alert.symbol}{end}",
            f"{'='*50}",
            f"  Price: ${alert.trigger_price:.2f}",
            f"  Pivot: ${alert.pivot_price:.2f}",
            f"  Distance to Pivot: {alert.distance_to_pivot_pct:.1f}%",
            f"  Score: {alert.score:.0f}",
            f"  Contractions: {alert.num_contractions}",
            f"  Time: {alert.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"{'='*50}",
        ]
        return "\n".join(lines)


class WebhookNotificationChannel(BaseNotificationChannel):
    """
    Notification channel that sends alerts to a webhook URL.

    Supports generic webhook endpoints and can be customized
    for specific services (Slack, custom APIs, etc).
    """

    def __init__(
        self,
        name: str,
        webhook_url: str,
        alert_types: Optional[List[AlertType]] = None,
        headers: Optional[dict] = None,
        timeout: float = 10.0,
    ):
        """
        Initialize the webhook channel.

        Args:
            name: Channel name
            webhook_url: URL to POST alerts to
            alert_types: Alert types to handle
            headers: Optional HTTP headers
            timeout: Request timeout in seconds
        """
        super().__init__(name, alert_types)
        self.webhook_url = webhook_url
        self.headers = headers or {"Content-Type": "application/json"}
        self.timeout = timeout

    def send(self, alert: Alert) -> bool:
        """Send alert to webhook."""
        try:
            import urllib.request
            import urllib.error

            payload = self._build_payload(alert)
            data = json.dumps(payload).encode("utf-8")

            request = urllib.request.Request(
                self.webhook_url,
                data=data,
                headers=self.headers,
                method="POST",
            )

            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                if response.status == 200:
                    logger.info(f"Webhook sent successfully to {self.name}")
                    return True
                else:
                    logger.warning(f"Webhook returned status {response.status}")
                    return False

        except urllib.error.URLError as e:
            logger.error(f"Failed to send webhook: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending webhook: {e}")
            return False

    def _build_payload(self, alert: Alert) -> dict:
        """Build JSON payload for webhook."""
        return {
            "alert_id": alert.id,
            "symbol": alert.symbol,
            "alert_type": alert.alert_type.value,
            "state": alert.state.value,
            "trigger_price": alert.trigger_price,
            "pivot_price": alert.pivot_price,
            "distance_to_pivot_pct": alert.distance_to_pivot_pct,
            "score": alert.score,
            "num_contractions": alert.num_contractions,
            "created_at": alert.created_at.isoformat(),
            "message": self.format_message(alert),
        }


class CallbackNotificationChannel(BaseNotificationChannel):
    """
    Notification channel that calls a callback function.

    Useful for integrating with existing systems or for testing.
    """

    def __init__(
        self,
        name: str,
        callback: callable,
        alert_types: Optional[List[AlertType]] = None,
    ):
        """
        Initialize the callback channel.

        Args:
            name: Channel name
            callback: Function to call with (alert, message)
            alert_types: Alert types to handle
        """
        super().__init__(name, alert_types)
        self.callback = callback

    def send(self, alert: Alert) -> bool:
        """Call the callback with the alert."""
        try:
            message = self.format_message(alert)
            self.callback(alert, message)
            return True
        except Exception as e:
            logger.error(f"Callback failed: {e}")
            return False


class NotificationHub:
    """
    Central hub for dispatching notifications to multiple channels.

    Features:
    - Register multiple notification channels
    - Dispatch alerts to appropriate channels based on alert type
    - Track notification statistics
    """

    def __init__(self):
        """Initialize the notification hub."""
        self._channels: List[BaseNotificationChannel] = []
        self._stats = {
            "dispatched": 0,
            "successful": 0,
            "failed": 0,
            "by_type": {t.value: 0 for t in AlertType},
            "by_channel": {},
        }

    def register_channel(self, channel: BaseNotificationChannel) -> None:
        """
        Register a notification channel.

        Args:
            channel: Channel to register
        """
        self._channels.append(channel)
        self._stats["by_channel"][channel.name] = {"sent": 0, "failed": 0}
        logger.info(f"Registered notification channel: {channel.name}")

    def unregister_channel(self, name: str) -> bool:
        """
        Unregister a notification channel by name.

        Args:
            name: Name of channel to unregister

        Returns:
            True if channel was found and removed
        """
        for i, channel in enumerate(self._channels):
            if channel.name == name:
                self._channels.pop(i)
                logger.info(f"Unregistered notification channel: {name}")
                return True
        return False

    def dispatch(self, alert: Alert) -> int:
        """
        Dispatch an alert to all appropriate channels.

        Args:
            alert: Alert to dispatch

        Returns:
            Number of successful notifications
        """
        self._stats["dispatched"] += 1
        self._stats["by_type"][alert.alert_type.value] += 1

        successful = 0

        for channel in self._channels:
            # Check if channel handles this alert type
            if alert.alert_type not in channel.alert_types:
                continue

            try:
                if channel.send(alert):
                    successful += 1
                    self._stats["successful"] += 1
                    self._stats["by_channel"][channel.name]["sent"] += 1
                else:
                    self._stats["failed"] += 1
                    self._stats["by_channel"][channel.name]["failed"] += 1
            except Exception as e:
                logger.error(f"Error dispatching to {channel.name}: {e}")
                self._stats["failed"] += 1
                self._stats["by_channel"][channel.name]["failed"] += 1

        return successful

    def dispatch_batch(self, alerts: List[Alert]) -> int:
        """
        Dispatch multiple alerts.

        Args:
            alerts: List of alerts to dispatch

        Returns:
            Total number of successful notifications
        """
        total_successful = 0
        for alert in alerts:
            total_successful += self.dispatch(alert)
        return total_successful

    def get_stats(self) -> dict:
        """Get notification statistics."""
        return self._stats.copy()

    def reset_stats(self) -> None:
        """Reset notification statistics."""
        self._stats = {
            "dispatched": 0,
            "successful": 0,
            "failed": 0,
            "by_type": {t.value: 0 for t in AlertType},
            "by_channel": {c.name: {"sent": 0, "failed": 0} for c in self._channels},
        }

    @property
    def channels(self) -> List[BaseNotificationChannel]:
        """Get registered channels."""
        return self._channels.copy()

    @property
    def channel_names(self) -> List[str]:
        """Get names of registered channels."""
        return [c.name for c in self._channels]
