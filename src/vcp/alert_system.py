"""
VCP Alert System - Main Orchestrator

The VCPAlertSystem coordinates all components of the alert system:
- VCPDetector for pattern detection
- AlertManager for alert lifecycle
- NotificationHub for multi-channel notifications
- Repository for persistence

This is the main entry point for using the VCP alert system.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any

import pandas as pd

from .models import (
    Alert,
    AlertChain,
    AlertType,
    ConversionStats,
    EntrySignal,
    VCPPattern,
)
from .detector import VCPDetector, DetectorConfig
from .alert_manager import AlertManager, AlertConfig
from .repository import SQLiteAlertRepository, InMemoryAlertRepository, AlertRepository
from .notifications import (
    NotificationHub,
    LogNotificationChannel,
    ConsoleNotificationChannel,
    CallbackNotificationChannel,
)


logger = logging.getLogger(__name__)


@dataclass
class SystemConfig:
    """Configuration for the VCP Alert System."""
    # Database
    db_path: str = "data/alerts.db"
    use_memory_db: bool = False  # Use in-memory DB for testing

    # Detection
    detector_config: Optional[DetectorConfig] = None

    # Alerts
    alert_config: Optional[AlertConfig] = None

    # Notifications
    enable_console_notifications: bool = True
    enable_log_notifications: bool = True

    # Processing
    auto_expire_alerts: bool = True


class VCPAlertSystem:
    """
    Main orchestrator for the VCP Alert System.

    Coordinates:
    - Pattern detection from price data
    - Alert creation and lifecycle management
    - Notification dispatch to multiple channels
    - Alert persistence and history

    Usage:
        system = VCPAlertSystem()
        system.process_symbol("AAPL", price_data)
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        """
        Initialize the VCP Alert System.

        Args:
            config: System configuration
        """
        self.config = config or SystemConfig()

        # Initialize repository
        if self.config.use_memory_db:
            self._repository: AlertRepository = InMemoryAlertRepository()
        else:
            self._repository = SQLiteAlertRepository(self.config.db_path)

        # Initialize detector
        detector_config = self.config.detector_config or DetectorConfig()
        self._detector = VCPDetector(detector_config)

        # Initialize alert manager
        alert_config = self.config.alert_config or AlertConfig()
        self._alert_manager = AlertManager(self._repository, alert_config)

        # Initialize notification hub
        self._notification_hub = NotificationHub()
        self._setup_default_channels()

        # Connect alert manager to notification hub
        self._alert_manager.subscribe(self._on_alert)

        logger.info("VCPAlertSystem initialized")

    def _setup_default_channels(self) -> None:
        """Set up default notification channels based on config."""
        if self.config.enable_log_notifications:
            self._notification_hub.register_channel(
                LogNotificationChannel(name="log")
            )

        if self.config.enable_console_notifications:
            self._notification_hub.register_channel(
                ConsoleNotificationChannel(name="console", use_colors=True)
            )

    def _on_alert(self, alert: Alert) -> None:
        """Handle new alerts from alert manager."""
        self._notification_hub.dispatch(alert)

    # === Public API ===

    def process_symbol(
        self,
        symbol: str,
        df: pd.DataFrame,
        check_entry: bool = True,
    ) -> List[Alert]:
        """
        Process price data for a symbol and emit alerts.

        This is the main method for processing market data.
        It detects VCP patterns and emits appropriate alerts.

        Args:
            symbol: Stock symbol
            df: DataFrame with OHLCV data
            check_entry: Whether to check for entry signals

        Returns:
            List of alerts created
        """
        # Auto-expire stale alerts if enabled
        if self.config.auto_expire_alerts:
            self._alert_manager.expire_stale_alerts()

        # Detect pattern
        pattern = self._detector.analyze(df, symbol)

        if pattern is None:
            logger.debug(f"No VCP pattern detected for {symbol}")
            return []

        # Get current price
        current_price = df["Close"].iloc[-1]

        # Check for entry signal if requested
        entry_signal = None
        if check_entry and pattern.is_valid:
            signals = self._detector.get_entry_signals(df, pattern)
            if signals:
                entry_signal = signals[0]  # Use first signal

        # Emit alerts
        alerts = self._alert_manager.check_and_emit(
            pattern=pattern,
            current_price=current_price,
            entry_signal=entry_signal,
        )

        if alerts:
            logger.info(f"Created {len(alerts)} alerts for {symbol}")

        return alerts

    def process_symbols(
        self,
        symbols: List[str],
        data_fetcher: callable,
        check_entry: bool = True,
    ) -> Dict[str, List[Alert]]:
        """
        Process multiple symbols using a data fetcher function.

        Args:
            symbols: List of stock symbols
            data_fetcher: Function that takes symbol and returns DataFrame
            check_entry: Whether to check for entry signals

        Returns:
            Dict mapping symbols to their alerts
        """
        results = {}

        for symbol in symbols:
            try:
                df = data_fetcher(symbol)
                if df is not None and not df.empty:
                    alerts = self.process_symbol(symbol, df, check_entry)
                    results[symbol] = alerts
                else:
                    logger.warning(f"No data for {symbol}")
                    results[symbol] = []
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                results[symbol] = []

        return results

    def analyze_pattern(
        self,
        symbol: str,
        df: pd.DataFrame,
    ) -> Optional[VCPPattern]:
        """
        Analyze price data for VCP pattern without emitting alerts.

        Useful for preview/research without triggering notifications.

        Args:
            symbol: Stock symbol
            df: DataFrame with OHLCV data

        Returns:
            VCPPattern if found, None otherwise
        """
        return self._detector.analyze(df, symbol)

    def get_entry_signals(
        self,
        df: pd.DataFrame,
        pattern: VCPPattern,
        risk_pct: float = 7.0,
        rr_ratio: float = 3.0,
    ) -> List[EntrySignal]:
        """
        Get entry signals for a VCP pattern.

        Args:
            df: DataFrame with OHLCV data
            pattern: Detected VCP pattern
            risk_pct: Maximum risk percentage
            rr_ratio: Reward to risk ratio

        Returns:
            List of entry signals
        """
        return self._detector.get_entry_signals(df, pattern, risk_pct, rr_ratio)

    # === Alert Management ===

    def get_active_alerts(
        self,
        symbol: Optional[str] = None,
    ) -> List[Alert]:
        """
        Get all active (pending/notified) alerts.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of active alerts
        """
        return self._alert_manager.get_active_alerts(symbol)

    def get_alert_by_id(self, alert_id: str) -> Optional[Alert]:
        """
        Get an alert by its ID.

        Args:
            alert_id: Alert ID

        Returns:
            Alert if found, None otherwise
        """
        return self._repository.get_by_id(alert_id)

    def get_alert_chain(self, alert_id: str) -> Optional[AlertChain]:
        """
        Get the complete alert chain for an alert.

        Args:
            alert_id: Any alert in the chain

        Returns:
            AlertChain if found, None otherwise
        """
        return self._alert_manager.get_alert_chain(alert_id)

    def get_alert_history(
        self,
        symbol: Optional[str] = None,
        alert_type: Optional[AlertType] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Alert]:
        """
        Get alert history with filters.

        Args:
            symbol: Filter by symbol
            alert_type: Filter by alert type
            start_date: Filter by start date
            end_date: Filter by end date

        Returns:
            List of matching alerts
        """
        return self._alert_manager.get_alert_history(
            symbol=symbol,
            alert_type=alert_type,
            start_date=start_date,
            end_date=end_date,
        )

    def mark_alert_notified(self, alert_id: str) -> bool:
        """
        Mark an alert as notified.

        Args:
            alert_id: Alert ID

        Returns:
            True if successful
        """
        return self._alert_manager.mark_notified(alert_id)

    def mark_trade_completed(self, alert_id: str) -> bool:
        """
        Mark a trade alert as completed.

        Args:
            alert_id: Trade alert ID

        Returns:
            True if successful
        """
        return self._alert_manager.mark_completed(alert_id)

    def expire_stale_alerts(self) -> int:
        """
        Manually expire stale alerts.

        Returns:
            Number of alerts expired
        """
        return self._alert_manager.expire_stale_alerts()

    # === Statistics ===

    def get_conversion_stats(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> ConversionStats:
        """
        Get alert conversion statistics for a period.

        Args:
            start_date: Start of period
            end_date: End of period

        Returns:
            ConversionStats for the period
        """
        return self._alert_manager.get_conversion_stats(start_date, end_date)

    def get_notification_stats(self) -> dict:
        """
        Get notification dispatch statistics.

        Returns:
            Statistics dictionary
        """
        return self._notification_hub.get_stats()

    # === Notification Management ===

    def add_notification_channel(
        self,
        channel,
    ) -> None:
        """
        Add a notification channel.

        Args:
            channel: Notification channel to add
        """
        self._notification_hub.register_channel(channel)

    def remove_notification_channel(self, name: str) -> bool:
        """
        Remove a notification channel by name.

        Args:
            name: Channel name

        Returns:
            True if channel was found and removed
        """
        return self._notification_hub.unregister_channel(name)

    def add_callback_handler(
        self,
        name: str,
        callback: callable,
        alert_types: Optional[List[AlertType]] = None,
    ) -> None:
        """
        Add a callback function as a notification handler.

        Convenience method for adding callback-based notifications.

        Args:
            name: Handler name
            callback: Function(alert, message) to call
            alert_types: Optional filter for alert types
        """
        channel = CallbackNotificationChannel(
            name=name,
            callback=callback,
            alert_types=alert_types,
        )
        self._notification_hub.register_channel(channel)

    # === Properties ===

    @property
    def detector(self) -> VCPDetector:
        """Get the VCP detector."""
        return self._detector

    @property
    def alert_manager(self) -> AlertManager:
        """Get the alert manager."""
        return self._alert_manager

    @property
    def notification_hub(self) -> NotificationHub:
        """Get the notification hub."""
        return self._notification_hub

    @property
    def repository(self) -> AlertRepository:
        """Get the alert repository."""
        return self._repository


def create_system(
    db_path: str = "data/alerts.db",
    min_score: float = 60.0,
    pre_alert_proximity: float = 3.0,
    enable_console: bool = True,
) -> VCPAlertSystem:
    """
    Factory function to create a configured VCPAlertSystem.

    Args:
        db_path: Path to SQLite database
        min_score: Minimum pattern score for alerts
        pre_alert_proximity: Distance % threshold for pre-alerts
        enable_console: Whether to enable console notifications

    Returns:
        Configured VCPAlertSystem
    """
    config = SystemConfig(
        db_path=db_path,
        alert_config=AlertConfig(
            min_score_contraction=min_score,
            pre_alert_proximity_pct=pre_alert_proximity,
        ),
        enable_console_notifications=enable_console,
    )
    return VCPAlertSystem(config)
