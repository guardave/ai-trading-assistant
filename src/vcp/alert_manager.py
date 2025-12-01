"""
VCP Alert System - Alert Manager

Manages the alert lifecycle including:
- Alert creation and deduplication
- State transitions
- Alert chain tracking
- Subscriber notifications
- Expiration handling
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, List, Optional, Protocol

from .models import (
    Alert,
    AlertChain,
    AlertState,
    AlertType,
    ConversionStats,
    VCPPattern,
    EntrySignal,
)
from .repository import AlertRepository


@dataclass
class AlertConfig:
    """Configuration for alert manager behavior."""
    # Deduplication settings
    dedup_window_days: int = 7  # Days to check for duplicate alerts

    # Pre-alert settings
    pre_alert_proximity_pct: float = 3.0  # Trigger pre-alert within X% of pivot

    # Expiration settings
    contraction_ttl_days: int = 30  # Days before contraction alert expires
    pre_alert_ttl_days: int = 14    # Days before pre-alert expires
    trade_ttl_days: int = 7         # Days before trade alert expires

    # Score thresholds
    min_score_contraction: float = 60.0
    min_contractions: int = 2


class AlertSubscriber(Protocol):
    """Protocol for alert subscribers."""

    def on_alert(self, alert: Alert) -> None:
        """Called when a new alert is created."""
        ...


class AlertManager:
    """
    Manages the VCP alert lifecycle.

    Responsibilities:
    - Create alerts based on pattern detection
    - Deduplicate alerts to avoid spam
    - Track state transitions
    - Notify subscribers of new alerts
    - Handle alert expiration
    - Build alert chains
    """

    def __init__(
        self,
        repository: AlertRepository,
        config: Optional[AlertConfig] = None,
    ):
        """
        Initialize the alert manager.

        Args:
            repository: Storage backend for alerts
            config: Configuration settings
        """
        self.repository = repository
        self.config = config or AlertConfig()
        self._subscribers: List[Callable[[Alert], None]] = []

    def subscribe(self, handler: Callable[[Alert], None]) -> None:
        """
        Subscribe to alert notifications.

        Args:
            handler: Callback function to receive alerts
        """
        self._subscribers.append(handler)

    def unsubscribe(self, handler: Callable[[Alert], None]) -> None:
        """
        Unsubscribe from alert notifications.

        Args:
            handler: Previously subscribed callback
        """
        if handler in self._subscribers:
            self._subscribers.remove(handler)

    def _notify_subscribers(self, alert: Alert) -> None:
        """Notify all subscribers of a new alert."""
        for subscriber in self._subscribers:
            try:
                subscriber(alert)
            except Exception:
                # Log error but don't fail on subscriber errors
                pass

    def _is_duplicate(self, symbol: str, alert_type: AlertType) -> bool:
        """
        Check if an alert would be a duplicate.

        An alert is considered duplicate if there's already an active
        (PENDING or NOTIFIED) alert of the same type for the same symbol.
        """
        existing = self.repository.query(
            symbol=symbol,
            alert_type=alert_type,
        )
        # Check for active alerts within dedup window
        cutoff = datetime.now() - timedelta(days=self.config.dedup_window_days)
        for alert in existing:
            if alert.state in (AlertState.PENDING, AlertState.NOTIFIED):
                if alert.created_at >= cutoff:
                    return True
        return False

    def check_contraction_alert(
        self,
        pattern: VCPPattern,
        current_price: float,
    ) -> Optional[Alert]:
        """
        Check if a contraction alert should be created.

        Criteria:
        - Pattern is valid
        - Score meets minimum threshold
        - At least minimum contractions
        - No duplicate alert exists

        Args:
            pattern: Detected VCP pattern
            current_price: Current market price

        Returns:
            Created alert or None if criteria not met
        """
        # Check validity
        if not pattern.is_valid:
            return None

        # Check score threshold
        if pattern.proximity_score < self.config.min_score_contraction:
            return None

        # Check minimum contractions
        if pattern.num_contractions < self.config.min_contractions:
            return None

        # Check for duplicates
        if self._is_duplicate(pattern.symbol, AlertType.CONTRACTION):
            return None

        # Create the alert
        distance_pct = ((pattern.pivot_price - current_price) / current_price) * 100

        alert = Alert(
            symbol=pattern.symbol,
            alert_type=AlertType.CONTRACTION,
            trigger_price=current_price,
            pivot_price=pattern.pivot_price,
            distance_to_pivot_pct=distance_pct,
            score=pattern.proximity_score,
            num_contractions=pattern.num_contractions,
            pattern_snapshot=pattern.to_dict(),
        )

        # Save and notify
        self.repository.save(alert)
        self._notify_subscribers(alert)

        return alert

    def check_pre_alert(
        self,
        symbol: str,
        current_price: float,
        pivot_price: float,
        pattern_score: float,
        num_contractions: int,
        parent_alert_id: Optional[str] = None,
    ) -> Optional[Alert]:
        """
        Check if a pre-alert should be created.

        Criteria:
        - Price is within proximity threshold of pivot
        - No duplicate pre-alert exists

        Args:
            symbol: Stock symbol
            current_price: Current market price
            pivot_price: Pivot/breakout price
            pattern_score: Pattern quality score
            num_contractions: Number of contractions
            parent_alert_id: ID of parent contraction alert

        Returns:
            Created alert or None if criteria not met
        """
        # Calculate distance to pivot
        distance_pct = ((pivot_price - current_price) / current_price) * 100

        # Check proximity threshold
        if distance_pct > self.config.pre_alert_proximity_pct:
            return None

        # Check for duplicates
        if self._is_duplicate(symbol, AlertType.PRE_ALERT):
            return None

        # Create the alert
        alert = Alert(
            symbol=symbol,
            alert_type=AlertType.PRE_ALERT,
            trigger_price=current_price,
            pivot_price=pivot_price,
            distance_to_pivot_pct=distance_pct,
            score=pattern_score,
            num_contractions=num_contractions,
            parent_alert_id=parent_alert_id,
        )

        # Mark parent as converted if exists
        if parent_alert_id:
            parent = self.repository.get_by_id(parent_alert_id)
            if parent and parent.state in (AlertState.PENDING, AlertState.NOTIFIED):
                parent.mark_converted(alert.id)
                self.repository.update(parent)

        # Save and notify
        self.repository.save(alert)
        self._notify_subscribers(alert)

        return alert

    def check_trade_alert(
        self,
        symbol: str,
        entry_signal: EntrySignal,
        pivot_price: float,
        pattern_score: float,
        num_contractions: int,
        parent_alert_id: Optional[str] = None,
    ) -> Optional[Alert]:
        """
        Check if a trade alert should be created.

        Criteria:
        - Entry signal is valid
        - No duplicate trade alert exists

        Args:
            symbol: Stock symbol
            entry_signal: Entry signal details
            pivot_price: Pivot/breakout price
            pattern_score: Pattern quality score
            num_contractions: Number of contractions
            parent_alert_id: ID of parent pre-alert

        Returns:
            Created alert or None if criteria not met
        """
        # Check for duplicates
        if self._is_duplicate(symbol, AlertType.TRADE):
            return None

        # Calculate distance (negative = above pivot)
        distance_pct = ((pivot_price - entry_signal.price) / entry_signal.price) * 100

        # Create the alert
        alert = Alert(
            symbol=symbol,
            alert_type=AlertType.TRADE,
            trigger_price=entry_signal.price,
            pivot_price=pivot_price,
            distance_to_pivot_pct=distance_pct,
            score=pattern_score,
            num_contractions=num_contractions,
            parent_alert_id=parent_alert_id,
            pattern_snapshot=entry_signal.to_dict(),
        )

        # Mark parent as converted if exists
        if parent_alert_id:
            parent = self.repository.get_by_id(parent_alert_id)
            if parent and parent.state in (AlertState.PENDING, AlertState.NOTIFIED):
                parent.mark_converted(alert.id)
                self.repository.update(parent)

        # Save and notify
        self.repository.save(alert)
        self._notify_subscribers(alert)

        return alert

    def check_and_emit(
        self,
        pattern: VCPPattern,
        current_price: float,
        entry_signal: Optional[EntrySignal] = None,
    ) -> List[Alert]:
        """
        Check all alert conditions and emit applicable alerts.

        This is the main method for processing a pattern and price update.
        It checks each stage in order and emits alerts as appropriate.

        Args:
            pattern: Detected VCP pattern
            current_price: Current market price
            entry_signal: Optional entry signal if triggered

        Returns:
            List of created alerts
        """
        alerts = []

        # Get existing active alerts for this symbol
        active_alerts = self.repository.query(
            symbol=pattern.symbol,
            state=AlertState.PENDING,
        ) + self.repository.query(
            symbol=pattern.symbol,
            state=AlertState.NOTIFIED,
        )

        # Find parent alerts by type
        contraction_parent = None
        pre_alert_parent = None
        for alert in active_alerts:
            if alert.alert_type == AlertType.CONTRACTION:
                contraction_parent = alert
            elif alert.alert_type == AlertType.PRE_ALERT:
                pre_alert_parent = alert

        # Check for trade alert first (highest priority)
        if entry_signal:
            parent_id = pre_alert_parent.id if pre_alert_parent else (
                contraction_parent.id if contraction_parent else None
            )
            trade_alert = self.check_trade_alert(
                symbol=pattern.symbol,
                entry_signal=entry_signal,
                pivot_price=pattern.pivot_price,
                pattern_score=pattern.proximity_score,
                num_contractions=pattern.num_contractions,
                parent_alert_id=parent_id,
            )
            if trade_alert:
                alerts.append(trade_alert)
                return alerts  # Don't create other alerts if trade triggered

        # Check for pre-alert
        distance_pct = ((pattern.pivot_price - current_price) / current_price) * 100
        if 0 < distance_pct <= self.config.pre_alert_proximity_pct:
            parent_id = contraction_parent.id if contraction_parent else None
            pre_alert = self.check_pre_alert(
                symbol=pattern.symbol,
                current_price=current_price,
                pivot_price=pattern.pivot_price,
                pattern_score=pattern.proximity_score,
                num_contractions=pattern.num_contractions,
                parent_alert_id=parent_id,
            )
            if pre_alert:
                alerts.append(pre_alert)

        # Check for contraction alert (only if no pre-alert/trade exists)
        if not pre_alert_parent and not contraction_parent:
            contraction_alert = self.check_contraction_alert(
                pattern=pattern,
                current_price=current_price,
            )
            if contraction_alert:
                alerts.append(contraction_alert)

        return alerts

    def mark_notified(self, alert_id: str) -> bool:
        """
        Mark an alert as notified.

        Args:
            alert_id: ID of the alert to mark

        Returns:
            True if successful, False if alert not found or invalid state
        """
        alert = self.repository.get_by_id(alert_id)
        if not alert:
            return False

        if alert.state != AlertState.PENDING:
            return False

        alert.mark_notified()
        self.repository.update(alert)
        return True

    def mark_converted(self, alert_id: str, child_alert_id: str) -> bool:
        """
        Mark an alert as converted to next stage.

        Args:
            alert_id: ID of the alert to mark
            child_alert_id: ID of the child alert

        Returns:
            True if successful, False if alert not found or invalid state
        """
        alert = self.repository.get_by_id(alert_id)
        if not alert:
            return False

        if alert.state not in (AlertState.PENDING, AlertState.NOTIFIED):
            return False

        alert.mark_converted(child_alert_id)
        self.repository.update(alert)
        return True

    def mark_completed(self, alert_id: str) -> bool:
        """
        Mark a trade alert as completed.

        Args:
            alert_id: ID of the trade alert

        Returns:
            True if successful, False if alert not found or invalid
        """
        alert = self.repository.get_by_id(alert_id)
        if not alert:
            return False

        if alert.alert_type != AlertType.TRADE:
            return False

        if alert.state != AlertState.NOTIFIED:
            return False

        alert.mark_completed()
        self.repository.update(alert)
        return True

    def expire_stale_alerts(self) -> int:
        """
        Expire alerts that have exceeded their TTL.

        Returns:
            Number of alerts expired
        """
        now = datetime.now()
        expired_count = 0

        # Get potentially expiring alerts by type
        for alert_type, ttl_days in [
            (AlertType.CONTRACTION, self.config.contraction_ttl_days),
            (AlertType.PRE_ALERT, self.config.pre_alert_ttl_days),
            (AlertType.TRADE, self.config.trade_ttl_days),
        ]:
            cutoff = now - timedelta(days=ttl_days)
            alerts = self.repository.get_expiring(cutoff)

            for alert in alerts:
                if alert.alert_type == alert_type:
                    alert.mark_expired()
                    self.repository.update(alert)
                    expired_count += 1

        return expired_count

    def get_active_alerts(self, symbol: Optional[str] = None) -> List[Alert]:
        """
        Get all active (non-expired, non-completed) alerts.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of active alerts
        """
        pending = self.repository.query(symbol=symbol, state=AlertState.PENDING)
        notified = self.repository.query(symbol=symbol, state=AlertState.NOTIFIED)
        return pending + notified

    def get_alert_chain(self, alert_id: str) -> Optional[AlertChain]:
        """
        Build the complete alert chain for an alert.

        Traverses parent/child relationships to build the full chain.

        Args:
            alert_id: ID of any alert in the chain

        Returns:
            AlertChain or None if alert not found
        """
        alert = self.repository.get_by_id(alert_id)
        if not alert:
            return None

        chain = AlertChain(symbol=alert.symbol)

        # Walk up to find root (contraction alert)
        current = alert
        while current.parent_alert_id:
            parent = self.repository.get_by_id(current.parent_alert_id)
            if not parent:
                break
            current = parent

        # Now we have the root, walk down to build chain
        root = current

        if root.alert_type == AlertType.CONTRACTION:
            chain.contraction_alert = root
        elif root.alert_type == AlertType.PRE_ALERT:
            chain.add_pre_alert(root)
        elif root.alert_type == AlertType.TRADE:
            chain.trade_alert = root

        # Get children recursively
        def add_children(parent: Alert):
            children = self.repository.get_children(parent.id)
            for child in children:
                if child.alert_type == AlertType.PRE_ALERT:
                    chain.add_pre_alert(child)
                elif child.alert_type == AlertType.TRADE:
                    chain.trade_alert = child
                add_children(child)

        add_children(root)

        return chain

    def get_alert_history(
        self,
        symbol: Optional[str] = None,
        alert_type: Optional[AlertType] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Alert]:
        """
        Get alert history with optional filters.

        Args:
            symbol: Filter by symbol
            alert_type: Filter by alert type
            start_date: Filter by start date
            end_date: Filter by end date

        Returns:
            List of alerts matching criteria
        """
        return self.repository.query(
            symbol=symbol,
            alert_type=alert_type,
            start_date=start_date,
            end_date=end_date,
        )

    def get_conversion_stats(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> ConversionStats:
        """
        Calculate conversion statistics for a time period.

        Args:
            start_date: Start of period
            end_date: End of period

        Returns:
            ConversionStats for the period
        """
        # Get all alerts in period
        contraction_alerts = self.repository.query(
            alert_type=AlertType.CONTRACTION,
            start_date=start_date,
            end_date=end_date,
        )
        pre_alerts = self.repository.query(
            alert_type=AlertType.PRE_ALERT,
            start_date=start_date,
            end_date=end_date,
        )
        trade_alerts = self.repository.query(
            alert_type=AlertType.TRADE,
            start_date=start_date,
            end_date=end_date,
        )

        # Count totals
        total_contraction = len(contraction_alerts)
        total_pre = len(pre_alerts)
        total_trade = len(trade_alerts)

        # Count conversions
        contraction_to_pre = sum(
            1 for a in contraction_alerts if a.state == AlertState.CONVERTED
        )
        contraction_to_trade = sum(
            1 for a in contraction_alerts
            if a.state == AlertState.CONVERTED
            # Check if any child is a trade alert
        )
        pre_to_trade = sum(
            1 for a in pre_alerts if a.state == AlertState.CONVERTED
        )

        # Calculate rates
        contraction_to_pre_rate = (
            contraction_to_pre / total_contraction if total_contraction > 0 else 0.0
        )
        contraction_to_trade_rate = (
            total_trade / total_contraction if total_contraction > 0 else 0.0
        )
        pre_to_trade_rate = (
            pre_to_trade / total_pre if total_pre > 0 else 0.0
        )

        # Calculate lead times
        lead_times_contraction_to_trade = []
        lead_times_pre_to_trade = []

        for trade in trade_alerts:
            chain = self.get_alert_chain(trade.id)
            if chain:
                if chain.total_lead_time_days is not None:
                    lead_times_contraction_to_trade.append(chain.total_lead_time_days)
                if chain.pre_alert_lead_time_days is not None:
                    lead_times_pre_to_trade.append(chain.pre_alert_lead_time_days)

        avg_contraction_to_trade = (
            sum(lead_times_contraction_to_trade) / len(lead_times_contraction_to_trade)
            if lead_times_contraction_to_trade else 0.0
        )
        avg_pre_to_trade = (
            sum(lead_times_pre_to_trade) / len(lead_times_pre_to_trade)
            if lead_times_pre_to_trade else 0.0
        )

        return ConversionStats(
            period_start=start_date,
            period_end=end_date,
            total_contraction_alerts=total_contraction,
            total_pre_alerts=total_pre,
            total_trade_alerts=total_trade,
            contraction_to_pre_alert_rate=contraction_to_pre_rate,
            contraction_to_trade_rate=contraction_to_trade_rate,
            pre_alert_to_trade_rate=pre_to_trade_rate,
            avg_days_contraction_to_trade=avg_contraction_to_trade,
            avg_days_pre_alert_to_trade=avg_pre_to_trade,
        )
