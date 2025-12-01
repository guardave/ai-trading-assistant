"""
VCP Alert System - Core Data Models

Defines the data structures for the three-stage VCP alert system:
- Alert types and states
- Alert model with timestamps and linkage
- VCP pattern components (SwingPoint, Contraction, VCPPattern)
- Entry signals
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
import uuid


class AlertType(Enum):
    """Types of alerts in the three-stage system."""
    CONTRACTION = "contraction"
    PRE_ALERT = "pre_alert"
    TRADE = "trade"


class AlertState(Enum):
    """Alert lifecycle states."""
    PENDING = "pending"      # Created, awaiting notification
    NOTIFIED = "notified"    # User has been notified
    CONVERTED = "converted"  # Progressed to next stage
    EXPIRED = "expired"      # TTL exceeded without conversion
    COMPLETED = "completed"  # Trade closed (trade alerts only)


class EntryType(Enum):
    """Types of entry signals."""
    PIVOT_BREAKOUT = "pivot_breakout"
    HANDLE_BREAK = "handle_break"


@dataclass
class SwingPoint:
    """A swing high or swing low point in price data."""
    index: int
    date: datetime
    price: float
    point_type: str  # 'high' or 'low'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "index": self.index,
            "date": self.date.isoformat() if isinstance(self.date, datetime) else str(self.date),
            "price": self.price,
            "point_type": self.point_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SwingPoint":
        """Create from dictionary."""
        date = data["date"]
        if isinstance(date, str):
            date = datetime.fromisoformat(date)
        return cls(
            index=data["index"],
            date=date,
            price=data["price"],
            point_type=data["point_type"],
        )


@dataclass
class Contraction:
    """A single VCP contraction defined by swing highs and lows."""
    swing_high: SwingPoint
    swing_low: SwingPoint
    range_pct: float  # (high - low) / low * 100
    duration_days: int
    avg_volume_ratio: float  # vs 50-day average (< 1 means volume dry-up)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "swing_high": self.swing_high.to_dict(),
            "swing_low": self.swing_low.to_dict(),
            "range_pct": self.range_pct,
            "duration_days": self.duration_days,
            "avg_volume_ratio": self.avg_volume_ratio,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Contraction":
        """Create from dictionary."""
        return cls(
            swing_high=SwingPoint.from_dict(data["swing_high"]),
            swing_low=SwingPoint.from_dict(data["swing_low"]),
            range_pct=data["range_pct"],
            duration_days=data["duration_days"],
            avg_volume_ratio=data["avg_volume_ratio"],
        )


@dataclass
class VCPPattern:
    """Complete VCP pattern analysis."""
    symbol: str
    contractions: List[Contraction]
    is_valid: bool
    validity_reasons: List[str]

    # Scores (0-100)
    contraction_quality: float  # Are contractions progressively tighter?
    volume_quality: float       # Is volume drying up?
    proximity_score: float      # Overall pattern quality

    # Pattern bounds
    pivot_price: float          # Breakout level (highest high in pattern)
    support_price: float        # Lowest low in final contraction

    # Metadata
    detection_date: datetime = field(default_factory=datetime.now)

    @property
    def num_contractions(self) -> int:
        """Number of contractions in the pattern."""
        return len(self.contractions)

    @property
    def last_contraction_range_pct(self) -> float:
        """Range percentage of the final contraction."""
        if self.contractions:
            return self.contractions[-1].range_pct
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "contractions": [c.to_dict() for c in self.contractions],
            "is_valid": self.is_valid,
            "validity_reasons": self.validity_reasons,
            "contraction_quality": self.contraction_quality,
            "volume_quality": self.volume_quality,
            "proximity_score": self.proximity_score,
            "pivot_price": self.pivot_price,
            "support_price": self.support_price,
            "detection_date": self.detection_date.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VCPPattern":
        """Create from dictionary."""
        return cls(
            symbol=data["symbol"],
            contractions=[Contraction.from_dict(c) for c in data["contractions"]],
            is_valid=data["is_valid"],
            validity_reasons=data["validity_reasons"],
            contraction_quality=data["contraction_quality"],
            volume_quality=data["volume_quality"],
            proximity_score=data["proximity_score"],
            pivot_price=data["pivot_price"],
            support_price=data["support_price"],
            detection_date=datetime.fromisoformat(data["detection_date"]),
        )


@dataclass
class EntrySignal:
    """An entry signal for trading."""
    entry_type: EntryType
    date: datetime
    price: float
    stop_price: float
    target_price: float
    volume_ratio: float  # vs average

    @property
    def risk_pct(self) -> float:
        """Percentage risk from entry to stop."""
        if self.price > 0:
            return ((self.price - self.stop_price) / self.price) * 100
        return 0.0

    @property
    def reward_risk_ratio(self) -> float:
        """Reward to risk ratio."""
        risk = self.price - self.stop_price
        if risk > 0:
            reward = self.target_price - self.price
            return reward / risk
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "entry_type": self.entry_type.value,
            "date": self.date.isoformat(),
            "price": self.price,
            "stop_price": self.stop_price,
            "target_price": self.target_price,
            "volume_ratio": self.volume_ratio,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EntrySignal":
        """Create from dictionary."""
        return cls(
            entry_type=EntryType(data["entry_type"]),
            date=datetime.fromisoformat(data["date"]),
            price=data["price"],
            stop_price=data["stop_price"],
            target_price=data["target_price"],
            volume_ratio=data["volume_ratio"],
        )


@dataclass
class Alert:
    """
    A VCP alert representing one stage in the alert chain.

    Alerts progress through states:
    - PENDING -> NOTIFIED (user notified)
    - PENDING/NOTIFIED -> CONVERTED (progressed to next stage)
    - PENDING/NOTIFIED -> EXPIRED (TTL exceeded)
    - NOTIFIED -> COMPLETED (trade closed, trade alerts only)
    """
    symbol: str
    alert_type: AlertType

    # Pattern data at alert time
    trigger_price: float
    pivot_price: float
    distance_to_pivot_pct: float
    score: float
    num_contractions: int
    pattern_snapshot: Dict[str, Any] = field(default_factory=dict)

    # State
    state: AlertState = AlertState.PENDING

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    notified_at: Optional[datetime] = None
    converted_at: Optional[datetime] = None
    expired_at: Optional[datetime] = None

    # Linkage
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_alert_id: Optional[str] = None

    def mark_notified(self) -> None:
        """Mark alert as notified."""
        if self.state == AlertState.PENDING:
            self.state = AlertState.NOTIFIED
            self.notified_at = datetime.now()
            self.updated_at = datetime.now()

    def mark_converted(self, child_alert_id: str) -> None:
        """Mark alert as converted to next stage."""
        if self.state in (AlertState.PENDING, AlertState.NOTIFIED):
            self.state = AlertState.CONVERTED
            self.converted_at = datetime.now()
            self.updated_at = datetime.now()

    def mark_expired(self) -> None:
        """Mark alert as expired."""
        if self.state in (AlertState.PENDING, AlertState.NOTIFIED):
            self.state = AlertState.EXPIRED
            self.expired_at = datetime.now()
            self.updated_at = datetime.now()

    def mark_completed(self) -> None:
        """Mark trade alert as completed (trade closed)."""
        if self.alert_type == AlertType.TRADE and self.state == AlertState.NOTIFIED:
            self.state = AlertState.COMPLETED
            self.updated_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "alert_type": self.alert_type.value,
            "state": self.state.value,
            "trigger_price": self.trigger_price,
            "pivot_price": self.pivot_price,
            "distance_to_pivot_pct": self.distance_to_pivot_pct,
            "score": self.score,
            "num_contractions": self.num_contractions,
            "pattern_snapshot": self.pattern_snapshot,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "notified_at": self.notified_at.isoformat() if self.notified_at else None,
            "converted_at": self.converted_at.isoformat() if self.converted_at else None,
            "expired_at": self.expired_at.isoformat() if self.expired_at else None,
            "parent_alert_id": self.parent_alert_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Alert":
        """Create from dictionary."""
        alert = cls(
            id=data["id"],
            symbol=data["symbol"],
            alert_type=AlertType(data["alert_type"]),
            state=AlertState(data["state"]),
            trigger_price=data["trigger_price"],
            pivot_price=data["pivot_price"],
            distance_to_pivot_pct=data["distance_to_pivot_pct"],
            score=data["score"],
            num_contractions=data["num_contractions"],
            pattern_snapshot=data.get("pattern_snapshot", {}),
            parent_alert_id=data.get("parent_alert_id"),
        )
        alert.created_at = datetime.fromisoformat(data["created_at"])
        alert.updated_at = datetime.fromisoformat(data["updated_at"])
        if data.get("notified_at"):
            alert.notified_at = datetime.fromisoformat(data["notified_at"])
        if data.get("converted_at"):
            alert.converted_at = datetime.fromisoformat(data["converted_at"])
        if data.get("expired_at"):
            alert.expired_at = datetime.fromisoformat(data["expired_at"])
        return alert


@dataclass
class AlertChain:
    """
    A chain of related alerts for a single symbol's VCP progression.

    Tracks the full alert journey:
    Contraction Alert -> Pre-Alert(s) -> Trade Alert
    """
    symbol: str
    contraction_alert: Optional[Alert] = None
    pre_alerts: List[Alert] = field(default_factory=list)
    trade_alert: Optional[Alert] = None

    @property
    def total_lead_time_days(self) -> Optional[int]:
        """Days from first contraction alert to trade alert."""
        if self.contraction_alert and self.trade_alert:
            delta = self.trade_alert.created_at - self.contraction_alert.created_at
            return delta.days
        return None

    @property
    def pre_alert_lead_time_days(self) -> Optional[int]:
        """Days from first pre-alert to trade alert."""
        if self.pre_alerts and self.trade_alert:
            first_pre_alert = min(self.pre_alerts, key=lambda a: a.created_at)
            delta = self.trade_alert.created_at - first_pre_alert.created_at
            return delta.days
        return None

    @property
    def has_full_chain(self) -> bool:
        """Whether chain has all three stages."""
        return (
            self.contraction_alert is not None
            and len(self.pre_alerts) > 0
            and self.trade_alert is not None
        )

    def add_pre_alert(self, alert: Alert) -> None:
        """Add a pre-alert to the chain."""
        if alert.alert_type == AlertType.PRE_ALERT:
            self.pre_alerts.append(alert)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "contraction_alert": self.contraction_alert.to_dict() if self.contraction_alert else None,
            "pre_alerts": [a.to_dict() for a in self.pre_alerts],
            "trade_alert": self.trade_alert.to_dict() if self.trade_alert else None,
        }


@dataclass
class ConversionStats:
    """Statistics about alert conversions over a time period."""
    period_start: datetime
    period_end: datetime

    # Counts
    total_contraction_alerts: int
    total_pre_alerts: int
    total_trade_alerts: int

    # Conversion rates (0.0 - 1.0)
    contraction_to_pre_alert_rate: float
    contraction_to_trade_rate: float
    pre_alert_to_trade_rate: float

    # Lead times (days)
    avg_days_contraction_to_trade: float
    avg_days_pre_alert_to_trade: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "total_contraction_alerts": self.total_contraction_alerts,
            "total_pre_alerts": self.total_pre_alerts,
            "total_trade_alerts": self.total_trade_alerts,
            "contraction_to_pre_alert_rate": self.contraction_to_pre_alert_rate,
            "contraction_to_trade_rate": self.contraction_to_trade_rate,
            "pre_alert_to_trade_rate": self.pre_alert_to_trade_rate,
            "avg_days_contraction_to_trade": self.avg_days_contraction_to_trade,
            "avg_days_pre_alert_to_trade": self.avg_days_pre_alert_to_trade,
        }
