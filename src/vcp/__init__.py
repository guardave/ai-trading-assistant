"""
VCP Alert System Package

Three-stage alert system for VCP (Volatility Contraction Pattern) trading:
1. Contraction Alert - Pattern detected with qualified contractions
2. Pre-Alert - Price within proximity of pivot
3. Trade Alert - Entry signal triggered

Usage:
    from src.vcp import VCPAlertSystem, create_system

    # Quick start
    system = create_system()
    alerts = system.process_symbol("AAPL", price_data)

    # Custom configuration
    from src.vcp import SystemConfig, DetectorConfig, AlertConfig
    config = SystemConfig(
        detector_config=DetectorConfig(min_contractions=3),
        alert_config=AlertConfig(min_score_contraction=70.0),
    )
    system = VCPAlertSystem(config)
"""

from .models import (
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
from .detector import VCPDetector, DetectorConfig
from .alert_manager import AlertManager, AlertConfig
from .repository import (
    AlertRepository,
    SQLiteAlertRepository,
    InMemoryAlertRepository,
)
from .notifications import (
    NotificationHub,
    NotificationChannel,
    LogNotificationChannel,
    ConsoleNotificationChannel,
    WebhookNotificationChannel,
    CallbackNotificationChannel,
)
from .alert_system import VCPAlertSystem, SystemConfig, create_system

__all__ = [
    # Models
    "AlertType",
    "AlertState",
    "Alert",
    "AlertChain",
    "ConversionStats",
    "VCPPattern",
    "Contraction",
    "SwingPoint",
    "EntrySignal",
    "EntryType",
    # Detector
    "VCPDetector",
    "DetectorConfig",
    # Alert Manager
    "AlertManager",
    "AlertConfig",
    # Repository
    "AlertRepository",
    "SQLiteAlertRepository",
    "InMemoryAlertRepository",
    # Notifications
    "NotificationHub",
    "NotificationChannel",
    "LogNotificationChannel",
    "ConsoleNotificationChannel",
    "WebhookNotificationChannel",
    "CallbackNotificationChannel",
    # Main System
    "VCPAlertSystem",
    "SystemConfig",
    "create_system",
]
