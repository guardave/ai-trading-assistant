"""
VCP Alert System - Repository Layer

Provides data persistence for alerts with protocol-based abstraction:
- AlertRepository protocol defines the interface
- SQLiteAlertRepository implements SQLite storage
- PostgreSQLAlertRepository can be added for production
"""

import json
import sqlite3
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Protocol, Iterator

from .models import Alert, AlertType, AlertState


class AlertRepository(Protocol):
    """
    Protocol defining the alert repository interface.

    Implementations must provide methods for:
    - CRUD operations on alerts
    - Query by various criteria
    - Alert chain traversal
    """

    def save(self, alert: Alert) -> None:
        """Save a new alert to the repository."""
        ...

    def get_by_id(self, alert_id: str) -> Optional[Alert]:
        """Get an alert by its ID."""
        ...

    def update(self, alert: Alert) -> None:
        """Update an existing alert."""
        ...

    def delete(self, alert_id: str) -> None:
        """Delete an alert by ID."""
        ...

    def query(
        self,
        symbol: Optional[str] = None,
        alert_type: Optional[AlertType] = None,
        state: Optional[AlertState] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        parent_alert_id: Optional[str] = None,
    ) -> List[Alert]:
        """Query alerts by various criteria."""
        ...

    def get_children(self, parent_id: str) -> List[Alert]:
        """Get all child alerts for a parent alert."""
        ...

    def get_expiring(self, before: datetime) -> List[Alert]:
        """Get alerts that should be expired by the given datetime."""
        ...


class SQLiteAlertRepository:
    """
    SQLite implementation of AlertRepository.

    Stores alerts in a local SQLite database with support for:
    - All CRUD operations
    - Flexible querying
    - JSON storage for pattern snapshots
    """

    def __init__(self, db_path: str = "data/alerts.db"):
        """
        Initialize the SQLite repository.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vcp_alerts (
                    id VARCHAR(36) PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    alert_type VARCHAR(20) NOT NULL,
                    state VARCHAR(20) NOT NULL DEFAULT 'pending',

                    -- Timestamps
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    notified_at TIMESTAMP,
                    converted_at TIMESTAMP,
                    expired_at TIMESTAMP,

                    -- Linkage
                    parent_alert_id VARCHAR(36) REFERENCES vcp_alerts(id),

                    -- Pattern data
                    trigger_price DECIMAL(12,4) NOT NULL,
                    pivot_price DECIMAL(12,4) NOT NULL,
                    distance_to_pivot_pct DECIMAL(8,4) NOT NULL,
                    score DECIMAL(5,2) NOT NULL,
                    num_contractions INTEGER NOT NULL,
                    pattern_snapshot TEXT
                )
            """)
            # Create indexes
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_vcp_alerts_symbol
                ON vcp_alerts(symbol)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_vcp_alerts_type_state
                ON vcp_alerts(alert_type, state)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_vcp_alerts_created
                ON vcp_alerts(created_at)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_vcp_alerts_parent
                ON vcp_alerts(parent_alert_id)
            """)

    def _row_to_alert(self, row: sqlite3.Row) -> Alert:
        """Convert a database row to an Alert object."""
        pattern_snapshot = {}
        if row["pattern_snapshot"]:
            pattern_snapshot = json.loads(row["pattern_snapshot"])

        alert = Alert(
            id=row["id"],
            symbol=row["symbol"],
            alert_type=AlertType(row["alert_type"]),
            state=AlertState(row["state"]),
            trigger_price=row["trigger_price"],
            pivot_price=row["pivot_price"],
            distance_to_pivot_pct=row["distance_to_pivot_pct"],
            score=row["score"],
            num_contractions=row["num_contractions"],
            pattern_snapshot=pattern_snapshot,
            parent_alert_id=row["parent_alert_id"],
        )

        # Set timestamps
        alert.created_at = datetime.fromisoformat(row["created_at"])
        alert.updated_at = datetime.fromisoformat(row["updated_at"])
        if row["notified_at"]:
            alert.notified_at = datetime.fromisoformat(row["notified_at"])
        if row["converted_at"]:
            alert.converted_at = datetime.fromisoformat(row["converted_at"])
        if row["expired_at"]:
            alert.expired_at = datetime.fromisoformat(row["expired_at"])

        return alert

    def save(self, alert: Alert) -> None:
        """Save a new alert to the database."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO vcp_alerts (
                    id, symbol, alert_type, state,
                    created_at, updated_at, notified_at, converted_at, expired_at,
                    parent_alert_id,
                    trigger_price, pivot_price, distance_to_pivot_pct,
                    score, num_contractions, pattern_snapshot
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    alert.id,
                    alert.symbol,
                    alert.alert_type.value,
                    alert.state.value,
                    alert.created_at.isoformat(),
                    alert.updated_at.isoformat(),
                    alert.notified_at.isoformat() if alert.notified_at else None,
                    alert.converted_at.isoformat() if alert.converted_at else None,
                    alert.expired_at.isoformat() if alert.expired_at else None,
                    alert.parent_alert_id,
                    alert.trigger_price,
                    alert.pivot_price,
                    alert.distance_to_pivot_pct,
                    alert.score,
                    alert.num_contractions,
                    json.dumps(alert.pattern_snapshot) if alert.pattern_snapshot else None,
                ),
            )

    def get_by_id(self, alert_id: str) -> Optional[Alert]:
        """Get an alert by its ID."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM vcp_alerts WHERE id = ?",
                (alert_id,)
            )
            row = cursor.fetchone()
            if row:
                return self._row_to_alert(row)
            return None

    def update(self, alert: Alert) -> None:
        """Update an existing alert."""
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE vcp_alerts SET
                    symbol = ?,
                    alert_type = ?,
                    state = ?,
                    updated_at = ?,
                    notified_at = ?,
                    converted_at = ?,
                    expired_at = ?,
                    parent_alert_id = ?,
                    trigger_price = ?,
                    pivot_price = ?,
                    distance_to_pivot_pct = ?,
                    score = ?,
                    num_contractions = ?,
                    pattern_snapshot = ?
                WHERE id = ?
                """,
                (
                    alert.symbol,
                    alert.alert_type.value,
                    alert.state.value,
                    alert.updated_at.isoformat(),
                    alert.notified_at.isoformat() if alert.notified_at else None,
                    alert.converted_at.isoformat() if alert.converted_at else None,
                    alert.expired_at.isoformat() if alert.expired_at else None,
                    alert.parent_alert_id,
                    alert.trigger_price,
                    alert.pivot_price,
                    alert.distance_to_pivot_pct,
                    alert.score,
                    alert.num_contractions,
                    json.dumps(alert.pattern_snapshot) if alert.pattern_snapshot else None,
                    alert.id,
                ),
            )

    def delete(self, alert_id: str) -> None:
        """Delete an alert by ID."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM vcp_alerts WHERE id = ?", (alert_id,))

    def query(
        self,
        symbol: Optional[str] = None,
        alert_type: Optional[AlertType] = None,
        state: Optional[AlertState] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        parent_alert_id: Optional[str] = None,
    ) -> List[Alert]:
        """Query alerts by various criteria."""
        conditions = []
        params = []

        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)

        if alert_type:
            conditions.append("alert_type = ?")
            params.append(alert_type.value)

        if state:
            conditions.append("state = ?")
            params.append(state.value)

        if start_date:
            conditions.append("created_at >= ?")
            params.append(start_date.isoformat())

        if end_date:
            conditions.append("created_at <= ?")
            params.append(end_date.isoformat())

        if parent_alert_id:
            conditions.append("parent_alert_id = ?")
            params.append(parent_alert_id)

        query = "SELECT * FROM vcp_alerts"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at DESC"

        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return [self._row_to_alert(row) for row in cursor.fetchall()]

    def get_children(self, parent_id: str) -> List[Alert]:
        """Get all child alerts for a parent alert."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM vcp_alerts WHERE parent_alert_id = ? ORDER BY created_at",
                (parent_id,)
            )
            return [self._row_to_alert(row) for row in cursor.fetchall()]

    def get_expiring(self, before: datetime) -> List[Alert]:
        """
        Get alerts that should be expired.

        Returns alerts that are:
        - In PENDING or NOTIFIED state
        - Created before the specified datetime
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM vcp_alerts
                WHERE state IN ('pending', 'notified')
                AND created_at < ?
                ORDER BY created_at
                """,
                (before.isoformat(),)
            )
            return [self._row_to_alert(row) for row in cursor.fetchall()]

    def get_active_by_symbol(self, symbol: str) -> List[Alert]:
        """Get all active (non-expired, non-completed) alerts for a symbol."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM vcp_alerts
                WHERE symbol = ?
                AND state IN ('pending', 'notified')
                ORDER BY created_at DESC
                """,
                (symbol,)
            )
            return [self._row_to_alert(row) for row in cursor.fetchall()]

    def count_by_state(self) -> dict:
        """Get count of alerts by state."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT state, COUNT(*) as count
                FROM vcp_alerts
                GROUP BY state
                """
            )
            return {row["state"]: row["count"] for row in cursor.fetchall()}

    def count_by_type(self) -> dict:
        """Get count of alerts by type."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT alert_type, COUNT(*) as count
                FROM vcp_alerts
                GROUP BY alert_type
                """
            )
            return {row["alert_type"]: row["count"] for row in cursor.fetchall()}

    def clear_all(self) -> int:
        """Clear all alerts from the database. Returns count deleted."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM vcp_alerts")
            count = cursor.fetchone()[0]
            conn.execute("DELETE FROM vcp_alerts")
            return count


class InMemoryAlertRepository:
    """
    In-memory implementation of AlertRepository for testing.

    Stores alerts in a dictionary - no persistence.
    """

    def __init__(self):
        self._alerts: dict[str, Alert] = {}

    def save(self, alert: Alert) -> None:
        """Save a new alert."""
        self._alerts[alert.id] = alert

    def get_by_id(self, alert_id: str) -> Optional[Alert]:
        """Get an alert by ID."""
        return self._alerts.get(alert_id)

    def update(self, alert: Alert) -> None:
        """Update an existing alert."""
        if alert.id in self._alerts:
            self._alerts[alert.id] = alert

    def delete(self, alert_id: str) -> None:
        """Delete an alert by ID."""
        self._alerts.pop(alert_id, None)

    def query(
        self,
        symbol: Optional[str] = None,
        alert_type: Optional[AlertType] = None,
        state: Optional[AlertState] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        parent_alert_id: Optional[str] = None,
    ) -> List[Alert]:
        """Query alerts by criteria."""
        results = list(self._alerts.values())

        if symbol:
            results = [a for a in results if a.symbol == symbol]
        if alert_type:
            results = [a for a in results if a.alert_type == alert_type]
        if state:
            results = [a for a in results if a.state == state]
        if start_date:
            results = [a for a in results if a.created_at >= start_date]
        if end_date:
            results = [a for a in results if a.created_at <= end_date]
        if parent_alert_id:
            results = [a for a in results if a.parent_alert_id == parent_alert_id]

        return sorted(results, key=lambda a: a.created_at, reverse=True)

    def get_children(self, parent_id: str) -> List[Alert]:
        """Get all child alerts for a parent."""
        children = [a for a in self._alerts.values() if a.parent_alert_id == parent_id]
        return sorted(children, key=lambda a: a.created_at)

    def get_expiring(self, before: datetime) -> List[Alert]:
        """Get alerts that should be expired."""
        results = [
            a for a in self._alerts.values()
            if a.state in (AlertState.PENDING, AlertState.NOTIFIED)
            and a.created_at < before
        ]
        return sorted(results, key=lambda a: a.created_at)

    def get_active_by_symbol(self, symbol: str) -> List[Alert]:
        """Get active alerts for a symbol."""
        results = [
            a for a in self._alerts.values()
            if a.symbol == symbol
            and a.state in (AlertState.PENDING, AlertState.NOTIFIED)
        ]
        return sorted(results, key=lambda a: a.created_at, reverse=True)

    def clear_all(self) -> int:
        """Clear all alerts."""
        count = len(self._alerts)
        self._alerts.clear()
        return count
