#!/usr/bin/env python3
"""
Backtest Three-Stage Alert System

Implements a three-stage alert system:
1. Contraction Alert (earliest): New qualified contraction detected - "Pattern Building"
2. Pre-Alert (Day -N to -1): Within 3% of pivot - "Watch Closely"
3. Trade Alert (Day 0): Entry signal triggered - "Execute"

This allows traders to:
- Build watchlist early when patterns are forming
- Review and validate setups as they approach pivot
- Prepare position sizing and orders
- Enter at optimal timing
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
import pandas as pd
import numpy as np
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

from vcp_detector_v5 import VCPDetectorV5, VCPPatternV5, EntryType, EntrySignal

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ContractionAlert:
    """Alert when a new qualified contraction is detected (earliest signal)"""
    symbol: str
    alert_date: str
    alert_type: str  # 'new_pattern' for first contraction, 'new_contraction' for additional
    num_contractions: int
    latest_contraction_pct: float  # Range % of latest contraction
    tightening_ratio: float  # vs previous contraction (0 if first)
    overall_score: float
    prior_advance_pct: float
    pivot_price: float
    current_price: float
    distance_to_pivot_pct: float
    converted_to_pre_alert: bool = False
    converted_to_trade: bool = False
    pre_alert_date: Optional[str] = None
    trade_date: Optional[str] = None


@dataclass
class ContractionSequence:
    """Tracks contraction alerts for a pattern"""
    symbol: str
    first_alert_date: str
    alerts: List[ContractionAlert] = field(default_factory=list)
    converted_to_pre_alert: bool = False
    converted_to_trade: bool = False
    pre_alert_date: Optional[str] = None
    trade_date: Optional[str] = None

    @property
    def num_alerts(self) -> int:
        return len(self.alerts)

    @property
    def max_contractions(self) -> int:
        return max((a.num_contractions for a in self.alerts), default=0)


@dataclass
class PreAlert:
    """Pre-alert when pattern is forming near pivot"""
    symbol: str
    alert_date: str
    close_price: float
    pivot_price: float
    distance_to_pivot_pct: float
    overall_score: float
    num_contractions: int
    final_contraction_pct: float
    prior_advance_pct: float
    base_low: float
    position_in_base: str  # Where current price is in the base
    volume_vs_avg: float  # Current volume vs 50-day avg
    alert_sequence: int = 1  # Which pre-alert in the sequence (1st, 2nd, 3rd...)
    converted_to_trade: bool = False
    trade_date: Optional[str] = None
    trade_entry_type: Optional[str] = None


@dataclass
class PreAlertSequence:
    """Tracks a sequence of pre-alerts for the same pattern"""
    symbol: str
    first_alert_date: str
    alerts: List[PreAlert] = field(default_factory=list)
    converted_to_trade: bool = False
    trade_date: Optional[str] = None

    @property
    def num_alerts(self) -> int:
        return len(self.alerts)

    @property
    def last_alert(self) -> Optional[PreAlert]:
        return self.alerts[-1] if self.alerts else None


@dataclass
class Trade:
    """Represents a completed trade"""
    symbol: str
    # Contraction alert tracking
    first_contraction_alert_date: Optional[str] = None
    num_contraction_alerts: int = 0
    max_contractions_seen: int = 0
    # Pre-alert tracking
    first_pre_alert_date: Optional[str] = None
    last_pre_alert_date: Optional[str] = None
    num_pre_alerts: int = 0
    # Trade details
    signal_date: str = ""
    entry_date: str = ""
    entry_type: str = ""
    entry_price: float = 0.0
    stop_loss: float = 0.0
    target_price: float = 0.0
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl_pct: Optional[float] = None
    days_held: Optional[int] = None
    risk_pct: Optional[float] = None
    overall_score: Optional[float] = None
    # Alert status
    had_contraction_alert: bool = False
    had_pre_alert: bool = False
    days_from_first_contraction_alert: Optional[int] = None
    days_from_first_pre_alert: Optional[int] = None


@dataclass
class BacktestResult:
    """Results from three-stage alert backtest"""
    # Contraction alert stats
    total_contraction_sequences: int = 0
    total_contraction_alerts: int = 0
    contraction_to_pre_alert_rate: float = 0.0
    contraction_to_trade_rate: float = 0.0
    avg_contraction_alerts_per_seq: float = 0.0
    # Pre-alert stats
    total_pre_alert_sequences: int = 0
    total_pre_alerts: int = 0
    pre_alert_to_trade_rate: float = 0.0
    avg_pre_alerts_per_seq: float = 0.0
    # Trade stats
    total_trades: int = 0
    trades_with_contraction_alert: int = 0
    trades_with_pre_alert: int = 0
    trades_with_both_alerts: int = 0
    trades_with_no_alerts: int = 0
    win_rate_with_contraction_alert: float = 0.0
    win_rate_with_pre_alert: float = 0.0
    win_rate_with_both: float = 0.0
    win_rate_with_none: float = 0.0
    avg_days_contraction_to_trade: float = 0.0
    avg_days_pre_alert_to_trade: float = 0.0
    profit_factor: float = 0.0
    total_return_pct: float = 0.0
    # Data
    contraction_sequences: List[ContractionSequence] = field(default_factory=list)
    pre_alert_sequences: List[PreAlertSequence] = field(default_factory=list)
    trades: List[Trade] = field(default_factory=list)


class DataCache:
    """Cache for historical price data"""

    def __init__(self, cache_dir: str = 'cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.memory_cache: Dict[str, pd.DataFrame] = {}

    def get_data(self, symbol: str, period: str = '5y') -> Optional[pd.DataFrame]:
        cache_key = f"{symbol}_{period}"

        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]

        cache_file = os.path.join(self.cache_dir, f"{cache_key}.parquet")
        if os.path.exists(cache_file):
            try:
                file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
                if datetime.now() - file_time < timedelta(days=1):
                    df = pd.read_parquet(cache_file)
                    self.memory_cache[cache_key] = df
                    return df
            except Exception:
                pass

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)

            if df.empty:
                return None

            df.columns = [c.title() for c in df.columns]

            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            df.to_parquet(cache_file)
            self.memory_cache[cache_key] = df
            return df

        except Exception as e:
            logger.warning(f"Error fetching {symbol}: {e}")
            return None


class ThreeStageAlertBacktest:
    """Backtest engine for three-stage alert system"""

    def __init__(
        self,
        min_contractions_for_alert: int = 2,  # Minimum contractions to trigger contraction alert
        pre_alert_distance_pct: float = 3.0,  # Within 3% of pivot for pre-alert
        min_overall_score: float = 70.0,
        target_rr: float = 3.0,
        max_hold_days: int = 60,
        use_trailing_stop: bool = True,
        trailing_activation_pct: float = 8.0,
        trailing_stop_pct: float = 5.0,
        max_contraction_alert_days: int = 20,  # Max days contraction alert active
        max_pre_alert_days: int = 5  # Max days to wait for conversion
    ):
        self.min_contractions_for_alert = min_contractions_for_alert
        self.pre_alert_distance_pct = pre_alert_distance_pct
        self.min_overall_score = min_overall_score
        self.target_rr = target_rr
        self.max_hold_days = max_hold_days
        self.use_trailing_stop = use_trailing_stop
        self.trailing_activation_pct = trailing_activation_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.max_contraction_alert_days = max_contraction_alert_days
        self.max_pre_alert_days = max_pre_alert_days

        self.detector = VCPDetectorV5()
        self.cache = DataCache()

        # Allowed entry types (recommended config)
        self.allowed_entry_types = [EntryType.LOW_CHEAT, EntryType.HANDLE, EntryType.PIVOT_BREAKOUT]

    def check_contraction_alert_conditions(
        self,
        df: pd.DataFrame,
        pattern: VCPPatternV5,
        prev_num_contractions: int = 0
    ) -> Optional[ContractionAlert]:
        """Check if a new contraction should trigger an alert."""
        if pattern is None or not pattern.is_valid:
            return None

        # Need minimum contractions
        if pattern.num_contractions < self.min_contractions_for_alert:
            return None

        # Only alert on NEW contractions (not same pattern)
        if pattern.num_contractions <= prev_num_contractions:
            return None

        # Check score threshold (use lower threshold for early alerts)
        if pattern.overall_score < self.min_overall_score - 10:  # Allow 60+ for early alerts
            return None

        current = df.iloc[-1]
        current_price = current['Close']
        pivot = pattern.pivot_price

        # Calculate distance to pivot
        distance_to_pivot_pct = (pivot - current_price) / current_price * 100

        # Get latest contraction info
        latest_contraction = pattern.contractions[-1] if pattern.contractions else None
        if not latest_contraction:
            return None

        alert_type = "new_pattern" if pattern.num_contractions == self.min_contractions_for_alert else "new_contraction"

        return ContractionAlert(
            symbol=df.name if hasattr(df, 'name') else 'UNKNOWN',
            alert_date=current.name.strftime('%Y-%m-%d'),
            alert_type=alert_type,
            num_contractions=pattern.num_contractions,
            latest_contraction_pct=latest_contraction.range_pct,
            tightening_ratio=latest_contraction.tightening_ratio,
            overall_score=pattern.overall_score,
            prior_advance_pct=pattern.prior_uptrend.advance_pct,
            pivot_price=pivot,
            current_price=current_price,
            distance_to_pivot_pct=distance_to_pivot_pct
        )

    def check_pre_alert_conditions(
        self,
        df: pd.DataFrame,
        pattern: VCPPatternV5
    ) -> Optional[PreAlert]:
        """Check if current bar qualifies for a pre-alert."""
        if pattern is None or not pattern.is_valid:
            return None

        if pattern.overall_score < self.min_overall_score:
            return None

        current = df.iloc[-1]
        current_price = current['Close']
        pivot = pattern.pivot_price
        base_low = pattern.base_low

        # Calculate distance to pivot
        distance_to_pivot_pct = (pivot - current_price) / current_price * 100

        # Pre-alert: price within X% BELOW pivot (not yet broken out)
        if distance_to_pivot_pct < 0 or distance_to_pivot_pct > self.pre_alert_distance_pct:
            return None

        # Check if already has an entry signal (if so, it's a trade alert, not pre-alert)
        if pattern.entry_signals:
            for signal in pattern.entry_signals:
                if signal.entry_type in self.allowed_entry_types:
                    return None  # Already has entry signal

        # Calculate position in base
        base_range = pivot - base_low
        if base_range > 0:
            position_ratio = (current_price - base_low) / base_range
            if position_ratio < 0.33:
                position = "lower_third"
            elif position_ratio < 0.67:
                position = "middle_third"
            else:
                position = "upper_third"
        else:
            position = "unknown"

        # Volume analysis
        vol_ma50 = df['Volume'].rolling(50).mean().iloc[-1]
        volume_vs_avg = current['Volume'] / vol_ma50 if vol_ma50 > 0 else 1.0

        return PreAlert(
            symbol=df.name if hasattr(df, 'name') else 'UNKNOWN',
            alert_date=current.name.strftime('%Y-%m-%d'),
            close_price=current_price,
            pivot_price=pivot,
            distance_to_pivot_pct=distance_to_pivot_pct,
            overall_score=pattern.overall_score,
            num_contractions=pattern.num_contractions,
            final_contraction_pct=pattern.final_contraction_pct,
            prior_advance_pct=pattern.prior_uptrend.advance_pct,
            base_low=base_low,
            position_in_base=position,
            volume_vs_avg=volume_vs_avg
        )

    def get_best_entry(self, pattern: VCPPatternV5) -> Optional[EntrySignal]:
        """Get the best available entry signal."""
        if not pattern.entry_signals:
            return None

        allowed = [s for s in pattern.entry_signals
                   if s.entry_type in self.allowed_entry_types]

        if not allowed:
            return None

        priority = {
            EntryType.LOW_CHEAT: 1,
            EntryType.CHEAT: 2,
            EntryType.HANDLE: 3,
            EntryType.PIVOT_BREAKOUT: 4
        }

        allowed.sort(key=lambda x: priority.get(x.entry_type, 5))
        return allowed[0]

    def simulate_trade(
        self,
        df: pd.DataFrame,
        entry_idx: int,
        entry_price: float,
        stop_loss: float,
        target_price: float
    ) -> Tuple[Optional[str], Optional[float], Optional[str], int, float, float]:
        """Simulate a trade from entry to exit."""
        if entry_idx >= len(df) - 1:
            return None, None, 'open', 0, 0, 0

        max_price = entry_price
        trailing_stop = None
        max_gain = 0
        max_dd = 0

        for i in range(entry_idx + 1, min(entry_idx + self.max_hold_days + 1, len(df))):
            row = df.iloc[i]
            days_held = i - entry_idx

            if row['High'] > max_price:
                max_price = row['High']
                current_gain = (max_price - entry_price) / entry_price * 100
                max_gain = max(max_gain, current_gain)

                if self.use_trailing_stop and current_gain >= self.trailing_activation_pct:
                    trailing_stop = max_price * (1 - self.trailing_stop_pct / 100)

            current_dd = (entry_price - row['Low']) / entry_price * 100
            max_dd = max(max_dd, current_dd)

            effective_stop = stop_loss
            if trailing_stop and trailing_stop > stop_loss:
                effective_stop = trailing_stop

            if row['Low'] <= effective_stop:
                exit_price = effective_stop
                exit_reason = 'trailing_stop' if trailing_stop and trailing_stop > stop_loss else 'stop'
                return row.name.strftime('%Y-%m-%d'), exit_price, exit_reason, days_held, max_gain, max_dd

            if not self.use_trailing_stop or trailing_stop is None:
                if row['High'] >= target_price:
                    return row.name.strftime('%Y-%m-%d'), target_price, 'target', days_held, max_gain, max_dd

        last_row = df.iloc[min(entry_idx + self.max_hold_days, len(df) - 1)]
        days_held = min(self.max_hold_days, len(df) - entry_idx - 1)
        return last_row.name.strftime('%Y-%m-%d'), last_row['Close'], 'time', days_held, max_gain, max_dd

    def backtest_symbol(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        spy_df: pd.DataFrame = None
    ) -> Tuple[List[ContractionSequence], List[PreAlertSequence], List[Trade]]:
        """Run backtest on a single symbol."""
        contraction_sequences = []
        pre_alert_sequences = []
        trades = []

        try:
            df = self.cache.get_data(symbol, period='5y')
            if df is None or len(df) < 300:
                return contraction_sequences, pre_alert_sequences, trades

            df = df[start_date:end_date]
            if len(df) < 150:
                return contraction_sequences, pre_alert_sequences, trades

            df.name = symbol

            lookback = 120
            last_trade_exit_idx = -1

            # Track active sequences
            active_contraction_seq: Optional[Tuple[ContractionSequence, int, int, int]] = None  # (seq, first_idx, last_idx, num_contractions)
            active_pre_alert_seq: Optional[Tuple[PreAlertSequence, int, int]] = None  # (seq, first_idx, last_idx)

            for i in range(lookback + 50, len(df) - 5):
                if i <= last_trade_exit_idx:
                    continue

                df_to_date = df.iloc[:i+1]
                pattern = self.detector.analyze_pattern_v5(df_to_date, lookback_days=lookback, spy_df=spy_df)

                current_date = df.iloc[i].name.strftime('%Y-%m-%d')

                # Skip if pattern invalid (but check for sequence expiry first)
                if pattern is None or not pattern.is_valid:
                    # Check for expired sequences
                    if active_contraction_seq is not None:
                        seq, first_idx, last_idx, _ = active_contraction_seq
                        if i - last_idx > self.max_contraction_alert_days:
                            contraction_sequences.append(seq)
                            active_contraction_seq = None
                    if active_pre_alert_seq is not None:
                        seq, first_idx, last_idx = active_pre_alert_seq
                        if i - last_idx > self.max_pre_alert_days:
                            pre_alert_sequences.append(seq)
                            active_pre_alert_seq = None
                    continue

                # Get previous contraction count for comparison
                prev_num_contractions = 0
                if active_contraction_seq is not None:
                    prev_num_contractions = active_contraction_seq[3]

                # Check for trade entry signal first (highest priority)
                entry_signal = self.get_best_entry(pattern) if pattern.overall_score >= self.min_overall_score else None

                if entry_signal:
                    # Trade alert triggered
                    entry_row = df.iloc[i]
                    entry_price = entry_signal.entry_price
                    stop_loss = entry_signal.stop_loss

                    risk = entry_price - stop_loss
                    target_price = entry_price + (risk * self.target_rr)
                    risk_pct = (risk / entry_price) * 100

                    exit_date, exit_price, exit_reason, days_held, max_gain, max_dd = self.simulate_trade(
                        df, i, entry_price, stop_loss, target_price
                    )

                    pnl_pct = (exit_price - entry_price) / entry_price * 100 if exit_price else 0

                    # Check if we had contraction alerts for this
                    had_contraction_alert = False
                    first_contraction_alert_date = None
                    num_contraction_alerts = 0
                    max_contractions_seen = 0
                    days_from_first_contraction = None

                    if active_contraction_seq is not None:
                        seq, first_idx, last_idx, num_c = active_contraction_seq
                        days_since_last = i - last_idx
                        if days_since_last <= self.max_contraction_alert_days:
                            had_contraction_alert = True
                            first_contraction_alert_date = seq.first_alert_date
                            num_contraction_alerts = seq.num_alerts
                            max_contractions_seen = seq.max_contractions
                            days_from_first_contraction = i - first_idx
                            seq.converted_to_trade = True
                            seq.trade_date = current_date
                            for ca in seq.alerts:
                                ca.converted_to_trade = True
                                ca.trade_date = current_date
                        contraction_sequences.append(seq)
                        active_contraction_seq = None

                    # Check if we had pre-alerts for this
                    had_pre_alert = False
                    first_pre_alert_date = None
                    last_pre_alert_date = None
                    num_pre_alerts = 0
                    days_from_first_pre_alert = None

                    if active_pre_alert_seq is not None:
                        seq, first_idx, last_idx = active_pre_alert_seq
                        days_since_last = i - last_idx
                        if days_since_last <= self.max_pre_alert_days:
                            had_pre_alert = True
                            first_pre_alert_date = seq.first_alert_date
                            last_pre_alert_date = seq.last_alert.alert_date if seq.last_alert else None
                            num_pre_alerts = seq.num_alerts
                            days_from_first_pre_alert = i - first_idx
                            seq.converted_to_trade = True
                            seq.trade_date = current_date
                            for pa in seq.alerts:
                                pa.converted_to_trade = True
                                pa.trade_date = current_date
                                pa.trade_entry_type = entry_signal.entry_type.value
                        pre_alert_sequences.append(seq)
                        active_pre_alert_seq = None

                    trade = Trade(
                        symbol=symbol,
                        first_contraction_alert_date=first_contraction_alert_date,
                        num_contraction_alerts=num_contraction_alerts,
                        max_contractions_seen=max_contractions_seen,
                        first_pre_alert_date=first_pre_alert_date,
                        last_pre_alert_date=last_pre_alert_date,
                        num_pre_alerts=num_pre_alerts,
                        signal_date=current_date,
                        entry_date=current_date,
                        entry_type=entry_signal.entry_type.value,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        target_price=target_price,
                        exit_date=exit_date,
                        exit_price=exit_price,
                        exit_reason=exit_reason,
                        pnl_pct=pnl_pct,
                        days_held=days_held,
                        risk_pct=risk_pct,
                        overall_score=pattern.overall_score,
                        had_contraction_alert=had_contraction_alert,
                        had_pre_alert=had_pre_alert,
                        days_from_first_contraction_alert=days_from_first_contraction,
                        days_from_first_pre_alert=days_from_first_pre_alert
                    )

                    trades.append(trade)

                    if days_held:
                        last_trade_exit_idx = i + days_held
                    else:
                        last_trade_exit_idx = i + 1

                else:
                    # No entry signal yet - check for contraction alert first
                    contraction_alert = self.check_contraction_alert_conditions(
                        df_to_date, pattern, prev_num_contractions
                    )

                    if contraction_alert:
                        contraction_alert.symbol = symbol

                        if active_contraction_seq is None:
                            # Start new contraction sequence
                            seq = ContractionSequence(
                                symbol=symbol,
                                first_alert_date=current_date,
                                alerts=[contraction_alert]
                            )
                            active_contraction_seq = (seq, i, i, pattern.num_contractions)
                        else:
                            # Add to existing sequence
                            seq, first_idx, last_idx, _ = active_contraction_seq
                            seq.alerts.append(contraction_alert)
                            active_contraction_seq = (seq, first_idx, i, pattern.num_contractions)

                    # Also check for pre-alert (closer to breakout)
                    if pattern.overall_score >= self.min_overall_score:
                        pre_alert = self.check_pre_alert_conditions(df_to_date, pattern)

                        if pre_alert:
                            pre_alert.symbol = symbol

                            # Link contraction to pre-alert if exists
                            if active_contraction_seq is not None:
                                cseq, _, _, _ = active_contraction_seq
                                if not cseq.converted_to_pre_alert:
                                    cseq.converted_to_pre_alert = True
                                    cseq.pre_alert_date = current_date
                                    for ca in cseq.alerts:
                                        ca.converted_to_pre_alert = True
                                        ca.pre_alert_date = current_date

                            if active_pre_alert_seq is None:
                                # Start new pre-alert sequence
                                pre_alert.alert_sequence = 1
                                seq = PreAlertSequence(
                                    symbol=symbol,
                                    first_alert_date=current_date,
                                    alerts=[pre_alert]
                                )
                                active_pre_alert_seq = (seq, i, i)
                            else:
                                # Add to existing sequence
                                seq, first_idx, last_idx = active_pre_alert_seq
                                pre_alert.alert_sequence = seq.num_alerts + 1
                                seq.alerts.append(pre_alert)
                                active_pre_alert_seq = (seq, first_idx, i)

                # Check for expired sequences
                if active_contraction_seq is not None:
                    seq, first_idx, last_idx, _ = active_contraction_seq
                    if i - last_idx > self.max_contraction_alert_days:
                        contraction_sequences.append(seq)
                        active_contraction_seq = None

                if active_pre_alert_seq is not None:
                    seq, first_idx, last_idx = active_pre_alert_seq
                    if i - last_idx > self.max_pre_alert_days:
                        # Sequence expired without converting
                        pre_alert_sequences.append(seq)
                        active_pre_alert_seq = None

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")

        # Don't forget any remaining active sequences
        if active_contraction_seq is not None:
            contraction_sequences.append(active_contraction_seq[0])
        if active_pre_alert_seq is not None:
            pre_alert_sequences.append(active_pre_alert_seq[0])

        return contraction_sequences, pre_alert_sequences, trades

    def run_backtest(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        max_workers: int = 4
    ) -> BacktestResult:
        """Run backtest across multiple symbols."""
        logger.info(f"Starting three-stage alert backtest: {len(symbols)} symbols")

        spy_df = self.cache.get_data('SPY', period='5y')
        all_contraction_sequences = []
        all_pre_alert_sequences = []
        all_trades = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.backtest_symbol, symbol, start_date, end_date, spy_df): symbol
                for symbol in symbols
            }

            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    contraction_seqs, pre_alert_seqs, trades = future.result()
                    all_contraction_sequences.extend(contraction_seqs)
                    all_pre_alert_sequences.extend(pre_alert_seqs)
                    all_trades.extend(trades)
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")

        # Contraction sequence statistics
        total_contraction_seqs = len(all_contraction_sequences)
        contraction_to_pre_alert = len([s for s in all_contraction_sequences if s.converted_to_pre_alert])
        contraction_to_trade = len([s for s in all_contraction_sequences if s.converted_to_trade])
        contraction_to_pre_alert_rate = contraction_to_pre_alert / total_contraction_seqs * 100 if total_contraction_seqs > 0 else 0
        contraction_to_trade_rate = contraction_to_trade / total_contraction_seqs * 100 if total_contraction_seqs > 0 else 0
        total_contraction_alerts = sum(s.num_alerts for s in all_contraction_sequences)
        avg_contraction_alerts = total_contraction_alerts / total_contraction_seqs if total_contraction_seqs > 0 else 0

        # Pre-alert sequence statistics
        total_pre_alert_seqs = len(all_pre_alert_sequences)
        pre_alert_to_trade = len([s for s in all_pre_alert_sequences if s.converted_to_trade])
        pre_alert_to_trade_rate = pre_alert_to_trade / total_pre_alert_seqs * 100 if total_pre_alert_seqs > 0 else 0
        total_pre_alerts = sum(s.num_alerts for s in all_pre_alert_sequences)
        avg_pre_alerts = total_pre_alerts / total_pre_alert_seqs if total_pre_alert_seqs > 0 else 0

        # Trade statistics by alert type
        total_trades = len(all_trades)
        trades_with_contraction = len([t for t in all_trades if t.had_contraction_alert])
        trades_with_pre_alert = len([t for t in all_trades if t.had_pre_alert])
        trades_with_both = len([t for t in all_trades if t.had_contraction_alert and t.had_pre_alert])
        trades_with_none = len([t for t in all_trades if not t.had_contraction_alert and not t.had_pre_alert])

        # Win rates by alert type
        with_ca = [t for t in all_trades if t.had_contraction_alert]
        with_pa = [t for t in all_trades if t.had_pre_alert]
        with_both = [t for t in all_trades if t.had_contraction_alert and t.had_pre_alert]
        with_none = [t for t in all_trades if not t.had_contraction_alert and not t.had_pre_alert]

        win_rate_ca = len([t for t in with_ca if t.pnl_pct and t.pnl_pct > 0]) / len(with_ca) * 100 if with_ca else 0
        win_rate_pa = len([t for t in with_pa if t.pnl_pct and t.pnl_pct > 0]) / len(with_pa) * 100 if with_pa else 0
        win_rate_both = len([t for t in with_both if t.pnl_pct and t.pnl_pct > 0]) / len(with_both) * 100 if with_both else 0
        win_rate_none = len([t for t in with_none if t.pnl_pct and t.pnl_pct > 0]) / len(with_none) * 100 if with_none else 0

        # Average days from alerts to trade
        days_contraction_list = [t.days_from_first_contraction_alert for t in all_trades if t.days_from_first_contraction_alert]
        days_pre_alert_list = [t.days_from_first_pre_alert for t in all_trades if t.days_from_first_pre_alert]
        avg_days_contraction = np.mean(days_contraction_list) if days_contraction_list else 0
        avg_days_pre_alert = np.mean(days_pre_alert_list) if days_pre_alert_list else 0

        # Overall P&L
        winning = [t for t in all_trades if t.pnl_pct and t.pnl_pct > 0]
        losing = [t for t in all_trades if t.pnl_pct and t.pnl_pct < 0]
        total_wins = sum(t.pnl_pct for t in winning) if winning else 0
        total_losses = abs(sum(t.pnl_pct for t in losing)) if losing else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        total_return = sum(t.pnl_pct for t in all_trades if t.pnl_pct)

        return BacktestResult(
            total_contraction_sequences=total_contraction_seqs,
            total_contraction_alerts=total_contraction_alerts,
            contraction_to_pre_alert_rate=contraction_to_pre_alert_rate,
            contraction_to_trade_rate=contraction_to_trade_rate,
            avg_contraction_alerts_per_seq=avg_contraction_alerts,
            total_pre_alert_sequences=total_pre_alert_seqs,
            total_pre_alerts=total_pre_alerts,
            pre_alert_to_trade_rate=pre_alert_to_trade_rate,
            avg_pre_alerts_per_seq=avg_pre_alerts,
            total_trades=total_trades,
            trades_with_contraction_alert=trades_with_contraction,
            trades_with_pre_alert=trades_with_pre_alert,
            trades_with_both_alerts=trades_with_both,
            trades_with_no_alerts=trades_with_none,
            win_rate_with_contraction_alert=win_rate_ca,
            win_rate_with_pre_alert=win_rate_pa,
            win_rate_with_both=win_rate_both,
            win_rate_with_none=win_rate_none,
            avg_days_contraction_to_trade=avg_days_contraction,
            avg_days_pre_alert_to_trade=avg_days_pre_alert,
            profit_factor=profit_factor,
            total_return_pct=total_return,
            contraction_sequences=all_contraction_sequences,
            pre_alert_sequences=all_pre_alert_sequences,
            trades=all_trades
        )


def generate_sample_charts(result: BacktestResult, output_dir: str):
    """Generate sample charts for dual-alert trades."""
    os.makedirs(output_dir, exist_ok=True)

    if not result.trades:
        return

    cache = DataCache()
    detector = VCPDetectorV5()

    # Select trades with and without pre-alerts
    with_pa = [t for t in result.trades if t.had_pre_alert and t.pnl_pct]
    without_pa = [t for t in result.trades if not t.had_pre_alert and t.pnl_pct]

    # Get winners and losers from each group
    samples = []

    # With pre-alert samples
    with_pa_winners = sorted([t for t in with_pa if t.pnl_pct > 0], key=lambda x: -x.pnl_pct)[:3]
    with_pa_losers = sorted([t for t in with_pa if t.pnl_pct < 0], key=lambda x: x.pnl_pct)[:2]

    # Without pre-alert samples
    without_pa_winners = sorted([t for t in without_pa if t.pnl_pct > 0], key=lambda x: -x.pnl_pct)[:3]
    without_pa_losers = sorted([t for t in without_pa if t.pnl_pct < 0], key=lambda x: x.pnl_pct)[:2]

    samples = with_pa_winners + with_pa_losers + without_pa_winners + without_pa_losers

    for idx, trade in enumerate(samples):
        try:
            generate_trade_chart(trade, cache, detector, output_dir, idx + 1)
        except Exception as e:
            logger.error(f"Error generating chart for {trade.symbol}: {e}")


def generate_trade_chart(trade: Trade, cache: DataCache, detector: VCPDetectorV5,
                         output_dir: str, chart_num: int):
    """Generate a single trade chart showing three-stage alerts and trade."""
    symbol = trade.symbol
    signal_date = trade.signal_date
    first_contraction_alert_date = trade.first_contraction_alert_date
    num_contraction_alerts = trade.num_contraction_alerts
    first_pre_alert_date = trade.first_pre_alert_date
    last_pre_alert_date = trade.last_pre_alert_date
    num_pre_alerts = trade.num_pre_alerts
    exit_date = trade.exit_date
    pnl_pct = trade.pnl_pct

    df = cache.get_data(symbol, period='5y')
    if df is None:
        return

    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    signal_dt = pd.Timestamp(signal_date)
    exit_dt = pd.Timestamp(exit_date) if exit_date else signal_dt
    first_contraction_dt = pd.Timestamp(first_contraction_alert_date) if first_contraction_alert_date else None

    # Extend display start if we have contraction alerts
    if first_contraction_dt:
        display_start = min(signal_dt - timedelta(days=220), first_contraction_dt - timedelta(days=30))
    else:
        display_start = signal_dt - timedelta(days=220)
    display_end = exit_dt + timedelta(days=5)
    df_display = df[(df.index >= display_start) & (df.index <= display_end)]

    if len(df_display) < 20:
        return

    df_to_signal = df[df.index <= signal_dt]
    pattern = detector.analyze_pattern_v5(df_to_signal, lookback_days=120)

    fig, (ax_price, ax_volume) = plt.subplots(2, 1, figsize=(18, 11),
                                               gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    dates = df_display.index.tolist()
    date_to_pos = {date: i for i, date in enumerate(dates)}

    is_winner = pnl_pct > 0
    outcome_str = "WINNER" if is_winner else "LOSER"
    outcome_color = '#4CAF50' if is_winner else '#f44336'

    entry_type = trade.entry_type.upper().replace('_', ' ')

    # Build alert summary string
    alert_parts = []
    if num_contraction_alerts > 0:
        alert_parts.append(f"Contr: {num_contraction_alerts}")
    if num_pre_alerts > 0:
        alert_parts.append(f"Pre: {num_pre_alerts}")
    alert_str = " | ".join(alert_parts) if alert_parts else "No Prior Alerts"

    fig.suptitle(
        f'{symbol} - {entry_type} Entry - {outcome_str}\n'
        f'P&L: {pnl_pct:+.1f}% | Score: {trade.overall_score:.0f} | {alert_str}',
        fontsize=14, fontweight='bold', color=outcome_color
    )

    # Plot candlesticks
    for i, (date, row) in enumerate(df_display.iterrows()):
        if date not in date_to_pos:
            continue
        pos = date_to_pos[date]
        color = '#26a69a' if row['Close'] >= row['Open'] else '#ef5350'
        ax_price.plot([pos, pos], [row['Low'], row['High']], color=color, linewidth=0.8)
        body_bottom = min(row['Open'], row['Close'])
        body_height = abs(row['Close'] - row['Open'])
        rect = Rectangle((pos - 0.35, body_bottom), 0.7, body_height,
                         facecolor=color, edgecolor=color, linewidth=0.5)
        ax_price.add_patch(rect)

    # Plot MAs
    df_display = df_display.copy()
    df_display['MA50'] = df['Close'].rolling(50).mean()
    df_display['MA150'] = df['Close'].rolling(150).mean()
    df_display['MA200'] = df['Close'].rolling(200).mean()

    ma_positions = list(range(len(df_display)))
    ax_price.plot(ma_positions, df_display['MA50'].values, color='blue', linewidth=1, alpha=0.7, label='50 MA')
    ax_price.plot(ma_positions, df_display['MA150'].values, color='orange', linewidth=1, alpha=0.7, label='150 MA')
    ax_price.plot(ma_positions, df_display['MA200'].values, color='red', linewidth=1, alpha=0.7, label='200 MA')

    # Plot VCP contractions
    contraction_colors = ['#9C27B0', '#E91E63', '#FF5722', '#795548']
    if pattern and pattern.contractions:
        for c_idx, contraction in enumerate(pattern.contractions):
            color = contraction_colors[c_idx % len(contraction_colors)]
            high_date = contraction.swing_high.date
            low_date = contraction.swing_low.date

            if high_date in date_to_pos and low_date in date_to_pos:
                high_pos = date_to_pos[high_date]
                low_pos = date_to_pos[low_date]

                ax_price.plot([high_pos, low_pos],
                            [contraction.swing_high.price, contraction.swing_low.price],
                            color=color, linewidth=2.5, linestyle='-', marker='v', markersize=8, zorder=8)

                mid_pos = (high_pos + low_pos) / 2
                mid_price = (contraction.swing_high.price + contraction.swing_low.price) / 2
                ax_price.annotate(f'C{c_idx+1}\n{contraction.range_pct:.1f}%',
                                xy=(mid_pos, mid_price), fontsize=9, fontweight='bold',
                                color='white', bbox=dict(boxstyle='round,pad=0.3',
                                facecolor=color, alpha=0.9), ha='center', va='center', zorder=10)

    # Entry/exit markers
    entry_price = trade.entry_price
    exit_price = trade.exit_price
    stop_loss = trade.stop_loss

    first_contraction_pos = None
    first_pre_alert_pos = None
    last_pre_alert_pos = None
    signal_pos = None
    exit_pos = None

    for date in dates:
        date_str = date.strftime('%Y-%m-%d')
        if first_contraction_alert_date and date_str == first_contraction_alert_date:
            first_contraction_pos = date_to_pos[date]
        if first_pre_alert_date and date_str == first_pre_alert_date:
            first_pre_alert_pos = date_to_pos[date]
        if last_pre_alert_date and date_str == last_pre_alert_date:
            last_pre_alert_pos = date_to_pos[date]
        if date_str == signal_date:
            signal_pos = date_to_pos[date]
        if exit_date and date_str == exit_date:
            exit_pos = date_to_pos[date]

    # Contraction alert marker (purple diamond) - earliest signal
    if first_contraction_pos is not None:
        first_contraction_price = df_display.iloc[first_contraction_pos]['Close']
        ax_price.scatter([first_contraction_pos], [first_contraction_price], marker='D', s=200,
                        color='#9C27B0', edgecolor='black', linewidth=1.5, zorder=14,
                        label=f'1st Contr Alert ({num_contraction_alerts})')
        ax_price.axvline(x=first_contraction_pos, color='#9C27B0', linestyle='--', alpha=0.4, linewidth=1)

    # Pre-alert markers (yellow stars) - show first and last if sequence
    if first_pre_alert_pos is not None:
        first_pre_alert_price = df_display.iloc[first_pre_alert_pos]['Close']
        ax_price.scatter([first_pre_alert_pos], [first_pre_alert_price], marker='*', s=300,
                        color='#FFC107', edgecolor='black', linewidth=1.5, zorder=16,
                        label=f'1st Pre-Alert {first_pre_alert_date}')
        ax_price.axvline(x=first_pre_alert_pos, color='#FFC107', linestyle='--', alpha=0.5, linewidth=1)

        # If there are multiple pre-alerts, shade the pre-alert period
        if last_pre_alert_pos is not None and last_pre_alert_pos != first_pre_alert_pos:
            last_pre_alert_price = df_display.iloc[last_pre_alert_pos]['Close']
            ax_price.scatter([last_pre_alert_pos], [last_pre_alert_price], marker='*', s=200,
                            color='#FF9800', edgecolor='black', linewidth=1.5, zorder=16,
                            label=f'Last Pre-Alert ({num_pre_alerts} total)')
            ax_price.axvspan(first_pre_alert_pos, last_pre_alert_pos, alpha=0.1, color='#FFC107')

    # Trade entry marker (blue triangle)
    if signal_pos is not None:
        ax_price.scatter([signal_pos], [entry_price], marker='^', s=200,
                        color='#2196F3', edgecolor='black', linewidth=1.5, zorder=15,
                        label=f'Entry ${entry_price:.2f}')
        ax_price.axhline(y=entry_price, color='#2196F3', linestyle='--', alpha=0.5, linewidth=1)

    # Exit marker
    if exit_pos is not None and exit_price:
        exit_marker = 'o' if is_winner else 'X'
        exit_color = '#4CAF50' if is_winner else '#f44336'
        ax_price.scatter([exit_pos], [exit_price], marker=exit_marker, s=200,
                        color=exit_color, edgecolor='black', linewidth=1.5, zorder=15,
                        label=f'Exit ${exit_price:.2f}')

    ax_price.axhline(y=stop_loss, color='#f44336', linestyle=':', alpha=0.7, linewidth=1.5,
                    label=f'Stop ${stop_loss:.2f}')

    if signal_pos is not None and exit_pos is not None:
        shade_color = '#4CAF50' if is_winner else '#f44336'
        ax_price.axvspan(signal_pos, exit_pos, alpha=0.1, color=shade_color)

    # Volume
    vol_colors = ['#26a69a' if df_display.iloc[i]['Close'] >= df_display.iloc[i]['Open']
                  else '#ef5350' for i in range(len(df_display))]
    ax_volume.bar(range(len(df_display)), df_display['Volume'].values / 1e6,
                  color=vol_colors, alpha=0.7, width=0.8)
    vol_ma = df_display['Volume'].rolling(50).mean() / 1e6
    ax_volume.plot(range(len(df_display)), vol_ma.values, color='blue', linewidth=1.5, label='50-day avg')

    ax_volume.set_ylabel('Volume (M)', fontsize=10)
    ax_volume.legend(loc='upper right', fontsize=8)

    tick_positions = list(range(0, len(dates), max(1, len(dates) // 10)))
    tick_labels = [dates[i].strftime('%Y-%m-%d') for i in tick_positions]
    ax_volume.set_xticks(tick_positions)
    ax_volume.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)

    ax_price.set_ylabel('Price ($)', fontsize=10)
    ax_price.legend(loc='lower left', fontsize=8, ncol=2)
    ax_price.grid(True, alpha=0.3)
    ax_volume.grid(True, alpha=0.3)

    # Info box
    info_lines = []
    if first_contraction_alert_date:
        info_lines.append(f"Contr Alerts: {num_contraction_alerts}")
        info_lines.append(f"1st Contr: {first_contraction_alert_date}")
        if trade.days_from_first_contraction_alert:
            info_lines.append(f"Days Contr→Trade: {trade.days_from_first_contraction_alert}")
    if first_pre_alert_date:
        if num_pre_alerts > 1:
            info_lines.append(f"Pre-Alerts: {num_pre_alerts}")
            info_lines.append(f"First: {first_pre_alert_date}")
            info_lines.append(f"Last: {last_pre_alert_date}")
        else:
            info_lines.append(f"Pre-Alert: {first_pre_alert_date}")
        info_lines.append(f"Days Pre→Trade: {trade.days_from_first_pre_alert}")
    info_lines.append(f"Trade: {signal_date}")
    info_lines.append(f"Exit: {exit_date}")
    info_lines.append(f"Days Held: {trade.days_held}")
    info_lines.append(f"Reason: {trade.exit_reason}")

    info_text = "\n".join(info_lines)
    ax_price.text(0.99, 0.99, info_text, transform=ax_price.transAxes,
                  fontsize=9, verticalalignment='top', horizontalalignment='right',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    plt.tight_layout()

    outcome = "winner" if is_winner else "loser"
    pa_str = "with_prealert" if trade.had_pre_alert else "no_prealert"
    filename = f"{chart_num:02d}_{symbol}_{trade.entry_type}_{pa_str}_{outcome}_{signal_date}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def get_test_universe() -> List[str]:
    """Get list of stocks for backtesting."""
    return [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
        'AMD', 'AVGO', 'QCOM', 'INTC', 'MU', 'AMAT', 'LRCX', 'KLAC',
        'CRM', 'ADBE', 'NOW', 'ORCL', 'INTU', 'PANW', 'CRWD', 'ZS',
        'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'AXP',
        'UNH', 'JNJ', 'LLY', 'PFE', 'ABBV', 'MRK', 'TMO', 'DHR',
        'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'COST', 'WMT', 'PG',
        'CAT', 'DE', 'GE', 'HON', 'UNP', 'UPS', 'LMT', 'RTX',
        'XOM', 'CVX', 'COP', 'SLB', 'EOG',
        'NFLX', 'PYPL', 'SHOP', 'UBER', 'ABNB', 'COIN'
    ]


def main():
    """Run three-stage alert backtest."""
    print("="*70)
    print("VCP Three-Stage Alert System Backtest")
    print("="*70)
    print()
    print("Three-stage alert system:")
    print("1. CONTRACTION ALERT: New qualified contraction detected - Build watchlist")
    print("2. PRE-ALERT: Price within 3% of pivot - Watch closely")
    print("3. TRADE ALERT: Entry signal triggered - Execute")
    print()

    symbols = get_test_universe()
    start_date = '2020-01-01'
    end_date = '2024-12-31'

    engine = ThreeStageAlertBacktest(
        min_contractions_for_alert=2,
        pre_alert_distance_pct=3.0,
        min_overall_score=70.0,
        max_contraction_alert_days=20,
        max_pre_alert_days=5
    )

    result = engine.run_backtest(symbols, start_date, end_date)

    # Stage 1: Contraction Alert Statistics
    print(f"\n{'='*70}")
    print("STAGE 1: CONTRACTION ALERT STATISTICS")
    print('='*70)
    print(f"Total Contraction Sequences:      {result.total_contraction_sequences}")
    print(f"Total Contraction Alerts:         {result.total_contraction_alerts}")
    print(f"Avg Alerts per Sequence:          {result.avg_contraction_alerts_per_seq:.1f}")
    print(f"Converted to Pre-Alert:           {result.contraction_to_pre_alert_rate:.1f}%")
    print(f"Converted to Trade:               {result.contraction_to_trade_rate:.1f}%")
    print(f"Avg Days Contraction → Trade:     {result.avg_days_contraction_to_trade:.1f}")

    # Stage 2: Pre-Alert Statistics
    print(f"\n{'='*70}")
    print("STAGE 2: PRE-ALERT STATISTICS")
    print('='*70)
    print(f"Total Pre-Alert Sequences:        {result.total_pre_alert_sequences}")
    print(f"Total Pre-Alerts:                 {result.total_pre_alerts}")
    print(f"Avg Alerts per Sequence:          {result.avg_pre_alerts_per_seq:.1f}")
    print(f"Converted to Trade:               {result.pre_alert_to_trade_rate:.1f}%")
    print(f"Avg Days Pre-Alert → Trade:       {result.avg_days_pre_alert_to_trade:.1f}")

    # Stage 3: Trade Statistics
    print(f"\n{'='*70}")
    print("STAGE 3: TRADE STATISTICS")
    print('='*70)
    print(f"Total Trades:                     {result.total_trades}")
    print(f"Trades with Contraction Alert:    {result.trades_with_contraction_alert}")
    print(f"Trades with Pre-Alert:            {result.trades_with_pre_alert}")
    print(f"Trades with BOTH alerts:          {result.trades_with_both_alerts}")
    print(f"Trades with NO prior alerts:      {result.trades_with_no_alerts}")

    # Win Rates by Alert Type
    print(f"\n{'='*70}")
    print("WIN RATES BY ALERT TYPE")
    print('='*70)
    print(f"With Contraction Alert:           {result.win_rate_with_contraction_alert:.1f}%")
    print(f"With Pre-Alert:                   {result.win_rate_with_pre_alert:.1f}%")
    print(f"With BOTH alerts:                 {result.win_rate_with_both:.1f}%")
    print(f"With NO prior alerts:             {result.win_rate_with_none:.1f}%")
    print(f"\nOverall Profit Factor:            {result.profit_factor:.2f}")
    print(f"Total Return:                     {result.total_return_pct:.1f}%")

    # Analysis
    print(f"\n{'='*70}")
    print("ANALYSIS")
    print('='*70)

    # Compare win rates
    best_wr = max(result.win_rate_with_both, result.win_rate_with_contraction_alert,
                  result.win_rate_with_pre_alert, result.win_rate_with_none)
    if result.win_rate_with_both == best_wr and result.trades_with_both_alerts > 0:
        print(f"✓ Trades with BOTH alerts have highest win rate ({result.win_rate_with_both:.1f}%)")
        print("  Full alert sequence provides best trade selection.")
    elif result.win_rate_with_contraction_alert == best_wr and result.trades_with_contraction_alert > 0:
        print(f"✓ Trades with Contraction Alert have highest win rate ({result.win_rate_with_contraction_alert:.1f}%)")
        print("  Early pattern detection adds value.")
    elif result.win_rate_with_pre_alert == best_wr and result.trades_with_pre_alert > 0:
        print(f"○ Trades with Pre-Alert have highest win rate ({result.win_rate_with_pre_alert:.1f}%)")

    if result.contraction_to_trade_rate > 30:
        print(f"✓ Good contraction→trade conversion ({result.contraction_to_trade_rate:.1f}%)")
    else:
        print(f"○ Low contraction→trade rate ({result.contraction_to_trade_rate:.1f}%)")
        print("  Many early patterns don't lead to entries.")

    # Save results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_base = os.path.join(script_dir, '..', 'results', 'three_stage_alert')
    os.makedirs(output_base, exist_ok=True)

    # Save contraction sequences
    if result.contraction_sequences:
        all_contraction_alerts = []
        for seq in result.contraction_sequences:
            for alert in seq.alerts:
                all_contraction_alerts.append(asdict(alert))
        ca_df = pd.DataFrame(all_contraction_alerts)
        ca_df.to_csv(f'{output_base}/contraction_alerts.csv', index=False)

        seq_data = []
        for seq in result.contraction_sequences:
            seq_data.append({
                'symbol': seq.symbol,
                'first_alert_date': seq.first_alert_date,
                'num_alerts': seq.num_alerts,
                'max_contractions': seq.max_contractions,
                'converted_to_pre_alert': seq.converted_to_pre_alert,
                'pre_alert_date': seq.pre_alert_date,
                'converted_to_trade': seq.converted_to_trade,
                'trade_date': seq.trade_date
            })
        seq_df = pd.DataFrame(seq_data)
        seq_df.to_csv(f'{output_base}/contraction_sequences.csv', index=False)

    # Save pre-alert sequences
    if result.pre_alert_sequences:
        all_pre_alerts = []
        for seq in result.pre_alert_sequences:
            for alert in seq.alerts:
                all_pre_alerts.append(asdict(alert))
        pa_df = pd.DataFrame(all_pre_alerts)
        pa_df.to_csv(f'{output_base}/pre_alerts.csv', index=False)

        seq_data = []
        for seq in result.pre_alert_sequences:
            seq_data.append({
                'symbol': seq.symbol,
                'first_alert_date': seq.first_alert_date,
                'num_alerts': seq.num_alerts,
                'converted_to_trade': seq.converted_to_trade,
                'trade_date': seq.trade_date
            })
        seq_df = pd.DataFrame(seq_data)
        seq_df.to_csv(f'{output_base}/pre_alert_sequences.csv', index=False)

    # Save trades
    if result.trades:
        trades_df = pd.DataFrame([asdict(t) for t in result.trades])
        trades_df.to_csv(f'{output_base}/trades.csv', index=False)

    # Save summary
    summary = {
        'total_contraction_sequences': result.total_contraction_sequences,
        'total_contraction_alerts': result.total_contraction_alerts,
        'avg_contraction_alerts_per_seq': result.avg_contraction_alerts_per_seq,
        'contraction_to_pre_alert_rate': result.contraction_to_pre_alert_rate,
        'contraction_to_trade_rate': result.contraction_to_trade_rate,
        'total_pre_alert_sequences': result.total_pre_alert_sequences,
        'total_pre_alerts': result.total_pre_alerts,
        'avg_pre_alerts_per_seq': result.avg_pre_alerts_per_seq,
        'pre_alert_to_trade_rate': result.pre_alert_to_trade_rate,
        'total_trades': result.total_trades,
        'trades_with_contraction_alert': result.trades_with_contraction_alert,
        'trades_with_pre_alert': result.trades_with_pre_alert,
        'trades_with_both_alerts': result.trades_with_both_alerts,
        'trades_with_no_alerts': result.trades_with_no_alerts,
        'win_rate_with_contraction_alert': result.win_rate_with_contraction_alert,
        'win_rate_with_pre_alert': result.win_rate_with_pre_alert,
        'win_rate_with_both': result.win_rate_with_both,
        'win_rate_with_none': result.win_rate_with_none,
        'avg_days_contraction_to_trade': result.avg_days_contraction_to_trade,
        'avg_days_pre_alert_to_trade': result.avg_days_pre_alert_to_trade,
        'profit_factor': result.profit_factor,
        'total_return_pct': result.total_return_pct
    }
    with open(f'{output_base}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Generate sample charts
    print(f"\nGenerating sample charts -> {output_base}/charts")
    generate_sample_charts(result, f'{output_base}/charts')

    print(f"\n\nResults saved to {output_base}/")

    return result


if __name__ == '__main__':
    main()
