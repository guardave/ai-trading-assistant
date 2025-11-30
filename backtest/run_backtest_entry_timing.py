#!/usr/bin/env python3
"""
Backtest Entry Timing Comparison

Compares entry timing approaches:
1. Same-day Close (current implementation) - Enter at EOD close on signal day
2. Next-day Open - Enter at next day's open price (more realistic for alerts)

This helps determine how overnight gaps affect performance.
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
class Trade:
    """Represents a single trade"""
    symbol: str
    entry_type: str
    signal_date: str  # Day pattern was detected
    entry_date: str   # Day of actual entry
    entry_price: float
    signal_close: float  # Close price on signal day (for comparison)
    stop_loss: float
    target_price: float
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl_pct: Optional[float] = None
    days_held: Optional[int] = None
    max_gain_pct: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    risk_pct: Optional[float] = None
    overall_score: Optional[float] = None
    prior_advance_pct: Optional[float] = None
    position_in_base: Optional[str] = None
    slippage_pct: Optional[float] = None  # Gap from signal close to actual entry


@dataclass
class BacktestConfig:
    """Configuration for a backtest run"""
    name: str
    entry_timing: str  # 'same_day_close' or 'next_day_open'
    allowed_entry_types: List[EntryType]
    stop_loss_pct: float = 7.0
    target_rr: float = 3.0
    max_hold_days: int = 60
    use_trailing_stop: bool = True
    trailing_activation_pct: float = 8.0
    trailing_stop_pct: float = 5.0
    min_overall_score: float = 70.0


@dataclass
class BacktestResult:
    """Results from a backtest run"""
    config_name: str
    entry_timing: str
    entry_types: List[str]
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    total_return_pct: float
    avg_days_held: float
    avg_risk_pct: float
    avg_slippage_pct: float
    gap_up_trades: int
    gap_down_trades: int
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

            # Remove timezone
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            df.to_parquet(cache_file)
            self.memory_cache[cache_key] = df
            return df

        except Exception as e:
            logger.warning(f"Error fetching {symbol}: {e}")
            return None


class EntryTimingBacktest:
    """Backtest engine comparing entry timing approaches"""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.detector = VCPDetectorV5()
        self.cache = DataCache()

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

        # Start checking from the day AFTER entry
        for i in range(entry_idx + 1, min(entry_idx + self.config.max_hold_days + 1, len(df))):
            row = df.iloc[i]
            days_held = i - entry_idx

            if row['High'] > max_price:
                max_price = row['High']
                current_gain = (max_price - entry_price) / entry_price * 100
                max_gain = max(max_gain, current_gain)

                if self.config.use_trailing_stop and current_gain >= self.config.trailing_activation_pct:
                    trailing_stop = max_price * (1 - self.config.trailing_stop_pct / 100)

            current_dd = (entry_price - row['Low']) / entry_price * 100
            max_dd = max(max_dd, current_dd)

            effective_stop = stop_loss
            if trailing_stop and trailing_stop > stop_loss:
                effective_stop = trailing_stop

            if row['Low'] <= effective_stop:
                exit_price = effective_stop
                exit_reason = 'trailing_stop' if trailing_stop and trailing_stop > stop_loss else 'stop'
                return row.name.strftime('%Y-%m-%d'), exit_price, exit_reason, days_held, max_gain, max_dd

            if not self.config.use_trailing_stop or trailing_stop is None:
                if row['High'] >= target_price:
                    return row.name.strftime('%Y-%m-%d'), target_price, 'target', days_held, max_gain, max_dd

        last_row = df.iloc[min(entry_idx + self.config.max_hold_days, len(df) - 1)]
        days_held = min(self.config.max_hold_days, len(df) - entry_idx - 1)
        return last_row.name.strftime('%Y-%m-%d'), last_row['Close'], 'time', days_held, max_gain, max_dd

    def get_best_entry(self, pattern: VCPPatternV5) -> Optional[EntrySignal]:
        """Get the best available entry signal based on allowed types."""
        if not pattern.entry_signals:
            return None

        allowed = [s for s in pattern.entry_signals
                   if s.entry_type in self.config.allowed_entry_types]

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

    def backtest_symbol(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        spy_df: pd.DataFrame = None
    ) -> List[Trade]:
        """Run backtest on a single symbol."""
        trades = []

        try:
            df = self.cache.get_data(symbol, period='5y')
            if df is None or len(df) < 300:
                return trades

            df = df[start_date:end_date]
            if len(df) < 150:
                return trades

            lookback = 120
            last_trade_exit_idx = -1

            for i in range(lookback + 50, len(df) - 5):
                if i <= last_trade_exit_idx:
                    continue

                df_to_date = df.iloc[:i+1]

                pattern = self.detector.analyze_pattern_v5(df_to_date, lookback_days=lookback, spy_df=spy_df)

                if pattern is None or not pattern.is_valid:
                    continue

                if pattern.overall_score < self.config.min_overall_score:
                    continue

                entry_signal = self.get_best_entry(pattern)
                if not entry_signal:
                    continue

                signal_row = df.iloc[i]
                signal_date = signal_row.name.strftime('%Y-%m-%d')
                signal_close = signal_row['Close']

                # Determine entry timing
                if self.config.entry_timing == 'same_day_close':
                    # Current implementation: enter at signal day's close
                    entry_idx = i
                    entry_price = signal_close
                    entry_date = signal_date
                    slippage_pct = 0.0

                elif self.config.entry_timing == 'next_day_open':
                    # More realistic: enter at next day's open
                    if i + 1 >= len(df):
                        continue  # No next day data

                    entry_idx = i + 1
                    next_day = df.iloc[i + 1]
                    entry_price = next_day['Open']
                    entry_date = next_day.name.strftime('%Y-%m-%d')

                    # Calculate slippage (gap from signal close to actual entry)
                    slippage_pct = (entry_price - signal_close) / signal_close * 100

                else:
                    raise ValueError(f"Unknown entry timing: {self.config.entry_timing}")

                # Recalculate stop and target based on actual entry price
                # Use original signal's stop loss level (pattern-based)
                stop_loss = entry_signal.stop_loss

                # Recalculate risk and target based on actual entry
                risk = entry_price - stop_loss
                if risk <= 0:
                    # Entry price is at or below stop loss (gap down too large)
                    continue

                target_price = entry_price + (risk * self.config.target_rr)
                risk_pct = (risk / entry_price) * 100

                # Simulate trade
                exit_date, exit_price, exit_reason, days_held, max_gain, max_dd = self.simulate_trade(
                    df, entry_idx, entry_price, stop_loss, target_price
                )

                pnl_pct = (exit_price - entry_price) / entry_price * 100 if exit_price else 0

                trade = Trade(
                    symbol=symbol,
                    entry_type=entry_signal.entry_type.value,
                    signal_date=signal_date,
                    entry_date=entry_date,
                    entry_price=entry_price,
                    signal_close=signal_close,
                    stop_loss=stop_loss,
                    target_price=target_price,
                    exit_date=exit_date,
                    exit_price=exit_price,
                    exit_reason=exit_reason,
                    pnl_pct=pnl_pct,
                    days_held=days_held,
                    max_gain_pct=max_gain,
                    max_drawdown_pct=max_dd,
                    risk_pct=risk_pct,
                    overall_score=pattern.overall_score,
                    prior_advance_pct=pattern.prior_uptrend.advance_pct,
                    position_in_base=entry_signal.position_in_base,
                    slippage_pct=slippage_pct
                )

                trades.append(trade)

                if days_held:
                    last_trade_exit_idx = entry_idx + days_held
                else:
                    last_trade_exit_idx = entry_idx + 1

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")

        return trades

    def run_backtest(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        max_workers: int = 4
    ) -> BacktestResult:
        """Run backtest across multiple symbols."""
        logger.info(f"Starting backtest '{self.config.name}': {len(symbols)} symbols")

        spy_df = self.cache.get_data('SPY', period='5y')
        all_trades = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.backtest_symbol, symbol, start_date, end_date, spy_df): symbol
                for symbol in symbols
            }

            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    trades = future.result()
                    all_trades.extend(trades)
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")

        if not all_trades:
            return BacktestResult(
                config_name=self.config.name,
                entry_timing=self.config.entry_timing,
                entry_types=[et.value for et in self.config.allowed_entry_types],
                total_trades=0, winning_trades=0, losing_trades=0,
                win_rate=0, avg_win_pct=0, avg_loss_pct=0,
                profit_factor=0, total_return_pct=0,
                avg_days_held=0, avg_risk_pct=0, avg_slippage_pct=0,
                gap_up_trades=0, gap_down_trades=0, trades=[]
            )

        winning = [t for t in all_trades if t.pnl_pct and t.pnl_pct > 0]
        losing = [t for t in all_trades if t.pnl_pct and t.pnl_pct < 0]

        total_wins = sum(t.pnl_pct for t in winning) if winning else 0
        total_losses = abs(sum(t.pnl_pct for t in losing)) if losing else 0

        # Gap statistics
        gap_up = len([t for t in all_trades if t.slippage_pct and t.slippage_pct > 0.5])
        gap_down = len([t for t in all_trades if t.slippage_pct and t.slippage_pct < -0.5])

        return BacktestResult(
            config_name=self.config.name,
            entry_timing=self.config.entry_timing,
            entry_types=[et.value for et in self.config.allowed_entry_types],
            total_trades=len(all_trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=len(winning) / len(all_trades) * 100 if all_trades else 0,
            avg_win_pct=np.mean([t.pnl_pct for t in winning]) if winning else 0,
            avg_loss_pct=np.mean([t.pnl_pct for t in losing]) if losing else 0,
            profit_factor=total_wins / total_losses if total_losses > 0 else float('inf'),
            total_return_pct=sum(t.pnl_pct for t in all_trades if t.pnl_pct),
            avg_days_held=np.mean([t.days_held for t in all_trades if t.days_held]) if all_trades else 0,
            avg_risk_pct=np.mean([t.risk_pct for t in all_trades if t.risk_pct]) if all_trades else 0,
            avg_slippage_pct=np.mean([t.slippage_pct for t in all_trades if t.slippage_pct is not None]) if all_trades else 0,
            gap_up_trades=gap_up,
            gap_down_trades=gap_down,
            trades=all_trades
        )


def generate_sample_charts(result: BacktestResult, output_dir: str, max_charts: int = 10):
    """Generate sample charts for a backtest result."""
    os.makedirs(output_dir, exist_ok=True)

    if not result.trades:
        return

    trades_df = pd.DataFrame([asdict(t) for t in result.trades])

    # Select winners and losers
    winners = trades_df[trades_df['pnl_pct'] > 0].nlargest(min(5, len(trades_df[trades_df['pnl_pct'] > 0])), 'pnl_pct')
    losers = trades_df[trades_df['pnl_pct'] < 0].nsmallest(min(5, len(trades_df[trades_df['pnl_pct'] < 0])), 'pnl_pct')

    samples = pd.concat([winners, losers])

    cache = DataCache()
    detector = VCPDetectorV5()

    for idx, (_, trade) in enumerate(samples.iterrows()):
        try:
            generate_trade_chart(trade, cache, detector, output_dir, idx + 1, result.config_name, result.entry_timing)
        except Exception as e:
            logger.error(f"Error generating chart for {trade['symbol']}: {e}")


def generate_trade_chart(trade: dict, cache: DataCache, detector: VCPDetectorV5,
                         output_dir: str, chart_num: int, config_name: str, entry_timing: str):
    """Generate a single trade chart."""
    symbol = trade['symbol']
    signal_date = trade['signal_date']
    entry_date = trade['entry_date']
    exit_date = trade['exit_date']
    pnl_pct = trade['pnl_pct']
    slippage_pct = trade['slippage_pct']

    df = cache.get_data(symbol, period='5y')
    if df is None:
        return

    # Remove timezone if present
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    signal_dt = pd.Timestamp(signal_date)
    entry_dt = pd.Timestamp(entry_date)
    exit_dt = pd.Timestamp(exit_date) if exit_date else entry_dt

    # Extended lookback for full MA display
    display_start = signal_dt - timedelta(days=220)
    display_end = exit_dt + timedelta(days=5)
    df_display = df[(df.index >= display_start) & (df.index <= display_end)]

    if len(df_display) < 20:
        return

    # Get pattern for visualization
    df_to_signal = df[df.index <= signal_dt]
    pattern = detector.analyze_pattern_v5(df_to_signal, lookback_days=120)

    # Create figure
    fig, (ax_price, ax_volume) = plt.subplots(2, 1, figsize=(18, 11),
                                               gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    dates = df_display.index.tolist()
    date_to_pos = {date: i for i, date in enumerate(dates)}

    is_winner = pnl_pct > 0
    outcome_str = "WINNER" if is_winner else "LOSER"
    outcome_color = '#4CAF50' if is_winner else '#f44336'

    entry_type = trade['entry_type'].upper().replace('_', ' ')
    timing_str = "EOD Close" if entry_timing == 'same_day_close' else "Next-Day Open"

    fig.suptitle(
        f'{symbol} - {entry_type} Entry ({timing_str}) - {outcome_str}\n'
        f'P&L: {pnl_pct:+.1f}% | Score: {trade["overall_score"]:.0f} | '
        f'Risk: {trade["risk_pct"]:.1f}% | Slippage: {slippage_pct:+.1f}%',
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

                ax_price.scatter([high_pos], [contraction.swing_high.price],
                               marker='^', s=100, color=color, zorder=9)
                ax_price.scatter([low_pos], [contraction.swing_low.price],
                               marker='v', s=100, color=color, zorder=9)

                mid_pos = (high_pos + low_pos) / 2
                mid_price = (contraction.swing_high.price + contraction.swing_low.price) / 2
                ax_price.annotate(f'C{c_idx+1}\n{contraction.range_pct:.1f}%',
                                xy=(mid_pos, mid_price), fontsize=9, fontweight='bold',
                                color='white', bbox=dict(boxstyle='round,pad=0.3',
                                facecolor=color, alpha=0.9), ha='center', va='center', zorder=10)

    # Entry/exit markers
    entry_price = trade['entry_price']
    signal_close = trade['signal_close']
    exit_price = trade['exit_price']
    stop_loss = trade['stop_loss']

    signal_pos = None
    entry_pos = None
    exit_pos = None
    for date in dates:
        date_str = date.strftime('%Y-%m-%d')
        if date_str == signal_date:
            signal_pos = date_to_pos[date]
        if date_str == entry_date:
            entry_pos = date_to_pos[date]
        if exit_date and date_str == exit_date:
            exit_pos = date_to_pos[date]

    # Show signal day close (for next-day open timing)
    if signal_pos is not None and entry_timing == 'next_day_open':
        ax_price.scatter([signal_pos], [signal_close], marker='s', s=100,
                        color='#FFC107', edgecolor='black', linewidth=1.5, zorder=14,
                        label=f'Signal Close ${signal_close:.2f}')
        ax_price.axhline(y=signal_close, color='#FFC107', linestyle='--', alpha=0.3, linewidth=1)

    if entry_pos is not None:
        ax_price.scatter([entry_pos], [entry_price], marker='^', s=200,
                        color='#2196F3', edgecolor='black', linewidth=1.5, zorder=15,
                        label=f'Entry ${entry_price:.2f}')
        ax_price.axhline(y=entry_price, color='#2196F3', linestyle='--', alpha=0.5, linewidth=1)

    if exit_pos is not None and exit_price:
        exit_marker = 'o' if is_winner else 'X'
        exit_color = '#4CAF50' if is_winner else '#f44336'
        ax_price.scatter([exit_pos], [exit_price], marker=exit_marker, s=200,
                        color=exit_color, edgecolor='black', linewidth=1.5, zorder=15,
                        label=f'Exit ${exit_price:.2f}')

    ax_price.axhline(y=stop_loss, color='#f44336', linestyle=':', alpha=0.7, linewidth=1.5,
                    label=f'Stop ${stop_loss:.2f}')

    if entry_pos is not None and exit_pos is not None:
        shade_color = '#4CAF50' if is_winner else '#f44336'
        ax_price.axvspan(entry_pos, exit_pos, alpha=0.1, color=shade_color)

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

    # Rules box
    if pattern:
        rules_lines = [f"ENTRY: {entry_type}", f"TIMING: {timing_str}", "=" * 22]
        rules_lines.append(f"Prior Uptrend: {pattern.prior_uptrend.advance_pct:.1f}%")
        rules_lines.append(f"Score: {pattern.overall_score:.0f}/100")
        rules_lines.append(f"Contractions: {pattern.num_contractions}")
        rules_lines.append(f"Final: {pattern.final_contraction_pct:.1f}%")
        rules_lines.append(f"Risk: {trade['risk_pct']:.1f}%")
        rules_lines.append(f"Slippage: {slippage_pct:+.1f}%")
        rules_lines.append(f"R:R Target: 3:1")

        rules_text = "\n".join(rules_lines)
        ax_price.text(0.01, 0.99, rules_text, transform=ax_price.transAxes,
                      fontsize=8, verticalalignment='top', horizontalalignment='left',
                      fontfamily='monospace',
                      bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95, edgecolor='gray'))

    # Info box
    info_lines = [f"Signal: {signal_date}"]
    if entry_timing == 'next_day_open':
        info_lines.append(f"Entry: {entry_date}")
    info_lines.append(f"Exit: {exit_date}")
    info_lines.append(f"Days: {trade['days_held']}")
    info_lines.append(f"Reason: {trade['exit_reason']}")
    info_text = "\n".join(info_lines)
    ax_price.text(0.99, 0.99, info_text, transform=ax_price.transAxes,
                  fontsize=9, verticalalignment='top', horizontalalignment='right',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    plt.tight_layout()

    outcome = "winner" if is_winner else "loser"
    filename = f"{chart_num:02d}_{symbol}_{trade['entry_type']}_{outcome}_{entry_date}.png"
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
    """Run entry timing comparison backtest."""
    print("="*70)
    print("VCP Entry Timing Comparison Backtest")
    print("="*70)
    print()
    print("Comparing two entry timing approaches:")
    print("1. Same-day Close: Enter at signal day's close price (current)")
    print("2. Next-day Open: Enter at next trading day's open price (realistic)")
    print()

    symbols = get_test_universe()
    start_date = '2020-01-01'
    end_date = '2024-12-31'

    # Use recommended entry types (exclude cheat)
    allowed_entries = [EntryType.LOW_CHEAT, EntryType.HANDLE, EntryType.PIVOT_BREAKOUT]

    configs = [
        BacktestConfig(
            name="same_day_close",
            entry_timing="same_day_close",
            allowed_entry_types=allowed_entries
        ),
        BacktestConfig(
            name="next_day_open",
            entry_timing="next_day_open",
            allowed_entry_types=allowed_entries
        ),
    ]

    results = []
    output_base = '../results/entry_timing'
    os.makedirs(output_base, exist_ok=True)

    for config in configs:
        print(f"\n{'='*70}")
        print(f"Testing: {config.name}")
        print(f"Entry timing: {config.entry_timing}")
        print('='*70)

        engine = EntryTimingBacktest(config)
        result = engine.run_backtest(symbols, start_date, end_date)
        results.append(result)

        print(f"\n--- Results ---")
        print(f"Total Trades: {result.total_trades}")
        print(f"Win Rate: {result.win_rate:.1f}%")
        print(f"Profit Factor: {result.profit_factor:.2f}")
        print(f"Total Return: {result.total_return_pct:.1f}%")
        print(f"Avg Risk: {result.avg_risk_pct:.1f}%")
        print(f"Avg Slippage: {result.avg_slippage_pct:+.2f}%")
        print(f"Gap Up Trades (>0.5%): {result.gap_up_trades}")
        print(f"Gap Down Trades (<-0.5%): {result.gap_down_trades}")

        # Save trades
        if result.trades:
            trades_df = pd.DataFrame([asdict(t) for t in result.trades])
            trades_df.to_csv(f'{output_base}/{config.name}_trades.csv', index=False)

            # Generate sample charts
            chart_dir = f'{output_base}/{config.name}_charts'
            print(f"\nGenerating sample charts -> {chart_dir}")
            generate_sample_charts(result, chart_dir)

    # Create comparison summary
    print(f"\n{'='*70}")
    print("ENTRY TIMING COMPARISON SUMMARY")
    print('='*70)

    summary_data = []
    for r in results:
        summary_data.append({
            'Config': r.config_name,
            'Timing': r.entry_timing,
            'Trades': r.total_trades,
            'Win Rate': f"{r.win_rate:.1f}%",
            'Profit Factor': f"{r.profit_factor:.2f}",
            'Total Return': f"{r.total_return_pct:.1f}%",
            'Avg Win': f"{r.avg_win_pct:.1f}%",
            'Avg Loss': f"{r.avg_loss_pct:.1f}%",
            'Avg Risk': f"{r.avg_risk_pct:.1f}%",
            'Avg Slippage': f"{r.avg_slippage_pct:+.2f}%",
            'Gap Up': r.gap_up_trades,
            'Gap Down': r.gap_down_trades
        })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    # Save summary
    summary_df.to_csv(f'{output_base}/timing_comparison_summary.csv', index=False)

    # Calculate impact
    if len(results) == 2:
        same_day = results[0]
        next_day = results[1]

        print(f"\n{'='*70}")
        print("SLIPPAGE IMPACT ANALYSIS")
        print('='*70)

        return_diff = next_day.total_return_pct - same_day.total_return_pct
        wr_diff = next_day.win_rate - same_day.win_rate
        pf_diff = next_day.profit_factor - same_day.profit_factor

        print(f"Win Rate Impact:      {wr_diff:+.1f}%")
        print(f"Profit Factor Impact: {pf_diff:+.2f}")
        print(f"Total Return Impact:  {return_diff:+.1f}%")
        print(f"Average Overnight Gap: {next_day.avg_slippage_pct:+.2f}%")

        if next_day.avg_slippage_pct > 0:
            print("\n⚠️  Stocks tend to GAP UP after signal day")
            print("   This means next-day-open entries cost more than EOD entries")
        else:
            print("\n✓  Stocks tend to GAP DOWN after signal day")
            print("   This may improve risk/reward for next-day-open entries")

    print(f"\n\nResults saved to {output_base}/")

    return results


if __name__ == '__main__':
    main()
