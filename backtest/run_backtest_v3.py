"""
Backtesting Script v3 - Using Refined VCP Detector

This script uses the improved VCP detection algorithm from vcp_detector.py
which includes:
- Proper swing high/low detection
- "Extend until recovery" contraction identification
- Progressive tightening enforcement
- Consolidation base rule (rejects staircase patterns)
- Volume dry-up validation

Usage:
    python backtest/run_backtest_v3.py
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
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.vcp_detector import VCPDetector, VCPPattern
from backtest.strategy_params import VCP_PARAMS, RISK_PARAMS, RS_PARAMS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Trade:
    """Represents a single trade with tracking"""
    symbol: str
    pattern: str
    entry_date: str
    entry_price: float
    stop_loss: float
    target_price: float
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # 'stop', 'target', 'trailing_stop', 'time', 'open'
    pnl_pct: Optional[float] = None
    days_held: Optional[int] = None
    max_gain_pct: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    rs_rating: Optional[float] = None
    volume_ratio: Optional[float] = None
    # VCP-specific from refined detector
    num_contractions: Optional[int] = None
    proximity_score: Optional[float] = None
    contraction_quality: Optional[float] = None
    volume_quality: Optional[float] = None
    final_contraction_pct: Optional[float] = None
    # Risk metrics
    risk_pct: Optional[float] = None
    reward_pct: Optional[float] = None
    theoretical_rr: Optional[float] = None
    realized_rr: Optional[float] = None


@dataclass
class BacktestResult:
    """Results from a backtest run"""
    strategy_name: str
    params: Dict[str, Any]
    exit_method: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    open_trades: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    avg_rr_realized: float
    total_return_pct: float
    max_drawdown_pct: float
    avg_days_held: float
    # Proximity analysis
    proximity_correlation: Optional[float] = None
    high_proximity_win_rate: Optional[float] = None
    low_proximity_win_rate: Optional[float] = None
    trades: List[Trade] = field(default_factory=list)


# =============================================================================
# Data Cache
# =============================================================================

class DataCache:
    """Cache for historical price data"""

    def __init__(self, cache_dir: str = 'cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.memory_cache: Dict[str, pd.DataFrame] = {}

    def get_historical_data(self, symbol: str, period: str = '2y') -> Optional[pd.DataFrame]:
        """Get historical data with caching"""
        cache_key = f"{symbol}_{period}"

        # Check memory cache
        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]

        # Check file cache
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.parquet")
        if os.path.exists(cache_file):
            try:
                file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
                if datetime.now() - file_time < timedelta(days=1):
                    df = pd.read_parquet(cache_file)
                    self.memory_cache[cache_key] = df
                    return df
            except Exception as e:
                logger.warning(f"Error reading cache for {symbol}: {e}")

        # Fetch from yfinance
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)

            if df.empty:
                logger.warning(f"No data for {symbol}")
                return None

            # Normalize column names
            df.columns = [c.title() for c in df.columns]

            # Handle MultiIndex columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.to_parquet(cache_file)
            self.memory_cache[cache_key] = df

            return df

        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return None


# =============================================================================
# RS Rating Calculator
# =============================================================================

def calculate_rs_rating(df: pd.DataFrame, spy_df: pd.DataFrame) -> float:
    """Calculate RS rating vs SPY benchmark"""
    if df is None or spy_df is None or len(df) < 252 or len(spy_df) < 252:
        return 50.0

    try:
        periods = RS_PARAMS['current']['periods']
        weights = RS_PARAMS['current']['weights']

        stock_ratios = []
        spy_ratios = []

        for period in periods:
            if len(df) >= period and len(spy_df) >= period:
                stock_ratio = df['Close'].iloc[-1] / df['Close'].iloc[-period]
                spy_ratio = spy_df['Close'].iloc[-1] / spy_df['Close'].iloc[-period]
                stock_ratios.append(stock_ratio)
                spy_ratios.append(spy_ratio)

        if len(stock_ratios) < 4:
            return 50.0

        rs_stock = sum(r * w for r, w in zip(stock_ratios, weights))
        rs_spy = sum(r * w for r, w in zip(spy_ratios, weights))

        if rs_spy == 0:
            return 50.0

        total_rs_score = (rs_stock / rs_spy) * 100

        # Map to rating using thresholds
        thresholds = RS_PARAMS['current']['rating_thresholds']
        if total_rs_score >= thresholds[99]:
            return 99.0
        elif total_rs_score >= thresholds[90]:
            return 90.0 + ((total_rs_score - thresholds[90]) / (thresholds[99] - thresholds[90])) * 8.0
        elif total_rs_score >= thresholds[70]:
            return 70.0 + ((total_rs_score - thresholds[70]) / (thresholds[90] - thresholds[70])) * 19.0
        elif total_rs_score >= thresholds[50]:
            return 50.0 + ((total_rs_score - thresholds[50]) / (thresholds[70] - thresholds[50])) * 19.0
        else:
            return max(1.0, 50.0 * (total_rs_score / thresholds[50]))

    except Exception as e:
        logger.debug(f"RS calculation error: {e}")
        return 50.0


# =============================================================================
# VCP Scanner using Refined Detector
# =============================================================================

class VCPScanner:
    """Scans for VCP patterns using the refined detector"""

    def __init__(self, params: Dict[str, Any]):
        self.params = params
        # Initialize refined VCP detector
        self.vcp_detector = VCPDetector(
            swing_lookback=5,
            min_contractions=params.get('min_contractions', 2),
            max_contraction_range=params.get('contraction_threshold', 0.20) * 100
        )

    def scan_for_vcp(self, df: pd.DataFrame, spy_df: pd.DataFrame,
                     check_date_idx: int) -> Optional[Dict[str, Any]]:
        """
        Scan for VCP pattern at a specific date using refined detector.

        Returns signal dict if valid pattern found, None otherwise.
        """
        if check_date_idx < 252:  # Need enough history
            return None

        df_slice = df.iloc[:check_date_idx + 1].copy()

        if len(df_slice) < 120:
            return None

        params = self.params

        # Calculate indicators
        df_slice['SMA_50'] = df_slice['Close'].rolling(window=50).mean()
        df_slice['SMA_200'] = df_slice['Close'].rolling(window=200).mean()
        df_slice['Volume_Avg'] = df_slice['Volume'].rolling(window=50).mean()

        current_price = df_slice['Close'].iloc[-1]
        current_volume = df_slice['Volume'].iloc[-1]
        avg_volume = df_slice['Volume_Avg'].iloc[-1]

        if pd.isna(avg_volume) or avg_volume == 0:
            return None

        # Check price above 200 MA
        if len(df_slice) >= 200:
            sma_200 = df_slice['SMA_200'].iloc[-1]
            if pd.notna(sma_200) and current_price < sma_200:
                return None

        # Check distance from 52-week high
        lookback = min(252, len(df_slice))
        high_52w = df_slice['High'].tail(lookback).max()
        distance_from_high = (high_52w - current_price) / high_52w

        if distance_from_high > params.get('distance_from_52w_high_max', 0.25):
            return None

        # Use refined VCP detector
        pattern = self.vcp_detector.analyze_pattern(df_slice, lookback_days=120)

        if pattern is None or not pattern.is_valid:
            return None

        # Check minimum proximity score if specified
        min_proximity = params.get('min_proximity_score', 0)
        if pattern.proximity_score < min_proximity:
            return None

        # Check RS rating
        spy_slice = spy_df.iloc[:check_date_idx + 1] if spy_df is not None else None
        rs_rating = calculate_rs_rating(df_slice, spy_slice)

        if rs_rating < params.get('rs_rating_min', 70):
            return None

        # Check for breakout (strict criteria)
        pivot_price = pattern.pivot_price
        pivot_vol_mult = params.get('pivot_breakout_volume_multiplier', 1.5)

        current_open = df_slice['Open'].iloc[-1]
        current_high = df_slice['High'].iloc[-1]
        current_low = df_slice['Low'].iloc[-1]

        # 1. Close must be above pivot
        if current_price < pivot_price:
            return None

        # 2. Must have above-average volume
        if current_volume < avg_volume * pivot_vol_mult:
            return None

        # 3. Must be a bullish candle (close > open)
        if current_price <= current_open:
            return None

        # 4. Close should be in upper 50% of day's range (strong close)
        day_range = current_high - current_low
        if day_range > 0:
            close_position = (current_price - current_low) / day_range
            if close_position < 0.5:  # Weak close
                return None

        # 5. Must be breaking out TODAY (previous close was below pivot)
        if len(df_slice) >= 2:
            prev_close = df_slice['Close'].iloc[-2]
            if prev_close >= pivot_price:  # Already above pivot, not a fresh breakout
                return None

        # Valid VCP signal
        return {
            'pattern': 'VCP',
            'price': float(current_price),
            'pivot_price': float(pivot_price),
            'volume_ratio': float(current_volume / avg_volume),
            'distance_52w_high': float(distance_from_high * 100),
            'rs_rating': float(rs_rating),
            'date': df_slice.index[-1],
            # From refined VCP detector
            'num_contractions': len(pattern.contractions),
            'proximity_score': pattern.proximity_score,
            'contraction_quality': pattern.contraction_quality,
            'volume_quality': pattern.volume_quality,
            'final_contraction_pct': pattern.contractions[-1].range_pct if pattern.contractions else 0,
            'support_price': float(pattern.support_price),
        }


# =============================================================================
# Trade Simulator
# =============================================================================

class TradeSimulator:
    """Simulates trades with different exit methods"""

    def simulate_fixed_target(self, df: pd.DataFrame, signal: Dict[str, Any],
                               stop_loss_pct: float = 7.0,
                               target_pct: float = 20.0,
                               max_hold_days: int = 60) -> Trade:
        """Simulate trade with fixed stop and target"""
        entry_date = signal['date']
        entry_price = signal['price']
        stop_loss = entry_price * (1 - stop_loss_pct / 100)
        target_price = entry_price * (1 + target_pct / 100)

        try:
            entry_idx = df.index.get_loc(entry_date)
        except KeyError:
            entry_idx = df.index.searchsorted(entry_date)

        trade = Trade(
            symbol=signal.get('symbol', 'UNKNOWN'),
            pattern=signal['pattern'],
            entry_date=str(entry_date)[:10],
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_price=target_price,
            rs_rating=signal.get('rs_rating'),
            volume_ratio=signal.get('volume_ratio'),
            num_contractions=signal.get('num_contractions'),
            proximity_score=signal.get('proximity_score'),
            contraction_quality=signal.get('contraction_quality'),
            volume_quality=signal.get('volume_quality'),
            final_contraction_pct=signal.get('final_contraction_pct'),
            risk_pct=stop_loss_pct,
            reward_pct=target_pct,
            theoretical_rr=target_pct / stop_loss_pct,
        )

        max_price = entry_price
        min_price = entry_price

        for day_offset in range(1, max_hold_days + 1):
            check_idx = entry_idx + day_offset

            if check_idx >= len(df):
                trade.exit_date = str(df.index[-1])[:10]
                trade.exit_price = df['Close'].iloc[-1]
                trade.exit_reason = 'open'
                trade.days_held = len(df) - entry_idx - 1
                break

            day_data = df.iloc[check_idx]
            low = day_data['Low']
            high = day_data['High']
            close = day_data['Close']

            max_price = max(max_price, high)
            min_price = min(min_price, low)

            if low <= stop_loss:
                trade.exit_date = str(df.index[check_idx])[:10]
                trade.exit_price = stop_loss
                trade.exit_reason = 'stop'
                trade.days_held = day_offset
                break

            if high >= target_price:
                trade.exit_date = str(df.index[check_idx])[:10]
                trade.exit_price = target_price
                trade.exit_reason = 'target'
                trade.days_held = day_offset
                break

            if day_offset >= max_hold_days:
                trade.exit_date = str(df.index[check_idx])[:10]
                trade.exit_price = close
                trade.exit_reason = 'time'
                trade.days_held = day_offset
                break

        if trade.exit_price is not None:
            trade.pnl_pct = ((trade.exit_price - entry_price) / entry_price) * 100
            if trade.risk_pct and trade.risk_pct > 0:
                trade.realized_rr = trade.pnl_pct / trade.risk_pct

        trade.max_gain_pct = ((max_price - entry_price) / entry_price) * 100
        trade.max_drawdown_pct = ((entry_price - min_price) / entry_price) * 100

        return trade

    def simulate_trailing_stop(self, df: pd.DataFrame, signal: Dict[str, Any],
                                stop_loss_pct: float = 7.0,
                                trailing_activation_pct: float = 8.0,
                                trailing_distance_pct: float = 5.0,
                                max_hold_days: int = 90) -> Trade:
        """Simulate trade with trailing stop after activation"""
        entry_date = signal['date']
        entry_price = signal['price']
        initial_stop = entry_price * (1 - stop_loss_pct / 100)
        current_stop = initial_stop

        try:
            entry_idx = df.index.get_loc(entry_date)
        except KeyError:
            entry_idx = df.index.searchsorted(entry_date)

        trade = Trade(
            symbol=signal.get('symbol', 'UNKNOWN'),
            pattern=signal['pattern'],
            entry_date=str(entry_date)[:10],
            entry_price=entry_price,
            stop_loss=initial_stop,
            target_price=0,  # No fixed target in trailing mode
            rs_rating=signal.get('rs_rating'),
            volume_ratio=signal.get('volume_ratio'),
            num_contractions=signal.get('num_contractions'),
            proximity_score=signal.get('proximity_score'),
            contraction_quality=signal.get('contraction_quality'),
            volume_quality=signal.get('volume_quality'),
            final_contraction_pct=signal.get('final_contraction_pct'),
            risk_pct=stop_loss_pct,
        )

        max_price = entry_price
        min_price = entry_price
        trailing_active = False

        for day_offset in range(1, max_hold_days + 1):
            check_idx = entry_idx + day_offset

            if check_idx >= len(df):
                trade.exit_date = str(df.index[-1])[:10]
                trade.exit_price = df['Close'].iloc[-1]
                trade.exit_reason = 'open'
                trade.days_held = len(df) - entry_idx - 1
                break

            day_data = df.iloc[check_idx]
            low = day_data['Low']
            high = day_data['High']
            close = day_data['Close']

            # Update max price
            if high > max_price:
                max_price = high

                # Check if trailing should activate
                gain_pct = ((max_price - entry_price) / entry_price) * 100
                if gain_pct >= trailing_activation_pct and not trailing_active:
                    trailing_active = True
                    current_stop = entry_price  # Move to breakeven

                # Update trailing stop if active
                if trailing_active:
                    new_stop = max_price * (1 - trailing_distance_pct / 100)
                    if new_stop > current_stop:
                        current_stop = new_stop

            min_price = min(min_price, low)

            # Check stop
            if low <= current_stop:
                trade.exit_date = str(df.index[check_idx])[:10]
                trade.exit_price = current_stop
                trade.exit_reason = 'trailing_stop' if trailing_active else 'stop'
                trade.days_held = day_offset
                break

            if day_offset >= max_hold_days:
                trade.exit_date = str(df.index[check_idx])[:10]
                trade.exit_price = close
                trade.exit_reason = 'time'
                trade.days_held = day_offset
                break

        if trade.exit_price is not None:
            trade.pnl_pct = ((trade.exit_price - entry_price) / entry_price) * 100
            if trade.risk_pct and trade.risk_pct > 0:
                trade.realized_rr = trade.pnl_pct / trade.risk_pct

        trade.max_gain_pct = ((max_price - entry_price) / entry_price) * 100
        trade.max_drawdown_pct = ((entry_price - min_price) / entry_price) * 100
        trade.reward_pct = trade.max_gain_pct

        return trade


# =============================================================================
# Backtester
# =============================================================================

class Backtester:
    """Main backtesting engine"""

    def __init__(self, data_cache: DataCache):
        self.data_cache = data_cache
        self.spy_df = data_cache.get_historical_data('SPY', '2y')
        self.simulator = TradeSimulator()

    def run_backtest(self, symbols: List[str], strategy_name: str,
                     params: Dict[str, Any], risk_params: Dict[str, Any],
                     exit_method: str = 'trailing_stop',
                     start_date: str = '2023-01-01') -> BacktestResult:
        """Run backtest across multiple symbols"""
        scanner = VCPScanner(params)
        all_trades: List[Trade] = []

        start_dt = pd.Timestamp(start_date)

        logger.info(f"Running {strategy_name} ({exit_method}) on {len(symbols)} symbols...")

        for i, symbol in enumerate(symbols):
            if (i + 1) % 10 == 0:
                logger.info(f"  Processing {i+1}/{len(symbols)} symbols...")

            df = self.data_cache.get_historical_data(symbol, '2y')

            if df is None or len(df) < 252:
                continue

            last_signal_idx = -60  # Avoid overlapping trades

            for idx in range(252, len(df) - 60):
                check_date = df.index[idx]

                # Normalize timezone for comparison
                check_date_normalized = check_date.tz_localize(None) if hasattr(check_date, 'tz_localize') and check_date.tzinfo else check_date
                start_dt_normalized = start_dt.tz_localize(None) if hasattr(start_dt, 'tz_localize') and start_dt.tzinfo else start_dt

                if check_date_normalized < start_dt_normalized:
                    continue

                # Skip if too close to last signal
                if idx - last_signal_idx < 20:
                    continue

                # Detect VCP pattern
                signal = scanner.scan_for_vcp(df, self.spy_df, idx)

                if signal:
                    signal['symbol'] = symbol
                    last_signal_idx = idx

                    # Simulate trade
                    if exit_method == 'fixed_target':
                        trade = self.simulator.simulate_fixed_target(
                            df, signal,
                            stop_loss_pct=risk_params.get('default_stop_loss_pct', 7.0),
                            target_pct=risk_params.get('default_target_pct', 20.0),
                            max_hold_days=60
                        )
                    else:
                        trade = self.simulator.simulate_trailing_stop(
                            df, signal,
                            stop_loss_pct=risk_params.get('default_stop_loss_pct', 7.0),
                            trailing_activation_pct=risk_params.get('trailing_stop_activation_pct', 8.0),
                            trailing_distance_pct=risk_params.get('trailing_stop_distance_pct', 5.0),
                            max_hold_days=90
                        )

                    all_trades.append(trade)

        return self._calculate_stats(strategy_name, params, exit_method, all_trades)

    def _calculate_stats(self, strategy_name: str, params: Dict[str, Any],
                         exit_method: str, trades: List[Trade]) -> BacktestResult:
        """Calculate backtest statistics"""
        if not trades:
            return BacktestResult(
                strategy_name=strategy_name,
                params=params,
                exit_method=exit_method,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                open_trades=0,
                win_rate=0.0,
                avg_win_pct=0.0,
                avg_loss_pct=0.0,
                profit_factor=0.0,
                avg_rr_realized=0.0,
                total_return_pct=0.0,
                max_drawdown_pct=0.0,
                avg_days_held=0.0,
                trades=trades
            )

        closed_trades = [t for t in trades if t.exit_reason != 'open']
        winning = [t for t in closed_trades if t.pnl_pct and t.pnl_pct > 0]
        losing = [t for t in closed_trades if t.pnl_pct and t.pnl_pct <= 0]
        open_trades = [t for t in trades if t.exit_reason == 'open']

        win_rate = len(winning) / len(closed_trades) * 100 if closed_trades else 0
        avg_win = np.mean([t.pnl_pct for t in winning]) if winning else 0
        avg_loss = np.mean([t.pnl_pct for t in losing]) if losing else 0

        total_wins = sum(t.pnl_pct for t in winning) if winning else 0
        total_losses = abs(sum(t.pnl_pct for t in losing)) if losing else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        avg_rr_realized = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        total_return = sum(t.pnl_pct for t in closed_trades if t.pnl_pct) if closed_trades else 0
        avg_days = np.mean([t.days_held for t in closed_trades if t.days_held]) if closed_trades else 0
        max_dd = max([t.max_drawdown_pct for t in trades if t.max_drawdown_pct], default=0)

        # Proximity analysis
        proximity_correlation = None
        high_proximity_win_rate = None
        low_proximity_win_rate = None

        vcp_trades = [t for t in closed_trades if t.proximity_score is not None]
        if len(vcp_trades) >= 10:
            proximities = [t.proximity_score for t in vcp_trades]
            wins = [1 if t.pnl_pct and t.pnl_pct > 0 else 0 for t in vcp_trades]

            if len(set(proximities)) > 1:
                proximity_correlation = np.corrcoef(proximities, wins)[0, 1]

            high_prox = [t for t in vcp_trades if t.proximity_score >= 70]
            low_prox = [t for t in vcp_trades if t.proximity_score <= 30]

            if high_prox:
                high_proximity_win_rate = sum(1 for t in high_prox if t.pnl_pct and t.pnl_pct > 0) / len(high_prox) * 100
            if low_prox:
                low_proximity_win_rate = sum(1 for t in low_prox if t.pnl_pct and t.pnl_pct > 0) / len(low_prox) * 100

        return BacktestResult(
            strategy_name=strategy_name,
            params=params,
            exit_method=exit_method,
            total_trades=len(trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            open_trades=len(open_trades),
            win_rate=win_rate,
            avg_win_pct=avg_win,
            avg_loss_pct=avg_loss,
            profit_factor=profit_factor,
            avg_rr_realized=avg_rr_realized,
            total_return_pct=total_return,
            max_drawdown_pct=max_dd,
            avg_days_held=avg_days,
            proximity_correlation=proximity_correlation,
            high_proximity_win_rate=high_proximity_win_rate,
            low_proximity_win_rate=low_proximity_win_rate,
            trades=trades
        )


# =============================================================================
# Output Functions
# =============================================================================

def print_results(result: BacktestResult):
    """Pretty print backtest results"""
    print(f"\n{'='*70}")
    print(f"  {result.strategy_name} ({result.exit_method})")
    print(f"{'='*70}")
    print(f"  Total Trades:        {result.total_trades}")
    print(f"  Winning Trades:      {result.winning_trades}")
    print(f"  Losing Trades:       {result.losing_trades}")
    print(f"  Open Trades:         {result.open_trades}")
    print(f"  Win Rate:            {result.win_rate:.1f}%")
    print(f"  Avg Win:             +{result.avg_win_pct:.2f}%")
    print(f"  Avg Loss:            {result.avg_loss_pct:.2f}%")
    print(f"  Profit Factor:       {result.profit_factor:.2f}")
    print(f"  Avg R:R Realized:    {result.avg_rr_realized:.2f}")
    print(f"  Total Return:        {result.total_return_pct:.2f}%")
    print(f"  Max Drawdown:        {result.max_drawdown_pct:.2f}%")
    print(f"  Avg Days Held:       {result.avg_days_held:.1f}")

    if result.proximity_correlation is not None:
        print(f"\n  VCP Proximity Analysis:")
        print(f"    Proximity-Win Correlation: {result.proximity_correlation:.3f}")
        if result.high_proximity_win_rate is not None:
            print(f"    High Proximity (>=70) Win Rate: {result.high_proximity_win_rate:.1f}%")
        if result.low_proximity_win_rate is not None:
            print(f"    Low Proximity (<=30) Win Rate: {result.low_proximity_win_rate:.1f}%")

    print(f"{'='*70}")


def save_results(result: BacktestResult, output_dir: str = 'results/v3'):
    """Save results to files"""
    os.makedirs(output_dir, exist_ok=True)

    # Save summary
    summary = {
        'strategy_name': result.strategy_name,
        'exit_method': result.exit_method,
        'total_trades': result.total_trades,
        'winning_trades': result.winning_trades,
        'losing_trades': result.losing_trades,
        'win_rate': result.win_rate,
        'avg_win_pct': result.avg_win_pct,
        'avg_loss_pct': result.avg_loss_pct,
        'profit_factor': result.profit_factor,
        'total_return_pct': result.total_return_pct,
        'proximity_correlation': result.proximity_correlation,
        'high_proximity_win_rate': result.high_proximity_win_rate,
        'low_proximity_win_rate': result.low_proximity_win_rate,
    }

    with open(os.path.join(output_dir, f'{result.strategy_name}_{result.exit_method}_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Save trades
    if result.trades:
        trades_df = pd.DataFrame([asdict(t) for t in result.trades])
        trades_df.to_csv(os.path.join(output_dir, f'{result.strategy_name}_{result.exit_method}_trades.csv'), index=False)

    logger.info(f"Results saved to {output_dir}")


# =============================================================================
# Stock Universe
# =============================================================================

def get_sp500_symbols() -> List[str]:
    """Get S&P 500 symbols"""
    # Top 50 S&P 500 by market cap (simplified for testing)
    return [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
        'XOM', 'JPM', 'V', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'ABBV', 'LLY',
        'PFE', 'AVGO', 'COST', 'PEP', 'KO', 'TMO', 'CSCO', 'MCD', 'WMT', 'ABT',
        'ACN', 'DHR', 'CRM', 'CMCSA', 'VZ', 'ADBE', 'NEE', 'INTC', 'WFC', 'PM',
        'TXN', 'NKE', 'BMY', 'UPS', 'RTX', 'ORCL', 'QCOM', 'BA', 'HON', 'AMGN',
        'IBM', 'GE', 'CAT', 'SBUX', 'DE', 'GS', 'MS', 'BLK', 'AXP', 'MMM',
        'LMT', 'AMD', 'GILD', 'ISRG', 'MDT', 'SYK', 'ADI', 'BKNG', 'TJX', 'SPGI',
        'MDLZ', 'VRTX', 'ADP', 'LRCX', 'REGN', 'ZTS', 'CB', 'CI', 'SO', 'MO',
        'SCHW', 'DUK', 'CME', 'CL', 'BDX', 'APD', 'EOG', 'ITW', 'AON', 'NOC',
        'SLB', 'FDX', 'EMR', 'WM', 'PNC', 'TGT', 'USB', 'NSC', 'CSX', 'F',
    ]


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Run backtests with refined VCP detector"""
    print("\n" + "="*70)
    print("  VCP BACKTEST V3 - Using Refined VCP Detector")
    print("="*70)
    print("\nThis backtest uses the improved VCP detection algorithm with:")
    print("  - Proper swing high/low detection")
    print("  - 'Extend until recovery' contraction identification")
    print("  - Progressive tightening enforcement")
    print("  - Consolidation base rule (rejects staircase patterns)")
    print("  - Volume dry-up validation")
    print()

    # Initialize
    data_cache = DataCache('cache')
    backtester = Backtester(data_cache)
    symbols = get_sp500_symbols()

    # Risk parameters
    risk_params = RISK_PARAMS['current']

    results = []

    # ==========================================================================
    # Test 1: RS 70 + Trailing Stop (Previous best config)
    # ==========================================================================
    print("\n" + "-"*70)
    print("Test 1: RS 70 + Trailing Stop (Previous optimal)")
    print("-"*70)

    params_rs70 = {
        'rs_rating_min': 70,
        'min_contractions': 2,
        'contraction_threshold': 0.20,
        'distance_from_52w_high_max': 0.25,
        'pivot_breakout_volume_multiplier': 1.5,
        'min_proximity_score': 0,  # No filter first
    }

    result_rs70_trailing = backtester.run_backtest(
        symbols=symbols,
        strategy_name='VCP_v3_RS70',
        params=params_rs70,
        risk_params=risk_params,
        exit_method='trailing_stop',
        start_date='2023-01-01'
    )
    print_results(result_rs70_trailing)
    results.append(result_rs70_trailing)

    # ==========================================================================
    # Test 2: RS 70 + Proximity >= 50 + Trailing Stop
    # ==========================================================================
    print("\n" + "-"*70)
    print("Test 2: RS 70 + Proximity >= 50 + Trailing Stop")
    print("-"*70)

    params_rs70_prox = {
        'rs_rating_min': 70,
        'min_contractions': 2,
        'contraction_threshold': 0.20,
        'distance_from_52w_high_max': 0.25,
        'pivot_breakout_volume_multiplier': 1.5,
        'min_proximity_score': 50,  # Add proximity filter
    }

    result_rs70_prox = backtester.run_backtest(
        symbols=symbols,
        strategy_name='VCP_v3_RS70_Prox50',
        params=params_rs70_prox,
        risk_params=risk_params,
        exit_method='trailing_stop',
        start_date='2023-01-01'
    )
    print_results(result_rs70_prox)
    results.append(result_rs70_prox)

    # ==========================================================================
    # Test 3: RS 80 + Trailing Stop
    # ==========================================================================
    print("\n" + "-"*70)
    print("Test 3: RS 80 + Trailing Stop")
    print("-"*70)

    params_rs80 = {
        'rs_rating_min': 80,
        'min_contractions': 2,
        'contraction_threshold': 0.20,
        'distance_from_52w_high_max': 0.25,
        'pivot_breakout_volume_multiplier': 1.5,
        'min_proximity_score': 0,
    }

    result_rs80 = backtester.run_backtest(
        symbols=symbols,
        strategy_name='VCP_v3_RS80',
        params=params_rs80,
        risk_params=risk_params,
        exit_method='trailing_stop',
        start_date='2023-01-01'
    )
    print_results(result_rs80)
    results.append(result_rs80)

    # ==========================================================================
    # Test 4: RS 80 + Proximity >= 50 + Trailing Stop
    # ==========================================================================
    print("\n" + "-"*70)
    print("Test 4: RS 80 + Proximity >= 50 + Trailing Stop")
    print("-"*70)

    params_rs80_prox = {
        'rs_rating_min': 80,
        'min_contractions': 2,
        'contraction_threshold': 0.20,
        'distance_from_52w_high_max': 0.25,
        'pivot_breakout_volume_multiplier': 1.5,
        'min_proximity_score': 50,
    }

    result_rs80_prox = backtester.run_backtest(
        symbols=symbols,
        strategy_name='VCP_v3_RS80_Prox50',
        params=params_rs80_prox,
        risk_params=risk_params,
        exit_method='trailing_stop',
        start_date='2023-01-01'
    )
    print_results(result_rs80_prox)
    results.append(result_rs80_prox)

    # ==========================================================================
    # Test 5: Fixed Target comparison (RS 70 + Prox 50)
    # ==========================================================================
    print("\n" + "-"*70)
    print("Test 5: RS 70 + Proximity >= 50 + Fixed Target (7%/21%)")
    print("-"*70)

    result_fixed = backtester.run_backtest(
        symbols=symbols,
        strategy_name='VCP_v3_RS70_Prox50_Fixed',
        params=params_rs70_prox,
        risk_params=risk_params,
        exit_method='fixed_target',
        start_date='2023-01-01'
    )
    print_results(result_fixed)
    results.append(result_fixed)

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "="*70)
    print("  SUMMARY - VCP V3 Backtest Results")
    print("="*70)

    summary_data = []
    for r in results:
        summary_data.append({
            'Config': r.strategy_name.replace('VCP_v3_', ''),
            'Exit': r.exit_method,
            'Trades': r.total_trades,
            'Win%': f"{r.win_rate:.1f}%",
            'PF': f"{r.profit_factor:.2f}",
            'Return': f"{r.total_return_pct:.1f}%",
            'AvgDays': f"{r.avg_days_held:.0f}",
        })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    # Save all results
    for r in results:
        save_results(r)

    print(f"\nResults saved to results/v3/")
    print("\nBacktest complete!")


if __name__ == '__main__':
    main()
