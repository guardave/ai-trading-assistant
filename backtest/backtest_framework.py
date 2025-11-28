"""
Backtesting Framework for VCP and Pivot Strategies
Evaluates strategy performance on historical data
"""
import os
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single trade"""
    symbol: str
    pattern: str
    entry_date: str
    entry_price: float
    stop_loss: float
    target_price: float
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # 'stop', 'target', 'time', 'open'
    pnl_pct: Optional[float] = None
    pnl_dollars: Optional[float] = None
    days_held: Optional[int] = None
    max_gain_pct: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    rs_rating: Optional[float] = None
    volume_ratio: Optional[float] = None


@dataclass
class BacktestResult:
    """Results from a backtest run"""
    strategy_name: str
    params: Dict[str, Any]
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
    trades: List[Trade] = field(default_factory=list)


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
                # Check if cache is recent (less than 1 day old)
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

            # Ensure column names are correct
            df.columns = [c.title() for c in df.columns]

            # Save to cache
            df.to_parquet(cache_file)
            self.memory_cache[cache_key] = df

            return df

        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return None


class StrategyScanner:
    """Scans for VCP and Pivot patterns"""

    def __init__(self, params: Dict[str, Any]):
        self.params = params

    def calculate_rs_rating(self, df: pd.DataFrame, spy_df: pd.DataFrame) -> float:
        """Calculate simplified RS rating vs SPY"""
        if df is None or spy_df is None or len(df) < 252 or len(spy_df) < 252:
            return 50.0

        try:
            periods = [63, 126, 189, 252]
            weights = [0.4, 0.2, 0.2, 0.2]

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

            # Map to rating
            if total_rs_score >= 195.93:
                return 99.0
            elif total_rs_score >= 117.11:
                return 90.0 + ((total_rs_score - 117.11) / (195.93 - 117.11)) * 8.0
            elif total_rs_score >= 99.04:
                return 70.0 + ((total_rs_score - 99.04) / (117.11 - 99.04)) * 19.0
            elif total_rs_score >= 91.66:
                return 50.0 + ((total_rs_score - 91.66) / (99.04 - 91.66)) * 19.0
            else:
                return max(1.0, 50.0 * (total_rs_score / 91.66))

        except Exception as e:
            logger.error(f"RS calculation error: {e}")
            return 50.0

    def detect_vcp(self, df: pd.DataFrame, spy_df: pd.DataFrame,
                   check_date_idx: int) -> Optional[Dict[str, Any]]:
        """
        Detect VCP pattern at a specific point in history

        Args:
            df: Full price DataFrame
            spy_df: SPY DataFrame for RS calculation
            check_date_idx: Index to check (simulates "today")

        Returns:
            Signal dict or None
        """
        # Get data up to check date
        df_slice = df.iloc[:check_date_idx + 1]

        if len(df_slice) < 60:
            return None

        params = self.params

        # Calculate indicators
        df_slice = df_slice.copy()
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

        # Check consolidation
        weeks_to_check = params.get('min_consolidation_weeks', 4)
        days_to_check = weeks_to_check * 5

        if len(df_slice) < days_to_check:
            return None

        consolidation_df = df_slice.tail(days_to_check)

        # Calculate contraction
        high_in_period = consolidation_df['High'].max()
        low_in_period = consolidation_df['Low'].min()
        contraction = (high_in_period - low_in_period) / low_in_period

        if contraction > params.get('contraction_threshold', 0.20):
            return None

        # Check volume dry-up
        recent_volume_avg = consolidation_df['Volume'].tail(10).mean()
        volume_ratio = recent_volume_avg / avg_volume

        if volume_ratio > params.get('volume_dry_up_ratio', 0.5):
            return None

        # Check RS rating
        spy_slice = spy_df.iloc[:check_date_idx + 1] if spy_df is not None else None
        rs_rating = self.calculate_rs_rating(df_slice, spy_slice)

        if rs_rating < params.get('rs_rating_min', 90):
            return None

        # Check for breakout
        pivot_breakout = False
        if current_price > consolidation_df['High'].iloc[-2]:
            pivot_vol_mult = params.get('pivot_breakout_volume_multiplier', 1.5)
            if current_volume > avg_volume * pivot_vol_mult:
                pivot_breakout = True

        if not pivot_breakout:
            return None  # Only count actual breakouts

        return {
            'pattern': 'VCP',
            'price': float(current_price),
            'volume_ratio': float(current_volume / avg_volume),
            'contraction_pct': float(contraction * 100),
            'distance_52w_high': float(distance_from_high * 100),
            'rs_rating': float(rs_rating),
            'date': df_slice.index[-1],
        }

    def detect_pivot(self, df: pd.DataFrame, spy_df: pd.DataFrame,
                     check_date_idx: int) -> Optional[Dict[str, Any]]:
        """
        Detect Pivot Breakout pattern at a specific point in history
        """
        df_slice = df.iloc[:check_date_idx + 1]

        if len(df_slice) < 60:
            return None

        params = self.params

        # Calculate indicators
        df_slice = df_slice.copy()
        df_slice['SMA_50'] = df_slice['Close'].rolling(window=50).mean()
        df_slice['SMA_200'] = df_slice['Close'].rolling(window=200).mean()
        df_slice['Volume_Avg'] = df_slice['Volume'].rolling(window=50).mean()

        current_price = df_slice['Close'].iloc[-1]
        current_volume = df_slice['Volume'].iloc[-1]
        avg_volume = df_slice['Volume_Avg'].iloc[-1]

        if pd.isna(avg_volume) or avg_volume == 0:
            return None

        # Check MAs
        if params.get('price_above_50ma', True) and len(df_slice) >= 50:
            sma_50 = df_slice['SMA_50'].iloc[-1]
            if pd.notna(sma_50) and current_price < sma_50:
                return None

        if params.get('price_above_200ma', True) and len(df_slice) >= 200:
            sma_200 = df_slice['SMA_200'].iloc[-1]
            if pd.notna(sma_200) and current_price < sma_200:
                return None

        # Check distance from 52-week high
        lookback = min(252, len(df_slice))
        high_52w = df_slice['High'].tail(lookback).max()
        distance_from_high = (high_52w - current_price) / high_52w

        if distance_from_high > params.get('distance_from_52w_high_max', 0.30):
            return None

        # Look for base formation
        min_base_weeks = params.get('min_base_weeks', 4)
        max_base_weeks = params.get('max_base_weeks', 52)
        min_base_depth = params.get('min_base_depth', 0.08)
        max_base_depth = params.get('max_base_depth', 0.35)

        max_lookback = min(max_base_weeks * 5, len(df_slice))

        base_found = False
        base_high = None
        base_low = None
        base_weeks = 0

        for weeks_back in range(min_base_weeks * 5, max_lookback, 5):
            base_df = df_slice.tail(weeks_back)

            if len(base_df) < min_base_weeks * 5:
                continue

            base_high_candidate = base_df['High'].max()
            base_low_candidate = base_df['Low'].min()

            base_depth = (base_high_candidate - base_low_candidate) / base_high_candidate

            if min_base_depth <= base_depth <= max_base_depth:
                recent_high = df_slice.tail(10)['High'].max()

                if recent_high >= base_high_candidate * 0.98:
                    base_found = True
                    base_high = base_high_candidate
                    base_low = base_low_candidate
                    base_weeks = weeks_back // 5
                    break

        if not base_found:
            return None

        # Check for breakout
        pivot_vol_mult = params.get('pivot_volume_multiplier', 1.5)
        if current_price >= base_high and current_volume > avg_volume * pivot_vol_mult:
            # RS rating
            spy_slice = spy_df.iloc[:check_date_idx + 1] if spy_df is not None else None
            rs_rating = self.calculate_rs_rating(df_slice, spy_slice)

            if rs_rating < params.get('rs_rating_min', 80):
                return None

            base_depth_pct = ((base_high - base_low) / base_high) * 100

            return {
                'pattern': 'Pivot',
                'price': float(current_price),
                'pivot_price': float(base_high),
                'base_depth_pct': float(base_depth_pct),
                'base_weeks': base_weeks,
                'volume_ratio': float(current_volume / avg_volume),
                'distance_52w_high': float(distance_from_high * 100),
                'rs_rating': float(rs_rating),
                'date': df_slice.index[-1],
            }

        return None


class Backtester:
    """Main backtesting engine"""

    def __init__(self, data_cache: DataCache):
        self.data_cache = data_cache
        self.spy_df = data_cache.get_historical_data('SPY', '2y')

    def simulate_trade(self, df: pd.DataFrame, signal: Dict[str, Any],
                       stop_loss_pct: float = 7.0,
                       target_pct: float = 20.0,
                       max_hold_days: int = 60) -> Trade:
        """
        Simulate a trade from entry to exit

        Args:
            df: Full price DataFrame
            signal: Entry signal with date
            stop_loss_pct: Stop loss percentage
            target_pct: Target percentage
            max_hold_days: Maximum days to hold

        Returns:
            Trade object with results
        """
        entry_date = signal['date']
        entry_price = signal['price']
        stop_loss = entry_price * (1 - stop_loss_pct / 100)
        target_price = entry_price * (1 + target_pct / 100)

        # Find entry index
        try:
            entry_idx = df.index.get_loc(entry_date)
        except KeyError:
            # Find closest date
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
        )

        max_price = entry_price
        min_price = entry_price

        # Simulate day by day
        for day_offset in range(1, max_hold_days + 1):
            check_idx = entry_idx + day_offset

            if check_idx >= len(df):
                # Still open - use last available price
                trade.exit_date = str(df.index[-1])[:10]
                trade.exit_price = df['Close'].iloc[-1]
                trade.exit_reason = 'open'
                trade.days_held = len(df) - entry_idx - 1
                break

            day_data = df.iloc[check_idx]
            low = day_data['Low']
            high = day_data['High']
            close = day_data['Close']

            # Track max/min for analysis
            max_price = max(max_price, high)
            min_price = min(min_price, low)

            # Check stop loss (assumes hit during day)
            if low <= stop_loss:
                trade.exit_date = str(df.index[check_idx])[:10]
                trade.exit_price = stop_loss
                trade.exit_reason = 'stop'
                trade.days_held = day_offset
                break

            # Check target (assumes hit during day)
            if high >= target_price:
                trade.exit_date = str(df.index[check_idx])[:10]
                trade.exit_price = target_price
                trade.exit_reason = 'target'
                trade.days_held = day_offset
                break

            # Max holding period reached
            if day_offset >= max_hold_days:
                trade.exit_date = str(df.index[check_idx])[:10]
                trade.exit_price = close
                trade.exit_reason = 'time'
                trade.days_held = day_offset
                break

        # Calculate P&L
        if trade.exit_price is not None:
            trade.pnl_pct = ((trade.exit_price - entry_price) / entry_price) * 100
            trade.pnl_dollars = (trade.exit_price - entry_price) * 100  # Assume 100 shares

        trade.max_gain_pct = ((max_price - entry_price) / entry_price) * 100
        trade.max_drawdown_pct = ((entry_price - min_price) / entry_price) * 100

        return trade

    def run_backtest(self, symbols: List[str], strategy_name: str,
                     params: Dict[str, Any], risk_params: Dict[str, Any],
                     start_date: str = '2023-01-01') -> BacktestResult:
        """
        Run backtest across multiple symbols

        Args:
            symbols: List of stock symbols
            strategy_name: 'VCP' or 'Pivot'
            params: Strategy parameters
            risk_params: Risk management parameters
            start_date: Start date for backtest

        Returns:
            BacktestResult with all trades and statistics
        """
        scanner = StrategyScanner(params)
        all_trades: List[Trade] = []

        start_dt = pd.Timestamp(start_date)

        logger.info(f"Running {strategy_name} backtest on {len(symbols)} symbols...")

        for symbol in symbols:
            df = self.data_cache.get_historical_data(symbol, '2y')

            if df is None or len(df) < 252:
                continue

            # Scan each day for signals
            for idx in range(252, len(df) - 60):  # Need history and forward data
                check_date = df.index[idx]

                if check_date < start_dt:
                    continue

                # Detect pattern
                if strategy_name == 'VCP':
                    signal = scanner.detect_vcp(df, self.spy_df, idx)
                else:
                    signal = scanner.detect_pivot(df, self.spy_df, idx)

                if signal:
                    signal['symbol'] = symbol

                    # Simulate trade
                    trade = self.simulate_trade(
                        df, signal,
                        stop_loss_pct=risk_params.get('default_stop_loss_pct', 7.0),
                        target_pct=risk_params.get('default_target_pct', 20.0),
                        max_hold_days=60
                    )

                    all_trades.append(trade)

                    # Skip ahead to avoid overlapping trades on same symbol
                    # (This is a simplification)

        # Calculate statistics
        return self._calculate_stats(strategy_name, params, all_trades)

    def _calculate_stats(self, strategy_name: str, params: Dict[str, Any],
                         trades: List[Trade]) -> BacktestResult:
        """Calculate backtest statistics"""
        if not trades:
            return BacktestResult(
                strategy_name=strategy_name,
                params=params,
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

        avg_rr = avg_win / abs(avg_loss) if avg_loss != 0 else 0

        total_return = sum(t.pnl_pct for t in closed_trades if t.pnl_pct) if closed_trades else 0

        avg_days = np.mean([t.days_held for t in closed_trades if t.days_held]) if closed_trades else 0

        # Calculate max drawdown (simplified)
        max_dd = max([t.max_drawdown_pct for t in trades if t.max_drawdown_pct], default=0)

        return BacktestResult(
            strategy_name=strategy_name,
            params=params,
            total_trades=len(trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            open_trades=len(open_trades),
            win_rate=win_rate,
            avg_win_pct=avg_win,
            avg_loss_pct=avg_loss,
            profit_factor=profit_factor,
            avg_rr_realized=avg_rr,
            total_return_pct=total_return,
            max_drawdown_pct=max_dd,
            avg_days_held=avg_days,
            trades=trades
        )


def print_results(result: BacktestResult):
    """Pretty print backtest results"""
    print(f"\n{'='*60}")
    print(f"  {result.strategy_name} Strategy Backtest Results")
    print(f"{'='*60}")
    print(f"  Total Trades:      {result.total_trades}")
    print(f"  Winning Trades:    {result.winning_trades}")
    print(f"  Losing Trades:     {result.losing_trades}")
    print(f"  Open Trades:       {result.open_trades}")
    print(f"  Win Rate:          {result.win_rate:.1f}%")
    print(f"  Avg Win:           +{result.avg_win_pct:.2f}%")
    print(f"  Avg Loss:          {result.avg_loss_pct:.2f}%")
    print(f"  Profit Factor:     {result.profit_factor:.2f}")
    print(f"  Avg R:R Realized:  {result.avg_rr_realized:.2f}")
    print(f"  Total Return:      {result.total_return_pct:.2f}%")
    print(f"  Max Drawdown:      {result.max_drawdown_pct:.2f}%")
    print(f"  Avg Days Held:     {result.avg_days_held:.1f}")
    print(f"{'='*60}")

    # Show sample trades
    if result.trades:
        print("\n  Sample Trades (last 10):")
        print("-" * 80)
        print(f"  {'Symbol':<8} {'Pattern':<8} {'Entry':<10} {'Exit':<10} {'P&L':>8} {'Reason':<8}")
        print("-" * 80)
        for trade in result.trades[-10:]:
            pnl_str = f"{trade.pnl_pct:+.1f}%" if trade.pnl_pct else "N/A"
            print(f"  {trade.symbol:<8} {trade.pattern:<8} {trade.entry_date:<10} "
                  f"{trade.exit_date or 'Open':<10} {pnl_str:>8} {trade.exit_reason or '':<8}")


if __name__ == '__main__':
    # This will be called from run_backtest.py
    pass
