#!/usr/bin/env python3
"""
Backtest V4 - VCP Strategy using V4 Detector

Uses the improved VCP detector with:
1. Prior uptrend validation (30%+ advance)
2. Full trend template (8 criteria)
3. Swing point contraction detection
4. Progressive tightening enforcement
5. Minimum base duration

Compares results with V2 and V3 detectors.
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

# Import V4 detector
from vcp_detector_v4 import VCPDetectorV4, VCPConfig, VCPPattern

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single trade"""
    symbol: str
    entry_date: str
    entry_price: float
    stop_loss: float
    target_price: float
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl_pct: Optional[float] = None
    days_held: Optional[int] = None
    max_gain_pct: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    # V4 specific fields
    overall_score: Optional[float] = None
    trend_template_score: Optional[float] = None
    prior_advance_pct: Optional[float] = None
    num_contractions: Optional[int] = None
    final_contraction_pct: Optional[float] = None
    base_duration_days: Optional[int] = None


@dataclass
class BacktestResult:
    """Results from a backtest run"""
    detector_version: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    total_return_pct: float
    avg_days_held: float
    avg_overall_score: float
    trades: List[Trade] = field(default_factory=list)


class DataCache:
    """Cache for historical price data"""

    def __init__(self, cache_dir: str = 'cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.memory_cache: Dict[str, pd.DataFrame] = {}

    def get_data(self, symbol: str, period: str = '5y') -> Optional[pd.DataFrame]:
        """Get historical data with caching"""
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
            df.to_parquet(cache_file)
            self.memory_cache[cache_key] = df
            return df

        except Exception as e:
            logger.warning(f"Error fetching {symbol}: {e}")
            return None


class BacktestEngineV4:
    """Backtest engine using V4 VCP detector"""

    def __init__(
        self,
        stop_loss_pct: float = 7.0,
        target_rr: float = 3.0,
        max_hold_days: int = 60,
        use_trailing_stop: bool = True,
        trailing_activation_pct: float = 8.0,
        trailing_stop_pct: float = 5.0,
        min_overall_score: float = 70.0,
        require_valid_pattern: bool = True
    ):
        self.stop_loss_pct = stop_loss_pct
        self.target_rr = target_rr
        self.max_hold_days = max_hold_days
        self.use_trailing_stop = use_trailing_stop
        self.trailing_activation_pct = trailing_activation_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.min_overall_score = min_overall_score
        self.require_valid_pattern = require_valid_pattern

        self.detector = VCPDetectorV4()
        self.cache = DataCache()

    def detect_breakout(
        self,
        df: pd.DataFrame,
        pattern: VCPPattern,
        idx: int
    ) -> bool:
        """
        Detect if a breakout occurred on the given day.

        Criteria:
        1. Close above pivot price
        2. Volume > 1.5x 50-day average
        3. Bullish candle (close > open)
        4. Close in upper 50% of day's range
        5. Previous close was below pivot (fresh breakout)
        """
        if idx < 1 or idx >= len(df):
            return False

        row = df.iloc[idx]
        prev_row = df.iloc[idx - 1]
        pivot = pattern.pivot_price

        # Calculate 50-day average volume
        vol_start = max(0, idx - 50)
        avg_volume = df.iloc[vol_start:idx]['Volume'].mean()

        # Check criteria
        close_above_pivot = row['Close'] > pivot
        volume_spike = row['Volume'] > avg_volume * 1.5
        bullish_candle = row['Close'] > row['Open']

        day_range = row['High'] - row['Low']
        if day_range > 0:
            close_position = (row['Close'] - row['Low']) / day_range
            close_in_upper_half = close_position >= 0.5
        else:
            close_in_upper_half = True

        fresh_breakout = prev_row['Close'] < pivot

        return all([
            close_above_pivot,
            volume_spike,
            bullish_candle,
            close_in_upper_half,
            fresh_breakout
        ])

    def simulate_trade(
        self,
        df: pd.DataFrame,
        entry_idx: int,
        entry_price: float,
        stop_loss: float,
        target_price: float
    ) -> Tuple[Optional[str], Optional[float], Optional[str], int, float, float]:
        """
        Simulate a trade from entry to exit.

        Returns:
            exit_date, exit_price, exit_reason, days_held, max_gain_pct, max_drawdown_pct
        """
        if entry_idx >= len(df) - 1:
            return None, None, 'open', 0, 0, 0

        max_price = entry_price
        trailing_stop = None
        max_gain = 0
        max_dd = 0

        for i in range(entry_idx + 1, min(entry_idx + self.max_hold_days + 1, len(df))):
            row = df.iloc[i]
            days_held = i - entry_idx

            # Update max price and trailing stop
            if row['High'] > max_price:
                max_price = row['High']
                current_gain = (max_price - entry_price) / entry_price * 100
                max_gain = max(max_gain, current_gain)

                # Activate trailing stop if gain threshold reached
                if self.use_trailing_stop and current_gain >= self.trailing_activation_pct:
                    trailing_stop = max_price * (1 - self.trailing_stop_pct / 100)

            # Update max drawdown
            current_dd = (entry_price - row['Low']) / entry_price * 100
            max_dd = max(max_dd, current_dd)

            # Check stop loss hit (including trailing)
            effective_stop = stop_loss
            if trailing_stop and trailing_stop > stop_loss:
                effective_stop = trailing_stop

            if row['Low'] <= effective_stop:
                exit_price = effective_stop
                exit_reason = 'trailing_stop' if trailing_stop and trailing_stop > stop_loss else 'stop'
                return row.name.strftime('%Y-%m-%d'), exit_price, exit_reason, days_held, max_gain, max_dd

            # Check target hit (if not using trailing stop or before activation)
            if not self.use_trailing_stop or trailing_stop is None:
                if row['High'] >= target_price:
                    return row.name.strftime('%Y-%m-%d'), target_price, 'target', days_held, max_gain, max_dd

        # Time exit
        last_row = df.iloc[min(entry_idx + self.max_hold_days, len(df) - 1)]
        days_held = min(self.max_hold_days, len(df) - entry_idx - 1)
        return last_row.name.strftime('%Y-%m-%d'), last_row['Close'], 'time', days_held, max_gain, max_dd

    def backtest_symbol(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        spy_df: pd.DataFrame = None
    ) -> List[Trade]:
        """Run backtest on a single symbol"""
        trades = []

        try:
            df = self.cache.get_data(symbol, period='5y')
            if df is None or len(df) < 300:
                return trades

            # Filter to date range
            df = df[start_date:end_date]
            if len(df) < 150:
                return trades

            # Scan for VCP patterns and breakouts
            lookback = 120
            last_trade_exit_idx = -1

            for i in range(lookback + 50, len(df) - 5):
                # Skip if we're still in a trade
                if i <= last_trade_exit_idx:
                    continue

                # Get data up to this point for pattern detection
                df_to_date = df.iloc[:i+1]

                # Detect VCP pattern
                pattern = self.detector.analyze_pattern(df_to_date, lookback_days=lookback, spy_df=spy_df)

                if pattern is None:
                    continue

                # Filter by score and validity
                if self.require_valid_pattern and not pattern.is_valid:
                    continue

                if pattern.overall_score < self.min_overall_score:
                    continue

                # Check for breakout
                if not self.detect_breakout(df, pattern, i):
                    continue

                # We have a valid setup - enter trade
                entry_row = df.iloc[i]
                entry_price = entry_row['Close']
                entry_date = entry_row.name.strftime('%Y-%m-%d')

                # Calculate stop and target
                stop_loss = entry_price * (1 - self.stop_loss_pct / 100)
                risk = entry_price - stop_loss
                target_price = entry_price + (risk * self.target_rr)

                # Simulate trade
                exit_date, exit_price, exit_reason, days_held, max_gain, max_dd = self.simulate_trade(
                    df, i, entry_price, stop_loss, target_price
                )

                if exit_price is not None:
                    pnl_pct = (exit_price - entry_price) / entry_price * 100
                else:
                    pnl_pct = 0

                trade = Trade(
                    symbol=symbol,
                    entry_date=entry_date,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    target_price=target_price,
                    exit_date=exit_date,
                    exit_price=exit_price,
                    exit_reason=exit_reason,
                    pnl_pct=pnl_pct,
                    days_held=days_held,
                    max_gain_pct=max_gain,
                    max_drawdown_pct=max_dd,
                    overall_score=pattern.overall_score,
                    trend_template_score=pattern.trend_template_score,
                    prior_advance_pct=pattern.prior_uptrend.advance_pct,
                    num_contractions=pattern.num_contractions,
                    final_contraction_pct=pattern.final_contraction_pct,
                    base_duration_days=pattern.base_duration_days
                )

                trades.append(trade)

                # Update last trade exit - calculate from days held instead of date lookup
                if days_held:
                    last_trade_exit_idx = i + days_held
                else:
                    last_trade_exit_idx = i + 1

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
        """Run backtest across multiple symbols"""

        logger.info(f"Starting V4 backtest: {len(symbols)} symbols, {start_date} to {end_date}")

        # Get SPY data for RS calculation
        spy_df = self.cache.get_data('SPY', period='5y')

        all_trades = []

        # Process symbols
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
                    if trades:
                        logger.info(f"{symbol}: {len(trades)} trades")
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")

        # Calculate statistics
        if not all_trades:
            return BacktestResult(
                detector_version='V4',
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0,
                avg_win_pct=0,
                avg_loss_pct=0,
                profit_factor=0,
                total_return_pct=0,
                avg_days_held=0,
                avg_overall_score=0,
                trades=[]
            )

        winning = [t for t in all_trades if t.pnl_pct and t.pnl_pct > 0]
        losing = [t for t in all_trades if t.pnl_pct and t.pnl_pct < 0]

        total_wins = sum(t.pnl_pct for t in winning) if winning else 0
        total_losses = abs(sum(t.pnl_pct for t in losing)) if losing else 0

        result = BacktestResult(
            detector_version='V4',
            total_trades=len(all_trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=len(winning) / len(all_trades) * 100 if all_trades else 0,
            avg_win_pct=np.mean([t.pnl_pct for t in winning]) if winning else 0,
            avg_loss_pct=np.mean([t.pnl_pct for t in losing]) if losing else 0,
            profit_factor=total_wins / total_losses if total_losses > 0 else float('inf'),
            total_return_pct=sum(t.pnl_pct for t in all_trades if t.pnl_pct),
            avg_days_held=np.mean([t.days_held for t in all_trades if t.days_held]) if all_trades else 0,
            avg_overall_score=np.mean([t.overall_score for t in all_trades if t.overall_score]) if all_trades else 0,
            trades=all_trades
        )

        return result


def get_test_universe() -> List[str]:
    """Get list of stocks for backtesting"""
    # Large cap tech + quality growth stocks
    symbols = [
        # FAANG+
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
        # Semiconductors
        'AMD', 'AVGO', 'QCOM', 'INTC', 'MU', 'AMAT', 'LRCX', 'KLAC',
        # Software
        'CRM', 'ADBE', 'NOW', 'ORCL', 'INTU', 'PANW', 'CRWD', 'ZS',
        # Financials
        'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'AXP',
        # Healthcare
        'UNH', 'JNJ', 'LLY', 'PFE', 'ABBV', 'MRK', 'TMO', 'DHR',
        # Consumer
        'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'COST', 'WMT', 'PG',
        # Industrial
        'CAT', 'DE', 'GE', 'HON', 'UNP', 'UPS', 'LMT', 'RTX',
        # Energy
        'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD',
        # Other growth
        'NFLX', 'PYPL', 'SQ', 'SHOP', 'UBER', 'ABNB', 'COIN'
    ]
    return symbols


def main():
    """Run V4 backtest and display results"""
    print("="*70)
    print("VCP Backtest V4 - Full Minervini Methodology")
    print("="*70)

    # Configuration
    symbols = get_test_universe()
    start_date = '2020-01-01'
    end_date = '2024-12-31'

    print(f"\nConfiguration:")
    print(f"  Symbols: {len(symbols)}")
    print(f"  Period: {start_date} to {end_date}")
    print(f"  Stop Loss: 7%")
    print(f"  Target R:R: 3:1")
    print(f"  Trailing Stop: Yes (8% activation, 5% trail)")
    print(f"  Min Overall Score: 70")
    print(f"  Require Valid Pattern: Yes (trend template + prior uptrend)")

    # Run backtest
    engine = BacktestEngineV4(
        stop_loss_pct=7.0,
        target_rr=3.0,
        max_hold_days=60,
        use_trailing_stop=True,
        trailing_activation_pct=8.0,
        trailing_stop_pct=5.0,
        min_overall_score=70.0,
        require_valid_pattern=True
    )

    result = engine.run_backtest(symbols, start_date, end_date)

    # Display results
    print(f"\n{'='*70}")
    print("BACKTEST RESULTS - V4 DETECTOR")
    print('='*70)

    print(f"\n--- Trade Statistics ---")
    print(f"Total Trades: {result.total_trades}")
    print(f"Winning Trades: {result.winning_trades}")
    print(f"Losing Trades: {result.losing_trades}")
    print(f"Win Rate: {result.win_rate:.1f}%")

    print(f"\n--- P&L Statistics ---")
    print(f"Avg Win: {result.avg_win_pct:.1f}%")
    print(f"Avg Loss: {result.avg_loss_pct:.1f}%")
    print(f"Profit Factor: {result.profit_factor:.2f}")
    print(f"Total Return: {result.total_return_pct:.1f}%")

    print(f"\n--- Pattern Quality ---")
    print(f"Avg Overall Score: {result.avg_overall_score:.1f}")
    print(f"Avg Days Held: {result.avg_days_held:.1f}")

    # Save results
    os.makedirs('results/v4', exist_ok=True)

    # Save trades to CSV
    if result.trades:
        trades_df = pd.DataFrame([asdict(t) for t in result.trades])
        trades_df.to_csv('results/v4/trades.csv', index=False)
        print(f"\nTrades saved to results/v4/trades.csv")

        # Show sample trades
        print(f"\n--- Sample Trades ---")
        sample = trades_df.head(10)
        for _, t in sample.iterrows():
            outcome = "WIN" if t['pnl_pct'] > 0 else "LOSS"
            print(f"  {t['symbol']} {t['entry_date']}: {t['pnl_pct']:.1f}% ({outcome}) "
                  f"Score: {t['overall_score']:.0f}, Prior Adv: {t['prior_advance_pct']:.1f}%")

    # Save summary
    summary = {
        'detector_version': 'V4',
        'total_trades': result.total_trades,
        'winning_trades': result.winning_trades,
        'losing_trades': result.losing_trades,
        'win_rate': result.win_rate,
        'avg_win_pct': result.avg_win_pct,
        'avg_loss_pct': result.avg_loss_pct,
        'profit_factor': result.profit_factor,
        'total_return_pct': result.total_return_pct,
        'avg_days_held': result.avg_days_held,
        'avg_overall_score': result.avg_overall_score,
        'config': {
            'stop_loss_pct': 7.0,
            'target_rr': 3.0,
            'use_trailing_stop': True,
            'min_overall_score': 70.0,
            'require_valid_pattern': True,
            'start_date': start_date,
            'end_date': end_date,
            'num_symbols': len(symbols)
        }
    }

    with open('results/v4/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to results/v4/summary.json")

    # Also run with relaxed settings for comparison
    print(f"\n{'='*70}")
    print("Running comparison with RELAXED settings...")
    print('='*70)

    engine_relaxed = BacktestEngineV4(
        stop_loss_pct=7.0,
        target_rr=3.0,
        max_hold_days=60,
        use_trailing_stop=True,
        trailing_activation_pct=8.0,
        trailing_stop_pct=5.0,
        min_overall_score=50.0,  # Lower threshold
        require_valid_pattern=False  # Don't require all criteria
    )

    result_relaxed = engine_relaxed.run_backtest(symbols, start_date, end_date)

    print(f"\n--- Relaxed Settings Results ---")
    print(f"Total Trades: {result_relaxed.total_trades}")
    print(f"Win Rate: {result_relaxed.win_rate:.1f}%")
    print(f"Profit Factor: {result_relaxed.profit_factor:.2f}")
    print(f"Total Return: {result_relaxed.total_return_pct:.1f}%")

    if result_relaxed.trades:
        trades_df_relaxed = pd.DataFrame([asdict(t) for t in result_relaxed.trades])
        trades_df_relaxed.to_csv('results/v4/trades_relaxed.csv', index=False)

    return result, result_relaxed


if __name__ == '__main__':
    main()
