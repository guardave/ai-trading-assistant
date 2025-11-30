#!/usr/bin/env python3
"""
Comprehensive VCP Strategy Analysis

This script performs the following investigations:
1. V2 vs V3 detector comparison
2. Market regime filter testing (SPY > 200 MA)
3. Relaxed breakout criteria testing
4. Extended history backtest (2020-2024)
5. Trailing stop parameter optimization

Date: 2025-11-30
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
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.vcp_detector import VCPDetector, VCPPattern
from backtest.strategy_params import VCP_PARAMS, RISK_PARAMS, RS_PARAMS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Output directory
OUTPUT_DIR = 'results/comprehensive_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/charts', exist_ok=True)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Trade:
    """Represents a single trade"""
    symbol: str
    pattern: str
    detector: str  # 'v2' or 'v3'
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
    rs_rating: Optional[float] = None
    volume_ratio: Optional[float] = None
    num_contractions: Optional[int] = None
    proximity_score: Optional[float] = None
    # Market regime
    spy_above_200ma: Optional[bool] = None
    spy_distance_from_200ma: Optional[float] = None


@dataclass
class BacktestConfig:
    """Configuration for a backtest run"""
    name: str
    detector: str  # 'v2', 'v3', or 'both'
    rs_min: float = 70
    proximity_min: float = 0
    exit_method: str = 'trailing_stop'
    stop_loss_pct: float = 7.0
    trailing_activation_pct: float = 8.0
    trailing_distance_pct: float = 5.0
    target_pct: float = 21.0
    close_position_threshold: float = 0.5  # Upper % of day's range
    require_fresh_breakout: bool = True
    market_regime_filter: bool = False
    start_date: str = '2023-01-01'
    end_date: str = '2025-11-30'


# =============================================================================
# Data Cache with Extended History
# =============================================================================

class ExtendedDataCache:
    """Cache for historical price data with extended period support"""

    def __init__(self, cache_dir: str = 'cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.memory_cache: Dict[str, pd.DataFrame] = {}

    def get_historical_data(self, symbol: str, start: str = '2019-01-01',
                           end: str = None) -> Optional[pd.DataFrame]:
        """Get historical data with date range support"""
        if end is None:
            end = datetime.now().strftime('%Y-%m-%d')

        cache_key = f"{symbol}_{start}_{end}"

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
            except Exception as e:
                logger.warning(f"Error reading cache for {symbol}: {e}")

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start, end=end)

            if df.empty:
                logger.warning(f"No data for {symbol}")
                return None

            df.columns = [c.title() for c in df.columns]
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.to_parquet(cache_file)
            self.memory_cache[cache_key] = df

            return df

        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return None


# =============================================================================
# V2 Detector (Rolling Window Method)
# =============================================================================

class V2VCPAnalyzer:
    """V2 VCP detection using rolling window method (original approach)"""

    def detect_contractions(self, df: pd.DataFrame, lookback_days: int = 120) -> List[Dict]:
        """Detect contractions using rolling window approach"""
        if len(df) < lookback_days:
            return []

        df_slice = df.tail(lookback_days).copy()
        df_slice['Volume_Avg'] = df['Volume'].rolling(window=50).mean()

        contractions = []
        windows = [10, 15, 20, 25]

        for window in windows:
            df_slice[f'High_{window}'] = df_slice['High'].rolling(window=window).max()
            df_slice[f'Low_{window}'] = df_slice['Low'].rolling(window=window).min()
            df_slice[f'Range_{window}'] = (
                (df_slice[f'High_{window}'] - df_slice[f'Low_{window}']) /
                df_slice[f'Low_{window}'] * 100
            )

        df_slice['Range_20'] = df_slice['Range_20'].fillna(method='bfill')
        range_series = df_slice['Range_20'].values
        vol_avg = df_slice['Volume_Avg'].values

        i = 0
        while i < len(range_series) - 20:
            window_range = range_series[i:i+20]

            if len(window_range) < 20:
                break

            max_range = np.max(window_range[:10])
            min_range = np.min(window_range[10:])

            if max_range > 0 and (max_range - min_range) / max_range > 0.3:
                start_idx = i
                end_idx = i + 20

                high = df_slice['High'].iloc[start_idx:end_idx].max()
                low = df_slice['Low'].iloc[start_idx:end_idx].min()
                range_pct = (high - low) / low * 100

                contraction_vol = df_slice['Volume'].iloc[start_idx:end_idx].mean()
                avg_vol = vol_avg[end_idx] if not np.isnan(vol_avg[end_idx]) else contraction_vol
                vol_ratio = contraction_vol / avg_vol if avg_vol > 0 else 1.0

                contractions.append({
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'high': high,
                    'low': low,
                    'range_pct': range_pct,
                    'avg_volume_ratio': vol_ratio
                })

                i = end_idx
            else:
                i += 5

        return contractions

    def analyze_vcp(self, df: pd.DataFrame, check_date_idx: int) -> Optional[Dict]:
        """Perform VCP analysis using V2 method"""
        if check_date_idx < 120:
            return None

        df_slice = df.iloc[max(0, check_date_idx - 120):check_date_idx + 1]
        contractions = self.detect_contractions(df_slice)

        if len(contractions) < 2:
            return None

        contractions.sort(key=lambda x: x['end_idx'])
        recent = contractions[-4:] if len(contractions) >= 4 else contractions[-2:]

        ranges = [c['range_pct'] for c in recent]
        sequence_valid = all(ranges[i] >= ranges[i+1] for i in range(len(ranges)-1))

        # Calculate proximity score (V2 method)
        score = 100.0

        # Gap penalty
        total_gap = 0
        for i in range(len(recent) - 1):
            gap = recent[i+1]['start_idx'] - recent[i]['end_idx']
            total_gap += max(0, gap)
        avg_gap = total_gap / (len(recent) - 1) if len(recent) > 1 else 0
        gap_penalty = min(40, avg_gap * 4)
        score -= gap_penalty

        # Sequence penalty
        violations = sum(1 for i in range(len(ranges)-1) if ranges[i+1] > ranges[i])
        score -= min(30, violations * 15)

        # Volume penalty
        volumes = [c['avg_volume_ratio'] for c in recent]
        vol_increasing = sum(1 for i in range(len(volumes)-1) if volumes[i+1] > volumes[i])
        score -= min(30, vol_increasing * 10)

        pivot_price = max(c['high'] for c in recent)

        return {
            'num_contractions': len(recent),
            'sequence_valid': sequence_valid,
            'proximity_score': max(0, min(100, score)),
            'final_range': recent[-1]['range_pct'],
            'pivot_price': pivot_price,
            'support_price': recent[-1]['low']
        }


# =============================================================================
# RS Rating Calculator
# =============================================================================

def calculate_rs_rating(df: pd.DataFrame, spy_df: pd.DataFrame) -> float:
    """Calculate RS rating vs SPY benchmark"""
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

    except Exception:
        return 50.0


# =============================================================================
# Market Regime Filter
# =============================================================================

def check_market_regime(spy_df: pd.DataFrame, check_idx: int) -> Tuple[bool, float]:
    """Check if market is in bullish regime (SPY > 200 MA)"""
    if spy_df is None or check_idx < 200:
        return True, 0.0

    spy_slice = spy_df.iloc[:check_idx + 1]
    if len(spy_slice) < 200:
        return True, 0.0

    spy_close = spy_slice['Close'].iloc[-1]
    spy_200ma = spy_slice['Close'].rolling(200).mean().iloc[-1]

    if pd.isna(spy_200ma):
        return True, 0.0

    distance = (spy_close - spy_200ma) / spy_200ma * 100
    return spy_close > spy_200ma, distance


# =============================================================================
# Unified Scanner
# =============================================================================

class UnifiedScanner:
    """Scanner that can use either V2 or V3 detection"""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.v2_analyzer = V2VCPAnalyzer()
        self.v3_detector = VCPDetector(
            swing_lookback=5,
            min_contractions=2,
            max_contraction_range=20.0
        )

    def scan_v2(self, df: pd.DataFrame, spy_df: pd.DataFrame,
                check_date_idx: int) -> Optional[Dict]:
        """Scan using V2 detector"""
        if check_date_idx < 252:
            return None

        df_slice = df.iloc[:check_date_idx + 1].copy()

        if len(df_slice) < 120:
            return None

        # Calculate indicators
        df_slice['SMA_200'] = df_slice['Close'].rolling(window=200).mean()
        df_slice['Volume_Avg'] = df_slice['Volume'].rolling(window=50).mean()

        current_price = df_slice['Close'].iloc[-1]
        current_volume = df_slice['Volume'].iloc[-1]
        avg_volume = df_slice['Volume_Avg'].iloc[-1]

        if pd.isna(avg_volume) or avg_volume == 0:
            return None

        # Price above 200 MA
        if len(df_slice) >= 200:
            sma_200 = df_slice['SMA_200'].iloc[-1]
            if pd.notna(sma_200) and current_price < sma_200:
                return None

        # Distance from 52-week high
        lookback = min(252, len(df_slice))
        high_52w = df_slice['High'].tail(lookback).max()
        distance_from_high = (high_52w - current_price) / high_52w

        if distance_from_high > 0.25:
            return None

        # V2 VCP analysis
        vcp = self.v2_analyzer.analyze_vcp(df, check_date_idx)

        if vcp is None:
            return None

        if vcp['num_contractions'] < 2:
            return None

        if vcp['final_range'] > 20.0:
            return None

        if vcp['proximity_score'] < self.config.proximity_min:
            return None

        # RS rating
        spy_slice = spy_df.iloc[:check_date_idx + 1] if spy_df is not None else None
        rs_rating = calculate_rs_rating(df_slice, spy_slice)

        if rs_rating < self.config.rs_min:
            return None

        # Breakout check (V2 style - looser)
        if len(df_slice) >= 2:
            if current_price <= df_slice['High'].iloc[-2]:
                return None
            if current_volume <= avg_volume * 1.5:
                return None

        return {
            'pattern': 'VCP',
            'detector': 'v2',
            'price': float(current_price),
            'pivot_price': float(vcp['pivot_price']),
            'volume_ratio': float(current_volume / avg_volume),
            'rs_rating': float(rs_rating),
            'date': df_slice.index[-1],
            'num_contractions': vcp['num_contractions'],
            'proximity_score': vcp['proximity_score'],
            'support_price': vcp['support_price']
        }

    def scan_v3(self, df: pd.DataFrame, spy_df: pd.DataFrame,
                check_date_idx: int) -> Optional[Dict]:
        """Scan using V3 detector with configurable breakout criteria"""
        if check_date_idx < 252:
            return None

        df_slice = df.iloc[:check_date_idx + 1].copy()

        if len(df_slice) < 120:
            return None

        # Calculate indicators
        df_slice['SMA_200'] = df_slice['Close'].rolling(window=200).mean()
        df_slice['Volume_Avg'] = df_slice['Volume'].rolling(window=50).mean()

        current_price = df_slice['Close'].iloc[-1]
        current_volume = df_slice['Volume'].iloc[-1]
        avg_volume = df_slice['Volume_Avg'].iloc[-1]
        current_open = df_slice['Open'].iloc[-1]
        current_high = df_slice['High'].iloc[-1]
        current_low = df_slice['Low'].iloc[-1]

        if pd.isna(avg_volume) or avg_volume == 0:
            return None

        # Price above 200 MA
        if len(df_slice) >= 200:
            sma_200 = df_slice['SMA_200'].iloc[-1]
            if pd.notna(sma_200) and current_price < sma_200:
                return None

        # Distance from 52-week high
        lookback = min(252, len(df_slice))
        high_52w = df_slice['High'].tail(lookback).max()
        distance_from_high = (high_52w - current_price) / high_52w

        if distance_from_high > 0.25:
            return None

        # V3 VCP detection
        pattern = self.v3_detector.analyze_pattern(df_slice, lookback_days=120)

        if pattern is None or not pattern.is_valid:
            return None

        if pattern.proximity_score < self.config.proximity_min:
            return None

        # RS rating
        spy_slice = spy_df.iloc[:check_date_idx + 1] if spy_df is not None else None
        rs_rating = calculate_rs_rating(df_slice, spy_slice)

        if rs_rating < self.config.rs_min:
            return None

        # Breakout criteria
        pivot_price = pattern.pivot_price

        # 1. Close above pivot
        if current_price < pivot_price:
            return None

        # 2. Volume confirmation
        if current_volume < avg_volume * 1.5:
            return None

        # 3. Bullish candle
        if current_price <= current_open:
            return None

        # 4. Close position threshold (configurable)
        day_range = current_high - current_low
        if day_range > 0:
            close_position = (current_price - current_low) / day_range
            if close_position < self.config.close_position_threshold:
                return None

        # 5. Fresh breakout (configurable)
        if self.config.require_fresh_breakout and len(df_slice) >= 2:
            prev_close = df_slice['Close'].iloc[-2]
            if prev_close >= pivot_price:
                return None

        return {
            'pattern': 'VCP',
            'detector': 'v3',
            'price': float(current_price),
            'pivot_price': float(pivot_price),
            'volume_ratio': float(current_volume / avg_volume),
            'rs_rating': float(rs_rating),
            'date': df_slice.index[-1],
            'num_contractions': len(pattern.contractions),
            'proximity_score': pattern.proximity_score,
            'contraction_quality': pattern.contraction_quality,
            'volume_quality': pattern.volume_quality,
            'support_price': float(pattern.support_price)
        }

    def scan(self, df: pd.DataFrame, spy_df: pd.DataFrame,
             check_date_idx: int) -> Optional[Dict]:
        """Scan using configured detector"""
        if self.config.detector == 'v2':
            return self.scan_v2(df, spy_df, check_date_idx)
        elif self.config.detector == 'v3':
            return self.scan_v3(df, spy_df, check_date_idx)
        else:
            # Try both
            signal = self.scan_v3(df, spy_df, check_date_idx)
            if signal is None:
                signal = self.scan_v2(df, spy_df, check_date_idx)
            return signal


# =============================================================================
# Trade Simulator
# =============================================================================

class TradeSimulator:
    """Simulates trades with various exit methods"""

    def simulate_trailing_stop(self, df: pd.DataFrame, signal: Dict,
                               config: BacktestConfig) -> Trade:
        """Simulate trade with trailing stop"""
        entry_date = signal['date']
        entry_price = signal['price']
        initial_stop = entry_price * (1 - config.stop_loss_pct / 100)
        current_stop = initial_stop

        try:
            entry_idx = df.index.get_loc(entry_date)
        except KeyError:
            entry_idx = df.index.searchsorted(entry_date)

        trade = Trade(
            symbol=signal.get('symbol', 'UNKNOWN'),
            pattern=signal['pattern'],
            detector=signal.get('detector', 'unknown'),
            entry_date=str(entry_date)[:10],
            entry_price=entry_price,
            stop_loss=initial_stop,
            target_price=0,
            rs_rating=signal.get('rs_rating'),
            volume_ratio=signal.get('volume_ratio'),
            num_contractions=signal.get('num_contractions'),
            proximity_score=signal.get('proximity_score'),
            spy_above_200ma=signal.get('spy_above_200ma'),
            spy_distance_from_200ma=signal.get('spy_distance_from_200ma')
        )

        max_price = entry_price
        min_price = entry_price
        trailing_active = False

        for day_offset in range(1, 91):
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

            if high > max_price:
                max_price = high
                gain_pct = ((max_price - entry_price) / entry_price) * 100

                if gain_pct >= config.trailing_activation_pct and not trailing_active:
                    trailing_active = True
                    current_stop = entry_price

                if trailing_active:
                    new_stop = max_price * (1 - config.trailing_distance_pct / 100)
                    if new_stop > current_stop:
                        current_stop = new_stop

            min_price = min(min_price, low)

            if low <= current_stop:
                trade.exit_date = str(df.index[check_idx])[:10]
                trade.exit_price = current_stop
                trade.exit_reason = 'trailing_stop' if trailing_active else 'stop'
                trade.days_held = day_offset
                break

            if day_offset >= 90:
                trade.exit_date = str(df.index[check_idx])[:10]
                trade.exit_price = close
                trade.exit_reason = 'time'
                trade.days_held = day_offset
                break

        if trade.exit_price is not None:
            trade.pnl_pct = ((trade.exit_price - entry_price) / entry_price) * 100

        trade.max_gain_pct = ((max_price - entry_price) / entry_price) * 100
        trade.max_drawdown_pct = ((entry_price - min_price) / entry_price) * 100

        return trade

    def simulate_fixed_target(self, df: pd.DataFrame, signal: Dict,
                              config: BacktestConfig) -> Trade:
        """Simulate trade with fixed stop and target"""
        entry_date = signal['date']
        entry_price = signal['price']
        stop_loss = entry_price * (1 - config.stop_loss_pct / 100)
        target_price = entry_price * (1 + config.target_pct / 100)

        try:
            entry_idx = df.index.get_loc(entry_date)
        except KeyError:
            entry_idx = df.index.searchsorted(entry_date)

        trade = Trade(
            symbol=signal.get('symbol', 'UNKNOWN'),
            pattern=signal['pattern'],
            detector=signal.get('detector', 'unknown'),
            entry_date=str(entry_date)[:10],
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_price=target_price,
            rs_rating=signal.get('rs_rating'),
            volume_ratio=signal.get('volume_ratio'),
            num_contractions=signal.get('num_contractions'),
            proximity_score=signal.get('proximity_score'),
            spy_above_200ma=signal.get('spy_above_200ma'),
            spy_distance_from_200ma=signal.get('spy_distance_from_200ma')
        )

        max_price = entry_price
        min_price = entry_price

        for day_offset in range(1, 61):
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

            if day_offset >= 60:
                trade.exit_date = str(df.index[check_idx])[:10]
                trade.exit_price = close
                trade.exit_reason = 'time'
                trade.days_held = day_offset
                break

        if trade.exit_price is not None:
            trade.pnl_pct = ((trade.exit_price - entry_price) / entry_price) * 100

        trade.max_gain_pct = ((max_price - entry_price) / entry_price) * 100
        trade.max_drawdown_pct = ((entry_price - min_price) / entry_price) * 100

        return trade


# =============================================================================
# Backtester
# =============================================================================

class ComprehensiveBacktester:
    """Main backtesting engine"""

    def __init__(self, data_cache: ExtendedDataCache):
        self.data_cache = data_cache
        self.simulator = TradeSimulator()

    def run_backtest(self, symbols: List[str], config: BacktestConfig) -> Dict:
        """Run backtest with given configuration"""
        scanner = UnifiedScanner(config)
        all_trades: List[Trade] = []

        # Get SPY data for market regime
        spy_df = self.data_cache.get_historical_data(
            'SPY',
            start='2019-01-01',
            end=config.end_date
        )

        start_dt = pd.Timestamp(config.start_date)
        end_dt = pd.Timestamp(config.end_date)

        logger.info(f"Running {config.name} on {len(symbols)} symbols...")

        for i, symbol in enumerate(symbols):
            if (i + 1) % 20 == 0:
                logger.info(f"  Processing {i+1}/{len(symbols)} symbols...")

            df = self.data_cache.get_historical_data(
                symbol,
                start='2019-01-01',
                end=config.end_date
            )

            if df is None or len(df) < 252:
                continue

            last_signal_idx = -60

            for idx in range(252, len(df) - 10):
                check_date = df.index[idx]

                # Normalize for comparison
                check_date_norm = check_date.tz_localize(None) if hasattr(check_date, 'tz_localize') and check_date.tzinfo else check_date

                if check_date_norm < start_dt or check_date_norm > end_dt:
                    continue

                if idx - last_signal_idx < 20:
                    continue

                # Market regime check
                if config.market_regime_filter:
                    above_200ma, distance = check_market_regime(spy_df, idx)
                    if not above_200ma:
                        continue
                else:
                    above_200ma, distance = check_market_regime(spy_df, idx)

                signal = scanner.scan(df, spy_df, idx)

                if signal:
                    signal['symbol'] = symbol
                    signal['spy_above_200ma'] = above_200ma
                    signal['spy_distance_from_200ma'] = distance
                    last_signal_idx = idx

                    if config.exit_method == 'fixed_target':
                        trade = self.simulator.simulate_fixed_target(df, signal, config)
                    else:
                        trade = self.simulator.simulate_trailing_stop(df, signal, config)

                    all_trades.append(trade)

        return self._calculate_stats(config, all_trades)

    def _calculate_stats(self, config: BacktestConfig, trades: List[Trade]) -> Dict:
        """Calculate backtest statistics"""
        if not trades:
            return {
                'config': config.name,
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_return': 0,
                'trades': []
            }

        closed = [t for t in trades if t.exit_reason != 'open']
        winning = [t for t in closed if t.pnl_pct and t.pnl_pct > 0]
        losing = [t for t in closed if t.pnl_pct and t.pnl_pct <= 0]

        win_rate = len(winning) / len(closed) * 100 if closed else 0
        avg_win = np.mean([t.pnl_pct for t in winning]) if winning else 0
        avg_loss = np.mean([t.pnl_pct for t in losing]) if losing else 0

        total_wins = sum(t.pnl_pct for t in winning) if winning else 0
        total_losses = abs(sum(t.pnl_pct for t in losing)) if losing else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        total_return = sum(t.pnl_pct for t in closed if t.pnl_pct)
        avg_days = np.mean([t.days_held for t in closed if t.days_held]) if closed else 0

        # Monthly breakdown
        monthly = {}
        for t in closed:
            if t.entry_date and t.pnl_pct:
                month = t.entry_date[:7]
                if month not in monthly:
                    monthly[month] = {'trades': 0, 'wins': 0, 'pnl': 0}
                monthly[month]['trades'] += 1
                if t.pnl_pct > 0:
                    monthly[month]['wins'] += 1
                monthly[month]['pnl'] += t.pnl_pct

        # Market regime breakdown
        regime_bull = [t for t in closed if t.spy_above_200ma]
        regime_bear = [t for t in closed if not t.spy_above_200ma]

        bull_win_rate = sum(1 for t in regime_bull if t.pnl_pct and t.pnl_pct > 0) / len(regime_bull) * 100 if regime_bull else 0
        bear_win_rate = sum(1 for t in regime_bear if t.pnl_pct and t.pnl_pct > 0) / len(regime_bear) * 100 if regime_bear else 0

        return {
            'config': config.name,
            'detector': config.detector,
            'total_trades': len(trades),
            'closed_trades': len(closed),
            'winning_trades': len(winning),
            'losing_trades': len(losing),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_return': total_return,
            'avg_days_held': avg_days,
            'monthly': monthly,
            'bull_market_trades': len(regime_bull),
            'bull_market_win_rate': bull_win_rate,
            'bear_market_trades': len(regime_bear),
            'bear_market_win_rate': bear_win_rate,
            'trades': [asdict(t) for t in trades]
        }


# =============================================================================
# Visualization
# =============================================================================

def create_comparison_chart(results: List[Dict], title: str, filename: str):
    """Create comparison bar chart"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    names = [r['config'] for r in results]

    # Win Rate
    ax1 = axes[0, 0]
    win_rates = [r['win_rate'] for r in results]
    bars1 = ax1.bar(names, win_rates, color='steelblue')
    ax1.set_title('Win Rate (%)')
    ax1.set_ylabel('%')
    ax1.axhline(y=50, color='red', linestyle='--', alpha=0.5)
    for bar, val in zip(bars1, win_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    ax1.tick_params(axis='x', rotation=45)

    # Profit Factor
    ax2 = axes[0, 1]
    pfs = [r['profit_factor'] for r in results]
    colors = ['green' if pf >= 1 else 'red' for pf in pfs]
    bars2 = ax2.bar(names, pfs, color=colors)
    ax2.set_title('Profit Factor')
    ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    for bar, val in zip(bars2, pfs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    ax2.tick_params(axis='x', rotation=45)

    # Total Return
    ax3 = axes[1, 0]
    returns = [r['total_return'] for r in results]
    colors = ['green' if ret >= 0 else 'red' for ret in returns]
    bars3 = ax3.bar(names, returns, color=colors)
    ax3.set_title('Total Return (%)')
    ax3.set_ylabel('%')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    for bar, val in zip(bars3, returns):
        ax3.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (5 if val >= 0 else -10),
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    ax3.tick_params(axis='x', rotation=45)

    # Trade Count
    ax4 = axes[1, 1]
    trades = [r['total_trades'] for r in results]
    bars4 = ax4.bar(names, trades, color='purple', alpha=0.7)
    ax4.set_title('Total Trades')
    for bar, val in zip(bars4, trades):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(val), ha='center', va='bottom', fontsize=9)
    ax4.tick_params(axis='x', rotation=45)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved chart: {filename}")


def create_monthly_performance_chart(results: Dict, filename: str):
    """Create monthly performance chart"""
    monthly = results.get('monthly', {})
    if not monthly:
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    months = sorted(monthly.keys())
    win_rates = [monthly[m]['wins'] / monthly[m]['trades'] * 100 if monthly[m]['trades'] > 0 else 0 for m in months]
    pnls = [monthly[m]['pnl'] for m in months]
    trades = [monthly[m]['trades'] for m in months]

    # Win rate by month
    ax1 = axes[0]
    colors = ['green' if wr >= 50 else 'red' for wr in win_rates]
    bars1 = ax1.bar(months, win_rates, color=colors, alpha=0.7)
    ax1.axhline(y=50, color='black', linestyle='--', alpha=0.5)
    ax1.set_title(f"{results['config']} - Monthly Win Rate")
    ax1.set_ylabel('Win Rate (%)')
    ax1.tick_params(axis='x', rotation=45)

    # Add trade count labels
    for bar, t in zip(bars1, trades):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'n={t}', ha='center', va='bottom', fontsize=8)

    # Cumulative P&L
    ax2 = axes[1]
    cumulative = np.cumsum(pnls)
    ax2.plot(months, cumulative, 'b-o', linewidth=2, markersize=6)
    ax2.fill_between(months, 0, cumulative, alpha=0.3,
                     where=[c >= 0 for c in cumulative], color='green')
    ax2.fill_between(months, 0, cumulative, alpha=0.3,
                     where=[c < 0 for c in cumulative], color='red')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_title('Cumulative Return')
    ax2.set_ylabel('Return (%)')
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved chart: {filename}")


def create_market_regime_chart(results: List[Dict], filename: str):
    """Create market regime comparison chart"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    names = [r['config'] for r in results]

    # Bull market performance
    ax1 = axes[0]
    bull_rates = [r['bull_market_win_rate'] for r in results]
    bull_trades = [r['bull_market_trades'] for r in results]
    bars1 = ax1.bar(names, bull_rates, color='green', alpha=0.7)
    ax1.set_title('Bull Market (SPY > 200 MA)\nWin Rate')
    ax1.set_ylabel('Win Rate (%)')
    ax1.axhline(y=50, color='red', linestyle='--', alpha=0.5)
    for bar, wr, t in zip(bars1, bull_rates, bull_trades):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{wr:.1f}%\n(n={t})', ha='center', va='bottom', fontsize=9)
    ax1.tick_params(axis='x', rotation=45)

    # Bear market performance
    ax2 = axes[1]
    bear_rates = [r['bear_market_win_rate'] for r in results]
    bear_trades = [r['bear_market_trades'] for r in results]
    bars2 = ax2.bar(names, bear_rates, color='red', alpha=0.7)
    ax2.set_title('Bear Market (SPY < 200 MA)\nWin Rate')
    ax2.set_ylabel('Win Rate (%)')
    ax2.axhline(y=50, color='green', linestyle='--', alpha=0.5)
    for bar, wr, t in zip(bars2, bear_rates, bear_trades):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{wr:.1f}%\n(n={t})', ha='center', va='bottom', fontsize=9)
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved chart: {filename}")


# =============================================================================
# Stock Universe
# =============================================================================

def get_stock_universe() -> List[str]:
    """Get test stock universe"""
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
# Main Analysis
# =============================================================================

def run_comprehensive_analysis():
    """Run all analysis tests"""
    print("\n" + "="*80)
    print("  COMPREHENSIVE VCP STRATEGY ANALYSIS")
    print("  Date: 2025-11-30")
    print("="*80)

    data_cache = ExtendedDataCache('cache')
    backtester = ComprehensiveBacktester(data_cache)
    symbols = get_stock_universe()

    all_results = []

    # =========================================================================
    # Test 1: V2 vs V3 Detector Comparison (2023-present)
    # =========================================================================
    print("\n" + "-"*80)
    print("TEST 1: V2 vs V3 Detector Comparison")
    print("-"*80)

    configs_v2v3 = [
        BacktestConfig(
            name='V2_RS70_Trailing',
            detector='v2',
            rs_min=70,
            exit_method='trailing_stop',
            start_date='2023-01-01'
        ),
        BacktestConfig(
            name='V3_RS70_Trailing',
            detector='v3',
            rs_min=70,
            exit_method='trailing_stop',
            start_date='2023-01-01'
        ),
        BacktestConfig(
            name='V3_RS70_Relaxed',
            detector='v3',
            rs_min=70,
            exit_method='trailing_stop',
            close_position_threshold=0.4,  # Relaxed from 0.5
            require_fresh_breakout=False,
            start_date='2023-01-01'
        ),
    ]

    results_v2v3 = []
    for config in configs_v2v3:
        result = backtester.run_backtest(symbols, config)
        results_v2v3.append(result)
        print(f"\n{config.name}:")
        print(f"  Trades: {result['total_trades']}, Win Rate: {result['win_rate']:.1f}%")
        print(f"  Profit Factor: {result['profit_factor']:.2f}, Return: {result['total_return']:.1f}%")

    all_results.extend(results_v2v3)
    create_comparison_chart(results_v2v3, 'V2 vs V3 Detector Comparison',
                           f'{OUTPUT_DIR}/charts/v2_vs_v3_comparison.png')

    # =========================================================================
    # Test 2: Market Regime Filter
    # =========================================================================
    print("\n" + "-"*80)
    print("TEST 2: Market Regime Filter (SPY > 200 MA)")
    print("-"*80)

    configs_regime = [
        BacktestConfig(
            name='V3_No_Filter',
            detector='v3',
            rs_min=70,
            market_regime_filter=False,
            start_date='2023-01-01'
        ),
        BacktestConfig(
            name='V3_With_Filter',
            detector='v3',
            rs_min=70,
            market_regime_filter=True,
            start_date='2023-01-01'
        ),
        BacktestConfig(
            name='V2_No_Filter',
            detector='v2',
            rs_min=70,
            market_regime_filter=False,
            start_date='2023-01-01'
        ),
        BacktestConfig(
            name='V2_With_Filter',
            detector='v2',
            rs_min=70,
            market_regime_filter=True,
            start_date='2023-01-01'
        ),
    ]

    results_regime = []
    for config in configs_regime:
        result = backtester.run_backtest(symbols, config)
        results_regime.append(result)
        print(f"\n{config.name}:")
        print(f"  Trades: {result['total_trades']}, Win Rate: {result['win_rate']:.1f}%")
        print(f"  Profit Factor: {result['profit_factor']:.2f}, Return: {result['total_return']:.1f}%")
        print(f"  Bull Market Win Rate: {result['bull_market_win_rate']:.1f}% (n={result['bull_market_trades']})")
        print(f"  Bear Market Win Rate: {result['bear_market_win_rate']:.1f}% (n={result['bear_market_trades']})")

    all_results.extend(results_regime)
    create_comparison_chart(results_regime, 'Market Regime Filter Impact',
                           f'{OUTPUT_DIR}/charts/market_regime_comparison.png')
    create_market_regime_chart(results_regime, f'{OUTPUT_DIR}/charts/bull_bear_breakdown.png')

    # =========================================================================
    # Test 3: Extended History (2020-2024)
    # =========================================================================
    print("\n" + "-"*80)
    print("TEST 3: Extended History (2020-2024)")
    print("-"*80)

    configs_extended = [
        BacktestConfig(
            name='V2_2020-2024',
            detector='v2',
            rs_min=70,
            start_date='2020-01-01',
            end_date='2024-12-31'
        ),
        BacktestConfig(
            name='V3_2020-2024',
            detector='v3',
            rs_min=70,
            start_date='2020-01-01',
            end_date='2024-12-31'
        ),
        BacktestConfig(
            name='V3_2020-2024_Filter',
            detector='v3',
            rs_min=70,
            market_regime_filter=True,
            start_date='2020-01-01',
            end_date='2024-12-31'
        ),
    ]

    results_extended = []
    for config in configs_extended:
        result = backtester.run_backtest(symbols, config)
        results_extended.append(result)
        print(f"\n{config.name}:")
        print(f"  Trades: {result['total_trades']}, Win Rate: {result['win_rate']:.1f}%")
        print(f"  Profit Factor: {result['profit_factor']:.2f}, Return: {result['total_return']:.1f}%")

    all_results.extend(results_extended)
    create_comparison_chart(results_extended, 'Extended History (2020-2024)',
                           f'{OUTPUT_DIR}/charts/extended_history_comparison.png')

    # Create monthly chart for best performer
    best_extended = max(results_extended, key=lambda x: x['profit_factor'])
    create_monthly_performance_chart(best_extended,
                                    f'{OUTPUT_DIR}/charts/extended_monthly_{best_extended["config"]}.png')

    # =========================================================================
    # Test 4: Trailing Stop Optimization
    # =========================================================================
    print("\n" + "-"*80)
    print("TEST 4: Trailing Stop Parameter Optimization")
    print("-"*80)

    configs_trailing = [
        BacktestConfig(
            name='Trail_8%_5%',
            detector='v3',
            rs_min=70,
            trailing_activation_pct=8.0,
            trailing_distance_pct=5.0,
            start_date='2023-01-01'
        ),
        BacktestConfig(
            name='Trail_10%_5%',
            detector='v3',
            rs_min=70,
            trailing_activation_pct=10.0,
            trailing_distance_pct=5.0,
            start_date='2023-01-01'
        ),
        BacktestConfig(
            name='Trail_8%_3%',
            detector='v3',
            rs_min=70,
            trailing_activation_pct=8.0,
            trailing_distance_pct=3.0,
            start_date='2023-01-01'
        ),
        BacktestConfig(
            name='Trail_10%_3%',
            detector='v3',
            rs_min=70,
            trailing_activation_pct=10.0,
            trailing_distance_pct=3.0,
            start_date='2023-01-01'
        ),
        BacktestConfig(
            name='Fixed_7%_21%',
            detector='v3',
            rs_min=70,
            exit_method='fixed_target',
            stop_loss_pct=7.0,
            target_pct=21.0,
            start_date='2023-01-01'
        ),
    ]

    results_trailing = []
    for config in configs_trailing:
        result = backtester.run_backtest(symbols, config)
        results_trailing.append(result)
        print(f"\n{config.name}:")
        print(f"  Trades: {result['total_trades']}, Win Rate: {result['win_rate']:.1f}%")
        print(f"  Profit Factor: {result['profit_factor']:.2f}, Return: {result['total_return']:.1f}%")
        print(f"  Avg Win: {result['avg_win']:.2f}%, Avg Loss: {result['avg_loss']:.2f}%")

    all_results.extend(results_trailing)
    create_comparison_chart(results_trailing, 'Trailing Stop Optimization',
                           f'{OUTPUT_DIR}/charts/trailing_stop_comparison.png')

    # =========================================================================
    # Save All Results
    # =========================================================================
    print("\n" + "-"*80)
    print("Saving Results...")
    print("-"*80)

    # Save summary
    summary = []
    for r in all_results:
        summary.append({
            'config': r['config'],
            'detector': r.get('detector', 'unknown'),
            'total_trades': r['total_trades'],
            'win_rate': round(r['win_rate'], 1),
            'profit_factor': round(r['profit_factor'], 2),
            'total_return': round(r['total_return'], 1),
            'avg_win': round(r.get('avg_win', 0), 2),
            'avg_loss': round(r.get('avg_loss', 0), 2),
            'avg_days_held': round(r.get('avg_days_held', 0), 1),
            'bull_win_rate': round(r.get('bull_market_win_rate', 0), 1),
            'bear_win_rate': round(r.get('bear_market_win_rate', 0), 1),
        })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(f'{OUTPUT_DIR}/summary.csv', index=False)

    with open(f'{OUTPUT_DIR}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Save detailed results
    for r in all_results:
        if r['trades']:
            trades_df = pd.DataFrame(r['trades'])
            trades_df.to_csv(f'{OUTPUT_DIR}/{r["config"]}_trades.csv', index=False)

    print(f"\nResults saved to {OUTPUT_DIR}/")

    # Print final summary table
    print("\n" + "="*80)
    print("  FINAL SUMMARY")
    print("="*80)
    print(summary_df.to_string(index=False))

    return all_results


if __name__ == '__main__':
    results = run_comprehensive_analysis()
