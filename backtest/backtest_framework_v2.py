"""
Backtesting Framework v2 for VCP and Pivot Strategies
Enhanced with:
- Trailing stop simulation
- VCP contraction proximity analysis
- R:R optimization focus
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Contraction:
    """Represents a single contraction in VCP pattern"""
    start_idx: int
    end_idx: int
    high: float
    low: float
    range_pct: float  # (high - low) / low * 100
    duration_days: int
    avg_volume_ratio: float  # vs 50-day average


@dataclass
class VCPAnalysis:
    """Detailed VCP pattern analysis"""
    contractions: List[Contraction]
    num_contractions: int
    contraction_sequence_valid: bool  # T1 > T2 > T3?
    avg_contraction_ratio: float  # How much each contraction tightens
    proximity_score: float  # How close/continuous are the contractions (0-100)
    total_pattern_days: int
    final_contraction_range: float


@dataclass
class Trade:
    """Represents a single trade with enhanced tracking"""
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
    pnl_dollars: Optional[float] = None
    days_held: Optional[int] = None
    max_gain_pct: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    rs_rating: Optional[float] = None
    volume_ratio: Optional[float] = None
    # VCP-specific
    num_contractions: Optional[int] = None
    proximity_score: Optional[float] = None
    contraction_sequence_valid: Optional[bool] = None
    # Risk metrics
    risk_pct: Optional[float] = None  # Entry to stop distance
    reward_pct: Optional[float] = None  # Entry to target distance
    theoretical_rr: Optional[float] = None
    realized_rr: Optional[float] = None


@dataclass
class BacktestResult:
    """Results from a backtest run"""
    strategy_name: str
    params: Dict[str, Any]
    exit_method: str  # 'fixed_target' or 'trailing_stop'
    total_trades: int
    winning_trades: int
    losing_trades: int
    open_trades: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    avg_rr_realized: float
    avg_rr_theoretical: float
    total_return_pct: float
    max_drawdown_pct: float
    avg_days_held: float
    # VCP proximity analysis
    proximity_correlation: Optional[float] = None  # Correlation between proximity and win
    high_proximity_win_rate: Optional[float] = None  # Win rate for proximity > 70
    low_proximity_win_rate: Optional[float] = None  # Win rate for proximity < 30
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

            df.columns = [c.title() for c in df.columns]
            df.to_parquet(cache_file)
            self.memory_cache[cache_key] = df

            return df

        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return None


class VCPAnalyzer:
    """Analyzes VCP patterns with contraction proximity scoring"""

    def __init__(self):
        pass

    def detect_contractions(self, df: pd.DataFrame, lookback_days: int = 120) -> List[Contraction]:
        """
        Detect individual contractions in price data.

        A contraction is defined as a period where:
        1. Price makes a local high
        2. Price pulls back
        3. Price consolidates with decreasing range
        """
        if len(df) < lookback_days:
            return []

        df_slice = df.tail(lookback_days).copy()
        df_slice['Volume_Avg'] = df['Volume'].rolling(window=50).mean()

        contractions = []

        # Calculate rolling highs and lows for different windows
        windows = [10, 15, 20, 25]  # Different window sizes to detect contractions

        for window in windows:
            df_slice[f'High_{window}'] = df_slice['High'].rolling(window=window).max()
            df_slice[f'Low_{window}'] = df_slice['Low'].rolling(window=window).min()
            df_slice[f'Range_{window}'] = (df_slice[f'High_{window}'] - df_slice[f'Low_{window}']) / df_slice[f'Low_{window}'] * 100

        # Find periods of contracting ranges
        df_slice['Range_20'] = df_slice['Range_20'].fillna(method='bfill')

        # Segment into potential contractions based on range changes
        range_series = df_slice['Range_20'].values
        vol_avg = df_slice['Volume_Avg'].values

        i = 0
        while i < len(range_series) - 20:
            # Look for a local high in range followed by contraction
            window_range = range_series[i:i+20]

            if len(window_range) < 20:
                break

            max_range = np.max(window_range[:10])  # First half
            min_range = np.min(window_range[10:])  # Second half

            # If range contracted by at least 30%, record this contraction
            if max_range > 0 and (max_range - min_range) / max_range > 0.3:
                start_idx = i
                end_idx = i + 20

                high = df_slice['High'].iloc[start_idx:end_idx].max()
                low = df_slice['Low'].iloc[start_idx:end_idx].min()
                range_pct = (high - low) / low * 100

                # Volume during contraction vs average
                contraction_vol = df_slice['Volume'].iloc[start_idx:end_idx].mean()
                avg_vol = vol_avg[end_idx] if not np.isnan(vol_avg[end_idx]) else contraction_vol
                vol_ratio = contraction_vol / avg_vol if avg_vol > 0 else 1.0

                contractions.append(Contraction(
                    start_idx=start_idx,
                    end_idx=end_idx,
                    high=high,
                    low=low,
                    range_pct=range_pct,
                    duration_days=20,
                    avg_volume_ratio=vol_ratio
                ))

                i = end_idx  # Move past this contraction
            else:
                i += 5  # Step forward

        return contractions

    def analyze_vcp(self, df: pd.DataFrame, check_date_idx: int) -> Optional[VCPAnalysis]:
        """
        Perform detailed VCP analysis including contraction proximity.

        Proximity Score (0-100):
        - 100: Contractions are continuous, each tighter than previous
        - 50: Some gaps between contractions, mostly sequential
        - 0: Scattered contractions with no clear pattern
        """
        if check_date_idx < 120:
            return None

        df_slice = df.iloc[max(0, check_date_idx - 120):check_date_idx + 1]

        contractions = self.detect_contractions(df_slice)

        if len(contractions) < 2:
            return None

        # Sort by end index (chronological)
        contractions.sort(key=lambda x: x.end_idx)

        # Take last 2-4 contractions
        recent_contractions = contractions[-4:] if len(contractions) >= 4 else contractions[-2:]

        # Check if sequence is valid (each contraction tighter than previous)
        ranges = [c.range_pct for c in recent_contractions]
        sequence_valid = all(ranges[i] >= ranges[i+1] for i in range(len(ranges)-1))

        # Calculate contraction ratio (how much each one tightens)
        if len(ranges) >= 2:
            ratios = [ranges[i+1] / ranges[i] if ranges[i] > 0 else 1.0 for i in range(len(ranges)-1)]
            avg_ratio = np.mean(ratios)
        else:
            avg_ratio = 1.0

        # Calculate proximity score
        proximity_score = self._calculate_proximity_score(recent_contractions)

        # Total pattern duration
        total_days = recent_contractions[-1].end_idx - recent_contractions[0].start_idx

        return VCPAnalysis(
            contractions=recent_contractions,
            num_contractions=len(recent_contractions),
            contraction_sequence_valid=sequence_valid,
            avg_contraction_ratio=avg_ratio,
            proximity_score=proximity_score,
            total_pattern_days=total_days,
            final_contraction_range=recent_contractions[-1].range_pct
        )

    def _calculate_proximity_score(self, contractions: List[Contraction]) -> float:
        """
        Calculate how close/continuous the contractions are.

        Factors:
        1. Gap between contractions (less gap = higher score)
        2. Range sequence (tightening = higher score)
        3. Volume pattern (decreasing = higher score)

        Returns: Score 0-100
        """
        if len(contractions) < 2:
            return 50.0

        score = 100.0

        # Factor 1: Gaps between contractions (max 40 points deduction)
        total_gap = 0
        for i in range(len(contractions) - 1):
            gap = contractions[i+1].start_idx - contractions[i].end_idx
            total_gap += max(0, gap)

        avg_gap = total_gap / (len(contractions) - 1)
        gap_penalty = min(40, avg_gap * 4)  # 10+ day avg gap = 40 point penalty
        score -= gap_penalty

        # Factor 2: Range sequence (max 30 points deduction)
        ranges = [c.range_pct for c in contractions]
        violations = 0
        for i in range(len(ranges) - 1):
            if ranges[i+1] > ranges[i]:  # Range expanded instead of contracted
                violations += 1

        sequence_penalty = violations * 15  # 15 points per violation
        score -= min(30, sequence_penalty)

        # Factor 3: Volume pattern (max 30 points deduction)
        volumes = [c.avg_volume_ratio for c in contractions]
        vol_increasing = sum(1 for i in range(len(volumes)-1) if volumes[i+1] > volumes[i])

        vol_penalty = vol_increasing * 10  # Prefer decreasing volume
        score -= min(30, vol_penalty)

        return max(0, min(100, score))


class StrategyScanner:
    """Scans for VCP and Pivot patterns with enhanced analysis"""

    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.vcp_analyzer = VCPAnalyzer()

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
            return 50.0

    def detect_vcp_enhanced(self, df: pd.DataFrame, spy_df: pd.DataFrame,
                            check_date_idx: int) -> Optional[Dict[str, Any]]:
        """
        Enhanced VCP detection with contraction proximity analysis.
        """
        df_slice = df.iloc[:check_date_idx + 1]

        if len(df_slice) < 120:
            return None

        params = self.params

        # Calculate indicators
        df_calc = df_slice.copy()
        df_calc['SMA_50'] = df_calc['Close'].rolling(window=50).mean()
        df_calc['SMA_200'] = df_calc['Close'].rolling(window=200).mean()
        df_calc['Volume_Avg'] = df_calc['Volume'].rolling(window=50).mean()

        current_price = df_calc['Close'].iloc[-1]
        current_volume = df_calc['Volume'].iloc[-1]
        avg_volume = df_calc['Volume_Avg'].iloc[-1]

        if pd.isna(avg_volume) or avg_volume == 0:
            return None

        # Check price above 200 MA
        if len(df_calc) >= 200:
            sma_200 = df_calc['SMA_200'].iloc[-1]
            if pd.notna(sma_200) and current_price < sma_200:
                return None

        # Check distance from 52-week high
        lookback = min(252, len(df_calc))
        high_52w = df_calc['High'].tail(lookback).max()
        distance_from_high = (high_52w - current_price) / high_52w

        if distance_from_high > params.get('distance_from_52w_high_max', 0.25):
            return None

        # Perform detailed VCP analysis
        vcp_analysis = self.vcp_analyzer.analyze_vcp(df, check_date_idx)

        if vcp_analysis is None:
            return None

        # Check minimum contractions
        min_contractions = params.get('min_contractions', 2)
        if vcp_analysis.num_contractions < min_contractions:
            return None

        # Check final contraction range
        if vcp_analysis.final_contraction_range > params.get('contraction_threshold', 0.20) * 100:
            return None

        # Check contraction sequence if required
        if params.get('require_sequence', False) and not vcp_analysis.contraction_sequence_valid:
            return None

        # Check proximity score threshold
        min_proximity = params.get('min_proximity_score', 0)
        if vcp_analysis.proximity_score < min_proximity:
            return None

        # Check RS rating
        spy_slice = spy_df.iloc[:check_date_idx + 1] if spy_df is not None else None
        rs_rating = self.calculate_rs_rating(df_calc, spy_slice)

        if rs_rating < params.get('rs_rating_min', 85):
            return None

        # Check for breakout
        weeks_to_check = params.get('min_consolidation_weeks', 4)
        days_to_check = weeks_to_check * 5
        consolidation_df = df_calc.tail(days_to_check)

        pivot_breakout = False
        if len(consolidation_df) >= 2:
            if current_price > consolidation_df['High'].iloc[-2]:
                pivot_vol_mult = params.get('pivot_breakout_volume_multiplier', 1.5)
                if current_volume > avg_volume * pivot_vol_mult:
                    pivot_breakout = True

        if not pivot_breakout:
            return None

        return {
            'pattern': 'VCP',
            'price': float(current_price),
            'volume_ratio': float(current_volume / avg_volume),
            'distance_52w_high': float(distance_from_high * 100),
            'rs_rating': float(rs_rating),
            'date': df_calc.index[-1],
            # VCP-specific analysis
            'num_contractions': vcp_analysis.num_contractions,
            'contraction_sequence_valid': vcp_analysis.contraction_sequence_valid,
            'proximity_score': vcp_analysis.proximity_score,
            'avg_contraction_ratio': vcp_analysis.avg_contraction_ratio,
            'final_contraction_range': vcp_analysis.final_contraction_range,
            'total_pattern_days': vcp_analysis.total_pattern_days,
        }

    def detect_pivot(self, df: pd.DataFrame, spy_df: pd.DataFrame,
                     check_date_idx: int) -> Optional[Dict[str, Any]]:
        """Detect Pivot Breakout pattern"""
        df_slice = df.iloc[:check_date_idx + 1]

        if len(df_slice) < 60:
            return None

        params = self.params

        df_calc = df_slice.copy()
        df_calc['SMA_50'] = df_calc['Close'].rolling(window=50).mean()
        df_calc['SMA_200'] = df_calc['Close'].rolling(window=200).mean()
        df_calc['Volume_Avg'] = df_calc['Volume'].rolling(window=50).mean()

        current_price = df_calc['Close'].iloc[-1]
        current_volume = df_calc['Volume'].iloc[-1]
        avg_volume = df_calc['Volume_Avg'].iloc[-1]

        if pd.isna(avg_volume) or avg_volume == 0:
            return None

        # Check MAs
        if params.get('price_above_50ma', True) and len(df_calc) >= 50:
            sma_50 = df_calc['SMA_50'].iloc[-1]
            if pd.notna(sma_50) and current_price < sma_50:
                return None

        if params.get('price_above_200ma', True) and len(df_calc) >= 200:
            sma_200 = df_calc['SMA_200'].iloc[-1]
            if pd.notna(sma_200) and current_price < sma_200:
                return None

        # Check distance from 52-week high
        lookback = min(252, len(df_calc))
        high_52w = df_calc['High'].tail(lookback).max()
        distance_from_high = (high_52w - current_price) / high_52w

        if distance_from_high > params.get('distance_from_52w_high_max', 0.30):
            return None

        # Look for base formation
        min_base_weeks = params.get('min_base_weeks', 4)
        max_base_weeks = params.get('max_base_weeks', 52)
        min_base_depth = params.get('min_base_depth', 0.08)
        max_base_depth = params.get('max_base_depth', 0.35)

        max_lookback = min(max_base_weeks * 5, len(df_calc))

        base_found = False
        base_high = None
        base_low = None
        base_weeks = 0

        for weeks_back in range(min_base_weeks * 5, max_lookback, 5):
            base_df = df_calc.tail(weeks_back)

            if len(base_df) < min_base_weeks * 5:
                continue

            base_high_candidate = base_df['High'].max()
            base_low_candidate = base_df['Low'].min()

            base_depth = (base_high_candidate - base_low_candidate) / base_high_candidate

            if min_base_depth <= base_depth <= max_base_depth:
                recent_high = df_calc.tail(10)['High'].max()

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
            spy_slice = spy_df.iloc[:check_date_idx + 1] if spy_df is not None else None
            rs_rating = self.calculate_rs_rating(df_calc, spy_slice)

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
                'date': df_calc.index[-1],
            }

        return None


class Backtester:
    """Main backtesting engine with dual exit methods"""

    def __init__(self, data_cache: DataCache):
        self.data_cache = data_cache
        self.spy_df = data_cache.get_historical_data('SPY', '2y')

    def simulate_trade_fixed_target(self, df: pd.DataFrame, signal: Dict[str, Any],
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
            contraction_sequence_valid=signal.get('contraction_sequence_valid'),
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
            trade.pnl_dollars = (trade.exit_price - entry_price) * 100
            if trade.risk_pct and trade.risk_pct > 0:
                trade.realized_rr = trade.pnl_pct / trade.risk_pct

        trade.max_gain_pct = ((max_price - entry_price) / entry_price) * 100
        trade.max_drawdown_pct = ((entry_price - min_price) / entry_price) * 100

        return trade

    def simulate_trade_trailing_stop(self, df: pd.DataFrame, signal: Dict[str, Any],
                                      stop_loss_pct: float = 7.0,
                                      trailing_activation_pct: float = 8.0,
                                      trailing_distance_pct: float = 5.0,
                                      max_hold_days: int = 90) -> Trade:
        """
        Simulate trade with trailing stop after activation threshold.

        - Initial stop: fixed at entry - stop_loss_pct
        - After +trailing_activation_pct gain: move stop to breakeven
        - Continue trailing by trailing_distance_pct below highest high
        """
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
            contraction_sequence_valid=signal.get('contraction_sequence_valid'),
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
            trade.pnl_dollars = (trade.exit_price - entry_price) * 100
            if trade.risk_pct and trade.risk_pct > 0:
                trade.realized_rr = trade.pnl_pct / trade.risk_pct

        trade.max_gain_pct = ((max_price - entry_price) / entry_price) * 100
        trade.max_drawdown_pct = ((entry_price - min_price) / entry_price) * 100
        trade.reward_pct = trade.max_gain_pct  # Actual achieved

        return trade

    def run_backtest(self, symbols: List[str], strategy_name: str,
                     params: Dict[str, Any], risk_params: Dict[str, Any],
                     exit_method: str = 'fixed_target',
                     start_date: str = '2023-01-01') -> BacktestResult:
        """
        Run backtest across multiple symbols.

        Args:
            exit_method: 'fixed_target' or 'trailing_stop'
        """
        scanner = StrategyScanner(params)
        all_trades: List[Trade] = []

        start_dt = pd.Timestamp(start_date)

        logger.info(f"Running {strategy_name} backtest ({exit_method}) on {len(symbols)} symbols...")

        for symbol in symbols:
            df = self.data_cache.get_historical_data(symbol, '2y')

            if df is None or len(df) < 252:
                continue

            last_signal_idx = -60  # Track to avoid overlapping trades

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

                # Detect pattern
                if 'VCP' in strategy_name:
                    signal = scanner.detect_vcp_enhanced(df, self.spy_df, idx)
                else:
                    signal = scanner.detect_pivot(df, self.spy_df, idx)

                if signal:
                    signal['symbol'] = symbol
                    last_signal_idx = idx

                    # Simulate trade based on exit method
                    if exit_method == 'fixed_target':
                        trade = self.simulate_trade_fixed_target(
                            df, signal,
                            stop_loss_pct=risk_params.get('default_stop_loss_pct', 7.0),
                            target_pct=risk_params.get('default_target_pct', 20.0),
                            max_hold_days=60
                        )
                    else:
                        trade = self.simulate_trade_trailing_stop(
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
        """Calculate backtest statistics including proximity analysis"""
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
                avg_rr_theoretical=0.0,
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
        avg_rr_theoretical = np.mean([t.theoretical_rr for t in trades if t.theoretical_rr]) if trades else 0

        total_return = sum(t.pnl_pct for t in closed_trades if t.pnl_pct) if closed_trades else 0
        avg_days = np.mean([t.days_held for t in closed_trades if t.days_held]) if closed_trades else 0
        max_dd = max([t.max_drawdown_pct for t in trades if t.max_drawdown_pct], default=0)

        # Proximity analysis for VCP
        proximity_correlation = None
        high_proximity_win_rate = None
        low_proximity_win_rate = None

        vcp_trades = [t for t in closed_trades if t.proximity_score is not None]
        if len(vcp_trades) >= 10:
            proximities = [t.proximity_score for t in vcp_trades]
            wins = [1 if t.pnl_pct and t.pnl_pct > 0 else 0 for t in vcp_trades]

            if len(set(proximities)) > 1:  # Need variance for correlation
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
            avg_rr_theoretical=avg_rr_theoretical,
            total_return_pct=total_return,
            max_drawdown_pct=max_dd,
            avg_days_held=avg_days,
            proximity_correlation=proximity_correlation,
            high_proximity_win_rate=high_proximity_win_rate,
            low_proximity_win_rate=low_proximity_win_rate,
            trades=trades
        )


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
    print(f"  Avg R:R Theoretical: {result.avg_rr_theoretical:.2f}")
    print(f"  Total Return:        {result.total_return_pct:.2f}%")
    print(f"  Max Drawdown:        {result.max_drawdown_pct:.2f}%")
    print(f"  Avg Days Held:       {result.avg_days_held:.1f}")

    # Proximity analysis
    if result.proximity_correlation is not None:
        print(f"\n  VCP Proximity Analysis:")
        print(f"    Proximity-Win Correlation: {result.proximity_correlation:.3f}")
        if result.high_proximity_win_rate is not None:
            print(f"    High Proximity (≥70) Win Rate: {result.high_proximity_win_rate:.1f}%")
        if result.low_proximity_win_rate is not None:
            print(f"    Low Proximity (≤30) Win Rate: {result.low_proximity_win_rate:.1f}%")

    print(f"{'='*70}")


if __name__ == '__main__':
    pass
