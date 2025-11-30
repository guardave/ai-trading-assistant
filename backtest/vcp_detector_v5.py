#!/usr/bin/env python3
"""
VCP (Volatility Contraction Pattern) Detector V5

Extends V4 with multiple entry types:
1. Pivot Breakout - Traditional entry above the pivot (highest high)
2. Cheat Entry - Entry in the middle third of base on tight range breakout
3. Low Cheat Entry - Entry in the lower third of base on tight range breakout
4. Handle Entry - Entry after a small pullback (5-10%) near the pivot

Based on Mark Minervini's methodology from:
- "Trade Like a Stock Market Wizard"
- "Think & Trade Like a Champion"

Sources:
- https://thesetupfactory.substack.com/p/mastering-the-minervini-low-cheat
- https://sakatas.substack.com/p/entry-types-all-possible-technical
- https://traderlion.com/technical-analysis/volatility-contraction-pattern/
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Import base V4 classes
from vcp_detector_v4 import (
    VCPDetectorV4, VCPConfig, VCPPattern,
    SwingPoint, Contraction, TrendTemplateResult, PriorUptrendResult,
    ValidationStatus
)


class EntryType(Enum):
    """Types of entry signals"""
    PIVOT_BREAKOUT = "pivot_breakout"       # Traditional breakout above pivot
    CHEAT = "cheat"                          # Middle third of base
    LOW_CHEAT = "low_cheat"                  # Lower third of base
    HANDLE = "handle"                        # Pullback near pivot
    NONE = "none"                            # No valid entry


@dataclass
class EntrySignal:
    """Represents a potential entry signal"""
    entry_type: EntryType
    entry_price: float
    entry_date: pd.Timestamp
    stop_loss: float
    pivot_price: float
    risk_pct: float  # (entry - stop) / entry * 100
    position_in_base: str  # 'lower_third', 'middle_third', 'upper_third'
    volume_ratio: float  # vs 50-day average
    is_valid: bool
    notes: List[str] = field(default_factory=list)


@dataclass
class HandleFormation:
    """Represents a handle formation near the pivot"""
    start_date: pd.Timestamp
    low_date: pd.Timestamp
    end_date: pd.Timestamp
    handle_high: float
    handle_low: float
    depth_pct: float  # pullback depth
    duration_days: int
    is_valid: bool


@dataclass
class VCPPatternV5(VCPPattern):
    """Extended VCP pattern with entry signals"""
    entry_signals: List[EntrySignal] = field(default_factory=list)
    handle: Optional[HandleFormation] = None
    base_high: float = 0.0
    base_low: float = 0.0
    base_range_pct: float = 0.0


class VCPDetectorV5(VCPDetectorV4):
    """
    V5 VCP Detector with multiple entry type detection.

    Entry Hierarchy (from earliest to latest):
    1. Low Cheat - Lower third of base, tight range breakout
    2. Cheat - Middle third of base, tight range breakout
    3. Handle - Pullback 5-10% near pivot, then breakout
    4. Pivot Breakout - Traditional breakout above pivot
    """

    def __init__(self, config: VCPConfig = None):
        super().__init__(config)

        # Entry-specific parameters
        self.handle_min_depth_pct = 3.0   # Minimum handle pullback
        self.handle_max_depth_pct = 12.0  # Maximum handle pullback
        self.handle_max_duration = 15     # Max days for handle formation
        self.tight_range_threshold = 5.0  # Max % range for cheat entries
        self.volume_confirmation = 1.2    # Min volume ratio for entry

    def calculate_base_zones(self, pattern: VCPPattern) -> Tuple[float, float, float, float]:
        """
        Calculate the three zones of the base for entry classification.

        Returns: (base_low, lower_third, middle_third, base_high)
        """
        if not pattern or not pattern.contractions:
            return 0, 0, 0, 0

        # Base high is the pivot
        base_high = pattern.pivot_price

        # Base low is the lowest low across all contractions
        base_low = min(c.swing_low.price for c in pattern.contractions)

        # Calculate thirds
        base_range = base_high - base_low
        lower_third = base_low + base_range * 0.33
        middle_third = base_low + base_range * 0.67

        return base_low, lower_third, middle_third, base_high

    def get_position_in_base(self, price: float, base_low: float,
                             lower_third: float, middle_third: float,
                             base_high: float) -> str:
        """Determine which third of the base a price is in."""
        if price <= lower_third:
            return 'lower_third'
        elif price <= middle_third:
            return 'middle_third'
        else:
            return 'upper_third'

    def detect_tight_range(self, df: pd.DataFrame, lookback: int = 5) -> Tuple[bool, float]:
        """
        Detect if recent price action is in a tight range.

        Returns: (is_tight, range_pct)
        """
        if len(df) < lookback:
            return False, 0

        recent = df.tail(lookback)
        high = recent['High'].max()
        low = recent['Low'].min()

        if low == 0:
            return False, 0

        range_pct = (high - low) / low * 100

        return range_pct <= self.tight_range_threshold, range_pct

    def detect_handle(self, df: pd.DataFrame, pivot_price: float) -> Optional[HandleFormation]:
        """
        Detect handle formation near the pivot.

        A handle is a small pullback (5-10%) that forms after price
        approaches or touches the pivot, before the actual breakout.
        """
        if len(df) < 10:
            return None

        # Look for price approaching pivot then pulling back
        recent = df.tail(30)

        # Find if price got close to pivot (within 3%)
        approach_mask = recent['High'] >= pivot_price * 0.97

        if not approach_mask.any():
            return None

        # Find the approach point
        approach_idx = approach_mask.idxmax()
        approach_pos = recent.index.get_loc(approach_idx)

        if approach_pos >= len(recent) - 3:
            return None  # Not enough data after approach

        # Look for pullback after approach
        post_approach = recent.iloc[approach_pos:]

        if len(post_approach) < 3:
            return None

        handle_high = post_approach['High'].iloc[0]
        handle_low = post_approach['Low'].min()
        low_idx = post_approach['Low'].idxmin()

        # Calculate handle depth
        depth_pct = (handle_high - handle_low) / handle_high * 100

        # Validate handle
        if depth_pct < self.handle_min_depth_pct or depth_pct > self.handle_max_depth_pct:
            return None

        duration = (low_idx - approach_idx).days if hasattr(low_idx - approach_idx, 'days') else \
                   recent.index.get_loc(low_idx) - approach_pos

        if duration > self.handle_max_duration:
            return None

        return HandleFormation(
            start_date=approach_idx,
            low_date=low_idx,
            end_date=recent.index[-1],
            handle_high=handle_high,
            handle_low=handle_low,
            depth_pct=depth_pct,
            duration_days=duration,
            is_valid=True
        )

    def detect_cheat_entry(self, df: pd.DataFrame, pattern: VCPPattern,
                           base_low: float, lower_third: float,
                           middle_third: float, base_high: float) -> Optional[EntrySignal]:
        """
        Detect cheat entry opportunity in the middle third of base.

        Criteria:
        - Price in middle third of base
        - Tight range (< 5% over last 5 days)
        - Bullish candle breaking out of tight range
        - Volume above average
        """
        if len(df) < 10:
            return None

        current = df.iloc[-1]
        current_price = current['Close']

        # Check position in base
        position = self.get_position_in_base(current_price, base_low,
                                             lower_third, middle_third, base_high)

        if position != 'middle_third':
            return None

        # Check for tight range
        is_tight, range_pct = self.detect_tight_range(df, lookback=5)

        if not is_tight:
            return None

        # Check for bullish breakout candle
        if current['Close'] <= current['Open']:
            return None  # Not bullish

        # Check volume
        vol_ma = df['Volume'].rolling(50).mean().iloc[-1]
        vol_ratio = current['Volume'] / vol_ma if vol_ma > 0 else 0

        if vol_ratio < self.volume_confirmation:
            return None

        # Calculate stop loss (below recent low)
        recent_low = df.tail(5)['Low'].min()
        stop_loss = recent_low * 0.98  # 2% below recent low

        risk_pct = (current_price - stop_loss) / current_price * 100

        return EntrySignal(
            entry_type=EntryType.CHEAT,
            entry_price=current_price,
            entry_date=df.index[-1],
            stop_loss=stop_loss,
            pivot_price=pattern.pivot_price,
            risk_pct=risk_pct,
            position_in_base=position,
            volume_ratio=vol_ratio,
            is_valid=True,
            notes=[f"Tight range: {range_pct:.1f}%", f"Volume: {vol_ratio:.1f}x avg"]
        )

    def detect_low_cheat_entry(self, df: pd.DataFrame, pattern: VCPPattern,
                                base_low: float, lower_third: float,
                                middle_third: float, base_high: float) -> Optional[EntrySignal]:
        """
        Detect low cheat entry opportunity in the lower third of base.

        Criteria:
        - Price in lower third of base
        - Tight range or shakeout recovery
        - Strong bullish candle with volume
        - Price above key MAs (50/200)
        """
        if len(df) < 10:
            return None

        current = df.iloc[-1]
        current_price = current['Close']

        # Check position in base
        position = self.get_position_in_base(current_price, base_low,
                                             lower_third, middle_third, base_high)

        if position != 'lower_third':
            return None

        # Check for tight range or shakeout pattern
        is_tight, range_pct = self.detect_tight_range(df, lookback=5)

        # Check for shakeout (price dipped below recent lows then recovered)
        recent_5 = df.tail(5)
        recent_20 = df.tail(20)

        shakeout = (recent_5['Low'].min() < recent_20['Low'].iloc[:-5].min() and
                    current['Close'] > recent_5['Open'].iloc[0])

        if not is_tight and not shakeout:
            return None

        # Check for bullish candle
        if current['Close'] <= current['Open']:
            return None

        # Check volume
        vol_ma = df['Volume'].rolling(50).mean().iloc[-1]
        vol_ratio = current['Volume'] / vol_ma if vol_ma > 0 else 0

        if vol_ratio < self.volume_confirmation:
            return None

        # Check above 200 MA
        ma200 = df['Close'].rolling(200).mean().iloc[-1]
        if current_price < ma200:
            return None

        # Calculate stop loss
        recent_low = df.tail(5)['Low'].min()
        stop_loss = recent_low * 0.98

        risk_pct = (current_price - stop_loss) / current_price * 100

        notes = []
        if is_tight:
            notes.append(f"Tight range: {range_pct:.1f}%")
        if shakeout:
            notes.append("Shakeout recovery")
        notes.append(f"Volume: {vol_ratio:.1f}x avg")

        return EntrySignal(
            entry_type=EntryType.LOW_CHEAT,
            entry_price=current_price,
            entry_date=df.index[-1],
            stop_loss=stop_loss,
            pivot_price=pattern.pivot_price,
            risk_pct=risk_pct,
            position_in_base=position,
            volume_ratio=vol_ratio,
            is_valid=True,
            notes=notes
        )

    def detect_handle_entry(self, df: pd.DataFrame, pattern: VCPPattern,
                            handle: HandleFormation) -> Optional[EntrySignal]:
        """
        Detect handle breakout entry.

        Criteria:
        - Valid handle formation exists
        - Price breaking above handle high
        - Volume confirmation
        """
        if handle is None or not handle.is_valid:
            return None

        current = df.iloc[-1]
        current_price = current['Close']

        # Check if breaking above handle high
        if current_price <= handle.handle_high:
            return None

        # Check for bullish candle
        if current['Close'] <= current['Open']:
            return None

        # Check volume
        vol_ma = df['Volume'].rolling(50).mean().iloc[-1]
        vol_ratio = current['Volume'] / vol_ma if vol_ma > 0 else 0

        if vol_ratio < self.volume_confirmation:
            return None

        # Stop below handle low
        stop_loss = handle.handle_low * 0.98
        risk_pct = (current_price - stop_loss) / current_price * 100

        return EntrySignal(
            entry_type=EntryType.HANDLE,
            entry_price=current_price,
            entry_date=df.index[-1],
            stop_loss=stop_loss,
            pivot_price=pattern.pivot_price,
            risk_pct=risk_pct,
            position_in_base='upper_third',
            volume_ratio=vol_ratio,
            is_valid=True,
            notes=[f"Handle depth: {handle.depth_pct:.1f}%",
                   f"Handle duration: {handle.duration_days}d",
                   f"Volume: {vol_ratio:.1f}x avg"]
        )

    def detect_pivot_breakout(self, df: pd.DataFrame, pattern: VCPPattern) -> Optional[EntrySignal]:
        """
        Detect traditional pivot breakout entry.

        Criteria:
        - Price closes above pivot
        - Volume > 1.5x average
        - Bullish candle
        - Close in upper half of day's range
        """
        if len(df) < 2:
            return None

        current = df.iloc[-1]
        prev = df.iloc[-2]
        current_price = current['Close']
        pivot = pattern.pivot_price

        # Must close above pivot
        if current_price <= pivot:
            return None

        # Must be fresh breakout (previous close below pivot)
        if prev['Close'] >= pivot:
            return None

        # Volume confirmation (stronger for pivot breakout)
        vol_ma = df['Volume'].rolling(50).mean().iloc[-1]
        vol_ratio = current['Volume'] / vol_ma if vol_ma > 0 else 0

        if vol_ratio < 1.5:  # Require 1.5x for pivot breakout
            return None

        # Bullish candle
        if current['Close'] <= current['Open']:
            return None

        # Close in upper half of range
        day_range = current['High'] - current['Low']
        if day_range > 0:
            close_position = (current['Close'] - current['Low']) / day_range
            if close_position < 0.5:
                return None

        # Stop below final contraction low or pivot
        if pattern.contractions:
            stop_loss = pattern.contractions[-1].swing_low.price * 0.98
        else:
            stop_loss = pivot * 0.93  # 7% below pivot

        risk_pct = (current_price - stop_loss) / current_price * 100

        return EntrySignal(
            entry_type=EntryType.PIVOT_BREAKOUT,
            entry_price=current_price,
            entry_date=df.index[-1],
            stop_loss=stop_loss,
            pivot_price=pivot,
            risk_pct=risk_pct,
            position_in_base='upper_third',
            volume_ratio=vol_ratio,
            is_valid=True,
            notes=[f"Volume: {vol_ratio:.1f}x avg", "Fresh breakout"]
        )

    def analyze_pattern_v5(
        self,
        df: pd.DataFrame,
        lookback_days: int = 120,
        spy_df: pd.DataFrame = None
    ) -> Optional[VCPPatternV5]:
        """
        Analyze price data for VCP pattern with multiple entry signals.
        """
        # First, get the base V4 pattern
        base_pattern = self.analyze_pattern(df, lookback_days, spy_df)

        if base_pattern is None:
            return None

        # Calculate base zones
        base_low, lower_third, middle_third, base_high = self.calculate_base_zones(base_pattern)

        # Detect handle formation
        handle = self.detect_handle(df, base_pattern.pivot_price)

        # Collect all entry signals
        entry_signals = []

        # Check for low cheat entry
        low_cheat = self.detect_low_cheat_entry(
            df, base_pattern, base_low, lower_third, middle_third, base_high
        )
        if low_cheat:
            entry_signals.append(low_cheat)

        # Check for cheat entry
        cheat = self.detect_cheat_entry(
            df, base_pattern, base_low, lower_third, middle_third, base_high
        )
        if cheat:
            entry_signals.append(cheat)

        # Check for handle entry
        handle_entry = self.detect_handle_entry(df, base_pattern, handle)
        if handle_entry:
            entry_signals.append(handle_entry)

        # Check for pivot breakout
        pivot_entry = self.detect_pivot_breakout(df, base_pattern)
        if pivot_entry:
            entry_signals.append(pivot_entry)

        # Create V5 pattern
        return VCPPatternV5(
            contractions=base_pattern.contractions,
            is_valid=base_pattern.is_valid,
            validity_reasons=base_pattern.validity_reasons,
            contraction_quality=base_pattern.contraction_quality,
            volume_quality=base_pattern.volume_quality,
            trend_template_score=base_pattern.trend_template_score,
            overall_score=base_pattern.overall_score,
            trend_template=base_pattern.trend_template,
            prior_uptrend=base_pattern.prior_uptrend,
            pivot_price=base_pattern.pivot_price,
            support_price=base_pattern.support_price,
            stop_loss_price=base_pattern.stop_loss_price,
            base_duration_days=base_pattern.base_duration_days,
            num_contractions=base_pattern.num_contractions,
            first_contraction_pct=base_pattern.first_contraction_pct,
            final_contraction_pct=base_pattern.final_contraction_pct,
            avg_tightening_ratio=base_pattern.avg_tightening_ratio,
            entry_signals=entry_signals,
            handle=handle,
            base_high=base_high,
            base_low=base_low,
            base_range_pct=(base_high - base_low) / base_low * 100 if base_low > 0 else 0
        )


def test_detector_v5():
    """Test the V5 VCP detector."""
    import yfinance as yf

    print("="*70)
    print("VCP Detector V5 Test - Multiple Entry Types")
    print("="*70)

    test_symbols = ['NVDA', 'AAPL', 'MSFT']

    detector = VCPDetectorV5()

    for symbol in test_symbols:
        print(f"\n{'='*70}")
        print(f"Analyzing {symbol}")
        print('='*70)

        df = yf.download(symbol, period='2y', progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if len(df) < 252:
            print(f"Insufficient data for {symbol}")
            continue

        pattern = detector.analyze_pattern_v5(df, lookback_days=120)

        if pattern:
            print(f"\nVCP Pattern Found: {'VALID' if pattern.is_valid else 'INVALID'}")
            print(f"Overall Score: {pattern.overall_score:.1f}")
            print(f"Base Range: {pattern.base_range_pct:.1f}%")
            print(f"Pivot: ${pattern.pivot_price:.2f}")

            if pattern.handle:
                print(f"\nHandle Formation:")
                print(f"  Depth: {pattern.handle.depth_pct:.1f}%")
                print(f"  Duration: {pattern.handle.duration_days} days")

            print(f"\nEntry Signals ({len(pattern.entry_signals)}):")
            for signal in pattern.entry_signals:
                print(f"\n  {signal.entry_type.value.upper()}:")
                print(f"    Price: ${signal.entry_price:.2f}")
                print(f"    Stop: ${signal.stop_loss:.2f}")
                print(f"    Risk: {signal.risk_pct:.1f}%")
                print(f"    Position: {signal.position_in_base}")
                print(f"    Volume: {signal.volume_ratio:.1f}x")
                for note in signal.notes:
                    print(f"    - {note}")
        else:
            print("No VCP pattern found")


if __name__ == '__main__':
    test_detector_v5()
