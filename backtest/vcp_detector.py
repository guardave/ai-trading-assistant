#!/usr/bin/env python3
"""
VCP (Volatility Contraction Pattern) Detector

Proper implementation based on Mark Minervini's methodology:
1. Identify swing highs and swing lows
2. Track contractions as the distance between swing high and swing low decreases
3. Each contraction should be tighter than the previous (T1 > T2 > T3)
4. Volume should dry up during contractions

A contraction is measured from:
- Upper bound: Line connecting swing highs (descending)
- Lower bound: Line connecting swing lows (ascending)
- The convergence shows volatility contraction
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


@dataclass
class SwingPoint:
    """A swing high or swing low point"""
    index: int
    date: pd.Timestamp
    price: float
    point_type: str  # 'high' or 'low'


@dataclass
class Contraction:
    """A single VCP contraction defined by swing highs and lows"""
    swing_high: SwingPoint
    swing_low: SwingPoint
    range_pct: float  # (high - low) / low * 100
    duration_days: int
    avg_volume_ratio: float  # vs 50-day average (< 1 means volume dry-up)


@dataclass
class VCPPattern:
    """Complete VCP pattern analysis"""
    contractions: List[Contraction]
    is_valid: bool
    validity_reasons: List[str]
    # Scores
    contraction_quality: float  # 0-100: Are contractions progressively tighter?
    volume_quality: float  # 0-100: Is volume drying up?
    proximity_score: float  # 0-100: Overall pattern quality
    # Pattern bounds for visualization
    pivot_price: float  # Breakout level (highest high in pattern)
    support_price: float  # Lowest low in final contraction


class VCPDetector:
    """
    Detects VCP patterns using swing high/low analysis.

    Parameters:
    - swing_lookback: Number of bars to look back/forward to confirm swing point
    - min_contractions: Minimum number of contractions for valid VCP
    - max_contraction_range: Maximum range % for tightest contraction
    - contraction_ratio_threshold: Each contraction should be this % tighter than previous
    """

    def __init__(
        self,
        swing_lookback: int = 5,
        min_contractions: int = 2,
        max_contraction_range: float = 15.0,  # Final contraction should be < 15%
        contraction_ratio_threshold: float = 0.8  # Each contraction should be 80% or less of previous
    ):
        self.swing_lookback = swing_lookback
        self.min_contractions = min_contractions
        self.max_contraction_range = max_contraction_range
        self.contraction_ratio_threshold = contraction_ratio_threshold

    def find_swing_highs(self, df: pd.DataFrame, lookback: int = None) -> List[SwingPoint]:
        """
        Find swing highs in price data.

        A swing high is a bar where the high is higher than the highs of
        'lookback' bars on both sides.
        """
        if lookback is None:
            lookback = self.swing_lookback

        swing_highs = []
        highs = df['High'].values

        for i in range(lookback, len(df) - lookback):
            is_swing_high = True
            current_high = highs[i]

            # Check left side
            for j in range(1, lookback + 1):
                if highs[i - j] >= current_high:
                    is_swing_high = False
                    break

            # Check right side
            if is_swing_high:
                for j in range(1, lookback + 1):
                    if highs[i + j] >= current_high:
                        is_swing_high = False
                        break

            if is_swing_high:
                swing_highs.append(SwingPoint(
                    index=i,
                    date=df.index[i],
                    price=current_high,
                    point_type='high'
                ))

        return swing_highs

    def find_swing_lows(self, df: pd.DataFrame, lookback: int = None) -> List[SwingPoint]:
        """
        Find swing lows in price data.

        A swing low is a bar where the low is lower than the lows of
        'lookback' bars on both sides.
        """
        if lookback is None:
            lookback = self.swing_lookback

        swing_lows = []
        lows = df['Low'].values

        for i in range(lookback, len(df) - lookback):
            is_swing_low = True
            current_low = lows[i]

            # Check left side
            for j in range(1, lookback + 1):
                if lows[i - j] <= current_low:
                    is_swing_low = False
                    break

            # Check right side
            if is_swing_low:
                for j in range(1, lookback + 1):
                    if lows[i + j] <= current_low:
                        is_swing_low = False
                        break

            if is_swing_low:
                swing_lows.append(SwingPoint(
                    index=i,
                    date=df.index[i],
                    price=current_low,
                    point_type='low'
                ))

        return swing_lows

    def identify_contractions(
        self,
        df: pd.DataFrame,
        swing_highs: List[SwingPoint],
        swing_lows: List[SwingPoint]
    ) -> List[Contraction]:
        """
        Identify contractions using the "extend until recovery" approach.

        A contraction starts at a swing high and extends to the deepest low
        until price recovers above that high. Only then can a new contraction
        begin from a new (higher) high.

        This ensures:
        - Each contraction captures the full depth of pullback
        - No duplicate endpoints
        - Contractions are sequential and non-overlapping
        """
        contractions = []

        # Calculate 50-day volume average for ratio
        df['Volume_MA50'] = df['Volume'].rolling(50).mean()

        # Sort swing highs and lows by index
        sorted_highs = sorted(swing_highs, key=lambda x: x.index)
        sorted_lows = sorted(swing_lows, key=lambda x: x.index)

        if not sorted_highs or not sorted_lows:
            return contractions

        # Process chronologically - find contractions that don't overlap
        current_contraction_high = None
        current_contraction_low = None
        last_contraction_end_index = -1

        for swing_high in sorted_highs:
            # Skip if this high is before our last contraction ended
            if swing_high.index <= last_contraction_end_index:
                continue

            # Start a new potential contraction from this high
            current_contraction_high = swing_high
            current_contraction_low = None

            # Find the deepest low before price recovers above this high
            for swing_low in sorted_lows:
                # Must be after the high
                if swing_low.index <= swing_high.index:
                    continue

                # Must be lower than the high
                if swing_low.price >= swing_high.price:
                    continue

                # Check if price has recovered above the high before this low
                # by looking for a higher high between swing_high and swing_low
                recovered = False
                for check_high in sorted_highs:
                    if check_high.index > swing_high.index and check_high.index < swing_low.index:
                        if check_high.price > swing_high.price:
                            recovered = True
                            break

                if recovered:
                    # Price recovered before this low, so stop looking
                    break

                # This low is a candidate - keep if it's deeper than current
                if current_contraction_low is None or swing_low.price < current_contraction_low.price:
                    current_contraction_low = swing_low

            # If we found a valid contraction, record it
            if current_contraction_high and current_contraction_low:
                range_pct = (current_contraction_high.price - current_contraction_low.price) / current_contraction_low.price * 100
                duration = current_contraction_low.index - current_contraction_high.index

                # Calculate average volume ratio during contraction
                contraction_slice = df.iloc[current_contraction_high.index:current_contraction_low.index + 1]
                if len(contraction_slice) > 0:
                    avg_vol = contraction_slice['Volume'].mean()
                    vol_ma = contraction_slice['Volume_MA50'].mean()
                    vol_ratio = avg_vol / vol_ma if vol_ma > 0 else 1.0
                else:
                    vol_ratio = 1.0

                contractions.append(Contraction(
                    swing_high=current_contraction_high,
                    swing_low=current_contraction_low,
                    range_pct=range_pct,
                    duration_days=duration,
                    avg_volume_ratio=vol_ratio
                ))

                # Mark where this contraction ended so next one starts after
                last_contraction_end_index = current_contraction_low.index

        return contractions

    def filter_sequential_contractions(
        self,
        contractions: List[Contraction]
    ) -> List[Contraction]:
        """
        Filter contractions to keep only those that form a valid VCP sequence.

        Valid VCP sequence requirements:
        1. Each contraction should be TIGHTER (smaller %) than the previous
        2. Contractions should form a CONSOLIDATION BASE - swing highs should be
           at roughly the same level (not stepping higher like a staircase uptrend)
        3. Time proximity - contractions shouldn't be too far apart
        4. Lows should not decline significantly (not a downtrend)
        """
        if len(contractions) < 2:
            return contractions

        # Find the best VCP sequence by looking for progressively tighter contractions
        # forming a consolidation base (not a staircase pattern)
        best_sequence = []

        # Try starting from each contraction to find the longest valid sequence
        for start_idx in range(len(contractions)):
            sequence = [contractions[start_idx]]
            base_high_price = contractions[start_idx].swing_high.price  # Reference level

            for i in range(start_idx + 1, len(contractions)):
                current = contractions[i]
                prev = sequence[-1]

                # Rule 1: Must be tighter (smaller range %) than previous
                if current.range_pct >= prev.range_pct:
                    continue

                # Rule 2: Time proximity - contractions should start within
                # reasonable time after the previous one ends (max 30 trading days)
                days_gap = current.swing_high.index - prev.swing_low.index
                if days_gap > 30:
                    continue

                # Rule 3: CONSOLIDATION BASE - swing highs should NOT step significantly higher
                # In a true VCP, later highs retest or slightly undercut the base high
                # Stepping HIGHER indicates an uptrend with pullbacks, not consolidation
                high_change = (current.swing_high.price - base_high_price) / base_high_price
                # Allow highs up to 3% above base (slight overshoot ok), but reject higher
                # Also reject if too far below (more than 10% below = different pattern)
                if high_change > 0.03:  # Stepping more than 3% higher = staircase uptrend
                    continue
                if high_change < -0.10:  # More than 10% below = likely different base
                    continue

                # Rule 4: Lows should not decline significantly (not a downtrend)
                price_decline = (prev.swing_low.price - current.swing_low.price) / prev.swing_low.price
                if price_decline > 0.15:  # More than 15% lower = likely downtrend
                    continue

                sequence.append(current)

                # Limit to 4 contractions max
                if len(sequence) >= 4:
                    break

            # Keep the best (longest) valid sequence
            if len(sequence) > len(best_sequence):
                best_sequence = sequence

        return best_sequence

    def analyze_pattern(
        self,
        df: pd.DataFrame,
        lookback_days: int = 120
    ) -> Optional[VCPPattern]:
        """
        Analyze price data for VCP pattern.

        Args:
            df: DataFrame with OHLCV data
            lookback_days: Number of days to look back for pattern

        Returns:
            VCPPattern if valid pattern found, None otherwise
        """
        if len(df) < lookback_days:
            return None

        # Use only the lookback period
        df_analysis = df.tail(lookback_days).copy()

        # Find swing points
        swing_highs = self.find_swing_highs(df_analysis)
        swing_lows = self.find_swing_lows(df_analysis)

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return None

        # Identify contractions
        contractions = self.identify_contractions(df_analysis, swing_highs, swing_lows)

        if len(contractions) < self.min_contractions:
            return None

        # Filter to get valid sequential contractions
        valid_contractions = self.filter_sequential_contractions(contractions)

        if len(valid_contractions) < self.min_contractions:
            return None

        # Validate pattern
        is_valid, reasons = self._validate_pattern(valid_contractions)

        # Calculate quality scores
        contraction_quality = self._calculate_contraction_quality(valid_contractions)
        volume_quality = self._calculate_volume_quality(valid_contractions)
        proximity_score = (contraction_quality + volume_quality) / 2

        # Get pattern bounds
        pivot_price = max(c.swing_high.price for c in valid_contractions)
        support_price = valid_contractions[-1].swing_low.price

        return VCPPattern(
            contractions=valid_contractions,
            is_valid=is_valid,
            validity_reasons=reasons,
            contraction_quality=contraction_quality,
            volume_quality=volume_quality,
            proximity_score=proximity_score,
            pivot_price=pivot_price,
            support_price=support_price
        )

    def _validate_pattern(self, contractions: List[Contraction]) -> Tuple[bool, List[str]]:
        """Validate if contractions form a proper VCP pattern."""
        reasons = []
        is_valid = True

        # Check 1: Minimum contractions
        if len(contractions) < self.min_contractions:
            reasons.append(f"Only {len(contractions)} contractions (need {self.min_contractions}+)")
            is_valid = False
        else:
            reasons.append(f"Has {len(contractions)} contractions")

        # Check 2: Contractions are progressively tighter
        ranges = [c.range_pct for c in contractions]
        tightening = all(
            ranges[i+1] <= ranges[i] * (1 + 0.1)  # Allow 10% tolerance
            for i in range(len(ranges) - 1)
        )
        if tightening:
            reasons.append("Contractions are progressively tighter")
        else:
            reasons.append("Contractions NOT progressively tighter")
            # Don't invalidate, just note it

        # Check 3: Final contraction is tight enough
        final_range = contractions[-1].range_pct
        if final_range <= self.max_contraction_range:
            reasons.append(f"Final contraction tight ({final_range:.1f}%)")
        else:
            reasons.append(f"Final contraction too wide ({final_range:.1f}% > {self.max_contraction_range}%)")
            is_valid = False

        # Check 4: Volume dry-up in later contractions
        if len(contractions) >= 2:
            later_vol = np.mean([c.avg_volume_ratio for c in contractions[-2:]])
            if later_vol < 1.0:
                reasons.append(f"Volume dry-up present ({later_vol:.2f}x avg)")
            else:
                reasons.append(f"No volume dry-up ({later_vol:.2f}x avg)")

        return is_valid, reasons

    def _calculate_contraction_quality(self, contractions: List[Contraction]) -> float:
        """
        Calculate quality score based on contraction sequence.

        100 = Perfect sequential tightening
        0 = No tightening pattern
        """
        if len(contractions) < 2:
            return 50.0

        score = 100.0
        ranges = [c.range_pct for c in contractions]

        for i in range(len(ranges) - 1):
            ratio = ranges[i+1] / ranges[i] if ranges[i] > 0 else 1.0

            if ratio <= 0.7:  # Excellent tightening (30%+ reduction)
                score += 5
            elif ratio <= 0.85:  # Good tightening
                pass  # No change
            elif ratio <= 1.0:  # Slight tightening
                score -= 10
            else:  # Expansion (bad)
                score -= 25

        # Bonus for tight final contraction
        if ranges[-1] < 10:
            score += 10
        elif ranges[-1] < 15:
            score += 5

        return max(0, min(100, score))

    def _calculate_volume_quality(self, contractions: List[Contraction]) -> float:
        """
        Calculate quality score based on volume pattern.

        100 = Perfect volume dry-up
        0 = Volume expanding during consolidation
        """
        if len(contractions) < 2:
            return 50.0

        score = 100.0
        vol_ratios = [c.avg_volume_ratio for c in contractions]

        # Check for decreasing volume
        for i in range(len(vol_ratios) - 1):
            if vol_ratios[i+1] < vol_ratios[i]:
                score += 5  # Volume decreasing (good)
            else:
                score -= 10  # Volume increasing (bad)

        # Bonus for low volume in final contraction
        if vol_ratios[-1] < 0.5:
            score += 15  # Very dry
        elif vol_ratios[-1] < 0.7:
            score += 10
        elif vol_ratios[-1] < 1.0:
            score += 5
        else:
            score -= 10  # High volume in final contraction

        return max(0, min(100, score))


def test_detector():
    """Test the VCP detector with sample data."""
    import yfinance as yf

    # Test with a few stocks
    test_symbols = ['NVDA', 'AAPL', 'MSFT']

    detector = VCPDetector(
        swing_lookback=5,
        min_contractions=2,
        max_contraction_range=20.0
    )

    for symbol in test_symbols:
        print(f"\n{'='*60}")
        print(f"Analyzing {symbol}")
        print('='*60)

        # Get data
        df = yf.download(symbol, period='1y', progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Analyze
        pattern = detector.analyze_pattern(df, lookback_days=120)

        if pattern:
            print(f"VCP Pattern Found: {'VALID' if pattern.is_valid else 'INVALID'}")
            print(f"Contractions: {len(pattern.contractions)}")
            print(f"Proximity Score: {pattern.proximity_score:.1f}")
            print(f"Contraction Quality: {pattern.contraction_quality:.1f}")
            print(f"Volume Quality: {pattern.volume_quality:.1f}")
            print(f"Pivot Price: ${pattern.pivot_price:.2f}")
            print(f"Support Price: ${pattern.support_price:.2f}")
            print("\nContraction Details:")
            for i, c in enumerate(pattern.contractions):
                print(f"  C{i+1}: {c.swing_high.date.strftime('%Y-%m-%d')} to {c.swing_low.date.strftime('%Y-%m-%d')}")
                print(f"       High: ${c.swing_high.price:.2f}, Low: ${c.swing_low.price:.2f}")
                print(f"       Range: {c.range_pct:.1f}%, Volume Ratio: {c.avg_volume_ratio:.2f}x")
            print("\nValidity Reasons:")
            for reason in pattern.validity_reasons:
                print(f"  - {reason}")
        else:
            print("No VCP pattern found")


if __name__ == '__main__':
    test_detector()
