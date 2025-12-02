"""
VCP Alert System - VCP Detector

Detects Volatility Contraction Patterns (VCP) based on Mark Minervini's methodology.

Features:
- Swing high/low detection
- Contraction identification with progressive tightening
- Volume dry-up analysis
- Entry signal detection (pivot breakout, handle break)
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .models import (
    Contraction,
    EntrySignal,
    EntryType,
    SwingPoint,
    VCPPattern,
)


@dataclass
class DetectorConfig:
    """Configuration for VCP detector."""
    swing_lookback: int = 5          # Bars to confirm swing point
    min_contractions: int = 2         # Minimum contractions for valid VCP
    max_contraction_range: float = 15.0  # Max % range for tightest contraction
    contraction_ratio_threshold: float = 0.8  # Each must be 80% or less of previous
    max_contraction_gap_days: int = 30  # Max days between contractions
    consolidation_tolerance: float = 0.03  # 3% tolerance for highs above base
    lookback_days: int = 120          # Days to look back for pattern

    # Staleness detection settings
    enable_staleness_check: bool = True  # Set to False to use legacy behavior
    max_days_since_contraction: int = 42  # 6 weeks - pattern becomes stale after this
    stale_penalty_per_week: float = 15.0  # Score reduction per week over threshold
    max_pivot_violations: int = 2     # Pattern marked stale after this many violations
    max_support_violations: int = 1   # Pattern invalid after support break
    pivot_violation_buffer: float = 0.005  # 0.5% buffer above pivot to count as violation


class VCPDetector:
    """
    Detects VCP patterns using swing high/low analysis.

    A VCP pattern consists of:
    1. Multiple contractions with progressively tighter ranges
    2. Volume dry-up in later contractions
    3. Consolidation base (highs don't step significantly higher)
    """

    def __init__(self, config: Optional[DetectorConfig] = None):
        """
        Initialize the VCP detector.

        Args:
            config: Configuration settings
        """
        self.config = config or DetectorConfig()

    def analyze(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN",
    ) -> Optional[VCPPattern]:
        """
        Analyze price data for VCP pattern.

        Args:
            df: DataFrame with OHLCV data (columns: Open, High, Low, Close, Volume)
            symbol: Stock symbol for the pattern

        Returns:
            VCPPattern if valid pattern found, None otherwise
        """
        if len(df) < self.config.lookback_days:
            return None

        # Use only the lookback period
        df_analysis = df.tail(self.config.lookback_days).copy()

        # Find swing points
        swing_highs = self._find_swing_highs(df_analysis)
        swing_lows = self._find_swing_lows(df_analysis)

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return None

        # Identify contractions
        contractions = self._identify_contractions(df_analysis, swing_highs, swing_lows)

        if len(contractions) < self.config.min_contractions:
            return None

        # Filter to get valid sequential contractions
        valid_contractions = self._filter_sequential_contractions(contractions)

        if len(valid_contractions) < self.config.min_contractions:
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

        pattern = VCPPattern(
            symbol=symbol,
            contractions=valid_contractions,
            is_valid=is_valid,
            validity_reasons=reasons,
            contraction_quality=contraction_quality,
            volume_quality=volume_quality,
            proximity_score=proximity_score,
            pivot_price=pivot_price,
            support_price=support_price,
            detection_date=datetime.now(),
        )

        # Check for staleness (time decay and price violations)
        # This can be disabled via config.enable_staleness_check = False
        pattern = self._check_staleness(df_analysis, pattern)

        return pattern

    def detect_contractions(
        self,
        df: pd.DataFrame,
    ) -> List[Contraction]:
        """
        Detect all contractions in price data.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            List of detected contractions
        """
        swing_highs = self._find_swing_highs(df)
        swing_lows = self._find_swing_lows(df)
        return self._identify_contractions(df, swing_highs, swing_lows)

    def get_entry_signals(
        self,
        df: pd.DataFrame,
        pattern: VCPPattern,
        risk_pct: float = 7.0,
        rr_ratio: float = 3.0,
    ) -> List[EntrySignal]:
        """
        Detect entry signals for a VCP pattern.

        Args:
            df: DataFrame with OHLCV data
            pattern: Detected VCP pattern
            risk_pct: Maximum risk percentage for stop loss
            rr_ratio: Reward to risk ratio for target

        Returns:
            List of entry signals
        """
        signals = []

        if not pattern.is_valid or df.empty:
            return signals

        # Get the last row for current price
        last_row = df.iloc[-1]
        current_price = last_row["Close"]
        current_date = df.index[-1]

        # Calculate stop based on pattern support
        stop_price = pattern.support_price * 0.99  # 1% below support
        risk = current_price - stop_price

        # Verify risk is within acceptable range
        risk_pct_actual = (risk / current_price) * 100
        if risk_pct_actual > risk_pct:
            # Adjust stop to maximum acceptable risk
            stop_price = current_price * (1 - risk_pct / 100)
            risk = current_price - stop_price

        target_price = current_price + (risk * rr_ratio)

        # Calculate volume ratio
        vol_ma = df["Volume"].rolling(50).mean().iloc[-1]
        vol_ratio = last_row["Volume"] / vol_ma if vol_ma > 0 else 1.0

        # Check for pivot breakout
        if self._check_pivot_breakout(df, pattern):
            signals.append(EntrySignal(
                entry_type=EntryType.PIVOT_BREAKOUT,
                date=current_date if isinstance(current_date, datetime) else datetime.now(),
                price=current_price,
                stop_price=stop_price,
                target_price=target_price,
                volume_ratio=vol_ratio,
            ))

        # Check for handle break (less common)
        if self._check_handle_break(df, pattern):
            signals.append(EntrySignal(
                entry_type=EntryType.HANDLE_BREAK,
                date=current_date if isinstance(current_date, datetime) else datetime.now(),
                price=current_price,
                stop_price=stop_price,
                target_price=target_price,
                volume_ratio=vol_ratio,
            ))

        return signals

    def calculate_score(self, pattern: VCPPattern) -> float:
        """
        Calculate overall pattern score.

        Args:
            pattern: VCP pattern to score

        Returns:
            Score from 0-100
        """
        return pattern.proximity_score

    def _find_swing_highs(self, df: pd.DataFrame) -> List[SwingPoint]:
        """Find swing highs in price data."""
        lookback = self.config.swing_lookback
        swing_highs = []
        highs = df["High"].values

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
                date = df.index[i]
                if isinstance(date, pd.Timestamp):
                    date = date.to_pydatetime()
                swing_highs.append(SwingPoint(
                    index=i,
                    date=date,
                    price=current_high,
                    point_type="high",
                ))

        return swing_highs

    def _find_swing_lows(self, df: pd.DataFrame) -> List[SwingPoint]:
        """Find swing lows in price data."""
        lookback = self.config.swing_lookback
        swing_lows = []
        lows = df["Low"].values

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
                date = df.index[i]
                if isinstance(date, pd.Timestamp):
                    date = date.to_pydatetime()
                swing_lows.append(SwingPoint(
                    index=i,
                    date=date,
                    price=current_low,
                    point_type="low",
                ))

        return swing_lows

    def _identify_contractions(
        self,
        df: pd.DataFrame,
        swing_highs: List[SwingPoint],
        swing_lows: List[SwingPoint],
    ) -> List[Contraction]:
        """
        Identify contractions using the "extend until recovery" approach.

        A contraction starts at a swing high and extends to the deepest low
        until price recovers above that high.
        """
        contractions = []

        # Calculate 50-day volume average for ratio
        df = df.copy()
        df["Volume_MA50"] = df["Volume"].rolling(50).mean()

        # Sort swing highs and lows by index
        sorted_highs = sorted(swing_highs, key=lambda x: x.index)
        sorted_lows = sorted(swing_lows, key=lambda x: x.index)

        if not sorted_highs or not sorted_lows:
            return contractions

        # Process chronologically
        last_contraction_end_index = -1

        for swing_high in sorted_highs:
            # Skip if this high is before our last contraction ended
            if swing_high.index <= last_contraction_end_index:
                continue

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
                recovered = False
                for check_high in sorted_highs:
                    if (check_high.index > swing_high.index and
                        check_high.index < swing_low.index and
                        check_high.price > swing_high.price):
                        recovered = True
                        break

                if recovered:
                    break

                # This low is a candidate - keep if it's deeper than current
                if current_contraction_low is None or swing_low.price < current_contraction_low.price:
                    current_contraction_low = swing_low

            # If we found a valid contraction, record it
            if current_contraction_low:
                range_pct = (
                    (swing_high.price - current_contraction_low.price)
                    / current_contraction_low.price * 100
                )
                duration = current_contraction_low.index - swing_high.index

                # Calculate average volume ratio during contraction
                contraction_slice = df.iloc[swing_high.index:current_contraction_low.index + 1]
                if len(contraction_slice) > 0:
                    avg_vol = contraction_slice["Volume"].mean()
                    vol_ma = contraction_slice["Volume_MA50"].mean()
                    vol_ratio = avg_vol / vol_ma if vol_ma > 0 else 1.0
                else:
                    vol_ratio = 1.0

                contractions.append(Contraction(
                    swing_high=swing_high,
                    swing_low=current_contraction_low,
                    range_pct=range_pct,
                    duration_days=duration,
                    avg_volume_ratio=vol_ratio,
                ))

                last_contraction_end_index = current_contraction_low.index

        return contractions

    def _filter_sequential_contractions(
        self,
        contractions: List[Contraction],
    ) -> List[Contraction]:
        """
        Filter contractions to keep only those forming a valid VCP sequence.

        Valid VCP sequence requirements:
        1. Each contraction should be TIGHTER than the previous
        2. Highs should form a consolidation base (not stepping higher)
        3. Time proximity - contractions shouldn't be too far apart
        4. Lows should not decline significantly
        """
        if len(contractions) < 2:
            return contractions

        best_sequence = []

        # Try starting from each contraction to find the longest valid sequence
        for start_idx in range(len(contractions)):
            sequence = [contractions[start_idx]]
            base_high_price = contractions[start_idx].swing_high.price

            for i in range(start_idx + 1, len(contractions)):
                current = contractions[i]
                prev = sequence[-1]

                # Rule 1: Must be tighter than previous
                if current.range_pct >= prev.range_pct:
                    continue

                # Rule 2: Time proximity
                days_gap = current.swing_high.index - prev.swing_low.index
                if days_gap > self.config.max_contraction_gap_days:
                    continue

                # Rule 3: Consolidation base - highs should not step significantly higher
                high_change = (current.swing_high.price - base_high_price) / base_high_price
                if high_change > self.config.consolidation_tolerance:
                    continue
                if high_change < -0.10:
                    continue

                # Rule 4: Lows should not decline significantly
                price_decline = (prev.swing_low.price - current.swing_low.price) / prev.swing_low.price
                if price_decline > 0.15:
                    continue

                sequence.append(current)

                # Limit to 4 contractions max
                if len(sequence) >= 4:
                    break

            # Keep the best (longest) valid sequence
            if len(sequence) > len(best_sequence):
                best_sequence = sequence

        return best_sequence

    def _validate_pattern(
        self,
        contractions: List[Contraction],
    ) -> Tuple[bool, List[str]]:
        """Validate if contractions form a proper VCP pattern."""
        reasons = []
        is_valid = True

        # Check 1: Minimum contractions
        if len(contractions) < self.config.min_contractions:
            reasons.append(f"Only {len(contractions)} contractions (need {self.config.min_contractions}+)")
            is_valid = False
        else:
            reasons.append(f"Has {len(contractions)} contractions")

        # Check 2: Contractions are progressively tighter
        ranges = [c.range_pct for c in contractions]
        tightening = all(
            ranges[i + 1] <= ranges[i] * 1.1  # Allow 10% tolerance
            for i in range(len(ranges) - 1)
        )
        if tightening:
            reasons.append("Contractions are progressively tighter")
        else:
            reasons.append("Contractions NOT progressively tighter")

        # Check 3: Final contraction is tight enough
        final_range = float(contractions[-1].range_pct)
        if final_range <= self.config.max_contraction_range:
            reasons.append(f"Final contraction tight ({final_range:.1f}%)")
        else:
            reasons.append(f"Final contraction too wide ({final_range:.1f}% > {self.config.max_contraction_range}%)")
            is_valid = False

        # Check 4: Volume dry-up in later contractions
        if len(contractions) >= 2:
            later_vol = float(np.mean([c.avg_volume_ratio for c in contractions[-2:]]))
            if later_vol < 1.0:
                reasons.append(f"Volume dry-up present ({later_vol:.2f}x avg)")
            else:
                reasons.append(f"No volume dry-up ({later_vol:.2f}x avg)")

        return is_valid, reasons

    def _calculate_contraction_quality(self, contractions: List[Contraction]) -> float:
        """Calculate quality score based on contraction sequence."""
        if len(contractions) < 2:
            return 50.0

        score = 100.0
        ranges = [c.range_pct for c in contractions]

        for i in range(len(ranges) - 1):
            ratio = ranges[i + 1] / ranges[i] if ranges[i] > 0 else 1.0

            if ratio <= 0.7:  # Excellent tightening
                score += 5
            elif ratio <= 0.85:  # Good tightening
                pass
            elif ratio <= 1.0:  # Slight tightening
                score -= 10
            else:  # Expansion
                score -= 25

        # Bonus for tight final contraction
        if ranges[-1] < 10:
            score += 10
        elif ranges[-1] < 15:
            score += 5

        return max(0, min(100, score))

    def _calculate_volume_quality(self, contractions: List[Contraction]) -> float:
        """Calculate quality score based on volume pattern."""
        if len(contractions) < 2:
            return 50.0

        score = 100.0
        vol_ratios = [float(c.avg_volume_ratio) for c in contractions]

        # Check for decreasing volume
        for i in range(len(vol_ratios) - 1):
            if vol_ratios[i + 1] < vol_ratios[i]:
                score += 5
            else:
                score -= 10

        # Bonus for low volume in final contraction
        if vol_ratios[-1] < 0.5:
            score += 15
        elif vol_ratios[-1] < 0.7:
            score += 10
        elif vol_ratios[-1] < 1.0:
            score += 5
        else:
            score -= 10

        return max(0, min(100, score))

    def _check_pivot_breakout(self, df: pd.DataFrame, pattern: VCPPattern) -> bool:
        """
        Check if current price action is a pivot breakout.

        Criteria:
        1. Close above pivot price
        2. Volume > 1.5x average
        3. Bullish candle (close > open)
        4. Close in upper 50% of day's range
        5. Previous close was below pivot (fresh breakout)
        """
        if len(df) < 2:
            return False

        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]

        # Calculate volume average
        vol_ma = df["Volume"].rolling(50).mean().iloc[-1]
        vol_ratio = last_row["Volume"] / vol_ma if vol_ma > 0 else 1.0

        # Check criteria
        close_above_pivot = last_row["Close"] > pattern.pivot_price
        volume_surge = vol_ratio > 1.5
        bullish_candle = last_row["Close"] > last_row["Open"]

        day_range = last_row["High"] - last_row["Low"]
        if day_range > 0:
            close_position = (last_row["Close"] - last_row["Low"]) / day_range
            close_in_upper_half = close_position > 0.5
        else:
            close_in_upper_half = True

        fresh_breakout = prev_row["Close"] <= pattern.pivot_price

        return all([
            close_above_pivot,
            volume_surge,
            bullish_candle,
            close_in_upper_half,
            fresh_breakout,
        ])

    def _check_handle_break(self, df: pd.DataFrame, pattern: VCPPattern) -> bool:
        """
        Check if current price action is a handle break.

        A handle break occurs when price breaks above the last contraction's
        swing high after forming a small handle (pullback).
        """
        if len(df) < 5 or not pattern.contractions:
            return False

        last_contraction = pattern.contractions[-1]
        handle_high = last_contraction.swing_high.price

        last_row = df.iloc[-1]

        # Check if close is above handle high
        if last_row["Close"] <= handle_high:
            return False

        # Check volume
        vol_ma = df["Volume"].rolling(50).mean().iloc[-1]
        vol_ratio = last_row["Volume"] / vol_ma if vol_ma > 0 else 1.0

        if vol_ratio < 1.2:
            return False

        # Check for small pullback (handle) in recent bars
        recent_lows = df["Low"].tail(10)
        handle_depth = (handle_high - recent_lows.min()) / handle_high

        # Handle should be shallow (< 10% depth)
        return handle_depth < 0.10

    # =========================================================================
    # Staleness Detection Methods
    # =========================================================================

    def _check_staleness(
        self,
        df: pd.DataFrame,
        pattern: VCPPattern,
    ) -> VCPPattern:
        """
        Check pattern for staleness based on time and price violations.

        Updates the pattern's staleness metrics in place and returns it.

        Args:
            df: Full DataFrame with OHLCV data
            pattern: VCPPattern to check

        Returns:
            Updated VCPPattern with staleness metrics
        """
        if not self.config.enable_staleness_check:
            # Legacy mode - skip staleness checks
            return pattern

        if not pattern.contractions:
            return pattern

        staleness_reasons = []

        # Get last contraction's end point
        last_contraction = pattern.contractions[-1]
        last_contraction_end_idx = last_contraction.swing_low.index

        # Calculate days since last contraction
        days_since = len(df) - 1 - last_contraction_end_idx
        pattern.days_since_last_contraction = days_since

        # Check time-based staleness
        if days_since > self.config.max_days_since_contraction:
            weeks_over = (days_since - self.config.max_days_since_contraction) / 5  # trading days per week
            staleness_reasons.append(
                f"Pattern is {days_since} days old (max {self.config.max_days_since_contraction})"
            )
            # Apply time decay penalty to freshness score
            time_penalty = weeks_over * self.config.stale_penalty_per_week
            pattern.freshness_score = max(0, 100 - time_penalty)
        else:
            pattern.freshness_score = 100.0

        # Count pivot violations (price went above pivot then fell back)
        pivot_violations = self._count_pivot_violations(
            df, pattern.pivot_price, last_contraction_end_idx
        )
        pattern.pivot_violations = pivot_violations

        if pivot_violations > 0:
            staleness_reasons.append(
                f"Pivot violated {pivot_violations} time(s) since last contraction"
            )
            # Apply violation penalty
            violation_penalty = pivot_violations * 20
            pattern.freshness_score = max(0, pattern.freshness_score - violation_penalty)

        # Count support violations (close below support)
        support_violations = self._count_support_violations(
            df, pattern.support_price, last_contraction_end_idx
        )
        pattern.support_violations = support_violations

        if support_violations > 0:
            staleness_reasons.append(
                f"Support broken {support_violations} time(s) since last contraction"
            )
            # Support break is more severe
            pattern.freshness_score = max(0, pattern.freshness_score - support_violations * 30)

        # Determine if pattern is stale
        is_stale = (
            days_since > self.config.max_days_since_contraction
            or pivot_violations >= self.config.max_pivot_violations
            or support_violations >= self.config.max_support_violations
        )

        pattern.is_stale = is_stale
        pattern.staleness_reasons = staleness_reasons

        # If support was violated, invalidate the pattern
        if support_violations >= self.config.max_support_violations:
            pattern.is_valid = False
            if "Support violation invalidates pattern" not in pattern.validity_reasons:
                pattern.validity_reasons.append("Support violation invalidates pattern")

        return pattern

    def _count_pivot_violations(
        self,
        df: pd.DataFrame,
        pivot_price: float,
        start_idx: int,
    ) -> int:
        """
        Count how many times price crossed above pivot then fell back.

        A pivot violation occurs when:
        1. High goes above pivot + buffer
        2. Subsequent close falls back below pivot

        Args:
            df: DataFrame with OHLCV data
            pivot_price: The pivot price level
            start_idx: Index to start checking from (after last contraction)

        Returns:
            Number of pivot violations
        """
        if start_idx >= len(df) - 1:
            return 0

        violations = 0
        above_pivot = False
        buffer = pivot_price * self.config.pivot_violation_buffer

        # Get arrays for faster access
        highs = df["High"].values
        closes = df["Close"].values

        # Check from day after last contraction to end of data
        for i in range(start_idx + 1, len(df)):
            high = float(highs[i])
            close = float(closes[i])

            if high > pivot_price + buffer:
                above_pivot = True

            if above_pivot and close < pivot_price:
                violations += 1
                above_pivot = False  # Reset for next potential violation

        return violations

    def _count_support_violations(
        self,
        df: pd.DataFrame,
        support_price: float,
        start_idx: int,
    ) -> int:
        """
        Count how many times price closed below support.

        Args:
            df: DataFrame with OHLCV data
            support_price: The support price level
            start_idx: Index to start checking from (after last contraction)

        Returns:
            Number of support violations (closes below support)
        """
        if start_idx >= len(df) - 1:
            return 0

        violations = 0

        # Get array for faster access
        closes = df["Close"].values

        # Check from day after last contraction to end of data
        for i in range(start_idx + 1, len(df)):
            close = float(closes[i])

            if close < support_price:
                violations += 1

        return violations
