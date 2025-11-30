#!/usr/bin/env python3
"""
VCP (Volatility Contraction Pattern) Detector V4

Implements Mark Minervini's methodology with all critical elements:
1. Prior uptrend validation (30%+ advance before base)
2. Full Trend Template (8 criteria)
3. Swing high/low contraction detection
4. Progressive tightening enforcement (each ~50-70% of previous)
5. Base structure validation (not staircase)
6. Volume dry-up requirement
7. Minimum base duration

Based on research from:
- "Trade Like a Stock Market Wizard" by Mark Minervini
- "Think & Trade Like a Champion" by Mark Minervini
- Industry implementations (TradingView, QuantConnect, ThinkorSwim)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class ValidationStatus(Enum):
    """Status of each validation check"""
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"


@dataclass
class SwingPoint:
    """A swing high or swing low point"""
    index: int
    date: pd.Timestamp
    price: float
    point_type: str  # 'high' or 'low'
    volume: float = 0.0


@dataclass
class Contraction:
    """A single VCP contraction defined by swing highs and lows"""
    swing_high: SwingPoint
    swing_low: SwingPoint
    range_pct: float  # (high - low) / low * 100
    duration_days: int
    avg_volume_ratio: float  # vs 50-day average (< 1 means volume dry-up)
    tightening_ratio: float = 0.0  # ratio vs previous contraction (0 for first)


@dataclass
class TrendTemplateResult:
    """Result of trend template validation"""
    is_valid: bool
    checks: Dict[str, Tuple[ValidationStatus, str]]
    score: float  # 0-100 based on how many criteria passed


@dataclass
class PriorUptrendResult:
    """Result of prior uptrend validation"""
    is_valid: bool
    advance_pct: float
    lookback_days: int
    start_price: float
    peak_price: float
    message: str


@dataclass
class VCPPattern:
    """Complete VCP pattern analysis with V4 enhancements"""
    # Pattern data
    contractions: List[Contraction]
    is_valid: bool
    validity_reasons: List[str]

    # Quality scores
    contraction_quality: float  # 0-100: Are contractions progressively tighter?
    volume_quality: float  # 0-100: Is volume drying up?
    trend_template_score: float  # 0-100: How many trend criteria pass?
    overall_score: float  # 0-100: Combined quality score

    # Validation results
    trend_template: TrendTemplateResult
    prior_uptrend: PriorUptrendResult

    # Pattern bounds
    pivot_price: float  # Breakout level (highest high in pattern)
    support_price: float  # Lowest low in final contraction
    stop_loss_price: float  # Below final contraction low

    # Pattern characteristics
    base_duration_days: int
    num_contractions: int
    first_contraction_pct: float
    final_contraction_pct: float
    avg_tightening_ratio: float


@dataclass
class VCPConfig:
    """Configuration parameters for V4 VCP detection"""
    # Swing detection
    swing_lookback: int = 5  # 5-bar pivot detection

    # Contraction requirements
    min_contractions: int = 2
    max_contractions: int = 5
    max_first_contraction_pct: float = 35.0  # First pullback max 35%
    max_final_contraction_pct: float = 15.0  # Final should be <15%, ideally <10%
    ideal_final_contraction_pct: float = 10.0  # Bonus for <10%
    tightening_ratio_max: float = 0.80  # Each contraction must be ≤80% of previous
    tightening_ratio_ideal: float = 0.60  # Ideal is ~50-60% of previous

    # Base structure
    max_high_deviation_pct: float = 3.0  # Later highs can't be >3% above first
    max_days_between_contractions: int = 30
    min_base_duration_days: int = 15
    max_base_duration_days: int = 150  # ~6 months max

    # Prior uptrend
    min_prior_advance_pct: float = 30.0  # Must have 30%+ advance before base
    prior_advance_lookback_days: int = 90  # Look back 90 days for advance

    # Trend template
    min_rs_rating: int = 70
    require_ma_alignment: bool = True  # 50 > 150 > 200
    min_above_52wk_low_pct: float = 25.0  # Was 30%, relaxed slightly
    max_below_52wk_high_pct: float = 30.0  # Was 25%, relaxed slightly

    # Volume
    max_final_contraction_volume_ratio: float = 0.85  # <85% of 50-day avg
    ideal_final_contraction_volume_ratio: float = 0.70  # Ideal <70%
    min_breakout_volume_ratio: float = 1.50  # >150% of 50-day avg on breakout


class VCPDetectorV4:
    """
    V4 VCP Detector implementing full Minervini methodology.

    Key improvements over V3:
    1. Prior uptrend validation
    2. Full trend template
    3. Stricter tightening enforcement
    4. Base duration filtering
    5. Enhanced scoring
    """

    def __init__(self, config: VCPConfig = None):
        self.config = config or VCPConfig()

    def calculate_rs_rating(self, df: pd.DataFrame, spy_df: pd.DataFrame = None) -> float:
        """
        Calculate Relative Strength rating (0-100).

        If SPY data provided, calculates relative to market.
        Otherwise, uses price position within 52-week range as proxy.
        """
        if len(df) < 252:
            # Not enough data for 52-week calculation
            lookback = len(df)
        else:
            lookback = 252

        current_price = df['Close'].iloc[-1]

        if spy_df is not None and len(spy_df) >= lookback:
            # Calculate relative performance vs SPY
            stock_return = (df['Close'].iloc[-1] / df['Close'].iloc[-lookback] - 1) * 100
            spy_return = (spy_df['Close'].iloc[-1] / spy_df['Close'].iloc[-lookback] - 1) * 100

            # RS = stock outperformance, scaled to 0-100
            relative_return = stock_return - spy_return
            # Assume -50% to +50% relative return maps to 0-100
            rs_rating = min(100, max(0, 50 + relative_return))
        else:
            # Proxy: position within 52-week high/low range
            high_52wk = df['High'].tail(lookback).max()
            low_52wk = df['Low'].tail(lookback).min()

            if high_52wk == low_52wk:
                rs_rating = 50
            else:
                rs_rating = (current_price - low_52wk) / (high_52wk - low_52wk) * 100

        return rs_rating

    def validate_trend_template(self, df: pd.DataFrame, rs_rating: float = None) -> TrendTemplateResult:
        """
        Validate Minervini's 8-point Trend Template.

        Criteria:
        1. Current price > 150-day MA
        2. Current price > 200-day MA
        3. 150-day MA > 200-day MA
        4. 200-day MA trending UP for ≥22 trading days (1 month)
        5. 50-day MA > 150-day MA
        6. 50-day MA > 200-day MA
        7. Current price ≥ 25% above 52-week low
        8. Current price within 30% of 52-week high
        9. RS Rating ≥ 70 (bonus criterion)
        """
        checks = {}

        if len(df) < 252:
            return TrendTemplateResult(
                is_valid=False,
                checks={"data": (ValidationStatus.FAIL, "Insufficient data (<252 days)")},
                score=0
            )

        current_price = df['Close'].iloc[-1]

        # Calculate MAs
        ma50 = df['Close'].rolling(50).mean().iloc[-1]
        ma150 = df['Close'].rolling(150).mean().iloc[-1]
        ma200 = df['Close'].rolling(200).mean().iloc[-1]
        ma200_month_ago = df['Close'].rolling(200).mean().iloc[-22] if len(df) >= 222 else ma200

        # 52-week high/low
        high_52wk = df['High'].tail(252).max()
        low_52wk = df['Low'].tail(252).min()

        # Calculate RS if not provided
        if rs_rating is None:
            rs_rating = self.calculate_rs_rating(df)

        # Check 1: Price > 150 MA
        if current_price > ma150:
            checks["price_above_150ma"] = (ValidationStatus.PASS, f"${current_price:.2f} > ${ma150:.2f}")
        else:
            checks["price_above_150ma"] = (ValidationStatus.FAIL, f"${current_price:.2f} < ${ma150:.2f}")

        # Check 2: Price > 200 MA
        if current_price > ma200:
            checks["price_above_200ma"] = (ValidationStatus.PASS, f"${current_price:.2f} > ${ma200:.2f}")
        else:
            checks["price_above_200ma"] = (ValidationStatus.FAIL, f"${current_price:.2f} < ${ma200:.2f}")

        # Check 3: 150 MA > 200 MA
        if ma150 > ma200:
            checks["150ma_above_200ma"] = (ValidationStatus.PASS, f"${ma150:.2f} > ${ma200:.2f}")
        else:
            checks["150ma_above_200ma"] = (ValidationStatus.FAIL, f"${ma150:.2f} < ${ma200:.2f}")

        # Check 4: 200 MA trending up
        if ma200 > ma200_month_ago:
            checks["200ma_trending_up"] = (ValidationStatus.PASS, f"${ma200:.2f} > ${ma200_month_ago:.2f} (month ago)")
        else:
            checks["200ma_trending_up"] = (ValidationStatus.FAIL, f"${ma200:.2f} < ${ma200_month_ago:.2f} (month ago)")

        # Check 5: 50 MA > 150 MA
        if ma50 > ma150:
            checks["50ma_above_150ma"] = (ValidationStatus.PASS, f"${ma50:.2f} > ${ma150:.2f}")
        else:
            checks["50ma_above_150ma"] = (ValidationStatus.FAIL, f"${ma50:.2f} < ${ma150:.2f}")

        # Check 6: 50 MA > 200 MA
        if ma50 > ma200:
            checks["50ma_above_200ma"] = (ValidationStatus.PASS, f"${ma50:.2f} > ${ma200:.2f}")
        else:
            checks["50ma_above_200ma"] = (ValidationStatus.FAIL, f"${ma50:.2f} < ${ma200:.2f}")

        # Check 7: Price ≥ 25% above 52-week low
        pct_above_low = (current_price - low_52wk) / low_52wk * 100
        if pct_above_low >= self.config.min_above_52wk_low_pct:
            checks["above_52wk_low"] = (ValidationStatus.PASS, f"{pct_above_low:.1f}% above 52wk low")
        else:
            checks["above_52wk_low"] = (ValidationStatus.FAIL, f"Only {pct_above_low:.1f}% above 52wk low (need {self.config.min_above_52wk_low_pct}%)")

        # Check 8: Price within 30% of 52-week high
        pct_below_high = (high_52wk - current_price) / high_52wk * 100
        if pct_below_high <= self.config.max_below_52wk_high_pct:
            checks["near_52wk_high"] = (ValidationStatus.PASS, f"{pct_below_high:.1f}% below 52wk high")
        else:
            checks["near_52wk_high"] = (ValidationStatus.FAIL, f"{pct_below_high:.1f}% below 52wk high (max {self.config.max_below_52wk_high_pct}%)")

        # Check 9: RS Rating
        if rs_rating >= self.config.min_rs_rating:
            checks["rs_rating"] = (ValidationStatus.PASS, f"RS {rs_rating:.0f} >= {self.config.min_rs_rating}")
        else:
            checks["rs_rating"] = (ValidationStatus.WARN, f"RS {rs_rating:.0f} < {self.config.min_rs_rating}")

        # Calculate score and validity
        pass_count = sum(1 for status, _ in checks.values() if status == ValidationStatus.PASS)
        total_checks = len(checks)
        score = pass_count / total_checks * 100

        # Must pass at least 6 of 8 core criteria (excluding RS which is WARN)
        core_passes = sum(1 for k, (status, _) in checks.items()
                        if status == ValidationStatus.PASS and k != "rs_rating")
        is_valid = core_passes >= 6

        return TrendTemplateResult(
            is_valid=is_valid,
            checks=checks,
            score=score
        )

    def validate_prior_uptrend(self, df: pd.DataFrame) -> PriorUptrendResult:
        """
        Validate that stock had significant advance before base formation.

        Looks for 30%+ advance in the prior 60-90 days before the most recent
        consolidation period.
        """
        lookback = min(self.config.prior_advance_lookback_days, len(df) - 60)

        if lookback < 30:
            return PriorUptrendResult(
                is_valid=False,
                advance_pct=0,
                lookback_days=lookback,
                start_price=0,
                peak_price=0,
                message="Insufficient data for prior uptrend check"
            )

        # Look at the period before the recent base
        # Assume base is ~60 days, so look at days before that
        analysis_end = -60 if len(df) > 120 else -30
        analysis_start = analysis_end - lookback

        if abs(analysis_start) > len(df):
            analysis_start = -len(df)

        df_prior = df.iloc[analysis_start:analysis_end]

        if len(df_prior) < 20:
            return PriorUptrendResult(
                is_valid=False,
                advance_pct=0,
                lookback_days=len(df_prior),
                start_price=0,
                peak_price=0,
                message="Insufficient data in prior period"
            )

        # Find the lowest low and highest high in the prior period
        start_price = df_prior['Low'].min()
        peak_price = df_prior['High'].max()

        # Calculate advance percentage
        if start_price > 0:
            advance_pct = (peak_price - start_price) / start_price * 100
        else:
            advance_pct = 0

        is_valid = advance_pct >= self.config.min_prior_advance_pct

        if is_valid:
            message = f"Prior advance of {advance_pct:.1f}% (${start_price:.2f} to ${peak_price:.2f})"
        else:
            message = f"Insufficient prior advance: {advance_pct:.1f}% (need {self.config.min_prior_advance_pct}%)"

        return PriorUptrendResult(
            is_valid=is_valid,
            advance_pct=advance_pct,
            lookback_days=len(df_prior),
            start_price=start_price,
            peak_price=peak_price,
            message=message
        )

    def find_swing_highs(self, df: pd.DataFrame) -> List[SwingPoint]:
        """Find swing highs using N-bar lookback."""
        swing_highs = []
        highs = df['High'].values
        volumes = df['Volume'].values if 'Volume' in df.columns else np.zeros(len(df))
        lookback = self.config.swing_lookback

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
                    point_type='high',
                    volume=volumes[i]
                ))

        return swing_highs

    def find_swing_lows(self, df: pd.DataFrame) -> List[SwingPoint]:
        """Find swing lows using N-bar lookback."""
        swing_lows = []
        lows = df['Low'].values
        volumes = df['Volume'].values if 'Volume' in df.columns else np.zeros(len(df))
        lookback = self.config.swing_lookback

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
                    point_type='low',
                    volume=volumes[i]
                ))

        return swing_lows

    def identify_contractions(
        self,
        df: pd.DataFrame,
        swing_highs: List[SwingPoint],
        swing_lows: List[SwingPoint]
    ) -> List[Contraction]:
        """
        Identify contractions using swing high to deepest low approach.

        Each contraction:
        - Starts at a swing high
        - Extends to the deepest low before price recovers above that high
        - Must have a valid high > low relationship
        """
        contractions = []

        # Calculate 50-day volume average
        df = df.copy()
        df['Volume_MA50'] = df['Volume'].rolling(50).mean()

        sorted_highs = sorted(swing_highs, key=lambda x: x.index)
        sorted_lows = sorted(swing_lows, key=lambda x: x.index)

        if not sorted_highs or not sorted_lows:
            return contractions

        last_contraction_end_index = -1
        prev_range_pct = None

        for swing_high in sorted_highs:
            if swing_high.index <= last_contraction_end_index:
                continue

            current_contraction_low = None

            # Find deepest low before recovery
            for swing_low in sorted_lows:
                if swing_low.index <= swing_high.index:
                    continue

                if swing_low.price >= swing_high.price:
                    continue

                # Check for recovery before this low
                recovered = False
                for check_high in sorted_highs:
                    if check_high.index > swing_high.index and check_high.index < swing_low.index:
                        if check_high.price > swing_high.price:
                            recovered = True
                            break

                if recovered:
                    break

                if current_contraction_low is None or swing_low.price < current_contraction_low.price:
                    current_contraction_low = swing_low

            if current_contraction_low:
                range_pct = (swing_high.price - current_contraction_low.price) / current_contraction_low.price * 100
                duration = current_contraction_low.index - swing_high.index

                # Calculate volume ratio
                contraction_slice = df.iloc[swing_high.index:current_contraction_low.index + 1]
                if len(contraction_slice) > 0:
                    avg_vol = contraction_slice['Volume'].mean()
                    vol_ma = contraction_slice['Volume_MA50'].mean()
                    vol_ratio = avg_vol / vol_ma if vol_ma > 0 else 1.0
                else:
                    vol_ratio = 1.0

                # Calculate tightening ratio vs previous
                if prev_range_pct is not None and prev_range_pct > 0:
                    tightening_ratio = range_pct / prev_range_pct
                else:
                    tightening_ratio = 0.0

                contractions.append(Contraction(
                    swing_high=swing_high,
                    swing_low=current_contraction_low,
                    range_pct=range_pct,
                    duration_days=duration,
                    avg_volume_ratio=vol_ratio,
                    tightening_ratio=tightening_ratio
                ))

                last_contraction_end_index = current_contraction_low.index
                prev_range_pct = range_pct

        return contractions

    def filter_valid_vcp_sequence(
        self,
        contractions: List[Contraction]
    ) -> Tuple[List[Contraction], List[str]]:
        """
        Filter contractions to find valid VCP sequence with strict criteria.

        Returns filtered contractions and list of rejection reasons.
        """
        reasons = []

        if len(contractions) < self.config.min_contractions:
            reasons.append(f"Only {len(contractions)} contractions (need {self.config.min_contractions}+)")
            return [], reasons

        best_sequence = []

        for start_idx in range(len(contractions)):
            sequence = [contractions[start_idx]]
            base_high_price = contractions[start_idx].swing_high.price
            seq_reasons = []

            # Check first contraction depth
            if contractions[start_idx].range_pct > self.config.max_first_contraction_pct:
                seq_reasons.append(f"First contraction too deep: {contractions[start_idx].range_pct:.1f}%")
                continue

            for i in range(start_idx + 1, len(contractions)):
                current = contractions[i]
                prev = sequence[-1]

                # Rule 1: Must be tighter (with tolerance for near-equal)
                tightening = current.range_pct / prev.range_pct if prev.range_pct > 0 else 1.0
                if tightening > self.config.tightening_ratio_max:
                    seq_reasons.append(f"C{len(sequence)+1} not tighter: {current.range_pct:.1f}% vs {prev.range_pct:.1f}% (ratio {tightening:.2f})")
                    continue

                # Rule 2: Time proximity
                days_gap = current.swing_high.index - prev.swing_low.index
                if days_gap > self.config.max_days_between_contractions:
                    seq_reasons.append(f"Gap too large: {days_gap} days between contractions")
                    continue

                # Rule 3: Base structure (no staircase)
                high_change = (current.swing_high.price - base_high_price) / base_high_price * 100
                if high_change > self.config.max_high_deviation_pct:
                    seq_reasons.append(f"Staircase pattern: high {high_change:.1f}% above base")
                    continue

                # Rule 4: Lows not declining significantly
                price_decline = (prev.swing_low.price - current.swing_low.price) / prev.swing_low.price * 100
                if price_decline > 15:
                    seq_reasons.append(f"Downtrend: low declined {price_decline:.1f}%")
                    continue

                sequence.append(current)

                if len(sequence) >= self.config.max_contractions:
                    break

            if len(sequence) > len(best_sequence):
                best_sequence = sequence
                reasons = seq_reasons

        # Validate final contraction tightness
        if best_sequence:
            final_range = best_sequence[-1].range_pct
            if final_range > self.config.max_final_contraction_pct:
                reasons.append(f"Final contraction too wide: {final_range:.1f}% (max {self.config.max_final_contraction_pct}%)")
                # Don't invalidate, but note it

        return best_sequence, reasons

    def calculate_scores(
        self,
        contractions: List[Contraction],
        trend_template: TrendTemplateResult,
        prior_uptrend: PriorUptrendResult
    ) -> Tuple[float, float, float]:
        """Calculate quality scores for the pattern."""

        # Contraction quality (0-100)
        contraction_score = 100.0
        if len(contractions) >= 2:
            for i in range(1, len(contractions)):
                ratio = contractions[i].tightening_ratio
                if ratio <= self.config.tightening_ratio_ideal:
                    contraction_score += 5  # Excellent tightening
                elif ratio <= self.config.tightening_ratio_max:
                    pass  # Good
                else:
                    contraction_score -= 15  # Not tight enough

            # Bonus for tight final contraction
            final_range = contractions[-1].range_pct
            if final_range < self.config.ideal_final_contraction_pct:
                contraction_score += 10
            elif final_range < self.config.max_final_contraction_pct:
                contraction_score += 5

        contraction_score = max(0, min(100, contraction_score))

        # Volume quality (0-100)
        volume_score = 100.0
        if len(contractions) >= 2:
            vol_ratios = [c.avg_volume_ratio for c in contractions]

            # Check for decreasing volume
            for i in range(1, len(vol_ratios)):
                if vol_ratios[i] < vol_ratios[i-1]:
                    volume_score += 5
                else:
                    volume_score -= 10

            # Final contraction volume
            final_vol = vol_ratios[-1]
            if final_vol < self.config.ideal_final_contraction_volume_ratio:
                volume_score += 15
            elif final_vol < self.config.max_final_contraction_volume_ratio:
                volume_score += 5
            else:
                volume_score -= 10

        volume_score = max(0, min(100, volume_score))

        # Overall score combines all factors
        overall_score = (
            contraction_score * 0.35 +
            volume_score * 0.20 +
            trend_template.score * 0.30 +
            (100 if prior_uptrend.is_valid else 50) * 0.15
        )

        return contraction_score, volume_score, overall_score

    def analyze_pattern(
        self,
        df: pd.DataFrame,
        lookback_days: int = 120,
        spy_df: pd.DataFrame = None
    ) -> Optional[VCPPattern]:
        """
        Analyze price data for VCP pattern using V4 methodology.

        Args:
            df: DataFrame with OHLCV data
            lookback_days: Number of days to look back for pattern
            spy_df: Optional SPY data for RS calculation

        Returns:
            VCPPattern if valid pattern found, None otherwise
        """
        if len(df) < max(lookback_days, 252):
            return None

        # Step 1: Validate trend template
        rs_rating = self.calculate_rs_rating(df, spy_df)
        trend_template = self.validate_trend_template(df, rs_rating)

        # Step 2: Validate prior uptrend
        prior_uptrend = self.validate_prior_uptrend(df)

        # Step 3: Find swing points in lookback period
        df_analysis = df.tail(lookback_days).copy()
        swing_highs = self.find_swing_highs(df_analysis)
        swing_lows = self.find_swing_lows(df_analysis)

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return None

        # Step 4: Identify contractions
        contractions = self.identify_contractions(df_analysis, swing_highs, swing_lows)

        if len(contractions) < self.config.min_contractions:
            return None

        # Step 5: Filter to valid VCP sequence
        valid_contractions, filter_reasons = self.filter_valid_vcp_sequence(contractions)

        if len(valid_contractions) < self.config.min_contractions:
            return None

        # Step 6: Validate base duration
        base_duration = valid_contractions[-1].swing_low.index - valid_contractions[0].swing_high.index
        if base_duration < self.config.min_base_duration_days:
            return None

        # Step 7: Calculate scores
        contraction_quality, volume_quality, overall_score = self.calculate_scores(
            valid_contractions, trend_template, prior_uptrend
        )

        # Step 8: Determine validity
        validity_reasons = []
        is_valid = True

        # Check trend template
        if trend_template.is_valid:
            validity_reasons.append(f"Trend template: {trend_template.score:.0f}% criteria passed")
        else:
            validity_reasons.append(f"Trend template weak: {trend_template.score:.0f}% criteria passed")
            is_valid = False

        # Check prior uptrend
        if prior_uptrend.is_valid:
            validity_reasons.append(f"Prior uptrend: {prior_uptrend.advance_pct:.1f}% advance")
        else:
            validity_reasons.append(f"No prior uptrend: {prior_uptrend.advance_pct:.1f}% (need {self.config.min_prior_advance_pct}%)")
            is_valid = False

        # Check contractions
        validity_reasons.append(f"Contractions: {len(valid_contractions)} found")

        # Check final contraction
        final_range = valid_contractions[-1].range_pct
        if final_range <= self.config.ideal_final_contraction_pct:
            validity_reasons.append(f"Final contraction tight: {final_range:.1f}%")
        elif final_range <= self.config.max_final_contraction_pct:
            validity_reasons.append(f"Final contraction acceptable: {final_range:.1f}%")
        else:
            validity_reasons.append(f"Final contraction wide: {final_range:.1f}%")

        # Add filter reasons
        validity_reasons.extend(filter_reasons)

        # Calculate pattern bounds
        pivot_price = max(c.swing_high.price for c in valid_contractions)
        support_price = valid_contractions[-1].swing_low.price
        stop_loss_price = support_price * 0.98  # 2% below support

        # Calculate average tightening ratio
        tightening_ratios = [c.tightening_ratio for c in valid_contractions[1:] if c.tightening_ratio > 0]
        avg_tightening = np.mean(tightening_ratios) if tightening_ratios else 0

        return VCPPattern(
            contractions=valid_contractions,
            is_valid=is_valid,
            validity_reasons=validity_reasons,
            contraction_quality=contraction_quality,
            volume_quality=volume_quality,
            trend_template_score=trend_template.score,
            overall_score=overall_score,
            trend_template=trend_template,
            prior_uptrend=prior_uptrend,
            pivot_price=pivot_price,
            support_price=support_price,
            stop_loss_price=stop_loss_price,
            base_duration_days=base_duration,
            num_contractions=len(valid_contractions),
            first_contraction_pct=valid_contractions[0].range_pct,
            final_contraction_pct=final_range,
            avg_tightening_ratio=avg_tightening
        )


def test_detector():
    """Test the V4 VCP detector with sample data."""
    import yfinance as yf

    print("="*70)
    print("VCP Detector V4 Test")
    print("="*70)

    # Test symbols
    test_symbols = ['NVDA', 'AAPL', 'MSFT', 'GE', 'META']

    # Get SPY for RS calculation
    print("\nFetching SPY data for RS calculation...")
    spy_df = yf.download('SPY', period='2y', progress=False)
    if isinstance(spy_df.columns, pd.MultiIndex):
        spy_df.columns = spy_df.columns.get_level_values(0)

    detector = VCPDetectorV4()

    for symbol in test_symbols:
        print(f"\n{'='*70}")
        print(f"Analyzing {symbol}")
        print('='*70)

        # Get data
        df = yf.download(symbol, period='2y', progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if len(df) < 252:
            print(f"Insufficient data for {symbol}")
            continue

        # Analyze
        pattern = detector.analyze_pattern(df, lookback_days=120, spy_df=spy_df)

        if pattern:
            print(f"\nVCP Pattern Found: {'VALID' if pattern.is_valid else 'INVALID'}")
            print(f"\n--- Scores ---")
            print(f"Overall Score: {pattern.overall_score:.1f}")
            print(f"Contraction Quality: {pattern.contraction_quality:.1f}")
            print(f"Volume Quality: {pattern.volume_quality:.1f}")
            print(f"Trend Template Score: {pattern.trend_template_score:.1f}")

            print(f"\n--- Prior Uptrend ---")
            print(f"Valid: {pattern.prior_uptrend.is_valid}")
            print(f"Advance: {pattern.prior_uptrend.advance_pct:.1f}%")
            print(f"Message: {pattern.prior_uptrend.message}")

            print(f"\n--- Trend Template Checks ---")
            for check, (status, msg) in pattern.trend_template.checks.items():
                print(f"  {check}: {status.value} - {msg}")

            print(f"\n--- Pattern Details ---")
            print(f"Contractions: {pattern.num_contractions}")
            print(f"Base Duration: {pattern.base_duration_days} days")
            print(f"First Contraction: {pattern.first_contraction_pct:.1f}%")
            print(f"Final Contraction: {pattern.final_contraction_pct:.1f}%")
            print(f"Avg Tightening Ratio: {pattern.avg_tightening_ratio:.2f}")
            print(f"Pivot Price: ${pattern.pivot_price:.2f}")
            print(f"Support Price: ${pattern.support_price:.2f}")
            print(f"Stop Loss: ${pattern.stop_loss_price:.2f}")

            print(f"\n--- Contraction Details ---")
            for i, c in enumerate(pattern.contractions):
                print(f"  C{i+1}: {c.swing_high.date.strftime('%Y-%m-%d')} -> {c.swing_low.date.strftime('%Y-%m-%d')}")
                print(f"       High: ${c.swing_high.price:.2f}, Low: ${c.swing_low.price:.2f}")
                print(f"       Range: {c.range_pct:.1f}%, Duration: {c.duration_days}d, Vol Ratio: {c.avg_volume_ratio:.2f}x")
                if c.tightening_ratio > 0:
                    print(f"       Tightening: {c.tightening_ratio:.2f}x previous")

            print(f"\n--- Validity Reasons ---")
            for reason in pattern.validity_reasons:
                print(f"  - {reason}")
        else:
            print("No VCP pattern found")


if __name__ == '__main__':
    test_detector()
