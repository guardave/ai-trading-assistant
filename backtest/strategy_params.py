"""
Strategy Parameters Extracted from Reference Project
For backtesting and optimization
"""

# =============================================================================
# VCP (Volatility Contraction Pattern) Strategy - Minervini
# =============================================================================
VCP_PARAMS = {
    # Current parameters from reference project
    'current': {
        'contraction_threshold': 0.20,          # Max 20% range in consolidation
        'volume_dry_up_ratio': 0.5,             # Volume should be < 50% of average
        'min_consolidation_weeks': 4,           # Minimum 4 weeks of consolidation
        'pivot_breakout_volume_multiplier': 1.5, # 1.5x average volume on breakout
        'rs_rating_min': 90,                    # Minimum RS rating (top 10%)
        'distance_from_52w_high_max': 0.25,     # Within 25% of 52-week high
        'price_above_200ma': True,              # Must be above 200-day MA
    },

    # Parameter ranges for optimization
    'ranges': {
        'contraction_threshold': [0.10, 0.15, 0.20, 0.25, 0.30],
        'volume_dry_up_ratio': [0.3, 0.4, 0.5, 0.6, 0.7],
        'min_consolidation_weeks': [3, 4, 5, 6, 8],
        'pivot_breakout_volume_multiplier': [1.2, 1.5, 1.8, 2.0],
        'rs_rating_min': [70, 80, 85, 90, 95],
        'distance_from_52w_high_max': [0.15, 0.20, 0.25, 0.30, 0.35],
    }
}

# =============================================================================
# Pivot Breakout Strategy
# =============================================================================
PIVOT_PARAMS = {
    # Current parameters from reference project
    'current': {
        'min_base_weeks': 4,                    # Minimum base length
        'max_base_weeks': 52,                   # Maximum base length
        'max_base_depth': 0.35,                 # Maximum 35% correction
        'min_base_depth': 0.08,                 # Minimum 8% correction
        'pivot_volume_multiplier': 1.5,         # 1.5x volume on breakout
        'handle_max_depth': 0.12,               # Max 12% handle depth
        'rs_rating_min': 80,                    # Minimum RS rating
        'distance_from_52w_high_max': 0.30,     # Within 30% of 52-week high
        'price_above_50ma': True,               # Must be above 50-day MA
        'price_above_200ma': True,              # Must be above 200-day MA
    },

    # Parameter ranges for optimization
    'ranges': {
        'min_base_weeks': [3, 4, 5, 6],
        'max_base_weeks': [26, 40, 52, 65],
        'max_base_depth': [0.25, 0.30, 0.35, 0.40],
        'min_base_depth': [0.05, 0.08, 0.10, 0.12],
        'pivot_volume_multiplier': [1.2, 1.5, 1.8, 2.0],
        'handle_max_depth': [0.08, 0.10, 0.12, 0.15],
        'rs_rating_min': [70, 75, 80, 85, 90],
        'distance_from_52w_high_max': [0.20, 0.25, 0.30, 0.35],
    }
}

# =============================================================================
# Cup with Handle Strategy
# =============================================================================
CUP_PARAMS = {
    # Current parameters from reference project
    'current': {
        'min_cup_weeks': 7,                     # Minimum cup duration
        'max_cup_weeks': 65,                    # Maximum cup duration
        'min_cup_depth': 0.12,                  # Minimum 12% depth
        'max_cup_depth': 0.33,                  # Maximum 33% depth
        'handle_max_depth': 0.12,               # Max 12% handle depth
        'handle_min_weeks': 1,                  # Minimum handle weeks
        'handle_max_weeks': 4,                  # Maximum handle weeks
        'volume_multiplier': 1.5,               # Volume on breakout
        'u_shape_position_min': 0.3,            # Low must be after 30% of cup
        'u_shape_position_max': 0.7,            # Low must be before 70% of cup
    },

    # Parameter ranges for optimization
    'ranges': {
        'min_cup_weeks': [5, 7, 10],
        'max_cup_weeks': [40, 52, 65],
        'min_cup_depth': [0.10, 0.12, 0.15],
        'max_cup_depth': [0.30, 0.33, 0.40],
        'handle_max_depth': [0.08, 0.10, 0.12, 0.15],
        'volume_multiplier': [1.2, 1.5, 1.8, 2.0],
    }
}

# =============================================================================
# Risk Management Parameters
# =============================================================================
RISK_PARAMS = {
    'current': {
        'max_risk_per_trade_pct': 2.0,          # Maximum 2% risk per trade
        'trailing_stop_activation_pct': 8.0,    # Move stop to breakeven at +8%
        'trailing_stop_distance_pct': 5.0,      # Trail by 5% after activation
        'max_portfolio_heat_pct': 6.0,          # Maximum total portfolio risk
        'reward_risk_ratio_min': 3.0,           # Minimum 3:1 R:R ratio
        'default_stop_loss_pct': 7.0,           # Default 7% stop loss
        'default_target_pct': 20.0,             # Default 20% target
    },

    # Parameter ranges for optimization
    'ranges': {
        'default_stop_loss_pct': [5.0, 6.0, 7.0, 8.0, 10.0],
        'default_target_pct': [15.0, 20.0, 25.0, 30.0],
        'trailing_stop_activation_pct': [5.0, 8.0, 10.0, 12.0],
        'reward_risk_ratio_min': [2.0, 2.5, 3.0, 3.5],
    }
}

# =============================================================================
# J-Law / O'Neil Filters
# =============================================================================
FILTER_PARAMS = {
    'current': {
        'institutional_ownership_min': 0.40,    # Minimum 40% institutional ownership
        'profitable_company': True,             # Must have positive profit margin
        'earnings_acceleration': True,          # Requires earnings growth (not implemented)
    }
}

# =============================================================================
# RS (Relative Strength) Calculation
# =============================================================================
RS_PARAMS = {
    'current': {
        'periods': [63, 126, 189, 252],         # Trading days for each period
        'weights': [0.4, 0.2, 0.2, 0.2],        # Weights (recent weighted more)
        'benchmark': 'SPY',                     # Benchmark for US stocks

        # TradingView RS Score to Rating mapping thresholds
        'rating_thresholds': {
            99: 195.93,
            90: 117.11,
            70: 99.04,
            50: 91.66,
            30: 80.96,
            10: 53.64,
            2: 24.86,
        }
    }
}
