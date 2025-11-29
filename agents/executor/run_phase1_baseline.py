#!/usr/bin/env python3
"""
Phase 1 Baseline Backtest Executor
Executes VCP strategy with current parameters from reference project
"""
import sys
import os
import json
import pandas as pd
from datetime import datetime

# Add backtest folder to path
sys.path.insert(0, '/home/david/dev/ai-trading-assistant/backtest')

from backtest_framework_v2 import DataCache, Backtester, BacktestResult, print_results
from strategy_params import VCP_PARAMS, RISK_PARAMS

def get_stock_universe():
    """Return a representative sample of US stocks for testing"""
    # Start with a smaller universe to manage API limits
    stocks = {
        'Technology': ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'AMD', 'AVGO', 'ORCL', 'ADBE', 'CRM'],
        'Healthcare': ['LLY', 'UNH', 'JNJ', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'PFE', 'BMY'],
        'Industrials': ['GE', 'CAT', 'BA', 'HON', 'UPS', 'RTX', 'DE', 'LMT', 'MMM', 'GD'],
        'Financials': ['JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'AXP', 'BLK', 'C'],
        'Consumer': ['HD', 'NKE', 'SBUX', 'MCD', 'LOW', 'TGT', 'TJX', 'BKNG', 'CMG', 'COST'],
    }

    # Flatten to list
    all_stocks = []
    for sector, tickers in stocks.items():
        all_stocks.extend(tickers)

    return all_stocks

def main():
    print("=" * 70)
    print("Phase 1 - Baseline Backtest Execution")
    print("=" * 70)
    print()

    # Initialize data cache
    cache = DataCache(cache_dir='/home/david/dev/ai-trading-assistant/cache')

    # Get stock universe
    symbols = get_stock_universe()
    print(f"Testing on {len(symbols)} US stocks")
    print(f"Symbols: {', '.join(symbols[:10])}...")
    print()

    # Initialize backtester
    backtester = Backtester(cache)

    # Configure parameters
    vcp_params = VCP_PARAMS['current'].copy()

    # IMPORTANT: User requested R:R >= 3.0
    # Current reference params have 7% stop, 20% target (R:R = 2.86)
    # Adjust to meet requirement: 8% stop, 15% target would be 1.875 (too low)
    # Better: 7% stop, 21% target = 3.0 R:R
    # Or: 8% stop, 24% target = 3.0 R:R

    # Let's test with two configurations:
    # Config 1: Original reference (7% stop, 20% target) - baseline as-is
    # Config 2: Adjusted for R:R 3.0 (8% stop, 24% target)

    risk_params_baseline = {
        'default_stop_loss_pct': 7.0,
        'default_target_pct': 20.0,  # R:R = 2.86
        'trailing_stop_activation_pct': 8.0,
        'trailing_stop_distance_pct': 5.0,
    }

    risk_params_adjusted = {
        'default_stop_loss_pct': 8.0,
        'default_target_pct': 24.0,  # R:R = 3.0
        'trailing_stop_activation_pct': 8.0,
        'trailing_stop_distance_pct': 5.0,
    }

    print("Configuration 1: Reference Baseline")
    print(f"  Stop Loss: {risk_params_baseline['default_stop_loss_pct']}%")
    print(f"  Target: {risk_params_baseline['default_target_pct']}%")
    print(f"  Expected R:R: {risk_params_baseline['default_target_pct'] / risk_params_baseline['default_stop_loss_pct']:.2f}")
    print()

    # Run baseline backtest with fixed target
    print("Running baseline backtest (fixed target)...")
    result_baseline_fixed = backtester.run_backtest(
        symbols=symbols,
        strategy_name='VCP Baseline (Reference)',
        params=vcp_params,
        risk_params=risk_params_baseline,
        exit_method='fixed_target',
        start_date='2022-01-01'
    )

    print_results(result_baseline_fixed)

    # Run baseline backtest with trailing stop
    print("\nRunning baseline backtest (trailing stop)...")
    result_baseline_trailing = backtester.run_backtest(
        symbols=symbols,
        strategy_name='VCP Baseline (Reference)',
        params=vcp_params,
        risk_params=risk_params_baseline,
        exit_method='trailing_stop',
        start_date='2022-01-01'
    )

    print_results(result_baseline_trailing)

    # Run adjusted R:R backtest
    print("\nConfiguration 2: Adjusted for R:R >= 3.0")
    print(f"  Stop Loss: {risk_params_adjusted['default_stop_loss_pct']}%")
    print(f"  Target: {risk_params_adjusted['default_target_pct']}%")
    print(f"  Expected R:R: {risk_params_adjusted['default_target_pct'] / risk_params_adjusted['default_stop_loss_pct']:.2f}")
    print()

    print("Running adjusted R:R backtest (fixed target)...")
    result_adjusted_fixed = backtester.run_backtest(
        symbols=symbols,
        strategy_name='VCP Adjusted R:R 3.0',
        params=vcp_params,
        risk_params=risk_params_adjusted,
        exit_method='fixed_target',
        start_date='2022-01-01'
    )

    print_results(result_adjusted_fixed)

    # Save results
    results_dir = '/home/david/dev/ai-trading-assistant/agents/executor/results'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save raw JSON
    results_data = {
        'timestamp': timestamp,
        'test_period': '2022-01-01 to 2024-11-29',
        'stock_count': len(symbols),
        'symbols': symbols,
        'baseline_fixed': {
            'params': vcp_params,
            'risk_params': risk_params_baseline,
            'total_trades': result_baseline_fixed.total_trades,
            'win_rate': result_baseline_fixed.win_rate,
            'avg_rr_realized': result_baseline_fixed.avg_rr_realized,
            'profit_factor': result_baseline_fixed.profit_factor,
            'total_return_pct': result_baseline_fixed.total_return_pct,
            'avg_days_held': result_baseline_fixed.avg_days_held,
            'proximity_correlation': result_baseline_fixed.proximity_correlation,
            'high_proximity_win_rate': result_baseline_fixed.high_proximity_win_rate,
            'low_proximity_win_rate': result_baseline_fixed.low_proximity_win_rate,
        },
        'baseline_trailing': {
            'params': vcp_params,
            'risk_params': risk_params_baseline,
            'total_trades': result_baseline_trailing.total_trades,
            'win_rate': result_baseline_trailing.win_rate,
            'avg_rr_realized': result_baseline_trailing.avg_rr_realized,
            'profit_factor': result_baseline_trailing.profit_factor,
            'total_return_pct': result_baseline_trailing.total_return_pct,
            'avg_days_held': result_baseline_trailing.avg_days_held,
            'proximity_correlation': result_baseline_trailing.proximity_correlation,
            'high_proximity_win_rate': result_baseline_trailing.high_proximity_win_rate,
            'low_proximity_win_rate': result_baseline_trailing.low_proximity_win_rate,
        },
        'adjusted_rr_fixed': {
            'params': vcp_params,
            'risk_params': risk_params_adjusted,
            'total_trades': result_adjusted_fixed.total_trades,
            'win_rate': result_adjusted_fixed.win_rate,
            'avg_rr_realized': result_adjusted_fixed.avg_rr_realized,
            'profit_factor': result_adjusted_fixed.profit_factor,
            'total_return_pct': result_adjusted_fixed.total_return_pct,
            'avg_days_held': result_adjusted_fixed.avg_days_held,
            'proximity_correlation': result_adjusted_fixed.proximity_correlation,
            'high_proximity_win_rate': result_adjusted_fixed.high_proximity_win_rate,
            'low_proximity_win_rate': result_adjusted_fixed.low_proximity_win_rate,
        }
    }

    json_file = os.path.join(results_dir, 'phase1_baseline_raw.json')
    with open(json_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"\nRaw results saved to: {json_file}")

    # Create handoff CSV
    handoff_data = {
        'metric': [
            'test_period',
            'stock_count',
            'baseline_fixed_total_trades',
            'baseline_fixed_win_rate',
            'baseline_fixed_avg_rr_realized',
            'baseline_fixed_profit_factor',
            'baseline_fixed_total_return_pct',
            'baseline_fixed_avg_days_held',
            'baseline_trailing_total_trades',
            'baseline_trailing_win_rate',
            'baseline_trailing_avg_rr_realized',
            'baseline_trailing_profit_factor',
            'baseline_trailing_total_return_pct',
            'baseline_trailing_avg_days_held',
            'adjusted_rr_total_trades',
            'adjusted_rr_win_rate',
            'adjusted_rr_avg_rr_realized',
            'adjusted_rr_profit_factor',
            'adjusted_rr_total_return_pct',
            'adjusted_rr_avg_days_held',
            'proximity_correlation',
            'high_proximity_win_rate',
            'low_proximity_win_rate',
        ],
        'value': [
            '2022-01-01 to 2024-11-29',
            len(symbols),
            result_baseline_fixed.total_trades,
            f"{result_baseline_fixed.win_rate:.2f}",
            f"{result_baseline_fixed.avg_rr_realized:.2f}",
            f"{result_baseline_fixed.profit_factor:.2f}",
            f"{result_baseline_fixed.total_return_pct:.2f}",
            f"{result_baseline_fixed.avg_days_held:.1f}",
            result_baseline_trailing.total_trades,
            f"{result_baseline_trailing.win_rate:.2f}",
            f"{result_baseline_trailing.avg_rr_realized:.2f}",
            f"{result_baseline_trailing.profit_factor:.2f}",
            f"{result_baseline_trailing.total_return_pct:.2f}",
            f"{result_baseline_trailing.avg_days_held:.1f}",
            result_adjusted_fixed.total_trades,
            f"{result_adjusted_fixed.win_rate:.2f}",
            f"{result_adjusted_fixed.avg_rr_realized:.2f}",
            f"{result_adjusted_fixed.profit_factor:.2f}",
            f"{result_adjusted_fixed.total_return_pct:.2f}",
            f"{result_adjusted_fixed.avg_days_held:.1f}",
            f"{result_baseline_fixed.proximity_correlation:.3f}" if result_baseline_fixed.proximity_correlation else 'N/A',
            f"{result_baseline_fixed.high_proximity_win_rate:.1f}" if result_baseline_fixed.high_proximity_win_rate else 'N/A',
            f"{result_baseline_fixed.low_proximity_win_rate:.1f}" if result_baseline_fixed.low_proximity_win_rate else 'N/A',
        ]
    }

    df_handoff = pd.DataFrame(handoff_data)
    csv_file = '/home/david/dev/ai-trading-assistant/agents/executor/handoff/phase1_baseline_results.csv'
    df_handoff.to_csv(csv_file, index=False)

    print(f"Handoff CSV saved to: {csv_file}")

    # Create detailed trade log
    trades_df_list = []

    for trade in result_baseline_fixed.trades:
        trades_df_list.append({
            'config': 'baseline_fixed',
            'symbol': trade.symbol,
            'pattern': trade.pattern,
            'entry_date': trade.entry_date,
            'entry_price': trade.entry_price,
            'exit_date': trade.exit_date,
            'exit_price': trade.exit_price,
            'exit_reason': trade.exit_reason,
            'pnl_pct': trade.pnl_pct,
            'days_held': trade.days_held,
            'rs_rating': trade.rs_rating,
            'proximity_score': trade.proximity_score,
            'num_contractions': trade.num_contractions,
            'theoretical_rr': trade.theoretical_rr,
            'realized_rr': trade.realized_rr,
        })

    for trade in result_baseline_trailing.trades:
        trades_df_list.append({
            'config': 'baseline_trailing',
            'symbol': trade.symbol,
            'pattern': trade.pattern,
            'entry_date': trade.entry_date,
            'entry_price': trade.entry_price,
            'exit_date': trade.exit_date,
            'exit_price': trade.exit_price,
            'exit_reason': trade.exit_reason,
            'pnl_pct': trade.pnl_pct,
            'days_held': trade.days_held,
            'rs_rating': trade.rs_rating,
            'proximity_score': trade.proximity_score,
            'num_contractions': trade.num_contractions,
            'theoretical_rr': trade.theoretical_rr,
            'realized_rr': trade.realized_rr,
        })

    for trade in result_adjusted_fixed.trades:
        trades_df_list.append({
            'config': 'adjusted_rr_fixed',
            'symbol': trade.symbol,
            'pattern': trade.pattern,
            'entry_date': trade.entry_date,
            'entry_price': trade.entry_price,
            'exit_date': trade.exit_date,
            'exit_price': trade.exit_price,
            'exit_reason': trade.exit_reason,
            'pnl_pct': trade.pnl_pct,
            'days_held': trade.days_held,
            'rs_rating': trade.rs_rating,
            'proximity_score': trade.proximity_score,
            'num_contractions': trade.num_contractions,
            'theoretical_rr': trade.theoretical_rr,
            'realized_rr': trade.realized_rr,
        })

    trades_df = pd.DataFrame(trades_df_list)
    trades_file = os.path.join(results_dir, 'phase1_all_trades.csv')
    trades_df.to_csv(trades_file, index=False)

    print(f"All trades saved to: {trades_file}")

    print("\n" + "=" * 70)
    print("Phase 1 Baseline Backtest Complete")
    print("=" * 70)
    print(f"\nFiles created:")
    print(f"  - {json_file}")
    print(f"  - {csv_file}")
    print(f"  - {trades_file}")
    print("\nReady for handoff to Reviewer Agent")

if __name__ == '__main__':
    main()
