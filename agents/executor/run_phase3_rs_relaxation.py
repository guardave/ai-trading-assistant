#!/usr/bin/env python3
"""
Phase 3 RS Relaxation Backtest Executor
Tests RS thresholds [70, 75, 80, 85, 90] with and without proximity filtering
"""
import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

# Add backtest folder to path
sys.path.insert(0, '/home/david/dev/ai-trading-assistant/backtest')

from backtest_framework_v2 import DataCache, Backtester, BacktestResult, print_results
from strategy_params import VCP_PARAMS, RISK_PARAMS

def get_stock_universe():
    """Return same 50 stocks from Phase 1 for consistency"""
    stocks = {
        'Technology': ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'AMD', 'AVGO', 'ORCL', 'ADBE', 'CRM'],
        'Healthcare': ['LLY', 'UNH', 'JNJ', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'PFE', 'BMY'],
        'Industrials': ['GE', 'CAT', 'BA', 'HON', 'UPS', 'RTX', 'DE', 'LMT', 'MMM', 'GD'],
        'Financials': ['JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'AXP', 'BLK', 'C'],
        'Consumer': ['HD', 'NKE', 'SBUX', 'MCD', 'LOW', 'TGT', 'TJX', 'BKNG', 'CMG', 'COST'],
    }

    all_stocks = []
    for sector, tickers in stocks.items():
        all_stocks.extend(tickers)

    return all_stocks

def run_configuration(backtester, symbols, rs_threshold, proximity_filter, risk_params):
    """Run a single backtest configuration"""

    # Configure VCP parameters
    vcp_params = VCP_PARAMS['current'].copy()
    vcp_params['rs_rating_min'] = rs_threshold

    if proximity_filter:
        vcp_params['min_proximity_score'] = 50
    else:
        vcp_params['min_proximity_score'] = 0

    config_name = f"VCP RS{rs_threshold}"
    if proximity_filter:
        config_name += " + Proximity>=50"

    print(f"\n{'='*70}")
    print(f"Running: {config_name}")
    print(f"{'='*70}")

    result = backtester.run_backtest(
        symbols=symbols,
        strategy_name=config_name,
        params=vcp_params,
        risk_params=risk_params,
        exit_method='trailing_stop',
        start_date='2022-01-01'
    )

    print_results(result)

    return result

def main():
    print("=" * 70)
    print("Phase 3 - RS Relaxation Backtest Execution")
    print("=" * 70)
    print()

    # Initialize data cache
    cache = DataCache(cache_dir='/home/david/dev/ai-trading-assistant/cache')

    # Get stock universe
    symbols = get_stock_universe()
    print(f"Testing on {len(symbols)} US stocks (same as Phase 1)")
    print(f"Test period: 2022-01-01 to 2024-11-29")
    print()

    # Initialize backtester
    backtester = Backtester(cache)

    # Configure trailing stop risk parameters
    risk_params = {
        'default_stop_loss_pct': 7.0,
        'trailing_stop_activation_pct': 8.0,
        'trailing_stop_distance_pct': 5.0,
    }

    # RS thresholds to test
    rs_thresholds = [70, 75, 80, 85, 90]

    # Store all results
    all_results = {}

    # Run all configurations
    for rs_threshold in rs_thresholds:
        # Configuration 1: No proximity filter
        result_no_prox = run_configuration(
            backtester, symbols, rs_threshold,
            proximity_filter=False,
            risk_params=risk_params
        )

        config_key_no_prox = f"rs{rs_threshold}_no_proximity"
        all_results[config_key_no_prox] = result_no_prox

        # Configuration 2: With proximity filter >= 50
        result_with_prox = run_configuration(
            backtester, symbols, rs_threshold,
            proximity_filter=True,
            risk_params=risk_params
        )

        config_key_with_prox = f"rs{rs_threshold}_proximity_50"
        all_results[config_key_with_prox] = result_with_prox

    # Calculate total trades across all configurations
    total_trades_all = sum(r.total_trades for r in all_results.values())

    print("\n" + "=" * 70)
    print(f"Phase 3 Complete - Total Trades Across All Configs: {total_trades_all}")
    print("=" * 70)

    # Save raw results
    results_dir = '/home/david/dev/ai-trading-assistant/agents/executor/results'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Build JSON results
    results_data = {
        'timestamp': timestamp,
        'test_period': '2022-01-01 to 2024-11-29',
        'stock_count': len(symbols),
        'symbols': symbols,
        'total_trades_all_configs': total_trades_all,
        'configurations': {}
    }

    for config_key, result in all_results.items():
        results_data['configurations'][config_key] = {
            'strategy_name': result.strategy_name,
            'params': result.params,
            'total_trades': result.total_trades,
            'winning_trades': result.winning_trades,
            'losing_trades': result.losing_trades,
            'win_rate': result.win_rate,
            'avg_win_pct': result.avg_win_pct,
            'avg_loss_pct': result.avg_loss_pct,
            'profit_factor': result.profit_factor,
            'avg_rr_realized': result.avg_rr_realized,
            'total_return_pct': result.total_return_pct,
            'avg_days_held': result.avg_days_held,
            'proximity_correlation': result.proximity_correlation,
            'high_proximity_win_rate': result.high_proximity_win_rate,
            'low_proximity_win_rate': result.low_proximity_win_rate,
        }

    json_file = os.path.join(results_dir, 'phase3_rs_relaxation_raw.json')
    with open(json_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"\nRaw results saved to: {json_file}")

    # Create RS comparison table
    rs_comparison = []

    for rs_threshold in rs_thresholds:
        # No proximity filter
        key_no_prox = f"rs{rs_threshold}_no_proximity"
        result_no_prox = all_results[key_no_prox]

        # Calculate average proximity score across all trades
        avg_proximity_no_prox = np.mean([
            t.proximity_score for t in result_no_prox.trades
            if t.proximity_score is not None
        ]) if result_no_prox.trades else 0.0

        rs_comparison.append({
            'rs_threshold': rs_threshold,
            'proximity_filter': 'none',
            'total_trades': result_no_prox.total_trades,
            'win_rate': f"{result_no_prox.win_rate:.2f}",
            'profit_factor': f"{result_no_prox.profit_factor:.2f}",
            'total_return_pct': f"{result_no_prox.total_return_pct:.2f}",
            'avg_days_held': f"{result_no_prox.avg_days_held:.1f}",
            'avg_proximity': f"{avg_proximity_no_prox:.1f}",
        })

        # With proximity filter
        key_with_prox = f"rs{rs_threshold}_proximity_50"
        result_with_prox = all_results[key_with_prox]

        avg_proximity_with_prox = np.mean([
            t.proximity_score for t in result_with_prox.trades
            if t.proximity_score is not None
        ]) if result_with_prox.trades else 0.0

        rs_comparison.append({
            'rs_threshold': rs_threshold,
            'proximity_filter': 'min_50',
            'total_trades': result_with_prox.total_trades,
            'win_rate': f"{result_with_prox.win_rate:.2f}",
            'profit_factor': f"{result_with_prox.profit_factor:.2f}",
            'total_return_pct': f"{result_with_prox.total_return_pct:.2f}",
            'avg_days_held': f"{result_with_prox.avg_days_held:.1f}",
            'avg_proximity': f"{avg_proximity_with_prox:.1f}",
        })

    df_rs_comparison = pd.DataFrame(rs_comparison)
    rs_csv_file = '/home/david/dev/ai-trading-assistant/agents/executor/handoff/phase3_rs_comparison.csv'
    df_rs_comparison.to_csv(rs_csv_file, index=False)

    print(f"RS comparison saved to: {rs_csv_file}")

    # Create proximity impact table
    proximity_impact = []

    for rs_threshold in rs_thresholds:
        no_prox = all_results[f"rs{rs_threshold}_no_proximity"]
        with_prox = all_results[f"rs{rs_threshold}_proximity_50"]

        trade_reduction = no_prox.total_trades - with_prox.total_trades
        trade_reduction_pct = (trade_reduction / no_prox.total_trades * 100) if no_prox.total_trades > 0 else 0

        win_rate_change = with_prox.win_rate - no_prox.win_rate
        profit_factor_change = with_prox.profit_factor - no_prox.profit_factor

        proximity_impact.append({
            'rs_threshold': rs_threshold,
            'trades_no_prox': no_prox.total_trades,
            'trades_with_prox': with_prox.total_trades,
            'trade_reduction': trade_reduction,
            'trade_reduction_pct': f"{trade_reduction_pct:.1f}",
            'win_rate_no_prox': f"{no_prox.win_rate:.2f}",
            'win_rate_with_prox': f"{with_prox.win_rate:.2f}",
            'win_rate_change': f"{win_rate_change:+.2f}",
            'profit_factor_no_prox': f"{no_prox.profit_factor:.2f}",
            'profit_factor_with_prox': f"{with_prox.profit_factor:.2f}",
            'profit_factor_change': f"{profit_factor_change:+.2f}",
        })

    df_proximity_impact = pd.DataFrame(proximity_impact)
    proximity_csv_file = '/home/david/dev/ai-trading-assistant/agents/executor/handoff/phase3_proximity_impact.csv'
    df_proximity_impact.to_csv(proximity_csv_file, index=False)

    print(f"Proximity impact saved to: {proximity_csv_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("Phase 3 Summary")
    print("=" * 70)
    print(f"\nTotal trades across all configurations: {total_trades_all}")
    print(f"\nRS Threshold Impact:")
    print(df_rs_comparison.to_string(index=False))
    print(f"\nProximity Filter Impact:")
    print(df_proximity_impact.to_string(index=False))

    # Identify best configuration
    best_config = None
    best_profit_factor = 0

    for config_key, result in all_results.items():
        if result.profit_factor > best_profit_factor:
            best_profit_factor = result.profit_factor
            best_config = config_key

    if best_config:
        print(f"\nBest Configuration (by profit factor): {best_config}")
        print(f"  Profit Factor: {best_profit_factor:.2f}")
        print(f"  Trades: {all_results[best_config].total_trades}")
        print(f"  Win Rate: {all_results[best_config].win_rate:.1f}%")

    print("\n" + "=" * 70)
    print("Files created:")
    print(f"  - {json_file}")
    print(f"  - {rs_csv_file}")
    print(f"  - {proximity_csv_file}")
    print("\nReady for handoff to Reviewer Agent")
    print("=" * 70)

if __name__ == '__main__':
    main()
