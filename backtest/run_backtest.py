#!/usr/bin/env python3
"""
Run Strategy Backtests
Tests VCP and Pivot strategies with current and optimized parameters
"""
import os
import sys
import json
from datetime import datetime
from typing import List, Dict, Any
import itertools

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest_framework import DataCache, Backtester, BacktestResult, print_results
from strategy_params import VCP_PARAMS, PIVOT_PARAMS, RISK_PARAMS

# =============================================================================
# Test Universe - Mix of historically strong stocks
# =============================================================================
TEST_SYMBOLS = [
    # Tech Leaders
    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'AMZN', 'TSLA', 'AMD', 'AVGO', 'CRM',
    # Growth Stocks
    'NFLX', 'ADBE', 'NOW', 'PANW', 'SNOW', 'DDOG', 'NET', 'CRWD', 'ZS', 'MDB',
    # Healthcare
    'LLY', 'UNH', 'JNJ', 'ABBV', 'MRK', 'PFE', 'ISRG', 'VRTX', 'REGN', 'GILD',
    # Industrials
    'GE', 'CAT', 'BA', 'UNP', 'HON', 'RTX', 'DE', 'LMT', 'MMM', 'FDX',
    # Financials
    'JPM', 'V', 'MA', 'BAC', 'GS', 'MS', 'BLK', 'AXP', 'C', 'WFC',
    # Consumer
    'HD', 'LOW', 'NKE', 'SBUX', 'MCD', 'DIS', 'COST', 'WMT', 'TGT', 'LULU',
]


def save_results(results: List[BacktestResult], filename: str):
    """Save backtest results to JSON"""
    output = []
    for r in results:
        output.append({
            'strategy': r.strategy_name,
            'params': r.params,
            'total_trades': r.total_trades,
            'win_rate': r.win_rate,
            'avg_win_pct': r.avg_win_pct,
            'avg_loss_pct': r.avg_loss_pct,
            'profit_factor': r.profit_factor,
            'avg_rr_realized': r.avg_rr_realized,
            'total_return_pct': r.total_return_pct,
            'avg_days_held': r.avg_days_held,
        })

    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {filename}")


def run_parameter_sweep(backtester: Backtester, strategy_name: str,
                        base_params: Dict[str, Any],
                        param_ranges: Dict[str, List[Any]],
                        risk_params: Dict[str, Any],
                        symbols: List[str]) -> List[BacktestResult]:
    """
    Run parameter sweep to find optimal settings

    Args:
        backtester: Backtester instance
        strategy_name: 'VCP' or 'Pivot'
        base_params: Base parameters
        param_ranges: Parameter ranges to test
        risk_params: Risk parameters
        symbols: Symbols to test

    Returns:
        List of BacktestResult sorted by profit factor
    """
    results = []

    # Generate parameter combinations (limited to avoid explosion)
    # We'll test key parameters individually first
    key_params = ['rs_rating_min', 'contraction_threshold' if strategy_name == 'VCP' else 'max_base_depth']

    for key_param in key_params:
        if key_param not in param_ranges:
            continue

        for value in param_ranges[key_param]:
            test_params = base_params.copy()
            test_params[key_param] = value

            print(f"\nTesting {strategy_name} with {key_param}={value}...")

            result = backtester.run_backtest(
                symbols=symbols,
                strategy_name=f"{strategy_name}_{key_param}={value}",
                params=test_params,
                risk_params=risk_params,
                start_date='2023-01-01'
            )

            results.append(result)

            print(f"  Trades: {result.total_trades}, Win Rate: {result.win_rate:.1f}%, "
                  f"PF: {result.profit_factor:.2f}")

    # Sort by profit factor
    results.sort(key=lambda x: x.profit_factor, reverse=True)

    return results


def run_risk_param_sweep(backtester: Backtester, strategy_name: str,
                         strategy_params: Dict[str, Any],
                         risk_ranges: Dict[str, List[Any]],
                         symbols: List[str]) -> List[BacktestResult]:
    """
    Test different stop loss and target combinations
    """
    results = []

    stop_losses = risk_ranges.get('default_stop_loss_pct', [7.0])
    targets = risk_ranges.get('default_target_pct', [20.0])

    for stop, target in itertools.product(stop_losses, targets):
        risk_params = {
            'default_stop_loss_pct': stop,
            'default_target_pct': target,
        }

        print(f"\nTesting {strategy_name} with Stop={stop}%, Target={target}%...")

        result = backtester.run_backtest(
            symbols=symbols,
            strategy_name=f"{strategy_name}_S{stop}_T{target}",
            params=strategy_params,
            risk_params=risk_params,
            start_date='2023-01-01'
        )

        # Add risk params to result params for tracking
        result.params['stop_loss_pct'] = stop
        result.params['target_pct'] = target

        results.append(result)

        print(f"  Trades: {result.total_trades}, Win Rate: {result.win_rate:.1f}%, "
              f"PF: {result.profit_factor:.2f}, Avg R:R: {result.avg_rr_realized:.2f}")

    # Sort by profit factor
    results.sort(key=lambda x: x.profit_factor, reverse=True)

    return results


def main():
    print("=" * 70)
    print("  AI Trading Assistant - Strategy Backtesting")
    print("=" * 70)
    print(f"\nTest Universe: {len(TEST_SYMBOLS)} symbols")
    print(f"Test Period: 2023-01-01 to present")

    # Initialize
    cache_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'cache')
    data_cache = DataCache(cache_dir)
    backtester = Backtester(data_cache)

    all_results = []

    # =========================================================================
    # 1. Test VCP with Current Parameters
    # =========================================================================
    print("\n" + "=" * 70)
    print("  Phase 1: VCP Strategy - Current Parameters")
    print("=" * 70)

    vcp_current_result = backtester.run_backtest(
        symbols=TEST_SYMBOLS,
        strategy_name="VCP_Current",
        params=VCP_PARAMS['current'],
        risk_params=RISK_PARAMS['current'],
        start_date='2023-01-01'
    )
    all_results.append(vcp_current_result)
    print_results(vcp_current_result)

    # =========================================================================
    # 2. Test Pivot with Current Parameters
    # =========================================================================
    print("\n" + "=" * 70)
    print("  Phase 2: Pivot Strategy - Current Parameters")
    print("=" * 70)

    pivot_current_result = backtester.run_backtest(
        symbols=TEST_SYMBOLS,
        strategy_name="Pivot_Current",
        params=PIVOT_PARAMS['current'],
        risk_params=RISK_PARAMS['current'],
        start_date='2023-01-01'
    )
    all_results.append(pivot_current_result)
    print_results(pivot_current_result)

    # =========================================================================
    # 3. VCP Parameter Optimization
    # =========================================================================
    print("\n" + "=" * 70)
    print("  Phase 3: VCP Parameter Optimization")
    print("=" * 70)

    vcp_sweep_results = run_parameter_sweep(
        backtester=backtester,
        strategy_name='VCP',
        base_params=VCP_PARAMS['current'],
        param_ranges=VCP_PARAMS['ranges'],
        risk_params=RISK_PARAMS['current'],
        symbols=TEST_SYMBOLS[:30]  # Use subset for faster testing
    )
    all_results.extend(vcp_sweep_results)

    if vcp_sweep_results:
        print("\nBest VCP parameter variations:")
        for i, r in enumerate(vcp_sweep_results[:5]):
            print(f"  {i+1}. {r.strategy_name}: PF={r.profit_factor:.2f}, "
                  f"WR={r.win_rate:.1f}%, Trades={r.total_trades}")

    # =========================================================================
    # 4. Pivot Parameter Optimization
    # =========================================================================
    print("\n" + "=" * 70)
    print("  Phase 4: Pivot Parameter Optimization")
    print("=" * 70)

    pivot_sweep_results = run_parameter_sweep(
        backtester=backtester,
        strategy_name='Pivot',
        base_params=PIVOT_PARAMS['current'],
        param_ranges=PIVOT_PARAMS['ranges'],
        risk_params=RISK_PARAMS['current'],
        symbols=TEST_SYMBOLS[:30]
    )
    all_results.extend(pivot_sweep_results)

    if pivot_sweep_results:
        print("\nBest Pivot parameter variations:")
        for i, r in enumerate(pivot_sweep_results[:5]):
            print(f"  {i+1}. {r.strategy_name}: PF={r.profit_factor:.2f}, "
                  f"WR={r.win_rate:.1f}%, Trades={r.total_trades}")

    # =========================================================================
    # 5. Risk Parameter Optimization (Stop/Target)
    # =========================================================================
    print("\n" + "=" * 70)
    print("  Phase 5: Risk Parameter Optimization")
    print("=" * 70)

    risk_sweep_results = run_risk_param_sweep(
        backtester=backtester,
        strategy_name='VCP',
        strategy_params=VCP_PARAMS['current'],
        risk_ranges=RISK_PARAMS['ranges'],
        symbols=TEST_SYMBOLS[:30]
    )
    all_results.extend(risk_sweep_results)

    if risk_sweep_results:
        print("\nBest Stop/Target combinations for VCP:")
        for i, r in enumerate(risk_sweep_results[:5]):
            stop = r.params.get('stop_loss_pct', 7)
            target = r.params.get('target_pct', 20)
            print(f"  {i+1}. Stop={stop}%, Target={target}%: PF={r.profit_factor:.2f}, "
                  f"WR={r.win_rate:.1f}%, R:R={r.avg_rr_realized:.2f}")

    # =========================================================================
    # Save Results
    # =========================================================================
    output_file = os.path.join(os.path.dirname(__file__), 'backtest_results.json')
    save_results(all_results, output_file)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("  SUMMARY & RECOMMENDATIONS")
    print("=" * 70)

    print("\nVCP Strategy (Current Parameters):")
    print(f"  Win Rate: {vcp_current_result.win_rate:.1f}%")
    print(f"  Profit Factor: {vcp_current_result.profit_factor:.2f}")
    print(f"  Avg R:R: {vcp_current_result.avg_rr_realized:.2f}")

    print("\nPivot Strategy (Current Parameters):")
    print(f"  Win Rate: {pivot_current_result.win_rate:.1f}%")
    print(f"  Profit Factor: {pivot_current_result.profit_factor:.2f}")
    print(f"  Avg R:R: {pivot_current_result.avg_rr_realized:.2f}")

    # Recommendations based on results
    print("\n" + "-" * 70)
    print("RECOMMENDATIONS:")
    print("-" * 70)

    if vcp_current_result.win_rate < 40:
        print("  - VCP: Consider RELAXING rs_rating_min (try 85 instead of 90)")
        print("    to increase trade count while maintaining quality")

    if vcp_current_result.win_rate > 60 and vcp_current_result.total_trades < 20:
        print("  - VCP: Parameters may be TOO STRICT. Consider relaxing:")
        print("    - contraction_threshold: 0.25 (from 0.20)")
        print("    - distance_from_52w_high_max: 0.30 (from 0.25)")

    if pivot_current_result.profit_factor < 1.5:
        print("  - Pivot: Consider TIGHTENING max_base_depth to 0.30")
        print("    to improve signal quality")

    if risk_sweep_results and risk_sweep_results[0].avg_rr_realized > 2.0:
        best = risk_sweep_results[0]
        print(f"  - RISK: Optimal Stop/Target: {best.params.get('stop_loss_pct')}%/"
              f"{best.params.get('target_pct')}%")

    print("\n" + "=" * 70)
    print("  Backtest Complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
