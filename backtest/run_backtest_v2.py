#!/usr/bin/env python3
"""
Run Strategy Backtests v2
Enhanced with:
- Trailing stop comparison
- VCP proximity analysis
- R:R optimization focus (target R:R ≥ 3.0)
"""
import os
import sys
import json
from datetime import datetime
from typing import List, Dict, Any
import itertools

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest_framework_v2 import DataCache, Backtester, BacktestResult, print_results
from strategy_params import VCP_PARAMS, PIVOT_PARAMS, RISK_PARAMS

# =============================================================================
# Test Universe - 60 US Stocks across sectors
# =============================================================================
TEST_SYMBOLS = [
    # Tech Leaders (20)
    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'AMZN', 'TSLA', 'AMD', 'AVGO', 'CRM',
    'NFLX', 'ADBE', 'NOW', 'PANW', 'SNOW', 'DDOG', 'NET', 'CRWD', 'ZS', 'MDB',
    # Healthcare (10)
    'LLY', 'UNH', 'JNJ', 'ABBV', 'MRK', 'PFE', 'ISRG', 'VRTX', 'REGN', 'GILD',
    # Industrials (10)
    'GE', 'CAT', 'BA', 'UNP', 'HON', 'RTX', 'DE', 'LMT', 'MMM', 'FDX',
    # Financials (10)
    'JPM', 'V', 'MA', 'BAC', 'GS', 'MS', 'BLK', 'AXP', 'C', 'WFC',
    # Consumer (10)
    'HD', 'LOW', 'NKE', 'SBUX', 'MCD', 'DIS', 'COST', 'WMT', 'TGT', 'LULU',
]


def save_results_json(results: List[BacktestResult], filename: str):
    """Save backtest results to JSON"""
    output = []
    for r in results:
        result_dict = {
            'strategy': r.strategy_name,
            'exit_method': r.exit_method,
            'params': r.params,
            'total_trades': r.total_trades,
            'winning_trades': r.winning_trades,
            'losing_trades': r.losing_trades,
            'win_rate': round(r.win_rate, 2),
            'avg_win_pct': round(r.avg_win_pct, 2),
            'avg_loss_pct': round(r.avg_loss_pct, 2),
            'profit_factor': round(r.profit_factor, 2),
            'avg_rr_realized': round(r.avg_rr_realized, 2),
            'avg_rr_theoretical': round(r.avg_rr_theoretical, 2),
            'total_return_pct': round(r.total_return_pct, 2),
            'avg_days_held': round(r.avg_days_held, 1),
        }

        # Add proximity analysis if available
        if r.proximity_correlation is not None:
            result_dict['proximity_correlation'] = round(r.proximity_correlation, 3)
        if r.high_proximity_win_rate is not None:
            result_dict['high_proximity_win_rate'] = round(r.high_proximity_win_rate, 1)
        if r.low_proximity_win_rate is not None:
            result_dict['low_proximity_win_rate'] = round(r.low_proximity_win_rate, 1)

        output.append(result_dict)

    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {filename}")


def save_trades_csv(trades: List, filename: str):
    """Save individual trades to CSV for analysis"""
    import csv

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Symbol', 'Pattern', 'Entry Date', 'Entry Price', 'Exit Date', 'Exit Price',
            'Exit Reason', 'P&L %', 'Days Held', 'RS Rating', 'Volume Ratio',
            'Num Contractions', 'Proximity Score', 'Theoretical R:R', 'Realized R:R'
        ])

        for t in trades:
            writer.writerow([
                t.symbol, t.pattern, t.entry_date, f"{t.entry_price:.2f}",
                t.exit_date or '', f"{t.exit_price:.2f}" if t.exit_price else '',
                t.exit_reason or '', f"{t.pnl_pct:.2f}" if t.pnl_pct else '',
                t.days_held or '', f"{t.rs_rating:.1f}" if t.rs_rating else '',
                f"{t.volume_ratio:.2f}" if t.volume_ratio else '',
                t.num_contractions or '', f"{t.proximity_score:.1f}" if t.proximity_score else '',
                f"{t.theoretical_rr:.2f}" if t.theoretical_rr else '',
                f"{t.realized_rr:.2f}" if t.realized_rr else ''
            ])

    print(f"Trades saved to {filename}")


def test_rr_combinations(backtester: Backtester, strategy_name: str,
                         strategy_params: Dict[str, Any],
                         symbols: List[str],
                         min_rr: float = 3.0) -> List[BacktestResult]:
    """
    Test stop/target combinations that achieve R:R ≥ min_rr
    Focus on 7-8% stops as per user preference
    """
    results = []

    # Stop/Target combinations with R:R ≥ 3.0
    # R:R = Target / Stop
    combinations = [
        # 7% stop combinations
        (7, 21),   # R:R = 3.0
        (7, 24),   # R:R = 3.43
        (7, 28),   # R:R = 4.0

        # 8% stop combinations
        (8, 24),   # R:R = 3.0
        (8, 28),   # R:R = 3.5
        (8, 32),   # R:R = 4.0

        # More aggressive (smaller stop)
        (6, 18),   # R:R = 3.0
        (6, 21),   # R:R = 3.5
        (6, 24),   # R:R = 4.0
    ]

    for stop, target in combinations:
        rr = target / stop
        if rr < min_rr:
            continue

        risk_params = {
            'default_stop_loss_pct': stop,
            'default_target_pct': target,
        }

        print(f"\n  Testing {strategy_name} with Stop={stop}%, Target={target}% (R:R={rr:.2f})...")

        # Test with fixed target
        result_fixed = backtester.run_backtest(
            symbols=symbols,
            strategy_name=f"{strategy_name}_S{stop}_T{target}",
            params=strategy_params,
            risk_params=risk_params,
            exit_method='fixed_target',
            start_date='2023-01-01'
        )
        results.append(result_fixed)

        print(f"    Fixed Target: Trades={result_fixed.total_trades}, "
              f"WR={result_fixed.win_rate:.1f}%, R:R={result_fixed.avg_rr_realized:.2f}, "
              f"PF={result_fixed.profit_factor:.2f}")

    return results


def test_trailing_stop(backtester: Backtester, strategy_name: str,
                       strategy_params: Dict[str, Any],
                       symbols: List[str]) -> List[BacktestResult]:
    """
    Test trailing stop configurations
    """
    results = []

    # Trailing stop configurations
    configs = [
        # (stop_loss, activation, trail_distance)
        (7, 8, 5),    # Standard: 7% initial stop, activate at +8%, trail by 5%
        (7, 10, 5),   # Later activation
        (7, 8, 6),    # Wider trail
        (8, 8, 5),    # 8% initial stop
        (8, 10, 6),   # More conservative trailing
        (7, 12, 7),   # Very late activation, wide trail (let winners run)
    ]

    for stop, activation, trail in configs:
        risk_params = {
            'default_stop_loss_pct': stop,
            'trailing_stop_activation_pct': activation,
            'trailing_stop_distance_pct': trail,
        }

        print(f"\n  Testing Trailing Stop: Initial={stop}%, Activate=+{activation}%, Trail={trail}%...")

        result = backtester.run_backtest(
            symbols=symbols,
            strategy_name=f"{strategy_name}_Trail_S{stop}_A{activation}_T{trail}",
            params=strategy_params,
            risk_params=risk_params,
            exit_method='trailing_stop',
            start_date='2023-01-01'
        )
        results.append(result)

        print(f"    Results: Trades={result.total_trades}, WR={result.win_rate:.1f}%, "
              f"R:R={result.avg_rr_realized:.2f}, PF={result.profit_factor:.2f}, "
              f"Avg Days={result.avg_days_held:.1f}")

    return results


def test_proximity_effect(backtester: Backtester, symbols: List[str]) -> Dict[str, BacktestResult]:
    """
    Test the effect of VCP contraction proximity on outcomes.
    Compare high proximity vs low proximity setups.
    """
    results = {}

    # Base VCP params
    base_params = VCP_PARAMS['current'].copy()

    # Test different minimum proximity thresholds
    proximity_thresholds = [0, 30, 50, 70]

    for min_prox in proximity_thresholds:
        test_params = base_params.copy()
        test_params['min_proximity_score'] = min_prox

        print(f"\n  Testing VCP with min_proximity_score={min_prox}...")

        result = backtester.run_backtest(
            symbols=symbols,
            strategy_name=f"VCP_Proximity_{min_prox}",
            params=test_params,
            risk_params=RISK_PARAMS['current'],
            exit_method='fixed_target',
            start_date='2023-01-01'
        )

        results[f"min_proximity_{min_prox}"] = result

        print(f"    Trades={result.total_trades}, WR={result.win_rate:.1f}%, "
              f"R:R={result.avg_rr_realized:.2f}, PF={result.profit_factor:.2f}")

    return results


def test_rs_relaxation(backtester: Backtester, strategy_name: str,
                       base_params: Dict[str, Any],
                       symbols: List[str]) -> List[BacktestResult]:
    """
    Test relaxing RS requirements to increase trade frequency while maintaining quality
    """
    results = []

    rs_thresholds = [70, 75, 80, 85, 90, 95]

    for rs_min in rs_thresholds:
        test_params = base_params.copy()
        test_params['rs_rating_min'] = rs_min

        print(f"\n  Testing {strategy_name} with rs_rating_min={rs_min}...")

        result = backtester.run_backtest(
            symbols=symbols,
            strategy_name=f"{strategy_name}_RS{rs_min}",
            params=test_params,
            risk_params=RISK_PARAMS['current'],
            exit_method='fixed_target',
            start_date='2023-01-01'
        )
        results.append(result)

        print(f"    Trades={result.total_trades}, WR={result.win_rate:.1f}%, "
              f"R:R={result.avg_rr_realized:.2f}, PF={result.profit_factor:.2f}")

    return results


def main():
    print("=" * 80)
    print("  AI Trading Assistant - Strategy Backtesting v2")
    print("  Focus: Higher frequency with R:R ≥ 3.0")
    print("=" * 80)
    print(f"\nTest Universe: {len(TEST_SYMBOLS)} US stocks")
    print(f"Test Period: 2023-01-01 to present")
    print(f"Target: More frequent trades with R:R ≥ 3.0")

    # Initialize
    cache_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'cache')
    data_cache = DataCache(cache_dir)
    backtester = Backtester(data_cache)

    all_results = []

    # =========================================================================
    # Phase 1: Baseline VCP with Current Parameters
    # =========================================================================
    print("\n" + "=" * 80)
    print("  PHASE 1: VCP Baseline (Current Parameters)")
    print("=" * 80)

    vcp_baseline = backtester.run_backtest(
        symbols=TEST_SYMBOLS,
        strategy_name="VCP_Baseline",
        params=VCP_PARAMS['current'],
        risk_params=RISK_PARAMS['current'],
        exit_method='fixed_target',
        start_date='2023-01-01'
    )
    all_results.append(vcp_baseline)
    print_results(vcp_baseline)

    # =========================================================================
    # Phase 2: Baseline Pivot with Current Parameters
    # =========================================================================
    print("\n" + "=" * 80)
    print("  PHASE 2: Pivot Baseline (Current Parameters)")
    print("=" * 80)

    pivot_baseline = backtester.run_backtest(
        symbols=TEST_SYMBOLS,
        strategy_name="Pivot_Baseline",
        params=PIVOT_PARAMS['current'],
        risk_params=RISK_PARAMS['current'],
        exit_method='fixed_target',
        start_date='2023-01-01'
    )
    all_results.append(pivot_baseline)
    print_results(pivot_baseline)

    # =========================================================================
    # Phase 3: VCP Contraction Proximity Analysis
    # =========================================================================
    print("\n" + "=" * 80)
    print("  PHASE 3: VCP Contraction Proximity Analysis")
    print("=" * 80)
    print("  Testing whether tighter/closer contractions improve outcomes...")

    proximity_results = test_proximity_effect(backtester, TEST_SYMBOLS)
    all_results.extend(proximity_results.values())

    print("\n  Proximity Analysis Summary:")
    print("-" * 60)
    for name, result in proximity_results.items():
        print(f"  {name}: Trades={result.total_trades}, WR={result.win_rate:.1f}%, "
              f"R:R={result.avg_rr_realized:.2f}")

        if result.proximity_correlation is not None:
            print(f"    -> Correlation(Proximity, Win): {result.proximity_correlation:.3f}")
        if result.high_proximity_win_rate is not None:
            print(f"    -> High Proximity (≥70) WR: {result.high_proximity_win_rate:.1f}%")
        if result.low_proximity_win_rate is not None:
            print(f"    -> Low Proximity (≤30) WR: {result.low_proximity_win_rate:.1f}%")

    # =========================================================================
    # Phase 4: RS Rating Relaxation (for higher frequency)
    # =========================================================================
    print("\n" + "=" * 80)
    print("  PHASE 4: RS Rating Relaxation (Increase Frequency)")
    print("=" * 80)

    rs_vcp_results = test_rs_relaxation(backtester, "VCP", VCP_PARAMS['current'], TEST_SYMBOLS[:40])
    all_results.extend(rs_vcp_results)

    print("\n  RS Relaxation Summary (VCP):")
    print("-" * 60)
    for r in rs_vcp_results:
        rs = r.params.get('rs_rating_min', 90)
        print(f"  RS≥{rs}: Trades={r.total_trades}, WR={r.win_rate:.1f}%, "
              f"R:R={r.avg_rr_realized:.2f}, PF={r.profit_factor:.2f}")

    # =========================================================================
    # Phase 5: Stop/Target Optimization for R:R ≥ 3.0
    # =========================================================================
    print("\n" + "=" * 80)
    print("  PHASE 5: Stop/Target Optimization (R:R ≥ 3.0)")
    print("=" * 80)

    rr_results = test_rr_combinations(backtester, "VCP", VCP_PARAMS['current'], TEST_SYMBOLS[:40])
    all_results.extend(rr_results)

    print("\n  R:R Optimization Summary:")
    print("-" * 60)
    # Sort by profit factor
    rr_results_sorted = sorted(rr_results, key=lambda x: x.profit_factor, reverse=True)
    for r in rr_results_sorted[:5]:
        stop = r.params.get('default_stop_loss_pct', 7)
        target = r.params.get('default_target_pct', 20)
        print(f"  Stop={stop}% Target={target}%: Trades={r.total_trades}, "
              f"WR={r.win_rate:.1f}%, R:R={r.avg_rr_realized:.2f}, PF={r.profit_factor:.2f}")

    # =========================================================================
    # Phase 6: Trailing Stop vs Fixed Target Comparison
    # =========================================================================
    print("\n" + "=" * 80)
    print("  PHASE 6: Trailing Stop vs Fixed Target")
    print("=" * 80)

    trailing_results = test_trailing_stop(backtester, "VCP", VCP_PARAMS['current'], TEST_SYMBOLS[:40])
    all_results.extend(trailing_results)

    print("\n  Trailing Stop Summary:")
    print("-" * 60)
    trailing_sorted = sorted(trailing_results, key=lambda x: x.profit_factor, reverse=True)
    for r in trailing_sorted[:5]:
        print(f"  {r.strategy_name}: Trades={r.total_trades}, WR={r.win_rate:.1f}%, "
              f"R:R={r.avg_rr_realized:.2f}, PF={r.profit_factor:.2f}, Days={r.avg_days_held:.0f}")

    # =========================================================================
    # Save Results
    # =========================================================================
    output_dir = os.path.dirname(__file__)
    save_results_json(all_results, os.path.join(output_dir, 'backtest_results_v2.json'))

    # Save detailed trades for best strategies
    if vcp_baseline.trades:
        save_trades_csv(vcp_baseline.trades, os.path.join(output_dir, 'vcp_baseline_trades.csv'))

    # =========================================================================
    # Final Summary and Recommendations
    # =========================================================================
    print("\n" + "=" * 80)
    print("  FINAL SUMMARY & RECOMMENDATIONS")
    print("=" * 80)

    print("\n  1. BASELINE PERFORMANCE:")
    print(f"     VCP: {vcp_baseline.total_trades} trades, {vcp_baseline.win_rate:.1f}% WR, "
          f"{vcp_baseline.avg_rr_realized:.2f} R:R, {vcp_baseline.profit_factor:.2f} PF")
    print(f"     Pivot: {pivot_baseline.total_trades} trades, {pivot_baseline.win_rate:.1f}% WR, "
          f"{pivot_baseline.avg_rr_realized:.2f} R:R, {pivot_baseline.profit_factor:.2f} PF")

    print("\n  2. PROXIMITY EFFECT:")
    if proximity_results:
        base_result = proximity_results.get('min_proximity_0')
        high_prox_result = proximity_results.get('min_proximity_70')

        if base_result and high_prox_result:
            wr_diff = (high_prox_result.win_rate - base_result.win_rate) if base_result.total_trades > 0 else 0
            print(f"     Requiring high proximity (≥70) vs no filter:")
            print(f"     - Trade count: {base_result.total_trades} → {high_prox_result.total_trades}")
            print(f"     - Win rate change: {wr_diff:+.1f}%")

            if base_result.high_proximity_win_rate and base_result.low_proximity_win_rate:
                print(f"     - High proximity trades win rate: {base_result.high_proximity_win_rate:.1f}%")
                print(f"     - Low proximity trades win rate: {base_result.low_proximity_win_rate:.1f}%")

    print("\n  3. BEST FIXED TARGET CONFIGURATION:")
    if rr_results_sorted:
        best = rr_results_sorted[0]
        print(f"     Stop={best.params.get('default_stop_loss_pct')}%, "
              f"Target={best.params.get('default_target_pct')}%")
        print(f"     {best.total_trades} trades, {best.win_rate:.1f}% WR, "
              f"{best.avg_rr_realized:.2f} R:R, {best.profit_factor:.2f} PF")

    print("\n  4. BEST TRAILING STOP CONFIGURATION:")
    if trailing_sorted:
        best_trail = trailing_sorted[0]
        print(f"     {best_trail.strategy_name}")
        print(f"     {best_trail.total_trades} trades, {best_trail.win_rate:.1f}% WR, "
              f"{best_trail.avg_rr_realized:.2f} R:R, {best_trail.profit_factor:.2f} PF")
        print(f"     Avg holding period: {best_trail.avg_days_held:.0f} days")

    print("\n  5. RS RELAXATION IMPACT:")
    if rs_vcp_results:
        rs90 = next((r for r in rs_vcp_results if r.params.get('rs_rating_min') == 90), None)
        rs85 = next((r for r in rs_vcp_results if r.params.get('rs_rating_min') == 85), None)
        rs80 = next((r for r in rs_vcp_results if r.params.get('rs_rating_min') == 80), None)

        if rs90 and rs85:
            trade_increase = ((rs85.total_trades - rs90.total_trades) / rs90.total_trades * 100) if rs90.total_trades > 0 else 0
            wr_change = rs85.win_rate - rs90.win_rate
            print(f"     RS 90 → 85: {trade_increase:+.0f}% trades, {wr_change:+.1f}% win rate")

        if rs90 and rs80:
            trade_increase = ((rs80.total_trades - rs90.total_trades) / rs90.total_trades * 100) if rs90.total_trades > 0 else 0
            wr_change = rs80.win_rate - rs90.win_rate
            print(f"     RS 90 → 80: {trade_increase:+.0f}% trades, {wr_change:+.1f}% win rate")

    print("\n" + "=" * 80)
    print("  Backtest Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
