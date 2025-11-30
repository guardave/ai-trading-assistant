#!/usr/bin/env python3
"""
Regenerate sample charts for entry comparison backtest results.
Fixes timezone issues in the original chart generation.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Optional
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

from vcp_detector_v5 import VCPDetectorV5

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataCache:
    """Cache for historical price data with timezone handling."""

    def __init__(self, cache_dir: str = 'cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.memory_cache = {}

    def get_data(self, symbol: str, period: str = '5y') -> Optional[pd.DataFrame]:
        cache_key = f"{symbol}_{period}"

        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]

        cache_file = os.path.join(self.cache_dir, f"{cache_key}.parquet")
        if os.path.exists(cache_file):
            try:
                file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
                if datetime.now() - file_time < timedelta(days=1):
                    df = pd.read_parquet(cache_file)
                    # Ensure timezone is removed
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    self.memory_cache[cache_key] = df
                    return df
            except Exception:
                pass

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)

            if df.empty:
                return None

            df.columns = [c.title() for c in df.columns]

            # Remove timezone
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            df.to_parquet(cache_file)
            self.memory_cache[cache_key] = df
            return df

        except Exception as e:
            logger.warning(f"Error fetching {symbol}: {e}")
            return None


def generate_trade_chart(trade: dict, cache: DataCache, detector: VCPDetectorV5,
                         output_dir: str, chart_num: int, config_name: str):
    """Generate a single trade chart with proper timezone handling."""
    symbol = trade['symbol']
    entry_date = trade['entry_date']
    exit_date = trade['exit_date']
    pnl_pct = trade['pnl_pct']

    df = cache.get_data(symbol, period='5y')
    if df is None:
        logger.warning(f"No data for {symbol}")
        return False

    # Ensure timezone is removed (double check)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Parse dates - ensure they are timezone-naive
    entry_dt = pd.Timestamp(entry_date).tz_localize(None) if pd.Timestamp(entry_date).tz else pd.Timestamp(entry_date)
    exit_dt = pd.Timestamp(exit_date).tz_localize(None) if exit_date and pd.Timestamp(exit_date).tz else pd.Timestamp(exit_date) if exit_date else entry_dt

    # Extended lookback for full MA display
    display_start = entry_dt - timedelta(days=220)
    display_end = exit_dt + timedelta(days=5)

    # Use loc with date strings to avoid timezone issues
    df_display = df.loc[(df.index >= display_start) & (df.index <= display_end)].copy()

    if len(df_display) < 20:
        logger.warning(f"Not enough data for {symbol}")
        return False

    # Get pattern for visualization
    df_to_entry = df[df.index <= entry_dt]
    pattern = detector.analyze_pattern_v5(df_to_entry, lookback_days=120)

    # Create figure
    fig, (ax_price, ax_volume) = plt.subplots(2, 1, figsize=(18, 11),
                                               gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    dates = df_display.index.tolist()
    date_to_pos = {date: i for i, date in enumerate(dates)}

    is_winner = pnl_pct > 0
    outcome_str = "WINNER" if is_winner else "LOSER"
    outcome_color = '#4CAF50' if is_winner else '#f44336'

    entry_type = trade['entry_type'].upper().replace('_', ' ')

    fig.suptitle(
        f'{symbol} - {entry_type} Entry ({config_name}) - {outcome_str}\n'
        f'P&L: {pnl_pct:+.1f}% | Score: {trade["overall_score"]:.0f} | '
        f'Risk: {trade["risk_pct"]:.1f}% | Position: {trade["position_in_base"]}',
        fontsize=14, fontweight='bold', color=outcome_color
    )

    # Plot candlesticks
    for i, (date, row) in enumerate(df_display.iterrows()):
        if date not in date_to_pos:
            continue
        pos = date_to_pos[date]
        color = '#26a69a' if row['Close'] >= row['Open'] else '#ef5350'
        ax_price.plot([pos, pos], [row['Low'], row['High']], color=color, linewidth=0.8)
        body_bottom = min(row['Open'], row['Close'])
        body_height = abs(row['Close'] - row['Open'])
        rect = Rectangle((pos - 0.35, body_bottom), 0.7, body_height,
                         facecolor=color, edgecolor=color, linewidth=0.5)
        ax_price.add_patch(rect)

    # Plot MAs using full dataframe
    df_display['MA50'] = df['Close'].rolling(50).mean().loc[df_display.index]
    df_display['MA150'] = df['Close'].rolling(150).mean().loc[df_display.index]
    df_display['MA200'] = df['Close'].rolling(200).mean().loc[df_display.index]

    ma_positions = list(range(len(df_display)))
    ax_price.plot(ma_positions, df_display['MA50'].values, color='blue', linewidth=1, alpha=0.7, label='50 MA')
    ax_price.plot(ma_positions, df_display['MA150'].values, color='orange', linewidth=1, alpha=0.7, label='150 MA')
    ax_price.plot(ma_positions, df_display['MA200'].values, color='red', linewidth=1, alpha=0.7, label='200 MA')

    # Plot VCP contractions if pattern found
    contraction_colors = ['#9C27B0', '#E91E63', '#FF5722', '#795548']
    if pattern and pattern.contractions:
        for c_idx, contraction in enumerate(pattern.contractions):
            color = contraction_colors[c_idx % len(contraction_colors)]
            high_date = contraction.swing_high.date
            low_date = contraction.swing_low.date

            # Ensure dates are timezone-naive for comparison
            if hasattr(high_date, 'tz') and high_date.tz:
                high_date = high_date.tz_localize(None)
            if hasattr(low_date, 'tz') and low_date.tz:
                low_date = low_date.tz_localize(None)

            if high_date in date_to_pos and low_date in date_to_pos:
                high_pos = date_to_pos[high_date]
                low_pos = date_to_pos[low_date]

                ax_price.plot([high_pos, low_pos],
                            [contraction.swing_high.price, contraction.swing_low.price],
                            color=color, linewidth=2.5, linestyle='-', marker='v', markersize=8, zorder=8)

                ax_price.scatter([high_pos], [contraction.swing_high.price],
                               marker='^', s=100, color=color, zorder=9)
                ax_price.scatter([low_pos], [contraction.swing_low.price],
                               marker='v', s=100, color=color, zorder=9)

                mid_pos = (high_pos + low_pos) / 2
                mid_price = (contraction.swing_high.price + contraction.swing_low.price) / 2
                ax_price.annotate(f'C{c_idx+1}\n{contraction.range_pct:.1f}%',
                                xy=(mid_pos, mid_price), fontsize=9, fontweight='bold',
                                color='white', bbox=dict(boxstyle='round,pad=0.3',
                                facecolor=color, alpha=0.9), ha='center', va='center', zorder=10)

    # Entry/exit markers
    entry_price = trade['entry_price']
    exit_price = trade['exit_price']
    stop_loss = trade['stop_loss']
    target_price = trade['target_price']

    entry_pos = None
    exit_pos = None
    for date in dates:
        date_str = date.strftime('%Y-%m-%d')
        if date_str == entry_date:
            entry_pos = date_to_pos[date]
        if exit_date and date_str == exit_date:
            exit_pos = date_to_pos[date]

    if entry_pos is not None:
        ax_price.scatter([entry_pos], [entry_price], marker='^', s=200,
                        color='#2196F3', edgecolor='black', linewidth=1.5, zorder=15,
                        label=f'Entry ${entry_price:.2f}')
        ax_price.axhline(y=entry_price, color='#2196F3', linestyle='--', alpha=0.5, linewidth=1)

    if exit_pos is not None and exit_price:
        exit_marker = 'o' if is_winner else 'X'
        exit_color = '#4CAF50' if is_winner else '#f44336'
        ax_price.scatter([exit_pos], [exit_price], marker=exit_marker, s=200,
                        color=exit_color, edgecolor='black', linewidth=1.5, zorder=15,
                        label=f'Exit ${exit_price:.2f}')

    ax_price.axhline(y=stop_loss, color='#f44336', linestyle=':', alpha=0.7, linewidth=1.5,
                    label=f'Stop ${stop_loss:.2f}')
    ax_price.axhline(y=target_price, color='#4CAF50', linestyle=':', alpha=0.5, linewidth=1,
                    label=f'Target ${target_price:.2f}')

    if entry_pos is not None and exit_pos is not None:
        shade_color = '#4CAF50' if is_winner else '#f44336'
        ax_price.axvspan(entry_pos, exit_pos, alpha=0.1, color=shade_color)

    # Volume
    vol_colors = ['#26a69a' if df_display.iloc[i]['Close'] >= df_display.iloc[i]['Open']
                  else '#ef5350' for i in range(len(df_display))]
    ax_volume.bar(range(len(df_display)), df_display['Volume'].values / 1e6,
                  color=vol_colors, alpha=0.7, width=0.8)
    vol_ma = df_display['Volume'].rolling(50).mean() / 1e6
    ax_volume.plot(range(len(df_display)), vol_ma.values, color='blue', linewidth=1.5, label='50-day avg')

    ax_volume.set_ylabel('Volume (M)', fontsize=10)
    ax_volume.legend(loc='upper right', fontsize=8)

    tick_positions = list(range(0, len(dates), max(1, len(dates) // 10)))
    tick_labels = [dates[i].strftime('%Y-%m-%d') for i in tick_positions]
    ax_volume.set_xticks(tick_positions)
    ax_volume.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)

    ax_price.set_ylabel('Price ($)', fontsize=10)
    ax_price.legend(loc='lower left', fontsize=8, ncol=2)
    ax_price.grid(True, alpha=0.3)
    ax_volume.grid(True, alpha=0.3)

    # Rules box
    if pattern:
        rules_lines = [f"ENTRY: {entry_type}", "=" * 22]
        rules_lines.append(f"Prior Uptrend: {pattern.prior_uptrend.advance_pct:.1f}%")
        rules_lines.append(f"Score: {pattern.overall_score:.0f}/100")
        rules_lines.append(f"Contractions: {pattern.num_contractions}")
        rules_lines.append(f"Final: {pattern.final_contraction_pct:.1f}%")
        rules_lines.append(f"Risk: {trade['risk_pct']:.1f}%")
        rules_lines.append(f"R:R Target: 3:1")

        rules_text = "\n".join(rules_lines)
        ax_price.text(0.01, 0.99, rules_text, transform=ax_price.transAxes,
                      fontsize=8, verticalalignment='top', horizontalalignment='left',
                      fontfamily='monospace',
                      bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95, edgecolor='gray'))

    # Info box
    info_text = f"Entry: {entry_date}\nExit: {exit_date}\nDays: {trade['days_held']}\nReason: {trade['exit_reason']}"
    ax_price.text(0.99, 0.99, info_text, transform=ax_price.transAxes,
                  fontsize=9, verticalalignment='top', horizontalalignment='right',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    plt.tight_layout()

    outcome = "winner" if is_winner else "loser"
    filename = f"{chart_num:02d}_{symbol}_{trade['entry_type']}_{outcome}_{entry_date}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    logger.info(f"Generated: {filename}")
    return True


def regenerate_charts_for_config(config_name: str, results_dir: str = '../results/entry_comparison'):
    """Regenerate sample charts for a specific configuration."""
    trades_file = os.path.join(results_dir, f'{config_name}_trades.csv')

    if not os.path.exists(trades_file):
        logger.error(f"Trades file not found: {trades_file}")
        return

    trades_df = pd.read_csv(trades_file)

    if trades_df.empty:
        logger.warning(f"No trades for {config_name}")
        return

    # Select winners and losers
    winners = trades_df[trades_df['pnl_pct'] > 0].nlargest(min(5, len(trades_df[trades_df['pnl_pct'] > 0])), 'pnl_pct')
    losers = trades_df[trades_df['pnl_pct'] < 0].nsmallest(min(5, len(trades_df[trades_df['pnl_pct'] < 0])), 'pnl_pct')

    samples = pd.concat([winners, losers])

    chart_dir = os.path.join(results_dir, f'{config_name}_charts')
    os.makedirs(chart_dir, exist_ok=True)

    cache = DataCache()
    detector = VCPDetectorV5()

    success_count = 0
    for idx, (_, trade) in enumerate(samples.iterrows()):
        trade_dict = trade.to_dict()
        if generate_trade_chart(trade_dict, cache, detector, chart_dir, idx + 1, config_name):
            success_count += 1

    logger.info(f"Generated {success_count}/{len(samples)} charts for {config_name}")


def main():
    """Regenerate all charts."""
    print("=" * 70)
    print("Regenerating Entry Comparison Sample Charts")
    print("=" * 70)

    configs = [
        'pivot_only',
        'cheat_only',
        'low_cheat_only',
        'handle_only',
        'all_entries',
        'cheat_and_low_cheat'
    ]

    for config_name in configs:
        print(f"\n--- {config_name} ---")
        regenerate_charts_for_config(config_name)

    print("\n" + "=" * 70)
    print("Chart regeneration complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
