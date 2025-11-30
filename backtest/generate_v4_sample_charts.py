#!/usr/bin/env python3
"""
Generate sample charts for V4 VCP trades with pattern visualization.

Shows:
- Candlestick chart with entry/exit markers
- VCP contractions with trendlines and percentages
- Trend template indicators (MAs)
- Prior uptrend region
- Trade outcome (win/loss)
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')

# Import V4 detector
from vcp_detector_v4 import VCPDetectorV4, VCPPattern


def get_stock_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch stock data for the given period."""
    # Extend start date to get enough history for pattern detection
    start_dt = pd.Timestamp(start_date) - timedelta(days=180)

    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_dt.strftime('%Y-%m-%d'), end=end_date)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Remove timezone info to avoid comparison issues
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    return df


def plot_candlesticks(ax, df: pd.DataFrame, date_to_pos: dict):
    """Plot candlestick chart."""
    for i, (date, row) in enumerate(df.iterrows()):
        if date not in date_to_pos:
            continue
        pos = date_to_pos[date]

        open_price = row['Open']
        close_price = row['Close']
        high_price = row['High']
        low_price = row['Low']

        color = '#26a69a' if close_price >= open_price else '#ef5350'

        # Draw wick
        ax.plot([pos, pos], [low_price, high_price], color=color, linewidth=0.8)

        # Draw body
        body_bottom = min(open_price, close_price)
        body_height = abs(close_price - open_price)
        rect = Rectangle((pos - 0.35, body_bottom), 0.7, body_height,
                         facecolor=color, edgecolor=color, linewidth=0.5)
        ax.add_patch(rect)


def generate_v4_chart(
    symbol: str,
    entry_date: str,
    exit_date: str,
    entry_price: float,
    exit_price: float,
    stop_loss: float,
    pnl_pct: float,
    overall_score: float,
    prior_advance_pct: float,
    num_contractions: int,
    output_path: str
):
    """Generate a detailed chart for a V4 VCP trade."""

    # Get data with extended history (250 days for full 200 MA display)
    end_dt = pd.Timestamp(exit_date) + timedelta(days=10)
    start_dt = pd.Timestamp(entry_date) - timedelta(days=280)

    df = get_stock_data(symbol, start_dt.strftime('%Y-%m-%d'), end_dt.strftime('%Y-%m-%d'))

    if df.empty:
        print(f"No data for {symbol}")
        return False

    # Get data up to entry for pattern detection
    entry_dt = pd.Timestamp(entry_date)
    df_to_entry = df[df.index <= entry_dt]

    if len(df_to_entry) < 120:
        print(f"Insufficient data for {symbol}")
        return False

    # Detect VCP pattern
    detector = VCPDetectorV4()
    pattern = detector.analyze_pattern(df_to_entry, lookback_days=120)

    # Define display range (220 days before entry for full MA display, to exit + 5 days)
    display_start = entry_dt - timedelta(days=220)
    display_end = pd.Timestamp(exit_date) + timedelta(days=5)
    df_display = df[(df.index >= display_start) & (df.index <= display_end)]

    if len(df_display) < 20:
        print(f"Display range too small for {symbol}")
        return False

    # Create figure (wider for more data)
    fig, (ax_price, ax_volume) = plt.subplots(
        2, 1, figsize=(18, 11),
        gridspec_kw={'height_ratios': [3, 1]},
        sharex=True
    )

    # Create position mapping for x-axis
    dates = df_display.index.tolist()
    date_to_pos = {date: i for i, date in enumerate(dates)}

    # Determine outcome
    is_winner = pnl_pct > 0
    outcome_str = "WINNER" if is_winner else "LOSER"
    outcome_color = '#4CAF50' if is_winner else '#f44336'

    # Title
    fig.suptitle(
        f'{symbol} - VCP Trade (V4 Detector) - {outcome_str}\n'
        f'P&L: {pnl_pct:+.1f}% | Score: {overall_score:.0f} | '
        f'Prior Advance: {prior_advance_pct:.1f}% | Contractions: {num_contractions}',
        fontsize=14, fontweight='bold', color=outcome_color
    )

    # Plot candlesticks
    plot_candlesticks(ax_price, df_display, date_to_pos)

    # Plot moving averages
    df_display = df_display.copy()
    df_display['MA50'] = df['Close'].rolling(50).mean()
    df_display['MA150'] = df['Close'].rolling(150).mean()
    df_display['MA200'] = df['Close'].rolling(200).mean()

    ma_positions = [date_to_pos[d] for d in df_display.index if d in date_to_pos]

    ax_price.plot(ma_positions, df_display['MA50'].values,
                  color='blue', linewidth=1, alpha=0.7, label='50 MA')
    ax_price.plot(ma_positions, df_display['MA150'].values,
                  color='orange', linewidth=1, alpha=0.7, label='150 MA')
    ax_price.plot(ma_positions, df_display['MA200'].values,
                  color='red', linewidth=1, alpha=0.7, label='200 MA')

    # Plot VCP contractions if pattern found
    contraction_colors = ['#9C27B0', '#E91E63', '#FF5722', '#795548']

    if pattern and pattern.contractions:
        for c_idx, contraction in enumerate(pattern.contractions):
            color = contraction_colors[c_idx % len(contraction_colors)]

            high_date = contraction.swing_high.date
            low_date = contraction.swing_low.date

            if high_date in date_to_pos and low_date in date_to_pos:
                high_pos = date_to_pos[high_date]
                low_pos = date_to_pos[low_date]

                # Draw trendline from high to low
                ax_price.plot(
                    [high_pos, low_pos],
                    [contraction.swing_high.price, contraction.swing_low.price],
                    color=color, linewidth=2.5, linestyle='-',
                    marker='v', markersize=8, zorder=8
                )

                # Add triangle markers
                ax_price.scatter([high_pos], [contraction.swing_high.price],
                               marker='^', s=100, color=color, zorder=9)
                ax_price.scatter([low_pos], [contraction.swing_low.price],
                               marker='v', s=100, color=color, zorder=9)

                # Add contraction label
                mid_pos = (high_pos + low_pos) / 2
                mid_price = (contraction.swing_high.price + contraction.swing_low.price) / 2

                ax_price.annotate(
                    f'C{c_idx+1}\n{contraction.range_pct:.1f}%',
                    xy=(mid_pos, mid_price),
                    fontsize=9, fontweight='bold',
                    color='white',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.9),
                    ha='center', va='center',
                    zorder=10
                )

    # Find entry and exit positions
    entry_pos = None
    exit_pos = None

    for date in dates:
        if date.strftime('%Y-%m-%d') == entry_date:
            entry_pos = date_to_pos[date]
        if date.strftime('%Y-%m-%d') == exit_date:
            exit_pos = date_to_pos[date]

    # If exact dates not found, find closest
    if entry_pos is None:
        for date in dates:
            if date >= pd.Timestamp(entry_date):
                entry_pos = date_to_pos[date]
                break

    if exit_pos is None:
        for date in dates:
            if date >= pd.Timestamp(exit_date):
                exit_pos = date_to_pos[date]
                break

    # Plot entry marker
    if entry_pos is not None:
        ax_price.scatter([entry_pos], [entry_price], marker='^', s=200,
                        color='#2196F3', edgecolor='black', linewidth=1.5,
                        zorder=15, label=f'Entry ${entry_price:.2f}')
        ax_price.axhline(y=entry_price, color='#2196F3', linestyle='--',
                        alpha=0.5, linewidth=1)

    # Plot exit marker
    if exit_pos is not None:
        exit_marker = 'o' if is_winner else 'X'
        exit_color = '#4CAF50' if is_winner else '#f44336'
        ax_price.scatter([exit_pos], [exit_price], marker=exit_marker, s=200,
                        color=exit_color, edgecolor='black', linewidth=1.5,
                        zorder=15, label=f'Exit ${exit_price:.2f}')

    # Plot stop loss line
    ax_price.axhline(y=stop_loss, color='#f44336', linestyle=':',
                    alpha=0.7, linewidth=1.5, label=f'Stop ${stop_loss:.2f}')

    # Shade trade period
    if entry_pos is not None and exit_pos is not None:
        shade_color = '#4CAF50' if is_winner else '#f44336'
        ax_price.axvspan(entry_pos, exit_pos, alpha=0.1, color=shade_color)

    # Add pivot line if pattern exists
    if pattern:
        ax_price.axhline(y=pattern.pivot_price, color='purple', linestyle='-.',
                        alpha=0.5, linewidth=1, label=f'Pivot ${pattern.pivot_price:.2f}')

    # Plot volume
    vol_colors = ['#26a69a' if df_display.iloc[i]['Close'] >= df_display.iloc[i]['Open']
                  else '#ef5350' for i in range(len(df_display))]

    vol_positions = list(range(len(df_display)))
    ax_volume.bar(vol_positions, df_display['Volume'].values / 1e6,
                  color=vol_colors, alpha=0.7, width=0.8)

    # Volume MA
    vol_ma = df_display['Volume'].rolling(50).mean() / 1e6
    ax_volume.plot(vol_positions, vol_ma.values, color='blue',
                   linewidth=1.5, label='50-day avg')

    ax_volume.set_ylabel('Volume (M)', fontsize=10)
    ax_volume.legend(loc='upper right', fontsize=8)

    # Format x-axis with dates
    tick_positions = list(range(0, len(dates), max(1, len(dates) // 10)))
    tick_labels = [dates[i].strftime('%Y-%m-%d') for i in tick_positions]
    ax_volume.set_xticks(tick_positions)
    ax_volume.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)

    # Price axis formatting
    ax_price.set_ylabel('Price ($)', fontsize=10)
    ax_price.legend(loc='lower left', fontsize=8, ncol=2)
    ax_price.grid(True, alpha=0.3)
    ax_volume.grid(True, alpha=0.3)

    # Add trade info box (top right)
    info_text = (
        f"Entry: {entry_date}\n"
        f"Exit: {exit_date}\n"
        f"Days Held: {(pd.Timestamp(exit_date) - pd.Timestamp(entry_date)).days}"
    )

    ax_price.text(0.99, 0.99, info_text, transform=ax_price.transAxes,
                  fontsize=9, verticalalignment='top', horizontalalignment='right',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    # Add V4 Rules Validation Box (left side)
    if pattern:
        tt = pattern.trend_template
        pu = pattern.prior_uptrend

        # Build rules checklist
        rules_lines = ["V4 RULES VALIDATION", "=" * 22]

        # Prior Uptrend
        pu_status = "PASS" if pu.is_valid else "FAIL"
        rules_lines.append(f"Prior Uptrend (≥30%): {pu_status}")
        rules_lines.append(f"  Advance: {pu.advance_pct:.1f}%")

        rules_lines.append("")
        rules_lines.append("Trend Template:")

        # Trend template checks
        for check_name, (status, msg) in tt.checks.items():
            symbol_char = "✓" if status.value == "PASS" else ("!" if status.value == "WARN" else "✗")
            short_name = check_name.replace("_", " ").title()[:18]
            rules_lines.append(f"  {symbol_char} {short_name}")

        rules_lines.append("")
        rules_lines.append("VCP Pattern:")
        rules_lines.append(f"  Contractions: {pattern.num_contractions}")
        rules_lines.append(f"  First: {pattern.first_contraction_pct:.1f}%")
        rules_lines.append(f"  Final: {pattern.final_contraction_pct:.1f}%")
        rules_lines.append(f"  Base Duration: {pattern.base_duration_days}d")
        rules_lines.append(f"  Tightening: {pattern.avg_tightening_ratio:.2f}x")

        rules_lines.append("")
        rules_lines.append(f"Overall Score: {pattern.overall_score:.0f}/100")
        rules_lines.append(f"Pattern Valid: {'YES' if pattern.is_valid else 'NO'}")

        rules_text = "\n".join(rules_lines)

        ax_price.text(0.01, 0.99, rules_text, transform=ax_price.transAxes,
                      fontsize=8, verticalalignment='top', horizontalalignment='left',
                      fontfamily='monospace',
                      bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95, edgecolor='gray'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    return True


def main():
    """Generate sample charts from V4 backtest results."""

    # Load trades from CSV
    trades_file = 'results/v4/trades.csv'
    if not os.path.exists(trades_file):
        print(f"Trades file not found: {trades_file}")
        return

    trades_df = pd.read_csv(trades_file)
    print(f"Loaded {len(trades_df)} trades from V4 backtest")

    # Create output directory
    output_dir = '../results/v4/sample_charts'
    os.makedirs(output_dir, exist_ok=True)

    # Select sample trades - mix of winners and losers
    winners = trades_df[trades_df['pnl_pct'] > 0].nlargest(5, 'pnl_pct')
    losers = trades_df[trades_df['pnl_pct'] < 0].nsmallest(5, 'pnl_pct')

    samples = pd.concat([winners, losers])

    print(f"\nGenerating {len(samples)} sample charts...")
    print("="*60)

    for idx, (_, trade) in enumerate(samples.iterrows()):
        symbol = trade['symbol']
        entry_date = trade['entry_date']
        exit_date = trade['exit_date']
        pnl_pct = trade['pnl_pct']

        outcome = "winner" if pnl_pct > 0 else "loser"
        filename = f"{idx+1:02d}_{symbol}_v4_{outcome}_{entry_date}.png"
        output_path = os.path.join(output_dir, filename)

        print(f"Generating: {filename}")

        success = generate_v4_chart(
            symbol=symbol,
            entry_date=entry_date,
            exit_date=exit_date if pd.notna(exit_date) else entry_date,
            entry_price=trade['entry_price'],
            exit_price=trade['exit_price'] if pd.notna(trade['exit_price']) else trade['entry_price'],
            stop_loss=trade['stop_loss'],
            pnl_pct=pnl_pct,
            overall_score=trade['overall_score'],
            prior_advance_pct=trade['prior_advance_pct'],
            num_contractions=trade['num_contractions'],
            output_path=output_path
        )

        if success:
            print(f"  -> {symbol}: {pnl_pct:+.1f}% ({outcome})")
        else:
            print(f"  -> Failed to generate chart for {symbol}")

    print(f"\n{'='*60}")
    print(f"Charts saved to: {output_dir}")

    # List generated files
    files = sorted(os.listdir(output_dir))
    print(f"\nGenerated {len(files)} charts:")
    for f in files:
        print(f"  - {f}")


if __name__ == '__main__':
    main()
