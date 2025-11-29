#!/usr/bin/env python3
"""
Trade Visualization Tool
Generates candlestick charts with:
- Entry/exit markers
- VCP pattern annotations
- RS rating subplot
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle, FancyBboxPatch
import mplfinance as mpf
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import os
warnings.filterwarnings('ignore')

# Output directory
OUTPUT_DIR = 'charts/trades'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def calculate_rs_rating(df: pd.DataFrame, benchmark_df: pd.DataFrame) -> pd.Series:
    """Calculate rolling RS rating relative to benchmark (SPY)."""
    # Calculate returns for different periods
    periods = [63, 126, 189, 252]  # ~3, 6, 9, 12 months
    weights = [0.4, 0.2, 0.2, 0.2]

    rs_scores = pd.Series(index=df.index, dtype=float)

    for i in range(len(df)):
        if i < 252:
            rs_scores.iloc[i] = np.nan
            continue

        stock_returns = []
        bench_returns = []

        for period in periods:
            if i >= period:
                stock_ret = (df['Close'].iloc[i] / df['Close'].iloc[i-period] - 1) * 100
                bench_ret = (benchmark_df['Close'].iloc[i] / benchmark_df['Close'].iloc[i-period] - 1) * 100
                stock_returns.append(stock_ret)
                bench_returns.append(bench_ret)
            else:
                stock_returns.append(0)
                bench_returns.append(0)

        # Weighted average
        stock_score = sum(r * w for r, w in zip(stock_returns, weights))
        bench_score = sum(r * w for r, w in zip(bench_returns, weights))

        # Convert to 0-100 scale (simplified)
        relative_strength = stock_score - bench_score
        rs_rating = 50 + relative_strength  # Center around 50
        rs_rating = max(0, min(100, rs_rating))  # Clamp to 0-100
        rs_scores.iloc[i] = rs_rating

    return rs_scores


def identify_vcp_contractions(df: pd.DataFrame, lookback: int = 60) -> list:
    """Identify VCP contractions in price data."""
    contractions = []

    for i in range(lookback, len(df) - 10):
        # Look for local high-low ranges
        window = df.iloc[i-lookback:i]

        # Find swing highs and lows
        high = window['High'].max()
        low = window['Low'].min()
        range_pct = (high - low) / low * 100

        # Check for contraction (range < 20%)
        if range_pct < 20:
            recent_window = df.iloc[i-20:i]
            recent_high = recent_window['High'].max()
            recent_low = recent_window['Low'].min()
            recent_range = (recent_high - recent_low) / recent_low * 100

            if recent_range < range_pct * 0.8:  # Contracting
                contractions.append({
                    'date': df.index[i],
                    'high': recent_high,
                    'low': recent_low,
                    'range_pct': recent_range
                })

    return contractions


def plot_trade(trade: dict, output_path: str = None):
    """
    Plot a single trade on candlestick chart with annotations.

    Args:
        trade: Dictionary with trade details
        output_path: Path to save the chart
    """
    symbol = trade['symbol']
    entry_date = pd.to_datetime(trade['entry_date'])
    exit_date = pd.to_datetime(trade['exit_date'])
    entry_price = trade['entry_price']
    exit_price = trade['exit_price']
    pnl_pct = trade['pnl_pct']
    proximity_score = trade.get('proximity_score', 0)
    rs_rating = trade.get('rs_rating', 0)
    exit_reason = trade.get('exit_reason', 'unknown')

    # Fetch data with buffer before and after trade
    start_date = entry_date - timedelta(days=90)
    end_date = exit_date + timedelta(days=30)

    print(f"  Fetching data for {symbol}...")
    df = yf.download(symbol, start=start_date, end=end_date, progress=False)
    spy = yf.download('SPY', start=start_date, end=end_date, progress=False)

    if df.empty:
        print(f"  No data for {symbol}")
        return

    # Handle MultiIndex columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)

    # Align indices
    common_idx = df.index.intersection(spy.index)
    df = df.loc[common_idx]
    spy = spy.loc[common_idx]

    # Calculate RS rating
    rs_series = calculate_rs_rating(df, spy)

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))

    # Candlestick chart (top, larger)
    ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=2)

    # Volume (middle)
    ax2 = plt.subplot2grid((4, 1), (2, 0), rowspan=1, sharex=ax1)

    # RS Rating (bottom)
    ax3 = plt.subplot2grid((4, 1), (3, 0), rowspan=1, sharex=ax1)

    # Plot candlesticks manually
    width = 0.6
    width2 = 0.1

    up = df[df['Close'] >= df['Open']]
    down = df[df['Close'] < df['Open']]

    # Up candles
    ax1.bar(up.index, up['Close'] - up['Open'], width, bottom=up['Open'], color='green', edgecolor='green')
    ax1.bar(up.index, up['High'] - up['Close'], width2, bottom=up['Close'], color='green')
    ax1.bar(up.index, up['Low'] - up['Open'], width2, bottom=up['Open'], color='green')

    # Down candles
    ax1.bar(down.index, down['Close'] - down['Open'], width, bottom=down['Open'], color='red', edgecolor='red')
    ax1.bar(down.index, down['High'] - down['Open'], width2, bottom=down['Open'], color='red')
    ax1.bar(down.index, down['Low'] - down['Close'], width2, bottom=down['Close'], color='red')

    # Mark entry point
    ax1.scatter([entry_date], [entry_price], marker='^', color='blue', s=200, zorder=5, label='Entry')
    ax1.annotate(f'ENTRY\n${entry_price:.2f}',
                xy=(entry_date, entry_price),
                xytext=(entry_date, entry_price * 1.05),
                fontsize=10, fontweight='bold', color='blue',
                ha='center',
                arrowprops=dict(arrowstyle='->', color='blue'))

    # Mark exit point
    exit_color = 'green' if pnl_pct > 0 else 'red'
    ax1.scatter([exit_date], [exit_price], marker='v', color=exit_color, s=200, zorder=5, label='Exit')
    ax1.annotate(f'EXIT ({exit_reason})\n${exit_price:.2f}\n{pnl_pct:+.1f}%',
                xy=(exit_date, exit_price),
                xytext=(exit_date, exit_price * 0.95),
                fontsize=10, fontweight='bold', color=exit_color,
                ha='center',
                arrowprops=dict(arrowstyle='->', color=exit_color))

    # Draw trade period highlight
    ax1.axvspan(entry_date, exit_date, alpha=0.1, color='blue', label='Trade Period')

    # Add stop loss line (7% below entry)
    stop_price = entry_price * 0.93
    ax1.axhline(y=stop_price, color='red', linestyle='--', alpha=0.5, label=f'Stop Loss (${stop_price:.2f})')

    # Add 20% target line
    target_price = entry_price * 1.20
    ax1.axhline(y=target_price, color='green', linestyle='--', alpha=0.5, label=f'Target (${target_price:.2f})')

    # Title with trade info
    result_text = "WIN" if pnl_pct > 0 else "LOSS"
    ax1.set_title(f'{symbol} - VCP Trade ({result_text}: {pnl_pct:+.1f}%)\n'
                  f'RS: {rs_rating:.0f} | Proximity: {proximity_score:.0f} | '
                  f'Held: {trade.get("days_held", "N/A")} days',
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price ($)')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Volume chart
    colors = ['green' if df['Close'].iloc[i] >= df['Open'].iloc[i] else 'red'
              for i in range(len(df))]
    ax2.bar(df.index, df['Volume'], color=colors, alpha=0.7)
    ax2.set_ylabel('Volume')
    ax2.axvspan(entry_date, exit_date, alpha=0.1, color='blue')
    ax2.grid(True, alpha=0.3)

    # RS Rating chart
    ax3.plot(df.index, rs_series, color='purple', linewidth=2, label='RS Rating')
    ax3.axhline(y=70, color='green', linestyle='--', alpha=0.5, label='RS 70 (New Threshold)')
    ax3.axhline(y=90, color='orange', linestyle='--', alpha=0.5, label='RS 90 (Old Threshold)')
    ax3.axvspan(entry_date, exit_date, alpha=0.1, color='blue')

    # Mark RS at entry
    entry_idx = df.index.get_indexer([entry_date], method='nearest')[0]
    if entry_idx >= 0 and entry_idx < len(rs_series):
        rs_at_entry = rs_series.iloc[entry_idx]
        ax3.scatter([entry_date], [rs_at_entry], marker='o', color='blue', s=100, zorder=5)
        ax3.annotate(f'RS: {rs_at_entry:.0f}',
                    xy=(entry_date, rs_at_entry),
                    xytext=(entry_date + timedelta(days=5), rs_at_entry + 5),
                    fontsize=9, color='purple')

    ax3.set_ylabel('RS Rating')
    ax3.set_xlabel('Date')
    ax3.set_ylim(40, 100)
    ax3.legend(loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Format x-axis
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax3.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Add text box with trade details
    textstr = f'Entry: {entry_date.strftime("%Y-%m-%d")}\n'
    textstr += f'Exit: {exit_date.strftime("%Y-%m-%d")}\n'
    textstr += f'P&L: {pnl_pct:+.2f}%\n'
    textstr += f'Proximity Score: {proximity_score:.0f}'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    """Generate trade visualizations for key trades."""
    print("Loading trade data...")

    # Load Phase 1 trades
    trades_df = pd.read_csv('results/phase1/phase1_all_trades.csv')

    # Filter to trailing stop config (best performer)
    trailing_trades = trades_df[trades_df['config'] == 'baseline_trailing'].copy()

    print(f"Found {len(trailing_trades)} trades in Phase 1 (trailing stop)")

    # Select interesting trades to visualize
    # 1. Best winning trade
    # 2. Worst losing trade
    # 3. High proximity winning trade
    # 4. Low proximity losing trade

    trades_to_plot = []

    # Best win
    best_win = trailing_trades.loc[trailing_trades['pnl_pct'].idxmax()]
    trades_to_plot.append(('best_win', best_win))

    # Worst loss
    worst_loss = trailing_trades.loc[trailing_trades['pnl_pct'].idxmin()]
    trades_to_plot.append(('worst_loss', worst_loss))

    # High proximity example (>= 70)
    high_prox = trailing_trades[trailing_trades['proximity_score'] >= 70]
    if len(high_prox) > 0:
        high_prox_trade = high_prox.iloc[0]
        trades_to_plot.append(('high_proximity', high_prox_trade))

    # Low proximity example (<= 40)
    low_prox = trailing_trades[trailing_trades['proximity_score'] <= 40]
    if len(low_prox) > 0:
        low_prox_trade = low_prox.iloc[0]
        trades_to_plot.append(('low_proximity', low_prox_trade))

    print(f"\nGenerating {len(trades_to_plot)} trade charts...")

    for name, trade_row in trades_to_plot:
        trade = {
            'symbol': trade_row['symbol'],
            'entry_date': trade_row['entry_date'],
            'exit_date': trade_row['exit_date'],
            'entry_price': trade_row['entry_price'],
            'exit_price': trade_row['exit_price'],
            'pnl_pct': trade_row['pnl_pct'],
            'proximity_score': trade_row['proximity_score'],
            'rs_rating': trade_row['rs_rating'],
            'exit_reason': trade_row['exit_reason'],
            'days_held': trade_row['days_held']
        }

        print(f"\nPlotting {name}: {trade['symbol']} ({trade['pnl_pct']:+.1f}%)")
        output_path = f"{OUTPUT_DIR}/{name}_{trade['symbol']}.png"

        try:
            plot_trade(trade, output_path)
        except Exception as e:
            print(f"  Error: {e}")

    print(f"\nAll trade charts saved to {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
