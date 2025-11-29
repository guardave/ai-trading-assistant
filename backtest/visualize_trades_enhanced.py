#!/usr/bin/env python3
"""
Enhanced Trade Visualization Tool
Generates candlestick charts with:
- Entry/exit markers
- VCP contraction annotations (boxes showing each contraction)
- Proper RS rating calculation with trend
- All trades from backtest results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle, FancyBboxPatch
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import os
import json
warnings.filterwarnings('ignore')

# Output directory
OUTPUT_DIR = 'charts/trades'
os.makedirs(OUTPUT_DIR, exist_ok=True)


class RSCalculator:
    """Calculate RS Rating using IBD-style methodology."""

    def __init__(self, benchmark_symbol: str = 'SPY'):
        self.benchmark_symbol = benchmark_symbol
        self.benchmark_data = None

    def load_benchmark(self, start_date: str, end_date: str):
        """Load benchmark data."""
        self.benchmark_data = yf.download(
            self.benchmark_symbol,
            start=start_date,
            end=end_date,
            progress=False
        )
        if isinstance(self.benchmark_data.columns, pd.MultiIndex):
            self.benchmark_data.columns = self.benchmark_data.columns.get_level_values(0)

    def calculate(self, stock_df: pd.DataFrame) -> pd.Series:
        """
        Calculate RS Rating for each day.

        Uses weighted performance over 4 periods:
        - 63 days (3 months): 40% weight
        - 126 days (6 months): 20% weight
        - 189 days (9 months): 20% weight
        - 252 days (12 months): 20% weight
        """
        if self.benchmark_data is None:
            return pd.Series(index=stock_df.index, dtype=float)

        # Align data
        common_idx = stock_df.index.intersection(self.benchmark_data.index)
        stock = stock_df.loc[common_idx, 'Close']
        bench = self.benchmark_data.loc[common_idx, 'Close']

        periods = [63, 126, 189, 252]
        weights = [0.4, 0.2, 0.2, 0.2]

        rs_ratings = pd.Series(index=common_idx, dtype=float)

        for i in range(len(common_idx)):
            if i < max(periods):
                rs_ratings.iloc[i] = np.nan
                continue

            stock_score = 0
            bench_score = 0

            for period, weight in zip(periods, weights):
                if i >= period:
                    stock_ret = (stock.iloc[i] / stock.iloc[i-period] - 1) * 100
                    bench_ret = (bench.iloc[i] / bench.iloc[i-period] - 1) * 100
                    stock_score += stock_ret * weight
                    bench_score += bench_ret * weight

            # Calculate relative strength
            relative_perf = stock_score - bench_score

            # Convert to 1-99 scale (simplified - in reality this would be percentile ranked)
            # Assume relative_perf ranges from -50 to +50 typically
            rs_rating = 50 + relative_perf
            rs_rating = max(1, min(99, rs_rating))
            rs_ratings.iloc[i] = rs_rating

        return rs_ratings


class VCPDetector:
    """Detect VCP (Volatility Contraction Pattern) in price data."""

    def __init__(self, contraction_threshold: float = 0.20, min_contractions: int = 2):
        self.contraction_threshold = contraction_threshold
        self.min_contractions = min_contractions

    def detect_contractions(self, df: pd.DataFrame, entry_date: pd.Timestamp,
                           lookback_days: int = 90) -> list:
        """
        Detect VCP contractions before the entry date.

        Returns list of contraction dictionaries with:
        - start_date, end_date
        - high, low
        - range_pct
        """
        # Get data before entry
        mask = df.index < entry_date
        pre_entry = df[mask].tail(lookback_days)

        if len(pre_entry) < 20:
            return []

        contractions = []

        # Use rolling windows to find contractions
        window_sizes = [20, 15, 10, 7]  # Decreasing windows for VCP

        for i, window in enumerate(window_sizes):
            if len(pre_entry) < window:
                continue

            # Slide through the data looking for tight ranges
            for j in range(len(pre_entry) - window):
                segment = pre_entry.iloc[j:j+window]
                high = segment['High'].max()
                low = segment['Low'].min()
                range_pct = (high - low) / low

                # Check if this is a contraction (tight range)
                if range_pct < self.contraction_threshold:
                    # Check for volume dry-up
                    avg_vol = segment['Volume'].mean()
                    prior_vol = pre_entry.iloc[max(0, j-20):j]['Volume'].mean() if j > 0 else avg_vol

                    vol_ratio = avg_vol / prior_vol if prior_vol > 0 else 1

                    contraction = {
                        'start_date': segment.index[0],
                        'end_date': segment.index[-1],
                        'high': high,
                        'low': low,
                        'range_pct': range_pct * 100,
                        'volume_ratio': vol_ratio,
                        'window': window
                    }

                    # Avoid duplicates (overlapping contractions)
                    is_duplicate = False
                    for existing in contractions:
                        if abs((existing['start_date'] - contraction['start_date']).days) < 5:
                            is_duplicate = True
                            break

                    if not is_duplicate:
                        contractions.append(contraction)

        # Sort by date and keep most relevant (last few before entry)
        contractions.sort(key=lambda x: x['start_date'])

        # Filter to get sequential contractions (VCP pattern)
        if len(contractions) >= 2:
            # Keep contractions that show decreasing range
            filtered = [contractions[0]]
            for c in contractions[1:]:
                if c['range_pct'] <= filtered[-1]['range_pct'] * 1.2:  # Allow some tolerance
                    filtered.append(c)
            contractions = filtered[-4:]  # Keep last 4 max

        return contractions


def plot_trade_enhanced(trade: dict, rs_calculator: RSCalculator,
                        vcp_detector: VCPDetector, output_path: str = None):
    """
    Plot a single trade with enhanced features.

    Features:
    - Candlestick chart with entry/exit markers
    - VCP contraction boxes
    - RS rating trend line
    - Volume with dry-up highlighting
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
    num_contractions = trade.get('num_contractions', 0)

    # Fetch data with buffer
    start_date = entry_date - timedelta(days=120)
    end_date = exit_date + timedelta(days=30)

    df = yf.download(symbol, start=start_date, end=end_date, progress=False)

    if df.empty:
        print(f"    No data for {symbol}")
        return False

    # Handle MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Calculate RS rating
    rs_series = rs_calculator.calculate(df)

    # Detect VCP contractions
    contractions = vcp_detector.detect_contractions(df, entry_date)

    # Create figure
    fig = plt.figure(figsize=(16, 14))

    # Layout: Price (top, large), Volume (middle), RS (bottom)
    ax1 = plt.subplot2grid((5, 1), (0, 0), rowspan=3)
    ax2 = plt.subplot2grid((5, 1), (3, 0), rowspan=1, sharex=ax1)
    ax3 = plt.subplot2grid((5, 1), (4, 0), rowspan=1, sharex=ax1)

    # === CANDLESTICK CHART ===
    width = 0.6
    width2 = 0.1

    up = df[df['Close'] >= df['Open']]
    down = df[df['Close'] < df['Open']]

    # Up candles
    ax1.bar(up.index, up['Close'] - up['Open'], width, bottom=up['Open'],
            color='#26a69a', edgecolor='#26a69a')
    ax1.bar(up.index, up['High'] - up['Close'], width2, bottom=up['Close'], color='#26a69a')
    ax1.bar(up.index, up['Low'] - up['Open'], width2, bottom=up['Open'], color='#26a69a')

    # Down candles
    ax1.bar(down.index, down['Close'] - down['Open'], width, bottom=down['Open'],
            color='#ef5350', edgecolor='#ef5350')
    ax1.bar(down.index, down['High'] - down['Open'], width2, bottom=down['Open'], color='#ef5350')
    ax1.bar(down.index, down['Low'] - down['Close'], width2, bottom=down['Close'], color='#ef5350')

    # === VCP CONTRACTION BOXES ===
    colors = ['#ffeb3b', '#ffc107', '#ff9800', '#ff5722']  # Yellow to orange gradient
    for i, contraction in enumerate(contractions):
        color = colors[min(i, len(colors)-1)]
        rect = Rectangle(
            (mdates.date2num(contraction['start_date']), contraction['low']),
            mdates.date2num(contraction['end_date']) - mdates.date2num(contraction['start_date']),
            contraction['high'] - contraction['low'],
            linewidth=2,
            edgecolor=color,
            facecolor=color,
            alpha=0.3,
            label=f"C{i+1}: {contraction['range_pct']:.1f}%" if i == 0 else f"C{i+1}: {contraction['range_pct']:.1f}%"
        )
        ax1.add_patch(rect)

        # Add contraction label
        mid_date = contraction['start_date'] + (contraction['end_date'] - contraction['start_date']) / 2
        ax1.text(mid_date, contraction['high'] * 1.01, f"C{i+1}\n{contraction['range_pct']:.1f}%",
                fontsize=8, ha='center', va='bottom', color='#ff6f00', fontweight='bold')

    # === ENTRY/EXIT MARKERS ===
    # Entry
    ax1.scatter([entry_date], [entry_price], marker='^', color='#2196f3', s=250,
                zorder=10, edgecolor='white', linewidth=2)
    ax1.annotate(f'ENTRY\n${entry_price:.2f}',
                xy=(entry_date, entry_price),
                xytext=(entry_date + timedelta(days=3), entry_price * 1.04),
                fontsize=10, fontweight='bold', color='#2196f3',
                ha='left', va='bottom',
                arrowprops=dict(arrowstyle='->', color='#2196f3', lw=1.5))

    # Exit
    exit_color = '#4caf50' if pnl_pct > 0 else '#f44336'
    ax1.scatter([exit_date], [exit_price], marker='v', color=exit_color, s=250,
                zorder=10, edgecolor='white', linewidth=2)
    ax1.annotate(f'EXIT ({exit_reason})\n${exit_price:.2f}\n{pnl_pct:+.1f}%',
                xy=(exit_date, exit_price),
                xytext=(exit_date + timedelta(days=3), exit_price * 0.96),
                fontsize=10, fontweight='bold', color=exit_color,
                ha='left', va='top',
                arrowprops=dict(arrowstyle='->', color=exit_color, lw=1.5))

    # Trade period highlight
    ax1.axvspan(entry_date, exit_date, alpha=0.08, color='#2196f3')

    # Stop loss and target lines
    stop_price = entry_price * 0.93
    target_price = entry_price * 1.20
    ax1.axhline(y=stop_price, color='#f44336', linestyle='--', alpha=0.6, linewidth=1.5)
    ax1.axhline(y=target_price, color='#4caf50', linestyle='--', alpha=0.6, linewidth=1.5)

    # Add labels for stop/target
    ax1.text(df.index[2], stop_price, f' Stop: ${stop_price:.2f}',
             fontsize=8, color='#f44336', va='center')
    ax1.text(df.index[2], target_price, f' Target: ${target_price:.2f}',
             fontsize=8, color='#4caf50', va='center')

    # Title
    result_text = "WIN" if pnl_pct > 0 else "LOSS"
    result_color = '#4caf50' if pnl_pct > 0 else '#f44336'
    ax1.set_title(f'{symbol} - VCP Trade ({result_text}: {pnl_pct:+.1f}%)\n'
                  f'RS: {rs_rating:.0f} | Proximity: {proximity_score:.0f} | '
                  f'Contractions: {len(contractions)} | Held: {trade.get("days_held", "N/A")} days',
                  fontsize=14, fontweight='bold', color=result_color)
    ax1.set_ylabel('Price ($)', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelbottom=False)

    # === VOLUME CHART ===
    vol_colors = ['#26a69a' if df['Close'].iloc[i] >= df['Open'].iloc[i] else '#ef5350'
                  for i in range(len(df))]
    ax2.bar(df.index, df['Volume'], color=vol_colors, alpha=0.7, width=0.8)

    # Highlight volume dry-up periods in contractions
    for contraction in contractions:
        ax2.axvspan(contraction['start_date'], contraction['end_date'],
                   alpha=0.2, color='#ff9800')

    ax2.axvspan(entry_date, exit_date, alpha=0.08, color='#2196f3')
    ax2.set_ylabel('Volume', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelbottom=False)

    # Add 50-day volume MA
    vol_ma = df['Volume'].rolling(50).mean()
    ax2.plot(df.index, vol_ma, color='#ff9800', linewidth=1.5, alpha=0.8, label='50-day Avg')
    ax2.legend(loc='upper right', fontsize=8)

    # === RS RATING CHART ===
    valid_rs = rs_series.dropna()
    if len(valid_rs) > 0:
        ax3.plot(valid_rs.index, valid_rs.values, color='#9c27b0', linewidth=2, label='RS Rating')
        ax3.fill_between(valid_rs.index, 0, valid_rs.values, alpha=0.1, color='#9c27b0')

    # RS threshold lines
    ax3.axhline(y=70, color='#4caf50', linestyle='--', alpha=0.7, linewidth=1.5, label='RS 70 (New)')
    ax3.axhline(y=90, color='#ff9800', linestyle='--', alpha=0.7, linewidth=1.5, label='RS 90 (Old)')
    ax3.axhline(y=50, color='gray', linestyle=':', alpha=0.5, linewidth=1)

    ax3.axvspan(entry_date, exit_date, alpha=0.08, color='#2196f3')

    # Mark RS at entry
    if len(valid_rs) > 0:
        try:
            entry_idx = valid_rs.index.get_indexer([entry_date], method='nearest')[0]
            if 0 <= entry_idx < len(valid_rs):
                rs_at_entry = valid_rs.iloc[entry_idx]
                ax3.scatter([entry_date], [rs_at_entry], marker='o', color='#2196f3',
                           s=100, zorder=5, edgecolor='white', linewidth=2)
                ax3.annotate(f'RS: {rs_at_entry:.0f}',
                            xy=(entry_date, rs_at_entry),
                            xytext=(entry_date + timedelta(days=5), min(95, rs_at_entry + 8)),
                            fontsize=9, color='#9c27b0', fontweight='bold',
                            arrowprops=dict(arrowstyle='->', color='#9c27b0', lw=1))
        except:
            pass

    ax3.set_ylabel('RS Rating', fontsize=11)
    ax3.set_xlabel('Date', fontsize=11)
    ax3.set_ylim(30, 100)
    ax3.legend(loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Format x-axis
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax3.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Info box
    info_text = (f"Entry: {entry_date.strftime('%Y-%m-%d')}\n"
                 f"Exit: {exit_date.strftime('%Y-%m-%d')}\n"
                 f"P&L: {pnl_pct:+.2f}%\n"
                 f"Proximity: {proximity_score:.0f}\n"
                 f"Contractions: {len(contractions)}")

    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray')
    ax1.text(0.02, 0.97, info_text, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, fontfamily='monospace')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        return True
    else:
        plt.show()
        plt.close()
        return True


def main():
    """Generate enhanced trade visualizations for all trades."""
    print("=" * 60)
    print("Enhanced Trade Visualization Generator")
    print("=" * 60)

    print("\nLoading trade data...")
    trades_df = pd.read_csv('results/phase1/phase1_all_trades.csv')

    # Filter to trailing stop config (best performer)
    trailing_trades = trades_df[trades_df['config'] == 'baseline_trailing'].copy()
    print(f"Found {len(trailing_trades)} trades to visualize")

    # Initialize calculators
    print("\nInitializing RS calculator and VCP detector...")
    rs_calculator = RSCalculator(benchmark_symbol='SPY')
    vcp_detector = VCPDetector(contraction_threshold=0.20)

    # Load benchmark data for full period
    min_date = pd.to_datetime(trailing_trades['entry_date'].min()) - timedelta(days=400)
    max_date = pd.to_datetime(trailing_trades['exit_date'].max()) + timedelta(days=60)
    rs_calculator.load_benchmark(min_date.strftime('%Y-%m-%d'), max_date.strftime('%Y-%m-%d'))
    print(f"Loaded SPY benchmark data: {min_date.date()} to {max_date.date()}")

    # Generate charts for all trades
    print(f"\nGenerating {len(trailing_trades)} trade charts...")
    print("-" * 60)

    success_count = 0
    error_count = 0

    for idx, (_, trade_row) in enumerate(trailing_trades.iterrows()):
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
            'days_held': trade_row['days_held'],
            'num_contractions': trade_row.get('num_contractions', 2)
        }

        # Create filename
        result = "win" if trade['pnl_pct'] > 0 else "loss"
        filename = f"{idx+1:02d}_{trade['symbol']}_{result}_{trade['pnl_pct']:+.1f}pct.png"
        output_path = f"{OUTPUT_DIR}/{filename}"

        print(f"  [{idx+1}/{len(trailing_trades)}] {trade['symbol']}: {trade['pnl_pct']:+.1f}%...", end=" ")

        try:
            success = plot_trade_enhanced(trade, rs_calculator, vcp_detector, output_path)
            if success:
                print(f"OK -> {filename}")
                success_count += 1
            else:
                print("SKIPPED (no data)")
                error_count += 1
        except Exception as e:
            print(f"ERROR: {e}")
            error_count += 1

    print("-" * 60)
    print(f"\nComplete! Generated {success_count} charts, {error_count} errors")
    print(f"Output directory: {OUTPUT_DIR}/")

    # Generate summary
    print("\n" + "=" * 60)
    print("TRADE SUMMARY")
    print("=" * 60)

    wins = trailing_trades[trailing_trades['pnl_pct'] > 0]
    losses = trailing_trades[trailing_trades['pnl_pct'] <= 0]

    print(f"Total Trades: {len(trailing_trades)}")
    print(f"Wins: {len(wins)} ({len(wins)/len(trailing_trades)*100:.1f}%)")
    print(f"Losses: {len(losses)} ({len(losses)/len(trailing_trades)*100:.1f}%)")
    print(f"Best Trade: {trailing_trades.loc[trailing_trades['pnl_pct'].idxmax(), 'symbol']} ({trailing_trades['pnl_pct'].max():+.1f}%)")
    print(f"Worst Trade: {trailing_trades.loc[trailing_trades['pnl_pct'].idxmin(), 'symbol']} ({trailing_trades['pnl_pct'].min():+.1f}%)")
    print(f"Avg Win: {wins['pnl_pct'].mean():+.1f}%")
    print(f"Avg Loss: {losses['pnl_pct'].mean():+.1f}%")


if __name__ == '__main__':
    main()
