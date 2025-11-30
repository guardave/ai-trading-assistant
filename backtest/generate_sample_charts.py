#!/usr/bin/env python3
"""
Generate Sample VCP Pattern Charts for Report

Creates 10 sample charts showing detected VCP patterns from actual backtest trades:
- 5 winning trades (best performers)
- 5 losing trades (stopped out)

Charts illustrate the patterns detected by both V2 and V3 detectors.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path
import yfinance as yf
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / 'results' / 'comprehensive_analysis' / 'sample_patterns'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Data cache directory
CACHE_DIR = Path(__file__).parent.parent / 'cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_stock_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch stock data with caching."""
    cache_file = CACHE_DIR / f"{symbol}_{start_date}_{end_date}.pkl"

    if cache_file.exists():
        return pd.read_pickle(cache_file)

    # Add buffer days before and after for context
    start = pd.to_datetime(start_date) - timedelta(days=90)
    end = pd.to_datetime(end_date) + timedelta(days=30)

    df = yf.download(symbol, start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.to_pickle(cache_file)
    return df


def create_trade_chart(
    symbol: str,
    df: pd.DataFrame,
    entry_date: str,
    entry_price: float,
    exit_date: str,
    exit_price: float,
    exit_reason: str,
    pnl_pct: float,
    stop_loss: float,
    rs_rating: float,
    num_contractions: int,
    proximity_score: float,
    detector: str,
    max_gain_pct: float,
    save_path: Path
) -> None:
    """
    Create a trade chart showing:
    - 60 days before entry
    - Entry point marked
    - Exit point marked
    - Stop loss line
    - Trade outcome (profit/loss)
    """
    # Find entry and exit dates in data
    entry_dt = pd.to_datetime(entry_date)
    exit_dt = pd.to_datetime(exit_date)

    # Get 60 days before entry plus trade duration
    start_idx = df.index.get_indexer([entry_dt], method='nearest')[0] - 60
    if start_idx < 0:
        start_idx = 0

    end_idx = df.index.get_indexer([exit_dt], method='nearest')[0] + 5
    if end_idx >= len(df):
        end_idx = len(df) - 1

    df_plot = df.iloc[start_idx:end_idx].copy()

    if len(df_plot) < 10:
        logging.warning(f"Not enough data for {symbol}")
        return

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(14, 10),
                             gridspec_kw={'height_ratios': [3, 1]})

    ax_price = axes[0]
    ax_volume = axes[1]

    # Plot candlesticks
    for i, (idx, row) in enumerate(df_plot.iterrows()):
        color = 'green' if row['Close'] >= row['Open'] else 'red'

        # Wick
        ax_price.plot([i, i], [row['Low'], row['High']], color='black', linewidth=0.5)

        # Body
        body_bottom = min(row['Open'], row['Close'])
        body_height = abs(row['Close'] - row['Open'])
        if body_height < 0.01:
            body_height = 0.01

        rect = mpatches.Rectangle(
            (i - 0.4, body_bottom), 0.8, body_height,
            facecolor=color, edgecolor='black', linewidth=0.5
        )
        ax_price.add_patch(rect)

    # Find positions for entry and exit
    date_to_pos = {date: i for i, date in enumerate(df_plot.index)}

    entry_pos = None
    exit_pos = None

    for date in df_plot.index:
        if abs((date - entry_dt).days) <= 1:
            entry_pos = date_to_pos[date]
            break

    for date in df_plot.index:
        if abs((date - exit_dt).days) <= 1:
            exit_pos = date_to_pos[date]
            break

    # Draw entry marker
    if entry_pos is not None:
        ax_price.scatter(entry_pos, entry_price, marker='^', color='blue',
                        s=200, zorder=10, edgecolor='black', linewidth=2,
                        label=f'Entry: ${entry_price:.2f}')
        ax_price.axvline(x=entry_pos, color='blue', linestyle='--', alpha=0.5, linewidth=1)

    # Draw exit marker
    if exit_pos is not None:
        exit_color = 'green' if pnl_pct > 0 else 'red'
        marker = 'v' if pnl_pct > 0 else 'x'
        ax_price.scatter(exit_pos, exit_price, marker=marker, color=exit_color,
                        s=200, zorder=10, edgecolor='black', linewidth=2,
                        label=f'Exit: ${exit_price:.2f} ({exit_reason})')
        ax_price.axvline(x=exit_pos, color=exit_color, linestyle='--', alpha=0.5, linewidth=1)

    # Draw horizontal stop loss line
    ax_price.axhline(y=stop_loss, color='red', linestyle=':', linewidth=1.5,
                    alpha=0.7, label=f'Stop Loss: ${stop_loss:.2f}')

    # Shade the trade period
    if entry_pos is not None and exit_pos is not None:
        shade_color = 'lightgreen' if pnl_pct > 0 else 'lightcoral'
        ax_price.axvspan(entry_pos, exit_pos, alpha=0.2, color=shade_color)

    # Set axis properties
    x_ticks = list(range(0, len(df_plot), max(1, len(df_plot) // 8)))
    x_labels = [df_plot.index[i].strftime('%Y-%m-%d') for i in x_ticks]
    ax_price.set_xticks(x_ticks)
    ax_price.set_xticklabels(x_labels, rotation=45, ha='right')
    ax_price.set_xlim(-1, len(df_plot))
    ax_price.set_ylabel('Price ($)', fontsize=11)

    y_min = df_plot['Low'].min() * 0.97
    y_max = df_plot['High'].max() * 1.03
    ax_price.set_ylim(y_min, y_max)

    ax_price.legend(loc='upper left', fontsize=10)
    ax_price.grid(True, alpha=0.3)

    # Volume bars
    colors_vol = ['green' if df_plot.iloc[i]['Close'] >= df_plot.iloc[i]['Open']
                  else 'red' for i in range(len(df_plot))]
    ax_volume.bar(range(len(df_plot)), df_plot['Volume'] / 1e6, color=colors_vol, alpha=0.7)
    ax_volume.set_xlim(-1, len(df_plot))
    ax_volume.set_ylabel('Volume (M)', fontsize=11)
    ax_volume.set_xticks(x_ticks)
    ax_volume.set_xticklabels([])
    ax_volume.grid(True, alpha=0.3)

    # Add 50-day volume MA
    df_plot['Vol_MA50'] = df_plot['Volume'].rolling(50, min_periods=1).mean()
    ax_volume.plot(range(len(df_plot)), df_plot['Vol_MA50'] / 1e6,
                  color='blue', linewidth=1.5, label='50-day avg')
    ax_volume.legend(loc='upper right', fontsize=9)

    # Title with trade info
    outcome = "WINNER" if pnl_pct > 0 else "LOSER"
    outcome_color = 'green' if pnl_pct > 0 else 'red'

    title = (f'{symbol} - VCP Trade ({detector.upper()} Detector) - {outcome}\n'
             f'P&L: {pnl_pct:+.1f}% | Max Gain: {max_gain_pct:.1f}% | '
             f'RS: {rs_rating:.0f} | Contractions: {num_contractions} | '
             f'Proximity: {proximity_score:.0f}')
    ax_price.set_title(title, fontsize=12, fontweight='bold', color=outcome_color)

    # Add trade details text box
    details = (f'Entry: {entry_date[:10]}\n'
               f'Exit: {exit_date[:10]} ({exit_reason})\n'
               f'Duration: {(exit_dt - entry_dt).days} days')
    ax_price.text(0.98, 0.98, details, transform=ax_price.transAxes,
                 fontsize=9, verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    logging.info(f"Saved: {save_path.name}")


def generate_sample_charts():
    """Generate 10 sample pattern charts from backtest results."""

    # Read trade results
    results_dir = Path(__file__).parent.parent / 'results' / 'comprehensive_analysis'

    # Get V2 trades (best performing detector)
    v2_trades = pd.read_csv(results_dir / 'V2_RS70_Trailing_trades.csv')

    # Get V3 trades for comparison
    v3_trades = pd.read_csv(results_dir / 'V3_RS70_Trailing_trades.csv')

    # Select sample trades:
    # - 3 best V2 winners
    # - 2 V2 losers (stopped out)
    # - 3 best V3 winners
    # - 2 V3 losers

    samples = []

    # V2 winners (top 3 by P&L)
    v2_winners = v2_trades[v2_trades['pnl_pct'] > 0].nlargest(3, 'pnl_pct')
    for _, row in v2_winners.iterrows():
        samples.append({**row.to_dict(), 'category': 'V2 Winner'})

    # V2 losers (2 stopped out)
    v2_losers = v2_trades[v2_trades['exit_reason'] == 'stop'].sample(n=min(2, len(v2_trades[v2_trades['exit_reason'] == 'stop'])), random_state=42)
    for _, row in v2_losers.iterrows():
        samples.append({**row.to_dict(), 'category': 'V2 Loser'})

    # V3 winners (top 3 by P&L)
    v3_winners = v3_trades[v3_trades['pnl_pct'] > 0].nlargest(3, 'pnl_pct')
    for _, row in v3_winners.iterrows():
        samples.append({**row.to_dict(), 'category': 'V3 Winner'})

    # V3 losers (2 stopped out)
    v3_losers = v3_trades[v3_trades['exit_reason'] == 'stop'].sample(n=min(2, len(v3_trades[v3_trades['exit_reason'] == 'stop'])), random_state=42)
    for _, row in v3_losers.iterrows():
        samples.append({**row.to_dict(), 'category': 'V3 Loser'})

    logging.info(f"Selected {len(samples)} sample trades")

    # Generate charts
    for i, trade in enumerate(samples, 1):
        symbol = trade['symbol']
        entry_date = trade['entry_date']
        exit_date = trade['exit_date']

        logging.info(f"Processing {i}/{len(samples)}: {symbol} ({trade['category']})")

        try:
            # Get stock data
            df = get_stock_data(symbol, entry_date[:10], exit_date[:10])

            if df is None or len(df) < 20:
                logging.warning(f"Insufficient data for {symbol}")
                continue

            # Create filename
            outcome = 'winner' if trade['pnl_pct'] > 0 else 'loser'
            detector = trade['detector']
            filename = f"{i:02d}_{symbol}_{detector}_{outcome}_{trade['entry_date'][:10]}.png"
            save_path = OUTPUT_DIR / filename

            create_trade_chart(
                symbol=symbol,
                df=df,
                entry_date=trade['entry_date'],
                entry_price=trade['entry_price'],
                exit_date=trade['exit_date'],
                exit_price=trade['exit_price'],
                exit_reason=trade['exit_reason'],
                pnl_pct=trade['pnl_pct'],
                stop_loss=trade['stop_loss'],
                rs_rating=trade['rs_rating'],
                num_contractions=trade['num_contractions'],
                proximity_score=trade['proximity_score'],
                detector=trade['detector'],
                max_gain_pct=trade['max_gain_pct'],
                save_path=save_path
            )

        except Exception as e:
            logging.error(f"Error processing {symbol}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    logging.info(f"\n{'='*60}")
    logging.info("Sample Charts Generated")
    logging.info('='*60)
    logging.info(f"Output directory: {OUTPUT_DIR}")

    # List generated files
    charts = list(OUTPUT_DIR.glob('*.png'))
    logging.info(f"Charts generated: {len(charts)}")
    for chart in sorted(charts):
        logging.info(f"  - {chart.name}")

    return charts


if __name__ == '__main__':
    generate_sample_charts()
