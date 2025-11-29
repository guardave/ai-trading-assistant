#!/usr/bin/env python3
"""
VCP Visualization for Review

Creates charts with:
1. Candlestick price data
2. Swing highs marked with triangles
3. Swing lows marked with triangles
4. Contractions shown as trendlines from swing high to swing low
5. Contraction percentage labeled on each trendline
6. Volume bars at bottom
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import mplfinance as mpf
from pathlib import Path
import yfinance as yf
from vcp_detector import VCPDetector, VCPPattern

# Create output directory
OUTPUT_DIR = Path(__file__).parent / 'charts' / 'vcp_review'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_vcp_review_chart(
    symbol: str,
    df: pd.DataFrame,
    pattern: VCPPattern,
    swing_highs: list,
    swing_lows: list,
    save_path: Path = None
) -> None:
    """
    Create a review chart showing VCP detection with trendlines.

    Args:
        symbol: Stock symbol
        df: DataFrame with OHLCV data (last 120 days)
        pattern: VCPPattern object with detected contractions
        swing_highs: List of all swing high points
        swing_lows: List of all swing low points
        save_path: Path to save the chart
    """
    # Prepare data for mplfinance
    df_plot = df.copy()

    # Create figure with custom panels
    # Note: figsize * dpi must stay under 2000px for API compatibility
    # Using figsize=(13, 10) with dpi=150 = 1950x1500 pixels (safe)
    fig, axes = plt.subplots(3, 1, figsize=(13, 10),
                             gridspec_kw={'height_ratios': [3, 1, 1]})

    ax_price = axes[0]
    ax_volume = axes[1]
    ax_legend = axes[2]
    ax_legend.axis('off')

    # Plot candlesticks manually
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

    # Set x-axis labels
    x_ticks = list(range(0, len(df_plot), max(1, len(df_plot) // 10)))
    x_labels = [df_plot.index[i].strftime('%Y-%m-%d') for i in x_ticks]
    ax_price.set_xticks(x_ticks)
    ax_price.set_xticklabels(x_labels, rotation=45, ha='right')
    ax_price.set_xlim(-1, len(df_plot))
    ax_price.set_ylabel('Price ($)')

    # Adjust y-axis limits
    y_min = df_plot['Low'].min() * 0.98
    y_max = df_plot['High'].max() * 1.02
    ax_price.set_ylim(y_min, y_max)

    # Create index mapping (date to position)
    date_to_pos = {date: i for i, date in enumerate(df_plot.index)}

    # Plot all swing highs (small markers)
    for sh in swing_highs:
        if sh.date in date_to_pos:
            pos = date_to_pos[sh.date]
            ax_price.scatter(pos, sh.price * 1.005, marker='v', color='blue',
                           s=50, zorder=5, alpha=0.5)

    # Plot all swing lows (small markers)
    for sl in swing_lows:
        if sl.date in date_to_pos:
            pos = date_to_pos[sl.date]
            ax_price.scatter(pos, sl.price * 0.995, marker='^', color='orange',
                           s=50, zorder=5, alpha=0.5)

    # Plot contractions as trendlines with labels
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']  # Different colors for each contraction

    for i, contraction in enumerate(pattern.contractions):
        color = colors[i % len(colors)]

        # Get positions
        high_date = contraction.swing_high.date
        low_date = contraction.swing_low.date

        if high_date in date_to_pos and low_date in date_to_pos:
            high_pos = date_to_pos[high_date]
            low_pos = date_to_pos[low_date]
            high_price = contraction.swing_high.price
            low_price = contraction.swing_low.price

            # Draw trendline from swing high to swing low
            ax_price.plot([high_pos, low_pos], [high_price, low_price],
                         color=color, linewidth=2.5, linestyle='-',
                         marker='o', markersize=8, zorder=10,
                         label=f'C{i+1}: {contraction.range_pct:.1f}%')

            # Add contraction percentage label
            mid_pos = (high_pos + low_pos) / 2
            mid_price = (high_price + low_price) / 2

            # Position label to the right of the trendline
            label_offset = (y_max - y_min) * 0.02
            ax_price.annotate(
                f'C{i+1}\n{contraction.range_pct:.1f}%\n{contraction.duration_days}d',
                xy=(mid_pos, mid_price),
                xytext=(mid_pos + 3, mid_price + label_offset),
                fontsize=9,
                fontweight='bold',
                color=color,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor=color, alpha=0.9),
                arrowprops=dict(arrowstyle='->', color=color, lw=1),
                zorder=15
            )

            # Mark swing high with larger marker
            ax_price.scatter(high_pos, high_price, marker='v', color=color,
                           s=150, zorder=12, edgecolor='black', linewidth=1)

            # Mark swing low with larger marker
            ax_price.scatter(low_pos, low_price, marker='^', color=color,
                           s=150, zorder=12, edgecolor='black', linewidth=1)

    # Draw horizontal lines for pivot and support
    ax_price.axhline(y=pattern.pivot_price, color='green', linestyle='--',
                    linewidth=1.5, alpha=0.7, label=f'Pivot: ${pattern.pivot_price:.2f}')
    ax_price.axhline(y=pattern.support_price, color='red', linestyle='--',
                    linewidth=1.5, alpha=0.7, label=f'Support: ${pattern.support_price:.2f}')

    # Volume bars
    colors_vol = ['green' if df_plot.iloc[i]['Close'] >= df_plot.iloc[i]['Open']
                  else 'red' for i in range(len(df_plot))]
    ax_volume.bar(range(len(df_plot)), df_plot['Volume'] / 1e6, color=colors_vol, alpha=0.7)
    ax_volume.set_xlim(-1, len(df_plot))
    ax_volume.set_ylabel('Volume (M)')
    ax_volume.set_xticks(x_ticks)
    ax_volume.set_xticklabels([])

    # Add 50-day volume MA
    df_plot['Vol_MA50'] = df_plot['Volume'].rolling(50).mean()
    ax_volume.plot(range(len(df_plot)), df_plot['Vol_MA50'] / 1e6,
                  color='blue', linewidth=1, label='50-day avg')
    ax_volume.legend(loc='upper right', fontsize=8)

    # Create legend in bottom panel
    legend_elements = [
        Line2D([0], [0], marker='v', color='w', markerfacecolor='blue',
               markersize=10, alpha=0.5, label='All Swing Highs'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='orange',
               markersize=10, alpha=0.5, label='All Swing Lows'),
        Line2D([0], [0], color='green', linestyle='--', linewidth=1.5,
               label=f'Pivot Price: ${pattern.pivot_price:.2f}'),
        Line2D([0], [0], color='red', linestyle='--', linewidth=1.5,
               label=f'Support Price: ${pattern.support_price:.2f}'),
    ]

    # Add contraction legend items
    for i, c in enumerate(pattern.contractions):
        color = colors[i % len(colors)]
        legend_elements.append(
            Line2D([0], [0], color=color, linewidth=2.5, marker='o', markersize=6,
                   label=f'Contraction {i+1}: {c.range_pct:.1f}% over {c.duration_days} days, Vol: {c.avg_volume_ratio:.2f}x')
        )

    ax_legend.legend(handles=legend_elements, loc='center', ncol=2, fontsize=10,
                    frameon=True, fancybox=True)

    # Title with pattern info
    validity = "VALID" if pattern.is_valid else "INVALID"
    title = (f'{symbol} - VCP Pattern Analysis ({validity})\n'
             f'Contractions: {len(pattern.contractions)} | '
             f'Contraction Quality: {pattern.contraction_quality:.0f} | '
             f'Volume Quality: {pattern.volume_quality:.0f} | '
             f'Overall Score: {pattern.proximity_score:.0f}')
    ax_price.set_title(title, fontsize=12, fontweight='bold')

    # Add validity reasons as text box
    reasons_text = '\n'.join([f'• {r}' for r in pattern.validity_reasons])
    ax_price.text(0.02, 0.98, reasons_text, transform=ax_price.transAxes,
                 fontsize=8, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")

    plt.close()


def generate_review_charts(symbols: list = None, lookback_days: int = 120):
    """
    Generate VCP review charts for a list of symbols.

    Args:
        symbols: List of stock symbols to analyze
        lookback_days: Number of days to look back for pattern
    """
    if symbols is None:
        # Default test symbols - mix of likely VCP candidates
        symbols = ['AAPL', 'NVDA', 'MSFT', 'GOOGL', 'META', 'AMZN', 'TSLA', 'AMD']

    detector = VCPDetector(
        swing_lookback=5,
        min_contractions=2,
        max_contraction_range=20.0
    )

    patterns_found = []

    for symbol in symbols:
        print(f"\nAnalyzing {symbol}...")

        try:
            # Get data
            df = yf.download(symbol, period='1y', progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if len(df) < lookback_days:
                print(f"  Not enough data for {symbol}")
                continue

            # Get analysis window
            df_analysis = df.tail(lookback_days).copy()

            # Find all swing points for visualization
            swing_highs = detector.find_swing_highs(df_analysis)
            swing_lows = detector.find_swing_lows(df_analysis)

            # Analyze pattern
            pattern = detector.analyze_pattern(df, lookback_days=lookback_days)

            if pattern:
                print(f"  Pattern found: {len(pattern.contractions)} contractions, "
                      f"Score: {pattern.proximity_score:.0f}")

                # Create chart
                save_path = OUTPUT_DIR / f'{symbol}_vcp_review.png'
                create_vcp_review_chart(
                    symbol=symbol,
                    df=df_analysis,
                    pattern=pattern,
                    swing_highs=swing_highs,
                    swing_lows=swing_lows,
                    save_path=save_path
                )

                patterns_found.append({
                    'symbol': symbol,
                    'contractions': len(pattern.contractions),
                    'is_valid': pattern.is_valid,
                    'score': pattern.proximity_score,
                    'pivot': pattern.pivot_price,
                    'support': pattern.support_price
                })
            else:
                print(f"  No VCP pattern detected")

                # Still create a chart showing all swing points (for debugging)
                if swing_highs or swing_lows:
                    # Create a minimal pattern for visualization
                    from vcp_detector import Contraction

                    # Try to form any contractions for debug view
                    contractions = detector.identify_contractions(df_analysis, swing_highs, swing_lows)
                    if contractions:
                        from vcp_detector import VCPPattern
                        debug_pattern = VCPPattern(
                            contractions=contractions[:3],  # Show first 3
                            is_valid=False,
                            validity_reasons=["Debug view - no valid VCP pattern"],
                            contraction_quality=0,
                            volume_quality=0,
                            proximity_score=0,
                            pivot_price=max(c.swing_high.price for c in contractions[:3]),
                            support_price=min(c.swing_low.price for c in contractions[:3])
                        )

                        save_path = OUTPUT_DIR / f'{symbol}_debug.png'
                        create_vcp_review_chart(
                            symbol=symbol,
                            df=df_analysis,
                            pattern=debug_pattern,
                            swing_highs=swing_highs,
                            swing_lows=swing_lows,
                            save_path=save_path
                        )

        except Exception as e:
            print(f"  Error analyzing {symbol}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n{'='*60}")
    print("VCP Review Charts Summary")
    print('='*60)
    print(f"Symbols analyzed: {len(symbols)}")
    print(f"Patterns found: {len(patterns_found)}")
    print(f"\nCharts saved to: {OUTPUT_DIR}")

    if patterns_found:
        print("\nPatterns detected:")
        for p in patterns_found:
            validity = "✓" if p['is_valid'] else "✗"
            print(f"  {validity} {p['symbol']}: {p['contractions']} contractions, "
                  f"Score: {p['score']:.0f}, Pivot: ${p['pivot']:.2f}")

    return patterns_found


if __name__ == '__main__':
    generate_review_charts()
