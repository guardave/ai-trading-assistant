"""
VCP Alert System - Chart Visualization

Creates charts for VCP patterns with:
1. Candlestick price data
2. Swing highs/lows marked
3. Contractions shown as trendlines
4. Alert levels (pivot, support) marked
5. Current alert type indicator
6. Volume bars with moving average
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from .models import (
    Alert,
    AlertType,
    VCPPattern,
    SwingPoint,
    Contraction,
)


# Color scheme for consistency
COLORS = {
    "contraction_lines": ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"],
    "trade_alert": "#00C853",       # Green for trade alerts
    "pre_alert": "#FFD600",         # Yellow for pre-alerts
    "contraction_alert": "#2196F3", # Blue for contraction alerts
    "pivot_line": "#00C853",        # Green dashed line
    "support_line": "#F44336",      # Red dashed line
    "swing_high": "#1976D2",        # Blue triangle
    "swing_low": "#FF9800",         # Orange triangle
    "bullish_candle": "#26A69A",    # Teal green
    "bearish_candle": "#EF5350",    # Red
    "volume_up": "#26A69A",
    "volume_down": "#EF5350",
}


def create_alert_chart(
    symbol: str,
    df: pd.DataFrame,
    pattern: VCPPattern,
    alerts: List[Alert],
    swing_highs: Optional[List[SwingPoint]] = None,
    swing_lows: Optional[List[SwingPoint]] = None,
    save_path: Optional[Path] = None,
    show_all_swings: bool = False,
) -> Optional[plt.Figure]:
    """
    Create a comprehensive chart for VCP pattern with alerts.

    Args:
        symbol: Stock symbol
        df: DataFrame with OHLCV data
        pattern: VCPPattern object with detected contractions
        alerts: List of alerts to mark on chart
        swing_highs: Optional list of all swing high points
        swing_lows: Optional list of all swing low points
        save_path: Path to save the chart (if None, returns figure)
        show_all_swings: Whether to show all swing points (default: only pattern swings)

    Returns:
        matplotlib Figure if save_path is None, else None
    """
    # Prepare data
    df_plot = df.copy()

    # Create figure with panels
    # Note: figsize * dpi must stay under 2000px for API compatibility
    fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                             gridspec_kw={"height_ratios": [3, 1, 0.8]})

    ax_price = axes[0]
    ax_volume = axes[1]
    ax_legend = axes[2]
    ax_legend.axis("off")

    # Plot candlesticks
    _plot_candlesticks(ax_price, df_plot)

    # Set x-axis labels
    x_ticks = list(range(0, len(df_plot), max(1, len(df_plot) // 10)))
    x_labels = [df_plot.index[i].strftime("%Y-%m-%d") for i in x_ticks]
    ax_price.set_xticks(x_ticks)
    ax_price.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax_price.set_xlim(-1, len(df_plot))
    ax_price.set_ylabel("Price ($)", fontsize=10)

    # Adjust y-axis limits
    y_min = df_plot["Low"].min() * 0.97
    y_max = df_plot["High"].max() * 1.03
    ax_price.set_ylim(y_min, y_max)

    # Create index mapping (date to position)
    date_to_pos = {date: i for i, date in enumerate(df_plot.index)}

    # Plot all swing points if requested
    if show_all_swings:
        if swing_highs:
            for sh in swing_highs:
                if sh.date in date_to_pos:
                    pos = date_to_pos[sh.date]
                    ax_price.scatter(pos, sh.price * 1.003, marker="v",
                                   color=COLORS["swing_high"], s=30, zorder=4, alpha=0.4)

        if swing_lows:
            for sl in swing_lows:
                if sl.date in date_to_pos:
                    pos = date_to_pos[sl.date]
                    ax_price.scatter(pos, sl.price * 0.997, marker="^",
                                   color=COLORS["swing_low"], s=30, zorder=4, alpha=0.4)

    # Plot contractions as trendlines
    _plot_contractions(ax_price, df_plot, pattern, date_to_pos, y_max - y_min)

    # Draw pivot and support lines
    ax_price.axhline(y=pattern.pivot_price, color=COLORS["pivot_line"], linestyle="--",
                    linewidth=2, alpha=0.8, zorder=3)
    ax_price.axhline(y=pattern.support_price, color=COLORS["support_line"], linestyle="--",
                    linewidth=2, alpha=0.8, zorder=3)

    # Add price labels on the right
    ax_price.text(len(df_plot) + 0.5, pattern.pivot_price,
                 f" Pivot: ${pattern.pivot_price:.2f}",
                 fontsize=9, va="center", color=COLORS["pivot_line"], fontweight="bold")
    ax_price.text(len(df_plot) + 0.5, pattern.support_price,
                 f" Support: ${pattern.support_price:.2f}",
                 fontsize=9, va="center", color=COLORS["support_line"], fontweight="bold")

    # Mark current price with alert type
    _mark_current_price_and_alerts(ax_price, df_plot, pattern, alerts)

    # Plot volume bars
    _plot_volume(ax_volume, df_plot, x_ticks)

    # Create legend
    _create_legend(ax_legend, pattern, alerts)

    # Title with pattern and alert info
    alert_types = [a.alert_type.value.upper().replace("_", " ") for a in alerts]
    alert_str = ", ".join(alert_types) if alerts else "No Alert"

    current_price = df_plot["Close"].iloc[-1]
    distance_pct = ((pattern.pivot_price - current_price) / current_price) * 100

    title = (f"{symbol} - VCP Pattern Analysis\n"
             f"Alert: {alert_str} | Score: {pattern.proximity_score:.0f} | "
             f"Distance to Pivot: {distance_pct:+.1f}%")
    ax_price.set_title(title, fontsize=12, fontweight="bold", pad=10)

    # Add validity info box
    _add_validity_box(ax_price, pattern)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        return None
    else:
        return fig


def _plot_candlesticks(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Plot candlestick chart."""
    for i, (idx, row) in enumerate(df.iterrows()):
        is_bullish = row["Close"] >= row["Open"]
        color = COLORS["bullish_candle"] if is_bullish else COLORS["bearish_candle"]

        # Wick
        ax.plot([i, i], [row["Low"], row["High"]], color="black", linewidth=0.5)

        # Body
        body_bottom = min(row["Open"], row["Close"])
        body_height = abs(row["Close"] - row["Open"])
        if body_height < 0.001:  # Doji
            body_height = 0.001

        rect = mpatches.Rectangle(
            (i - 0.35, body_bottom), 0.7, body_height,
            facecolor=color, edgecolor="black", linewidth=0.5
        )
        ax.add_patch(rect)


def _plot_contractions(
    ax: plt.Axes,
    df: pd.DataFrame,
    pattern: VCPPattern,
    date_to_pos: Dict,
    y_range: float,
) -> None:
    """Plot contraction trendlines with labels."""
    colors = COLORS["contraction_lines"]

    for i, contraction in enumerate(pattern.contractions):
        color = colors[i % len(colors)]

        high_date = contraction.swing_high.date
        low_date = contraction.swing_low.date

        # Convert pandas Timestamp to datetime for comparison
        if hasattr(high_date, 'to_pydatetime'):
            high_date = high_date.to_pydatetime()
        if hasattr(low_date, 'to_pydatetime'):
            low_date = low_date.to_pydatetime()

        # Find matching position by comparing dates
        high_pos = None
        low_pos = None
        for d, pos in date_to_pos.items():
            d_compare = d.to_pydatetime() if hasattr(d, 'to_pydatetime') else d
            if high_date and d_compare.date() == high_date.date():
                high_pos = pos
            if low_date and d_compare.date() == low_date.date():
                low_pos = pos

        if high_pos is not None and low_pos is not None:
            high_price = contraction.swing_high.price
            low_price = contraction.swing_low.price

            # Draw trendline from swing high to swing low
            ax.plot([high_pos, low_pos], [high_price, low_price],
                   color=color, linewidth=2.5, linestyle="-",
                   marker="o", markersize=8, zorder=10)

            # Add contraction label
            mid_pos = (high_pos + low_pos) / 2
            mid_price = (high_price + low_price) / 2
            label_offset = y_range * 0.03

            ax.annotate(
                f"C{i+1}\n{contraction.range_pct:.1f}%\n{contraction.duration_days}d",
                xy=(mid_pos, mid_price),
                xytext=(mid_pos + 2, mid_price + label_offset),
                fontsize=8,
                fontweight="bold",
                color=color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                         edgecolor=color, alpha=0.9),
                arrowprops=dict(arrowstyle="->", color=color, lw=1),
                zorder=15
            )

            # Mark swing high with larger marker
            ax.scatter(high_pos, high_price, marker="v", color=color,
                      s=120, zorder=12, edgecolor="black", linewidth=1)

            # Mark swing low with larger marker
            ax.scatter(low_pos, low_price, marker="^", color=color,
                      s=120, zorder=12, edgecolor="black", linewidth=1)


def _mark_current_price_and_alerts(
    ax: plt.Axes,
    df: pd.DataFrame,
    pattern: VCPPattern,
    alerts: List[Alert],
) -> None:
    """Mark current price and alert type on chart."""
    current_price = df["Close"].iloc[-1]
    current_pos = len(df) - 1

    # Determine alert type color
    if alerts:
        # Use the highest priority alert type
        if any(a.alert_type == AlertType.TRADE for a in alerts):
            alert_color = COLORS["trade_alert"]
            alert_label = "TRADE"
        elif any(a.alert_type == AlertType.PRE_ALERT for a in alerts):
            alert_color = COLORS["pre_alert"]
            alert_label = "PRE-ALERT"
        else:
            alert_color = COLORS["contraction_alert"]
            alert_label = "CONTRACTION"
    else:
        alert_color = "#9E9E9E"  # Gray for no alert
        alert_label = "PATTERN"

    # Mark current price with a horizontal line
    ax.axhline(y=current_price, color=alert_color, linestyle="-",
              linewidth=1.5, alpha=0.5, xmin=0.9, xmax=1.0)

    # Add current price marker
    ax.scatter(current_pos, current_price, marker="D", color=alert_color,
              s=100, zorder=20, edgecolor="black", linewidth=1)

    # Add price/alert label
    ax.annotate(
        f"{alert_label}\n${current_price:.2f}",
        xy=(current_pos, current_price),
        xytext=(current_pos - 8, current_price),
        fontsize=9,
        fontweight="bold",
        color="white",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=alert_color,
                 edgecolor="black", alpha=0.95),
        zorder=25
    )


def _plot_volume(ax: plt.Axes, df: pd.DataFrame, x_ticks: List[int]) -> None:
    """Plot volume bars with moving average."""
    colors = [COLORS["volume_up"] if df.iloc[i]["Close"] >= df.iloc[i]["Open"]
              else COLORS["volume_down"] for i in range(len(df))]

    ax.bar(range(len(df)), df["Volume"] / 1e6, color=colors, alpha=0.7, width=0.8)
    ax.set_xlim(-1, len(df))
    ax.set_ylabel("Volume (M)", fontsize=9)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([])

    # Add 50-day volume MA
    vol_ma = df["Volume"].rolling(50).mean()
    ax.plot(range(len(df)), vol_ma / 1e6, color="#1976D2", linewidth=1.5,
           label="50-day avg", alpha=0.8)
    ax.legend(loc="upper right", fontsize=8)


def _create_legend(ax: plt.Axes, pattern: VCPPattern, alerts: List[Alert]) -> None:
    """Create legend panel."""
    legend_elements = [
        Line2D([0], [0], color=COLORS["pivot_line"], linestyle="--", linewidth=2,
               label=f"Pivot: ${pattern.pivot_price:.2f}"),
        Line2D([0], [0], color=COLORS["support_line"], linestyle="--", linewidth=2,
               label=f"Support: ${pattern.support_price:.2f}"),
    ]

    # Add contraction legend items
    colors = COLORS["contraction_lines"]
    for i, c in enumerate(pattern.contractions):
        color = colors[i % len(colors)]
        legend_elements.append(
            Line2D([0], [0], color=color, linewidth=2.5, marker="o", markersize=6,
                   label=f"C{i+1}: {c.range_pct:.1f}% range, {c.duration_days}d, Vol: {c.avg_volume_ratio:.2f}x")
        )

    # Add alert type indicators
    if alerts:
        for alert in alerts:
            if alert.alert_type == AlertType.TRADE:
                color = COLORS["trade_alert"]
                label = f"Trade Alert (Score: {alert.score:.0f})"
            elif alert.alert_type == AlertType.PRE_ALERT:
                color = COLORS["pre_alert"]
                label = f"Pre-Alert (Dist: {alert.distance_to_pivot_pct:.1f}%)"
            else:
                color = COLORS["contraction_alert"]
                label = f"Contraction Alert (Score: {alert.score:.0f})"

            legend_elements.append(
                mpatches.Patch(facecolor=color, edgecolor="black",
                              label=label)
            )

    ax.legend(handles=legend_elements, loc="center", ncol=3, fontsize=9,
             frameon=True, fancybox=True)


def _add_validity_box(ax: plt.Axes, pattern: VCPPattern) -> None:
    """Add pattern validity info box."""
    validity = "VALID" if pattern.is_valid else "INVALID"
    reasons_text = f"Pattern: {validity}\n"
    reasons_text += "\n".join([f"• {r}" for r in pattern.validity_reasons[:4]])

    ax.text(0.02, 0.98, reasons_text, transform=ax.transAxes,
           fontsize=8, verticalalignment="top",
           bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85))


class ChartGenerator:
    """
    Chart generator for VCP Alert System.

    Generates charts for patterns and alerts with consistent styling.
    """

    def __init__(self, output_dir: str = "charts"):
        """
        Initialize chart generator.

        Args:
            output_dir: Directory to save charts
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_alert_chart(
        self,
        symbol: str,
        df: pd.DataFrame,
        pattern: VCPPattern,
        alerts: List[Alert],
        swing_highs: Optional[List[SwingPoint]] = None,
        swing_lows: Optional[List[SwingPoint]] = None,
        filename: Optional[str] = None,
    ) -> Path:
        """
        Generate and save an alert chart.

        Args:
            symbol: Stock symbol
            df: DataFrame with OHLCV data
            pattern: VCPPattern object
            alerts: List of alerts
            swing_highs: Optional swing highs
            swing_lows: Optional swing lows
            filename: Optional custom filename

        Returns:
            Path to saved chart
        """
        if filename is None:
            # Determine alert type for filename
            if alerts:
                if any(a.alert_type == AlertType.TRADE for a in alerts):
                    alert_suffix = "trade"
                elif any(a.alert_type == AlertType.PRE_ALERT for a in alerts):
                    alert_suffix = "prealert"
                else:
                    alert_suffix = "contraction"
            else:
                alert_suffix = "pattern"

            filename = f"{symbol}_{alert_suffix}.png"

        save_path = self.output_dir / filename

        create_alert_chart(
            symbol=symbol,
            df=df,
            pattern=pattern,
            alerts=alerts,
            swing_highs=swing_highs,
            swing_lows=swing_lows,
            save_path=save_path,
        )

        return save_path

    def generate_batch_charts(
        self,
        results: List[Dict[str, Any]],
        max_charts: Optional[int] = None,
    ) -> List[Path]:
        """
        Generate charts for multiple scan results.

        Args:
            results: List of dicts with 'symbol', 'df', 'pattern', 'alerts' keys
            max_charts: Maximum number of charts to generate

        Returns:
            List of paths to saved charts
        """
        saved_paths = []

        for i, result in enumerate(results):
            if max_charts and i >= max_charts:
                break

            try:
                path = self.generate_alert_chart(
                    symbol=result["symbol"],
                    df=result["df"],
                    pattern=result["pattern"],
                    alerts=result.get("alerts", []),
                    swing_highs=result.get("swing_highs"),
                    swing_lows=result.get("swing_lows"),
                )
                saved_paths.append(path)
            except Exception as e:
                print(f"Error generating chart for {result.get('symbol', 'unknown')}: {e}")

        return saved_paths
