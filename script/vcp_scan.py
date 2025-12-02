#!/usr/bin/env python3
"""
VCP Pattern Scanner CLI

A command-line utility to scan stocks for VCP (Volatility Contraction Pattern) setups
and generate interactive HTML dashboards using TradingView Lightweight Charts.

Usage:
    python script/vcp_scan.py                           # Scan S&P 500 (default)
    python script/vcp_scan.py -w watchlist.txt          # Scan from watchlist file
    python script/vcp_scan.py -w watchlist.txt -o out/  # Custom output directory
    python script/vcp_scan.py --symbols AAPL,MSFT,NVDA  # Scan specific symbols
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yfinance as yf
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import json
import warnings

warnings.filterwarnings("ignore")

from src.vcp import (
    VCPAlertSystem,
    SystemConfig,
    DetectorConfig,
    AlertConfig,
    Alert,
    AlertType,
    LightweightChartGenerator,
)


# Current S&P 500 components as of December 2025
SP500_SYMBOLS = [
    "MMM", "AOS", "ABT", "ABBV", "ACN", "ADBE", "AMD", "AES", "AFL", "A",
    "APD", "ABNB", "AKAM", "ALB", "ARE", "ALGN", "ALLE", "LNT", "ALL", "GOOGL",
    "GOOG", "MO", "AMZN", "AMCR", "AEE", "AEP", "AXP", "AIG", "AMT", "AWK",
    "AMP", "AME", "AMGN", "APH", "ADI", "AON", "APA", "APO", "AAPL", "AMAT",
    "APP", "APTV", "ACGL", "ADM", "ANET", "AJG", "AIZ", "T", "ATO", "ADSK",
    "ADP", "AZO", "AVB", "AVY", "AXON", "BKR", "BALL", "BAC", "BAX", "BDX",
    "BRK-B", "BBY", "TECH", "BIIB", "BLK", "BX", "XYZ", "BK", "BA", "BKNG",
    "BSX", "BMY", "AVGO", "BR", "BRO", "BF-B", "BLDR", "BG", "BXP", "CHRW",
    "CDNS", "CPT", "CPB", "COF", "CAH", "CCL", "CARR", "CAT", "CBOE", "CBRE",
    "CDW", "COR", "CNC", "CNP", "CF", "CRL", "SCHW", "CHTR", "CVX", "CMG",
    "CB", "CHD", "CI", "CINF", "CTAS", "CSCO", "C", "CFG", "CLX", "CME",
    "CMS", "KO", "CTSH", "COIN", "CL", "CMCSA", "CAG", "COP", "ED", "STZ",
    "CEG", "COO", "CPRT", "GLW", "CPAY", "CTVA", "CSGP", "COST", "CTRA", "CRWD",
    "CCI", "CSX", "CMI", "CVS", "DHR", "DRI", "DDOG", "DVA", "DAY", "DECK",
    "DE", "DELL", "DAL", "DVN", "DXCM", "FANG", "DLR", "DG", "DLTR", "D",
    "DPZ", "DASH", "DOV", "DOW", "DHI", "DTE", "DUK", "DD", "ETN", "EBAY",
    "ECL", "EIX", "EW", "EA", "ELV", "EME", "EMR", "ETR", "EOG", "EPAM",
    "EQT", "EFX", "EQIX", "EQR", "ERIE", "ESS", "EL", "EG", "EVRG", "ES",
    "EXC", "EXE", "EXPE", "EXPD", "EXR", "XOM", "FFIV", "FDS", "FICO", "FAST",
    "FRT", "FDX", "FIS", "FITB", "FSLR", "FE", "FISV", "F", "FTNT", "FTV",
    "FOXA", "FOX", "BEN", "FCX", "GRMN", "IT", "GE", "GEHC", "GEV", "GEN",
    "GNRC", "GD", "GIS", "GM", "GPC", "GILD", "GPN", "GL", "GDDY", "GS",
    "HAL", "HIG", "HAS", "HCA", "DOC", "HSIC", "HSY", "HPE", "HLT", "HOLX",
    "HD", "HON", "HRL", "HST", "HWM", "HPQ", "HUBB", "HUM", "HBAN", "HII",
    "IBM", "IEX", "IDXX", "ITW", "INCY", "IR", "PODD", "INTC", "IBKR", "ICE",
    "IFF", "IP", "INTU", "ISRG", "IVZ", "INVH", "IQV", "IRM", "JBHT", "JBL",
    "JKHY", "J", "JNJ", "JCI", "JPM", "K", "KVUE", "KDP", "KEY", "KEYS",
    "KMB", "KIM", "KMI", "KKR", "KLAC", "KHC", "KR", "LHX", "LH", "LRCX",
    "LW", "LVS", "LDOS", "LEN", "LII", "LLY", "LIN", "LYV", "LKQ", "LMT",
    "L", "LOW", "LULU", "LYB", "MTB", "MPC", "MAR", "MMC", "MLM", "MAS",
    "MA", "MTCH", "MKC", "MCD", "MCK", "MDT", "MRK", "META", "MET", "MTD",
    "MGM", "MCHP", "MU", "MSFT", "MAA", "MRNA", "MHK", "MOH", "TAP", "MDLZ",
    "MPWR", "MNST", "MCO", "MS", "MOS", "MSI", "MSCI", "NDAQ", "NTAP", "NFLX",
    "NEM", "NWSA", "NWS", "NEE", "NKE", "NI", "NDSN", "NSC", "NTRS", "NOC",
    "NCLH", "NRG", "NUE", "NVDA", "NVR", "NXPI", "ORLY", "OXY", "ODFL", "OMC",
    "ON", "OKE", "ORCL", "OTIS", "PCAR", "PKG", "PLTR", "PANW", "PSKY", "PH",
    "PAYX", "PAYC", "PYPL", "PNR", "PEP", "PFE", "PCG", "PM", "PSX", "PNW",
    "PNC", "POOL", "PPG", "PPL", "PFG", "PG", "PGR", "PLD", "PRU", "PEG",
    "PTC", "PSA", "PHM", "PWR", "QCOM", "DGX", "Q", "RL", "RJF", "RTX",
    "O", "REG", "REGN", "RF", "RSG", "RMD", "RVTY", "HOOD", "ROK", "ROL",
    "ROP", "ROST", "RCL", "SPGI", "CRM", "SNDK", "SBAC", "SLB", "STX", "SRE",
    "NOW", "SHW", "SPG", "SWKS", "SJM", "SW", "SNA", "SOLS", "SOLV", "SO",
    "LUV", "SWK", "SBUX", "STT", "STLD", "STE", "SYK", "SMCI", "SYF", "SNPS",
    "SYY", "TMUS", "TROW", "TTWO", "TPR", "TRGP", "TGT", "TEL", "TDY", "TER",
    "TSLA", "TXN", "TPL", "TXT", "TMO", "TJX", "TKO", "TTD", "TSCO", "TT",
    "TDG", "TRV", "TRMB", "TFC", "TYL", "TSN", "USB", "UBER", "UDR", "ULTA",
    "UNP", "UAL", "UPS", "URI", "UNH", "UHS", "VLO", "VTR", "VLTO", "VRSN",
    "VRSK", "VZ", "VRTX", "VTRS", "VICI", "V", "VST", "VMC", "WRB", "GWW",
    "WAB", "WMT", "DIS", "WBD", "WM", "WAT", "WEC", "WFC", "WELL", "WST",
    "WDC", "WY", "WSM", "WMB", "WTW", "WDAY", "WYNN", "XEL", "XYL", "YUM",
    "ZBRA", "ZBH", "ZTS",
]


def load_watchlist(filepath: str) -> List[str]:
    """Load stock symbols from a watchlist file.

    Supports formats:
    - Comma-separated: AAPL, MSFT, NVDA
    - Line-separated: AAPL\\nMSFT\\nNVDA
    - Mixed: AAPL, MSFT\\nNVDA, GOOGL

    Args:
        filepath: Path to the watchlist file

    Returns:
        List of stock symbols
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Watchlist file not found: {filepath}")

    content = path.read_text()

    # Split by both commas and newlines
    symbols = []
    for line in content.split("\n"):
        # Remove comments (lines starting with #)
        line = line.split("#")[0].strip()
        if not line:
            continue
        # Split by comma
        for symbol in line.split(","):
            symbol = symbol.strip().upper()
            if symbol:
                symbols.append(symbol)

    # Remove duplicates while preserving order
    seen = set()
    unique_symbols = []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            unique_symbols.append(s)

    return unique_symbols


def fetch_data(symbol: str, period: str = "1y") -> pd.DataFrame:
    """Fetch historical OHLCV data for a symbol.

    Args:
        symbol: Stock ticker symbol
        period: Time period (e.g., "6mo", "1y"). Default is 1y to support 200MA.

    Returns:
        DataFrame with OHLCV data
    """
    try:
        df = yf.download(symbol, period=period, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return pd.DataFrame()


def run_scan(
    symbols: List[str],
    output_dir: str = "output",
    dashboard_filename: str = "vcp_dashboard.html",
    verbose: bool = True,
    enable_staleness_check: bool = True,
) -> dict:
    """Run VCP pattern scan on a list of symbols.

    Args:
        symbols: List of stock symbols to scan
        output_dir: Directory for output files
        dashboard_filename: Name of the HTML dashboard file
        verbose: Whether to print progress
        enable_staleness_check: Whether to check for stale patterns (default: True)
                               Set to False for legacy behavior without staleness detection

    Returns:
        Dictionary with scan results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 70)
        print("VCP PATTERN SCANNER")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        staleness_mode = "ENABLED" if enable_staleness_check else "DISABLED (legacy)"
        print(f"Staleness Check: {staleness_mode}")
        print("=" * 70)
        print()
        print(f"Scanning {len(symbols)} symbols...")
        print("-" * 70)

    # Configure the VCP detection system
    config = SystemConfig(
        db_path=str(output_path / "alerts.db"),
        use_memory_db=True,  # Use in-memory for CLI scans
        detector_config=DetectorConfig(
            swing_lookback=5,
            min_contractions=2,
            max_contraction_range=20.0,
            lookback_days=120,
            # Staleness detection settings
            enable_staleness_check=enable_staleness_check,
            max_days_since_contraction=42,  # 6 weeks
            max_pivot_violations=2,
            max_support_violations=1,
        ),
        alert_config=AlertConfig(
            min_score_contraction=60.0,
            pre_alert_proximity_pct=3.0,
            dedup_window_days=1,
        ),
        enable_console_notifications=False,
        enable_log_notifications=False,
    )

    system = VCPAlertSystem(config)

    # Track results
    patterns_found = []
    chart_data = []
    failed_symbols = []
    no_pattern_stocks = []  # Stocks with data but no VCP pattern

    for i, symbol in enumerate(symbols):
        # Progress indicator
        if verbose and (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{len(symbols)} ({len(patterns_found)} patterns)")

        df = fetch_data(symbol)
        if df.empty:
            failed_symbols.append({"symbol": symbol, "reason": "no data"})
            continue

        # For stocks with limited data, still include in dashboard but mark as insufficient
        has_sufficient_data = len(df) >= 120

        try:
            pattern = None
            if has_sufficient_data:
                pattern = system.analyze_pattern(symbol, df)

            current_price = df["Close"].iloc[-1]

            if pattern and pattern.is_valid:
                distance_pct = ((pattern.pivot_price - current_price) / current_price) * 100

                # Determine alert type
                if distance_pct <= 0:
                    alert_type = AlertType.TRADE
                elif distance_pct <= 3.0:
                    alert_type = AlertType.PRE_ALERT
                else:
                    alert_type = AlertType.CONTRACTION

                # Create alert for chart generation
                alert = Alert(
                    symbol=symbol,
                    alert_type=alert_type,
                    trigger_price=current_price,
                    pivot_price=pattern.pivot_price,
                    distance_to_pivot_pct=distance_pct,
                    score=pattern.proximity_score,
                    num_contractions=pattern.num_contractions,
                )

                patterns_found.append({
                    "symbol": symbol,
                    "score": pattern.proximity_score,
                    "contractions": pattern.num_contractions,
                    "pivot": pattern.pivot_price,
                    "support": pattern.support_price,
                    "current_price": current_price,
                    "distance_pct": distance_pct,
                    "last_range_pct": pattern.last_contraction_range_pct,
                    "alert_type": alert_type.value,
                    # Staleness data
                    "days_since_contraction": pattern.days_since_last_contraction,
                    "pivot_violations": pattern.pivot_violations,
                    "support_violations": pattern.support_violations,
                    "is_stale": pattern.is_stale,
                    "freshness_score": pattern.freshness_score,
                    "staleness_reasons": pattern.staleness_reasons,
                })

                chart_data.append({
                    "symbol": symbol,
                    "df": df,
                    "pattern": pattern,
                    "alerts": [alert],
                    "score": pattern.proximity_score,
                    "distance_pct": distance_pct,
                })
            else:
                # No valid pattern - still include in dashboard for charting
                reason = "insufficient data" if not has_sufficient_data else "no pattern"
                no_pattern_stocks.append({
                    "symbol": symbol,
                    "current_price": current_price,
                    "data_days": len(df),
                    "reason": reason,
                })

                # Add to chart_data with no pattern/alerts
                chart_data.append({
                    "symbol": symbol,
                    "df": df,
                    "pattern": None,
                    "alerts": [],
                    "score": 0,
                    "distance_pct": None,
                    "reason": reason,
                })
        except Exception as e:
            failed_symbols.append({"symbol": symbol, "reason": str(e)})
            continue

    # Sort by score
    patterns_found.sort(key=lambda x: x["score"], reverse=True)

    # Categorize alerts
    trade_alerts = [p for p in patterns_found if p["distance_pct"] <= 0]
    pre_alerts = [p for p in patterns_found if 0 < p["distance_pct"] <= 3.0]
    contraction_alerts = [p for p in patterns_found if p["distance_pct"] > 3.0]

    if verbose:
        print()
        print("=" * 70)
        print("SCAN COMPLETE")
        print("=" * 70)
        print()

        # Helper to format staleness indicator
        def staleness_indicator(p):
            if not enable_staleness_check:
                return ""
            if p.get("is_stale"):
                return " [STALE]"
            elif p.get("pivot_violations", 0) > 0:
                return " [!]"
            return ""

        # Print trade alerts
        if trade_alerts:
            print("TRADE ALERTS (Price above pivot)")
            print("-" * 85)
            header = f"{'Symbol':<7} {'Score':<6} {'Price':<10} {'Pivot':<10} {'Above %':<8}"
            if enable_staleness_check:
                header += f" {'Days':<5} {'PViol':<5} {'Fresh':<6}"
            print(header)
            print("-" * 85)
            for p in trade_alerts[:20]:
                line = f"{p['symbol']:<7} {p['score']:<6.0f} ${p['current_price']:<9.2f} ${p['pivot']:<9.2f} {abs(p['distance_pct']):<8.1f}"
                if enable_staleness_check:
                    line += f" {p.get('days_since_contraction', 0):<5} {p.get('pivot_violations', 0):<5} {p.get('freshness_score', 100):<6.0f}"
                line += staleness_indicator(p)
                print(line)
            if len(trade_alerts) > 20:
                print(f"  ... and {len(trade_alerts) - 20} more")
            print()

        # Print pre-alerts
        if pre_alerts:
            print("PRE-ALERTS (Within 3% of pivot)")
            print("-" * 85)
            header = f"{'Symbol':<7} {'Score':<6} {'Price':<10} {'Pivot':<10} {'Dist %':<8}"
            if enable_staleness_check:
                header += f" {'Days':<5} {'PViol':<5} {'Fresh':<6}"
            print(header)
            print("-" * 85)
            for p in sorted(pre_alerts, key=lambda x: x["distance_pct"])[:20]:
                line = f"{p['symbol']:<7} {p['score']:<6.0f} ${p['current_price']:<9.2f} ${p['pivot']:<9.2f} {p['distance_pct']:<8.1f}"
                if enable_staleness_check:
                    line += f" {p.get('days_since_contraction', 0):<5} {p.get('pivot_violations', 0):<5} {p.get('freshness_score', 100):<6.0f}"
                line += staleness_indicator(p)
                print(line)
            print()

        # Print contraction alerts
        if contraction_alerts:
            print("CONTRACTION ALERTS (Top 20 by score)")
            print("-" * 85)
            header = f"{'Symbol':<7} {'Score':<6} {'Price':<10} {'Pivot':<10} {'Dist %':<8}"
            if enable_staleness_check:
                header += f" {'Days':<5} {'PViol':<5} {'Fresh':<6}"
            print(header)
            print("-" * 85)
            for p in contraction_alerts[:20]:
                line = f"{p['symbol']:<7} {p['score']:<6.0f} ${p['current_price']:<9.2f} ${p['pivot']:<9.2f} {p['distance_pct']:<8.1f}"
                if enable_staleness_check:
                    line += f" {p.get('days_since_contraction', 0):<5} {p.get('pivot_violations', 0):<5} {p.get('freshness_score', 100):<6.0f}"
                line += staleness_indicator(p)
                print(line)
            print()

        # Summary
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"  Total symbols:      {len(symbols)}")
        print(f"  Successfully loaded: {len(symbols) - len(failed_symbols)}")
        print(f"  Failed to load:     {len(failed_symbols)}")
        print(f"  No VCP pattern:     {len(no_pattern_stocks)}")
        print(f"  VCP patterns found: {len(patterns_found)}")
        print(f"  - Trade alerts:     {len(trade_alerts)}")
        print(f"  - Pre-alerts:       {len(pre_alerts)}")
        print(f"  - Contractions:     {len(contraction_alerts)}")

        if enable_staleness_check:
            stale_count = sum(1 for p in patterns_found if p.get("is_stale"))
            fresh_count = len(patterns_found) - stale_count
            print(f"  Fresh patterns:     {fresh_count}")
            print(f"  Stale patterns:     {stale_count}")

        if len(symbols) - len(failed_symbols) > 0:
            hit_rate = len(patterns_found) / (len(symbols) - len(failed_symbols)) * 100
            print(f"  Hit rate:           {hit_rate:.1f}%")
        print()

    # Generate dashboard
    if chart_data:
        if verbose:
            print("Generating interactive dashboard...")

        generator = LightweightChartGenerator(output_dir=str(output_path))
        dashboard_path = generator.generate_dashboard(
            scan_results=chart_data,
            filename=dashboard_filename,
        )

        if verbose:
            print(f"Dashboard saved: {dashboard_path}")
            print()
            print("Open the HTML file in a web browser to view interactive charts.")

    # Save results to JSON
    results = {
        "scan_date": datetime.now().isoformat(),
        "symbols_total": len(symbols),
        "symbols_scanned": len(symbols) - len(failed_symbols),
        "patterns_found": len(patterns_found),
        "no_pattern_count": len(no_pattern_stocks),
        "summary": {
            "trade_alerts": len(trade_alerts),
            "pre_alerts": len(pre_alerts),
            "contraction_alerts": len(contraction_alerts),
        },
        "patterns": patterns_found,
        "no_pattern_stocks": no_pattern_stocks,
        "failed_symbols": failed_symbols,
    }

    results_file = output_path / "scan_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    if verbose:
        print(f"Results saved: {results_file}")
        print("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="VCP Pattern Scanner - Scan stocks for Volatility Contraction Patterns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Scan S&P 500 (default)
  %(prog)s -w watchlist.txt          # Scan symbols from watchlist file
  %(prog)s -w watchlist.txt -o out/  # Custom output directory
  %(prog)s --symbols AAPL,MSFT,NVDA  # Scan specific symbols
  %(prog)s -q                        # Quiet mode (less output)
  %(prog)s --no-staleness            # Disable staleness check (legacy mode)

Staleness Detection:
  By default, the scanner checks for stale VCP patterns:
  - Patterns older than 6 weeks are marked stale
  - Pivot violations (price crossed above pivot then fell back)
  - Support violations (price closed below support)
  Use --no-staleness to disable these checks.

Watchlist file format (comma or line separated):
  AAPL, MSFT, NVDA
  GOOGL, AMZN
  # This is a comment
  TSLA
        """,
    )

    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "-w", "--watchlist",
        type=str,
        help="Path to watchlist file with stock symbols (comma-separated)",
    )
    input_group.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated list of stock symbols to scan",
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default="output",
        help="Output directory for dashboard and results (default: output)",
    )

    parser.add_argument(
        "-f", "--filename",
        type=str,
        default="vcp_dashboard.html",
        help="Dashboard filename (default: vcp_dashboard.html)",
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Quiet mode - minimal output",
    )

    parser.add_argument(
        "--no-staleness",
        action="store_true",
        help="Disable staleness check (use legacy detection behavior)",
    )

    args = parser.parse_args()

    # Determine symbols to scan
    if args.watchlist:
        try:
            symbols = load_watchlist(args.watchlist)
            if not symbols:
                print(f"Error: No symbols found in watchlist: {args.watchlist}")
                sys.exit(1)
            print(f"Loaded {len(symbols)} symbols from {args.watchlist}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
    elif args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        if not symbols:
            print("Error: No valid symbols provided")
            sys.exit(1)
    else:
        # Default to S&P 500
        symbols = SP500_SYMBOLS
        if not args.quiet:
            print(f"Using default S&P 500 list ({len(symbols)} symbols)")

    # Run the scan
    run_scan(
        symbols=symbols,
        output_dir=args.output,
        dashboard_filename=args.filename,
        verbose=not args.quiet,
        enable_staleness_check=not args.no_staleness,
    )


if __name__ == "__main__":
    main()
