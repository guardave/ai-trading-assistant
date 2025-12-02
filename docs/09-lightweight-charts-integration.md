# TradingView Lightweight Charts Integration

## AI Trading Assistant - Charting Documentation

**Version:** 1.0.0
**Date:** 2025-12-01
**Status:** Implemented

---

## 1. Overview

The VCPAlertSystem includes two charting components for visualizing VCP patterns and alerts:

1. **Static Charts (Matplotlib)** - PNG images for individual stocks
2. **Interactive Dashboard (TradingView Lightweight Charts)** - HTML dashboard with all scan results

This document focuses on the TradingView Lightweight Charts integration.

---

## 2. Technology Selection

### 2.1 Why TradingView Lightweight Charts?

| Criteria | Lightweight Charts | Alternative (Plotly) |
|----------|-------------------|---------------------|
| Size | 35KB | 3MB+ |
| License | Apache 2.0 | MIT |
| Rendering | HTML5 Canvas | SVG/WebGL |
| Performance | Excellent | Good |
| Financial Features | Built-in | Requires plugins |
| Server Required | No | No |

### 2.2 Library Details

- **Library**: TradingView Lightweight Charts
- **Version**: 4.1.0
- **CDN**: `https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js`
- **Documentation**: https://tradingview.github.io/lightweight-charts/
- **License**: Apache 2.0

---

## 3. Architecture

### 3.1 Component Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                  LightweightChartGenerator                       │
│                    (src/vcp/chart_lightweight.py)                │
├─────────────────────────────────────────────────────────────────┤
│  Input: List[Dict] scan_results                                  │
│  Output: Single HTML file with embedded data                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Generated HTML Dashboard                      │
├─────────────────────────────────────────────────────────────────┤
│  - TradingView Lightweight Charts JS (CDN)                       │
│  - Embedded JSON data (all stocks)                               │
│  - Vanilla JavaScript UI logic                                   │
│  - CSS styling (embedded)                                        │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Data Flow

```
Scan Results (Python Dict)
         │
         ▼
┌─────────────────────┐
│ _prepare_chart_data │  Convert to JS-compatible format
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   JSON Embedding    │  Embed as `const VCP_DATA = {...}`
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Template Render    │  Insert data into HTML template
└─────────────────────┘
         │
         ▼
     HTML File
```

---

## 4. Usage

### 4.1 Basic Usage

```python
from src.vcp import LightweightChartGenerator

# Create generator
generator = LightweightChartGenerator(output_dir="temp")

# Generate dashboard from scan results
scan_results = [
    {
        "symbol": "AAPL",
        "df": price_dataframe,  # pandas DataFrame with OHLCV
        "pattern": vcp_pattern,  # VCPPattern object
        "alerts": [alert1, alert2],  # List of Alert objects
    },
    # ... more results
]

dashboard_path = generator.generate_dashboard(
    scan_results=scan_results,
    filename="vcp_dashboard.html"
)
print(f"Dashboard generated: {dashboard_path}")
```

### 4.2 Integration with Scan Script

```python
from src.vcp import VCPAlertSystem, LightweightChartGenerator

# Run scan
system = VCPAlertSystem()
results = []

for symbol in symbols:
    df = fetch_data(symbol)
    alerts = system.process_symbol(symbol, df)
    if alerts:
        results.append({
            "symbol": symbol,
            "df": df,
            "pattern": system.detector.get_pattern(df),
            "alerts": alerts,
        })

# Generate dashboard
generator = LightweightChartGenerator()
generator.generate_dashboard(results, "scan_dashboard.html")
```

---

## 5. Data Format

### 5.1 Input Format (Python)

```python
scan_result = {
    "symbol": str,              # Stock symbol (e.g., "AAPL")
    "df": pd.DataFrame,         # OHLCV data with columns:
                                #   Date (index), Open, High, Low, Close, Volume
    "pattern": VCPPattern,      # Pattern object with:
                                #   contractions: List[Contraction]
                                #   pivot_price: float
                                #   support_price: float
                                #   score: float
    "alerts": List[Alert],      # Alert objects with:
                                #   alert_type: AlertType
                                #   created_at: datetime
                                #   trigger_price: float
}
```

### 5.2 Output Format (JavaScript)

```javascript
const VCP_DATA = {
    "AAPL": {
        "ohlcv": [
            {
                "time": "2025-01-02",  // YYYY-MM-DD format
                "open": 150.00,
                "high": 152.00,
                "low": 149.00,
                "close": 151.50,
                "volume": 50000000
            },
            // ... more bars (120 days)
        ],
        "pattern": {
            "contractions": [
                {
                    "start_date": "2025-01-10",
                    "end_date": "2025-01-20",
                    "high": 155.00,
                    "low": 148.00,
                    "depth_pct": 4.5
                }
            ],
            "pivot_price": 156.00,
            "support_price": 148.00,
            "score": 85.0,
            "num_contractions": 3
        },
        "alerts": [
            {
                "type": "trade",     // trade, pre_alert, contraction
                "date": "2025-01-25",
                "price": 156.50
            }
        ],
        "alert_type": "trade"  // Highest priority alert for filtering
    }
};
```

---

## 6. Dashboard Features

### 6.1 UI Components

| Component | Description |
|-----------|-------------|
| Sidebar | Stock list with filter buttons and search |
| Filter Buttons | All / Trade / Pre-Alert / Contraction |
| Search Box | Real-time symbol filtering |
| Stock List | Clickable list with alert indicators |
| Candlestick Chart | Main price chart with markers |
| Volume Chart | Secondary volume chart |
| Info Panel | Pattern details (score, contractions, pivot) |

### 6.2 Keyboard Navigation

| Key | Action |
|-----|--------|
| ↑ | Select previous stock |
| ↓ | Select next stock |
| Enter | Confirm selection |

### 6.3 Alert Indicators

| Symbol | Alert Type |
|--------|------------|
| ★ | Trade Alert |
| ● | Pre-Alert |
| ◆ | Contraction Alert |

### 6.4 Chart Annotations

- **Contraction Zones**: Shaded rectangular regions showing each contraction
- **Pivot Line**: Horizontal green dashed line at pivot price
- **Support Line**: Horizontal red dashed line at support price
- **Alert Markers**: Triangle markers at alert trigger points
  - Green: Trade alerts
  - Yellow: Pre-alerts
  - Blue: Contraction alerts

---

## 7. Customization

### 7.1 Chart Colors

The default color scheme:

```javascript
const COLORS = {
    // Candles
    upColor: '#26a69a',        // Green
    downColor: '#ef5350',      // Red

    // Volume
    volumeUp: 'rgba(38, 166, 154, 0.5)',
    volumeDown: 'rgba(239, 83, 80, 0.5)',

    // Lines
    pivotLine: '#00ff00',      // Green
    supportLine: '#ff0000',    // Red

    // Contraction zones
    contractionFill: 'rgba(100, 100, 100, 0.2)',

    // Markers
    tradeMarker: '#00ff00',
    preAlertMarker: '#ffff00',
    contractionMarker: '#0088ff'
};
```

### 7.2 Chart Options

```javascript
const chartOptions = {
    layout: {
        background: { color: '#1e1e1e' },
        textColor: '#d1d4dc',
    },
    grid: {
        vertLines: { color: '#2B2B43' },
        horzLines: { color: '#2B2B43' },
    },
    crosshair: {
        mode: LightweightCharts.CrosshairMode.Normal,
    },
    rightPriceScale: {
        borderColor: '#2B2B43',
    },
    timeScale: {
        borderColor: '#2B2B43',
        timeVisible: true,
    },
};
```

---

## 8. Performance

### 8.1 File Size

| Stocks | Data Size | HTML File Size |
|--------|-----------|----------------|
| 50 | ~500KB | ~600KB |
| 100 | ~1MB | ~1.2MB |
| 260 | ~2.5MB | ~3MB |
| 500 | ~5MB | ~6MB |

### 8.2 Load Time

- Initial load: 1-3 seconds (depends on file size)
- Stock switching: Instant (<100ms)
- Filtering: Instant (<50ms)

### 8.3 Memory Usage

- Browser memory: ~50-100MB for 260 stocks
- Canvas rendering: Efficient (only visible data rendered)

---

## 9. Limitations

1. **No Real-time Updates**: Data is static at generation time
2. **Browser-based**: Requires modern browser (Chrome, Firefox, Safari, Edge)
3. **File Size**: Large scans produce large HTML files
4. **No Zoom Persistence**: Zoom resets on stock switch
5. **Single File**: All data in one file (no lazy loading)

---

## 10. Future Enhancements

1. **WebSocket Integration**: Real-time price updates
2. **Lazy Loading**: Load stock data on demand
3. **Export Features**: Save charts as images
4. **Drawing Tools**: User annotations
5. **Indicator Overlays**: Moving averages, RSI, etc.
6. **Multi-timeframe**: Switch between daily/weekly/monthly

---

## 11. Troubleshooting

### 11.1 Chart Not Loading

**Problem**: Dashboard shows blank chart area

**Solutions**:
1. Check browser console for JavaScript errors
2. Verify CDN is accessible
3. Check data format in VCP_DATA

### 11.2 Slow Performance

**Problem**: Dashboard is sluggish

**Solutions**:
1. Reduce number of stocks
2. Limit historical data (use 60 days instead of 120)
3. Use Chrome or Firefox for better canvas performance

### 11.3 Missing Annotations

**Problem**: Contractions or price lines not showing

**Solutions**:
1. Verify pattern data includes contractions
2. Check pivot_price and support_price values
3. Ensure dates are in correct format (YYYY-MM-DD)

---

## 12. References

- [TradingView Lightweight Charts Documentation](https://tradingview.github.io/lightweight-charts/)
- [GitHub Repository](https://github.com/tradingview/lightweight-charts)
- [API Reference](https://tradingview.github.io/lightweight-charts/docs/api)
- [Examples](https://tradingview.github.io/lightweight-charts/tutorials)

---

## 13. Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-01 | Claude | Initial documentation |
