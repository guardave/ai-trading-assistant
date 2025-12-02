"""
VCP Alert System - Lightweight Charts Integration

Generates interactive HTML dashboards using TradingView's Lightweight Charts library.
Features:
- Multi-stock selector with filtering by alert type
- Interactive candlestick charts with volume
- VCP contraction visualization
- Pivot/support level markers
- Moving Average indicators (10, 20, 50, 63, 150, 200)
- Visible Range Volume Profile
- Toggle controls for all indicators
- All data preloaded for fast switching
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd

from .models import (
    Alert,
    AlertType,
    VCPPattern,
)


# HTML Template with embedded Lightweight Charts
HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VCP Alert Dashboard</title>
    <script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #1a1a2e;
            color: #eee;
            height: 100vh;
            overflow: hidden;
        }

        .container {
            display: flex;
            height: 100vh;
        }

        /* Left Sidebar */
        .sidebar {
            width: 280px;
            background: #16213e;
            border-right: 1px solid #0f3460;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .sidebar-header {
            padding: 15px;
            background: #0f3460;
            border-bottom: 1px solid #1a1a2e;
        }

        .sidebar-header h1 {
            font-size: 18px;
            color: #00d4ff;
            margin-bottom: 10px;
        }

        .search-box {
            width: 100%;
            padding: 8px 12px;
            border: 1px solid #0f3460;
            border-radius: 4px;
            background: #1a1a2e;
            color: #eee;
            font-size: 14px;
        }

        .search-box:focus {
            outline: none;
            border-color: #00d4ff;
        }

        .filter-buttons {
            display: flex;
            gap: 5px;
            margin-top: 10px;
            flex-wrap: wrap;
        }

        .filter-btn {
            padding: 5px 10px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            transition: all 0.2s;
        }

        .filter-btn.all { background: #444; color: #fff; }
        .filter-btn.trade { background: #00c853; color: #000; }
        .filter-btn.pre-alert { background: #ffd600; color: #000; }
        .filter-btn.contraction { background: #2196f3; color: #fff; }
        .filter-btn.none { background: #666; color: #ccc; }
        .filter-btn.active { box-shadow: 0 0 0 2px #00d4ff; }

        .selection-controls {
            display: flex;
            gap: 5px;
            margin-top: 8px;
        }

        .select-all-btn {
            flex: 1;
            padding: 5px 8px;
            border: 1px solid #666;
            border-radius: 4px;
            cursor: pointer;
            font-size: 11px;
            background: transparent;
            color: #888;
            transition: all 0.2s;
        }

        .select-all-btn:hover {
            border-color: #00d4ff;
            color: #00d4ff;
        }

        .copy-csv-btn {
            flex: 2;
            padding: 5px 10px;
            border: 1px solid #00d4ff;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            background: transparent;
            color: #00d4ff;
            transition: all 0.2s;
        }

        .copy-csv-btn:hover {
            background: #00d4ff;
            color: #000;
        }

        .copy-csv-btn.copied {
            background: #00c853;
            border-color: #00c853;
            color: #000;
        }

        .copy-csv-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .stock-checkbox {
            width: 14px;
            height: 14px;
            cursor: pointer;
            accent-color: #00d4ff;
        }

        .stock-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 10px;
            margin: 2px 0;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s;
            gap: 8px;
        }

        .stock-info {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex: 1;
        }

        .selected-count {
            font-size: 11px;
            color: #00d4ff;
            margin-top: 5px;
        }

        .stock-list {
            flex: 1;
            overflow-y: auto;
            padding: 10px;
        }

        .stock-group {
            margin-bottom: 15px;
        }

        .stock-group-header {
            font-size: 12px;
            color: #888;
            padding: 5px 0;
            border-bottom: 1px solid #0f3460;
            margin-bottom: 5px;
            display: flex;
            align-items: center;
            gap: 5px;
        }

        .stock-item:hover {
            background: #0f3460;
        }

        .stock-item.active {
            background: #0f3460;
            border-left: 3px solid #00d4ff;
        }

        .stock-symbol {
            font-weight: 600;
            font-size: 14px;
        }

        .stock-score {
            font-size: 12px;
            color: #888;
        }

        .stock-item.trade .stock-symbol { color: #00c853; }
        .stock-item.pre-alert .stock-symbol { color: #ffd600; }
        .stock-item.contraction .stock-symbol { color: #2196f3; }
        .stock-item.none .stock-symbol { color: #888; }

        /* Main Content */
        .main-content {
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .chart-header {
            padding: 15px 20px;
            background: #16213e;
            border-bottom: 1px solid #0f3460;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 10px;
        }

        .chart-title {
            display: flex;
            align-items: center;
            gap: 15px;
        }

        .chart-title h2 {
            font-size: 24px;
            color: #fff;
        }

        .alert-badge {
            padding: 4px 12px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
        }

        .alert-badge.trade { background: #00c853; color: #000; }
        .alert-badge.pre-alert { background: #ffd600; color: #000; }
        .alert-badge.contraction { background: #2196f3; color: #fff; }

        .chart-stats {
            display: flex;
            gap: 20px;
            font-size: 13px;
        }

        .stat-item {
            display: flex;
            flex-direction: column;
        }

        .stat-label {
            color: #888;
            font-size: 11px;
        }

        .stat-value {
            color: #fff;
            font-weight: 600;
        }

        /* Indicator Controls */
        .indicator-controls {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            padding: 10px 20px;
            background: #0f3460;
            border-bottom: 1px solid #1a1a2e;
        }

        .indicator-toggle {
            display: flex;
            align-items: center;
            gap: 5px;
            padding: 4px 10px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 11px;
            background: #1a1a2e;
            border: 1px solid #2B2B43;
            transition: all 0.2s;
        }

        .indicator-toggle:hover {
            border-color: #00d4ff;
        }

        .indicator-toggle.active {
            background: #16213e;
            border-color: var(--indicator-color, #00d4ff);
        }

        .indicator-toggle .color-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }

        .indicator-toggle input {
            display: none;
        }

        .indicator-section {
            display: flex;
            align-items: center;
            gap: 5px;
        }

        .indicator-section-label {
            font-size: 10px;
            color: #666;
            margin-right: 5px;
        }

        .indicator-divider {
            width: 1px;
            height: 20px;
            background: #2B2B43;
            margin: 0 5px;
        }

        .chart-container {
            flex: 1;
            padding: 10px;
            display: flex;
            flex-direction: column;
            position: relative;
            overflow: hidden;
        }

        #price-chart {
            flex: 3;
            min-height: 300px;
        }

        #volume-chart {
            flex: 1;
            min-height: 100px;
            margin-top: 5px;
        }

        /* Volume Profile Overlay - Left Side */
        .volume-profile-container {
            position: absolute;
            left: 10px;
            top: 0;
            bottom: 0;
            width: 100px;
            pointer-events: none;
            z-index: 10;
        }

        .volume-profile-bar {
            position: absolute;
            left: 0;
            height: 4px;
            background: rgba(33, 150, 243, 0.6);
            border-radius: 2px;
        }

        .volume-profile-bar.poc {
            background: rgba(255, 214, 0, 0.8);
            height: 6px;
        }

        .pattern-info {
            padding: 15px 20px;
            background: #16213e;
            border-top: 1px solid #0f3460;
        }

        .pattern-info h3 {
            font-size: 14px;
            color: #00d4ff;
            margin-bottom: 10px;
        }

        .contractions-list {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }

        .contraction-item {
            background: #0f3460;
            padding: 10px 15px;
            border-radius: 4px;
            font-size: 12px;
        }

        .contraction-item .label {
            color: #888;
        }

        .contraction-item .value {
            color: #fff;
            font-weight: 600;
        }

        .validity-reasons {
            margin-top: 10px;
            font-size: 12px;
            color: #888;
        }

        .validity-reasons li {
            margin-left: 20px;
            margin-top: 3px;
        }

        /* Footer */
        .footer {
            padding: 10px 20px;
            background: #0f3460;
            font-size: 11px;
            color: #666;
            text-align: center;
        }

        .footer a {
            color: #00d4ff;
            text-decoration: none;
        }

        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }

        ::-webkit-scrollbar-track {
            background: #1a1a2e;
        }

        ::-webkit-scrollbar-thumb {
            background: #0f3460;
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: #00d4ff;
        }

        /* Keyboard shortcuts hint */
        .shortcuts-hint {
            font-size: 11px;
            color: #666;
            padding: 10px;
            border-top: 1px solid #0f3460;
        }

        kbd {
            background: #0f3460;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <div class="sidebar-header">
                <h1>VCP Alert Dashboard</h1>
                <input type="text" class="search-box" placeholder="Search symbol..." id="search-input">
                <div class="filter-buttons">
                    <button class="filter-btn all active" data-filter="all">All ({{total_count}})</button>
                    <button class="filter-btn trade" data-filter="trade">Trade ({{trade_count}})</button>
                    <button class="filter-btn pre-alert" data-filter="pre_alert">Pre ({{prealert_count}})</button>
                    <button class="filter-btn contraction" data-filter="contraction">Contr ({{contraction_count}})</button>
                    <button class="filter-btn none" data-filter="none">None ({{none_count}})</button>
                </div>
                <div class="selection-controls">
                    <button class="select-all-btn" id="select-all-btn">Select All</button>
                    <button class="select-all-btn" id="unselect-all-btn">Clear</button>
                    <button class="copy-csv-btn" id="copy-csv-btn" disabled>📋 Copy (0)</button>
                </div>
                <div class="selected-count" id="selected-count"></div>
            </div>
            <div class="stock-list" id="stock-list">
                <!-- Populated by JavaScript -->
            </div>
            <div class="shortcuts-hint">
                <kbd>↑</kbd><kbd>↓</kbd> Navigate &nbsp; <kbd>Enter</kbd> Select
            </div>
        </div>

        <div class="main-content">
            <div class="chart-header">
                <div class="chart-title">
                    <h2 id="current-symbol">Select a stock</h2>
                    <span class="alert-badge" id="alert-badge" style="display:none;"></span>
                </div>
                <div class="chart-stats">
                    <div class="stat-item">
                        <span class="stat-label">Score</span>
                        <span class="stat-value" id="stat-score">-</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Distance to Pivot</span>
                        <span class="stat-value" id="stat-distance">-</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Current Price</span>
                        <span class="stat-value" id="stat-price">-</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Pivot</span>
                        <span class="stat-value" id="stat-pivot">-</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Support</span>
                        <span class="stat-value" id="stat-support">-</span>
                    </div>
                </div>
            </div>

            <!-- Indicator Toggle Controls -->
            <div class="indicator-controls">
                <span class="indicator-section-label">MAs:</span>
                <label class="indicator-toggle" style="--indicator-color: #ff6b6b;">
                    <input type="checkbox" id="ma-10" checked>
                    <span class="color-dot" style="background: #ff6b6b;"></span>
                    <span>10</span>
                </label>
                <label class="indicator-toggle" style="--indicator-color: #4ecdc4;">
                    <input type="checkbox" id="ma-20" checked>
                    <span class="color-dot" style="background: #4ecdc4;"></span>
                    <span>20</span>
                </label>
                <label class="indicator-toggle" style="--indicator-color: #45b7d1;">
                    <input type="checkbox" id="ma-50" checked>
                    <span class="color-dot" style="background: #45b7d1;"></span>
                    <span>50</span>
                </label>
                <label class="indicator-toggle" style="--indicator-color: #96ceb4;">
                    <input type="checkbox" id="ma-63">
                    <span class="color-dot" style="background: #96ceb4;"></span>
                    <span>63</span>
                </label>
                <label class="indicator-toggle" style="--indicator-color: #ffd93d;">
                    <input type="checkbox" id="ma-150">
                    <span class="color-dot" style="background: #ffd93d;"></span>
                    <span>150</span>
                </label>
                <label class="indicator-toggle" style="--indicator-color: #ff8c42;">
                    <input type="checkbox" id="ma-200" checked>
                    <span class="color-dot" style="background: #ff8c42;"></span>
                    <span>200</span>
                </label>
                <div class="indicator-divider"></div>
                <span class="indicator-section-label">Other:</span>
                <label class="indicator-toggle" style="--indicator-color: #2196f3;">
                    <input type="checkbox" id="vol-profile">
                    <span class="color-dot" style="background: #2196f3;"></span>
                    <span>Vol Profile</span>
                </label>
            </div>

            <div class="chart-container">
                <div id="price-chart"></div>
                <div id="volume-chart"></div>
                <div id="volume-profile-container" class="volume-profile-container" style="display:none;"></div>
            </div>

            <div class="pattern-info">
                <h3>Pattern Details</h3>
                <div class="contractions-list" id="contractions-list">
                    <!-- Populated by JavaScript -->
                </div>
                <ul class="validity-reasons" id="validity-reasons">
                    <!-- Populated by JavaScript -->
                </ul>
            </div>

            <div class="footer">
                Generated: {{generation_date}} |
                Charts powered by <a href="https://www.tradingview.com/" target="_blank">TradingView Lightweight Charts</a>
            </div>
        </div>
    </div>

    <script>
        // Embedded stock data
        const VCP_DATA = {{vcp_data_json}};

        // Chart instances
        let priceChart = null;
        let volumeChart = null;
        let candlestickSeries = null;
        let volumeSeries = null;
        let volumeMaSeries = null;
        let currentSymbol = null;
        let pivotLine = null;
        let supportLine = null;
        let contractionLines = [];

        // MA Series
        let maSeries = {};
        const MA_PERIODS = [10, 20, 50, 63, 150, 200];
        const MA_COLORS = {
            10: '#ff6b6b',
            20: '#4ecdc4',
            50: '#45b7d1',
            63: '#96ceb4',
            150: '#ffd93d',
            200: '#ff8c42'
        };

        // Colors
        const COLORS = {
            trade: '#00c853',
            pre_alert: '#ffd600',
            contraction: '#2196f3',
            pivot: '#00c853',
            support: '#f44336',
            contractionLines: ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'],
            upColor: '#26a69a',
            downColor: '#ef5350',
        };

        // Calculate Simple Moving Average for price
        function calculateSMA(data, period) {
            const result = [];
            for (let i = 0; i < data.length; i++) {
                if (i < period - 1) {
                    continue;
                }
                let sum = 0;
                for (let j = 0; j < period; j++) {
                    sum += data[i - j].close;
                }
                result.push({
                    time: data[i].time,
                    value: sum / period
                });
            }
            return result;
        }

        // Calculate Simple Moving Average for volume
        function calculateVolumeSMA(data, period) {
            const result = [];
            for (let i = 0; i < data.length; i++) {
                if (i < period - 1) {
                    continue;
                }
                let sum = 0;
                for (let j = 0; j < period; j++) {
                    sum += data[i - j].volume;
                }
                result.push({
                    time: data[i].time,
                    value: sum / period
                });
            }
            return result;
        }

        // Calculate Volume Profile for visible range
        function calculateVolumeProfile(data, numBins = 20) {
            if (data.length === 0) return { bins: [], poc: null };

            const prices = data.map(d => d.close);
            const minPrice = Math.min(...prices);
            const maxPrice = Math.max(...prices);
            const priceRange = maxPrice - minPrice;
            const binSize = priceRange / numBins;

            const bins = Array(numBins).fill(0).map((_, i) => ({
                priceLevel: minPrice + (i + 0.5) * binSize,
                volume: 0
            }));

            // Distribute volume to bins
            data.forEach(bar => {
                const binIndex = Math.min(
                    Math.floor((bar.close - minPrice) / binSize),
                    numBins - 1
                );
                if (binIndex >= 0 && binIndex < numBins) {
                    bins[binIndex].volume += bar.volume;
                }
            });

            // Find POC (Point of Control - highest volume bin)
            const maxVolume = Math.max(...bins.map(b => b.volume));
            const pocIndex = bins.findIndex(b => b.volume === maxVolume);

            return {
                bins,
                poc: pocIndex >= 0 ? bins[pocIndex] : null,
                maxVolume,
                minPrice,
                maxPrice
            };
        }

        // Render Volume Profile using chart's coordinate system
        function renderVolumeProfile(profile) {
            const container = document.getElementById('volume-profile-container');
            container.innerHTML = '';

            if (!profile || profile.bins.length === 0 || !priceChart) return;

            const priceChartEl = document.getElementById('price-chart');
            const chartHeight = priceChartEl.clientHeight - 30; // Subtract time axis height

            // Get chart's visible price range
            const priceScale = priceChart.priceScale('right');

            profile.bins.forEach((bin, i) => {
                const bar = document.createElement('div');
                bar.className = 'volume-profile-bar';
                if (profile.poc && bin.priceLevel === profile.poc.priceLevel) {
                    bar.classList.add('poc');
                }

                // Calculate position based on price range in the data
                const priceRange = profile.maxPrice - profile.minPrice;
                if (priceRange === 0) return;

                const relativePrice = (bin.priceLevel - profile.minPrice) / priceRange;
                // Invert because CSS bottom is from bottom, chart price increases upward
                const topPosition = (1 - relativePrice) * chartHeight;
                const width = Math.max(5, (bin.volume / profile.maxVolume) * 100);

                bar.style.top = topPosition + 'px';
                bar.style.width = width + 'px';

                container.appendChild(bar);
            });
        }

        // Initialize charts
        function initCharts() {
            const priceContainer = document.getElementById('price-chart');
            const volumeContainer = document.getElementById('volume-chart');

            // Price chart
            priceChart = LightweightCharts.createChart(priceContainer, {
                layout: {
                    background: { type: 'solid', color: '#1a1a2e' },
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
            });

            candlestickSeries = priceChart.addCandlestickSeries({
                upColor: COLORS.upColor,
                downColor: COLORS.downColor,
                borderDownColor: COLORS.downColor,
                borderUpColor: COLORS.upColor,
                wickDownColor: COLORS.downColor,
                wickUpColor: COLORS.upColor,
            });

            // Create MA line series
            MA_PERIODS.forEach(period => {
                maSeries[period] = priceChart.addLineSeries({
                    color: MA_COLORS[period],
                    lineWidth: 1,
                    priceLineVisible: false,
                    lastValueVisible: false,
                    crosshairMarkerVisible: false,
                });
            });

            // Volume formatter for B/M/K
            const volumeFormatter = (value) => {
                if (value >= 1e9) {
                    return (value / 1e9).toFixed(1) + 'B';
                } else if (value >= 1e6) {
                    return (value / 1e6).toFixed(1) + 'M';
                } else if (value >= 1e3) {
                    return (value / 1e3).toFixed(0) + 'K';
                }
                return value.toFixed(0);
            };

            // Volume chart
            volumeChart = LightweightCharts.createChart(volumeContainer, {
                layout: {
                    background: { type: 'solid', color: '#1a1a2e' },
                    textColor: '#d1d4dc',
                },
                grid: {
                    vertLines: { color: '#2B2B43' },
                    horzLines: { color: '#2B2B43' },
                },
                rightPriceScale: {
                    borderColor: '#2B2B43',
                    scaleMargins: {
                        top: 0.1,
                        bottom: 0,
                    },
                },
                timeScale: {
                    borderColor: '#2B2B43',
                    visible: false,
                },
                localization: {
                    priceFormatter: volumeFormatter,
                },
            });

            volumeSeries = volumeChart.addHistogramSeries({
                priceFormat: {
                    type: 'custom',
                    formatter: volumeFormatter,
                },
                priceScaleId: 'right',
            });

            // Volume 50MA line
            volumeMaSeries = volumeChart.addLineSeries({
                color: '#ffd93d',
                lineWidth: 1,
                priceLineVisible: false,
                lastValueVisible: false,
                crosshairMarkerVisible: false,
                priceScaleId: 'right',
                priceFormat: {
                    type: 'custom',
                    formatter: volumeFormatter,
                },
            });

            // Sync time scales using logical range for smooth scrolling
            let isSyncing = false;

            priceChart.timeScale().subscribeVisibleLogicalRangeChange((logicalRange) => {
                if (isSyncing || !logicalRange) return;
                isSyncing = true;
                volumeChart.timeScale().setVisibleLogicalRange(logicalRange);
                isSyncing = false;
            });

            volumeChart.timeScale().subscribeVisibleLogicalRangeChange((logicalRange) => {
                if (isSyncing || !logicalRange) return;
                isSyncing = true;
                priceChart.timeScale().setVisibleLogicalRange(logicalRange);
                isSyncing = false;
            });

            // Also sync crosshair
            priceChart.subscribeCrosshairMove((param) => {
                if (!param || !param.time) {
                    volumeChart.clearCrosshairPosition();
                    return;
                }
                volumeChart.setCrosshairPosition(param.seriesData.get(candlestickSeries)?.close || 0, param.time, volumeSeries);
            });

            volumeChart.subscribeCrosshairMove((param) => {
                if (!param || !param.time) {
                    priceChart.clearCrosshairPosition();
                    return;
                }
                priceChart.setCrosshairPosition(param.seriesData.get(volumeSeries)?.value || 0, param.time, candlestickSeries);
            });

            // Handle resize
            window.addEventListener('resize', () => {
                priceChart.applyOptions({ width: priceContainer.clientWidth });
                volumeChart.applyOptions({ width: volumeContainer.clientWidth });
            });

            // Initial resize
            priceChart.applyOptions({ width: priceContainer.clientWidth });
            volumeChart.applyOptions({ width: volumeContainer.clientWidth });
        }

        // Load stock data into chart
        function loadStock(symbol) {
            if (!VCP_DATA[symbol]) return;

            currentSymbol = symbol;
            const data = VCP_DATA[symbol];

            // Update header
            document.getElementById('current-symbol').textContent = symbol;

            const badge = document.getElementById('alert-badge');
            badge.style.display = 'inline';
            badge.className = 'alert-badge ' + data.alert.type.replace('_', '-');
            badge.textContent = data.alert.type.replace('_', ' ').toUpperCase();

            // Update stats
            const hasPattern = data.alert.type !== 'none' && data.pattern.pivot_price !== null;
            document.getElementById('stat-score').textContent = hasPattern ? data.pattern.score.toFixed(0) : '-';
            document.getElementById('stat-distance').textContent = hasPattern && data.alert.distance_pct !== null
                ? (data.alert.distance_pct > 0 ? '+' : '') + data.alert.distance_pct.toFixed(1) + '%'
                : '-';
            document.getElementById('stat-price').textContent = '$' + data.alert.current_price.toFixed(2);
            document.getElementById('stat-pivot').textContent = hasPattern ? '$' + data.pattern.pivot_price.toFixed(2) : '-';
            document.getElementById('stat-support').textContent = hasPattern ? '$' + data.pattern.support_price.toFixed(2) : '-';

            // Set candlestick data
            candlestickSeries.setData(data.ohlcv);

            // Set volume data with colors
            const volumeData = data.ohlcv.map(d => ({
                time: d.time,
                value: d.volume,
                color: d.close >= d.open ? COLORS.upColor + '80' : COLORS.downColor + '80',
            }));
            volumeSeries.setData(volumeData);

            // Calculate and set Volume 50MA
            const vol50MaData = calculateVolumeSMA(data.ohlcv, 50);
            volumeMaSeries.setData(vol50MaData);

            // Calculate and set MA data
            MA_PERIODS.forEach(period => {
                const maData = calculateSMA(data.ohlcv, period);
                maSeries[period].setData(maData);

                // Apply visibility based on checkbox state
                const checkbox = document.getElementById(`ma-${period}`);
                if (checkbox) {
                    maSeries[period].applyOptions({
                        visible: checkbox.checked
                    });
                }
            });

            // Update volume profile if enabled
            const volProfileCheckbox = document.getElementById('vol-profile');
            if (volProfileCheckbox && volProfileCheckbox.checked) {
                const profile = calculateVolumeProfile(data.ohlcv);
                renderVolumeProfile(profile);
                document.getElementById('volume-profile-container').style.display = 'block';
            } else {
                document.getElementById('volume-profile-container').style.display = 'none';
            }

            // Remove old price lines
            if (pivotLine) {
                candlestickSeries.removePriceLine(pivotLine);
                pivotLine = null;
            }
            if (supportLine) {
                candlestickSeries.removePriceLine(supportLine);
                supportLine = null;
            }

            // Only add price lines and markers if stock has a valid pattern
            if (hasPattern) {
                // Add pivot line
                pivotLine = candlestickSeries.createPriceLine({
                    price: data.pattern.pivot_price,
                    color: COLORS.pivot,
                    lineWidth: 2,
                    lineStyle: LightweightCharts.LineStyle.Dashed,
                    axisLabelVisible: true,
                    title: 'Pivot',
                });

                // Add support line
                supportLine = candlestickSeries.createPriceLine({
                    price: data.pattern.support_price,
                    color: COLORS.support,
                    lineWidth: 2,
                    lineStyle: LightweightCharts.LineStyle.Dashed,
                    axisLabelVisible: true,
                    title: 'Support',
                });

                // Add markers for contraction highs/lows
                const markers = [];
                data.pattern.contractions.forEach((c, i) => {
                    const color = COLORS.contractionLines[i % COLORS.contractionLines.length];

                    // High marker
                    markers.push({
                        time: c.high_time,
                        position: 'aboveBar',
                        color: color,
                        shape: 'arrowDown',
                        text: 'C' + (i + 1) + ' High',
                    });

                    // Low marker
                    markers.push({
                        time: c.low_time,
                        position: 'belowBar',
                        color: color,
                        shape: 'arrowUp',
                        text: 'C' + (i + 1) + ' Low (' + c.range_pct.toFixed(1) + '%)',
                    });
                });

                candlestickSeries.setMarkers(markers);
            } else {
                // Clear markers for stocks without patterns
                candlestickSeries.setMarkers([]);
            }

            // Update contractions list
            const contractionsList = document.getElementById('contractions-list');
            contractionsList.innerHTML = data.pattern.contractions.map((c, i) => `
                <div class="contraction-item">
                    <div><span class="label">C${i + 1}:</span> <span class="value">${c.range_pct.toFixed(1)}%</span></div>
                    <div><span class="label">Duration:</span> <span class="value">${c.duration_days}d</span></div>
                    <div><span class="label">Volume:</span> <span class="value">${c.volume_ratio.toFixed(2)}x</span></div>
                </div>
            `).join('');

            // Update validity reasons
            const reasonsList = document.getElementById('validity-reasons');
            reasonsList.innerHTML = data.pattern.validity_reasons.map(r =>
                `<li>${r}</li>`
            ).join('');

            // Fit content
            priceChart.timeScale().fitContent();
            volumeChart.timeScale().fitContent();

            // Update active state in list
            document.querySelectorAll('.stock-item').forEach(el => {
                el.classList.toggle('active', el.dataset.symbol === symbol);
            });
        }

        // Populate stock list
        function populateStockList(filter = 'all', search = '') {
            const stockList = document.getElementById('stock-list');
            const symbols = Object.keys(VCP_DATA);

            // Filter and sort
            let filtered = symbols.filter(s => {
                const data = VCP_DATA[s];
                const matchesFilter = filter === 'all' || data.alert.type === filter;
                const matchesSearch = search === '' || s.toLowerCase().includes(search.toLowerCase());
                return matchesFilter && matchesSearch;
            });

            // Sort by score descending
            filtered.sort((a, b) => VCP_DATA[b].pattern.score - VCP_DATA[a].pattern.score);

            // Group by alert type
            const groups = {
                trade: filtered.filter(s => VCP_DATA[s].alert.type === 'trade'),
                pre_alert: filtered.filter(s => VCP_DATA[s].alert.type === 'pre_alert'),
                contraction: filtered.filter(s => VCP_DATA[s].alert.type === 'contraction'),
                none: filtered.filter(s => VCP_DATA[s].alert.type === 'none'),
            };

            let html = '';

            if (groups.trade.length > 0) {
                html += `
                    <div class="stock-group">
                        <div class="stock-group-header">🎯 TRADE ALERTS (${groups.trade.length})</div>
                        ${groups.trade.map(s => createStockItem(s)).join('')}
                    </div>
                `;
            }

            if (groups.pre_alert.length > 0) {
                html += `
                    <div class="stock-group">
                        <div class="stock-group-header">⚠️ PRE-ALERTS (${groups.pre_alert.length})</div>
                        ${groups.pre_alert.map(s => createStockItem(s)).join('')}
                    </div>
                `;
            }

            if (groups.contraction.length > 0) {
                html += `
                    <div class="stock-group">
                        <div class="stock-group-header">📊 CONTRACTION ALERTS (${groups.contraction.length})</div>
                        ${groups.contraction.map(s => createStockItem(s)).join('')}
                    </div>
                `;
            }

            if (groups.none.length > 0) {
                html += `
                    <div class="stock-group">
                        <div class="stock-group-header">📉 NO PATTERN (${groups.none.length})</div>
                        ${groups.none.map(s => createStockItem(s)).join('')}
                    </div>
                `;
            }

            stockList.innerHTML = html;

            // Add click handlers for stock items
            document.querySelectorAll('.stock-item').forEach(el => {
                el.addEventListener('click', () => loadStock(el.dataset.symbol));
            });

            // Add checkbox change handlers
            document.querySelectorAll('.stock-checkbox').forEach(cb => {
                cb.addEventListener('change', (e) => {
                    const symbol = e.target.dataset.symbol;
                    if (e.target.checked) {
                        selectedSymbols.add(symbol);
                    } else {
                        selectedSymbols.delete(symbol);
                    }
                    updateSelectionUI();
                });
            });
        }

        // Track selected symbols
        let selectedSymbols = new Set();

        function createStockItem(symbol) {
            const data = VCP_DATA[symbol];
            const alertClass = data.alert.type.replace('_', '-');
            const isActive = symbol === currentSymbol ? 'active' : '';
            const isChecked = selectedSymbols.has(symbol) ? 'checked' : '';
            return `
                <div class="stock-item ${alertClass} ${isActive}" data-symbol="${symbol}">
                    <input type="checkbox" class="stock-checkbox" data-symbol="${symbol}" ${isChecked} onclick="event.stopPropagation();">
                    <div class="stock-info">
                        <span class="stock-symbol">${symbol}</span>
                        <span class="stock-score">${data.pattern.score.toFixed(0)}</span>
                    </div>
                </div>
            `;
        }

        function updateSelectionUI() {
            const count = selectedSymbols.size;
            const btn = document.getElementById('copy-csv-btn');
            btn.textContent = '📋 Copy (' + count + ')';
            btn.disabled = count === 0;
        }

        // Filter buttons
        document.querySelectorAll('.filter-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                const filter = btn.dataset.filter;
                const search = document.getElementById('search-input').value;
                populateStockList(filter, search);
            });
        });

        // Search
        document.getElementById('search-input').addEventListener('input', (e) => {
            const activeFilter = document.querySelector('.filter-btn.active').dataset.filter;
            populateStockList(activeFilter, e.target.value);
        });

        // MA Toggle handlers
        MA_PERIODS.forEach(period => {
            const checkbox = document.getElementById(`ma-${period}`);
            if (checkbox) {
                // Set initial active state
                const label = checkbox.closest('.indicator-toggle');
                if (label) {
                    label.classList.toggle('active', checkbox.checked);
                }

                checkbox.addEventListener('change', () => {
                    const label = checkbox.closest('.indicator-toggle');
                    if (label) {
                        label.classList.toggle('active', checkbox.checked);
                    }
                    if (maSeries[period]) {
                        maSeries[period].applyOptions({
                            visible: checkbox.checked
                        });
                    }
                });
            }
        });

        // Volume Profile toggle handler
        const volProfileCheckbox = document.getElementById('vol-profile');
        if (volProfileCheckbox) {
            const label = volProfileCheckbox.closest('.indicator-toggle');
            if (label) {
                label.classList.toggle('active', volProfileCheckbox.checked);
            }

            volProfileCheckbox.addEventListener('change', () => {
                const label = volProfileCheckbox.closest('.indicator-toggle');
                if (label) {
                    label.classList.toggle('active', volProfileCheckbox.checked);
                }
                const container = document.getElementById('volume-profile-container');
                if (volProfileCheckbox.checked && currentSymbol && VCP_DATA[currentSymbol]) {
                    const profile = calculateVolumeProfile(VCP_DATA[currentSymbol].ohlcv);
                    renderVolumeProfile(profile);
                    container.style.display = 'block';
                } else {
                    container.style.display = 'none';
                }
            });
        }

        // Keyboard navigation
        document.addEventListener('keydown', (e) => {
            const items = document.querySelectorAll('.stock-item');
            if (items.length === 0) return;

            const currentIndex = Array.from(items).findIndex(el => el.dataset.symbol === currentSymbol);

            if (e.key === 'ArrowDown') {
                e.preventDefault();
                const nextIndex = currentIndex < items.length - 1 ? currentIndex + 1 : 0;
                loadStock(items[nextIndex].dataset.symbol);
                items[nextIndex].scrollIntoView({ block: 'nearest' });
            } else if (e.key === 'ArrowUp') {
                e.preventDefault();
                const prevIndex = currentIndex > 0 ? currentIndex - 1 : items.length - 1;
                loadStock(items[prevIndex].dataset.symbol);
                items[prevIndex].scrollIntoView({ block: 'nearest' });
            }
        });

        // Get currently visible (filtered) symbols
        function getFilteredSymbols() {
            const activeFilter = document.querySelector('.filter-btn.active').dataset.filter;
            const search = document.getElementById('search-input').value.toLowerCase();

            return Object.keys(VCP_DATA).filter(s => {
                const data = VCP_DATA[s];
                const matchesFilter = activeFilter === 'all' || data.alert.type === activeFilter;
                const matchesSearch = search === '' || s.toLowerCase().includes(search);
                return matchesFilter && matchesSearch;
            }).sort((a, b) => VCP_DATA[b].pattern.score - VCP_DATA[a].pattern.score);
        }

        // Select All - selects all currently visible/filtered stocks
        document.getElementById('select-all-btn').addEventListener('click', () => {
            const filtered = getFilteredSymbols();
            filtered.forEach(s => selectedSymbols.add(s));
            // Update checkboxes
            document.querySelectorAll('.stock-checkbox').forEach(cb => {
                if (filtered.includes(cb.dataset.symbol)) {
                    cb.checked = true;
                }
            });
            updateSelectionUI();
        });

        // Clear selection
        document.getElementById('unselect-all-btn').addEventListener('click', () => {
            selectedSymbols.clear();
            document.querySelectorAll('.stock-checkbox').forEach(cb => {
                cb.checked = false;
            });
            updateSelectionUI();
        });

        // Copy selected symbols
        document.getElementById('copy-csv-btn').addEventListener('click', async () => {
            const btn = document.getElementById('copy-csv-btn');
            if (selectedSymbols.size === 0) return;

            // Sort selected symbols by score
            const sortedSymbols = Array.from(selectedSymbols).sort(
                (a, b) => VCP_DATA[b].pattern.score - VCP_DATA[a].pattern.score
            );
            const symbolList = sortedSymbols.join(', ');
            const count = selectedSymbols.size;

            try {
                await navigator.clipboard.writeText(symbolList);
                btn.textContent = '✓ Copied!';
                btn.classList.add('copied');

                setTimeout(() => {
                    btn.textContent = '📋 Copy (' + count + ')';
                    btn.classList.remove('copied');
                }, 2000);
            } catch (err) {
                // Fallback for older browsers
                const textarea = document.createElement('textarea');
                textarea.value = symbolList;
                document.body.appendChild(textarea);
                textarea.select();
                document.execCommand('copy');
                document.body.removeChild(textarea);

                btn.textContent = '✓ Copied!';
                btn.classList.add('copied');

                setTimeout(() => {
                    btn.textContent = '📋 Copy (' + count + ')';
                    btn.classList.remove('copied');
                }, 2000);
            }
        });

        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            initCharts();
            populateStockList();

            // Load first stock
            const firstSymbol = Object.keys(VCP_DATA)[0];
            if (firstSymbol) {
                loadStock(firstSymbol);
            }
        });
    </script>
</body>
</html>
'''


class LightweightChartGenerator:
    """
    Generates interactive HTML dashboards using TradingView Lightweight Charts.

    Features:
    - Multi-stock selector with filtering
    - Interactive candlestick + volume charts
    - VCP contraction visualization with markers
    - Pivot/support level lines
    - All data preloaded for fast switching
    """

    def __init__(self, output_dir: str = "charts"):
        """
        Initialize the chart generator.

        Args:
            output_dir: Directory to save generated HTML files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_dashboard(
        self,
        scan_results: List[Dict[str, Any]],
        filename: str = "vcp_dashboard.html",
    ) -> Path:
        """
        Generate an interactive dashboard HTML file with all stock data.

        Args:
            scan_results: List of dicts with 'symbol', 'df', 'pattern', 'alerts' keys
            filename: Output filename

        Returns:
            Path to generated HTML file
        """
        # Prepare data for embedding
        vcp_data = {}
        trade_count = 0
        prealert_count = 0
        contraction_count = 0
        no_alert_count = 0

        for result in scan_results:
            symbol = result["symbol"]
            df = result["df"]
            pattern = result.get("pattern")
            alerts = result.get("alerts", [])

            # Get current price
            current_price = float(df["Close"].iloc[-1])

            # Convert OHLCV data (always available)
            ohlcv = self._convert_ohlcv(df)

            # Handle stocks without patterns
            if pattern is None:
                no_alert_count += 1
                reason = result.get("reason", "no pattern")
                vcp_data[symbol] = {
                    "ohlcv": ohlcv,
                    "pattern": {
                        "contractions": [],
                        "pivot_price": None,
                        "support_price": None,
                        "score": 0,
                        "validity_reasons": [reason],
                    },
                    "alert": {
                        "type": "none",
                        "distance_pct": None,
                        "current_price": current_price,
                    },
                }
                continue

            # Determine alert type for stocks with patterns
            if alerts:
                alert = alerts[0]
                alert_type = alert.alert_type.value
                distance_pct = alert.distance_to_pivot_pct
            else:
                # Infer from pattern
                distance_pct = ((pattern.pivot_price - current_price) / current_price) * 100
                if distance_pct <= 0:
                    alert_type = "trade"
                elif distance_pct <= 3.0:
                    alert_type = "pre_alert"
                else:
                    alert_type = "contraction"

            # Count by type
            if alert_type == "trade":
                trade_count += 1
            elif alert_type == "pre_alert":
                prealert_count += 1
            else:
                contraction_count += 1

            # Convert pattern data
            contractions = self._convert_contractions(pattern)

            vcp_data[symbol] = {
                "ohlcv": ohlcv,
                "pattern": {
                    "contractions": contractions,
                    "pivot_price": float(pattern.pivot_price),
                    "support_price": float(pattern.support_price),
                    "score": float(pattern.proximity_score),
                    "validity_reasons": pattern.validity_reasons[:5],  # Limit to 5
                },
                "alert": {
                    "type": alert_type,
                    "distance_pct": float(distance_pct),
                    "current_price": current_price,
                },
            }

        # Sort by score (stocks with patterns first, then no-pattern stocks)
        vcp_data = dict(sorted(
            vcp_data.items(),
            key=lambda x: (x[1]["alert"]["type"] != "none", x[1]["pattern"]["score"]),
            reverse=True
        ))

        # Generate HTML
        html = HTML_TEMPLATE.replace("{{vcp_data_json}}", json.dumps(vcp_data, indent=2))
        html = html.replace("{{total_count}}", str(len(vcp_data)))
        html = html.replace("{{none_count}}", str(no_alert_count))
        html = html.replace("{{trade_count}}", str(trade_count))
        html = html.replace("{{prealert_count}}", str(prealert_count))
        html = html.replace("{{contraction_count}}", str(contraction_count))
        html = html.replace("{{generation_date}}", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        # Save file
        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            f.write(html)

        return output_path

    def _convert_ohlcv(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Convert DataFrame to Lightweight Charts format."""
        ohlcv = []
        for idx, row in df.iterrows():
            # Convert timestamp to string format
            if hasattr(idx, 'strftime'):
                time_str = idx.strftime("%Y-%m-%d")
            else:
                time_str = str(idx)

            ohlcv.append({
                "time": time_str,
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": float(row["Volume"]),
            })

        return ohlcv

    def _convert_contractions(self, pattern: VCPPattern) -> List[Dict[str, Any]]:
        """Convert contractions to JavaScript-friendly format."""
        contractions = []
        for c in pattern.contractions:
            # Convert dates
            high_time = c.swing_high.date
            low_time = c.swing_low.date

            if hasattr(high_time, 'strftime'):
                high_time = high_time.strftime("%Y-%m-%d")
            else:
                high_time = str(high_time)[:10]

            if hasattr(low_time, 'strftime'):
                low_time = low_time.strftime("%Y-%m-%d")
            else:
                low_time = str(low_time)[:10]

            contractions.append({
                "high_time": high_time,
                "high_price": float(c.swing_high.price),
                "low_time": low_time,
                "low_price": float(c.swing_low.price),
                "range_pct": float(c.range_pct),
                "duration_days": int(c.duration_days),
                "volume_ratio": float(c.avg_volume_ratio),
            })

        return contractions
