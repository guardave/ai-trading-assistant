#!/usr/bin/env python3
"""Generate backtest result charts as PNG files."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

# Output directory
OUTPUT_DIR = 'charts'

print("Loading data...")

# Load Phase 1 results
phase1_summary = pd.read_csv('results/phase1/phase1_baseline_results.csv')
phase1_trades = pd.read_csv('results/phase1/phase1_all_trades.csv')

# Load Phase 3 results
phase3_rs = pd.read_csv('results/phase3/phase3_rs_comparison.csv')
phase3_prox = pd.read_csv('results/phase3/phase3_proximity_impact.csv')

# Parse Phase 1 summary
p1 = dict(zip(phase1_summary['metric'], phase1_summary['value']))

print("Generating charts...")

# =============================================================================
# Chart 1: Phase 1 Baseline Comparison
# =============================================================================
phase1_table = pd.DataFrame([
    {
        'Configuration': 'Fixed Target',
        'Win Rate %': float(p1['baseline_fixed_win_rate']),
        'Profit Factor': float(p1['baseline_fixed_profit_factor']),
        'Total Return %': float(p1['baseline_fixed_total_return_pct']),
    },
    {
        'Configuration': 'Trailing Stop',
        'Win Rate %': float(p1['baseline_trailing_win_rate']),
        'Profit Factor': float(p1['baseline_trailing_profit_factor']),
        'Total Return %': float(p1['baseline_trailing_total_return_pct']),
    },
    {
        'Configuration': 'Adjusted R:R',
        'Win Rate %': float(p1['adjusted_rr_win_rate']),
        'Profit Factor': float(p1['adjusted_rr_profit_factor']),
        'Total Return %': float(p1['adjusted_rr_total_return_pct']),
    }
])

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

configs = ['Fixed\nTarget', 'Trailing\nStop', 'Adjusted\nR:R']

# Win Rate
win_rates = phase1_table['Win Rate %'].values
bar_colors = ['#27ae60' if w >= 50 else '#e74c3c' for w in win_rates]
axes[0].bar(configs, win_rates, color=bar_colors, edgecolor='black')
axes[0].axhline(y=50, color='black', linestyle='--', alpha=0.5)
axes[0].set_ylabel('Win Rate %')
axes[0].set_title('Win Rate by Configuration')
axes[0].set_ylim(0, 60)

# Profit Factor
pf_values = phase1_table['Profit Factor'].values
bar_colors = ['#27ae60' if pf >= 1.0 else '#e74c3c' for pf in pf_values]
axes[1].bar(configs, pf_values, color=bar_colors, edgecolor='black')
axes[1].axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
axes[1].set_ylabel('Profit Factor')
axes[1].set_title('Profit Factor by Configuration')
axes[1].set_ylim(0, 1.5)

# Total Return
returns = phase1_table['Total Return %'].values
bar_colors = ['#27ae60' if r >= 0 else '#e74c3c' for r in returns]
axes[2].bar(configs, returns, color=bar_colors, edgecolor='black')
axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[2].set_ylabel('Total Return %')
axes[2].set_title('Total Return by Configuration')

plt.suptitle('Phase 1: Baseline Results (RS 90) - All Configurations Unprofitable', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/01_phase1_baseline.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 01_phase1_baseline.png")

# =============================================================================
# Chart 2: Phase 1 Proximity Analysis
# =============================================================================
baseline_trades = phase1_trades[phase1_trades['config'] == 'baseline_trailing'].copy()
baseline_trades['win'] = baseline_trades['pnl_pct'] > 0

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Proximity vs P&L scatter
colors = ['#27ae60' if w else '#e74c3c' for w in baseline_trades['win']]
axes[0].scatter(baseline_trades['proximity_score'], baseline_trades['pnl_pct'],
                c=colors, alpha=0.7, s=80, edgecolor='black')
axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[0].set_xlabel('Proximity Score')
axes[0].set_ylabel('P&L %')
axes[0].set_title('Proximity Score vs P&L')

# Win rate by proximity bucket
baseline_trades['prox_bucket'] = pd.cut(baseline_trades['proximity_score'],
                                         bins=[0, 30, 50, 70, 100],
                                         labels=['0-30', '30-50', '50-70', '70-100'])
win_by_prox = baseline_trades.groupby('prox_bucket', observed=True)['win'].agg(['mean', 'count'])
win_by_prox['win_rate'] = win_by_prox['mean'] * 100

bar_colors = ['#27ae60' if w >= 50 else '#e74c3c' for w in win_by_prox['win_rate'].fillna(0)]
axes[1].bar(win_by_prox.index.astype(str), win_by_prox['win_rate'].fillna(0), color=bar_colors, edgecolor='black')
axes[1].axhline(y=50, color='black', linestyle='--', alpha=0.5)
axes[1].set_xlabel('Proximity Bucket')
axes[1].set_ylabel('Win Rate %')
axes[1].set_title('Win Rate by Proximity Bucket')

for i, (idx, row) in enumerate(win_by_prox.iterrows()):
    if not pd.isna(row['win_rate']):
        axes[1].text(i, row['win_rate'] + 2, f'n={int(row["count"])}', ha='center', fontsize=9)

plt.suptitle('Phase 1: Proximity Score Validation - Higher Proximity = Better Results', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_phase1_proximity.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 02_phase1_proximity.png")

# =============================================================================
# Chart 3: Phase 3 RS Threshold Comparison
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

prox_filtered = phase3_rs[phase3_rs['proximity_filter'] == 'min_50'].copy()
no_filter = phase3_rs[phase3_rs['proximity_filter'] == 'none'].copy()

# Trade Count
axes[0, 0].plot(no_filter['rs_threshold'], no_filter['total_trades'], 'o-',
                label='No Filter', linewidth=2, markersize=8)
axes[0, 0].plot(prox_filtered['rs_threshold'], prox_filtered['total_trades'], 's-',
                label='Prox >= 50', linewidth=2, markersize=8)
axes[0, 0].set_xlabel('RS Threshold')
axes[0, 0].set_ylabel('Number of Trades')
axes[0, 0].set_title('Trade Count vs RS Threshold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Win Rate
axes[0, 1].plot(no_filter['rs_threshold'], no_filter['win_rate'], 'o-',
                label='No Filter', linewidth=2, markersize=8)
axes[0, 1].plot(prox_filtered['rs_threshold'], prox_filtered['win_rate'], 's-',
                label='Prox >= 50', linewidth=2, markersize=8)
axes[0, 1].axhline(y=50, color='black', linestyle='--', alpha=0.5)
axes[0, 1].set_xlabel('RS Threshold')
axes[0, 1].set_ylabel('Win Rate %')
axes[0, 1].set_title('Win Rate vs RS Threshold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Profit Factor
axes[1, 0].plot(no_filter['rs_threshold'], no_filter['profit_factor'], 'o-',
                label='No Filter', linewidth=2, markersize=8)
axes[1, 0].plot(prox_filtered['rs_threshold'], prox_filtered['profit_factor'], 's-',
                label='Prox >= 50', linewidth=2, markersize=8)
axes[1, 0].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Breakeven')
axes[1, 0].axhline(y=1.3, color='green', linestyle='--', alpha=0.7, label='Target (1.3)')
axes[1, 0].set_xlabel('RS Threshold')
axes[1, 0].set_ylabel('Profit Factor')
axes[1, 0].set_title('Profit Factor vs RS Threshold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Total Return
axes[1, 1].plot(no_filter['rs_threshold'], no_filter['total_return_pct'], 'o-',
                label='No Filter', linewidth=2, markersize=8)
axes[1, 1].plot(prox_filtered['rs_threshold'], prox_filtered['total_return_pct'], 's-',
                label='Prox >= 50', linewidth=2, markersize=8)
axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[1, 1].set_xlabel('RS Threshold')
axes[1, 1].set_ylabel('Total Return %')
axes[1, 1].set_title('Total Return vs RS Threshold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Phase 3: RS Threshold Impact - RS 70 is Optimal', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/03_phase3_rs_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 03_phase3_rs_comparison.png")

# =============================================================================
# Chart 4: Proximity Filter Impact
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

x = np.arange(len(phase3_prox))
width = 0.35

# Profit Factor comparison
bars1 = axes[0].bar(x - width/2, phase3_prox['profit_factor_no_prox'], width,
                     label='No Filter', color='#3498db', edgecolor='black')
bars2 = axes[0].bar(x + width/2, phase3_prox['profit_factor_with_prox'], width,
                     label='Prox >= 50', color='#27ae60', edgecolor='black')
axes[0].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Breakeven')
axes[0].set_xlabel('RS Threshold')
axes[0].set_ylabel('Profit Factor')
axes[0].set_title('Profit Factor: Filter vs No Filter')
axes[0].set_xticks(x)
axes[0].set_xticklabels(phase3_prox['rs_threshold'])
axes[0].legend()

# Profit Factor Change
pf_changes = phase3_prox['profit_factor_change'].str.replace('+', '').astype(float)
colors = ['#27ae60' if c > 0 else '#e74c3c' for c in pf_changes]
axes[1].bar(phase3_prox['rs_threshold'].astype(str), pf_changes, color=colors, edgecolor='black')
axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[1].set_xlabel('RS Threshold')
axes[1].set_ylabel('Profit Factor Change')
axes[1].set_title('Profit Factor Improvement from Proximity Filter')

plt.suptitle('Proximity Filter Improves Profit Factor at ALL RS Levels', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/04_proximity_filter_impact.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 04_proximity_filter_impact.png")

# =============================================================================
# Chart 5: Phase 1 vs Phase 3 Transformation
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

phase1_best = {'Trades': 34, 'Profit Factor': 0.88, 'Return': -14.85}
phase3_best = {'Trades': 62, 'Profit Factor': 1.31, 'Return': 65.26}

labels = ['Phase 1\n(RS 90)', 'Phase 3\n(RS 70 + Prox)']

# Trades
trades = [phase1_best['Trades'], phase3_best['Trades']]
axes[0].bar(labels, trades, color=['#e74c3c', '#27ae60'], edgecolor='black')
axes[0].set_ylabel('Number of Trades')
axes[0].set_title('Trade Count')
for i, v in enumerate(trades):
    axes[0].text(i, v + 2, str(v), ha='center', fontweight='bold')

# Profit Factor
pf = [phase1_best['Profit Factor'], phase3_best['Profit Factor']]
colors = ['#e74c3c' if p < 1 else '#27ae60' for p in pf]
axes[1].bar(labels, pf, color=colors, edgecolor='black')
axes[1].axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
axes[1].set_ylabel('Profit Factor')
axes[1].set_title('Profit Factor')
for i, v in enumerate(pf):
    axes[1].text(i, v + 0.05, f'{v:.2f}', ha='center', fontweight='bold')

# Return
returns = [phase1_best['Return'], phase3_best['Return']]
colors = ['#e74c3c' if r < 0 else '#27ae60' for r in returns]
axes[2].bar(labels, returns, color=colors, edgecolor='black')
axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[2].set_ylabel('Total Return %')
axes[2].set_title('Total Return')
for i, v in enumerate(returns):
    axes[2].text(i, v + (5 if v > 0 else -8), f'{v:.1f}%', ha='center', fontweight='bold')

plt.suptitle('Transformation: Unprofitable → Profitable Strategy (+80% Return Improvement)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/05_transformation.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 05_transformation.png")

# =============================================================================
# Chart 6: Summary Dashboard
# =============================================================================
fig = plt.figure(figsize=(16, 10))

# Create grid
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# Top row - RS comparison
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(no_filter['rs_threshold'], no_filter['profit_factor'], 'o-',
         label='No Filter', linewidth=2, markersize=10)
ax1.plot(prox_filtered['rs_threshold'], prox_filtered['profit_factor'], 's-',
         label='Prox >= 50', linewidth=2, markersize=10)
ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
ax1.axhline(y=1.3, color='green', linestyle='--', alpha=0.7)
ax1.fill_between([69, 71], 0, 1.5, alpha=0.2, color='green', label='Optimal Zone')
ax1.set_xlabel('RS Threshold', fontsize=12)
ax1.set_ylabel('Profit Factor', fontsize=12)
ax1.set_title('Profit Factor by RS Threshold', fontsize=14, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0.8, 1.4)

# Top right - Key metrics
ax2 = fig.add_subplot(gs[0, 2])
ax2.axis('off')
metrics_text = """
OPTIMAL CONFIGURATION
━━━━━━━━━━━━━━━━━━━━━━
RS Threshold:    70
Proximity Min:   50
Exit Method:     Trailing Stop

PERFORMANCE
━━━━━━━━━━━━━━━━━━━━━━
Trades:          62
Win Rate:        50.8%
Profit Factor:   1.31
Total Return:    +65.3%
Avg Days Held:   25.2

STATUS: PROFITABLE ✓
"""
ax2.text(0.1, 0.9, metrics_text, transform=ax2.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

# Bottom left - Transformation
ax3 = fig.add_subplot(gs[1, 0])
labels = ['Phase 1\n(RS 90)', 'Phase 3\n(RS 70+Prox)']
returns = [-14.85, 65.26]
colors = ['#e74c3c', '#27ae60']
bars = ax3.bar(labels, returns, color=colors, edgecolor='black')
ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax3.set_ylabel('Total Return %')
ax3.set_title('Return Transformation', fontweight='bold')
for bar, val in zip(bars, returns):
    ax3.text(bar.get_x() + bar.get_width()/2, val + (3 if val > 0 else -6),
             f'{val:.1f}%', ha='center', fontweight='bold')

# Bottom middle - Proximity impact
ax4 = fig.add_subplot(gs[1, 1])
pf_no = phase3_prox[phase3_prox['rs_threshold'] == 70]['profit_factor_no_prox'].values[0]
pf_with = phase3_prox[phase3_prox['rs_threshold'] == 70]['profit_factor_with_prox'].values[0]
bars = ax4.bar(['Without\nFilter', 'With Prox\n>= 50'], [pf_no, pf_with],
               color=['#3498db', '#27ae60'], edgecolor='black')
ax4.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
ax4.set_ylabel('Profit Factor')
ax4.set_title('Proximity Filter Impact (RS 70)', fontweight='bold')
ax4.set_ylim(0, 1.5)
for bar, val in zip(bars, [pf_no, pf_with]):
    ax4.text(bar.get_x() + bar.get_width()/2, val + 0.03, f'{val:.2f}',
             ha='center', fontweight='bold')

# Bottom right - Trade distribution
ax5 = fig.add_subplot(gs[1, 2])
rs_trades = prox_filtered.set_index('rs_threshold')['total_trades']
colors = ['#27ae60' if rs == 70 else '#3498db' for rs in rs_trades.index]
bars = ax5.bar(rs_trades.index.astype(str), rs_trades.values, color=colors, edgecolor='black')
ax5.set_xlabel('RS Threshold')
ax5.set_ylabel('Number of Trades')
ax5.set_title('Trade Count (Prox >= 50)', fontweight='bold')
for bar, val in zip(bars, rs_trades.values):
    ax5.text(bar.get_x() + bar.get_width()/2, val + 1, str(int(val)),
             ha='center', fontsize=9)

plt.suptitle('VCP Strategy Backtest Results Dashboard', fontsize=16, fontweight='bold', y=0.98)
plt.savefig(f'{OUTPUT_DIR}/06_summary_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 06_summary_dashboard.png")

print(f"\nAll charts saved to {OUTPUT_DIR}/")
print("Charts generated:")
print("  1. Phase 1 Baseline Comparison")
print("  2. Phase 1 Proximity Analysis")
print("  3. Phase 3 RS Threshold Comparison")
print("  4. Proximity Filter Impact")
print("  5. Phase 1 to Phase 3 Transformation")
print("  6. Summary Dashboard")
