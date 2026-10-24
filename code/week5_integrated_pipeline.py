"""
Week 5 - Integrated Analytics Pipeline and Executive Summary
YuvaIntern | Virtual Data Science with Python | Summer Training 2026
Author  : Arpit Vats

Purpose : Run Weeks 1-4 as one end-to-end pipeline (clean -> explore -> visualise -> test ->
          predict), collect their headline results into a single summary table, and produce an
          executive dashboard. Nothing is re-implemented here, so the numbers always match the
          weekly scripts.

HOW TO RUN: keep the five week*.py files in the same folder and run this script. The weekly
scripts' own printouts are suppressed; only the consolidated summary is shown.

DATA NOTE: All data are SYNTHETIC (NumPy, seed = 42), so the findings demonstrate the method,
not facts about real football players.

Output  : week5_executive_dashboard.png (the weekly PNG figures are regenerated as well)
"""

import io
import runpy
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    BASE = Path(__file__).resolve().parent
except NameError:               # running inside Jupyter / Colab
    BASE = Path.cwd()


def run_phase(filename):
    """Execute one weekly script silently and return its variables."""
    with redirect_stdout(io.StringIO()):
        return runpy.run_path(str(BASE / filename))


# ----------------------------------------------------------------------------
# STEP 1: RUN THE FOUR ANALYSIS PHASES
# ----------------------------------------------------------------------------
w1 = run_phase('week1_data_preprocessing.py')
w2 = run_phase('week2_visual_storytelling.py')
w3 = run_phase('week3_hypothesis_testing.py')
w4 = run_phase('week4_injury_risk_ml.py')

lr = w4['results']['Logistic Regression']
rf = w4['results']['Random Forest']

# ----------------------------------------------------------------------------
# STEP 2: CONSOLIDATED RESULTS TABLE
# ----------------------------------------------------------------------------
summary = pd.DataFrame([
    ('1. Preprocessing', 'Rows: raw -> unique -> clean',
     f"{w1['raw_rows']} -> {w1['unique_rows']} -> {w1['clean_rows']}"),
    ('1. Preprocessing', 'Duplicates / zero-match rows removed',
     f"{w1['duplicate_count']} / {w1['zero_match_rows']}"),
    ('1. Preprocessing', 'Correlation of goals / age with market value',
     f"{w1['corr_goals_value']:.2f} / {w1['corr_age_value']:.2f}"),
    ('2. Visual storytelling', 'Mean efficiency range across age groups',
     f"{w2['age_summary']['Mean_Efficiency'].min():.2f} - {w2['age_summary']['Mean_Efficiency'].max():.2f}"),
    ('2. Visual storytelling', 'Players aged 29 / 31 / 33',
     " / ".join(str(int(n)) for n in w2['age_summary'].loc[w2['age_summary']['Age'] >= 29, 'Players'])),
    ('3. Hypothesis testing', 'Goal contributions per 90: normal vs congested rest',
     f"{w3['normal_rest_output'].mean():.2f} vs {w3['congested_rest_output'].mean():.2f}"),
    ('3. Hypothesis testing', 'Welch t-test (one-sided) / Mann-Whitney p-values',
     f"{w3['t_pval_one']:.4f} / {w3['u_pval']:.4f}"),
    ('3. Hypothesis testing', 'Chi-square (position vs injury risk)',
     f"chi2 = {w3['chi2_stat']:.2f}, p < 0.0001, V = {w3['cramers_v']:.2f}"),
    ('4. Machine learning', 'Logistic Regression: AUC / recall / precision',
     f"{lr['ROC-AUC']:.3f} / {lr['Recall']:.1%} / {lr['Precision']:.1%}"),
    ('4. Machine learning', 'Random Forest: AUC / recall / precision',
     f"{rf['ROC-AUC']:.3f} / {rf['Recall']:.1%} / {rf['Precision']:.1%}"),
], columns=['Phase', 'Metric', 'Result'])

print("=" * 78)
print("CONSOLIDATED RESULTS - SPORTS ANALYTICS CAPSTONE")
print("=" * 78)
print(summary.to_string(index=False))

print("\nRECOMMENDATIONS (to be validated on real club data before use)")
print("1. Rotation protocol: limit forwards' minutes in congested fixture spells "
      f"(high-risk share: forwards {w3['high_risk_share']['Forward']:.0%}, "
      f"midfielders {w3['high_risk_share']['Midfielder']:.0%}, defenders {w3['high_risk_share']['Defender']:.0%}).")
print("2. Load management: use the Logistic Regression model as a daily screening tool and "
      "tune the decision threshold to raise recall.")
print("3. Evidence before spending: market value showed no age or goal-count pattern, so rely on "
      "validated efficiency metrics rather than age or raw totals.")

# ----------------------------------------------------------------------------
# STEP 3: EXECUTIVE DASHBOARD (2 x 2)
# ----------------------------------------------------------------------------
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# Panel A: data funnel
ax = axes[0, 0]
stages = ['Raw rows', 'After de-duplication', 'After cleaning']
counts = [w1['raw_rows'], w1['unique_rows'], w1['clean_rows']]
bars = ax.bar(stages, counts, color=['#9aa5b1', '#6c8ebf', '#2b5c8f'])
ax.bar_label(bars, padding=3)
ax.set_ylim(0, max(counts) * 1.12)
ax.set_title('A. Data preparation funnel (Week 1)')
ax.set_ylabel('Number of player records')

# Panel B: efficiency by age
ax = axes[0, 1]
age_summary = w2['age_summary']
x_pos = np.arange(len(age_summary))
ax.plot(x_pos, age_summary['Mean_Efficiency'], marker='o', linewidth=3, color='#2b5c8f')
ax.set_xticks(x_pos)
ax.set_xticklabels(age_summary['Age'])
ax.set_ylim(0.4, 0.75)
ax.set_xlabel('Player age')
ax.set_ylabel('Mean contributions per match')
ax.set_title('B. Efficiency by age group (Week 2)')

# Panel C: high-risk share by position
ax = axes[1, 0]
share = w3['high_risk_share'].sort_values(ascending=False)
bars = ax.bar(share.index, share.values * 100, color=['#c0504d', '#e28743', '#9aa5b1'])
ax.bar_label(bars, labels=[f"{v:.0%}" for v in share.values], padding=3)
ax.set_ylim(0, 100)
ax.set_ylabel('High-risk players (%)')
ax.set_title(f"C. Injury risk by position (Week 3, chi2 = {w3['chi2_stat']:.1f})")

# Panel D: ROC curves
ax = axes[1, 1]
ax.plot(w4['fpr_lr'], w4['tpr_lr'], color='blue', lw=2, label=f"Logistic Regression (AUC = {lr['ROC-AUC']:.3f})")
ax.plot(w4['fpr_rf'], w4['tpr_rf'], color='green', lw=2, label=f"Random Forest (AUC = {rf['ROC-AUC']:.3f})")
ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
ax.set_xlabel('False positive rate')
ax.set_ylabel('True positive rate')
ax.set_title('D. Injury-risk models (Week 4)')
ax.legend(loc='lower right')

fig.suptitle('Sports Analytics Capstone - Executive Dashboard (synthetic data)', fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig('week5_executive_dashboard.png', dpi=300)
plt.close(fig)

print("\nPipeline complete! Dashboard saved as week5_executive_dashboard.png")
