"""
Week 3 - Hypothesis Testing on Fixture Congestion and Injury Risk
YuvaIntern | Virtual Data Science with Python | Summer Training 2026
Author  : Arpit Vats

Test 1  : Does attacking output fall when rest between matches is short?
            H0: mean goal contributions per 90 are equal (normal rest = congested rest)
            H1: mean under NORMAL rest (>= 4 days) is HIGHER than under CONGESTED rest (< 4 days)
Test 2  : Is high injury risk associated with playing position?
            H0: injury risk and position are independent
            H1: injury risk depends on position
Alpha   : 0.05

DATA NOTE: Both datasets are SYNTHETIC (NumPy, seed = 42). The congestion effect and the
position effect were built into the simulation, so these results demonstrate the testing
workflow rather than facts about real players.

Outputs : stat_1_density_distribution.png, stat_2_boxplot_intervals.png, stat_3_chi_heatmap.png
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
np.random.seed(42)

ALPHA = 0.05

# ----------------------------------------------------------------------------
# STEP 1: GENERATE DATASETS
# ----------------------------------------------------------------------------
n_samples = 500

# Group 1: normal rest (>= 4 days) - right-skewed output
normal_rest_output = np.random.gamma(shape=2.5, scale=0.25, size=n_samples)
# Group 2: congested rest (< 4 days) - slightly lower output simulated
congested_rest_output = np.random.gamma(shape=2.1, scale=0.25, size=n_samples)

df_stats = pd.DataFrame({
    'Rest_Condition': ['Normal Rest'] * n_samples + ['Congested Rest'] * n_samples,
    'Goal_Contributions_Per_90': np.concatenate([normal_rest_output, congested_rest_output]),
})

# Categorical data for the Chi-square test
positions = ['Forward', 'Midfielder', 'Defender']
injury_risk = ['Low Risk', 'High Risk']
df_chi = pd.DataFrame({
    'Position': np.random.choice(positions, size=600, p=[0.35, 0.40, 0.25]),
    'Injury_Status': np.random.choice(injury_risk, size=600, p=[0.70, 0.30]),
})
# Built-in dependency: forwards carry extra fatigue risk
df_chi.loc[(df_chi['Position'] == 'Forward') & (np.random.rand(len(df_chi)) > 0.4),
           'Injury_Status'] = 'High Risk'

# ----------------------------------------------------------------------------
# STEP 2: STATISTICAL TESTING PIPELINE
# ----------------------------------------------------------------------------
# A. Normality (Shapiro-Wilk) for each group
stat_norm_a, p_norm_a = stats.shapiro(normal_rest_output)
stat_norm_b, p_norm_b = stats.shapiro(congested_rest_output)

# B. Equality of variances (Levene)
stat_lev, p_lev = stats.levene(normal_rest_output, congested_rest_output)

# C. Welch's t-test (does not assume equal variances).
#    CORRECTION: the original script reported only the default TWO-sided p-value although H1
#    is one-sided (normal > congested). Both versions are reported here.
t_stat, t_pval = stats.ttest_ind(normal_rest_output, congested_rest_output, equal_var=False)
_, t_pval_one = stats.ttest_ind(normal_rest_output, congested_rest_output, equal_var=False,
                                alternative='greater')

# Effect size (Cohen's d with pooled standard deviation)
pooled_sd = np.sqrt((normal_rest_output.var(ddof=1) + congested_rest_output.var(ddof=1)) / 2)
cohens_d = (normal_rest_output.mean() - congested_rest_output.mean()) / pooled_sd

# D. Mann-Whitney U (non-parametric; the data are right-skewed). One-sided: normal > congested
u_stat, u_pval = stats.mannwhitneyu(normal_rest_output, congested_rest_output, alternative='greater')

# E. Chi-square test of independence (position vs injury risk)
contingency_table = pd.crosstab(df_chi['Position'], df_chi['Injury_Status'])
chi2_stat, chi2_pval, dof, expected = stats.chi2_contingency(contingency_table)
cramers_v = np.sqrt(chi2_stat / (contingency_table.values.sum() * (min(contingency_table.shape) - 1)))
high_risk_share = contingency_table['High Risk'] / contingency_table.sum(axis=1)

def decision(p):
    return "Reject H0" if p < ALPHA else "Fail to reject H0"

print("--- STATISTICAL EXECUTION RESULTS (alpha = 0.05) ---")
print(f"Mean goal contributions per 90 : normal rest = {normal_rest_output.mean():.3f}, "
      f"congested rest = {congested_rest_output.mean():.3f}")
print(f"Shapiro-Wilk  : normal W={stat_norm_a:.3f}, p={p_norm_a:.2e} | "
      f"congested W={stat_norm_b:.3f}, p={p_norm_b:.2e}  -> data are not normal")
print(f"Levene        : F={stat_lev:.3f}, p={p_lev:.4f}  -> equal variances retained")
print(f"Welch t-test  : t={t_stat:.3f}, p(two-sided)={t_pval:.4f}, "
      f"p(one-sided)={t_pval_one:.4f}  -> {decision(t_pval_one)}")
print(f"Cohen's d     : {cohens_d:.3f} (small effect)")
print(f"Mann-Whitney U: U={u_stat:,.0f}, p(one-sided)={u_pval:.4f}  -> {decision(u_pval)}")
print(f"Chi-square    : chi2={chi2_stat:.2f}, dof={dof}, p={chi2_pval:.2e}, "
      f"Cramer's V={cramers_v:.3f}  -> {decision(chi2_pval)}")
print(f"Smallest expected count in Chi-square table: {expected.min():.1f} (rule of thumb: >= 5)")
print("\nObserved counts:")
print(contingency_table)
print("\nHigh-risk share by position:")
print(high_risk_share.round(3).to_string())

# ----------------------------------------------------------------------------
# STEP 3: VISUALISATION
# ----------------------------------------------------------------------------
# Figure 1: density plot with group means
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df_stats, x='Goal_Contributions_Per_90', hue='Rest_Condition', fill=True,
            common_norm=False, palette='Set1', alpha=0.4)
plt.axvline(normal_rest_output.mean(), color='blue', linestyle='--',
            label=f'Normal Mean ({normal_rest_output.mean():.2f})')
plt.axvline(congested_rest_output.mean(), color='red', linestyle='--',
            label=f'Congested Mean ({congested_rest_output.mean():.2f})')
plt.title('Figure 1: Performance Density Distribution Across Rest Conditions', pad=15)
plt.xlabel('Goal Contributions Per 90 Minutes')
plt.ylabel('Probability Density')
plt.legend()
plt.tight_layout()
plt.savefig('stat_1_density_distribution.png', dpi=300)
plt.close()

# Figure 2: boxplot comparison (white dot = group mean)
plt.figure(figsize=(8, 6))
sns.boxplot(data=df_stats, x='Rest_Condition', y='Goal_Contributions_Per_90',
            hue='Rest_Condition', legend=False, palette='Set2', showmeans=True,
            meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "black"})
plt.title('Figure 2: Performance Variance & Central Tendency Comparison', pad=15)
plt.xlabel('Rest Schedule Condition')
plt.ylabel('Goal Contributions Per 90 Minutes')
plt.tight_layout()
plt.savefig('stat_2_boxplot_intervals.png', dpi=300)
plt.close()

# Figure 3: Chi-square contingency heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlGnBu', cbar=False)
plt.title('Figure 3: Observed Injury Risk Contingency Matrix Across Positions', pad=15)
plt.xlabel('Injury Risk Classification')
plt.ylabel('Playing Position')
plt.tight_layout()
plt.savefig('stat_3_chi_heatmap.png', dpi=300)
plt.close()

print("\nVisualisation generation complete! PNG files saved locally.")
