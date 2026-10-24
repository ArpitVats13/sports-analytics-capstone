"""
Week 2 - Advanced Visual Storytelling for Recruitment Decisions
YuvaIntern | Virtual Data Science with Python | Summer Training 2026
Author  : Arpit Vats

Purpose : Engineer performance features and build five executive-style visualisations
          that help a club board question its recruitment assumptions.

DATA NOTE: The dataset is SYNTHETIC (NumPy, seed = 42); no real player records are used.

Outputs : story_1_joint_scarcity.png, story_2_violin_valuation.png,
          story_3_bubble_efficiency.png, story_4_facet_ecdf.png, story_5_dual_axis.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="white")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16,
})

# ----------------------------------------------------------------------------
# STEP 1: GENERATE DATASET AND ENGINEER FEATURES
# ----------------------------------------------------------------------------
np.random.seed(42)
n_records = 1000

data = {
    'Player_ID': [f"PL{1000 + i}" for i in range(n_records)],
    'Name': [f"Player_{i}" for i in range(n_records)],
    'Age': np.random.choice([21, 23, 25, 27, 29, 31, 33], size=n_records,
                            p=[0.1, 0.2, 0.25, 0.2, 0.13, 0.08, 0.04]),
    'Position': np.random.choice(['Forward', 'Midfielder', 'Defender', 'Goalkeeper'],
                                 size=n_records, p=[0.3, 0.35, 0.25, 0.1]),
    'Matches_Played': np.random.randint(5, 38, size=n_records),
    'Goals': np.random.poisson(lam=5, size=n_records),
    'Assists': np.random.poisson(lam=4, size=n_records),
    'Market_Value_M': np.random.exponential(scale=14.0, size=n_records),
}
df = pd.DataFrame(data)

df['Goal_Contribution'] = df['Goals'] + df['Assists']
df['Efficiency_Per_Match'] = df['Goal_Contribution'] / df['Matches_Played']   # Matches_Played >= 5
# Three equal-sized market-value tiers
df['Tier'] = pd.qcut(df['Market_Value_M'], q=3, labels=['Value Asset', 'Core Roster', 'Marquee Elite'])

# Summary statistics quoted in the report
age_summary = (df.groupby('Age')
                 .agg(Mean_Efficiency=('Efficiency_Per_Match', 'mean'),
                      Players=('Player_ID', 'count'))
                 .reset_index())
share_high_eff = (df['Efficiency_Per_Match'] > 0.4).mean()
share_low_eff = (df['Efficiency_Per_Match'] < 0.3).mean()
median_matches_by_tier = df.groupby('Tier', observed=True)['Matches_Played'].median()

print("--- SUMMARY STATISTICS ---")
print(age_summary.round(3).to_string(index=False))
print(f"\nShare of players above 0.4 contributions per match : {share_high_eff:.1%}")
print(f"Share of players below 0.3 contributions per match : {share_low_eff:.1%}")
print("\nMedian matches played by tier:")
print(median_matches_by_tier.to_string())

# ----------------------------------------------------------------------------
# STEP 2: FIVE VISUALISATIONS
# ----------------------------------------------------------------------------
# Visualisation 1: joint scatter with density contours (talent scarcity)
g = sns.jointplot(data=df, x='Age', y='Efficiency_Per_Match', hue='Position',
                  kind='scatter', alpha=0.5, palette='Set2', height=7)
g.plot_joint(sns.kdeplot, color="gray", alpha=0.3, zorder=0, levels=5)
g.fig.suptitle('Figure 1: The Scarcity Matrix - Production Efficiency vs. Age', y=1.03)
g.ax_joint.set_xlabel('Player Age')
g.ax_joint.set_ylabel('Contributions Per Match (Goals + Assists)')
plt.savefig('story_1_joint_scarcity.png', dpi=300, bbox_inches='tight')
plt.close()

# Visualisation 2: violin plot of market value by position and tier
plt.figure(figsize=(11, 6))
sns.violinplot(data=df, x='Position', y='Market_Value_M', hue='Tier',
               palette='muted', inner='quartile', split=False)
plt.title('Figure 2: Financial Risk and Market Valuation Distributions across Tiers', pad=20)
plt.xlabel('Playing Position')
plt.ylabel('Market Value (€ Millions)')
sns.despine(left=True, bottom=True)
plt.tight_layout()
plt.savefig('story_2_violin_valuation.png', dpi=300)
plt.close()

# Visualisation 3: bubble chart for forwards (goals, assists, value, age)
plt.figure(figsize=(12, 7))
forwards = df[df['Position'] == 'Forward']
bubble = plt.scatter(x=forwards['Goals'], y=forwards['Assists'],
                     s=forwards['Market_Value_M'] * 15, c=forwards['Age'],
                     cmap='coolwarm', alpha=0.6, edgecolors='black', linewidth=0.5)
cbar = plt.colorbar(bubble)
cbar.set_label('Player Age', rotation=270, labelpad=15)
plt.title('Figure 3: Identification Profile of Efficient Targets (Forwards Group)', pad=20)
plt.xlabel('Goals Scored')
plt.ylabel('Assists Provided')
plt.annotate('High Efficiency Target Zone\n(High Output / Low Value Bubble Size)',
             xy=(9, 8), xytext=(3, 9),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6))
plt.tight_layout()
plt.savefig('story_3_bubble_efficiency.png', dpi=300)
plt.close()

# Visualisation 4: ECDF grid (reliability of match availability)
g = sns.FacetGrid(df, col="Position", hue="Tier", palette="viridis", height=4, aspect=1)
g.map(sns.ecdfplot, "Matches_Played", linewidth=2)
g.add_legend(title="Market Segment")
g.fig.suptitle('Figure 4: Reliability Profiles - Cumulative Probability of Player Match Experience', y=1.08)
g.set_axis_labels("Total Matches Featured", "Cumulative Sample Probability")
plt.savefig('story_4_facet_ecdf.png', dpi=300, bbox_inches='tight')
plt.close()

# Visualisation 5: dual-axis chart - mean efficiency (line) and talent pool size (bars)
# CORRECTION: the original script drew the line at numeric x = 21..33 while the bars sat at
# categorical positions 0..6 on the same shared axis, so the efficiency line fell outside the
# visible area. Both series now use the same positions (0..6) with age labels on the x-axis.
x_pos = np.arange(len(age_summary))
fig, ax1 = plt.subplots(figsize=(12, 6))
ax2 = ax1.twinx()

bars = ax2.bar(x_pos, age_summary['Players'], color='#e28743', alpha=0.3,
               label='Available Talent Pool (players)')
ax2.bar_label(bars, padding=2, fontsize=9, color='#a85a1f')
ax2.set_ylabel('Total Number of Available Players', color='#e28743')
ax2.tick_params(axis='y', labelcolor='#e28743')
ax2.set_ylim(0, age_summary['Players'].max() * 1.15)

ax1.plot(x_pos, age_summary['Mean_Efficiency'], color='#2b5c8f', marker='o', linewidth=3,
         label='Mean Efficiency (contributions per match)')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(age_summary['Age'])
ax1.set_xlabel('Player Age')
ax1.set_ylabel('Mean Performance Efficiency', color='#2b5c8f')
ax1.tick_params(axis='y', labelcolor='#2b5c8f')
ax1.set_ylim(age_summary['Mean_Efficiency'].min() - 0.05,
             age_summary['Mean_Efficiency'].max() + 0.05)
ax1.set_zorder(ax2.get_zorder() + 1)    # draw the line above the bars
ax1.patch.set_visible(False)

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc='upper right')

plt.title('Figure 5: Executive Overview - Efficiency by Age vs. Talent Pool Size', pad=20)
fig.tight_layout()
plt.savefig('story_5_dual_axis.png', dpi=300)
plt.close()

print("\nExecution successful! All 5 storytelling visualisations saved as PNG files.")
