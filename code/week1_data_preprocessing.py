"""
Week 1 - Data Preprocessing and Exploratory Data Analysis
YuvaIntern | Virtual Data Science with Python | Summer Training 2026
Author  : Arpit Vats

Project : Sports Analytics Capstone (Premier League-style player metrics)
Purpose : Generate a SYNTHETIC player dataset that contains deliberate data-quality
          problems, clean it with a documented pipeline, and explore it visually.

DATA NOTE: No real or scraped data is used. All records are generated with NumPy
(seed = 42), so every number printed by this script is reproducible.

Outputs : visual_1_missing_heatmap.png, visual_2_age_vs_value.png,
          visual_3_correlation_matrix.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# ----------------------------------------------------------------------------
# STEP 1: GENERATE SYNTHETIC DATASET WITH DELIBERATE DATA-QUALITY ISSUES
# ----------------------------------------------------------------------------
np.random.seed(42)
n_records = 1000

# Injury_History is built from an OBJECT array so that np.nan stays a real missing
# value. (np.random.choice(['Yes', np.nan]) silently converts np.nan into the text
# 'nan', which pandas does not treat as missing, so fillna() would never fire.)
injury_choices = np.array(['Yes', np.nan], dtype=object)

data = {
    'Player_ID': [f"PL{1000 + i}" for i in range(n_records)],
    'Name': [f"Player_{i}" for i in range(n_records)],
    'Age': np.random.choice([22, 25, 28, 31, 34, np.nan], size=n_records,
                            p=[0.2, 0.3, 0.2, 0.15, 0.1, 0.05]),
    'Position': np.random.choice(['Forward', 'Midfielder', 'Defender', 'Goalkeeper'],
                                 size=n_records),
    'Matches_Played': np.random.randint(0, 38, size=n_records),
    'Goals': np.random.poisson(lam=4, size=n_records),
    'Assists': np.random.poisson(lam=3, size=n_records),
    'Market_Value_M': np.random.exponential(scale=15.0, size=n_records),
    'Injury_History': np.random.choice(injury_choices, size=n_records, p=[0.15, 0.85]),
}
df = pd.DataFrame(data)

# Inject intentional problems for the cleaning demonstration
df = pd.concat([df, df.iloc[:15]], ignore_index=True)           # 15 duplicate rows
df.loc[df['Market_Value_M'] > 60, 'Market_Value_M'] = np.nan    # extreme values -> missing

raw_df = df.copy()          # snapshot of the raw table (used for the Figure 1 heatmap)
raw_rows = len(raw_df)

print("--- RAW DATA INITIAL ASSESSMENT ---")
df.info()

# ----------------------------------------------------------------------------
# STEP 2: DATA CLEANING PIPELINE
# ----------------------------------------------------------------------------
# 1. Remove exact duplicates (same Player_ID)
duplicate_count = int(df.duplicated(subset=['Player_ID']).sum())
df = df.drop_duplicates(subset=['Player_ID'], keep='first')
unique_rows = len(df)

# Missing-value profile of the de-duplicated data (before imputation)
age_nulls = int(df['Age'].isnull().sum())
value_nulls = int(df['Market_Value_M'].isnull().sum())
injury_nulls = int(df['Injury_History'].isnull().sum())

# 2. Missing categorical data: a blank injury entry means "no recorded injury"
df['Injury_History'] = df['Injury_History'].fillna('No')

# 3. Missing market values: median of the player's own position (values are right-skewed)
df['Market_Value_M'] = df.groupby('Position')['Market_Value_M'].transform(
    lambda x: x.fillna(x.median()))

# 4. Missing ages: overall median, then cast to integer
df['Age'] = df['Age'].fillna(df['Age'].median()).astype(int)

# 5. Remove players with zero matches (avoids division by zero in per-match ratios)
zero_match_rows = int((df['Matches_Played'] == 0).sum())
df = df[df['Matches_Played'] > 0].copy()
clean_rows = len(df)

# Validation: the cleaned table must be complete and free of duplicates
assert df.isnull().sum().sum() == 0, "Missing values remain after cleaning"
assert not df['Player_ID'].duplicated().any(), "Duplicates remain after cleaning"

print("\n--- CLEANED DATA ASSESSMENT ---")
df.info()

print("\n--- CLEANING SUMMARY ---")
print(f"Raw rows                      : {raw_rows}")
print(f"Duplicate rows removed        : {duplicate_count}  -> {unique_rows} unique players")
print(f"Missing Age (unique rows)     : {age_nulls} ({age_nulls / unique_rows:.1%})")
print(f"Missing Market_Value_M        : {value_nulls}")
print(f"Missing Injury_History        : {injury_nulls} ({injury_nulls / unique_rows:.1%})")
print(f"Zero-match players removed    : {zero_match_rows}")
print(f"Clean rows for analysis       : {clean_rows}")
print("Injury_History after cleaning  :", df['Injury_History'].value_counts().to_dict())

# ----------------------------------------------------------------------------
# STEP 3: EXPLORATORY ANALYSIS
# ----------------------------------------------------------------------------
numerical_features = df[['Age', 'Matches_Played', 'Goals', 'Assists', 'Market_Value_M']]
correlation_matrix = numerical_features.corr()
corr_goals_value = correlation_matrix.loc['Goals', 'Market_Value_M']
corr_age_value = correlation_matrix.loc['Age', 'Market_Value_M']

print("\n--- CORRELATION MATRIX ---")
print(correlation_matrix.round(2))
print("\nMean market value (EUR m) by age:")
print(df.groupby('Age')['Market_Value_M'].mean().round(1).to_string())

# ----------------------------------------------------------------------------
# STEP 4: VISUALISATION
# ----------------------------------------------------------------------------
# Figure 1: Missing-data heatmap of the RAW table (before cleaning)
plt.figure(figsize=(10, 5))
sns.heatmap(raw_df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title('Figure 1: Spatial Heatmap of Missing System Values (Pre-Clean)', fontsize=14, pad=15)
plt.ylabel('Row Index')
plt.tight_layout()
plt.savefig('visual_1_missing_heatmap.png', dpi=300)
plt.close()

# Figure 2: Age vs market value with a second-degree trend line
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Age', y='Market_Value_M', hue='Position', alpha=0.6, palette='deep')
sns.regplot(data=df, x='Age', y='Market_Value_M', scatter=False, color='red', order=2,
            label='Valuation Curve')
plt.title('Figure 2: Non-Linear Regression of Player Market Value Against Age', fontsize=14, pad=15)
plt.xlabel('Player Age (Years)')
plt.ylabel('Market Value (€ Millions)')
plt.legend()
plt.tight_layout()
plt.savefig('visual_2_age_vs_value.png', dpi=300)
plt.close()

# Figure 3: Pearson correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Figure 3: Pearson Correlation Matrix of On-Field Metrics', fontsize=14, pad=15)
plt.tight_layout()
plt.savefig('visual_3_correlation_matrix.png', dpi=300)
plt.close()

print("\nProcessing complete! Visualisations saved locally as PNG files.")
