"""
Week 4 - Injury-Risk Prediction with Machine Learning
YuvaIntern | Virtual Data Science with Python | Summer Training 2026
Author  : Arpit Vats

Purpose : Predict whether an athlete is at high (1) or low (0) risk of a soft-tissue injury
          in the next match micro-cycle, and compare two classifiers:
            - Logistic Regression (baseline, features standardised)
            - Random Forest (100 trees, max depth 5, untuned)

DATA NOTE: The dataset is SYNTHETIC (NumPy, seed = 42). The risk label is generated from a
logistic formula of workload, sleep and soreness, so the models are learning a known rule.

Outputs : ml_1_confusion_matrices.png, ml_2_roc_curves.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, classification_report, roc_curve, auc,
                             accuracy_score, precision_score, recall_score, f1_score)

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
np.random.seed(42)

# ----------------------------------------------------------------------------
# STEP 1: GENERATE SYNTHETIC ATHLETE WORKLOAD DATASET
# ----------------------------------------------------------------------------
n_samples = 1200

age = np.random.randint(18, 36, size=n_samples)
acwr = np.random.normal(loc=1.1, scale=0.3, size=n_samples)         # Acute:Chronic Workload Ratio
load_28d = np.random.normal(loc=120, scale=25, size=n_samples)      # 28-day load (not used in the label)
sleep = np.random.randint(3, 10, size=n_samples)
soreness = np.random.randint(1, 10, size=n_samples)

# Linear logit plus noise; the flag is 1 when the resulting probability exceeds 0.5
logit = (-3.5 + (1.8 * acwr) + (0.03 * age) + (0.35 * soreness) - (0.4 * sleep)
         + np.random.normal(0, 0.8, n_samples))
prob = 1 / (1 + np.exp(-logit))
injury_flag = (prob > 0.5).astype(int)

df = pd.DataFrame({
    'Age': age,
    'ACWR': acwr,
    'Cumulative_28Day_Load': load_28d,
    'Sleep_Quality': sleep,
    'Soreness_Score': soreness,
    'Injury_Risk_Flag': injury_flag,
})

# ----------------------------------------------------------------------------
# STEP 2: PREPROCESSING AND MODEL TRAINING
# ----------------------------------------------------------------------------
X = df.drop('Injury_Risk_Flag', axis=1)
y = df['Injury_Risk_Flag']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)      # fit on training data only
X_test_scaled = scaler.transform(X_test)

# Model 1: Logistic Regression
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)
y_prob_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

# Model 2: Random Forest (trees do not need scaling)
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

# ----------------------------------------------------------------------------
# STEP 3: EVALUATION (every metric quoted in the report is printed here)
# ----------------------------------------------------------------------------
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
auc_lr = auc(fpr_lr, tpr_lr)
auc_rf = auc(fpr_rf, tpr_rf)

cm_lr = confusion_matrix(y_test, y_pred_lr)
cm_rf = confusion_matrix(y_test, y_pred_rf)

results = {}
for name, y_pred, cm, model_auc in [('Logistic Regression', y_pred_lr, cm_lr, auc_lr),
                                    ('Random Forest', y_pred_rf, cm_rf, auc_rf)]:
    tn, fp, fn, tp = cm.ravel()
    results[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'ROC-AUC': model_auc,
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
    }

print("--- DATASET AND SPLIT ---")
print(f"Records: {len(df)} | train: {len(X_train)} | test: {len(X_test)}")
print(f"High-risk share (all data): {y.mean():.1%} | high-risk cases in test set: {int(y_test.sum())}")

print("\n--- MODEL COMPARISON (test set) ---")
print(pd.DataFrame(results).T.round(3).to_string())
for name, y_pred in [('Logistic Regression', y_pred_lr), ('Random Forest', y_pred_rf)]:
    print(f"\n{name} - classification report")
    print(classification_report(y_test, y_pred, target_names=['Low risk', 'High risk']))

print("--- FEATURE INFLUENCE ---")
print("Logistic Regression coefficients (standardised features):")
print(pd.Series(lr_model.coef_[0], index=X.columns).round(2).sort_values(ascending=False).to_string())
print("\nRandom Forest feature importances:")
print(pd.Series(rf_model.feature_importances_, index=X.columns).round(3).sort_values(ascending=False).to_string())

# ----------------------------------------------------------------------------
# STEP 4: VISUALISATION
# ----------------------------------------------------------------------------
# Figure 1: side-by-side confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
axes[0].set_title('Logistic Regression Confusion Matrix')
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[1], cbar=False)
axes[1].set_title('Random Forest Confusion Matrix')
axes[1].set_xlabel('Predicted Label')
axes[1].set_ylabel('True Label')
plt.tight_layout()
plt.savefig('ml_1_confusion_matrices.png', dpi=300)
plt.close()

# Figure 2: ROC curve comparison
plt.figure(figsize=(9, 6))
plt.plot(fpr_lr, tpr_lr, color='blue', lw=2, label=f'Logistic Regression (AUC = {auc_lr:.3f})')
plt.plot(fpr_rf, tpr_rf, color='green', lw=2, label=f'Random Forest (AUC = {auc_rf:.3f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('Figure 2: Receiver Operating Characteristic (ROC) Comparison', pad=15)
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Recall)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('ml_2_roc_curves.png', dpi=300)
plt.close()

print("\nModel pipeline execution complete! PNG charts saved locally.")
