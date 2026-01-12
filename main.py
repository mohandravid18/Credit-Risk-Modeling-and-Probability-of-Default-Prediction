# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve, 
                             auc, confusion_matrix, classification_report)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import shap
import warnings
import os

warnings.filterwarnings('ignore')

# ===================================================================
# Create folders for plots and data
# ===================================================================
plots_dir = 'credit_risk_plots'
data_dir = 'credit_risk_data'

os.makedirs(plots_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

print(f"Plots will be saved in: ./{plots_dir}/")
print(f"Processed data will be saved in: ./{data_dir}/\n")

# ===================================================================
# 1. Load German Credit Risk Dataset
# ===================================================================
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
columns = ['Status', 'Duration', 'CreditHistory', 'Purpose', 'CreditAmount', 
           'Savings', 'Employment', 'InstallmentRate', 'PersonalStatus', 'OtherDebtors',
           'Residence', 'Property', 'Age', 'OtherPlans', 'Housing', 
           'ExistingCredits', 'Job', 'Dependents', 'Telephone', 'ForeignWorker', 'Risk']

df = pd.read_csv(url, sep=' ', header=None, names=columns)

# Target: Risk 1 = Good, 2 = Bad → map to 0=Good (low PD), 1=Bad (default/high PD)
df['Risk'] = df['Risk'].map({1: 0, 2: 1})
print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
print(f"Default rate: {df['Risk'].mean():.2%}\n")

# Save raw dataset
raw_csv = f"{data_dir}/german_credit_raw.csv"
df.to_csv(raw_csv, index=False)
print(f"Raw dataset saved: {raw_csv}")

# ===================================================================
# 2. Exploratory Data Analysis (EDA)
# ===================================================================
plt.figure(figsize=(12, 10))

# Target distribution
plt.subplot(2, 2, 1)
sns.countplot(x='Risk', data=df)
plt.title('Distribution of Credit Risk (0=Good, 1=Bad/Default)')
plt.xlabel('Risk')

# Age distribution by risk
plt.subplot(2, 2, 2)
sns.boxplot(x='Risk', y='Age', data=df)
plt.title('Age vs Credit Risk')

# Credit amount by risk
plt.subplot(2, 2, 3)
sns.boxplot(x='Risk', y='CreditAmount', data=df)
plt.title('Credit Amount vs Credit Risk')

# Duration by risk
plt.subplot(2, 2, 4)
sns.boxplot(x='Risk', y='Duration', data=df)
plt.title('Loan Duration vs Credit Risk')

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '01_eda_basic_plots.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved: 01_eda_basic_plots.png")

# Correlation heatmap (numeric features only)
numeric_cols = df.select_dtypes(include=np.number).columns
plt.figure(figsize=(10, 8))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numeric Features')
plt.savefig(os.path.join(plots_dir, '02_correlation_heatmap.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved: 02_correlation_heatmap.png")

# ===================================================================
# 3. Preprocessing
# ===================================================================
# Separate features and target
X = df.drop('Risk', axis=1)
y = df['Risk']

# Encode categorical variables
categorical_cols = X.select_dtypes(include='object').columns
X_encoded = X.copy()
for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X[col])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, 
                                                    random_state=42, stratify=y)

# Scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save processed features
processed_df = pd.DataFrame(X_train_scaled, columns=X_encoded.columns)
processed_df['Risk'] = y_train.values
processed_csv = f"{data_dir}/german_credit_processed_train.csv"
processed_df.to_csv(processed_csv, index=False)
print(f"Processed training data saved: {processed_csv}\n")

# ===================================================================
# 4. Handle Class Imbalance & Train Models
# ===================================================================
print("Class distribution before SMOTE:", np.bincount(y_train))
smote = SMOTE(random_state=42)

# Model 1: Logistic Regression with class weights (no SMOTE)
lr_weighted = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)

# Model 2: Logistic Regression with SMOTE
lr_smote = ImbPipeline([
    ('smote', smote),
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=1000, random_state=42))
])

# Fit models
lr_weighted.fit(X_train_scaled, y_train)
lr_smote.fit(X_train, y_train)  # SMOTE pipeline handles scaling internally

# Predictions
pred_weighted = lr_weighted.predict_proba(X_test_scaled)[:, 1]
pred_smote = lr_smote.predict_proba(X_test_scaled)[:, 1]
y_pred_weighted = lr_weighted.predict(X_test_scaled)
y_pred_smote = lr_smote.predict(X_test_scaled)

# ===================================================================
# 5. Evaluation
# ===================================================================
roc_weighted = roc_auc_score(y_test, pred_weighted)
roc_smote = roc_auc_score(y_test, pred_smote)
pr_weighted = auc(*precision_recall_curve(y_test, pred_weighted)[1::-1])
pr_smote = auc(*precision_recall_curve(y_test, pred_smote)[1::-1])

print(f"ROC-AUC (Weighted): {roc_weighted:.3f}")
print(f"ROC-AUC (SMOTE): {roc_smote:.3f}")
print(f"PR-AUC (Weighted): {pr_weighted:.3f}")
print(f"PR-AUC (SMOTE): {pr_smote:.3f}\n")

# ROC Curve
plt.figure(figsize=(12, 5))
fpr1, tpr1, _ = roc_curve(y_test, pred_weighted)
fpr2, tpr2, _ = roc_curve(y_test, pred_smote)
plt.subplot(1, 2, 1)
plt.plot(fpr1, tpr1, label=f'Weighted (AUC={roc_weighted:.3f})')
plt.plot(fpr2, tpr2, label=f'SMOTE (AUC={roc_smote:.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()

# Precision-Recall Curve
plt.subplot(1, 2, 2)
prec1, rec1, _ = precision_recall_curve(y_test, pred_weighted)
prec2, rec2, _ = precision_recall_curve(y_test, pred_smote)
plt.plot(rec1, prec1, label=f'Weighted (AUC={pr_weighted:.3f})')
plt.plot(rec2, prec2, label=f'SMOTE (AUC={pr_smote:.3f})')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '03_roc_pr_curves.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved: 03_roc_pr_curves.png")

# ===================================================================
# 6. Explainability with SHAP (using SMOTE model)
# ===================================================================
explainer = shap.Explainer(lr_smote.named_steps['lr'], 
                          lr_smote.named_steps['smote'].fit_resample(X_train, y_train)[0])
shap_values = explainer(X_test_scaled)

# Summary plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_scaled, feature_names=X_encoded.columns, show=False)
plt.savefig(os.path.join(plots_dir, '04_shap_summary.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved: 04_shap_summary.png")

# Feature importance (mean absolute SHAP)
shap_importance = pd.DataFrame({
    'feature': X_encoded.columns,
    'importance': np.abs(shap_values.values).mean(0)
}).sort_values('importance', ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(x='importance', y='feature', data=shap_importance.head(10))
plt.title('Top 10 Features by SHAP Importance')
plt.savefig(os.path.join(plots_dir, '05_shap_feature_importance.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Saved: 05_shap_feature_importance.png")

# ===================================================================
# Final Insights
# ===================================================================
print("\n" + "="*60)
print("CREDIT RISK MODELING INSIGHTS")
print("="*60)
print(f"• Dataset: German Credit (Default rate: {df['Risk'].mean():.2%})")
print(f"• Best ROC-AUC: {max(roc_weighted, roc_smote):.3f}")
print(f"• Top risk drivers (SHAP): {', '.join(shap_importance['feature'].head(5))}")
print("• SMOTE often improves recall for minority class (defaults) — crucial for risk detection")
print("• Logistic Regression remains highly interpretable and suitable for regulatory compliance")
print("• Use predicted PD for loan approval, pricing, or capital allocation")
print("="*60)

print(f"\nProject Complete!")
print(f"All 5 plots saved in: './{plots_dir}/'")
print(f"Raw & processed datasets saved in: './{data_dir}/'")