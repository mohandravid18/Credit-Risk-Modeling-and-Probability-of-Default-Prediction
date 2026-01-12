Markdown

# Credit Risk Modeling and Probability of Default Prediction

## Project Overview
This project builds an **explainable credit risk prediction system** to estimate the **Probability of Default (PD)** for loan applicants using the classic **German Credit Risk dataset**. It implements a baseline **Logistic Regression** model with two approaches to handle class imbalance: class weighting and **SMOTE** oversampling. The focus is on performance, interpretability, and regulatory compliance.

**Key Highlights:**
- Comprehensive **EDA** on credit risk factors (age, amount, duration, etc.)
- Class imbalance handling using **SMOTE** and balanced class weights
- Evaluation using **ROC-AUC** and **Precision-Recall AUC** (suitable for imbalanced data)
- Model **explainability** via **SHAP** values (global and local feature importance)
- Comparison of two Logistic Regression variants
- Saves all plots and processed data in organized folders

## Dataset
- **Source**: UCI Machine Learning Repository  
  URL: https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data
- **Size**: 1,000 loan applications
- **Features**: 20 (mix of categorical and numerical)
  - Key numeric: `Duration`, `CreditAmount`, `Age`, `InstallmentRate`
  - Key categorical: `CreditHistory`, `Purpose`, `Savings`, `Employment`, etc.
- **Target**: `Risk` → 0 = Good (low PD), 1 = Bad (default/high PD)
- **Default Rate**: ~30%

## Project Structure

.
├── credit_risk_plots/
│   ├── 01_eda_basic_plots.png               # Target distribution & key feature boxplots
│   ├── 02_correlation_heatmap.png           # Numeric feature correlations
│   ├── 03_roc_pr_curves.png                 # ROC & Precision-Recall curves
│   ├── 04_shap_summary.png                  # SHAP summary plot (beeswarm)
│   └── 05_shap_feature_importance.png       # Top 10 features by mean |SHAP|
├── credit_risk_data/
│   ├── german_credit_raw.csv                # Original downloaded dataset
│   └── german_credit_processed_train.csv    # Scaled & encoded training split
├── credit_risk_modeling.py                  # Main script
└── README.md                                # This file
text

## Requirements
Install the required packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn shap

How to Run

    Save the script as credit_risk_modeling.py
    Run the script:

Bash

python credit_risk_modeling.py

    Output:
        Automatically downloads the dataset from UCI
        Creates credit_risk_plots/ and credit_risk_data/ folders
        Saves 5 high-quality plots
        Saves raw and processed datasets
        Prints performance metrics, top risk drivers (SHAP), and business insights

Key Outputs & Insights

    EDA Plots: Reveal patterns such as higher credit amounts and longer durations associated with defaults
    Model Performance:
        ROC-AUC typically ~0.75–0.78
        PR-AUC improved with SMOTE in many cases
    Top Risk Drivers (from SHAP, typical results):
        CreditHistory, Duration, CreditAmount, Purpose, Age
    Interpretability: SHAP values show direction and magnitude of feature impact on PD

Business Applications

    Loan approval automation
    Risk-based pricing (higher PD → higher interest rate)
    Regulatory reporting (highly interpretable model)
    Portfolio risk monitoring and capital allocation

Future Improvements

    Compare with tree-based models (XGBoost/Random Forest) for higher performance
    Hyperparameter tuning and threshold optimization
    Local explanations (SHAP force plots for individual applicants)
    Integration with credit scoring pipeline
