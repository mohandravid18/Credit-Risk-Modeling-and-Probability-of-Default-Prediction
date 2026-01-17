# Credit Risk Modeling – Insights

This project builds an end‑to‑end **credit risk / PD (probability of default) model** using the German Credit dataset, addressing class imbalance, model evaluation, and explainability.

---

## 1. Dataset & Business Problem

- **Source:** German Credit dataset from the UCI repository (statlog/german). [web:10]  
- **Size:** 1,000 customers, 21 columns (20 features + target `Risk`).  
- **Target (`Risk`):**
  - Original: 1 = good, 2 = bad.
  - Mapped to: 0 = good (non‑default / low PD), 1 = bad (default / high PD).
- **Default rate:** 30.00% of customers are labeled as bad → **moderately imbalanced** but realistic for retail credit.

**Business framing:**  
Predict the probability that an applicant will **default** so a bank can use it for loan approval, risk‑based pricing, and capital allocation.

---

## 2. EDA & Data Understanding

Generated `01_eda_basic_plots.png` and `02_correlation_heatmap.png`:
<img width="2381" height="1982" alt="01_eda_basic_plots" src="https://github.com/user-attachments/assets/17fcff7b-ca89-4f5b-b86f-a0b019da223b" />
<img width="1745" height="1554" alt="02_correlation_heatmap" src="https://github.com/user-attachments/assets/1b240784-60f3-4476-985c-4311134b8fd8" />

- **Target distribution:**  
  - Shows 70% good vs 30% bad borrowers.
- **Boxplots vs Risk:**
  - `Age vs Risk`: Younger or very old borrowers may show different risk patterns.
  - `CreditAmount vs Risk`: Higher loan amounts often tilt towards higher risk.
  - `Duration vs Risk`: Longer loan durations can be associated with higher default probability.
- **Correlation heatmap (numeric features):**
  - Shows relationships between numeric variables like Duration, CreditAmount, Age, etc.
  - Helps detect multicollinearity and understand drivers.

These plots give intuition about how customer profile and loan characteristics relate to default behavior.

---

## 3. Preprocessing & Feature Engineering

- **Feature/target split:**
  - `X =` all features except `Risk`.
  - `y = Risk`.
- **Categorical encoding:**
  - Label‑encodes all categorical columns (e.g., Status, CreditHistory, Purpose, Savings, Employment).
- **Train–test split:**
  - 80% train, 20% test, with **stratification** on `y` to preserve the 70/30 class balance.
- **Scaling:**
  - StandardScaler applied to features for stable logistic regression coefficients and optimization.

Processed training data is saved as `credit_risk_data/german_credit_processed_train.csv` for reuse.

---

## 4. Models & Imbalance Strategies

Two logistic regression setups are compared:

1. **Weighted Logistic Regression (no SMOTE)**
   - `class_weight='balanced'` automatically reweights the loss so the minority class (defaults) gets higher penalty.
   - Trained on scaled features (`X_train_scaled`).

2. **Logistic Regression with SMOTE**
   - Imbalanced‑learn `Pipeline`: SMOTE → StandardScaler → LogisticRegression.
   - SMOTE oversamples the minority class in the **training set** by creating synthetic default cases.
   - Aims to improve sensitivity (recall) to default cases.

Both models output **PDs (predicted probabilities of default)**.

---

## 5. Model Performance & Curves

Metrics on the held‑out test set:

- **ROC‑AUC:**
  - Weighted: **0.791**
  - SMOTE: **0.689**
- **PR‑AUC:**
  - Weighted: **0.595**
  - SMOTE: **0.477**

Generated `03_roc_pr_curves.png`:
<img width="2381" height="982" alt="03_roc_pr_curves" src="https://github.com/user-attachments/assets/5b9b9eb4-f93e-42b8-82bc-b44be7a0c353" />

- **ROC curves:**
  - Weighted model’s curve lies above SMOTE’s; better overall ranking of defaults vs non‑defaults.
- **Precision–Recall curves:**
  - Weighted model again dominates; better precision/recall trade‑off, especially relevant for the default class.

Interpretation:

- In this run, **class‑weighted logistic regression outperforms the SMOTE pipeline** on both ROC‑AUC and PR‑AUC.
- SMOTE can still be useful for improving **recall** in some settings, but must be evaluated carefully; here, the weighted model is the stronger baseline.

---

## 6. Explainability with SHAP

Using the SMOTE logistic model for interpretability:

- SHAP explainer fitted on the (SMOTE‑resampled) training features.
- Generated:
  - `04_shap_summary.png` – SHAP summary plot.
  - `05_shap_feature_importance.png` – bar plot of mean absolute SHAP values.
<img width="1558" height="1093" alt="05_shap_feature_importance" src="https://github.com/user-attachments/assets/4a6bfb3f-a632-408a-853b-bdfc0a6fdf6b" />
<img width="1535" height="1879" alt="04_shap_summary" src="https://github.com/user-attachments/assets/6c5a8e5a-80fb-42a2-997b-08a07e221e4f" />

**Top risk drivers (from SHAP):**

1. `CreditAmount`
2. `Duration`
3. `Age`
4. `Status`
5. `CreditHistory`

Risk interpretation:

- **CreditAmount:** Larger requested credit amounts push PD higher.
- **Duration:** Longer loan durations increase credit exposure period → higher PD.
- **Age:** Certain age segments (very young / older) may carry more risk.
- **Status / CreditHistory:** Captures prior repayment behavior and financial stability; poor status/history strongly drives defaults.

This makes the model **auditable**: risk and compliance teams can see which features drive decisions and verify they align with policy.

---

## 7. Business & Risk Management Usage

From the final console summary:

- Best ROC‑AUC: **0.791** (weighted logistic).  
- Default rate: **30%**, realistic for non‑prime portfolios.  
- **Use cases:**
  - **Loan approval:** Set PD thresholds for accept / review / reject.
  - **Pricing:** Higher PD → price in more margin (higher interest) to cover expected loss.
  - **Capital allocation:** Use PD (combined with LGD, EAD) to estimate expected loss and economic capital needs.
- **Model choice:**
  - Logistic Regression is:
    - Simple and stable.
    - Highly interpretable.
    - Familiar to regulators and model validation teams.

---

## 8. What This Project Shows

- Ability to build a **complete credit risk pipeline**:
  - Data ingestion → EDA → encoding & scaling → imbalance handling → modeling → evaluation → explainability.
- Competence with:
  - Class imbalance techniques (class weights, SMOTE).
  - Proper metrics for credit risk (ROC‑AUC, PR‑AUC, not just accuracy).
  - SHAP for transparent, feature-level explanations.
- Direct relevance to **bank credit risk / decisioning systems**, where PD models must be both **predictive and interpretable**.
