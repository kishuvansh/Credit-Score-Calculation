import pandas as pd
import numpy as np
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

warnings.simplefilter(action='ignore', category=FutureWarning)

print("=== Project Nova: Advanced Credit Score Pipeline ===")

# =============================================================================
# 1. DATA PROCESSING & FEATURE ENGINEERING
# =============================================================================
print("\n--- Step 1: Loading and Preprocessing Data ---")
try:
    cols = ['loan_status', 'loan_amnt', 'term', 'purpose', 'home_ownership', 
            'annual_inc', 'emp_length', 'dti', 'revol_util', 'inq_last_6mths', 
            'pub_rec_bankruptcies', 'fico_range_low', 'fico_range_high', 
            'int_rate', 'installment', 'open_acc', 'total_acc', 
            'revol_bal', 'total_rev_hi_lim', 'mort_acc']
    
    # Load 10% sample for fast local iteration
    df_full = pd.read_csv('project_nova_dataset.csv', usecols=cols, low_memory=False)
    df = df_full.sample(frac=0.1, random_state=42)
    
except FileNotFoundError:
    print("Error: 'project_nova_dataset.csv' not found.")
    exit()

# Target Variable handling
df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])].copy()
df['loan_status'] = np.where(df['loan_status'] == 'Charged Off', 1, 0)

# Extract Term length
if df['term'].dtype == 'object':
    df['term'] = df['term'].str.extract(r'(\d+)').astype(float)

# Map employment length
emp_map = {'< 1 year': 0.5, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
           '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9,
           '10+ years': 10}
if df['emp_length'].dtype == 'object':
    df['emp_length'] = df['emp_length'].map(emp_map)

# Missing Values Imputation (Median or Zero where appropriate)
df['emp_length'].fillna(0, inplace=True)
df['revol_util'].fillna(0, inplace=True)
df['pub_rec_bankruptcies'].fillna(0, inplace=True)
df['dti'].fillna(df['dti'].median(), inplace=True)
df['total_rev_hi_lim'].fillna(df['total_rev_hi_lim'].median(), inplace=True)
df['mort_acc'].fillna(0, inplace=True)

# Outlier Capping
inc_cap = df['annual_inc'].quantile(0.995)
df['annual_inc'] = df['annual_inc'].clip(upper=inc_cap)

print("\n--- Step 2: Extracting New Compound Features ---")

# 1. Average FICO Baseline
df['fico_avg'] = (df['fico_range_low'] + df['fico_range_high']) / 2

# 2. Monthly Debt Burden Ratio (Installment relative to monthly income)
df['monthly_income'] = df['annual_inc'] / 12
df['monthly_debt_burden'] = df['installment'] / (df['monthly_income'] + 1)

# 3. Overall Credit Utilization Behavior
df['balance_to_limit_ratio'] = df['revol_bal'] / (df['total_rev_hi_lim'] + 1)

# 4. Old engineered features
df['loan_to_income_ratio'] = df['loan_amnt'] / (df['annual_inc'] + 1)

# Drop redundant raw columns we used for compounds
drop_cols = ['monthly_income', 'fico_range_low', 'fico_range_high']
df.drop(columns=drop_cols, inplace=True)

# Categorical encodings
df = pd.get_dummies(df, columns=['home_ownership', 'purpose'], drop_first=True)

# Prepare Model Inputs
X = df.drop('loan_status', axis=1)
y = df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =============================================================================
# 2. MACHINE LEARNING MODEL BUILDING (LIGHTGBM)
# =============================================================================
print("\n--- Step 3: Training LightGBM Default Probability Model ---")

num_neg = y_train.value_counts()[0]
num_pos = y_train.value_counts()[1]
scale_weight = num_neg / num_pos
print(f"Applying scale_pos_weight: {scale_weight:.2f} due to class imbalance.")

model_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'n_estimators': 1500,
    'learning_rate': 0.03,
    'num_leaves': 63,
    'max_depth': -1,
    'seed': 42,
    'n_jobs': -1,
    'verbose': -1,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'scale_pos_weight': scale_weight
}

model = lgb.LGBMClassifier(**model_params)

model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          eval_metric='auc',
          callbacks=[lgb.early_stopping(50, verbose=False)])

print("Model training complete.")

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("\n--- Model Evaluation ---")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =============================================================================
# 3. FICO-SCALED CREDIT SCORE LOGIC
# =============================================================================

def calculate_credit_score(probabilities, base_score=600, pdo=20, odds_at_base=1.0):
    """
    Transforms a probability of default into a traditional scaled credit score (300 to 850).
    Uses the Score = Offset - Factor * ln(Odds).
    """
    # Factor calculation based on Points to Double Odds (PDO)
    factor = pdo / np.log(2)
    # Offset calculation
    offset = base_score - (factor * np.log(odds_at_base))
    
    # Clip probabilities heavily bounded to prevent infinite odds
    eps = 1e-6
    probs_clipped = np.clip(probabilities, eps, 1 - eps)
    
    # Calculate odds of Non-Default vs Default
    # standard scorecards define odds as Good / Bad = (1 - P(Default)) / P(Default)
    odds = (1 - probs_clipped) / probs_clipped
    
    # Scale scores
    scores = offset + factor * np.log(odds)
    
    # Clip final scores to a traditional 300 - 850 bounds range
    scores = np.clip(scores, 300, 850)
    return np.round(scores).astype(int)

print("\n--- Step 4: Applying Credit Score Calculation ---")

# Calculate credit scores for the test set
X_test_eval = X_test.copy()
X_test_eval['prob_default'] = y_pred_proba
X_test_eval['true_label'] = y_test.values
X_test_eval['Credit_Score'] = calculate_credit_score(y_pred_proba)

mean_score_good = X_test_eval[X_test_eval['true_label'] == 0]['Credit_Score'].mean()
mean_score_bad = X_test_eval[X_test_eval['true_label'] == 1]['Credit_Score'].mean()

print(f"Average Credit Score for Fully Paid Clients: {mean_score_good:.1f}")
print(f"Average Credit Score for Charged Off Clients: {mean_score_bad:.1f}")

# Optional: Plot the separation (if run interactively or saved)
plt.figure(figsize=(10, 6))
sns.histplot(data=X_test_eval, x='Credit_Score', hue='true_label', bins=50, kde=True, 
             palette={0: "green", 1: "red"})
plt.title("Generated Credit Score Distribution (Red=Defaulters, Green=Repaid)")
plt.xlabel("Credit Score (Base 600, PDO 20)")
plt.ylabel("Count")
plt.legend(["Charged Off (Default)", "Fully Paid"])
plt.savefig('credit_score_distribution.png')
print("Successfully generated and saved credit score visualization to 'credit_score_distribution.png'!")

print("=== Pipeline Finished! ===")
