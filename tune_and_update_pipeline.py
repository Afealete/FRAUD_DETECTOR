import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score
import warnings
warnings.filterwarnings('ignore')

# Load saved hybrid features
print('Loading X_hybrid_with_unsup.pkl')
X = joblib.load('X_hybrid_with_unsup.pkl')
print('X shape:', getattr(X, 'shape', None))

# Load labels
print('Loading labels from CSV')
df = pd.read_csv('Bank_Transaction_Fraud_Detection.csv')
y = df['Is_Fraud'].reset_index(drop=True)

# Ensure X index aligns with y length
if len(X) != len(y):
    print('Warning: X and y length mismatch, attempting to align by taking first n rows')
    n = min(len(X), len(y))
    X = X.iloc[:n].reset_index(drop=True)
    y = y.iloc[:n].reset_index(drop=True)

# Create a single stratified split matching notebook settings
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in sss.split(X, y):
    X_train, X_test = X.iloc[train_idx].reset_index(drop=True), X.iloc[test_idx].reset_index(drop=True)
    y_train, y_test = y.iloc[train_idx].reset_index(drop=True), y.iloc[test_idx].reset_index(drop=True)

print('Train/test shapes:', X_train.shape, X_test.shape)

# Sample smaller subset for tuning
sample_frac_tune = 0.25
X_tune = X_train.sample(frac=sample_frac_tune, random_state=42).reset_index(drop=True)
y_tune = y_train.loc[X_tune.index].reset_index(drop=True)
X_tr_inner, X_val_inner, y_tr_inner, y_val_inner = train_test_split(X_tune, y_tune, stratify=y_tune, test_size=0.2, random_state=42)

# RandomForest tuning
print('\nTuning RandomForest...')
rf_param_dist = {
    'n_estimators': [100, 200, 400],
    'max_depth': [6, 12, None],
    'max_features': ['sqrt', 'log2'] ,
    'min_samples_split': [2, 5]
}
rf_base = RandomForestClassifier(class_weight='balanced_subsample', n_jobs=-1, random_state=42)
rf_rs = RandomizedSearchCV(rf_base, rf_param_dist, n_iter=6, scoring='average_precision', cv=3, random_state=42, n_jobs=-1, verbose=1)
rf_rs.fit(X_tr_inner, y_tr_inner)
best_rf = rf_rs.best_estimator_
print('RF best params:', rf_rs.best_params_, 'CV AP:', rf_rs.best_score_)

# Evaluate on inner validation
yval_prob_rf = best_rf.predict_proba(X_val_inner)[:, 1]
print('RF val AP:', round(average_precision_score(y_val_inner, yval_prob_rf), 4))

# XGBoost tuning
print('\nTuning XGBoost...')
xgb_param_dist = {
    'n_estimators': [100, 200, 400],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0],
}
xgb_base = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, verbosity=0)
xgb_rs = RandomizedSearchCV(xgb_base, xgb_param_dist, n_iter=6, scoring='average_precision', cv=3, random_state=42, n_jobs=-1, verbose=1)
xgb_rs.fit(X_tr_inner, y_tr_inner)
best_xgb = xgb_rs.best_estimator_
print('XGB best params:', xgb_rs.best_params_, 'CV AP:', xgb_rs.best_score_)

# Evaluate on inner validation
yval_prob_xgb = best_xgb.predict_proba(X_val_inner)[:, 1]
print('XGB val AP:', round(average_precision_score(y_val_inner, yval_prob_xgb), 4))

# Save tuned models
joblib.dump(best_rf, 'rf_best_tuned.pkl')
joblib.dump(best_xgb, 'xgb_best_tuned.pkl')
print('Saved tuned models: rf_best_tuned.pkl, xgb_best_tuned.pkl')

# Try to update pipeline by replacing classifier step
try:
    pipeline = joblib.load('fraud_detector_pipeline.pkl')
    print('Loaded existing pipeline')
    # Replace classifier if step exists
    if 'classifier' in pipeline.named_steps:
        pipeline.named_steps['classifier'] = best_xgb
        joblib.dump(pipeline, 'fraud_detector_pipeline.pkl')
        print('Updated fraud_detector_pipeline.pkl with tuned XGBoost classifier')
    else:
        print('Pipeline has no "classifier" step; skipping replacement')
except Exception as e:
    print('Failed to update pipeline:', e)

print('\nDone')
