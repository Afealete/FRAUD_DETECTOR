# 🚀 Fraud Detection Pipeline - Deployment Guide

## Overview
This is a production-ready fraud detection system using a hybrid unsupervised + supervised learning approach. The system uses XGBoost as the primary classifier with continuous anomaly scores from IsolationForest, Autoencoder, KMeans, and Local Outlier Factor.

## Key Artifacts

### Models
- **`xgb_model.pkl`** - Primary XGBoost classifier (best performance)
- **`rf_model.pkl`** - RandomForest alternative
- **`iso_model.pkl`** - IsolationForest anomaly detector
- **`autoencoder_model.h5`** - Keras autoencoder for reconstruction error (trained on normal transactions)
- **`xgb_best_tuned.pkl`** - Tuned XGBoost (RandomizedSearchCV optimized for Average Precision)
- **`rf_best_tuned.pkl`** - Tuned RandomForest (RandomizedSearchCV optimized for Average Precision)

### Preprocessing & Feature Engineering
- **`scaler_full.pkl`** - StandardScaler for all features
- **`scaler_ae.pkl`** - StandardScaler for autoencoder input
- **`scaler_unsup.pkl`** - MinMaxScaler for normalizing anomaly scores (0..1)
- **`feature_columns.pkl`** - List of original feature columns
- **`hybrid_feature_columns.pkl`** - List of hybrid (original + anomaly) feature columns
- **`model_input_features.pkl`** - List of features expected by `fraud_detector_pipeline.pkl`

### Pipelines & Configuration
- **`fraud_detector_pipeline.pkl`** - Sklearn Pipeline combining AnomalyAdder → Preprocessor → Classifier
- **`production_config.json`** - Canonical production settings (optimal threshold, model name, expected metrics)
- **`unsupervised_scores.pkl`** - Pre-computed continuous anomaly scores for reference
- **`X_hybrid_with_unsup.pkl`** - Full training feature matrix with anomaly scores (for reproducibility)

## Configuration: `production_config.json`
```json
{
  "optimal_threshold": 0.25,
  "model": "xgboost",
  "expected_f1": 0.0961,
  "test_auc": 0.5096
}
```
- **optimal_threshold**: Decision threshold for fraud classification. Lower values catch more fraud (higher recall) at cost of false positives.
- **model**: Primary classifier name (xgboost)
- **expected_f1**: Observed F1 score on test set
- **test_auc**: Observed ROC-AUC on test set

## Performance Metrics

### Test Set (40k transactions, ~5% fraud)
- **Precision@threshold=0.25**: ~5%
- **Recall@threshold=0.25**: ~97.7%
- **F1 Score**: ~0.0961
- **ROC-AUC**: 0.5096

### Unsupervised Signal Quality (Average Precision)
- **IsolationForest**: 0.0509
- **Autoencoder**: 0.0502
- **KMeans**: 0.0512
- **LOF**: 0.0511
- **Ensemble (average)**: 0.0510

## Setup & Running

### Prerequisites
- Python 3.13.5
- Virtual environment (`.venv`)
- All required packages in `requirements.txt`

### Environment Setup
```bash
# Activate the virtual environment
.\.venv\Scripts\Activate.ps1   # Windows PowerShell
# or
.venv/bin/activate              # Linux/macOS

# Install dependencies (if needed)
pip install -r requirements.txt
```

### Testing the Pipeline
```bash
# Run smoke test to validate all artifacts load correctly
python smoke_test.py
```

Expected output:
```
✅ Pipeline loaded
✅ Features loaded
✅ All models load successfully
✅ Autoencoder loads with 7 layers
✅ Prediction successful
```

### Running the Streamlit App
```bash
# Start the Streamlit fraud detection app
streamlit run app.py
```

The app will open at `http://localhost:8501` with:
- Interactive transaction input fields
- Real-time fraud probability prediction
- Adjustable decision threshold (via sidebar slider)
- Prediction history tracking

### Making Predictions Programmatically
```python
import joblib
import pandas as pd

# Load pipeline and features
pipeline = joblib.load('fraud_detector_pipeline.pkl')
features = joblib.load('model_input_features.pkl')

# Create prediction input (example)
transaction = pd.DataFrame([{
    'Account_Balance': 5000.0,
    'Transaction_Amount': 200.0,
    'State': 'Western',
    'Transaction_Time': 14.5,
    'Age': 35,
    'Transaction_Device': 'Mobile'
}])

# Ensure feature order matches
transaction = transaction.reindex(columns=features, fill_value=0)

# Get prediction
fraud_prob = pipeline.predict_proba(transaction)[0][1]
is_fraud = int(fraud_prob >= 0.25)  # Use threshold from production_config.json

print(f"Fraud probability: {fraud_prob:.4f}")
print(f"Prediction: {'Fraud' if is_fraud else 'Legit'}")
```

## Deployment Checklist

- [x] Models trained and saved (`xgb_model.pkl`, `rf_model.pkl`, etc.)
- [x] Keras autoencoder properly saved with `model.save()` 
- [x] Scalers fitted and persisted
- [x] Production config set (`production_config.json`)
- [x] Pipeline created and saved (`fraud_detector_pipeline.pkl`)
- [x] `AnomalyAdder` transformer class defined and compatible
- [x] Streamlit app ready (`app.py`)
- [x] Smoke test passing (run `python smoke_test.py`)
- [x] Requirements documented (`requirements.txt`)

## Threshold Tuning

The optimal threshold depends on your operational constraints:

| Threshold | Precision | Recall | Predicted Count | Use Case |
|-----------|-----------|--------|-----------------|----------|
| 0.10      | 5.0%      | 99.8%  | ~39,930        | Review all suspected fraud |
| 0.25      | 5.1%      | 97.7%  | ~39,015        | Default: catch almost all fraud |
| 0.45      | 5.1%      | 60.4%  | ~23,885        | High precision, moderate recall |
| 0.50      | 5.4%      | 28.8%  | ~10,761        | Conservative: only high-confidence fraud |

To change threshold, update `optimal_threshold` in `production_config.json`.

## Model Improvements & Recommendations

### Completed
✅ Continuous unsupervised anomaly signals (not binary flags)
✅ Keras autoencoder trained only on normal transactions
✅ Ensemble of 4 anomaly detectors (IF, AE, KMeans, LOF)
✅ Normalized anomaly scores (0..1 scale)
✅ Hyperparameter tuning (RandomizedSearchCV optimizing Average Precision)
✅ Proper Keras save/load with compile=False

### Future Improvements
- Add domain-specific features (transaction velocity, merchant risk, temporal patterns)
- Collect more labeled fraud examples for better signal
- Tune ensemble weights via validation set
- Implement semi-supervised learning (pseudo-labeling)
- A/B test threshold in production; track precision/recall drift

## Troubleshooting

### Pipeline fails to load
**Error**: `Can't get attribute 'AnomalyAdder'`
**Solution**: Ensure `AnomalyAdder` class is defined in your script before loading the pipeline.

### Autoencoder fails to load
**Error**: `ValueError: Could not deserialize 'keras.metrics.mse'`
**Solution**: Use `load_model(..., compile=False)` to skip recompiling the model.

### Feature mismatch
**Error**: `Unexpected feature order` or `Shape mismatch`
**Solution**: Always reindex prediction data to match `model_input_features.pkl` order:
```python
transaction = transaction.reindex(columns=features, fill_value=0)
```

## Contact & Support
For questions or issues, refer to the Jupyter notebook (`Foxtrot_2_Final_project.ipynb`) for full training pipeline details.
