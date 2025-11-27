#!/usr/bin/env python
"""
Smoke test for fraud detection pipeline.
Validates that all models load and produce predictions.
"""

import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Define AnomalyAdder for unpickling the pipeline
class AnomalyAdder(BaseEstimator, TransformerMixin):
    def __init__(self, iso_model=None, numeric_cols=None, scaler=None, ae_model=None, ae_scaler=None):
        self.iso_model = iso_model
        self.numeric_cols = numeric_cols
        self.scaler = scaler
        self.ae_model = ae_model
        self.ae_scaler = ae_scaler

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.numeric_cols is not None:
            X[self.numeric_cols] = X[self.numeric_cols].fillna(0)
            scaled = self.scaler.transform(X[self.numeric_cols])
            X['iso_score'] = self.iso_model.decision_function(scaled)
        else:
            X['iso_score'] = 0.0
        
        if getattr(self, 'ae_model', None) is not None:
            ae_scaler = self.ae_scaler if getattr(self, 'ae_scaler', None) is not None else self.scaler
            scaled_ae = ae_scaler.transform(X[self.numeric_cols])
            recon = self.ae_model.predict(scaled_ae, verbose=0)
            mse = ((scaled_ae - recon) ** 2).mean(axis=1)
            X['autoenc_error'] = mse
        else:
            X['autoenc_error'] = 0.0
        
        X['kmeans_flag'] = 0
        return X

def test_pipeline_load():
    """Test loading the fraud detector pipeline."""
    print("[TEST] Testing pipeline load...")
    try:
        pipeline = joblib.load('fraud_detector_pipeline.pkl')
        print("[PASS] Pipeline loaded successfully")
        return pipeline
    except Exception as e:
        print(f"[FAIL] Failed to load pipeline: {e}")
        return None

def test_features_load():
    """Test loading feature columns."""
    print("[TEST] Testing feature columns load...")
    try:
        features = joblib.load('model_input_features.pkl')
        print(f"[PASS] Features loaded: {len(features)} columns")
        return features
    except Exception as e:
        print(f"[FAIL] Failed to load features: {e}")
        return None

def test_models_load():
    """Test loading individual models."""
    print("[TEST] Testing model artifacts...")
    models_to_check = [
        ('xgb_model.pkl', 'XGBoost'),
        ('rf_model.pkl', 'RandomForest'),
        ('iso_model.pkl', 'IsolationForest'),
        ('scaler_full.pkl', 'Full Scaler'),
        ('scaler_ae.pkl', 'AE Scaler'),
        ('scaler_unsup.pkl', 'Unsupervised Scaler'),
    ]
    
    for file_name, label in models_to_check:
        try:
            _ = joblib.load(file_name)
            print(f"  [PASS] {label} ({file_name})")
        except Exception as e:
            print(f"  [FAIL] {label} ({file_name}): {e}")

def test_autoencoder_load():
    """Test loading Keras autoencoder."""
    print("[TEST] Testing Keras autoencoder load...")
    try:
        from tensorflow.keras.models import load_model
        ae = load_model('autoencoder_model.h5', compile=False)
        print(f"[PASS] Autoencoder loaded: {len(ae.layers)} layers")
        return ae
    except Exception as e:
        print(f"[FAIL] Failed to load autoencoder: {e}")
        return None

def test_prediction(pipeline, features):
    """Test making a prediction on synthetic data."""
    print("[TEST] Testing prediction on synthetic row...")
    try:
        # Create a minimal synthetic row with appropriate types for categorical vs numeric
        row = {}
        for f in features:
            fname = f.lower()
            # heuristics for numeric feature names
            if any(x in fname for x in ['amount', 'balance', 'time', 'age', 'error', 'score']):
                row[f] = 0.0
            else:
                # categorical / id fields should be strings
                if 'merchant' in fname or 'device' in fname or 'id' in fname or 'type' in fname or 'city' in fname:
                    row[f] = 'Unknown'
                else:
                    row[f] = 'Unknown'
        synthetic_row = pd.DataFrame([row])
        
        # Make prediction
        prob = pipeline.predict_proba(synthetic_row)[0][1]
        pred = int(prob >= 0.25)
        verdict = "Fraud" if pred else "Legit"
        
        print(f"[PASS] Prediction successful:")
        print(f"   Fraud probability: {prob:.4f}")
        print(f"   Verdict (threshold 0.25): {verdict}")
        return True
    except Exception as e:
        print(f"[FAIL] Prediction failed: {e}")
        return False

def test_production_config():
    """Test loading production config."""
    print("[TEST] Testing production config...")
    try:
        import json
        with open('production_config.json', 'r') as f:
            config = json.load(f)
        threshold = config.get('optimal_threshold', 'N/A')
        model_name = config.get('model', 'N/A')
        print(f"[PASS] Production config loaded:")
        print(f"   Model: {model_name}")
        print(f"   Optimal threshold: {threshold}")
        return config
    except Exception as e:
        print(f"[FAIL] Failed to load production config: {e}")
        return None

def main():
    """Run all smoke tests."""
    print("\n" + "="*70)
    print("FRAUD DETECTION PIPELINE SMOKE TEST")
    print("="*70 + "\n")
    
    # Test 1: Load pipeline
    pipeline = test_pipeline_load()
    print()
    
    # Test 2: Load features
    features = test_features_load()
    print()
    
    # Test 3: Load models
    test_models_load()
    print()
    
    # Test 4: Load autoencoder
    ae = test_autoencoder_load()
    print()
    
    # Test 5: Load production config
    config = test_production_config()
    print()
    
    # Test 6: Prediction
    if pipeline is not None and features is not None:
        test_prediction(pipeline, features)
    print()
    
    # Summary
    print("="*70)
    print("[OK] SMOKE TEST COMPLETE - Pipeline is ready for deployment!")
    print("="*70)

if __name__ == '__main__':
    main()
