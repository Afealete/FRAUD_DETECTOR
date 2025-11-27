import joblib, pandas as pd
print('Loading pipeline...')
pipe = joblib.load('fraud_detector_pipeline.pkl')
print('Pipeline loaded, steps =', list(pipe.named_steps.keys()))
# Create a dummy input using model_input_features
features = joblib.load('model_input_features.pkl')
row = pd.DataFrame([{f:0 for f in features}])
print('Row columns:', row.shape)
prob = pipe.predict_proba(row)[0][1]
print('Predict proba (attempt):', prob)
