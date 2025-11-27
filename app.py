import streamlit as st
import pandas as pd
import joblib
import os
import json
import pickle
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px


# === 🔧 AnomalyAdder (keeps pipeline unpickling compatible) ===
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
            try:
                scaled = self.scaler.transform(X[self.numeric_cols])
                # IsolationForest: higher -> more normal, invert to represent anomaly magnitude
                X['iso_score'] = -self.iso_model.decision_function(scaled)
            except Exception:
                X['iso_score'] = 0.0
        else:
            X['iso_score'] = 0.0

        # Autoencoder reconstruction error
        try:
            if getattr(self, 'ae_model', None) is not None:
                ae_scaler = self.ae_scaler if getattr(self, 'ae_scaler', None) is not None else self.scaler
                scaled_ae = ae_scaler.transform(X[self.numeric_cols])
                recon = self.ae_model.predict(scaled_ae, verbose=0)
                mse = np.mean((scaled_ae - recon) ** 2, axis=1)
                X['autoenc_error'] = mse
            else:
                X['autoenc_error'] = 0.0
        except Exception:
            X['autoenc_error'] = 0.0

        X['kmeans_flag'] = 0
        return X


# === 📦 Helpers to load artifacts ===
@st.cache_resource
def load_pipeline():
    try:
        return joblib.load("fraud_detector_pipeline.pkl")
    except Exception:
        try:
            with open("fraud_detector_pipeline_pickle.pkl", "rb") as f:
                return pickle.load(f)
        except Exception as e:
            st.error(f"Failed to load pipeline: {e}")
            raise


@st.cache_data
def load_features():
    try:
        return joblib.load("model_input_features.pkl")
    except Exception as e:
        st.error(f"Failed to load feature list: {e}")
        raise


try:
    pipeline = load_pipeline()
    input_features = load_features()
except Exception:
    st.error("Failed to initialize app. Regenerate the pipeline in the notebook and ensure artifacts are in the app folder.")
    st.stop()


# Attach autoencoder and scaler if available
try:
    from tensorflow.keras.models import load_model
    ae_model = load_model('autoencoder_model.h5', compile=False)
    scaler_ae = joblib.load('scaler_ae.pkl')
    if hasattr(pipeline, 'named_steps') and 'add_anomalies' in pipeline.named_steps:
        pipeline.named_steps['add_anomalies'].ae_model = ae_model
        pipeline.named_steps['add_anomalies'].ae_scaler = scaler_ae
except Exception:
    ae_model = None
    scaler_ae = None


# Load production config (threshold, defaults)
try:
    with open('production_config.json', 'r') as f:
        production_config = json.load(f)
        default_threshold = float(production_config.get('optimal_threshold', 0.25))
except Exception:
    production_config = {}
    default_threshold = 0.25


# === 🎨 Streamlit UI ===
st.set_page_config(page_title="Foxtrot Fraud Detector", page_icon=":shield:", layout="wide")

logo_path = 'logo.png'
col_logo, col_title = st.columns([1, 6])
with col_logo:
    if os.path.exists(logo_path):
        st.image(logo_path, width=80)
    else:
        st.image("https://placehold.co/80x80?text=Foxtrot", width=80)
with col_title:
    st.title("Foxtrot Fraud Detection — Live Prediction")
    st.caption("Hybrid unsupervised + supervised fraud detection — production UI")

st.markdown("---")

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Detection threshold", 0.0, 1.0, value=default_threshold, step=0.01)
    st.markdown("---")
    st.subheader("Runtime")
    # Show the production model name (read-only) to avoid confusing choices
    st.write("**Production model:**", production_config.get('model', 'xgboost'))
    st.markdown("---")
    st.write("Artifacts loaded from app folder. Ensure `fraud_detector_pipeline.pkl` and scalers are present.")


# --- Input form (horizontal layout) ---
st.subheader("Transaction Inputs")

# Row 1: Account Balance and Transaction Amount
col1, col2 = st.columns(2)
with col1:
    account_balance = st.number_input("Account Balance ($)", min_value=0.0, value=1000.0, format="%.2f")
with col2:
    transaction_amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0, format="%.2f")

# Row 2: State and Device
col3, col4 = st.columns(2)
with col3:
    state = st.selectbox("State", ["Western", "Eastern", "Greater Accra", "Unknown"], key="state_sel")
with col4:
    transaction_device = st.selectbox("Device", ["Mobile", "Web", "POS"], key="device_sel")

# Row 3: Age and Time
col5, col6 = st.columns(2)
with col5:
    age = st.number_input("Age", min_value=15, max_value=120, value=30, step=1)
with col6:
    transaction_time = st.number_input("Time (24h, e.g. 14.5)", min_value=0.0, max_value=24.0, value=13.5, step=0.5, format="%.1f")

# Row 4: Merchant ID
col7, col8 = st.columns([2, 1])
with col7:
    merchant_id = st.text_input("Merchant ID (optional)", value="", placeholder="e.g., GenericMerchant001")
with col8:
    st.write("")  # spacer


def build_input_df():
    data = {
        'Account_Balance': [account_balance],
        'Transaction_Amount': [transaction_amount],
        'State': [state],
        'Transaction_Time': [transaction_time],
        'Age': [age],
        'Transaction_Device': [transaction_device],
        'Merchant_ID': [merchant_id]
    }
    df_in = pd.DataFrame(data)
    # don't accept technical anomaly hints from users; the pipeline will compute any needed signals
    df_in['autoenc_error'] = 0.0
    df_in['iso_score'] = 0.0
    return df_in

df_input = build_input_df()


# Session history
if 'history' not in st.session_state:
    st.session_state.history = []


def predict_and_display(df_model):
    # Ensure columns in correct order for pipeline
    try:
        df_model = df_model.reindex(columns=input_features, fill_value=0)
    except Exception:
        pass

    prob = pipeline.predict_proba(df_model)[0][1]
    pred_flag = int(prob >= threshold)
    verdict = "Fraud Detected" if pred_flag else "Legitimate"
    color = "red" if pred_flag else "green"

    # Results layout
    st.markdown("---")
    st.subheader("Prediction Result")
    
    # Main verdict with large display
    col_verdict, col_prob = st.columns([2, 2])
    with col_verdict:
        st.markdown(f"<h2 style='color:{color};'>{verdict}</h2>", unsafe_allow_html=True)
    with col_prob:
        st.metric("Risk Score", f"{prob*100:.1f}%")
    
    # Gauge chart showing fraud risk
    fig_gauge = go.Figure(data=[go.Indicator(
        mode="gauge+number+delta",
        value=prob * 100,
        title={'text': "Fraud Risk Level"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 33], 'color': "#D3F8D3"},
                {'range': [33, 66], 'color': "#FFF4E6"},
                {'range': [66, 100], 'color': "#FFE6E6"}],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': threshold * 100}},
        number={'suffix': "%"}
    )])
    fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Explanation text based on threshold
    st.write("")
    if prob < 0.2:
        st.success("✅ **Very Safe**: This transaction looks normal. No red flags detected.")
    elif prob < 0.4:
        st.info("ℹ️ **Safe**: This transaction appears legitimate, though it has some unusual characteristics.")
    elif prob < 0.6:
        st.warning("⚠️ **Caution**: This transaction has moderate risk factors. It may need review.")
    else:
        st.error("🛑 **High Risk**: This transaction shows significant fraud indicators and should be reviewed.")
    
    # Threshold indicator
    st.markdown(f"**Detection threshold**: {threshold*100:.0f}% risk level")
    
    st.markdown("---")
    st.subheader("Transaction Analysis")
    
    # Transaction profile summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Balance", f"${account_balance:.2f}")
    with col2:
        st.metric("Amount", f"${transaction_amount:.2f}")
    with col3:
        ratio = (transaction_amount / max(account_balance, 1)) * 100
        st.metric("% of Balance", f"{ratio:.1f}%")
    with col4:
        st.metric("Time of Day", f"{int(transaction_time)}:{int((transaction_time % 1) * 60):02d}")
    
    # Chart: Transaction Amount vs Account Balance
    fig_ratio = go.Figure()
    fig_ratio.add_trace(go.Bar(
        x=['Account Balance', 'Transaction Amount'],
        y=[account_balance, transaction_amount],
        marker=dict(color=['green', 'blue']),
        text=[f"${account_balance:.0f}", f"${transaction_amount:.0f}"],
        textposition='auto'
    ))
    fig_ratio.update_layout(
        title="Transaction Size Comparison",
        xaxis_title="",
        yaxis_title="Amount ($)",
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False
    )
    st.plotly_chart(fig_ratio, use_container_width=True)
    
    # Anomaly diagnostics (if available)
    st.markdown("---")
    st.subheader("System Analysis")
    try:
        if hasattr(pipeline, 'named_steps') and 'add_anomalies' in pipeline.named_steps:
            aug = pipeline.named_steps['add_anomalies'].transform(df_model.copy())
            try:
                scaler_unsup = joblib.load('scaler_unsup.pkl')
                raw = aug[['iso_score', 'autoenc_error', 'kmeans_flag']].fillna(0)
                scaled = scaler_unsup.transform(raw)
                ensemble = float(np.mean(scaled, axis=1)[0])
                
                # Show anomaly score as a simple number without technical jargon
                col_anom1, col_anom2 = st.columns([2, 2])
                with col_anom1:
                    # Create a simple interpretation
                    anom_pct = min(100, ensemble * 100)
                    st.metric("Anomaly Detection", f"{anom_pct:.0f}%", help="How unusual this transaction is")
                with col_anom2:
                    if ensemble < 0.3:
                        st.success("Normal pattern detected")
                    elif ensemble < 0.6:
                        st.info("Slightly unusual")
                    else:
                        st.warning("Very unusual")
            except Exception:
                pass
    except Exception:
        pass

    # Save to session history
    st.session_state.history.insert(0, {
        'ts': datetime.utcnow().isoformat(),
        'prob': float(prob),
        'verdict': verdict,
        'threshold': float(threshold),
        'amount': float(transaction_amount),
        'device': transaction_device
    })


if st.button("Predict"):
    predict_and_display(df_input)


if len(st.session_state.history) > 0:
    st.markdown("---")
    st.subheader('Recent Predictions')
    hist_df = pd.DataFrame(st.session_state.history).head(10)
    # Format the display
    display_cols = ['ts', 'amount', 'device', 'verdict', 'prob']
    if 'amount' in hist_df.columns and 'device' in hist_df.columns:
        display_df = hist_df[display_cols].copy()
        display_df['prob'] = (display_df['prob'] * 100).round(1).astype(str) + '%'
        display_df = display_df.rename(columns={'ts': 'Time', 'amount': 'Amount', 'device': 'Device', 'verdict': 'Result', 'prob': 'Risk'})
        st.table(display_df)
