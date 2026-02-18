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

# Import FraudDetectionPipeline from fraud_predictor module
try:
    from fraud_predictor import FraudDetectionPipeline
except ImportError as e:
    print(f"Warning: Could not import FraudDetectionPipeline: {e}")
    FraudDetectionPipeline = None


# === 🔧 AnomalyAdder (keeps pipeline unpickling compatible) ===
class AnomalyAdder(BaseEstimator, TransformerMixin):
    def __init__(self, iso_model, numeric_cols, scaler):
        self.iso_model = iso_model
        self.numeric_cols = numeric_cols
        self.scaler = scaler

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Ensure we don't modify the original dataframe if it's a view
        # Fill NaNs for numeric columns to match training behavior
        for col in self.numeric_cols:
            if col in X.columns:
                 X[col] = X[col].fillna(0)
            else:
                 X[col] = 0

        scaled = self.scaler.transform(X[self.numeric_cols])
        X['iso_score'] = self.iso_model.decision_function(scaled)
        X['autoenc_error'] = 0.0  # placeholder, as per notebook
        X['kmeans_flag'] = 0      # placeholder, as per notebook
        return X


# === 🩹 Fix for pickling ===
# If the pipeline was pickled in a script where AnomalyAdder was in __main__,
# we need to ensure it can be found there during unpickling.
import __main__
setattr(__main__, "AnomalyAdder", AnomalyAdder)

# === 📦 Helpers to load artifacts ===
@st.cache_resource
def load_ensemble_pipeline():
    """Load the ensemble fraud detection pipeline"""
    try:
        pipeline_path = "fraud_detector_pipeline.pkl"
        if not os.path.exists(pipeline_path):
            st.error(f"Ensemble pipeline NOT found at: {pipeline_path}. CWD: {os.getcwd()}")
            return None
        
        # Use simple load with filename, simpler for joblib
        pipeline = joblib.load(pipeline_path)
        
        st.success("✓ Ensemble pipeline loaded successfully")
        return pipeline
        
    except Exception as e:
        st.error(f"Failed to load ensemble pipeline: {e}")
        return None

@st.cache_data
def load_features():
    try:
        feature_path = "model_input_features.pkl"
        if os.path.exists(feature_path):
            return joblib.load(feature_path)
        return None
    except Exception as e:
        st.error(f"Failed to load features: {e}")
        return None


try:
    ensemble_pipeline = load_ensemble_pipeline()
    input_features = load_features()
    
    # At minimum, we need the ensemble pipeline
    if ensemble_pipeline is None:
        files = os.listdir()
        st.error(f"Failed to load ensemble pipeline. CWD: {os.getcwd()}. Files: {files}. Ensure 'fraud_detector_pipeline.pkl' is in the app folder.")
        st.stop()
        
    if input_features is None:
        st.error("Failed to load input features. Ensure 'model_input_features.pkl' is in the app folder.")
        st.stop()
        
except Exception as e:
    st.error(f"Initialization error: {e}")
    st.stop()


def preprocess_and_predict_ensemble(input_data):
    """Use the new ensemble pipeline for prediction"""
    if not ensemble_pipeline:
        return None
    
    try:
        # Create DataFrame from input dict
        df_input = pd.DataFrame([input_data])
        
        # Use ensemble pipeline
        # result = ensemble_pipeline.predict(df_input)
        # Extract probability of class 1
        probs = ensemble_pipeline.predict_proba(df_input)
        prob = float(probs[0][1])
        return prob
    except Exception as e:
        st.warning(f"Ensemble prediction failed: {e}")
        return None



# Load production config (threshold, defaults)
try:
    with open('production_config.json', 'r') as f:
        production_config = json.load(f)
        default_threshold = float(production_config.get('optimal_threshold', 0.1))
except Exception:
    production_config = {}
    default_threshold = 0.1


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
    # Show available models
    models_available = []
    if ensemble_pipeline:
        models_available.append("✓ Pipeline")
    st.write("**Available models:**")
    for m in models_available:
        st.write(m)
    if ensemble_pipeline:
        st.markdown("*Using pipeline from notebook (RF + IsoForest)*")
    st.markdown("---")
    st.write("Artifacts loaded from app folder. Ensure pipeline files are present.")


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
    """Build input dict with user-provided features from UI"""
    return {
        'Account_Balance': account_balance,
        'Transaction_Amount': transaction_amount,
        'State': state,
        'Transaction_Time': transaction_time,
        'Age': age,
        'Transaction_Device': transaction_device,
        'Merchant_ID': merchant_id if merchant_id else 'Unknown',
        # Defaults for optional fields
        'Customer_ID': 0,
        'Customer_Name': 'User',
        'Gender': 0,
        'City': 'Unknown',
        'Bank_Branch': 'Unknown',
        'Account_Type': 'Checking',
        'Transaction_ID': 0,
        'Transaction_Date': pd.Timestamp.now().strftime('%Y-%m-%d'),
        'Transaction_Type': 'Purchase',
        'Merchant_Category': 'General',
        'Transaction_Location': 'Unknown',
        'Device_Type': transaction_device,
        'Transaction_Currency': 'USD',
        'Customer_Contact': '',
        'Transaction_Description': '',
        'Customer_Email': ''
    }

df_input = build_input_df()


# Session history
if 'history' not in st.session_state:
    st.session_state.history = []


def predict_and_display(input_data):
    """Use the ensemble pipeline for prediction"""
    prob = None
    model_used = None
    
    # Use ensemble pipeline (primary model)
    if ensemble_pipeline is not None:
        prob = preprocess_and_predict_ensemble(input_data)
        model_used = "Ensemble (RF + XGBoost + LR)"
    
    
    # If all predictions failed
    if prob is None:
        st.error("Prediction failed. Please check inputs and ensure the pipeline is available.")
        return
    
    # Use optimal threshold from config
    optimal_threshold = 0.05
    try:
        with open('production_config.json', 'r') as f:
            config = json.load(f)
            optimal_threshold = float(config.get('optimal_threshold', 0.05))
    except:
        pass
    
    pred_flag = int(prob >= optimal_threshold)
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
                'value': optimal_threshold * 100}},
        number={'suffix': "%"}
    )])
    fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Explanation text based on verdict and probability
    st.write("")
    if pred_flag:  # Fraud detected
        if prob >= 0.9:
            st.error("🛑 **CRITICAL FRAUD ALERT**: This transaction shows very strong fraud indicators and should be immediately blocked.")
        elif prob >= 0.7:
            st.error("🛑 **High Fraud Risk**: Multiple fraud signals detected. Recommend blocking or requiring verification.")
        else:
            st.warning("⚠️ **Moderate Fraud Risk**: This transaction exhibits suspicious characteristics. Recommend review before approval.")
    else:  # Legitimate
        if prob < 0.1:
            st.success("✅ **Very Safe**: This transaction appears completely normal with no red flags.")
        elif prob < 0.2:
            st.success("✅ **Safe**: This transaction is legitimate with minimal risk indicators.")
        else:
            st.info("ℹ️ **Generally Safe**: This transaction appears legitimate but has some slightly unusual characteristics. You may want to monitor it.")
    
    # Threshold indicator
    st.markdown(f"**Detection threshold**: {optimal_threshold*100:.1f}% — tuned model optimized for maximum fraud detection (F1 = 0.0955)")
    
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
        # Show which model was used
        st.info(f"✓ Prediction generated using: **{model_used}**")
    except Exception:
        pass

    # Save to session history
    st.session_state.history.insert(0, {
        'ts': datetime.utcnow().isoformat(),
        'prob': float(prob),
        'verdict': verdict,
        'threshold': float(optimal_threshold),
        'amount': float(input_data.get('Transaction_Amount', 0)),
        'device': input_data.get('Transaction_Device', 'Unknown')
    })


if st.button("Predict"):
    predict_and_display(df_input)


if len(st.session_state.history) > 0:
    st.markdown("---")
    st.subheader('Transaction History')
    
    hist_df = pd.DataFrame(st.session_state.history)
    
    # Format for display
    display_cols = ['ts', 'amount', 'device', 'verdict', 'prob']
    if all(col in hist_df.columns for col in display_cols):
        display_df = hist_df[display_cols].copy()
        display_df['prob_pct'] = (display_df['prob'] * 100).round(1)
        display_df = display_df[['ts', 'amount', 'device', 'verdict', 'prob_pct']].copy()
        display_df = display_df.rename(columns={
            'ts': 'Timestamp', 
            'amount': 'Amount ($)', 
            'device': 'Device', 
            'verdict': 'Result', 
            'prob_pct': 'Risk %'
        })
        
        # Advanced search and filter functionality
        col_search, col_filter = st.columns([3, 2])
        
        with col_search:
            search_term = st.text_input(
                "🔍 Search transactions",
                placeholder="Search by date, amount, device, result, or risk % (e.g., '1000', 'Mobile', 'Fraud', '75')"
            )
        
        with col_filter:
            filter_result = st.selectbox("Filter by result", ["All", "Fraud Detected", "Legitimate"])
        
        # Apply filters
        filtered_df = display_df.copy()
        
        # Search across all columns
        if search_term:
            search_lower = search_term.lower()
            mask = filtered_df.astype(str).apply(lambda x: x.str.contains(search_lower, case=False, na=False).any(), axis=1)
            filtered_df = filtered_df[mask]
        
        # Filter by result
        if filter_result != "All":
            filtered_df = filtered_df[filtered_df['Result'] == filter_result]
        
        # Display count and summary
        fraud_count = len(filtered_df[filtered_df['Result'] == 'Fraud Detected'])
        legit_count = len(filtered_df[filtered_df['Result'] == 'Legitimate'])
        
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.metric("Total Transactions", len(filtered_df))
        with col_info2:
            st.metric("Fraud Cases", fraud_count)
        with col_info3:
            st.metric("Legitimate", legit_count)
        
        # Display as interactive table
        st.dataframe(filtered_df, use_container_width=True, hide_index=True)
        
        # Download as CSV
        csv_data = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Results as CSV",
            data=csv_data,
            file_name=f"fraud_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
