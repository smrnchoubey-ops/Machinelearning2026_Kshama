import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# --- Page Config ---
st.set_page_config(
    page_title="Placement Predictor",
    page_icon="🎓",
    layout="centered"
)

# --- Clean CSS ---
st.markdown("""
<style>
    .stApp {
        background-color: #fafafa;
    }
    .main-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 4px;
    }
    .sub-title {
        font-size: 0.95rem;
        color: #64748b;
        margin-bottom: 20px;
    }
    .res-card {
        padding: 18px;
        border-radius: 10px;
        margin-top: 16px;
        font-size: 1.25rem;
        font-weight: 700;
        text-align: center;
    }
    .res-placed {
        background-color: #ecfdf5;
        color: #065f46;
        border: 1.5px solid #10b981;
    }
    .res-not-placed {
        background-color: #fef2f2;
        color: #991b1b;
        border: 1.5px solid #ef4444;
    }
</style>
""", unsafe_allow_html=True)


# --- Load Model (Fast & Cached) ---
@st.cache_resource
def get_model():
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    with open(model_path, "rb") as f:
        return pickle.load(f)

# Pre-fitted scaler parameters from training data split (instant execution)
SCALER_MEAN = np.array([6.01666667, 120.98888889])
SCALER_SCALE = np.array([1.1399074, 37.88975072])

model = get_model()


# --- UI Header ---
st.markdown('<div class="main-title">🎓 Student Placement Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Enter student metrics to get instant placement prediction</div>', unsafe_allow_html=True)
st.divider()


# --- Input Controls ---
col1, col2 = st.columns(2)

with col1:
    cgpa = st.number_input(
        "CGPA",
        min_value=0.0,
        max_value=10.0,
        value=7.0,
        step=0.1,
        format="%.2f",
        help="Scale 0.0 to 10.0"
    )

with col2:
    iq = st.number_input(
        "IQ Score",
        min_value=30.0,
        max_value=250.0,
        value=120.0,
        step=1.0,
        format="%.1f",
        help="Scale 30 to 250"
    )

st.write("")

# --- Instant Prediction ---
if st.button("Predict Placement", type="primary", use_container_width=True):
    # Instant standard scaling: (X - mean) / scale
    scaled_input = (np.array([[cgpa, iq]]) - SCALER_MEAN) / SCALER_SCALE
    
    # Model inference
    pred = int(model.predict(scaled_input)[0])
    probs = model.predict_proba(scaled_input)[0]
    placed_prob = probs[1] * 100.0

    st.write("")
    if pred == 1:
        st.markdown(
            '<div class="res-card res-placed">Prediction: Placed</div>',
            unsafe_allow_html=True
        )
        st.success(f"Placement Probability: **{placed_prob:.1f}%**")
    else:
        st.markdown(
            '<div class="res-card res-not-placed">Prediction: Not Placed</div>',
            unsafe_allow_html=True
        )
        st.error(f"Placement Probability: **{placed_prob:.1f}%** (Non-Placement: **{100.0 - placed_prob:.1f}%**)")
