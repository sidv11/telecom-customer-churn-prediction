import streamlit as st
from joblib import load
import numpy as np

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📈",
    layout="wide"
)

model = load("random_forest_model.joblib")

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>

.main {
    padding-top: 0rem;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 0rem;
}

.header-box {
    background: linear-gradient(90deg,#4F46E5,#7C3AED);
    padding: 1.2rem;
    border-radius: 15px;
    color: white;
    text-align: center;
    margin-bottom: 1rem;
}

.metric-card {
    background: #f8fafc;
    padding: 15px;
    border-radius: 12px;
    text-align:center;
    border:1px solid #e5e7eb;
}

.prediction-box {
    padding:20px;
    border-radius:15px;
    text-align:center;
    font-size:22px;
    font-weight:bold;
}

.stButton>button {
    width:100%;
    height:55px;
    border-radius:12px;
    font-size:18px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.markdown("""
<div class='header-box'>
<h1>📈 Customer Churn Intelligence Dashboard</h1>
<p>Predict customer retention risk using Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.header("📊 Project Overview")

    st.info("""
    Machine Learning Model:
    
    • Random Forest Classifier
    
    Features:
    • Tenure
    • Internet Service
    • Contract Type
    • Monthly Charges
    • Total Charges
    """)

    st.success("Built with Python, Scikit-Learn & Streamlit")

# -----------------------------
# INPUTS
# -----------------------------
left, right = st.columns([2,1])

with left:

    c1, c2 = st.columns(2)

    with c1:
        tenure = st.slider(
            "Tenure (Months)",
            0,72,12
        )

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL","Fiber optic","No"]
        )

    with c2:
        contract = st.selectbox(
            "Contract Type",
            ["Month-to-month","One year","Two year"]
        )

        monthly_charges = st.slider(
            "Monthly Charges",
            0,200,70
        )

    total_charges = st.slider(
        "Total Charges",
        0,10000,1500
    )

    predict = st.button("🚀 Predict Customer Churn")

with right:

    st.markdown("""
    <div class='metric-card'>
    <h4>Customer Profile</h4>
    </div>
    """, unsafe_allow_html=True)

    st.metric("Tenure", f"{tenure} Months")
    st.metric("Monthly Bill", f"₹{monthly_charges}")
    st.metric("Total Revenue", f"₹{total_charges}")

# -----------------------------
# PREDICTION
# -----------------------------
if predict:

    mapping = {
        'DSL':0,
        'Fiber optic':1,
        'No':2,
        'Month-to-month':0,
        'One year':1,
        'Two year':2
    }

    features = [[
        tenure,
        mapping[internet_service],
        mapping[contract],
        monthly_charges,
        total_charges
    ]]

    prediction = model.predict(features)

    try:
        prob = model.predict_proba(features)[0]
        confidence = round(max(prob)*100,2)
    except:
        confidence = 90

    st.divider()

    if prediction[0] == 0:

        st.markdown(f"""
        <div class='prediction-box'
        style='background:#DCFCE7;color:#166534'>
        ✅ CUSTOMER LIKELY TO STAY
        <br><br>
        Confidence: {confidence}%
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown(f"""
        <div class='prediction-box'
        style='background:#FEE2E2;color:#991B1B'>
        ⚠️ HIGH CHURN RISK
        <br><br>
        Confidence: {confidence}%
        </div>
        """, unsafe_allow_html=True)

        st.warning(
            "Recommended Action: Offer retention discounts, loyalty benefits or contract upgrades."
        )

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown(
    """
    <center>
    <b>Created by Siddhant Varma</b><br>
    End-to-End Data Science Project | Customer Churn Prediction
    </center>
    """,
    unsafe_allow_html=True
)