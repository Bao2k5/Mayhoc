import streamlit as st
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go

st.set_page_config(
    page_title="Credit Card Default Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme selector in sidebar
theme = st.sidebar.radio("🎨 Theme", ["Light", "Dark"], horizontal=True, key="theme_select")

def apply_theme(theme_name: str):
    if theme_name == "Dark":
        css = """<style>
* { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
html, body, .main, .block-container { background: linear-gradient(135deg, #0a0e27 0%, #0f1535 100%) !important; color: #e8f0ff !important; }
.stSidebar { background: linear-gradient(180deg, #0f1535 0%, #0a0e27 100%) !important; border-right: 1px solid rgba(96, 165, 250, 0.1); }
.stButton>button { background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important; color: white !important; border-radius: 8px !important; padding: 12px 24px !important; }
.stNumberInput>div>div>input, .stSelectbox>div>div>select { background: rgba(30, 41, 59, 0.8) !important; color: #e8f0ff !important; border-radius: 8px !important; padding: 8px 12px !important; }
.success-box { padding: 16px; border-radius: 8px; background: rgba(16,185,129,0.06); border: 1px solid #10b981; }
.danger-box { padding: 16px; border-radius: 8px; background: rgba(239,68,68,0.06); border: 1px solid #ef4444; }
</style>"""
    else:
        css = """<style>
* { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
html, body, .main, .block-container { background: linear-gradient(135deg, #f8fafc 0%, #f0f4f8 100%) !important; color: #1e293b !important; }
.stSidebar { background: linear-gradient(180deg, #f5f7fa 0%, #ffffff 100%) !important; border-right: 1px solid #e2e8f0; }
.stButton>button { background: linear-gradient(135deg, #0366d6 0%, #0256c7 100%) !important; color: white !important; border-radius: 8px !important; padding: 12px 24px !important; }
.stNumberInput>div>div>input, .stSelectbox>div>div>select { background: rgba(248,250,252,0.9) !important; color: #1e293b !important; border-radius: 8px !important; padding: 8px 12px !important; }
.success-box { padding: 16px; border-radius: 8px; background: rgba(34,197,94,0.06); border: 1px solid #22c55e; }
.danger-box { padding: 16px; border-radius: 8px; background: rgba(248,81,73,0.06); border: 1px solid #ef4444; }
</style>"""
    st.markdown(css, unsafe_allow_html=True)

apply_theme(theme)

@st.cache_resource
def load_model():
    model_path = Path('models/best_model.pkl')
    scaler_path = Path('models/scaler.pkl')
    feature_path = Path('data/processed/feature_names.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(feature_path, 'rb') as f:
        feature_names = pickle.load(f)
    return model, scaler, feature_names


def engineer_features(df):
    pay_cols = [f'PAY_{i}' for i in [0, 2, 3, 4, 5, 6]]
    df['MAX_PAY_DELAY'] = df[pay_cols].max(axis=1)
    df['AVG_PAY_DELAY'] = df[pay_cols].mean(axis=1)
    df['PAY_DELAY_SUM'] = df[pay_cols].sum(axis=1)
    bill_cols = [f'BILL_AMT{i}' for i in range(1, 7)]
    df['AVG_BILL_AMT'] = df[bill_cols].mean(axis=1)
    df['MAX_BILL_AMT'] = df[bill_cols].max(axis=1)
    df['MIN_BILL_AMT'] = df[bill_cols].min(axis=1)
    df['STD_BILL_AMT'] = df[bill_cols].std(axis=1)
    df['BILL_AMT_TREND'] = df['BILL_AMT1'] - df['BILL_AMT6']
    pay_amt_cols = [f'PAY_AMT{i}' for i in range(1, 7)]
    df['AVG_PAY_AMT'] = df[pay_amt_cols].mean(axis=1)
    df['MAX_PAY_AMT'] = df[pay_amt_cols].max(axis=1)
    df['MIN_PAY_AMT'] = df[pay_amt_cols].min(axis=1)
    df['STD_PAY_AMT'] = df[pay_amt_cols].std(axis=1)
    df['PAY_AMT_TREND'] = df['PAY_AMT1'] - df['PAY_AMT6']
    df['UTILIZATION_RATE'] = (df['AVG_BILL_AMT'] / df['LIMIT_BAL'] * 100).clip(upper=100)
    df['MAX_UTILIZATION'] = (df['MAX_BILL_AMT'] / df['LIMIT_BAL'] * 100).clip(upper=100)
    df['PAYMENT_RATIO'] = np.where(df['AVG_BILL_AMT'] > 0, (df['AVG_PAY_AMT'] / df['AVG_BILL_AMT'] * 100).clip(upper=200), 0)
    df['TIMES_DELAYED'] = (df[pay_cols] > 0).sum(axis=1)
    df['CREDIT_USAGE_CONSISTENCY'] = 1 / (1 + df['STD_BILL_AMT'])
    df['PAYMENT_CONSISTENCY'] = 1 / (1 + df['STD_PAY_AMT'])
    df['AGE_LIMIT'] = df['AGE'] * df['LIMIT_BAL'] / 1000000
    df['EDUCATION_LIMIT'] = df['EDUCATION'] * df['LIMIT_BAL'] / 100000
    df['PAY_TO_LIMIT_RATIO'] = df['AVG_PAY_AMT'] / df['LIMIT_BAL']
    df['RECENT_PAY_TREND'] = df['PAY_0'] - df['PAY_2']
    df['RECENT_BILL_TREND'] = df['BILL_AMT1'] - df['BILL_AMT2']
    df['RECENT_PAYMENT_TREND'] = df['PAY_AMT1'] - df['PAY_AMT2']
    return df


def predict_default(customer_data, model, scaler, feature_names):
    df = pd.DataFrame([customer_data])
    df = engineer_features(df)
    for feature in feature_names:
        if feature not in df.columns:
            df[feature] = 0
    df = df[feature_names]
    X_scaled = scaler.transform(df)
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0]
    return prediction, probability


def create_gauge(probability, title, theme_name):
    value = float(probability) * 100
    font_color = "#60a5fa" if theme_name == "Dark" else "#0366d6"
    bar_color = "#3b82f6" if theme_name == "Dark" else "#0366d6"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16, 'color': font_color}},
        delta={'reference': 50, 'font': {'color': font_color}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': font_color},
            'bar': {'color': bar_color, 'thickness': 0.6},
            'bgcolor': 'rgba(0,0,0,0)',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 20], 'color': 'rgba(34, 197, 94, 0.2)'},
                {'range': [20, 40], 'color': 'rgba(59, 130, 246, 0.15)'},
                {'range': [40, 60], 'color': 'rgba(251, 146, 60, 0.15)'},
                {'range': [60, 80], 'color': 'rgba(244, 63, 94, 0.12)'},
                {'range': [80, 100], 'color': 'rgba(239, 68, 68, 0.2)'}
            ],
            'threshold': {'line': {'color': "#ef4444", 'width': 2}, 'thickness': 0.7, 'value': 50}
        }
    ))
    fig.update_layout(height=260, margin=dict(l=8, r=8, t=24, b=8), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': font_color, 'family': "Segoe UI"})
    return fig


def main():
    with st.sidebar:
        st.markdown("### 📊 Model Stats")
        col1, col2 = st.columns(2)
        col1.metric("Accuracy", "80.43%", "✓")
        col2.metric("ROC-AUC", "0.7692", "✓")
        col1.metric("F1-Score", "0.5092", "✓")
        col2.metric("Precision", "62.3%", "✓")
        st.markdown("---")
        st.markdown("### 💼 Business Impact")
        st.metric("Savings", "19B VND", "+41% Risk Reduction")
        st.markdown("---")
        st.markdown("### ⚙️ Model Info")
        st.caption("**Algorithm:** LightGBM\n**Training Samples:** 30,000\n**Features:** 30+ Engineered")

    st.markdown("<div style='text-align: center; margin-bottom: 2rem;'><h1>🏦 CREDIT RISK ASSESSMENT</h1><p style='font-size: 16px; opacity: 0.85; margin: 0;'>Advanced Default Prediction Demo</p></div>", unsafe_allow_html=True)

    try:
        model, scaler, feature_names = load_model()
    except Exception as e:
        st.error(f"❌ Model Loading Error: {e}")
        return

    tab1, tab2, tab3 = st.tabs(["🔍 Assessment", "📋 Demo", "📖 Guide"])

    with tab1:
        st.markdown("### Customer Profile")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            age = st.number_input("Age", 21, 75, 35)
        with col2:
            limit_bal = st.number_input("Credit Limit (K NT$)", 10, 1000, 100, step=10) * 1000
        with col3:
            sex = st.selectbox("Gender", [1, 2], format_func=lambda x: "M" if x == 1 else "F")
        with col4:
            education = st.selectbox("Education", [1, 2, 3, 4], format_func=lambda x: ["Grad", "Uni", "HS", "Other"][x-1])
        with col5:
            marriage = st.selectbox("Status", [1, 2, 3], format_func=lambda x: ["Married", "Single", "Other"][x-1])

        st.markdown("---")
        st.markdown("### Payment History (Last 6 Months)")
        st.caption("Status: -2=No Use | -1=On-Time | 0=Revolving | 1-9=Months Delayed")
        # Use selectboxes with explicit options to prevent invalid user input
        pay_features = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
        cols = st.columns(len(pay_features))
        pay_vals = []
        options = [-2, -1, 0] + list(range(1, 10))
        def fmt(x):
            if x == -2:
                return 'No Use (-2)'
            if x == -1:
                return 'On-Time (-1)'
            if x == 0:
                return 'Revolving (0)'
            return f"{x} month{'s' if x>1 else ''} delayed ({x})"

        for i, feat in enumerate(pay_features):
            with cols[i]:
                sel = st.selectbox(feat, options, index=1, format_func=fmt, key=f"pay_{feat}")
                pay_vals.append(sel)

        # Unpack in the same order as feature list
        pay_0, pay_2, pay_3, pay_4, pay_5, pay_6 = pay_vals

        st.markdown("---")
        st.markdown("### Financial Records")
        col_bill, col_pay = st.columns(2)
        with col_bill:
            st.markdown("**Bills (NT$)**")
            bill_vals = [st.number_input(f"M{10-m}", 0, 1000000, 50000-m*1000, step=1000, key=f"bill_{m}", label_visibility="collapsed") for m in range(1, 7)]
            bill_amt1, bill_amt2, bill_amt3, bill_amt4, bill_amt5, bill_amt6 = bill_vals
        with col_pay:
            st.markdown("**Payments (NT$)**")
            pay_amounts = [st.number_input(f"M{10-m}", 0, 1000000, 20000-m*1000, step=1000, key=f"pamt_{m}", label_visibility="collapsed") for m in range(1, 7)]
            pay_amt1, pay_amt2, pay_amt3, pay_amt4, pay_amt5, pay_amt6 = pay_amounts

        st.markdown("---")
        if st.button("🚀 ANALYZE RISK", use_container_width=True, type="primary"):
            customer_data = {'LIMIT_BAL': limit_bal, 'SEX': sex, 'EDUCATION': education, 'MARRIAGE': marriage, 'AGE': age,
                             'PAY_0': pay_0, 'PAY_2': pay_2, 'PAY_3': pay_3, 'PAY_4': pay_4, 'PAY_5': pay_5, 'PAY_6': pay_6,
                             'BILL_AMT1': bill_amt1, 'BILL_AMT2': bill_amt2, 'BILL_AMT3': bill_amt3, 'BILL_AMT4': bill_amt4, 'BILL_AMT5': bill_amt5, 'BILL_AMT6': bill_amt6,
                             'PAY_AMT1': pay_amt1, 'PAY_AMT2': pay_amt2, 'PAY_AMT3': pay_amt3, 'PAY_AMT4': pay_amt4, 'PAY_AMT5': pay_amt5, 'PAY_AMT6': pay_amt6}
            prediction, probability = predict_default(customer_data, model, scaler, feature_names)
            prob_no_default, prob_default = probability[0], probability[1]

            st.markdown("---")
            st.markdown("## 📊 PREDICTION RESULTS")
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                st.plotly_chart(create_gauge(prob_no_default, "No Default Risk", theme), use_container_width=True, config={"displayModeBar": False})
            with col_g2:
                st.plotly_chart(create_gauge(prob_default, "Default Risk", theme), use_container_width=True, config={"displayModeBar": False})

            st.markdown("---")
            if prediction == 0:
                st.markdown(f"<div class='success-box'><h3>✓ APPROVED</h3><p>Low-risk customer profile</p><p><strong>Confidence:</strong> {prob_no_default*100:.1f}% | <strong>Risk:</strong> {'Low 🟢' if prob_default < 0.2 else 'Medium 🟡'}</p></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='danger-box'><h3>⚠ REVIEW REQUIRED</h3><p>High-risk customer profile</p><p><strong>Risk Level:</strong> {prob_default*100:.1f}% | <strong>Status:</strong> {'High 🟠' if prob_default < 0.6 else 'Critical 🔴'}</p></div>", unsafe_allow_html=True)

            st.markdown("### 📈 Key Metrics")
            mc1, mc2, mc3 = st.columns(3)
            utilization = (bill_amt1 / limit_bal * 100) if limit_bal > 0 else 0
            payment_ratio = (pay_amt1 / bill_amt1 * 100) if bill_amt1 > 0 else 0
            times_delayed = sum([1 for p in pay_vals if p > 0])
            with mc1:
                st.metric("Utilization", f"{utilization:.0f}%", delta="⚠ High" if utilization > 70 else "✓ Normal", delta_color="inverse" if utilization > 70 else "normal")
            with mc2:
                st.metric("Payment Ratio", f"{payment_ratio:.0f}%", delta="⚠ Low" if payment_ratio < 30 else "✓ Good", delta_color="inverse" if payment_ratio < 30 else "normal")
            with mc3:
                st.metric("Late Payments", f"{times_delayed}x", delta="⚠ High" if times_delayed > 2 else "✓ Low", delta_color="inverse" if times_delayed > 2 else "normal")

    with tab2:
        st.markdown("### Pre-Built Scenarios")
        demo_type = st.radio("Select Profile", ["✓ Good Customer", "⚠ Risk Customer"], horizontal=True)
        if demo_type == "✓ Good Customer":
            demo = {'LIMIT_BAL': 200000, 'SEX': 1, 'EDUCATION': 1, 'MARRIAGE': 1, 'AGE': 35,
                    'PAY_0': -1, 'PAY_2': -1, 'PAY_3': -1, 'PAY_4': -1, 'PAY_5': -1, 'PAY_6': -1,
                    'BILL_AMT1': 50000, 'BILL_AMT2': 48000, 'BILL_AMT3': 47000, 'BILL_AMT4': 45000, 'BILL_AMT5': 43000, 'BILL_AMT6': 42000,
                    'PAY_AMT1': 50000, 'PAY_AMT2': 48000, 'PAY_AMT3': 47000, 'PAY_AMT4': 45000, 'PAY_AMT5': 43000, 'PAY_AMT6': 42000}
            st.info("High credit limit, excellent payment history, consistent full payments")
        else:
            demo = {'LIMIT_BAL': 50000, 'SEX': 2, 'EDUCATION': 3, 'MARRIAGE': 2, 'AGE': 25,
                    'PAY_0': 2, 'PAY_2': 2, 'PAY_3': 1, 'PAY_4': 3, 'PAY_5': 2, 'PAY_6': 1,
                    'BILL_AMT1': 48000, 'BILL_AMT2': 47000, 'BILL_AMT3': 49000, 'BILL_AMT4': 48500, 'BILL_AMT5': 47500, 'BILL_AMT6': 48000,
                    'PAY_AMT1': 1000, 'PAY_AMT2': 1500, 'PAY_AMT3': 1200, 'PAY_AMT4': 1000, 'PAY_AMT5': 1100, 'PAY_AMT6': 1000}
            st.warning("Low credit limit, frequent delays, minimal payments")
        st.json(demo)
        if st.button("RUN ANALYSIS", use_container_width=True, type="primary"):
            pred, prob = predict_default(demo, model, scaler, feature_names)
            if pred == 0:
                st.success(f"Result: No Default ({prob[0]*100:.1f}% confidence)")
            else:
                st.error(f"Result: Default Risk ({prob[1]*100:.1f}% confidence)")

    with tab3:
        st.markdown("""
        ### How to Use
        1. **Enter customer profile** in Assessment tab
        2. **Input payment history** for last 6 months
        3. **Provide financial data** (bills & payments)
        4. **Click ANALYZE** to get prediction

        ### Risk Interpretation
        | Probability | Action |
        |---|---|
        | > 80% | ✓ Strong Approve |
        | 60-80% | ✓ Approve |
        | 40-60% | ⚠ Review |
        | < 40% | ✗ Decline |
        """)

if __name__ == "__main__":
    main()

