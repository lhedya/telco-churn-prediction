"""
Streamlit App — Telco Customer Churn Prediction
Portfolio Project: Data Analyst & Data Scientist — Lhedya Monica Ismon

Model: Random Forest (manual params: n_estimators=200, max_depth=10, min_samples_leaf=5)
Dilatih persis sesuai pipeline Telco_Churn_Final_Combined.ipynb
ROC-AUC: 0.834 | Accuracy: 76.7% | Precision: 55.0% | Recall: 68.2% | F1: 60.9%

Cara menjalankan secara lokal:
    streamlit run streamlit_app.py

Cara deploy ke Streamlit Community Cloud:
    1. Push folder ini (streamlit_app.py, rf_model.pkl, scaler.pkl, feature_cols.pkl,
       label_encoders.pkl, requirements.txt) ke GitHub repository.
    2. Buka https://share.streamlit.io -> New app -> pilih repo & file streamlit_app.py.
    3. Deploy.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Telco Churn Prediction",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# LOAD MODEL & ARTIFACTS
# =========================================================
@st.cache_resource
def load_artifacts():
    with open("rf_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)
    with open("label_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return model, scaler, feature_cols, encoders

try:
    model, scaler, feature_cols, encoders = load_artifacts()
    artifacts_loaded = True
except FileNotFoundError:
    artifacts_loaded = False

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("📡 Telco Churn Prediction")
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Tentang Aplikasi**

Aplikasi ini men-deploy model **Random Forest** yang dilatih untuk
memprediksi risiko churn pelanggan telekomunikasi, sebagai bagian dari
portofolio Data Analyst & Data Scientist.

**Performa Model (test set):**
- ROC-AUC: 0.834
- Accuracy: 76.7%
- Recall: 68.2%
- F1-Score: 60.9%

**Dataset:** IBM Telco Customer Churn (7.043 pelanggan)
""")
st.sidebar.markdown("---")
st.sidebar.caption("Dibuat oleh Lhedya Monica Ismon — Data Analyst & Data Scientist Portfolio")

if not artifacts_loaded:
    st.error(
        "⚠️ File model (`rf_model.pkl`, `scaler.pkl`, `feature_cols.pkl`, `label_encoders.pkl`) "
        "tidak ditemukan di direktori ini. Pastikan keempat file tersebut berada di folder yang "
        "sama dengan `streamlit_app.py`."
    )
    st.stop()

# =========================================================
# HEADER
# =========================================================
st.title("📡 Telco customer churn prediction")
st.markdown("""
Masukkan profil pelanggan untuk memprediksi probabilitas churn menggunakan model
**Random Forest** yang telah dilatih dan dievaluasi pada 1.409 pelanggan test set.
Model ini dapat diintegrasikan ke sistem CRM perusahaan sebagai *early-warning system*
bagi tim Customer Retention.
""")

st.markdown("---")

# =========================================================
# INPUT FORM
# =========================================================
with st.form("prediction_form"):
    st.subheader("📋 Profil pelanggan")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demografi**")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Memiliki Partner", ["No", "Yes"])
        dependents = st.selectbox("Memiliki Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (bulan berlangganan)", 0, 72, 12)

    with col2:
        st.markdown("**Layanan**")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    with col3:
        st.markdown("**Kontrak & Billing**")
        contract = st.selectbox("Tipe Kontrak", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Metode Pembayaran", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=65.0, step=0.5)
        total_charges = st.number_input(
            "Total Charges ($)", min_value=0.0, max_value=10000.0,
            value=float(round(monthly_charges * max(tenure, 1), 2)), step=10.0
        )

    submitted = st.form_submit_button("🔮 Prediksi Churn", use_container_width=True)

# =========================================================
# PREDICTION LOGIC
# =========================================================
if submitted:
    raw_input = {
        'gender': gender, 'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
        'Partner': partner, 'Dependents': dependents, 'tenure': tenure,
        'PhoneService': phone_service, 'MultipleLines': multiple_lines,
        'InternetService': internet_service, 'OnlineSecurity': online_security,
        'OnlineBackup': online_backup, 'DeviceProtection': device_protection,
        'TechSupport': tech_support, 'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies, 'Contract': contract,
        'PaperlessBilling': paperless_billing, 'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
    }
    input_df = pd.DataFrame([raw_input])

    # Feature engineering — persis sesuai notebook
    input_df['ChargesPerTenure'] = input_df['TotalCharges'] / (input_df['tenure'] + 1)
    input_df['IsNewCustomer']    = (input_df['tenure'] <= 6).astype(int)
    input_df['IsMonthToMonth']   = (input_df['Contract'] == 'Month-to-month').astype(int)

    # Encoding biner/multi-kelas — pakai label_encoders.pkl yang sama dengan training
    binary_cols = ['gender','Partner','Dependents','PhoneService','PaperlessBilling',
                   'MultipleLines','OnlineSecurity','OnlineBackup','DeviceProtection',
                   'TechSupport','StreamingTV','StreamingMovies']
    for col in binary_cols:
        mapping = encoders[col]
        input_df[col] = input_df[col].map(mapping)

    # One-hot encoding — sesuaikan manual karena hanya 1 baris (get_dummies tidak reliable untuk 1 baris)
    input_df['InternetService_Fiber optic'] = 1 if internet_service == 'Fiber optic' else 0
    input_df['InternetService_No'] = 1 if internet_service == 'No' else 0
    input_df['Contract_One year'] = 1 if contract == 'One year' else 0
    input_df['Contract_Two year'] = 1 if contract == 'Two year' else 0
    input_df['PaymentMethod_Credit card (automatic)'] = 1 if payment_method == 'Credit card (automatic)' else 0
    input_df['PaymentMethod_Electronic check'] = 1 if payment_method == 'Electronic check' else 0
    input_df['PaymentMethod_Mailed check'] = 1 if payment_method == 'Mailed check' else 0

    # Drop kolom kategorikal asli yang sudah di-encode
    input_df = input_df.drop(columns=['InternetService', 'Contract', 'PaymentMethod'])

    # Selaraskan urutan kolom persis seperti saat training
    final_input = input_df.reindex(columns=feature_cols, fill_value=0).astype(float)

    # Scaling — pakai scaler yang sama dengan training
    final_input_scaled = scaler.transform(final_input)

    # Prediksi
    churn_proba = model.predict_proba(final_input_scaled)[0][1]
    churn_pred = model.predict(final_input_scaled)[0]

    st.markdown("---")
    st.subheader("📊 Hasil prediksi")

    col_a, col_b = st.columns([1, 2])

    with col_a:
        if churn_proba >= 0.75:
            risk_label, risk_color = "HIGH RISK", "🔴"
        elif churn_proba >= 0.50:
            risk_label, risk_color = "MEDIUM RISK", "🟡"
        else:
            risk_label, risk_color = "LOW RISK", "🟢"

        st.metric("Probabilitas Churn", f"{churn_proba*100:.1f}%")
        st.markdown(f"### {risk_color} Status Risiko: **{risk_label}**")

        if churn_pred == 1:
            st.error("⚠️ Pelanggan ini diprediksi **AKAN CHURN**. Disarankan tindakan retensi proaktif.")
        else:
            st.success("✅ Pelanggan ini diprediksi **TIDAK akan churn** dalam waktu dekat.")

    with col_b:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(churn_proba * 100, 1),
            title={'text': "Churn risk score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "black"},
                'steps': [
                    {'range': [0, 50], 'color': "#4CAF50"},
                    {'range': [50, 75], 'color': "#FF9800"},
                    {'range': [75, 100], 'color': "#F44336"}
                ],
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("#### 💡 Rekomendasi aksi")
    recommendations = []
    if contract == "Month-to-month":
        recommendations.append("Tawarkan insentif migrasi ke kontrak tahunan (diskon/cashback) — kontrak month-to-month memiliki churn rate 42.7% vs 2.8% untuk two-year.")
    if online_security in ["No"] or tech_support in ["No"]:
        recommendations.append("Tawarkan free trial Online Security / Tech Support — pelanggan tanpa layanan ini berisiko lebih tinggi churn.")
    if payment_method == "Electronic check":
        recommendations.append("Dorong migrasi ke metode pembayaran otomatis (auto-debit/credit card) — electronic check adalah metode dengan churn rate tertinggi (45%).")
    if tenure <= 6:
        recommendations.append("Pelanggan masih dalam periode onboarding kritis (0-6 bulan) — pastikan program proactive check-in aktif.")
    if internet_service == "Fiber optic":
        recommendations.append("Pelanggan fiber optic punya churn rate lebih tinggi — pertimbangkan satisfaction survey & review pricing kompetitif.")
    if not recommendations:
        recommendations.append("Profil pelanggan relatif stabil — lanjutkan monitoring rutin tanpa intervensi khusus.")

    for rec in recommendations:
        st.markdown(f"- {rec}")

    with st.expander("🔍 Lihat detail fitur yang dikirim ke model"):
        st.dataframe(final_input.T.rename(columns={0: "Nilai"}), use_container_width=True)

st.markdown("---")
st.caption("Portfolio Project — Telco Customer Churn Analysis | Model: Random Forest | Built with Streamlit")
