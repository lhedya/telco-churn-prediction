# ============================================================
# app.py — Telco Customer Churn | Business Intelligence Dashboard
# Deploy: streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Intelligence | Telco",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

[data-testid="stSidebar"] { background: #0f1923; border-right: 1px solid #1e2d3d; }
[data-testid="stSidebar"] * { color: #c9d6e3 !important; }

.main .block-container { background: #0b1520; padding: 1.5rem 2rem; max-width: 1400px; }
.page-title { font-size: 22px; font-weight: 700; color: #e8f4fd; letter-spacing: -0.3px; margin-bottom: 2px; }
.page-sub   { font-size: 13px; color: #5a8aad; margin-bottom: 1.2rem; }

.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 20px; }
.kpi      { background: #111e2d; border: 1px solid #1a2d42; border-radius: 10px; padding: 16px 18px; position: relative; overflow: hidden; }
.kpi::before { content:''; position:absolute; top:0; left:0; right:0; height:3px; }
.kpi.red::before    { background: linear-gradient(90deg,#e63946,#ff6b6b); }
.kpi.green::before  { background: linear-gradient(90deg,#2dc653,#56ef7e); }
.kpi.blue::before   { background: linear-gradient(90deg,#0077b6,#48cae4); }
.kpi.amber::before  { background: linear-gradient(90deg,#e9830f,#ffc300); }
.kpi-label { font-size: 11px; color: #5a8aad; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 6px; }
.kpi-value { font-size: 28px; font-weight: 700; color: #e8f4fd; line-height: 1; margin-bottom: 4px; }
.kpi-delta { font-size: 11px; color: #5a8aad; }
.kpi-delta.up   { color: #e63946; }
.kpi-delta.down { color: #2dc653; }

.sec-head { font-size: 13px; font-weight: 600; color: #7a9bb5; text-transform: uppercase;
            letter-spacing: 1px; margin: 20px 0 10px; padding-bottom: 6px;
            border-bottom: 1px solid #1a2d42; }

.alert-high   { background:#1c0a0a; border:1px solid #e63946; border-radius:8px; padding:12px 16px; margin:8px 0; color:#ff9a9a; font-size:13px; }
.alert-medium { background:#1a1200; border:1px solid #e9830f; border-radius:8px; padding:12px 16px; margin:8px 0; color:#ffc96b; font-size:13px; }
.alert-low    { background:#071a0e; border:1px solid #2dc653; border-radius:8px; padding:12px 16px; margin:8px 0; color:#7effa0; font-size:13px; }

.stTabs [data-baseweb="tab-list"] { background:#0f1923; border-radius:8px; padding:4px; gap:4px; }
.stTabs [data-baseweb="tab"]      { border-radius:6px; color:#5a8aad; font-size:13px; font-weight:500; padding:6px 16px; }
.stTabs [aria-selected="true"]    { background:#111e2d !important; color:#e8f4fd !important; }
</style>
""", unsafe_allow_html=True)


# ── Load Model ────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open('rf_model.pkl',     'rb') as f: model     = pickle.load(f)
    with open('scaler.pkl',       'rb') as f: scaler    = pickle.load(f)
    with open('feature_cols.pkl', 'rb') as f: feat_cols = pickle.load(f)
    return model, scaler, feat_cols

@st.cache_data
def load_dataset():
    URL = 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv'
    df = pd.read_csv(URL)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    df['Churn_bin'] = df['Churn'].map({'Yes': 1, 'No': 0})
    return df

try:
    model, scaler, feat_cols = load_artifacts()
    model_loaded = True
except:
    model_loaded = False

df = load_dataset()


# ── Helpers ───────────────────────────────────────────────────
def fmt_usd(v):
    if v >= 1_000_000: return f"${v/1_000_000:.2f}M"
    if v >= 1_000:     return f"${v/1_000:.1f}K"
    return f"${v:.0f}"

def fig_style(fig, axes=None):
    fig.patch.set_facecolor('#111e2d')
    if axes is None: return
    lst = axes if hasattr(axes, '__iter__') else [axes]
    for ax in lst:
        ax.set_facecolor('#111e2d')
        ax.tick_params(colors='#5a8aad', labelsize=10)
        ax.xaxis.label.set_color('#5a8aad')
        ax.yaxis.label.set_color('#5a8aad')
        for spine in ax.spines.values():
            spine.set_edgecolor('#1a2d42')

def build_features(inp, feat_cols):
    row = {col: 0 for col in feat_cols}
    row['tenure']           = inp['tenure']
    row['MonthlyCharges']   = inp['MonthlyCharges']
    row['TotalCharges']     = inp['TotalCharges']
    row['ChargesPerTenure'] = inp['TotalCharges'] / (inp['tenure'] + 1)
    row['IsNewCustomer']    = 1 if inp['tenure'] <= 6 else 0
    row['IsMonthToMonth']   = 1 if inp['Contract'] == 'Month-to-month' else 0
    binary_map = {
        'gender':           {'Female':0,'Male':1},
        'Partner':          {'No':0,'Yes':1},
        'Dependents':       {'No':0,'Yes':1},
        'PhoneService':     {'No':0,'Yes':1},
        'PaperlessBilling': {'No':0,'Yes':1},
        'MultipleLines':    {'No':0,'No phone service':1,'Yes':2},
        'OnlineSecurity':   {'No':0,'No internet service':1,'Yes':2},
        'OnlineBackup':     {'No':0,'No internet service':1,'Yes':2},
        'DeviceProtection': {'No':0,'No internet service':1,'Yes':2},
        'TechSupport':      {'No':0,'No internet service':1,'Yes':2},
        'StreamingTV':      {'No':0,'No internet service':1,'Yes':2},
        'StreamingMovies':  {'No':0,'No internet service':1,'Yes':2},
        'SeniorCitizen':    {0:0,1:1},
    }
    for col, mapping in binary_map.items():
        if col in row:
            row[col] = mapping.get(inp.get(col, list(mapping.keys())[0]), 0)
    if inp['InternetService'] == 'Fiber optic':
        if 'InternetService_Fiber optic' in row: row['InternetService_Fiber optic'] = 1
    elif inp['InternetService'] == 'No':
        if 'InternetService_No' in row: row['InternetService_No'] = 1
    if inp['Contract'] == 'One year':
        if 'Contract_One year' in row: row['Contract_One year'] = 1
    elif inp['Contract'] == 'Two year':
        if 'Contract_Two year' in row: row['Contract_Two year'] = 1
    pm_map = {
        'Electronic check':          'PaymentMethod_Electronic check',
        'Mailed check':              'PaymentMethod_Mailed check',
        'Bank transfer (automatic)': 'PaymentMethod_Bank transfer (automatic)',
    }
    pm_col = pm_map.get(inp['PaymentMethod'], None)
    if pm_col and pm_col in row: row[pm_col] = 1
    return pd.DataFrame([row])[feat_cols]


# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📡 Churn Intelligence")
    st.markdown("---")
    nav = st.radio("Menu", ["📊 Business Dashboard", "🔮 Prediksi Pelanggan"],
                   label_visibility="collapsed")
    st.markdown("---")

    if nav == "🔮 Prediksi Pelanggan":
        st.markdown("**Input Data Pelanggan**")
        tenure          = st.slider("Tenure (bulan)", 0, 72, 12)
        contract        = st.selectbox("Tipe Kontrak", ["Month-to-month","One year","Two year"])
        monthly_charges = st.number_input("Monthly Charges (USD)", 18.0, 120.0, 65.0, 0.5)
        total_charges   = st.number_input("Total Charges (USD)", 0.0, 9000.0, float(monthly_charges*tenure), 10.0)
        internet        = st.selectbox("Internet Service", ["Fiber optic","DSL","No"])
        online_sec      = st.selectbox("Online Security", ["No","Yes","No internet service"])
        online_bkp      = st.selectbox("Online Backup", ["No","Yes","No internet service"])
        tech_support    = st.selectbox("Tech Support", ["No","Yes","No internet service"])
        dev_prot        = st.selectbox("Device Protection", ["No","Yes","No internet service"])
        streaming_tv    = st.selectbox("Streaming TV", ["No","Yes","No internet service"])
        streaming_mv    = st.selectbox("Streaming Movies", ["No","Yes","No internet service"])
        payment_method  = st.selectbox("Payment Method",
                            ["Electronic check","Mailed check",
                             "Bank transfer (automatic)","Credit card (automatic)"])
        paperless       = st.selectbox("Paperless Billing", ["Yes","No"])
        phone_svc       = st.selectbox("Phone Service", ["Yes","No"])
        multi_lines     = st.selectbox("Multiple Lines", ["No","Yes","No phone service"])
        gender          = st.selectbox("Gender", ["Male","Female"])
        senior          = st.selectbox("Senior Citizen", [0,1], format_func=lambda x:"Ya" if x else "Tidak")
        partner         = st.selectbox("Partner", ["Yes","No"])
        dependents      = st.selectbox("Dependents", ["No","Yes"])
        predict_btn     = st.button("🔮 Prediksi Sekarang", use_container_width=True)
    else:
        predict_btn = False


# ══════════════════════════════════════════════════════════════
# HALAMAN 1 — BUSINESS DASHBOARD
# ══════════════════════════════════════════════════════════════
if nav == "📊 Business Dashboard":
    st.markdown('<div class="page-title">📊 Business Intelligence Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Telco Customer Churn Analysis · IBM Dataset · Model: Random Forest AUC 0.834</div>', unsafe_allow_html=True)

    # KPI
    total         = len(df)
    churned       = int(df['Churn_bin'].sum())
    retained      = total - churned
    churn_rate    = churned / total * 100
    monthly_lost  = df[df['Churn']=='Yes']['MonthlyCharges'].sum()
    annual_risk   = monthly_lost * 12
    avg_tenure_c  = df[df['Churn']=='Yes']['tenure'].mean()
    avg_tenure_r  = df[df['Churn']=='No']['tenure'].mean()

    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi red">
        <div class="kpi-label">Churn Rate</div>
        <div class="kpi-value">{churn_rate:.1f}%</div>
        <div class="kpi-delta up">▲ {churned:,} dari {total:,} pelanggan</div>
      </div>
      <div class="kpi amber">
        <div class="kpi-label">Revenue At Risk / Bulan</div>
        <div class="kpi-value">{fmt_usd(monthly_lost)}</div>
        <div class="kpi-delta up">▲ Proyeksi {fmt_usd(annual_risk)} / tahun</div>
      </div>
      <div class="kpi green">
        <div class="kpi-label">Pelanggan Retain</div>
        <div class="kpi-value">{retained:,}</div>
        <div class="kpi-delta down">✓ {100-churn_rate:.1f}% dari total</div>
      </div>
      <div class="kpi blue">
        <div class="kpi-label">Avg Tenure Churn</div>
        <div class="kpi-value">{avg_tenure_c:.0f} bln</div>
        <div class="kpi-delta">vs {avg_tenure_r:.0f} bln (retain)</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Row 1: Contract + Tenure ──────────────────────────────
    st.markdown('<div class="sec-head">Segmentasi Risiko Churn</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        cg = df.groupby('Contract')['Churn_bin'].agg(['mean','sum','count']).reset_index()
        cg.columns = ['Contract','Rate','Churned','Total']
        cg['Rate'] *= 100
        cg = cg.sort_values('Rate', ascending=True)

        fig, ax = plt.subplots(figsize=(6, 3.2))
        fig_style(fig, ax)
        colors = ['#2dc653','#e9830f','#e63946']
        bars = ax.barh(cg['Contract'], cg['Rate'], color=colors, height=0.55, edgecolor='none')
        ax.set_xlabel('Churn Rate (%)', fontsize=10)
        ax.set_title('Churn Rate per Tipe Kontrak', color='#c9d6e3', fontsize=12, fontweight='600', pad=10)
        for bar, val, ch, tot in zip(bars, cg['Rate'], cg['Churned'], cg['Total']):
            ax.text(val+0.8, bar.get_y()+bar.get_height()/2,
                    f'{val:.1f}%  ({ch:,}/{tot:,})', va='center', color='#c9d6e3', fontsize=9, fontweight='600')
        ax.set_xlim(0, 62)
        ax.axvline(churn_rate, color='#5a8aad', linestyle='--', linewidth=0.8, alpha=0.6)
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown('<div style="font-size:11px;color:#5a8aad">💡 Month-to-Month 3× lebih berisiko dari kontrak tahunan</div>', unsafe_allow_html=True)

    with col2:
        df['tenure_bucket'] = pd.cut(df['tenure'], bins=[0,6,12,24,48,72],
                                      labels=['0–6 bln','7–12 bln','13–24 bln','25–48 bln','49–72 bln'])
        tg = df.groupby('tenure_bucket', observed=True)['Churn_bin'].agg(['mean','count']).reset_index()
        tg.columns = ['Bucket','Rate','Total']
        tg['Rate'] *= 100

        fig, ax = plt.subplots(figsize=(6, 3.2))
        fig_style(fig, ax)
        clr = ['#e63946','#ff8c42','#e9830f','#5cb8e4','#2dc653']
        bars = ax.bar(tg['Bucket'], tg['Rate'], color=clr, width=0.6, edgecolor='none')
        ax.set_ylabel('Churn Rate (%)', fontsize=10)
        ax.set_title('Churn Rate per Tenure Pelanggan', color='#c9d6e3', fontsize=12, fontweight='600', pad=10)
        ax.tick_params(axis='x', rotation=15)
        for bar, val in zip(bars, tg['Rate']):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                    f'{val:.1f}%', ha='center', color='#c9d6e3', fontsize=9, fontweight='600')
        ax.axhline(churn_rate, color='#5a8aad', linestyle='--', linewidth=0.8, alpha=0.6)
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown('<div style="font-size:11px;color:#5a8aad">💡 Pelanggan baru (0–6 bln) punya churn rate tertinggi</div>', unsafe_allow_html=True)

    # ── Row 2: Monthly Charges + Payment Method ───────────────
    st.markdown('<div class="sec-head">Revenue Impact & Payment Behavior</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)

    with col3:
        fig, ax = plt.subplots(figsize=(6, 3.2))
        fig_style(fig, ax)
        ax.hist(df[df['Churn']=='No']['MonthlyCharges'],  bins=30, alpha=0.6, color='#2dc653', label='Retain', edgecolor='none')
        ax.hist(df[df['Churn']=='Yes']['MonthlyCharges'], bins=30, alpha=0.7, color='#e63946', label='Churn',  edgecolor='none')
        med_c = df[df['Churn']=='Yes']['MonthlyCharges'].median()
        med_r = df[df['Churn']=='No']['MonthlyCharges'].median()
        ax.axvline(med_c, color='#ff6b6b', linestyle='--', linewidth=1.5)
        ax.axvline(med_r, color='#7effa0', linestyle='--', linewidth=1.5)
        ax.set_xlabel('Monthly Charges (USD)', fontsize=10)
        ax.set_ylabel('Jumlah Pelanggan', fontsize=10)
        ax.set_title('Distribusi Tagihan: Churn vs Retain', color='#c9d6e3', fontsize=12, fontweight='600', pad=10)
        ax.legend(fontsize=9, facecolor='#1a2d42', edgecolor='none', labelcolor='#c9d6e3')
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown(f'<div style="font-size:11px;color:#5a8aad">💡 Median churn <b style="color:#ff6b6b">${med_c:.0f}</b> vs retain <b style="color:#7effa0">${med_r:.0f}</b> — pelanggan tagihan tinggi lebih berisiko</div>', unsafe_allow_html=True)

    with col4:
        pm = df.groupby('PaymentMethod')['Churn_bin'].mean().sort_values(ascending=True) * 100
        short = {
            'Bank transfer (automatic)':'Bank Transfer (Auto)',
            'Credit card (automatic)': 'Credit Card (Auto)',
            'Mailed check':            'Mailed Check',
            'Electronic check':        'Electronic Check'
        }
        pm.index = [short.get(i,i) for i in pm.index]
        fig, ax = plt.subplots(figsize=(6, 3.2))
        fig_style(fig, ax)
        bars = ax.barh(pm.index, pm.values, color=['#2dc653','#5cb8e4','#e9830f','#e63946'], height=0.55, edgecolor='none')
        ax.set_xlabel('Churn Rate (%)', fontsize=10)
        ax.set_title('Churn Rate per Metode Pembayaran', color='#c9d6e3', fontsize=12, fontweight='600', pad=10)
        for bar, val in zip(bars, pm.values):
            ax.text(val+0.5, bar.get_y()+bar.get_height()/2,
                    f'{val:.1f}%', va='center', color='#c9d6e3', fontsize=9, fontweight='600')
        ax.set_xlim(0, 55)
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown('<div style="font-size:11px;color:#5a8aad">💡 Electronic Check churn 2× lebih tinggi dari auto-payment</div>', unsafe_allow_html=True)

    # ── Row 3: Internet Service + Heatmap ─────────────────────
    st.markdown('<div class="sec-head">Analisis Layanan & Kombinasi Risiko</div>', unsafe_allow_html=True)
    col5, col6 = st.columns(2)

    with col5:
        ig = df.groupby('InternetService')['Churn_bin'].agg(['mean','sum']).reset_index()
        ig.columns = ['Service','Rate','Churned']
        ig['Rate'] *= 100
        ig['RevRisk'] = [df[(df['Churn']=='Yes') & (df['InternetService']==s)]['MonthlyCharges'].sum()
                         for s in ig['Service']]
        fig, ax = plt.subplots(figsize=(6, 3.2))
        fig_style(fig, ax)
        x = np.arange(len(ig))
        bars = ax.bar(x, ig['Rate'], color=['#5cb8e4','#e63946','#2dc653'], width=0.5, edgecolor='none')
        ax.set_xticks(x); ax.set_xticklabels(ig['Service'], fontsize=10)
        ax.set_ylabel('Churn Rate (%)', fontsize=10)
        ax.set_title('Churn Rate per Layanan Internet', color='#c9d6e3', fontsize=12, fontweight='600', pad=10)
        for bar, val, risk in zip(bars, ig['Rate'], ig['RevRisk']):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                    f'{val:.1f}%', ha='center', color='#c9d6e3', fontsize=9, fontweight='600')
            if risk > 0:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()/2,
                        f'${risk/1000:.0f}K\nat risk', ha='center', va='center',
                        color='white', fontsize=8, fontweight='600')
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown('<div style="font-size:11px;color:#5a8aad">💡 Fiber Optic — churn tertinggi sekaligus revenue at risk terbesar</div>', unsafe_allow_html=True)

    with col6:
        pivot = df.groupby(['Contract','InternetService'])['Churn_bin'].mean().unstack() * 100
        fig, ax = plt.subplots(figsize=(6, 3.2))
        fig_style(fig, ax)
        im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=75)
        ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, fontsize=9, color='#c9d6e3')
        ax.set_yticks(range(len(pivot.index)));   ax.set_yticklabels(pivot.index,   fontsize=9, color='#c9d6e3')
        ax.set_title('Heatmap: Kontrak × Internet (%)', color='#c9d6e3', fontsize=12, fontweight='600', pad=10)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                            color='white' if val > 40 else '#1a2d42', fontsize=11, fontweight='700')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(colors='#5a8aad', labelsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown('<div style="font-size:11px;color:#5a8aad">💡 Month-to-Month + Fiber Optic = kombinasi risiko tertinggi</div>', unsafe_allow_html=True)

    # ── Rekomendasi Bisnis ────────────────────────────────────
    st.markdown('<div class="sec-head">Rekomendasi Bisnis Berdasarkan Data</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="alert-high">
        <b>🔴 Prioritas 1 — Retensi Kontrak M2M</b><br><br>
        Pelanggan Month-to-Month memiliki churn rate <b>~43%</b>.
        Tawarkan diskon <b>15–20%</b> untuk upgrade ke kontrak tahunan
        di bulan ke-3. Potensi recovery revenue signifikan per tahun.
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="alert-medium">
        <b>🟡 Prioritas 2 — Onboarding Pelanggan Baru</b><br><br>
        Tenure <b>0–6 bulan</b> adalah periode paling kritis.
        Program check-in aktif di bulan ke-2 & ke-3 disertai
        trial TechSupport gratis 1 bulan terbukti efektif.
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="alert-low">
        <b>🟢 Prioritas 3 — Migrasi Metode Bayar</b><br><br>
        Electronic Check churn rate <b>~45%</b> vs auto-payment <b>~15%</b>.
        Dorong migrasi ke Bank Transfer / Credit Card auto
        dengan insentif cashback di bulan pertama.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"""
    <div style="display:flex;gap:24px;font-size:12px;color:#5a8aad;padding:4px 0;flex-wrap:wrap">
      <span>🤖 Model: Random Forest · AUC 0.834</span>
      <span>📦 Dataset: IBM Telco (7,043 records)</span>
      <span>🛠 Stack: Python · Scikit-learn · SQL · Streamlit</span>
      <span>👤 dibimbing.id Talent Showcase</span>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# HALAMAN 2 — PREDIKSI PELANGGAN
# ══════════════════════════════════════════════════════════════
else:
    st.markdown('<div class="page-title">🔮 Prediksi Churn Pelanggan</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Masukkan data pelanggan di sidebar → klik Prediksi Sekarang</div>', unsafe_allow_html=True)

    if not model_loaded:
        st.error("⚠️ File model tidak ditemukan. Pastikan rf_model.pkl, scaler.pkl, feature_cols.pkl ada di folder yang sama.")
        st.stop()

    if not predict_btn:
        total      = len(df)
        churned    = int(df['Churn_bin'].sum())
        churn_rate = churned / total * 100
        monthly_lost = df[df['Churn']=='Yes']['MonthlyCharges'].sum()

        col_a, col_b = st.columns([1.2, 1])
        with col_a:
            fig, ax = plt.subplots(figsize=(5.5, 3.5))
            fig_style(fig, ax)
            sizes  = [total - churned, churned]
            colors = ['#2dc653', '#e63946']
            wedges, texts, autos = ax.pie(sizes, labels=['Retain','Churn'],
                                           autopct='%1.1f%%', colors=colors,
                                           startangle=90, pctdistance=0.75,
                                           wedgeprops={'edgecolor':'#0b1520','linewidth':2})
            for t in texts: t.set_color('#c9d6e3'); t.set_fontsize(11)
            for a in autos: a.set_color('white'); a.set_fontsize(10); a.set_fontweight('600')
            ax.set_title('Distribusi Churn Keseluruhan', color='#c9d6e3', fontsize=12, fontweight='600')
            plt.tight_layout()
            st.pyplot(fig)

        with col_b:
            st.markdown(f"""
            <br>
            <div class="kpi red" style="margin-bottom:12px">
              <div class="kpi-label">Churn Rate Keseluruhan</div>
              <div class="kpi-value">{churn_rate:.1f}%</div>
              <div class="kpi-delta up">▲ {churned:,} pelanggan churn</div>
            </div>
            <div class="kpi amber">
              <div class="kpi-label">Revenue At Risk / Tahun</div>
              <div class="kpi-value">{fmt_usd(monthly_lost*12)}</div>
              <div class="kpi-delta">Jika tidak ada intervensi</div>
            </div>
            """, unsafe_allow_html=True)
            st.info("👈 Isi data pelanggan di sidebar, lalu klik **Prediksi Sekarang**")
    else:
        inp = {
            'tenure':tenure, 'MonthlyCharges':monthly_charges, 'TotalCharges':total_charges,
            'Contract':contract, 'PaymentMethod':payment_method, 'PaperlessBilling':paperless,
            'InternetService':internet, 'OnlineSecurity':online_sec, 'OnlineBackup':online_bkp,
            'TechSupport':tech_support, 'StreamingTV':streaming_tv, 'StreamingMovies':streaming_mv,
            'PhoneService':phone_svc, 'MultipleLines':multi_lines, 'DeviceProtection':dev_prot,
            'gender':gender, 'SeniorCitizen':senior, 'Partner':partner, 'Dependents':dependents,
        }
        X_in  = build_features(inp, feat_cols)
        X_sc  = scaler.transform(X_in)
        prob  = model.predict_proba(X_sc)[0][1]
        pct   = prob * 100
        risk  = "HIGH" if prob >= 0.70 else ("MEDIUM" if prob >= 0.45 else "LOW")
        icon  = "🔴" if risk=="HIGH" else ("🟡" if risk=="MEDIUM" else "🟢")
        kpi_c = "red" if prob >= 0.70 else ("amber" if prob >= 0.45 else "green")
        ann_risk = monthly_charges * 12 if prob >= 0.45 else 0

        # KPI row
        st.markdown(f"""
        <div class="kpi-grid">
          <div class="kpi {kpi_c}">
            <div class="kpi-label">Probabilitas Churn</div>
            <div class="kpi-value">{pct:.1f}%</div>
            <div class="kpi-delta {'up' if prob>=0.5 else 'down'}">{icon} {risk} RISK</div>
          </div>
          <div class="kpi green">
            <div class="kpi-label">Probabilitas Retain</div>
            <div class="kpi-value">{(1-prob)*100:.1f}%</div>
          </div>
          <div class="kpi amber">
            <div class="kpi-label">Revenue At Risk / Tahun</div>
            <div class="kpi-value">{fmt_usd(ann_risk)}</div>
            <div class="kpi-delta">Jika tidak dicegah</div>
          </div>
          <div class="kpi blue">
            <div class="kpi-label">Monthly Charges</div>
            <div class="kpi-value">{fmt_usd(monthly_charges)}</div>
            <div class="kpi-delta">Tenure: {tenure} bulan</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        col_g, col_r = st.columns([1, 1.5])

        with col_g:
            # Gauge chart
            fig, ax = plt.subplots(figsize=(5, 3.2), subplot_kw={'projection':'polar'})
            fig.patch.set_facecolor('#111e2d')
            ax.set_facecolor('#111e2d')
            ax.barh(1, np.pi*0.45, left=np.pi,      height=0.4, color='#2dc653', alpha=0.35)
            ax.barh(1, np.pi*0.25, left=np.pi*0.55, height=0.4, color='#e9830f', alpha=0.35)
            ax.barh(1, np.pi*0.30, left=np.pi*0.30, height=0.4, color='#e63946', alpha=0.35)
            needle = np.pi - (prob * np.pi)
            ax.annotate('', xy=(needle, 0.95), xytext=(needle, 0.1),
                        arrowprops=dict(arrowstyle='->', color='#e8f4fd', lw=2.5))
            ax.text(np.pi*1.1, 1.38, 'LOW',  ha='left',   color='#2dc653', fontsize=9, fontweight='700')
            ax.text(np.pi*0.5, 1.38, 'MED',  ha='center', color='#e9830f', fontsize=9, fontweight='700')
            ax.text(np.pi*0.0, 1.38, 'HIGH', ha='right',  color='#e63946', fontsize=9, fontweight='700')
            ax.text(np.pi/2, 0.38, f'{pct:.1f}%', ha='center', va='center',
                    color='#e8f4fd', fontsize=22, fontweight='700')
            ax.set_ylim(0,1.5); ax.set_xlim(0,np.pi)
            ax.set_yticklabels([]); ax.set_xticklabels([])
            ax.spines['polar'].set_visible(False); ax.grid(False)
            plt.tight_layout()
            st.pyplot(fig)

            # Probability bar
            fig2, ax2 = plt.subplots(figsize=(5, 1.6))
            fig_style(fig2, ax2)
            ax2.barh(['Retain','Churn'], [1-prob, prob],
                     color=['#2dc653','#e63946'], height=0.45, edgecolor='none')
            ax2.set_xlim(0, 1)
            for i, v in enumerate([1-prob, prob]):
                ax2.text(v+0.01, i, f'{v*100:.1f}%', va='center', color='#c9d6e3', fontsize=10, fontweight='600')
            for sp in ax2.spines.values(): sp.set_visible(False)
            plt.tight_layout()
            st.pyplot(fig2)

        with col_r:
            # Risk Factors
            flags = []
            if contract == 'Month-to-month':                    flags.append(("🔴 Kontrak Month-to-Month", "high"))
            if tenure <= 6:                                     flags.append(("🔴 Pelanggan baru (≤6 bln)", "high"))
            if tech_support == 'No' and internet != 'No':      flags.append(("🟡 Tanpa Tech Support", "med"))
            if internet == 'Fiber optic':                       flags.append(("🟡 Fiber Optic (high churn segment)", "med"))
            if monthly_charges > 70:                            flags.append(("🟡 Monthly Charges > $70", "med"))
            if payment_method == 'Electronic check':            flags.append(("🟡 Electronic Check Payment", "med"))
            if not flags:                                       flags.append(("🟢 Tidak ada faktor risiko signifikan", "low"))

            st.markdown('<div class="sec-head">Faktor Risiko Aktif</div>', unsafe_allow_html=True)
            for flag, level in flags:
                cls = "alert-high" if level=="high" else ("alert-medium" if level=="med" else "alert-low")
                st.markdown(f'<div class="{cls}" style="padding:8px 14px;margin:4px 0">{flag}</div>', unsafe_allow_html=True)

            # Rekomendasi
            recs = []
            if contract == 'Month-to-month' and prob >= 0.5:
                recs.append("💼 Tawarkan diskon 15–20% upgrade ke kontrak tahunan")
            if tenure <= 6:
                recs.append("🤝 Aktifkan onboarding check-in bulan ke-2 dan ke-3")
            if tech_support == 'No' and internet != 'No':
                recs.append("🛠 Trial TechSupport 1 bulan gratis")
            if payment_method == 'Electronic check' and prob >= 0.5:
                recs.append("🏦 Dorong migrasi ke auto-payment + cashback")
            if monthly_charges > 70 and prob >= 0.5:
                recs.append("💰 Review bundling paket value-for-money")
            if not recs:
                recs.append("✅ Pelanggan risiko rendah — pertahankan kualitas layanan")

            st.markdown('<div class="sec-head">Rekomendasi Tindakan</div>', unsafe_allow_html=True)
            for rec in recs:
                st.markdown(f'<div class="alert-medium" style="padding:8px 14px;margin:4px 0">{rec}</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.caption("🤖 Random Forest · AUC 0.834 · IBM Telco Dataset · dibimbing.id Talent Showcase")
