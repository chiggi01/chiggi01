"""
AI-Based Tax Fraud Risk Prediction System
==========================================
Streamlit frontend integrating pre-trained Random Forest + Isolation Forest models.
Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Optional: use plotly if available, fall back to matplotlib
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    USE_PLOTLY = True
except ImportError:
    USE_PLOTLY = False

# ══════════════════════════════════════════════════════
#  PAGE CONFIGURATION
# ══════════════════════════════════════════════════════
st.set_page_config(
    page_title="Tax Fraud Risk Prediction",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS — Government of India inspired theme
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600;700&family=Source+Serif+Pro:wght@600;700&display=swap');

/* ── Global */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #f4f6f9;
    font-family: 'Source Sans Pro', sans-serif;
    color: #1a1a2e;
}

/* ── Main content area */
[data-testid="stMain"] {
    background-color: #f4f6f9;
}

/* ── Sidebar */
[data-testid="stSidebar"] {
    background-color: #0d2137 !important;
}
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown div,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: #c8d8e8 !important;
}
[data-testid="stSidebar"] b,
[data-testid="stSidebar"] strong {
    color: #ffffff !important;
}

/* ── Metric cards */
[data-testid="metric-container"] {
    background-color: #ffffff !important;
    border: 1px solid #d0dae8;
    border-top: 4px solid #1a4f8a;
    border-radius: 6px;
    padding: 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}
[data-testid="metric-container"] label {
    color: #4a5568 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #0d2137 !important;
    font-size: 28px !important;
    font-weight: 700 !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    color: #4a5568 !important;
}

/* ── Headers */
h1 {
    color: #0d2137 !important;
    font-family: 'Source Serif Pro', serif !important;
    font-weight: 700 !important;
}
h2 {
    color: #0d2137 !important;
    font-family: 'Source Serif Pro', serif !important;
    border-bottom: 3px solid #FF6B00;
    padding-bottom: 8px;
    margin-bottom: 20px !important;
}
h3 {
    color: #0d2137 !important;
    font-weight: 600 !important;
}

/* ── Risk badge */
.risk-high   {background:#c0392b; color:#fff; padding:5px 14px; border-radius:4px; font-weight:700; font-size:13px; display:inline-block;}
.risk-medium {background:#d97706; color:#fff; padding:5px 14px; border-radius:4px; font-weight:700; font-size:13px; display:inline-block;}
.risk-low    {background:#166534; color:#fff; padding:5px 14px; border-radius:4px; font-weight:700; font-size:13px; display:inline-block;}

/* ── Top banner */
.top-banner {
    background: linear-gradient(135deg, #0d2137 0%, #1a4f8a 60%, #0d2137 100%);
    color: white;
    padding: 24px 32px;
    border-radius: 8px;
    margin-bottom: 24px;
    border-left: 6px solid #FF6B00;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}
.top-banner h1 {
    color: white !important;
    font-size: 26px !important;
    margin: 0 0 6px 0 !important;
}
.top-banner p {
    color: #b8cce4 !important;
    font-size: 14px;
    margin: 0;
}

/* ── Info card — WHITE background, DARK text */
.info-card {
    background: #ffffff;
    border-left: 5px solid #1a4f8a;
    border-radius: 6px;
    padding: 16px 18px;
    margin: 8px 0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    color: #1a1a2e !important;
}
.info-card b, .info-card strong {
    color: #0d2137 !important;
}
.info-card p, .info-card span, .info-card div {
    color: #2d3748 !important;
}
.info-card h3 {
    color: #0d2137 !important;
    margin-top: 0 !important;
}
.info-card ul li {
    color: #2d3748 !important;
}
.info-card table td {
    color: #2d3748 !important;
}
.info-card pre {
    color: #1a1a2e !important;
}

/* ── Alert card — light red, DARK text */
.alert-card {
    background: #fff5f5;
    border: 1px solid #e53e3e;
    border-left: 5px solid #e53e3e;
    border-radius: 6px;
    padding: 16px 18px;
    margin: 8px 0;
    color: #1a1a2e !important;
}
.alert-card b, .alert-card strong {
    color: #742a2a !important;
}

/* ── Success / neutral card */
.ok-card {
    background: #f0fff4;
    border: 1px solid #38a169;
    border-left: 5px solid #38a169;
    border-radius: 6px;
    padding: 16px 18px;
    margin: 8px 0;
    color: #1a1a2e !important;
}

/* ── Sidebar nav button */
.stButton > button {
    width: 100%;
    background-color: #1a4f8a;
    color: #ffffff !important;
    border: none;
    border-radius: 4px;
    padding: 10px 14px;
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 4px;
    text-align: left;
    transition: background 0.2s;
    letter-spacing: 0.2px;
}
.stButton > button:hover {
    background-color: #FF6B00 !important;
    color: #ffffff !important;
}

/* ── Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 6px;
    border: 1px solid #d0dae8;
    overflow: hidden;
}

/* ── Selectbox / input labels */
label[data-testid="stWidgetLabel"] p {
    color: #2d3748 !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}

/* ── Tab styling */
[data-testid="stHorizontalBlock"] {
    gap: 16px;
}

/* ── Government emblem strip */
.gov-strip {
    background: #0d2137;
    color: #b8cce4;
    font-size: 11px;
    padding: 6px 16px;
    border-radius: 4px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Section sub-label */
.section-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #1a4f8a;
    margin-bottom: 6px;
}

/* ── Streamlit text elements override */
p, li, td, th, span {
    color: #2d3748;
}

/* ── Markdown in main area */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] td,
[data-testid="stMarkdownContainer"] th {
    color: #2d3748 !important;
}

/* ── Caption */
[data-testid="stCaptionContainer"] p {
    color: #718096 !important;
}

/* ── Download button */
[data-testid="stDownloadButton"] button {
    background-color: #1a4f8a !important;
    color: white !important;
    border: none;
    border-radius: 4px;
    font-weight: 600;
}
[data-testid="stDownloadButton"] button:hover {
    background-color: #FF6B00 !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  DATA & MODEL LOADING
# ══════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading ML models…")
def load_models():
    model        = joblib.load("fraud_model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    return model, preprocessor

@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    df = pd.read_csv("taxpayer_data.csv")
    if 'Risk_Level' not in df.columns:
        df['Risk_Level'] = pd.cut(df['Fraud_Probability'],
                                   bins=[0, 0.35, 0.65, 1.0],
                                   labels=['Low', 'Medium', 'High'])
    if 'Anomaly_Flag' not in df.columns:
        df['Anomaly_Flag'] = 0
    return df

try:
    rf_model, preprocessor = load_models()
    df = load_data()
    feature_cols = preprocessor['feature_cols']
    le_prof = preprocessor['label_encoder_profession']
    le_city = preprocessor['label_encoder_city']
    MODEL_LOADED = True
except FileNotFoundError as e:
    MODEL_LOADED = False
    LOAD_ERROR = str(e)


# ══════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════
def risk_badge(level):
    level = str(level)
    cls = {'High': 'risk-high', 'Medium': 'risk-medium', 'Low': 'risk-low'}.get(level, 'risk-low')
    return f'<span class="{cls}">⚠ {level} Risk</span>'

def format_inr(value):
    return f"₹{value:,.0f}"

def get_investigation_suggestions(row):
    tips = []
    if row['Declared_Expenses'] > 0.80 * row['Annual_Income']:
        ratio = row['Declared_Expenses'] / row['Annual_Income'] * 100
        tips.append(f"📋 <b>Expense Verification:</b> Declared expenses are {ratio:.1f}% of income — "
                    f"obtain original receipts and invoices for all claimed deductions.")
    if row['Investment_Claims'] > 0.30 * row['Annual_Income']:
        ratio = row['Investment_Claims'] / row['Annual_Income'] * 100
        tips.append(f"📈 <b>Investment Audit:</b> Investment claims represent {ratio:.1f}% of income — "
                    f"verify supporting documents (ELSS, PPF, insurance policies).")
    if row['Late_Filing_Count'] > 3:
        tips.append(f"⏰ <b>Filing History Review:</b> {int(row['Late_Filing_Count'])} late filings detected — "
                    f"cross-check amendment history and late payment penalties.")
    if row['Previous_Penalty'] == 1:
        tips.append("⚖️ <b>Penalty Record:</b> Taxpayer has a prior penalty on record — "
                    "retrieve historical assessment orders.")
    if row.get('Anomaly_Flag', 0) == 1:
        tips.append("🔎 <b>Statistical Anomaly:</b> Financial figures deviate significantly from "
                    "peer group — conduct third-party income verification.")
    if not tips:
        tips.append("✅ <b>No Critical Red Flags:</b> This taxpayer's profile is within normal ranges. "
                    "Standard periodic review recommended.")
    return tips


# ══════════════════════════════════════════════════════
#  SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:16px 0 20px 0;'>
        <div style='font-size:44px; line-height:1;'>🧾</div>
        <div style='font-size:14px; font-weight:700; letter-spacing:2px; color:#ffffff; margin-top:10px;'>
            TAX FRAUD
        </div>
        <div style='font-size:12px; font-weight:600; letter-spacing:1px; color:#FF6B00;'>
            RISK INTELLIGENCE SYSTEM
        </div>
        <div style='font-size:10px; color:#7a9bbf; margin-top:4px; letter-spacing:0.5px;'>
            Powered by AI · ML Edition
        </div>
    </div>
    <div style='border-top: 1px solid #1e3a5f; margin: 0 0 18px 0;'></div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:10px; font-weight:700; letter-spacing:2px; color:#7a9bbf; margin-bottom:8px;'>NAVIGATION</div>", unsafe_allow_html=True)

    pages = {
        "🏠  Home"             : "Home",
        "📊  Dashboard"        : "Dashboard",
        "🔍  Taxpayer Search"  : "Taxpayer Search",
        "⚠️  Risk Analysis"    : "Risk Analysis",
        "ℹ️  About"            : "About",
    }
    if "page" not in st.session_state:
        st.session_state.page = "Home"

    for label, key in pages.items():
        if st.button(label, key=f"nav_{key}"):
            st.session_state.page = key

    st.markdown("<div style='border-top: 1px solid #1e3a5f; margin: 18px 0;'></div>", unsafe_allow_html=True)

    if MODEL_LOADED:
        total      = len(df)
        high_count = (df['Risk_Level'] == 'High').sum()
        anom_count = df['Anomaly_Flag'].sum()
        st.markdown(f"""
        <div style='background:#0a1929; border-radius:6px; padding:14px; font-size:12px;'>
            <div style='font-size:10px; font-weight:700; letter-spacing:2px; color:#7a9bbf; margin-bottom:10px;'>LIVE STATISTICS</div>
            <div style='display:flex; justify-content:space-between; margin-bottom:6px;'>
                <span style='color:#8aa8c8;'>Total Records</span>
                <span style='color:#ffffff; font-weight:700;'>{total:,}</span>
            </div>
            <div style='display:flex; justify-content:space-between; margin-bottom:6px;'>
                <span style='color:#8aa8c8;'>High Risk</span>
                <span style='color:#ef4444; font-weight:700;'>{high_count:,}</span>
            </div>
            <div style='display:flex; justify-content:space-between;'>
                <span style='color:#8aa8c8;'>Anomalies</span>
                <span style='color:#f59e0b; font-weight:700;'>{anom_count:,}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

page = st.session_state.page


# ══════════════════════════════════════════════════════
#  GUARD: model not found
# ══════════════════════════════════════════════════════
if not MODEL_LOADED:
    st.error(f"❌ Could not load model files: `{LOAD_ERROR}`")
    st.info("Please ensure `fraud_model.pkl`, `preprocessor.pkl`, and `taxpayer_data.csv` "
            "are in the same directory as `app.py`.")
    st.stop()


# ══════════════════════════════════════════════════════
#  PAGE: HOME
# ══════════════════════════════════════════════════════
if page == "Home":
    st.markdown("""
    <div class='top-banner'>
        <div style='font-size:11px; letter-spacing:2px; color:#FF6B00; font-weight:700; margin-bottom:8px;'>
            GOVERNMENT TAX INTELLIGENCE PLATFORM
        </div>
        <h1>🧾 AI-Based Tax Fraud Risk Prediction System</h1>
        <p>An intelligent platform for automated tax audit prioritization using Machine Learning
        and Anomaly Detection. For authorized personnel only.</p>
    </div>
    """, unsafe_allow_html=True)

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    total  = len(df)
    high_c = int((df['Risk_Level'] == 'High').sum())
    med_c  = int((df['Risk_Level'] == 'Medium').sum())
    low_c  = int((df['Risk_Level'] == 'Low').sum())
    anom_c = int(df['Anomaly_Flag'].sum())

    col1.metric("Total Taxpayers", f"{total:,}", delta="2025 Dataset")
    col2.metric("🔴 High Risk",    f"{high_c:,}",  delta=f"{high_c/total*100:.1f}% of total", delta_color="inverse")
    col3.metric("🟡 Medium Risk",  f"{med_c:,}",   delta=f"{med_c/total*100:.1f}%", delta_color="off")
    col4.metric("🚨 Anomalies",    f"{anom_c:,}",  delta="Isolation Forest", delta_color="inverse")

    st.markdown("---")
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.subheader("How This System Works")
        steps = [
            ("Step 1 — Data Ingestion",
             "Taxpayer financial data is ingested and standardised. Categorical attributes (Profession, City) are encoded for machine learning."),
            ("Step 2 — Fraud Risk Classification",
             "A <em>Random Forest Classifier</em> (200 trees) trained on historical patterns assigns each taxpayer a fraud probability score from 0–100%."),
            ("Step 3 — Anomaly Detection",
             "An <em>Isolation Forest</em> model independently flags statistical outliers in income, expenses, and investment claims."),
            ("Step 4 — Risk Triage",
             "Taxpayers are ranked into High / Medium / Low risk tiers, enabling auditors to prioritise investigations efficiently."),
        ]
        for title, body in steps:
            st.markdown(f"""
            <div class='info-card'>
                <div style='font-size:10px; font-weight:700; letter-spacing:1.5px; color:#FF6B00; margin-bottom:4px; text-transform:uppercase;'>◆ {title}</div>
                <div style='color:#2d3748; font-size:14px; line-height:1.6;'>{body}</div>
            </div>
            """, unsafe_allow_html=True)

    with col_r:
        st.subheader("Risk Level Guide")
        st.markdown("""
        <div class='info-card'>
            <table style='width:100%; border-collapse:collapse; font-size:13px;'>
                <thead>
                    <tr style='border-bottom:2px solid #e2e8f0;'>
                        <th style='padding:8px 6px; text-align:left; color:#0d2137;'>Badge</th>
                        <th style='padding:8px 6px; text-align:left; color:#0d2137;'>Range</th>
                        <th style='padding:8px 6px; text-align:left; color:#0d2137;'>Action</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style='border-bottom:1px solid #f0f4f8;'>
                        <td style='padding:8px 6px;'><span class='risk-high'>● High</span></td>
                        <td style='padding:8px 6px; color:#2d3748;'>&gt; 65%</td>
                        <td style='padding:8px 6px; color:#2d3748;'>Immediate audit</td>
                    </tr>
                    <tr style='border-bottom:1px solid #f0f4f8;'>
                        <td style='padding:8px 6px;'><span class='risk-medium'>● Medium</span></td>
                        <td style='padding:8px 6px; color:#2d3748;'>35–65%</td>
                        <td style='padding:8px 6px; color:#2d3748;'>Scheduled review</td>
                    </tr>
                    <tr>
                        <td style='padding:8px 6px;'><span class='risk-low'>● Low</span></td>
                        <td style='padding:8px 6px; color:#2d3748;'>&lt; 35%</td>
                        <td style='padding:8px 6px; color:#2d3748;'>Standard monitoring</td>
                    </tr>
                </tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("Fraud Trigger Conditions")
        st.markdown("""
        <div class='info-card'>
            <div style='color:#2d3748; font-size:14px; line-height:2;'>
                📌 &nbsp;Expenses &gt; 80% of annual income<br>
                📌 &nbsp;Investment claims &gt; 30% of income<br>
                📌 &nbsp;Late filings &gt; 3 times<br>
                📌 &nbsp;Previous tax penalty on record
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  PAGE: DASHBOARD
# ══════════════════════════════════════════════════════
elif page == "Dashboard":
    st.markdown("<h2>📊 Executive Dashboard</h2>", unsafe_allow_html=True)

    k1, k2, k3, k4, k5 = st.columns(5)
    total   = len(df)
    fraud_r = df['Fraud_Label'].mean()
    avg_inc = df['Annual_Income'].mean()
    anom_r  = df['Anomaly_Flag'].mean()

    k1.metric("Total Taxpayers", f"{total:,}")
    k2.metric("Fraud Rate",      f"{fraud_r:.1%}")
    k3.metric("Avg. Income",     f"₹{avg_inc/1e5:.1f}L")
    k4.metric("Anomaly Rate",    f"{anom_r:.1%}")
    k5.metric("High Risk",       f"{(df['Risk_Level']=='High').sum():,}")

    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Risk Distribution")
        risk_counts = df['Risk_Level'].value_counts()
        colors = {'High': '#c0392b', 'Medium': '#d97706', 'Low': '#166534'}
        if USE_PLOTLY:
            fig = px.pie(
                values=risk_counts.values, names=risk_counts.index,
                color=risk_counts.index, color_discrete_map=colors, hole=0.42
            )
            fig.update_layout(height=320, margin=dict(t=10, b=10, l=10, r=10),
                              font=dict(color='#2d3748'),
                              legend=dict(orientation='h', y=-0.1, font=dict(color='#2d3748')))
            fig.update_traces(textfont_color='white')
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            wedge_colors = [colors.get(str(k), '#7f8c8d') for k in risk_counts.index]
            ax.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%',
                   colors=wedge_colors, startangle=140,
                   textprops={'color': '#2d3748', 'fontsize': 10})
            ax.set_title('Risk Level Distribution', fontweight='bold', color='#0d2137')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with c2:
        st.subheader("Fraud Rate by Profession")
        fraud_prof = df.groupby('Profession')['Fraud_Label'].mean().sort_values(ascending=False)
        if USE_PLOTLY:
            fig = px.bar(
                x=fraud_prof.values, y=fraud_prof.index, orientation='h',
                color=fraud_prof.values,
                color_continuous_scale=['#166534', '#d97706', '#c0392b'],
                labels={'x': 'Fraud Rate', 'y': 'Profession'},
                text=[f'{v:.1%}' for v in fraud_prof.values]
            )
            fig.update_layout(height=320, margin=dict(t=10, b=10, l=10, r=10),
                              coloraxis_showscale=False,
                              font=dict(color='#2d3748'),
                              yaxis={'categoryorder': 'total ascending'})
            fig.update_traces(textposition='outside', textfont_color='#2d3748')
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.barh(fraud_prof.index, fraud_prof.values, color='#1a4f8a', alpha=0.85)
            ax.set_xlabel('Fraud Rate', color='#2d3748')
            ax.set_title('Fraud Rate by Profession', fontweight='bold', color='#0d2137')
            ax.tick_params(colors='#2d3748')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    c3, c4 = st.columns(2)

    with c3:
        st.subheader("Income vs Declared Expenses")
        sample = df.sample(min(800, len(df)), random_state=42)
        if USE_PLOTLY:
            color_map = {'High': '#c0392b', 'Medium': '#d97706', 'Low': '#166534'}
            fig = px.scatter(
                sample, x='Annual_Income', y='Declared_Expenses', color='Risk_Level',
                color_discrete_map=color_map, opacity=0.55,
                labels={'Annual_Income': 'Annual Income (₹)', 'Declared_Expenses': 'Declared Expenses (₹)'}
            )
            fig.update_layout(height=320, margin=dict(t=10, b=10, l=10, r=10),
                              font=dict(color='#2d3748'))
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            for level, clr in [('High', '#c0392b'), ('Medium', '#d97706'), ('Low', '#166534')]:
                s = sample[sample['Risk_Level'] == level]
                ax.scatter(s['Annual_Income']/1e5, s['Declared_Expenses']/1e5,
                           c=clr, label=level, alpha=0.4, s=12)
            ax.set_xlabel('Income (₹ Lakhs)', color='#2d3748')
            ax.set_ylabel('Expenses (₹ Lakhs)', color='#2d3748')
            ax.legend()
            ax.set_title('Income vs Expenses', fontweight='bold', color='#0d2137')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with c4:
        st.subheader("City-wise High Risk Count")
        city_risk = df[df['Risk_Level'] == 'High'].groupby('City').size().sort_values(ascending=False)
        if USE_PLOTLY:
            fig = px.bar(
                x=city_risk.index, y=city_risk.values,
                color=city_risk.values,
                color_continuous_scale=['#1a4f8a', '#c0392b'],
                labels={'x': 'City', 'y': 'High Risk Count'},
                text=city_risk.values
            )
            fig.update_layout(height=320, margin=dict(t=10, b=10, l=10, r=10),
                              coloraxis_showscale=False, font=dict(color='#2d3748'))
            fig.update_traces(textposition='outside', textfont_color='#2d3748')
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.bar(city_risk.index, city_risk.values, color='#1a4f8a', alpha=0.85)
            ax.set_xticklabels(city_risk.index, rotation=45, ha='right', color='#2d3748')
            ax.set_ylabel('Count', color='#2d3748')
            ax.set_title('High Risk by City', fontweight='bold', color='#0d2137')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    st.subheader("Fraud Probability Distribution Across All Taxpayers")
    if USE_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df[df['Fraud_Label'] == 0]['Fraud_Probability'],
            name='Legitimate', marker_color='#166534', opacity=0.65, nbinsx=40
        ))
        fig.add_trace(go.Histogram(
            x=df[df['Fraud_Label'] == 1]['Fraud_Probability'],
            name='Fraud', marker_color='#c0392b', opacity=0.65, nbinsx=40
        ))
        fig.add_vline(x=0.5, line_dash='dash', line_color='#0d2137',
                      annotation_text='Threshold 0.50', annotation_font_color='#0d2137')
        fig.update_layout(barmode='overlay', height=280,
                          margin=dict(t=10, b=10, l=10, r=10),
                          xaxis_title='Fraud Probability', yaxis_title='Count',
                          font=dict(color='#2d3748'),
                          legend=dict(orientation='h', y=1.1, font=dict(color='#2d3748')))
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.hist(df[df['Fraud_Label'] == 0]['Fraud_Probability'], bins=40,
                alpha=0.65, color='#166534', label='Legitimate')
        ax.hist(df[df['Fraud_Label'] == 1]['Fraud_Probability'], bins=40,
                alpha=0.65, color='#c0392b', label='Fraud')
        ax.axvline(0.5, ls='--', color='#0d2137', label='Threshold')
        ax.set_xlabel('Fraud Probability', color='#2d3748')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ══════════════════════════════════════════════════════
#  PAGE: TAXPAYER SEARCH
# ══════════════════════════════════════════════════════
elif page == "Taxpayer Search":
    st.markdown("<h2>🔍 Taxpayer Search</h2>", unsafe_allow_html=True)

    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        search_id = st.text_input("Search by Taxpayer ID", placeholder="e.g. TXP10042")
    with fc2:
        prof_options = ["All"] + sorted(df['Profession'].unique().tolist())
        filter_prof = st.selectbox("Filter by Profession", prof_options)
    with fc3:
        risk_options = ["All", "High", "Medium", "Low"]
        filter_risk = st.selectbox("Filter by Risk Level", risk_options)
    with fc4:
        city_options = ["All"] + sorted(df['City'].unique().tolist())
        filter_city = st.selectbox("Filter by City", city_options)

    filtered = df.copy()
    if search_id.strip():
        filtered = filtered[filtered['Taxpayer_ID'].str.contains(search_id.strip(), case=False)]
    if filter_prof != "All":
        filtered = filtered[filtered['Profession'] == filter_prof]
    if filter_risk != "All":
        filtered = filtered[filtered['Risk_Level'] == filter_risk]
    if filter_city != "All":
        filtered = filtered[filtered['City'] == filter_city]

    st.markdown(f"""
    <div style='background:#1a4f8a; color:#ffffff; padding:10px 16px; border-radius:6px;
                font-size:14px; font-weight:600; margin-bottom:12px;'>
        🔎 &nbsp; <b>{len(filtered):,} taxpayers</b> found matching your criteria
    </div>
    """, unsafe_allow_html=True)

    display_cols = ['Taxpayer_ID', 'Age', 'Profession', 'City', 'Annual_Income',
                    'Declared_Expenses', 'Investment_Claims', 'Late_Filing_Count',
                    'Previous_Penalty', 'Fraud_Label', 'Risk_Level', 'Anomaly_Flag', 'Fraud_Probability']
    display_df = filtered[display_cols].copy()
    display_df['Annual_Income']     = display_df['Annual_Income'].apply(lambda x: f"₹{x:,.0f}")
    display_df['Declared_Expenses'] = display_df['Declared_Expenses'].apply(lambda x: f"₹{x:,.0f}")
    display_df['Investment_Claims'] = display_df['Investment_Claims'].apply(lambda x: f"₹{x:,.0f}")
    display_df['Fraud_Probability'] = display_df['Fraud_Probability'].apply(lambda x: f"{x:.2%}")
    display_df['Anomaly_Flag']      = display_df['Anomaly_Flag'].map({0: 'Normal', 1: '⚠ Anomaly'})
    display_df['Fraud_Label']       = display_df['Fraud_Label'].map({0: 'Legitimate', 1: '🚨 Fraud'})

    st.dataframe(display_df.head(100), use_container_width=True, height=420)

    if len(filtered) > 100:
        st.caption(f"Showing first 100 of {len(filtered):,} results. Refine filters to narrow down.")

    csv = filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Filtered Results as CSV",
        data=csv, file_name="filtered_taxpayers.csv", mime="text/csv"
    )


# ══════════════════════════════════════════════════════
#  PAGE: RISK ANALYSIS
# ══════════════════════════════════════════════════════
elif page == "Risk Analysis":
    st.markdown("<h2>⚠️ Individual Risk Analysis</h2>", unsafe_allow_html=True)

    taxpayer_ids = sorted(df['Taxpayer_ID'].tolist())
    col_sel1, col_sel2 = st.columns([2, 3])
    with col_sel1:
        selected_id = st.selectbox(
            "Select Taxpayer ID for Analysis", taxpayer_ids, index=0,
            help="Choose any taxpayer to view their complete risk profile."
        )

    row = df[df['Taxpayer_ID'] == selected_id].iloc[0]
    risk = str(row['Risk_Level'])
    fraud_prob = float(row['Fraud_Probability'])

    h1, h2, h3 = st.columns(3)

    with h1:
        st.markdown(f"""
        <div class='info-card'>
            <div style='font-size:10px; font-weight:700; letter-spacing:1.5px; color:#1a4f8a; margin-bottom:6px;'>TAXPAYER PROFILE</div>
            <div style='font-size:24px; font-weight:800; color:#0d2137;'>{selected_id}</div>
            <div style='font-size:14px; color:#4a5568; margin-top:4px;'>{row['Profession']} &nbsp;·&nbsp; {row['City']}</div>
            <div style='font-size:13px; color:#718096; margin-top:2px;'>Age: {int(row['Age'])}</div>
        </div>
        """, unsafe_allow_html=True)

    with h2:
        risk_color = {'High': '#c0392b', 'Medium': '#d97706', 'Low': '#166534'}.get(risk, '#666')
        bar_width  = int(fraud_prob * 100)
        st.markdown(f"""
        <div class='info-card'>
            <div style='font-size:10px; font-weight:700; letter-spacing:1.5px; color:#1a4f8a; margin-bottom:6px;'>FRAUD RISK SCORE</div>
            <div style='font-size:40px; font-weight:800; color:{risk_color}; line-height:1;'>{fraud_prob:.1%}</div>
            <div style='background:#e2e8f0; border-radius:4px; height:8px; margin:10px 0;'>
                <div style='width:{bar_width}%; background:{risk_color}; height:8px; border-radius:4px;'></div>
            </div>
            {risk_badge(risk)}
        </div>
        """, unsafe_allow_html=True)

    with h3:
        anomaly_icon  = "⚠️ Flagged as Anomaly" if row['Anomaly_Flag'] == 1 else "✅ No Anomaly Detected"
        anomaly_color = "#c0392b" if row['Anomaly_Flag'] == 1 else "#166534"
        fraud_icon    = "🚨 Fraud Record Exists" if row['Fraud_Label'] == 1 else "✅ Clean Record"
        fraud_color   = "#c0392b" if row['Fraud_Label'] == 1 else "#166534"
        st.markdown(f"""
        <div class='info-card'>
            <div style='font-size:10px; font-weight:700; letter-spacing:1.5px; color:#1a4f8a; margin-bottom:6px;'>FLAGS & HISTORY</div>
            <div style='font-size:14px; font-weight:700; color:{anomaly_color}; margin-bottom:4px;'>{anomaly_icon}</div>
            <div style='font-size:14px; font-weight:700; color:{fraud_color}; margin-bottom:8px;'>{fraud_icon}</div>
            <div style='font-size:13px; color:#4a5568;'>
                Late Filings: <b style='color:#0d2137;'>{int(row['Late_Filing_Count'])}</b>
                &nbsp;&nbsp;|&nbsp;&nbsp;
                Prior Penalty: <b style='color:#0d2137;'>{"Yes" if row['Previous_Penalty'] == 1 else "No"}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Income & Expense Breakdown")
        peer_income  = df[df['Profession'] == row['Profession']]['Annual_Income'].mean()
        peer_expense = df[df['Profession'] == row['Profession']]['Declared_Expenses'].mean()
        peer_invest  = df[df['Profession'] == row['Profession']]['Investment_Claims'].mean()

        categories    = ['Annual Income', 'Declared Expenses', 'Investment Claims']
        taxpayer_vals = [row['Annual_Income'], row['Declared_Expenses'], row['Investment_Claims']]
        peer_vals     = [peer_income, peer_expense, peer_invest]

        if USE_PLOTLY:
            fig = go.Figure(data=[
                go.Bar(name='This Taxpayer', x=categories, y=taxpayer_vals,
                       marker_color='#c0392b', opacity=0.85),
                go.Bar(name=f'Peer Avg ({row["Profession"]})', x=categories, y=peer_vals,
                       marker_color='#1a4f8a', opacity=0.75)
            ])
            fig.update_layout(barmode='group', height=320,
                              yaxis_tickprefix='₹', yaxis_tickformat=',',
                              margin=dict(t=10, b=10, l=10, r=10),
                              font=dict(color='#2d3748'),
                              legend=dict(orientation='h', y=1.1, font=dict(color='#2d3748')))
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            x = np.arange(len(categories)); w = 0.35
            ax.bar(x - w/2, taxpayer_vals, w, label='This Taxpayer', color='#c0392b', alpha=0.85)
            ax.bar(x + w/2, peer_vals,     w, label='Peer Avg',      color='#1a4f8a', alpha=0.75)
            ax.set_xticks(x)
            ax.set_xticklabels(categories, fontsize=8, color='#2d3748')
            ax.set_ylabel('₹', color='#2d3748')
            ax.legend(fontsize=8)
            ax.set_title('Income & Expense vs Peer Avg', fontweight='bold', color='#0d2137')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown(f"""
        <div class='info-card' style='border-left-color:#1a4f8a;'>
            <div style='font-size:10px; font-weight:700; letter-spacing:1.5px; color:#1a4f8a; margin-bottom:8px;'>
                PEER BENCHMARK — {row['Profession'].upper()}
            </div>
            <div style='display:grid; grid-template-columns:1fr 1fr; gap:4px; font-size:13px;'>
                <span style='color:#4a5568;'>Average Income</span>
                <span style='color:#0d2137; font-weight:700;'>{format_inr(peer_income)}</span>
                <span style='color:#4a5568;'>Average Expenses</span>
                <span style='color:#0d2137; font-weight:700;'>{format_inr(peer_expense)}</span>
                <span style='color:#4a5568;'>Expense Ratio</span>
                <span style='color:#0d2137; font-weight:700;'>{peer_expense/peer_income:.1%}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.subheader("Feature Importance (Model View)")
        importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values()
        if USE_PLOTLY:
            bar_colors_fi = ['#c0392b' if v >= importances.median() else '#1a4f8a'
                             for v in importances.values]
            fig = go.Figure(go.Bar(
                x=importances.values, y=importances.index, orientation='h',
                marker_color=bar_colors_fi,
                text=[f'{v:.3f}' for v in importances.values],
                textposition='outside'
            ))
            fig.update_layout(height=320, margin=dict(t=10, b=10, l=10, r=10),
                              xaxis_title='Importance Score',
                              font=dict(color='#2d3748'))
            fig.update_traces(textfont_color='#2d3748')
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            bar_colors_fi = ['#c0392b' if v >= importances.median() else '#1a4f8a'
                             for v in importances.values]
            ax.barh(importances.index, importances.values, color=bar_colors_fi, alpha=0.9)
            ax.set_xlabel('Importance', color='#2d3748')
            ax.set_title('Feature Importance', fontweight='bold', color='#0d2137')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # Risk condition checklist
    st.markdown("---")
    st.subheader("Risk Condition Checklist")
    exp_ratio = row['Declared_Expenses'] / row['Annual_Income']
    inv_ratio = row['Investment_Claims']  / row['Annual_Income']
    lfc = int(row['Late_Filing_Count'])
    pp  = int(row['Previous_Penalty'])

    r1, r2, r3, r4 = st.columns(4)

    def condition_card(col, icon, label, value_text, threshold_text, triggered):
        card_class = 'alert-card' if triggered else 'ok-card'
        status_color = '#c0392b' if triggered else '#166534'
        status_text  = '⚠ TRIGGERED' if triggered else '✅ WITHIN LIMIT'
        col.markdown(f"""
        <div class='{card_class}'>
            <div style='font-size:10px; font-weight:700; letter-spacing:1px; color:#4a5568; margin-bottom:4px;'>{icon} {label.upper()}</div>
            <div style='font-size:28px; font-weight:800; color:{status_color};'>{value_text}</div>
            <div style='font-size:12px; color:#718096; margin-top:2px;'>Threshold: {threshold_text}</div>
            <div style='font-size:12px; font-weight:700; color:{status_color}; margin-top:4px;'>{status_text}</div>
        </div>
        """, unsafe_allow_html=True)

    condition_card(r1, "💰", "Expense Ratio",    f"{exp_ratio:.1%}", "80%",  exp_ratio > 0.80)
    condition_card(r2, "📈", "Investment Ratio",  f"{inv_ratio:.1%}", "30%",  inv_ratio > 0.30)
    condition_card(r3, "⏰", "Late Filings",      str(lfc),           "> 3",  lfc > 3)
    condition_card(r4, "⚖️", "Prior Penalty",     "Yes" if pp==1 else "No", "Any", pp == 1)

    # Investigation suggestions
    st.markdown("---")
    st.subheader("📋 Audit Investigation Suggestions")
    suggestions = get_investigation_suggestions(row)
    for tip in suggestions:
        st.markdown(f"<div class='info-card'><div style='color:#2d3748; font-size:14px; line-height:1.7;'>{tip}</div></div>",
                    unsafe_allow_html=True)

    # Similar taxpayers
    st.markdown("---")
    st.subheader(f"👥 Similar High-Risk Taxpayers — {row['Profession']}")
    similar = df[
        (df['Profession'] == row['Profession']) &
        (df['Risk_Level'] == 'High') &
        (df['Taxpayer_ID'] != selected_id)
    ][['Taxpayer_ID', 'Age', 'City', 'Annual_Income', 'Fraud_Probability', 'Anomaly_Flag']].head(8)
    if len(similar) > 0:
        similar['Annual_Income']     = similar['Annual_Income'].apply(lambda x: f"₹{x:,.0f}")
        similar['Fraud_Probability'] = similar['Fraud_Probability'].apply(lambda x: f"{x:.2%}")
        similar['Anomaly_Flag']      = similar['Anomaly_Flag'].map({0: 'Normal', 1: '⚠ Anomaly'})
        st.dataframe(similar, use_container_width=True, hide_index=True)
    else:
        st.info("No similar high-risk taxpayers found for this profession.")


# ══════════════════════════════════════════════════════
#  PAGE: ABOUT
# ══════════════════════════════════════════════════════
elif page == "About":
    st.markdown("<h2>ℹ️ About This System</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        <div class='info-card'>
            <div style='font-size:10px; font-weight:700; letter-spacing:1.5px; color:#FF6B00; margin-bottom:8px;'>◆ PROJECT OBJECTIVE</div>
            <p style='color:#2d3748; font-size:14px; line-height:1.7; margin:0;'>
            This system demonstrates the application of <strong style='color:#0d2137;'>Machine Learning</strong> to
            tax fraud risk detection — a critical challenge for revenue authorities worldwide.
            By combining supervised classification with unsupervised anomaly detection,
            it enables intelligent prioritisation of audit cases, reducing manual workload
            while increasing detection accuracy.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='info-card'>
            <div style='font-size:10px; font-weight:700; letter-spacing:1.5px; color:#FF6B00; margin-bottom:8px;'>◆ ML TECHNIQUES USED</div>
            <p style='color:#0d2137; font-weight:700; margin:0 0 4px 0;'>1. Random Forest Classifier</p>
            <p style='color:#2d3748; font-size:14px; margin:0 0 12px 0; line-height:1.6;'>
            An ensemble of 200 decision trees trained on historical taxpayer patterns.
            Uses <em>class_weight='balanced'</em> to handle the natural imbalance between
            legitimate and fraudulent taxpayers. Outputs fraud probability scores.
            </p>
            <p style='color:#0d2137; font-weight:700; margin:0 0 4px 0;'>2. Isolation Forest</p>
            <p style='color:#2d3748; font-size:14px; margin:0 0 12px 0; line-height:1.6;'>
            An unsupervised anomaly detection algorithm that isolates outliers by randomly
            partitioning the feature space. Applied to financial figures to flag statistical
            outliers independently of labels.
            </p>
            <p style='color:#0d2137; font-weight:700; margin:0 0 4px 0;'>3. Label Encoding</p>
            <p style='color:#2d3748; font-size:14px; margin:0; line-height:1.6;'>
            Categorical features (Profession, City) are ordinally encoded to allow
            numerical processing by tree-based algorithms.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='info-card'>
            <div style='font-size:10px; font-weight:700; letter-spacing:1.5px; color:#FF6B00; margin-bottom:8px;'>◆ MODEL PERFORMANCE</div>
            <table style='width:100%; border-collapse:collapse; font-size:13px;'>
                <thead>
                    <tr style='background:#f0f4f8; border-bottom:2px solid #d0dae8;'>
                        <th style='padding:10px 12px; text-align:left; color:#0d2137;'>Metric</th>
                        <th style='padding:10px 12px; text-align:left; color:#0d2137;'>Value</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style='border-bottom:1px solid #e8eef5;'><td style='padding:9px 12px; color:#2d3748;'>Accuracy</td><td style='padding:9px 12px; color:#0d2137; font-weight:700;'>~87–90%</td></tr>
                    <tr style='border-bottom:1px solid #e8eef5;'><td style='padding:9px 12px; color:#2d3748;'>Precision</td><td style='padding:9px 12px; color:#0d2137; font-weight:700;'>~73–80%</td></tr>
                    <tr style='border-bottom:1px solid #e8eef5;'><td style='padding:9px 12px; color:#2d3748;'>Recall</td><td style='padding:9px 12px; color:#0d2137; font-weight:700;'>~42–55%</td></tr>
                    <tr><td style='padding:9px 12px; color:#2d3748;'>F1 Score</td><td style='padding:9px 12px; color:#0d2137; font-weight:700;'>~55–65%</td></tr>
                </tbody>
            </table>
            <p style='font-size:12px; color:#718096; margin:10px 0 0 0;'>
            ℹ️ Recall is intentionally tuned conservatively to minimise false accusations.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='alert-card'>
            <div style='font-size:10px; font-weight:700; letter-spacing:1.5px; color:#c0392b; margin-bottom:8px;'>⚖️ ETHICAL DISCLAIMER</div>
            <p style='font-size:13px; color:#2d3748; line-height:1.6; margin:0 0 10px 0;'>
            This system is built on <strong style='color:#0d2137;'>synthetic data</strong> for educational and
            demonstration purposes only.
            </p>
            <p style='font-size:13px; color:#2d3748; margin:0 0 8px 0;'>In real-world deployments, AI-based fraud detection systems must be:</p>
            <ul style='font-size:13px; color:#2d3748; margin:0 0 10px 0; padding-left:18px; line-height:1.9;'>
                <li>Audited for algorithmic bias</li>
                <li>Reviewed by qualified human officers</li>
                <li>Compliant with data protection laws</li>
                <li>Transparent to flagged individuals</li>
                <li>Regularly retrained on fresh data</li>
            </ul>
            <p style='font-size:13px; color:#742a2a; font-weight:600; margin:0;'>
            A high risk score is an investigative signal, not a verdict.
            No enforcement action should be taken solely on ML predictions.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='info-card'>
            <div style='font-size:10px; font-weight:700; letter-spacing:1.5px; color:#FF6B00; margin-bottom:8px;'>◆ TECHNOLOGY STACK</div>
            <table style='font-size:13px; width:100%; border-collapse:collapse;'>
                <tr style='border-bottom:1px solid #e8eef5;'><td style='padding:7px 0; color:#1a4f8a; font-weight:700;'>🐍 Python 3.9+</td><td style='padding:7px 0; color:#4a5568;'>Core language</td></tr>
                <tr style='border-bottom:1px solid #e8eef5;'><td style='padding:7px 0; color:#1a4f8a; font-weight:700;'>🌲 scikit-learn</td><td style='padding:7px 0; color:#4a5568;'>ML models</td></tr>
                <tr style='border-bottom:1px solid #e8eef5;'><td style='padding:7px 0; color:#1a4f8a; font-weight:700;'>🐼 pandas</td><td style='padding:7px 0; color:#4a5568;'>Data processing</td></tr>
                <tr style='border-bottom:1px solid #e8eef5;'><td style='padding:7px 0; color:#1a4f8a; font-weight:700;'>📦 joblib</td><td style='padding:7px 0; color:#4a5568;'>Model persistence</td></tr>
                <tr style='border-bottom:1px solid #e8eef5;'><td style='padding:7px 0; color:#1a4f8a; font-weight:700;'>📊 Plotly</td><td style='padding:7px 0; color:#4a5568;'>Visualisations</td></tr>
                <tr><td style='padding:7px 0; color:#1a4f8a; font-weight:700;'>🖥️ Streamlit</td><td style='padding:7px 0; color:#4a5568;'>Web frontend</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='info-card'>
            <div style='font-size:10px; font-weight:700; letter-spacing:1.5px; color:#FF6B00; margin-bottom:8px;'>◆ FILE STRUCTURE</div>
            <pre style='font-size:12px; background:#f0f4f8; color:#0d2137; padding:12px; border-radius:4px; margin:0; line-height:1.8;'>project/
├── app.py
├── Tax_Fraud_Risk_
│   Prediction.ipynb
├── fraud_model.pkl
├── preprocessor.pkl
└── taxpayer_data.csv</pre>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; color:#718096; font-size:13px; padding:12px;
                background:#ffffff; border-radius:6px; border:1px solid #e2e8f0;'>
        🧾 &nbsp; AI-Based Tax Fraud Risk Prediction System &nbsp;·&nbsp;
        Built with Streamlit & scikit-learn &nbsp;·&nbsp;
        For educational and demonstration purposes only.
    </div>
    """, unsafe_allow_html=True)