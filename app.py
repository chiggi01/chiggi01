"""
AI-Based Tax Fraud Risk Prediction System
==========================================
Streamlit frontend integrating pre-trained Random Forest + Isolation Forest models.
Color scheme: Official Income Tax India portal (incometax.gov.in)
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
    page_title="Income Tax Fraud Risk System | CBDT",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Official Income Tax India color palette:
#    Primary Navy  : #003366   (header/sidebar dark blue)
#    Saffron/Orange: #FF6600   (IT Dept accent / tricolor saffron)
#    India Green   : #138808   (success / low-risk)
#    Alert Red     : #CC0000   (high-risk / danger)
#    Gold accent   : #F5A623   (medium / caution)
#    Light BG      : #F4F7FA
#    Card White    : #FFFFFF
#    Border grey   : #D0D8E4
#    Text dark     : #1A2A4A
#    Text muted    : #5A6A7A

st.markdown("""
<style>
/* ── Google Font: Noto Sans for body, Tiro Devanagari for headings (govt feel) */
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans:wght@300;400;500;600;700&family=Playfair+Display:wght@600;700&family=Source+Sans+3:wght@300;400;600;700&display=swap');

/* ── Root CSS variables */
:root {
    --navy:      #003366;
    --navy-dark: #00224d;
    --navy-mid:  #004080;
    --saffron:   #FF6600;
    --saffron-lt:#FF8833;
    --green:     #138808;
    --red:       #CC0000;
    --gold:      #E8860A;
    --gold-lt:   #F5A623;
    --bg:        #EEF2F7;
    --card:      #FFFFFF;
    --border:    #C8D4E4;
    --text:      #1A2A4A;
    --muted:     #5A6A7A;
    --tricolor-gradient: linear-gradient(90deg, #FF6600 33%, #E8EFF8 33%, #E8EFF8 66%, #138808 66%);
}

/* ── Global */
* { font-family: 'Source Sans 3', 'Noto Sans', sans-serif !important; }
[data-testid="stAppViewContainer"] {
    background-color: var(--bg);
}
[data-testid="stMainBlockContainer"] { padding-top: 0.5rem; }

/* ── Sidebar — dark navy, never overridden */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--navy-dark) 0%, var(--navy) 60%, #004d99 100%) !important;
    border-right: 3px solid var(--saffron) !important;
}
[data-testid="stSidebar"] > div:first-child {
    background: transparent !important;
}
/* ALL text inside sidebar → white */
[data-testid="stSidebar"] *,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] b,
[data-testid="stSidebar"] strong { color: #FFFFFF !important; }
/* Prevent any block/vertical containers inside sidebar from going light */
[data-testid="stSidebar"] [data-testid="stVerticalBlock"],
[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
    background: transparent !important;
}

/* ── Metric cards */
[data-testid="metric-container"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-top: 3px solid var(--saffron);
    border-radius: 6px;
    padding: 16px 20px;
    box-shadow: 0 2px 8px rgba(0,51,102,0.08);
    transition: box-shadow 0.2s;
}
[data-testid="metric-container"]:hover {
    box-shadow: 0 4px 16px rgba(0,51,102,0.14);
}
[data-testid="stMetricValue"] { color: var(--navy) !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 12px !important; text-transform: uppercase; letter-spacing: 0.5px; }

/* ── Headings */
h1 { font-family: 'Playfair Display', serif !important; color: var(--navy) !important; font-weight: 700 !important; }
/* h1 inside the dark banner → always white */
.top-banner h1,
.top-banner h1 * { color: #FFFFFF !important; }
h2 { font-family: 'Source Sans 3', sans-serif !important; color: var(--navy) !important; font-weight: 700 !important;
     border-bottom: 2px solid var(--saffron); padding-bottom: 6px; margin-bottom: 18px; letter-spacing: 0.3px; }
h3 { font-family: 'Source Sans 3', sans-serif !important; color: var(--navy) !important; font-weight: 600 !important; }

/* ── Risk badges */
.risk-high   { background: var(--red);    color: #fff; padding: 3px 12px; border-radius: 4px; font-weight: 700; font-size: 13px; letter-spacing: 0.3px; }
.risk-medium { background: var(--gold);   color: #fff; padding: 3px 12px; border-radius: 4px; font-weight: 700; font-size: 13px; }
.risk-low    { background: var(--green);  color: #fff; padding: 3px 12px; border-radius: 4px; font-weight: 700; font-size: 13px; }

/* ── Top banner */
.top-banner {
    background: linear-gradient(135deg, var(--navy-dark) 0%, var(--navy) 55%, var(--navy-mid) 100%);
    color: white;
    padding: 22px 32px;
    border-radius: 8px;
    margin-bottom: 22px;
    border-left: 6px solid var(--saffron);
    box-shadow: 0 4px 20px rgba(0,34,77,0.25);
    position: relative;
    overflow: hidden;
}
.top-banner::before {
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 200px; height: 100%;
    background: linear-gradient(135deg, transparent 50%, rgba(255,102,0,0.08) 100%);
}
.top-banner::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 4px;
    background: var(--tricolor-gradient);
}

/* ── Tricolor bar */
.tricolor-bar {
    height: 5px;
    background: var(--tricolor-gradient);
    border-radius: 2px;
    margin: 12px 0;
}

/* ── Info card */
.info-card {
    background: var(--card);
    border-left: 4px solid var(--saffron);
    border-radius: 6px;
    padding: 14px 18px;
    margin: 8px 0;
    box-shadow: 0 1px 6px rgba(0,51,102,0.07);
    font-size: 14px;
    color: var(--text);
    line-height: 1.6;
}
.info-card b { color: var(--navy); }

/* ── Alert card */
.alert-card {
    background: #FFF5F5;
    border: 1px solid #FFAAAA;
    border-left: 4px solid var(--red);
    border-radius: 6px;
    padding: 14px 18px;
    margin: 6px 0;
    font-size: 14px;
    color: var(--text);
}

/* ── CBDT header strip */
.cbdt-strip {
    background: var(--navy-dark);
    color: #C8D8F0 !important;
    font-size: 11px;
    padding: 5px 20px;
    border-radius: 4px 4px 0 0;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    font-weight: 500;
}

/* ── Section heading pill */
.section-tag {
    display: inline-block;
    background: var(--saffron);
    color: white;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding: 2px 10px;
    border-radius: 2px;
    margin-bottom: 6px;
}

/* ── Sidebar nav buttons */
[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    background: rgba(255,255,255,0.06) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 5px !important;
    padding: 10px 14px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    margin-bottom: 4px !important;
    text-align: left !important;
    transition: all 0.2s !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: var(--saffron) !important;
    color: #FFFFFF !important;
    border-color: var(--saffron) !important;
}
[data-testid="stSidebar"] .stButton > button p,
[data-testid="stSidebar"] .stButton > button span,
[data-testid="stSidebar"] .stButton > button div {
    color: #FFFFFF !important;
    font-weight: 600 !important;
}

/* ── Dataframe */
[data-testid="stDataFrame"] { border-radius: 6px; overflow: hidden; }

/* ── Download button */
[data-testid="stDownloadButton"] > button {
    background: var(--navy) !important;
    color: white !important;
    border: none !important;
    border-radius: 5px !important;
    padding: 9px 20px !important;
    font-weight: 600 !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: var(--saffron) !important;
}

/* ── Horizontal rules */
hr { border-color: var(--border) !important; }

/* ── Tabs */
.stTabs [data-baseweb="tab"] {
    font-weight: 600;
    color: var(--muted);
}
.stTabs [aria-selected="true"] {
    color: var(--navy) !important;
    border-bottom-color: var(--saffron) !important;
}

/* ── Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #e0e8f4; }
::-webkit-scrollbar-thumb { background: var(--navy); border-radius: 3px; }

/* ════════════════════════════════════════════════════
   FIX: White-on-white — Streamlit native widget overrides
   Scoped strictly to stMain so sidebar is NEVER affected
   ════════════════════════════════════════════════════ */

/* ── Main content area background only */
[data-testid="stMain"] { background-color: var(--bg) !important; }
[data-testid="stMain"] > div { background-color: var(--bg) !important; }
.main .block-container { background-color: var(--bg) !important; }

/* ── General text in main area — scoped tightly */
/* Plain markdown paragraphs only — inline style="" colors take precedence */
[data-testid="stMain"] [data-testid="stMarkdownContainer"] > p:not([style]) { color: var(--text) !important; }
[data-testid="stMain"] li { color: var(--text) !important; }
[data-testid="stMain"] td { color: var(--text) !important; }
[data-testid="stMain"] th { color: var(--text) !important; }

/* ── Selectbox: container, label, selected value */
[data-testid="stSelectbox"] > label { color: var(--navy) !important; font-weight: 600 !important; font-size: 13px !important; }
[data-testid="stSelectbox"] [data-baseweb="select"] { background-color: #ffffff !important; border: 1.5px solid var(--border) !important; border-radius: 5px !important; }
[data-testid="stSelectbox"] [data-baseweb="select"] * { color: var(--text) !important; background-color: transparent !important; }
[data-testid="stSelectbox"] [data-baseweb="select"]:focus-within { border-color: var(--saffron) !important; box-shadow: 0 0 0 2px rgba(255,102,0,0.15) !important; }

/* ── Selectbox dropdown menu (popover) */
[data-baseweb="popover"] { background: #ffffff !important; border: 1px solid var(--border) !important; box-shadow: 0 4px 16px rgba(0,51,102,0.12) !important; }
[data-baseweb="popover"] * { color: var(--text) !important; background: transparent !important; }
[data-baseweb="menu"] li:hover { background: #EEF4FF !important; color: var(--navy) !important; }
[data-baseweb="option"] { color: var(--text) !important; }
[aria-selected="true"][data-baseweb="option"] { background: #E8F0FF !important; color: var(--navy) !important; }

/* ── Text input */
[data-testid="stTextInput"] > label { color: var(--navy) !important; font-weight: 600 !important; font-size: 13px !important; }
[data-testid="stTextInput"] input {
    background: #ffffff !important;
    color: var(--text) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 5px !important;
    padding: 8px 12px !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: var(--saffron) !important;
    box-shadow: 0 0 0 2px rgba(255,102,0,0.15) !important;
}
[data-testid="stTextInput"] input::placeholder { color: #9AAABB !important; }

/* ── Number input */
[data-testid="stNumberInput"] > label { color: var(--navy) !important; font-weight: 600 !important; }
[data-testid="stNumberInput"] input { background: #ffffff !important; color: var(--text) !important; border: 1.5px solid var(--border) !important; border-radius: 5px !important; }

/* ── Slider */
[data-testid="stSlider"] > label { color: var(--navy) !important; font-weight: 600 !important; }
[data-testid="stSlider"] [data-testid="stMarkdownContainer"] p { color: var(--text) !important; }
[data-baseweb="slider"] [role="slider"] { background: var(--saffron) !important; border-color: var(--saffron) !important; }
[data-baseweb="slider"] [data-testid="stSlider"] div[class*="thumb"] { background: var(--saffron) !important; }

/* ── Checkbox */
[data-testid="stCheckbox"] label { color: var(--text) !important; }
[data-testid="stCheckbox"] input:checked + span { background: var(--saffron) !important; border-color: var(--saffron) !important; }

/* ── Radio buttons */
[data-testid="stRadio"] > label { color: var(--navy) !important; font-weight: 600 !important; }
[data-testid="stRadio"] label { color: var(--text) !important; }

/* ── Multiselect */
[data-testid="stMultiSelect"] > label { color: var(--navy) !important; font-weight: 600 !important; }
[data-testid="stMultiSelect"] [data-baseweb="select"] { background: #ffffff !important; border: 1.5px solid var(--border) !important; border-radius: 5px !important; }
[data-testid="stMultiSelect"] [data-baseweb="tag"] { background: var(--navy) !important; color: white !important; }
[data-testid="stMultiSelect"] [data-baseweb="tag"] span { color: white !important; }

/* ── st.info / st.success / st.warning / st.error boxes */
[data-testid="stAlert"] { border-radius: 6px !important; }
[data-testid="stAlert"][data-baseweb="notification"] { border-radius: 6px !important; }
.stAlert { background: #EDF3FF !important; border-left: 4px solid var(--navy) !important; border-radius: 6px !important; }
.stAlert p, .stAlert div, .stAlert span { color: var(--navy) !important; }
/* Info */
div[data-testid="stAlert"] > div[role="alert"] { background: #EDF3FF !important; }
div[data-testid="stAlert"] svg { color: var(--navy) !important; }

/* ── st.info specifically */
[data-testid="stInfo"] { background: #EDF3FF !important; border-left: 4px solid #2563EB !important; }
[data-testid="stInfo"] * { color: #1A2A4A !important; }

/* ── st.success */
[data-testid="stSuccess"] { background: #EDFAEE !important; border-left: 4px solid var(--green) !important; }
[data-testid="stSuccess"] * { color: #0D5C0D !important; }

/* ── st.warning */
[data-testid="stWarning"] { background: #FFF8E6 !important; border-left: 4px solid var(--gold) !important; }
[data-testid="stWarning"] * { color: #7A4A00 !important; }

/* ── st.error */
[data-testid="stError"] { background: #FFF0F0 !important; border-left: 4px solid var(--red) !important; }
[data-testid="stError"] * { color: #800000 !important; }

/* ── Expander */
[data-testid="stExpander"] { background: #ffffff !important; border: 1px solid var(--border) !important; border-radius: 6px !important; }
[data-testid="stExpander"] summary { color: var(--navy) !important; font-weight: 600 !important; background: #F5F8FF !important; border-radius: 6px !important; }
[data-testid="stExpander"] summary:hover { background: #EBF0FF !important; }
[data-testid="stExpander"] summary * { color: var(--navy) !important; }
[data-testid="stExpander"] > div { background: #ffffff !important; color: var(--text) !important; }
[data-testid="stExpander"] > div * { color: var(--text) !important; }

/* ── Dataframe / table */
[data-testid="stDataFrame"] { background: #ffffff !important; border: 1px solid var(--border) !important; border-radius: 6px !important; }
[data-testid="stDataFrame"] * { color: var(--text) !important; }
[data-testid="stDataFrame"] th { background: var(--navy) !important; color: white !important; font-weight: 600 !important; }
[data-testid="stDataFrame"] th * { color: white !important; }
[data-testid="stDataFrame"] tr:nth-child(even) td { background: #F5F8FF !important; }
[data-testid="stDataFrame"] tr:hover td { background: #EBF0FA !important; }

/* ── Caption / small text */
[data-testid="stCaptionContainer"] p { color: var(--muted) !important; font-size: 12px !important; }

/* ── Metric delta text */
[data-testid="stMetricDelta"] { font-size: 12px !important; }
[data-testid="stMetricDelta"] > div { color: var(--muted) !important; }

/* ── Tabs (main content) */
[data-baseweb="tab-list"] { background: transparent !important; border-bottom: 2px solid var(--border) !important; }
[data-baseweb="tab"] { color: var(--muted) !important; font-weight: 600 !important; background: transparent !important; }
[data-baseweb="tab"][aria-selected="true"] { color: var(--navy) !important; border-bottom: 3px solid var(--saffron) !important; }
[data-baseweb="tab-panel"] { background: transparent !important; }

/* ── Spinner / loading text */
[data-testid="stSpinner"] p { color: var(--navy) !important; }

/* ── Column containers — prevent white box flash */
[data-testid="column"] { background: transparent !important; }

/* ── Markdown rendered inside columns or blocks */
/* Markdown text — scoped to main content only, NOT sidebar */
[data-testid="stMain"] [data-testid="stMarkdownContainer"] > p { color: var(--text) !important; }
[data-testid="stMain"] [data-testid="stMarkdownContainer"] > ul li { color: var(--text) !important; }

/* ── Subheader (st.subheader renders as h3) */
[data-testid="stHeadingWithActionElements"] h3 { color: var(--navy) !important; font-weight: 700 !important; }

/* ── Form submit button */
[data-testid="stFormSubmitButton"] > button { background: var(--navy) !important; color: white !important; border: none !important; border-radius: 5px !important; font-weight: 600 !important; }
[data-testid="stFormSubmitButton"] > button:hover { background: var(--saffron) !important; }

/* ── Tooltip */
[data-baseweb="tooltip"] { background: var(--navy) !important; color: white !important; border-radius: 4px !important; }

/* ── Code blocks */
code { background: #E8EFF8 !important; color: var(--navy) !important; border-radius: 3px !important; padding: 1px 5px !important; }
pre { background: #E8EFF8 !important; color: var(--navy) !important; border-radius: 6px !important; }

/* ── Tables rendered in markdown */
table { border-collapse: collapse !important; width: 100% !important; }
table th { background: var(--navy) !important; color: white !important; padding: 8px 12px !important; }
table td { color: var(--text) !important; padding: 7px 12px !important; border-bottom: 1px solid var(--border) !important; }
table tr:nth-child(even) td { background: #F5F8FF !important; }
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
#  COLOUR CONSTANTS (for Plotly / matplotlib)
# ══════════════════════════════════════════════════════
C_NAVY    = '#003366'
C_SAFFRON = '#FF6600'
C_GREEN   = '#138808'
C_RED     = '#CC0000'
C_GOLD    = '#E8860A'
C_MID     = '#004080'
C_MUTED   = '#5A6A7A'

RISK_COLORS = {'High': C_RED, 'Medium': C_GOLD, 'Low': C_GREEN}

# Plotly layout defaults
PLOTLY_LAYOUT = dict(
    font=dict(family="Source Sans 3, Noto Sans, sans-serif", color='#1A2A4A', size=13),
    plot_bgcolor='#FFFFFF',
    paper_bgcolor='#FFFFFF',
    margin=dict(t=20, b=40, l=10, r=10),
    xaxis=dict(
        color='#1A2A4A',
        tickfont=dict(color='#1A2A4A', size=12),
        title_font=dict(color='#1A2A4A', size=13),
        gridcolor='#E0E8F4',
        linecolor='#C8D4E4',
        zerolinecolor='#C8D4E4',
    ),
    yaxis=dict(
        color='#1A2A4A',
        tickfont=dict(color='#1A2A4A', size=12),
        title_font=dict(color='#1A2A4A', size=13),
        gridcolor='#E0E8F4',
        linecolor='#C8D4E4',
        zerolinecolor='#C8D4E4',
    ),
    legend=dict(
        font=dict(color='#1A2A4A', size=12),
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor='#C8D4E4',
        borderwidth=1,
    ),
)


# ══════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════
def risk_badge(level):
    level = str(level)
    cls = {'High': 'risk-high', 'Medium': 'risk-medium', 'Low': 'risk-low'}.get(level, 'risk-low')
    return f'<span class="{cls}">▲ {level} Risk</span>'

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
                    f"cross-check amendment history and late payment penalties under §234F.")
    if row['Previous_Penalty'] == 1:
        tips.append("⚖️ <b>Penalty Record:</b> Taxpayer has a prior penalty on record — "
                    "retrieve historical assessment orders under §144/§143(3).")
    if row.get('Anomaly_Flag', 0) == 1:
        tips.append("🔎 <b>Statistical Anomaly:</b> Financial figures deviate significantly from "
                    "peer group — conduct third-party income verification via Form 26AS / AIS.")
    if not tips:
        tips.append("✅ <b>No Critical Red Flags:</b> This taxpayer's profile is within normal ranges. "
                    "Standard periodic scrutiny under §143(1) recommended.")
    return tips


# ══════════════════════════════════════════════════════
#  SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════
with st.sidebar:
    # Govt emblem-style header
    st.markdown("""
    <div style='text-align:center; padding:16px 0 10px 0;'>
        <div style='font-size:46px; margin-bottom:6px;'>🇮🇳</div>
        <div style='font-size:11px; letter-spacing:2px; color:#90A8CC; text-transform:uppercase; font-weight:600;'>
            Government of India
        </div>
        <div style='font-size:10px; letter-spacing:1px; color:#7090B8; margin-top:2px;'>
            Ministry of Finance · CBDT
        </div>
    </div>
    <div style='background:linear-gradient(90deg,#FF6600,#ffffff22,#138808); height:3px; border-radius:2px; margin:10px 0 14px 0;'></div>
    <div style='text-align:center; padding:0 0 16px 0;'>
        <div style='font-size:15px; font-weight:700; color:#E0ECF8; letter-spacing:0.5px; line-height:1.4;'>
            Tax Fraud Risk<br>Prediction System
        </div>
        <div style='font-size:10px; color:#7090B8; margin-top:5px; letter-spacing:0.5px;'>
            AI/ML Intelligence Platform · FY 2025–26
        </div>
    </div>
    <hr style='border-color:rgba(255,255,255,0.1); margin:0 0 14px 0;'>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:10px; color:#8090A8; letter-spacing:1.5px; text-transform:uppercase; margin-bottom:8px;'>Navigation</div>", unsafe_allow_html=True)

    pages = {
        "🏠  Home"            : "Home",
        "📊  Dashboard"       : "Dashboard",
        "🔍  Taxpayer Search" : "Taxpayer Search",
        "⚠️  Risk Analysis"   : "Risk Analysis",
        "ℹ️  About"           : "About",
    }
    if "page" not in st.session_state:
        st.session_state.page = "Home"

    for label, key in pages.items():
        if st.button(label, key=f"nav_{key}"):
            st.session_state.page = key

    st.markdown("<hr style='border-color:rgba(255,255,255,0.1); margin:16px 0 12px 0;'>", unsafe_allow_html=True)

    if MODEL_LOADED:
        total      = len(df)
        high_count = int((df['Risk_Level'] == 'High').sum())
        anom_count = int(df['Anomaly_Flag'].sum())
        fraud_count = int(df['Fraud_Label'].sum())
        st.markdown(f"""
        <div style='font-size:12px; color:#FFFFFF; line-height:2.2;'>
            <div style='font-size:10px; color:#FFFFFF; letter-spacing:1.5px; text-transform:uppercase; margin-bottom:6px; font-weight:700; border-bottom:1px solid rgba(255,255,255,0.2); padding-bottom:4px;'>Live Statistics</div>
            <div style='display:flex;justify-content:space-between;'><span style='color:#FFFFFF;'>Total Records</span><b style='color:#FFFFFF;'>{total:,}</b></div>
            <div style='display:flex;justify-content:space-between;'><span style='color:#FFFFFF;'>High Risk Cases</span><b style='color:#FF8866;'>{high_count:,}</b></div>
            <div style='display:flex;justify-content:space-between;'><span style='color:#FFFFFF;'>Fraud Flagged</span><b style='color:#FFD080;'>{fraud_count:,}</b></div>
            <div style='display:flex;justify-content:space-between;'><span style='color:#FFFFFF;'>Anomalies</span><b style='color:#FFD080;'>{anom_count:,}</b></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-top:auto; padding-top:20px; font-size:10px; color:#506070; text-align:center; border-top:1px solid rgba(255,255,255,0.06); margin-top:30px; padding-top:10px;'>
        For internal use only · CBDT Confidential<br>
        Powered by scikit-learn · Streamlit
    </div>
    """, unsafe_allow_html=True)

page = st.session_state.page


# ══════════════════════════════════════════════════════
#  GUARD
# ══════════════════════════════════════════════════════
if not MODEL_LOADED:
    st.error(f"❌ Could not load model files: `{LOAD_ERROR}`")
    st.info("Please ensure `fraud_model.pkl`, `preprocessor.pkl`, and `taxpayer_data.csv` are present.")
    st.stop()


# ══════════════════════════════════════════════════════
#  PAGE: HOME
# ══════════════════════════════════════════════════════
if page == "Home":

    # CBDT top strip
    st.markdown("""
    <div class='cbdt-strip'>
        Central Board of Direct Taxes (CBDT) · Income Tax Department · Government of India
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='top-banner'>
        <div style='font-size:12px; letter-spacing:2px; color:#FF9955; text-transform:uppercase; margin-bottom:8px; font-weight:600;'>
            AI / ML Intelligence Platform
        </div>
        <h1 style='color:white; margin:0; font-size:26px; font-family:Playfair Display,serif; font-weight:700;'>
            Tax Fraud Risk Prediction System
        </h1>
        <p style='margin:8px 0 0 0; color:#FFFFFF; font-size:14px; max-width:620px; line-height:1.6;'>
            An intelligent platform for automated audit prioritisation using
            Random Forest Classification and Isolation Forest Anomaly Detection.
            Built for Income Tax officers under CBDT.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # KPI row
    total  = len(df)
    high_c = int((df['Risk_Level'] == 'High').sum())
    med_c  = int((df['Risk_Level'] == 'Medium').sum())
    low_c  = int((df['Risk_Level'] == 'Low').sum())
    anom_c = int(df['Anomaly_Flag'].sum())

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Taxpayers",  f"{total:,}",          delta="FY 2025–26")
    col2.metric("🔴 High Risk",     f"{high_c:,}",         delta=f"{high_c/total*100:.1f}%", delta_color="inverse")
    col3.metric("🟡 Medium Risk",   f"{med_c:,}",          delta=f"{med_c/total*100:.1f}%",  delta_color="off")
    col4.metric("🟢 Low Risk",      f"{low_c:,}",          delta=f"{low_c/total*100:.1f}%",  delta_color="normal")
    col5.metric("🔎 Anomalies",     f"{anom_c:,}",         delta="Isolation Forest",          delta_color="inverse")

    st.markdown("<div class='tricolor-bar'></div>", unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.subheader("System Overview")
        steps = [
            ("01", "Data Ingestion & Normalisation",
             "Taxpayer financial data is ingested, standardised, and validated. Categorical attributes (Profession, City) are ordinally encoded for numerical processing by tree-based ML algorithms."),
            ("02", "Fraud Risk Classification (Random Forest)",
             "An ensemble of 200 decision trees trained on historical fraud patterns assigns each taxpayer a fraud probability score from 0–100%. Uses <em>class_weight='balanced'</em> to handle class imbalance."),
            ("03", "Anomaly Detection (Isolation Forest)",
             "An unsupervised Isolation Forest model independently flags statistical outliers in income, expenses, and investment claims — without requiring labelled fraud data."),
            ("04", "Risk Triage & Audit Prioritisation",
             "Taxpayers are automatically ranked into High / Medium / Low risk tiers, enabling Income Tax officers to prioritise investigations and maximise detection yield per audit manhour."),
        ]
        for num, title, desc in steps:
            st.markdown(f"""
            <div class='info-card' style='display:flex; gap:16px; align-items:flex-start;'>
                <div style='background:var(--navy); color:#FF9944; font-weight:800; font-size:18px;
                            width:38px; height:38px; border-radius:50%; display:flex; align-items:center;
                            justify-content:center; flex-shrink:0; font-family:Playfair Display,serif;'>{num}</div>
                <div>
                    <div style='font-weight:700; color:var(--navy); font-size:14px; margin-bottom:3px;'>{title}</div>
                    <div style='color:var(--muted); font-size:13px; line-height:1.6;'>{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_r:
        st.subheader("Risk Level Guide")
        for level, color, rng, action in [
            ("High Risk",   C_RED,    "> 65%",   "Immediate scrutiny / audit required"),
            ("Medium Risk", C_GOLD,   "35 – 65%","Scheduled review within quarter"),
            ("Low Risk",    C_GREEN,  "< 35%",   "Standard periodic monitoring"),
        ]:
            st.markdown(f"""
            <div style='background:white; border-left:4px solid {color}; border-radius:6px;
                        padding:12px 16px; margin:6px 0; box-shadow:0 1px 5px rgba(0,51,102,0.07);
                        display:flex; justify-content:space-between; align-items:center;'>
                <div>
                    <div style='font-weight:700; color:{color}; font-size:14px;'>{level}</div>
                    <div style='font-size:12px; color:var(--muted);'>{action}</div>
                </div>
                <div style='background:{color}; color:white; border-radius:4px;
                            padding:3px 12px; font-size:12px; font-weight:700;'>{rng}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
        st.subheader("Fraud Trigger Conditions")
        triggers = [
            ("§37(1)", "Expenses > 80% of annual income"),
            ("§80C",   "Investment claims > 30% of income"),
            ("§234F",  "Late filings > 3 times"),
            ("§271",   "Previous penalty on record"),
        ]
        for section, desc in triggers:
            st.markdown(f"""
            <div style='background:white; border-radius:5px; padding:9px 14px; margin:5px 0;
                        box-shadow:0 1px 4px rgba(0,51,102,0.06); display:flex; gap:10px; align-items:center;'>
                <span style='background:#FF6600; color:white; font-size:9px; font-weight:700;
                             padding:2px 7px; border-radius:2px; white-space:nowrap;'>{section}</span>
                <span style='font-size:13px; color:var(--text);'>{desc}</span>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  PAGE: DASHBOARD
# ══════════════════════════════════════════════════════
elif page == "Dashboard":
    st.markdown("""
    <div class='cbdt-strip'>
        Central Board of Direct Taxes · Executive Intelligence Dashboard
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<h2>📊 Executive Dashboard</h2>", unsafe_allow_html=True)

    k1, k2, k3, k4, k5 = st.columns(5)
    total   = len(df)
    fraud_r = df['Fraud_Label'].mean()
    avg_inc = df['Annual_Income'].mean()
    anom_r  = df['Anomaly_Flag'].mean()

    k1.metric("Total Taxpayers",  f"{total:,}")
    k2.metric("Overall Fraud Rate", f"{fraud_r:.1%}")
    k3.metric("Average Income",     f"₹{avg_inc/1e5:.1f}L")
    k4.metric("Anomaly Rate",       f"{anom_r:.1%}")
    k5.metric("High Risk Cases",    f"{(df['Risk_Level']=='High').sum():,}")

    st.markdown("<div class='tricolor-bar'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Risk Distribution")
        risk_counts = df['Risk_Level'].value_counts()
        if USE_PLOTLY:
            fig = px.pie(
                values=risk_counts.values, names=risk_counts.index,
                color=risk_counts.index, color_discrete_map=RISK_COLORS, hole=0.45
            )
            fig.update_traces(
                textfont=dict(family="Source Sans 3", size=13, color='#1A2A4A'),
                marker=dict(line=dict(color='white', width=2))
            )
            fig.update_layout(height=300,
                plot_bgcolor='#FFFFFF', paper_bgcolor='#FFFFFF',
                font=dict(color='#1A2A4A', size=13),
                legend=dict(orientation='h', y=-0.1, font=dict(color='#1A2A4A', size=12)),
                margin=dict(t=20, b=40, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            clrs = [RISK_COLORS.get(str(k), '#999') for k in risk_counts.index]
            ax.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%',
                   colors=clrs, startangle=140, wedgeprops=dict(linewidth=2, edgecolor='white'))
            ax.set_title('Risk Distribution', fontweight='bold', color=C_NAVY)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with c2:
        st.subheader("Fraud Rate by Profession")
        fraud_prof = df.groupby('Profession')['Fraud_Label'].mean().sort_values(ascending=False)
        if USE_PLOTLY:
            fig = px.bar(
                x=fraud_prof.values, y=fraud_prof.index, orientation='h',
                color=fraud_prof.values,
                color_continuous_scale=[[0, C_GREEN], [0.5, C_GOLD], [1, C_RED]],
                text=[f'{v:.1%}' for v in fraud_prof.values]
            )
            fig.update_layout(height=300, coloraxis_showscale=False,
                plot_bgcolor='#FFFFFF', paper_bgcolor='#FFFFFF',
                font=dict(color='#1A2A4A', size=13),
                xaxis=dict(tickfont=dict(color='#1A2A4A', size=12), title_font=dict(color='#1A2A4A'), gridcolor='#E0E8F4', linecolor='#C8D4E4'),
                yaxis=dict(categoryorder='total ascending', tickfont=dict(color='#1A2A4A', size=12), title_font=dict(color='#1A2A4A'), linecolor='#C8D4E4'),
                margin=dict(t=20, b=40, l=10, r=10))
            fig.update_traces(textposition='outside', marker_line_width=0,
                              textfont=dict(color='#1A2A4A', size=12))
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.barh(fraud_prof.index, fraud_prof.values, color=C_SAFFRON, alpha=0.9)
            ax.set_xlabel('Fraud Rate'); ax.set_title('Fraud Rate by Profession', fontweight='bold', color=C_NAVY)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    c3, c4 = st.columns(2)

    with c3:
        st.subheader("Income vs Declared Expenses")
        sample = df.sample(min(800, len(df)), random_state=42)
        if USE_PLOTLY:
            fig = px.scatter(
                sample, x='Annual_Income', y='Declared_Expenses', color='Risk_Level',
                color_discrete_map=RISK_COLORS, opacity=0.50,
                labels={'Annual_Income': 'Annual Income (₹)', 'Declared_Expenses': 'Declared Expenses (₹)'}
            )
            fig.update_layout(height=300,
                plot_bgcolor='#FFFFFF', paper_bgcolor='#FFFFFF',
                font=dict(color='#1A2A4A', size=13),
                xaxis=dict(tickfont=dict(color='#1A2A4A', size=11), title_font=dict(color='#1A2A4A', size=13), gridcolor='#E0E8F4', linecolor='#C8D4E4'),
                yaxis=dict(tickfont=dict(color='#1A2A4A', size=11), title_font=dict(color='#1A2A4A', size=13), gridcolor='#E0E8F4', linecolor='#C8D4E4'),
                legend=dict(font=dict(color='#1A2A4A', size=12), bgcolor='rgba(255,255,255,0.9)', bordercolor='#C8D4E4', borderwidth=1),
                margin=dict(t=20, b=40, l=10, r=10))
            fig.update_traces(marker=dict(size=5, line=dict(width=0)))
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            for level, clr in RISK_COLORS.items():
                s = sample[sample['Risk_Level'] == level]
                ax.scatter(s['Annual_Income']/1e5, s['Declared_Expenses']/1e5,
                           c=clr, label=level, alpha=0.4, s=10)
            ax.set_xlabel('Income (₹L)'); ax.set_ylabel('Expenses (₹L)')
            ax.legend(); ax.set_title('Income vs Expenses', fontweight='bold', color=C_NAVY)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with c4:
        st.subheader("City-wise High Risk Count")
        city_risk = df[df['Risk_Level'] == 'High'].groupby('City').size().sort_values(ascending=False)
        if USE_PLOTLY:
            fig = px.bar(
                x=city_risk.index, y=city_risk.values,
                color=city_risk.values,
                color_continuous_scale=[[0, C_GOLD], [1, C_RED]],
                text=city_risk.values
            )
            fig.update_layout(height=300, coloraxis_showscale=False,
                plot_bgcolor='#FFFFFF', paper_bgcolor='#FFFFFF',
                font=dict(color='#1A2A4A', size=13),
                xaxis=dict(tickfont=dict(color='#1A2A4A', size=12), title_font=dict(color='#1A2A4A'), gridcolor='#E0E8F4', linecolor='#C8D4E4'),
                yaxis=dict(tickfont=dict(color='#1A2A4A', size=12), title_font=dict(color='#1A2A4A'), gridcolor='#E0E8F4', linecolor='#C8D4E4'),
                margin=dict(t=20, b=40, l=10, r=10))
            fig.update_traces(textposition='outside', marker_line_width=0,
                              textfont=dict(color='#1A2A4A', size=12))
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.bar(city_risk.index, city_risk.values, color=C_RED, alpha=0.85)
            ax.set_xticklabels(city_risk.index, rotation=45, ha='right')
            ax.set_ylabel('Count'); ax.set_title('High Risk by City', fontweight='bold', color=C_NAVY)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    st.subheader("Fraud Probability Distribution")
    if USE_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df[df['Fraud_Label'] == 0]['Fraud_Probability'],
            name='Legitimate', marker_color=C_GREEN, opacity=0.65, nbinsx=40
        ))
        fig.add_trace(go.Histogram(
            x=df[df['Fraud_Label'] == 1]['Fraud_Probability'],
            name='Fraud', marker_color=C_RED, opacity=0.65, nbinsx=40
        ))
        fig.add_vline(x=0.5, line_dash='dot', line_color='#003366',
                      annotation_text='Threshold 0.50',
                      annotation_font=dict(color='#1A2A4A', size=12))
        fig.update_layout(barmode='overlay', height=260,
                          plot_bgcolor='#FFFFFF', paper_bgcolor='#FFFFFF',
                          font=dict(color='#1A2A4A', size=13),
                          xaxis=dict(title='Fraud Probability', tickfont=dict(color='#1A2A4A', size=12), title_font=dict(color='#1A2A4A', size=13), gridcolor='#E0E8F4', linecolor='#C8D4E4'),
                          yaxis=dict(title='Count', tickfont=dict(color='#1A2A4A', size=12), title_font=dict(color='#1A2A4A', size=13), gridcolor='#E0E8F4', linecolor='#C8D4E4'),
                          legend=dict(orientation='h', y=1.1, font=dict(color='#1A2A4A', size=12)),
                          margin=dict(t=20, b=40, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.hist(df[df['Fraud_Label'] == 0]['Fraud_Probability'], bins=40, alpha=0.65, color=C_GREEN, label='Legitimate')
        ax.hist(df[df['Fraud_Label'] == 1]['Fraud_Probability'], bins=40, alpha=0.65, color=C_RED, label='Fraud')
        ax.axvline(0.5, ls='--', color=C_NAVY, label='Threshold 0.50')
        ax.set_xlabel('Fraud Probability'); ax.legend()
        ax.set_title('Fraud Probability Distribution', fontweight='bold', color=C_NAVY)
        plt.tight_layout(); st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════════════
#  PAGE: TAXPAYER SEARCH
# ══════════════════════════════════════════════════════
elif page == "Taxpayer Search":
    st.markdown("""
    <div class='cbdt-strip'>
        Central Board of Direct Taxes · Taxpayer Records Search
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<h2>🔍 Taxpayer Search</h2>", unsafe_allow_html=True)

    st.markdown("""
    <div class='info-card' style='border-left-color:#003366; background:#F0F5FF;'>
        <b>Search & Filter</b> — Use the fields below to locate taxpayer records.
        Results can be exported as CSV for offline analysis.
    </div>
    """, unsafe_allow_html=True)

    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        search_id = st.text_input("🔎 Taxpayer ID", placeholder="e.g. TXP10042")
    with fc2:
        prof_options = ["All"] + sorted(df['Profession'].unique().tolist())
        filter_prof = st.selectbox("Profession", prof_options)
    with fc3:
        risk_options = ["All", "High", "Medium", "Low"]
        filter_risk = st.selectbox("Risk Level", risk_options)
    with fc4:
        city_options = ["All"] + sorted(df['City'].unique().tolist())
        filter_city = st.selectbox("City", city_options)

    filtered = df.copy()
    if search_id.strip():
        filtered = filtered[filtered['Taxpayer_ID'].str.contains(search_id.strip(), case=False)]
    if filter_prof != "All":
        filtered = filtered[filtered['Profession'] == filter_prof]
    if filter_risk != "All":
        filtered = filtered[filtered['Risk_Level'] == filter_risk]
    if filter_city != "All":
        filtered = filtered[filtered['City'] == filter_city]

    r1, r2, r3 = st.columns(3)
    r1.metric("Records Found",  f"{len(filtered):,}")
    r2.metric("High Risk in Results", f"{(filtered['Risk_Level']=='High').sum():,}")
    r3.metric("Anomalies in Results", f"{filtered['Anomaly_Flag'].sum():,}")

    display_cols = ['Taxpayer_ID', 'Age', 'Profession', 'City', 'Annual_Income',
                    'Declared_Expenses', 'Investment_Claims', 'Late_Filing_Count',
                    'Previous_Penalty', 'Fraud_Label', 'Risk_Level', 'Anomaly_Flag', 'Fraud_Probability']
    display_df = filtered[display_cols].copy()
    display_df['Annual_Income']      = display_df['Annual_Income'].apply(lambda x: f"₹{x:,.0f}")
    display_df['Declared_Expenses']  = display_df['Declared_Expenses'].apply(lambda x: f"₹{x:,.0f}")
    display_df['Investment_Claims']  = display_df['Investment_Claims'].apply(lambda x: f"₹{x:,.0f}")
    display_df['Fraud_Probability']  = display_df['Fraud_Probability'].apply(lambda x: f"{x:.2%}")
    display_df['Anomaly_Flag']       = display_df['Anomaly_Flag'].map({0: 'Normal', 1: '⚠ Anomaly'})
    display_df['Fraud_Label']        = display_df['Fraud_Label'].map({0: 'Legitimate', 1: '🚨 Fraud'})

    st.dataframe(display_df.head(100), use_container_width=True, height=420)

    if len(filtered) > 100:
        st.caption(f"Showing first 100 of {len(filtered):,} results. Refine filters to narrow down.")

    csv = filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️  Download Filtered Results as CSV",
        data=csv, file_name="filtered_taxpayers.csv", mime="text/csv"
    )


# ══════════════════════════════════════════════════════
#  PAGE: RISK ANALYSIS
# ══════════════════════════════════════════════════════
elif page == "Risk Analysis":
    st.markdown("""
    <div class='cbdt-strip'>
        Central Board of Direct Taxes · Individual Taxpayer Risk Profile
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<h2>⚠️ Individual Risk Analysis</h2>", unsafe_allow_html=True)

    col_sel, _ = st.columns([2, 3])
    with col_sel:
        taxpayer_ids = sorted(df['Taxpayer_ID'].tolist())
        selected_id = st.selectbox("Select Taxpayer ID", taxpayer_ids, index=0)

    row = df[df['Taxpayer_ID'] == selected_id].iloc[0]
    risk = str(row['Risk_Level'])
    fraud_prob = float(row['Fraud_Probability'])
    risk_color = RISK_COLORS.get(risk, C_MUTED)
    bar_width  = int(fraud_prob * 100)

    h1, h2, h3 = st.columns(3)
    with h1:
        st.markdown(f"""
        <div class='info-card' style='border-left-color:{C_NAVY};'>
            <div class='section-tag'>Taxpayer Profile</div>
            <div style='font-size:24px; font-weight:800; color:{C_NAVY}; font-family:Playfair Display,serif;'>{selected_id}</div>
            <div style='font-size:14px; color:{C_MUTED}; margin-top:4px;'>{row['Profession']} &nbsp;·&nbsp; {row['City']}</div>
            <div style='font-size:13px; color:{C_MUTED};'>Age: {int(row['Age'])} &nbsp;|&nbsp; Income: {format_inr(row['Annual_Income'])}</div>
        </div>
        """, unsafe_allow_html=True)
    with h2:
        st.markdown(f"""
        <div class='info-card' style='border-left-color:{risk_color};'>
            <div class='section-tag' style='background:{risk_color};'>Risk Score</div>
            <div style='font-size:40px; font-weight:900; color:{risk_color}; line-height:1.1;'>{fraud_prob:.1%}</div>
            <div style='background:#E8EFF8; border-radius:4px; height:8px; margin:8px 0 10px 0;'>
                <div style='width:{bar_width}%; background:{risk_color}; height:8px; border-radius:4px;
                            transition:width 0.8s ease;'></div>
            </div>
            {risk_badge(risk)}
        </div>
        """, unsafe_allow_html=True)
    with h3:
        anom_icon  = "⚠️ Anomaly Flagged" if row['Anomaly_Flag'] == 1 else "✅ Normal Profile"
        anom_color = C_RED if row['Anomaly_Flag'] == 1 else C_GREEN
        fraud_icon = "🚨 Fraud Record" if row['Fraud_Label'] == 1 else "✅ Clean Record"
        st.markdown(f"""
        <div class='info-card' style='border-left-color:{anom_color};'>
            <div class='section-tag' style='background:{anom_color};'>Flags</div>
            <div style='font-size:16px; font-weight:700; color:{anom_color}; margin:6px 0 4px 0;'>{anom_icon}</div>
            <div style='font-size:14px; margin-bottom:6px;'>{fraud_icon}</div>
            <div style='font-size:12px; color:{C_MUTED};'>
                Late Filings: <b>{int(row['Late_Filing_Count'])}</b> &nbsp;|&nbsp;
                Prior Penalty: <b>{"Yes" if row['Previous_Penalty']==1 else "No"}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='tricolor-bar'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Income & Expense Comparison")
        peer_income  = df[df['Profession'] == row['Profession']]['Annual_Income'].mean()
        peer_expense = df[df['Profession'] == row['Profession']]['Declared_Expenses'].mean()
        peer_invest  = df[df['Profession'] == row['Profession']]['Investment_Claims'].mean()
        categories   = ['Annual Income', 'Declared Expenses', 'Investment Claims']
        tp_vals      = [row['Annual_Income'], row['Declared_Expenses'], row['Investment_Claims']]
        peer_vals    = [peer_income, peer_expense, peer_invest]

        if USE_PLOTLY:
            fig = go.Figure(data=[
                go.Bar(name='This Taxpayer', x=categories, y=tp_vals,
                       marker_color=C_SAFFRON, marker_line_width=0, opacity=0.9),
                go.Bar(name=f'Peer Avg ({row["Profession"]})', x=categories, y=peer_vals,
                       marker_color=C_NAVY, marker_line_width=0, opacity=0.7)
            ])
            fig.update_layout(barmode='group', height=300,
                plot_bgcolor='#FFFFFF', paper_bgcolor='#FFFFFF',
                font=dict(color='#1A2A4A', size=13),
                yaxis=dict(tickprefix='₹', tickformat=',', tickfont=dict(color='#1A2A4A', size=12), title_font=dict(color='#1A2A4A'), gridcolor='#E0E8F4', linecolor='#C8D4E4'),
                xaxis=dict(tickfont=dict(color='#1A2A4A', size=12), linecolor='#C8D4E4'),
                legend=dict(orientation='h', y=1.12, font=dict(color='#1A2A4A', size=12)),
                margin=dict(t=20, b=40, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            x = np.arange(len(categories)); w = 0.35
            ax.bar(x-w/2, tp_vals,   w, label='This Taxpayer',  color=C_SAFFRON, alpha=0.9)
            ax.bar(x+w/2, peer_vals, w, label='Peer Avg',        color=C_NAVY,    alpha=0.7)
            ax.set_xticks(x); ax.set_xticklabels(categories, fontsize=8)
            ax.set_ylabel('₹'); ax.legend(fontsize=8)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown(f"""
        <div class='info-card' style='border-left-color:{C_NAVY}; background:#F0F5FF;'>
            <b>Peer Benchmark · {row['Profession']}</b><br>
            <span style='font-size:12px; color:{C_MUTED};'>
                Avg Income: {format_inr(peer_income)} &nbsp;|&nbsp;
                Avg Expenses: {format_inr(peer_expense)} &nbsp;|&nbsp;
                Expense Ratio: {peer_expense/peer_income:.1%}
            </span>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.subheader("Feature Importance (Model View)")
        importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values()
        if USE_PLOTLY:
            bar_colors_fi = [C_SAFFRON if v >= importances.median() else C_NAVY for v in importances.values]
            fig = go.Figure(go.Bar(
                x=importances.values, y=importances.index, orientation='h',
                marker_color=bar_colors_fi, marker_line_width=0,
                text=[f'{v:.3f}' for v in importances.values], textposition='outside',
                textfont=dict(color='#1A2A4A', size=11)
            ))
            fig.update_layout(height=300,
                plot_bgcolor='#FFFFFF', paper_bgcolor='#FFFFFF',
                font=dict(color='#1A2A4A', size=13),
                xaxis=dict(title='Importance Score', tickfont=dict(color='#1A2A4A', size=12), title_font=dict(color='#1A2A4A', size=13), gridcolor='#E0E8F4', linecolor='#C8D4E4'),
                yaxis=dict(tickfont=dict(color='#1A2A4A', size=12), linecolor='#C8D4E4'),
                margin=dict(t=20, b=40, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            bar_colors_fi = [C_SAFFRON if v >= importances.median() else C_NAVY for v in importances.values]
            ax.barh(importances.index, importances.values, color=bar_colors_fi, alpha=0.9)
            ax.set_xlabel('Importance'); ax.set_title('Feature Importance', fontweight='bold', color=C_NAVY)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("<div class='tricolor-bar'></div>", unsafe_allow_html=True)
    st.subheader("Risk Condition Checklist")

    exp_ratio = row['Declared_Expenses'] / row['Annual_Income']
    inv_ratio = row['Investment_Claims']  / row['Annual_Income']

    r1, r2, r3, r4 = st.columns(4)
    for col, label, value, display, threshold, triggered in [
        (r1, "Expense Ratio",    exp_ratio, f"{exp_ratio:.1%}", "80%",  exp_ratio > 0.80),
        (r2, "Investment Ratio", inv_ratio, f"{inv_ratio:.1%}", "30%",  inv_ratio > 0.30),
        (r3, "Late Filings",     None,       str(int(row['Late_Filing_Count'])), "> 3", int(row['Late_Filing_Count']) > 3),
        (r4, "Prior Penalty",    None,       "Yes" if row['Previous_Penalty']==1 else "No", "Any", row['Previous_Penalty']==1),
    ]:
        card_class = "alert-card" if triggered else "info-card"
        icon_color = C_RED if triggered else C_GREEN
        status     = "⚠ TRIGGERED" if triggered else "✅ CLEAR"
        col.markdown(f"""
        <div class='{card_class}' style='text-align:center;'>
            <div style='font-size:11px; color:{C_MUTED}; letter-spacing:1px; text-transform:uppercase; margin-bottom:6px;'>{label}</div>
            <div style='font-size:28px; font-weight:800; color:{icon_color};'>{display}</div>
            <div style='font-size:11px; color:{C_MUTED}; margin:4px 0;'>Threshold: {threshold}</div>
            <div style='font-weight:700; font-size:12px; color:{icon_color};'>{status}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='tricolor-bar'></div>", unsafe_allow_html=True)
    st.subheader("📋 Audit Investigation Suggestions")
    suggestions = get_investigation_suggestions(row)
    for i, tip in enumerate(suggestions, 1):
        st.markdown(f"""
        <div class='info-card' style='display:flex; gap:12px; align-items:flex-start;'>
            <div style='background:{C_SAFFRON}; color:white; width:24px; height:24px; border-radius:50%;
                        display:flex; align-items:center; justify-content:center; font-size:11px;
                        font-weight:800; flex-shrink:0;'>{i}</div>
            <div style='font-size:13px; line-height:1.6; color:{C_MUTED};'>{tip}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='tricolor-bar'></div>", unsafe_allow_html=True)
    st.subheader(f"👥 Similar High-Risk Taxpayers — {row['Profession']}")
    similar = df[
        (df['Profession'] == row['Profession']) &
        (df['Risk_Level'] == 'High') &
        (df['Taxpayer_ID'] != selected_id)
    ][['Taxpayer_ID', 'Age', 'City', 'Annual_Income', 'Fraud_Probability', 'Anomaly_Flag']].head(8)

    if len(similar) > 0:
        similar = similar.copy()
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
    st.markdown("""
    <div class='cbdt-strip'>
        Central Board of Direct Taxes · System Information & Methodology
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<h2>ℹ️ About This System</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown(f"""
        <div class='info-card'>
            <div class='section-tag'>Project Objective</div>
            <p style='margin:10px 0 0 0; font-size:14px; line-height:1.7; color:{C_MUTED};'>
            This system demonstrates the application of <b style='color:{C_NAVY};'>Machine Learning</b> to
            tax fraud risk detection — a critical challenge for revenue authorities under CBDT.
            By combining supervised classification with unsupervised anomaly detection,
            it enables intelligent prioritisation of audit cases, reducing manual workload
            while increasing detection accuracy and coverage.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class='info-card'>
            <div class='section-tag'>ML Techniques</div>
            <p style='margin:10px 0 6px 0; font-size:14px; color:{C_NAVY}; font-weight:700;'>1. Random Forest Classifier</p>
            <p style='margin:0 0 10px 0; font-size:13px; line-height:1.6; color:{C_MUTED};'>
            An ensemble of 200 decision trees trained on historical taxpayer patterns.
            Uses <em>class_weight='balanced'</em> to handle the natural class imbalance
            between legitimate and fraudulent cases. Outputs calibrated fraud probability scores (0–100%).
            </p>
            <p style='margin:0 0 6px 0; font-size:14px; color:{C_NAVY}; font-weight:700;'>2. Isolation Forest</p>
            <p style='margin:0 0 10px 0; font-size:13px; line-height:1.6; color:{C_MUTED};'>
            An unsupervised anomaly detection algorithm that isolates outliers by randomly
            partitioning the feature space. Applied to income, expenses, and investment claims
            to flag statistical outliers independently of labels.
            </p>
            <p style='margin:0 0 6px 0; font-size:14px; color:{C_NAVY}; font-weight:700;'>3. Label Encoding</p>
            <p style='margin:0; font-size:13px; line-height:1.6; color:{C_MUTED};'>
            Categorical features (Profession, City) are ordinally encoded to allow
            numerical processing by tree-based algorithms.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class='info-card'>
            <div class='section-tag'>Model Performance</div>
            <table style='width:100%; font-size:13px; margin-top:10px; border-collapse:collapse;'>
            <tr style='background:{C_NAVY}; color:white;'>
                <th style='padding:8px 12px; text-align:left; border-radius:4px 0 0 0;'>Metric</th>
                <th style='padding:8px 12px; text-align:right;'>Value</th>
                <th style='padding:8px 12px; text-align:left; border-radius:0 4px 0 0;'>Note</th>
            </tr>
            {''.join([
                f"<tr style='background:{'#F8FBFF' if i%2==0 else 'white'}; border-bottom:1px solid #E0E8F0;'>"
                f"<td style='padding:7px 12px; font-weight:600; color:{C_NAVY};'>{m}</td>"
                f"<td style='padding:7px 12px; text-align:right; font-weight:700; color:{c};'>{v}</td>"
                f"<td style='padding:7px 12px; color:{C_MUTED}; font-size:12px;'>{n}</td></tr>"
                for i, (m, v, c, n) in enumerate([
                    ("Accuracy",  "87–90%", C_GREEN, "Overall correctness"),
                    ("Precision", "73–80%", C_GOLD,  "Positive predictive value"),
                    ("Recall",    "42–55%", C_SAFFRON,"Conservative — minimises false accusations"),
                    ("F1 Score",  "55–65%", C_MID,   "Harmonic mean"),
                ])
            ])}
            </table>
            <p style='font-size:12px; color:{C_MUTED}; margin-top:10px;'>
            ℹ Recall is intentionally tuned conservatively to minimise wrongful flagging of legitimate taxpayers.
            Threshold can be adjusted based on audit capacity and risk tolerance.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='alert-card'>
            <div class='section-tag' style='background:{C_RED};'>⚖️ Ethical Disclaimer</div>
            <p style='font-size:13px; line-height:1.7; color:{C_MUTED}; margin-top:10px;'>
            This system is built on <b style='color:{C_NAVY};'>synthetic data</b> for
            educational and demonstration purposes only.
            </p>
            <p style='font-size:13px; line-height:1.7; color:{C_MUTED};'>
            In real-world deployments, AI-based fraud detection systems must be:
            </p>
            <ul style='font-size:13px; color:{C_MUTED}; line-height:1.9; padding-left:18px;'>
                <li>Audited for algorithmic bias across demographic groups</li>
                <li>Reviewed by qualified Income Tax officers</li>
                <li>Compliant with the Personal Data Protection Act</li>
                <li>Transparent to flagged individuals (Natural Justice)</li>
                <li>Regularly retrained on fresh assessment data</li>
            </ul>
            <p style='font-size:12px; color:{C_RED}; font-weight:700; border-top:1px solid #FFAAAA; padding-top:8px; margin-top:8px;'>
            A high risk score is an investigative signal — not a verdict.
            No enforcement action under §148 / §131 should be taken solely on ML output.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class='info-card'>
            <div class='section-tag'>Technology Stack</div>
            <table style='width:100%; font-size:13px; margin-top:10px; border-collapse:collapse;'>
            {''.join([
                f"<tr style='border-bottom:1px solid #E8EFF8;'>"
                f"<td style='padding:7px 0; font-weight:600; color:{C_NAVY};'>{icon} {tech}</td>"
                f"<td style='padding:7px 0; color:{C_MUTED};'>{role}</td></tr>"
                for icon, tech, role in [
                    ("🐍","Python 3.9+","Core language"),
                    ("🌲","scikit-learn","ML models"),
                    ("🐼","pandas","Data processing"),
                    ("📦","joblib","Model persistence"),
                    ("📊","Plotly","Visualisations"),
                    ("🖥️","Streamlit","Web frontend"),
                ]
            ])}
            </table>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class='info-card'>
            <div class='section-tag'>File Structure</div>
            <pre style='font-size:12px; background:#F0F5FA; padding:12px; border-radius:4px;
                        color:{C_NAVY}; margin-top:10px; line-height:1.8;'>
project/
├── app.py                  ← Streamlit app
├── Tax_Fraud_Risk_
│   Prediction.ipynb        ← ML notebook
├── fraud_model.pkl         ← RF model
├── preprocessor.pkl        ← Encoders
└── taxpayer_data.csv       ← Dataset
            </pre>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='tricolor-bar'></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='text-align:center; color:{C_MUTED}; font-size:12px; padding:16px;
                background:white; border-radius:6px; border:1px solid #D0DCF0;'>
        <b style='color:{C_NAVY};'>Income Tax Department · Government of India</b><br>
        AI-Based Tax Fraud Risk Prediction System &nbsp;·&nbsp;
        Built with Streamlit &amp; scikit-learn &nbsp;·&nbsp;
        For demonstration purposes only &nbsp;·&nbsp; CBDT Internal Use
    </div>
    """, unsafe_allow_html=True)