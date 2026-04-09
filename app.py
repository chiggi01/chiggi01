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

# ── Custom CSS for professional government-style look
st.markdown("""
<style>
/* ── Global */
[data-testid="stAppViewContainer"] {background-color: #f0f2f6;}
[data-testid="stSidebar"]          {background-color: #1a2340;}
[data-testid="stSidebar"] * {color: #e0e6f0 !important;}

/* ── Metric cards */
[data-testid="metric-container"] {
    background-color: #ffffff;
    border: 1px solid #d1d8e8;
    border-radius: 10px;
    padding: 14px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.07);
}

/* ── Headers */
h1 {color: #1a2340 !important; font-weight: 700;}
h2 {color: #1a2340 !important; border-bottom: 2px solid #c0392b; padding-bottom: 5px;}
h3 {color: #2c3e50 !important;}

/* ── Risk badge */
.risk-high   {background:#c0392b;color:#fff;padding:4px 12px;border-radius:12px;font-weight:600;}
.risk-medium {background:#e67e22;color:#fff;padding:4px 12px;border-radius:12px;font-weight:600;}
.risk-low    {background:#27ae60;color:#fff;padding:4px 12px;border-radius:12px;font-weight:600;}

/* ── Top banner */
.top-banner {
    background: linear-gradient(135deg, #1a2340 0%, #c0392b 100%);
    color: white;
    padding: 20px 28px;
    border-radius: 12px;
    margin-bottom: 20px;
}

/* ── Info card */
.info-card {
    background: #ffffff;
    border-left: 5px solid #c0392b;
    border-radius: 8px;
    padding: 16px;
    margin: 8px 0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
}

/* ── Alert card */
.alert-card {
    background: #fff3f3;
    border: 1px solid #e74c3c;
    border-radius: 8px;
    padding: 14px;
    margin: 6px 0;
}

/* ── Sidebar nav button */
.stButton>button {
    width: 100%;
    background-color: #2c3e6e;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px;
    font-size: 14px;
    font-weight: 500;
    margin-bottom: 4px;
    transition: background 0.2s;
}
.stButton>button:hover {background-color: #c0392b;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  DATA & MODEL LOADING
# ══════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading ML models…")
def load_models():
    model       = joblib.load("fraud_model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    return model, preprocessor

@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    df = pd.read_csv("taxpayer_data.csv")
    # Ensure required derived columns exist
    if 'Risk_Level' not in df.columns:
        df['Risk_Level'] = pd.cut(df['Fraud_Probability'],
                                   bins=[0,0.35,0.65,1.0],
                                   labels=['Low','Medium','High'])
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
    cls = {'High':'risk-high','Medium':'risk-medium','Low':'risk-low'}.get(level, 'risk-low')
    return f'<span class="{cls}">⚠ {level} Risk</span>'

def format_inr(value):
    """Format number as Indian Rupees."""
    return f"₹{value:,.0f}"

def get_investigation_suggestions(row):
    """Return a list of investigation tips based on taxpayer profile."""
    tips = []
    if row['Declared_Expenses'] > 0.80 * row['Annual_Income']:
        ratio = row['Declared_Expenses'] / row['Annual_Income'] * 100
        tips.append(f"📋 **Expense Verification:** Declared expenses are {ratio:.1f}% of income — "
                    f"obtain original receipts and invoices for all claimed deductions.")
    if row['Investment_Claims'] > 0.30 * row['Annual_Income']:
        ratio = row['Investment_Claims'] / row['Annual_Income'] * 100
        tips.append(f"📈 **Investment Audit:** Investment claims represent {ratio:.1f}% of income — "
                    f"verify supporting documents (ELSS, PPF, insurance policies).")
    if row['Late_Filing_Count'] > 3:
        tips.append(f"⏰ **Filing History Review:** {int(row['Late_Filing_Count'])} late filings detected — "
                    f"cross-check amendment history and late payment penalties.")
    if row['Previous_Penalty'] == 1:
        tips.append("⚖️ **Penalty Record:** Taxpayer has a prior penalty on record — "
                    "retrieve historical assessment orders.")
    if row.get('Anomaly_Flag', 0) == 1:
        tips.append("🔎 **Statistical Anomaly:** Financial figures deviate significantly from "
                    "peer group — conduct third-party income verification.")
    if not tips:
        tips.append("✅ **No Critical Red Flags:** This taxpayer's profile is within normal ranges. "
                    "Standard periodic review recommended.")
    return tips


# ══════════════════════════════════════════════════════
#  SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:10px 0 20px 0;'>
        <div style='font-size:40px;'>🧾</div>
        <div style='font-size:15px; font-weight:700; letter-spacing:1px;'>
            TAX FRAUD<br>RISK SYSTEM
        </div>
        <div style='font-size:11px; color:#aab; margin-top:4px;'>
            Powered by AI · ML Edition
        </div>
    </div>
    <hr style='border-color:#3a4a70; margin:0 0 16px 0;'>
    """, unsafe_allow_html=True)

    # Navigation
    st.markdown("**NAVIGATION**")
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

    st.markdown("<hr style='border-color:#3a4a70; margin:16px 0;'>", unsafe_allow_html=True)

    if MODEL_LOADED:
        total      = len(df)
        high_count = (df['Risk_Level'] == 'High').sum()
        anom_count = df['Anomaly_Flag'].sum()
        st.markdown(f"""
        <div style='font-size:12px; color:#99b;'>
            <b style='color:#e0e6f0;'>LIVE STATS</b><br>
            Total Records : <b style='color:#fff;'>{total:,}</b><br>
            High Risk     : <b style='color:#e74c3c;'>{high_count:,}</b><br>
            Anomalies     : <b style='color:#f39c12;'>{anom_count:,}</b>
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
        <h1 style='color:white!important;margin:0;font-size:28px;'>
            🧾 AI-Based Tax Fraud Risk Prediction System
        </h1>
        <p style='margin:8px 0 0 0; color:#ddd; font-size:14px;'>
            An intelligent platform for automated tax audit prioritization using
            Machine Learning and Anomaly Detection.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    total = len(df)
    high_c  = int((df['Risk_Level'] == 'High').sum())
    med_c   = int((df['Risk_Level'] == 'Medium').sum())
    low_c   = int((df['Risk_Level'] == 'Low').sum())
    fraud_c = int(df['Fraud_Label'].sum())
    anom_c  = int(df['Anomaly_Flag'].sum())

    col1.metric("Total Taxpayers", f"{total:,}", delta="2025 Dataset")
    col2.metric("🔴 High Risk",    f"{high_c:,}",  delta=f"{high_c/total*100:.1f}% of total", delta_color="inverse")
    col3.metric("🟡 Medium Risk",  f"{med_c:,}",   delta=f"{med_c/total*100:.1f}%", delta_color="off")
    col4.metric("🚨 Anomalies",    f"{anom_c:,}",  delta="Isolation Forest", delta_color="inverse")

    st.markdown("---")
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.subheader("How This System Works")
        st.markdown("""
        <div class='info-card'>
        <b>Step 1 — Data Ingestion</b><br>
        Taxpayer financial data is ingested and standardised. Categorical attributes
        (Profession, City) are encoded for machine learning.
        </div>
        <div class='info-card'>
        <b>Step 2 — Fraud Risk Classification</b><br>
        A <em>Random Forest Classifier</em> (200 trees) trained on historical patterns
        assigns each taxpayer a fraud probability score from 0–100%.
        </div>
        <div class='info-card'>
        <b>Step 3 — Anomaly Detection</b><br>
        An <em>Isolation Forest</em> model independently flags statistical outliers in
        income, expenses, and investment claims.
        </div>
        <div class='info-card'>
        <b>Step 4 — Risk Triage</b><br>
        Taxpayers are ranked into High / Medium / Low risk tiers, enabling auditors
        to prioritise investigations efficiently.
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.subheader("Risk Level Guide")
        st.markdown("""
        | Badge | Range | Action |
        |-------|-------|--------|
        | 🔴 High   | > 65% | Immediate audit required |
        | 🟡 Medium | 35–65% | Scheduled review |
        | 🟢 Low    | < 35% | Standard monitoring |
        """)
        st.subheader("Fraud Trigger Conditions")
        st.markdown("""
        - 📌 Expenses > 80% of annual income
        - 📌 Investment claims > 30% of income
        - 📌 Late filings > 3 times
        - 📌 Previous tax penalty on record
        """)


# ══════════════════════════════════════════════════════
#  PAGE: DASHBOARD
# ══════════════════════════════════════════════════════
elif page == "Dashboard":
    st.markdown("<h2>📊 Executive Dashboard</h2>", unsafe_allow_html=True)

    # Top KPIs
    k1, k2, k3, k4, k5 = st.columns(5)
    total   = len(df)
    fraud_r = df['Fraud_Label'].mean()
    avg_inc = df['Annual_Income'].mean()
    anom_r  = df['Anomaly_Flag'].mean()

    k1.metric("Total Taxpayers",  f"{total:,}")
    k2.metric("Fraud Rate",        f"{fraud_r:.1%}")
    k3.metric("Avg. Income",       f"₹{avg_inc/1e5:.1f}L")
    k4.metric("Anomaly Rate",      f"{anom_r:.1%}")
    k5.metric("High Risk",         f"{(df['Risk_Level']=='High').sum():,}")

    st.markdown("---")

    # Row 1
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Risk Distribution")
        risk_counts = df['Risk_Level'].value_counts()
        colors = {'High':'#c0392b','Medium':'#e67e22','Low':'#27ae60'}
        if USE_PLOTLY:
            fig = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                color=risk_counts.index,
                color_discrete_map=colors,
                hole=0.4
            )
            fig.update_layout(height=320, margin=dict(t=10,b=10,l=10,r=10),
                              legend=dict(orientation='h',y=-0.1))
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            wedge_colors = [colors.get(str(k),'#7f8c8d') for k in risk_counts.index]
            ax.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%',
                   colors=wedge_colors, startangle=140)
            ax.set_title('Risk Level Distribution', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with c2:
        st.subheader("Fraud Rate by Profession")
        fraud_prof = df.groupby('Profession')['Fraud_Label'].mean().sort_values(ascending=False)
        if USE_PLOTLY:
            fig = px.bar(
                x=fraud_prof.values, y=fraud_prof.index,
                orientation='h',
                color=fraud_prof.values,
                color_continuous_scale=['#27ae60','#e67e22','#c0392b'],
                labels={'x':'Fraud Rate','y':'Profession'},
                text=[f'{v:.1%}' for v in fraud_prof.values]
            )
            fig.update_layout(height=320, margin=dict(t=10,b=10,l=10,r=10),
                              coloraxis_showscale=False,
                              yaxis={'categoryorder':'total ascending'})
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.barh(fraud_prof.index, fraud_prof.values, color='#c0392b', alpha=0.85)
            ax.set_xlabel('Fraud Rate')
            ax.set_title('Fraud Rate by Profession', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # Row 2
    c3, c4 = st.columns(2)

    with c3:
        st.subheader("Income vs Declared Expenses")
        sample = df.sample(min(800, len(df)), random_state=42)
        if USE_PLOTLY:
            fig = px.scatter(
                sample,
                x='Annual_Income', y='Declared_Expenses',
                color='Risk_Level',
                color_discrete_map={'High':'#c0392b','Medium':'#e67e22','Low':'#27ae60'},
                opacity=0.55,
                labels={'Annual_Income':'Annual Income (₹)','Declared_Expenses':'Declared Expenses (₹)'}
            )
            fig.update_layout(height=320, margin=dict(t=10,b=10,l=10,r=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(6,3.5))
            for level, clr in [('High','#c0392b'),('Medium','#e67e22'),('Low','#27ae60')]:
                s = sample[sample['Risk_Level']==level]
                ax.scatter(s['Annual_Income']/1e5, s['Declared_Expenses']/1e5,
                           c=clr, label=level, alpha=0.4, s=12)
            ax.set_xlabel('Income (₹ Lakhs)'); ax.set_ylabel('Expenses (₹ Lakhs)')
            ax.legend(); ax.set_title('Income vs Expenses', fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with c4:
        st.subheader("City-wise High Risk Count")
        city_risk = df[df['Risk_Level']=='High'].groupby('City').size().sort_values(ascending=False)
        if USE_PLOTLY:
            fig = px.bar(
                x=city_risk.index, y=city_risk.values,
                color=city_risk.values,
                color_continuous_scale=['#f9a825','#c0392b'],
                labels={'x':'City','y':'High Risk Count'},
                text=city_risk.values
            )
            fig.update_layout(height=320, margin=dict(t=10,b=10,l=10,r=10),
                              coloraxis_showscale=False)
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(6,3.5))
            ax.bar(city_risk.index, city_risk.values, color='#c0392b', alpha=0.85)
            ax.set_xticklabels(city_risk.index, rotation=45, ha='right')
            ax.set_ylabel('Count'); ax.set_title('High Risk by City', fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()

    # Row 3
    st.subheader("Fraud Probability Distribution Across All Taxpayers")
    if USE_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df[df['Fraud_Label']==0]['Fraud_Probability'],
            name='Legitimate', marker_color='#27ae60', opacity=0.65, nbinsx=40
        ))
        fig.add_trace(go.Histogram(
            x=df[df['Fraud_Label']==1]['Fraud_Probability'],
            name='Fraud', marker_color='#c0392b', opacity=0.65, nbinsx=40
        ))
        fig.add_vline(x=0.5, line_dash='dash', line_color='black', annotation_text='Threshold 0.50')
        fig.update_layout(barmode='overlay', height=280,
                          margin=dict(t=10,b=10,l=10,r=10),
                          xaxis_title='Fraud Probability', yaxis_title='Count',
                          legend=dict(orientation='h',y=1.1))
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(12,3))
        ax.hist(df[df['Fraud_Label']==0]['Fraud_Probability'], bins=40,
                alpha=0.65, color='#27ae60', label='Legitimate')
        ax.hist(df[df['Fraud_Label']==1]['Fraud_Probability'], bins=40,
                alpha=0.65, color='#c0392b', label='Fraud')
        ax.axvline(0.5, ls='--', color='black', label='Threshold')
        ax.set_xlabel('Fraud Probability'); ax.legend()
        plt.tight_layout(); st.pyplot(fig); plt.close()


# ══════════════════════════════════════════════════════
#  PAGE: TAXPAYER SEARCH
# ══════════════════════════════════════════════════════
elif page == "Taxpayer Search":
    st.markdown("<h2>🔍 Taxpayer Search</h2>", unsafe_allow_html=True)

    # Filters
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

    # Apply filters
    filtered = df.copy()
    if search_id.strip():
        filtered = filtered[filtered['Taxpayer_ID'].str.contains(search_id.strip(), case=False)]
    if filter_prof != "All":
        filtered = filtered[filtered['Profession'] == filter_prof]
    if filter_risk != "All":
        filtered = filtered[filtered['Risk_Level'] == filter_risk]
    if filter_city != "All":
        filtered = filtered[filtered['City'] == filter_city]

    st.markdown(f"**{len(filtered):,} taxpayers found** matching your criteria.")

    # Display table with colour coding
    display_cols = ['Taxpayer_ID','Age','Profession','City','Annual_Income',
                    'Declared_Expenses','Investment_Claims','Late_Filing_Count',
                    'Previous_Penalty','Fraud_Label','Risk_Level','Anomaly_Flag','Fraud_Probability']
    display_df = filtered[display_cols].copy()
    display_df['Annual_Income']     = display_df['Annual_Income'].apply(lambda x: f"₹{x:,.0f}")
    display_df['Declared_Expenses'] = display_df['Declared_Expenses'].apply(lambda x: f"₹{x:,.0f}")
    display_df['Investment_Claims'] = display_df['Investment_Claims'].apply(lambda x: f"₹{x:,.0f}")
    display_df['Fraud_Probability'] = display_df['Fraud_Probability'].apply(lambda x: f"{x:.2%}")
    display_df['Anomaly_Flag'] = display_df['Anomaly_Flag'].map({0:'Normal', 1:'⚠ Anomaly'})
    display_df['Fraud_Label']  = display_df['Fraud_Label'].map({0:'Legitimate', 1:'🚨 Fraud'})

    st.dataframe(
        display_df.head(100),
        use_container_width=True,
        height=420
    )

    if len(filtered) > 100:
        st.caption(f"Showing first 100 of {len(filtered):,} results. Refine filters to narrow down.")

    # Download button
    csv = filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Filtered Results as CSV",
        data=csv,
        file_name="filtered_taxpayers.csv",
        mime="text/csv"
    )


# ══════════════════════════════════════════════════════
#  PAGE: RISK ANALYSIS
# ══════════════════════════════════════════════════════
elif page == "Risk Analysis":
    st.markdown("<h2>⚠️ Individual Risk Analysis</h2>", unsafe_allow_html=True)

    # Taxpayer selector
    taxpayer_ids = sorted(df['Taxpayer_ID'].tolist())
    col_sel1, col_sel2 = st.columns([2, 3])
    with col_sel1:
        selected_id = st.selectbox(
            "Select Taxpayer ID for Analysis",
            taxpayer_ids,
            index=0,
            help="Choose any taxpayer to view their complete risk profile."
        )

    row = df[df['Taxpayer_ID'] == selected_id].iloc[0]
    risk = str(row['Risk_Level'])
    fraud_prob = float(row['Fraud_Probability'])

    # ── Header row
    h1, h2, h3 = st.columns(3)
    with h1:
        st.markdown(f"""
        <div class='info-card'>
            <div style='font-size:13px; color:#666;'>Taxpayer</div>
            <div style='font-size:22px; font-weight:700; color:#1a2340;'>{selected_id}</div>
            <div style='font-size:14px; color:#555;'>{row['Profession']} · {row['City']}</div>
            <div style='font-size:13px; color:#777;'>Age: {int(row['Age'])}</div>
        </div>
        """, unsafe_allow_html=True)
    with h2:
        risk_color = {'High':'#c0392b','Medium':'#e67e22','Low':'#27ae60'}.get(risk,'#666')
        bar_width  = int(fraud_prob * 100)
        st.markdown(f"""
        <div class='info-card'>
            <div style='font-size:13px; color:#666;'>Fraud Risk Score</div>
            <div style='font-size:36px; font-weight:800; color:{risk_color};'>{fraud_prob:.1%}</div>
            <div style='background:#eee;border-radius:8px;height:10px;margin:6px 0;'>
                <div style='width:{bar_width}%;background:{risk_color};height:10px;border-radius:8px;'></div>
            </div>
            {risk_badge(risk)}
        </div>
        """, unsafe_allow_html=True)
    with h3:
        anomaly_icon = "⚠️ Flagged" if row['Anomaly_Flag'] == 1 else "✅ Normal"
        anomaly_color = "#c0392b" if row['Anomaly_Flag'] == 1 else "#27ae60"
        fraud_icon = "🚨 Fraud Record" if row['Fraud_Label'] == 1 else "✅ Clean Record"
        st.markdown(f"""
        <div class='info-card'>
            <div style='font-size:13px; color:#666;'>Flags</div>
            <div style='font-size:18px; font-weight:700; color:{anomaly_color}; margin-top:6px;'>
                {anomaly_icon}
            </div>
            <div style='font-size:15px; margin-top:4px;'>{fraud_icon}</div>
            <div style='font-size:13px; color:#777;'>Late Filings: {int(row['Late_Filing_Count'])} | Prior Penalty: {"Yes" if row['Previous_Penalty']==1 else "No"}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Financial comparison
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Income & Expense Breakdown")
        peer_income = df[df['Profession'] == row['Profession']]['Annual_Income'].mean()
        peer_expense = df[df['Profession'] == row['Profession']]['Declared_Expenses'].mean()
        peer_invest  = df[df['Profession'] == row['Profession']]['Investment_Claims'].mean()

        categories = ['Annual Income', 'Declared Expenses', 'Investment Claims']
        taxpayer_vals = [row['Annual_Income'], row['Declared_Expenses'], row['Investment_Claims']]
        peer_vals     = [peer_income, peer_expense, peer_invest]

        if USE_PLOTLY:
            fig = go.Figure(data=[
                go.Bar(name='This Taxpayer', x=categories, y=taxpayer_vals,
                       marker_color='#c0392b', opacity=0.85),
                go.Bar(name=f'Peer Avg ({row["Profession"]})', x=categories, y=peer_vals,
                       marker_color='#2c3e6e', opacity=0.75)
            ])
            fig.update_layout(barmode='group', height=320,
                              yaxis_tickprefix='₹', yaxis_tickformat=',',
                              margin=dict(t=10,b=10,l=10,r=10),
                              legend=dict(orientation='h',y=1.1))
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(6,3.5))
            x = np.arange(len(categories)); w = 0.35
            ax.bar(x-w/2, taxpayer_vals, w, label='This Taxpayer', color='#c0392b', alpha=0.85)
            ax.bar(x+w/2, peer_vals, w, label=f'Peer Avg', color='#2c3e6e', alpha=0.75)
            ax.set_xticks(x); ax.set_xticklabels(categories, fontsize=8)
            ax.set_ylabel('₹'); ax.legend(fontsize=8)
            ax.set_title('Income & Expense vs Peer Avg', fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()

        # Peer income note
        st.markdown(f"""
        <div class='info-card' style='border-left-color:#2c3e6e;'>
            <b>Peer Benchmark ({row['Profession']})</b><br>
            Average Income  : {format_inr(peer_income)}<br>
            Average Expenses: {format_inr(peer_expense)}<br>
            Expense Ratio   : {peer_expense/peer_income:.1%}
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.subheader("Feature Importance (Model View)")
        importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values()
        if USE_PLOTLY:
            bar_colors_fi = ['#c0392b' if v >= importances.median() else '#2c3e6e'
                             for v in importances.values]
            fig = go.Figure(go.Bar(
                x=importances.values,
                y=importances.index,
                orientation='h',
                marker_color=bar_colors_fi,
                text=[f'{v:.3f}' for v in importances.values],
                textposition='outside'
            ))
            fig.update_layout(height=320, margin=dict(t=10,b=10,l=10,r=10),
                              xaxis_title='Importance Score')
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(6,3.5))
            bar_colors_fi = ['#c0392b' if v >= importances.median() else '#2c3e6e'
                             for v in importances.values]
            ax.barh(importances.index, importances.values, color=bar_colors_fi, alpha=0.9)
            ax.set_xlabel('Importance')
            ax.set_title('Feature Importance', fontweight='bold')
            plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── Risk condition indicators
    st.markdown("---")
    st.subheader("Risk Condition Checklist")
    exp_ratio = row['Declared_Expenses'] / row['Annual_Income']
    inv_ratio = row['Investment_Claims']  / row['Annual_Income']

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        icon = "🔴" if exp_ratio > 0.80 else "🟢"
        st.markdown(f"""
        <div class='{"alert-card" if exp_ratio > 0.80 else "info-card"}'>
            {icon} <b>Expense Ratio</b><br>
            <span style='font-size:22px;font-weight:700;'>{exp_ratio:.1%}</span><br>
            Threshold: 80%<br>
            {'⚠ TRIGGERED' if exp_ratio > 0.80 else '✅ OK'}
        </div>""", unsafe_allow_html=True)
    with r2:
        icon = "🔴" if inv_ratio > 0.30 else "🟢"
        st.markdown(f"""
        <div class='{"alert-card" if inv_ratio > 0.30 else "info-card"}'>
            {icon} <b>Investment Ratio</b><br>
            <span style='font-size:22px;font-weight:700;'>{inv_ratio:.1%}</span><br>
            Threshold: 30%<br>
            {'⚠ TRIGGERED' if inv_ratio > 0.30 else '✅ OK'}
        </div>""", unsafe_allow_html=True)
    with r3:
        lfc = int(row['Late_Filing_Count'])
        icon = "🔴" if lfc > 3 else "🟢"
        st.markdown(f"""
        <div class='{"alert-card" if lfc > 3 else "info-card"}'>
            {icon} <b>Late Filings</b><br>
            <span style='font-size:22px;font-weight:700;'>{lfc}</span><br>
            Threshold: > 3<br>
            {'⚠ TRIGGERED' if lfc > 3 else '✅ OK'}
        </div>""", unsafe_allow_html=True)
    with r4:
        pp = int(row['Previous_Penalty'])
        icon = "🔴" if pp == 1 else "🟢"
        st.markdown(f"""
        <div class='{"alert-card" if pp == 1 else "info-card"}'>
            {icon} <b>Prior Penalty</b><br>
            <span style='font-size:22px;font-weight:700;'>{"Yes" if pp==1 else "No"}</span><br>
            Threshold: Any<br>
            {'⚠ TRIGGERED' if pp == 1 else '✅ OK'}
        </div>""", unsafe_allow_html=True)

    # ── Investigation suggestions
    st.markdown("---")
    st.subheader("📋 Audit Investigation Suggestions")
    suggestions = get_investigation_suggestions(row)
    for tip in suggestions:
        st.markdown(f"<div class='info-card'>{tip}</div>", unsafe_allow_html=True)

    # ── Similar risk taxpayers
    st.markdown("---")
    st.subheader(f"👥 Similar High-Risk Taxpayers (Same Profession: {row['Profession']})")
    similar = df[
        (df['Profession'] == row['Profession']) &
        (df['Risk_Level'] == 'High') &
        (df['Taxpayer_ID'] != selected_id)
    ][['Taxpayer_ID','Age','City','Annual_Income','Fraud_Probability','Anomaly_Flag']].head(8)
    if len(similar) > 0:
        similar['Annual_Income']    = similar['Annual_Income'].apply(lambda x: f"₹{x:,.0f}")
        similar['Fraud_Probability']= similar['Fraud_Probability'].apply(lambda x: f"{x:.2%}")
        similar['Anomaly_Flag']     = similar['Anomaly_Flag'].map({0:'Normal',1:'⚠ Anomaly'})
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
        <h3 style='color:#1a2340;margin-top:0;'>🎯 Project Objective</h3>
        This system demonstrates the application of <strong>Machine Learning</strong> to 
        tax fraud risk detection — a critical challenge for revenue authorities worldwide.
        By combining supervised classification with unsupervised anomaly detection, 
        it enables intelligent prioritisation of audit cases, reducing manual workload 
        while increasing detection accuracy.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='info-card'>
        <h3 style='color:#1a2340;margin-top:0;'>🤖 ML Techniques Used</h3>
        <b>1. Random Forest Classifier</b><br>
        An ensemble of 200 decision trees trained on historical taxpayer patterns.
        Uses <em>class_weight='balanced'</em> to handle the natural imbalance between 
        legitimate and fraudulent taxpayers. Outputs fraud probability scores.<br><br>
        <b>2. Isolation Forest</b><br>
        An unsupervised anomaly detection algorithm that isolates outliers by randomly 
        partitioning the feature space. Applied to financial figures (income, expenses, 
        investment claims) to flag statistical outliers independently of labels.<br><br>
        <b>3. Label Encoding</b><br>
        Categorical features (Profession, City) are ordinally encoded to allow 
        numerical processing by tree-based algorithms.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='info-card'>
        <h3 style='color:#1a2340;margin-top:0;'>📐 Model Performance</h3>

        | Metric | Value |
        |--------|-------|
        | Accuracy  | ~87–90% |
        | Precision | ~73–80% |
        | Recall    | ~42–55% |
        | F1 Score  | ~55–65% |

        > ℹ️ Recall is intentionally tuned conservatively to minimise false accusations.
        > A real deployment would tune the threshold based on audit resource capacity.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='info-card' style='border-left-color:#c0392b;'>
        <h3 style='color:#1a2340;margin-top:0;'>⚖️ Ethical Disclaimer</h3>
        <p style='font-size:13px;'>
        This system is built on <strong>synthetic data</strong> for educational and 
        demonstration purposes only.
        </p>
        <p style='font-size:13px;'>
        In real-world deployments, AI-based fraud detection systems must be:
        </p>
        <ul style='font-size:13px;'>
            <li>Audited for algorithmic bias</li>
            <li>Reviewed by qualified human officers</li>
            <li>Compliant with data protection laws</li>
            <li>Transparent to flagged individuals</li>
            <li>Regularly retrained on fresh data</li>
        </ul>
        <p style='font-size:13px;'>
        A high risk score is an <strong>investigative signal</strong>, not a verdict. 
        No enforcement action should be taken solely on ML predictions without 
        independent human review.
        </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='info-card'>
        <h3 style='color:#1a2340;margin-top:0;'>🛠️ Technology Stack</h3>
        <table style='font-size:13px;width:100%;'>
        <tr><td>🐍 Python 3.9+</td><td>Core language</td></tr>
        <tr><td>🌲 scikit-learn</td><td>ML models</td></tr>
        <tr><td>🐼 pandas</td><td>Data processing</td></tr>
        <tr><td>📦 joblib</td><td>Model persistence</td></tr>
        <tr><td>📊 Plotly</td><td>Visualisations</td></tr>
        <tr><td>🖥️ Streamlit</td><td>Web frontend</td></tr>
        </table>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='info-card'>
        <h3 style='color:#1a2340;margin-top:0;'>📁 File Structure</h3>
        <pre style='font-size:12px; background:#f0f2f6; padding:8px; border-radius:4px;'>
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

    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; color:#888; font-size:13px; padding:10px;'>
        AI-Based Tax Fraud Risk Prediction System · Built with Streamlit & scikit-learn
        · For educational and demonstration purposes only.
    </div>
    """, unsafe_allow_html=True)