import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from io import StringIO

# ── Page config ──
st.set_page_config(
    page_title="JetLag — Engine Health Monitor",
    page_icon="✈️",
    layout="wide"
)

# ── Custom CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main { background-color: #0D1B2A; }

h1, h2, h3 { font-family: 'Space Mono', monospace; }

.stApp { background-color: #0D1B2A; color: #E2E8F0; }

.metric-card {
    background: #112233;
    border: 1px solid #1E3A5F;
    border-radius: 8px;
    padding: 20px;
    text-align: center;
}

.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2.5rem;
    font-weight: 700;
    color: #14B8A6;
}

.metric-label {
    font-size: 0.8rem;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-top: 4px;
}

.alert-critical {
    background: linear-gradient(135deg, #7F1D1D, #991B1B);
    border: 2px solid #EF4444;
    border-radius: 8px;
    padding: 16px 24px;
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem;
    color: #FEE2E2;
    text-align: center;
    animation: pulse 1.5s infinite;
}

.alert-safe {
    background: linear-gradient(135deg, #064E3B, #065F46);
    border: 2px solid #10B981;
    border-radius: 8px;
    padding: 16px 24px;
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem;
    color: #D1FAE5;
    text-align: center;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.85; }
}

.header-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.8rem;
    font-weight: 700;
    color: #FFFFFF;
    letter-spacing: -1px;
}

.header-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 1rem;
    color: #14B8A6;
    font-style: italic;
    margin-top: -8px;
}

.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #F59E0B;
    text-transform: uppercase;
    letter-spacing: 3px;
    margin-bottom: 8px;
}

div[data-testid="stFileUploader"] {
    background: #112233;
    border: 2px dashed #1E3A5F;
    border-radius: 8px;
    padding: 10px;
}

.stButton > button {
    background: #0D9488;
    color: white;
    border: none;
    border-radius: 6px;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    letter-spacing: 1px;
    padding: 10px 24px;
    width: 100%;
}

.stButton > button:hover {
    background: #14B8A6;
}
</style>
""", unsafe_allow_html=True)

# ── Model definition (must match training) ──
class RULTransformer(nn.Module):
    def __init__(self, n_sensors=15, model_dim=64, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(n_sensors, model_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, 30, model_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=n_heads,
            dim_feedforward=128, dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.regressor = nn.Sequential(
            nn.Linear(model_dim, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.input_projection(x)
        x = x + self.pos_encoding
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.regressor(x).squeeze(-1)

SENSOR_COLS = ['s2','s3','s4','s6','s7','s8','s9','s11','s12','s13','s14','s15','s17','s20','s21']
WINDOW_SIZE = 30

# ── Load model and scaler ──
@st.cache_resource
def load_model():
    model = RULTransformer()
    model.load_state_dict(torch.load('jetlag_model.pth', map_location='cpu'))
    model.eval()
    return model

@st.cache_resource
def load_scaler():
    with open('jetlag_scaler.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()
scaler = load_scaler()

# ── Header ──
col_logo, col_title = st.columns([1, 8])
with col_title:
    st.markdown('<div class="header-title">✈ JetLag</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-subtitle">Turbofan Engine Remaining Useful Life Predictor — NASA C-MAPSS</div>', unsafe_allow_html=True)

st.markdown("---")

# ── Sidebar ──
with st.sidebar:
    st.markdown('<div class="section-label">About</div>', unsafe_allow_html=True)
    st.markdown("""
    **JetLag** predicts how many cycles a turbofan engine has left before failure.

    Built on NASA C-MAPSS simulated data using a Transformer neural network with critical zone loss weighting.

    **Model Performance:**
    - Overall RMSE: **3.56**
    - Critical Zone RMSE: **1.21**
    - Baseline RMSE: **41.37**
    """)

    st.markdown("---")
    st.markdown('<div class="section-label">How to use</div>', unsafe_allow_html=True)
    st.markdown("""
    1. Upload a CSV file with engine sensor readings
    2. Each row = one cycle
    3. Need at least 30 cycles
    4. Required columns: s2, s3, s4, s6, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21
    """)

    st.markdown("---")
    st.markdown('<div class="section-label">Try demo data</div>', unsafe_allow_html=True)
    use_demo = st.button("Load Demo Engine")

# ── Main content ──
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="section-label">Input</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload engine sensor CSV", type=['csv'], label_visibility="collapsed")

# ── Generate demo data ──
def make_demo_data(n_cycles=150):
    np.random.seed(42)
    data = {}
    for s in SENSOR_COLS:
        base = np.random.uniform(0, 1)
        trend = np.linspace(0, np.random.uniform(0.5, 2.0), n_cycles)
        noise = np.random.normal(0, 0.05, n_cycles)
        data[s] = base + trend + noise
    return pd.DataFrame(data)

# ── Load data ──
df_input = None
if use_demo:
    df_input = make_demo_data(150)
    with left:
        st.success("Demo engine loaded — 150 cycles")
elif uploaded_file:
    df_input = pd.read_csv(uploaded_file)
    with left:
        st.success(f"Loaded {len(df_input)} cycles")

# ── Predict and display ──
if df_input is not None:
    missing = [c for c in SENSOR_COLS if c not in df_input.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
    elif len(df_input) < WINDOW_SIZE:
        st.error(f"Need at least {WINDOW_SIZE} cycles. Got {len(df_input)}.")
    else:
        # Normalize
        sensor_data = df_input[SENSOR_COLS].copy()
        sensor_data_norm = scaler.transform(sensor_data)

        # Sliding windows — predict RUL at each cycle
        rul_predictions = []
        cycles = []
        for i in range(WINDOW_SIZE - 1, len(sensor_data_norm)):
            window = sensor_data_norm[i - WINDOW_SIZE + 1:i + 1]
            x = torch.FloatTensor(window).unsqueeze(0)
            with torch.no_grad():
                pred = model(x).item()
            pred = max(0, min(125, pred))
            rul_predictions.append(pred)
            cycles.append(i + 1)

        latest_rul = rul_predictions[-1]
        is_critical = latest_rul < 30

        # ── Metrics ──
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{latest_rul:.0f}</div>
                <div class="metric-label">Predicted RUL (cycles)</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(df_input)}</div>
                <div class="metric-label">Cycles Observed</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            health_pct = max(0, min(100, (latest_rul / 125) * 100))
            color = "#EF4444" if is_critical else "#14B8A6"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{color}">{health_pct:.0f}%</div>
                <div class="metric-label">Health Index</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Alert ──
        if is_critical:
            st.markdown(f'<div class="alert-critical">⚠ CRITICAL ZONE — RUL = {latest_rul:.0f} cycles — MAINTENANCE REQUIRED</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-safe">✓ ENGINE NOMINAL — RUL = {latest_rul:.0f} cycles remaining</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── RUL Curve ──
        col_plot1, col_plot2 = st.columns(2)

        with col_plot1:
            st.markdown('<div class="section-label">RUL Degradation Curve</div>', unsafe_allow_html=True)
            fig1, ax1 = plt.subplots(figsize=(6, 3.5))
            fig1.patch.set_facecolor('#0D1B2A')
            ax1.set_facecolor('#112233')

            # Color segments
            crit_idx = [i for i, r in enumerate(rul_predictions) if r < 30]
            safe_idx = [i for i, r in enumerate(rul_predictions) if r >= 30]

            if safe_idx:
                ax1.plot([cycles[i] for i in safe_idx], [rul_predictions[i] for i in safe_idx],
                         color='#14B8A6', linewidth=2, label='Safe zone')
            if crit_idx:
                ax1.plot([cycles[i] for i in crit_idx], [rul_predictions[i] for i in crit_idx],
                         color='#EF4444', linewidth=2, label='Critical zone')

            ax1.axhline(y=30, color='#F59E0B', linestyle='--', linewidth=1, alpha=0.7, label='Critical threshold (30)')
            ax1.fill_between(cycles, 0, 30, alpha=0.1, color='#EF4444')

            ax1.set_xlabel('Cycle', color='#94A3B8', fontsize=9)
            ax1.set_ylabel('Predicted RUL', color='#94A3B8', fontsize=9)
            ax1.set_title('Remaining Useful Life Over Time', color='#E2E8F0', fontsize=10, pad=10)
            ax1.tick_params(colors='#64748B', labelsize=8)
            for spine in ax1.spines.values():
                spine.set_edgecolor('#1E3A5F')
            ax1.legend(fontsize=7, facecolor='#112233', labelcolor='#E2E8F0', edgecolor='#1E3A5F')
            plt.tight_layout()
            st.pyplot(fig1)
            plt.close()

        with col_plot2:
            st.markdown('<div class="section-label">Sensor Importance</div>', unsafe_allow_html=True)

            # Pre-computed importance from training
            importance_scores = {
                's11': 19.77, 's13': 17.54, 's4': 14.38, 's14': 13.51, 's8': 13.30,
                's3': 12.09, 's15': 11.89, 's2': 11.72, 's7': 11.65, 's12': 10.93,
                's9': 10.23, 's21': 10.19, 's17': 6.54, 's20': 6.21, 's6': 1.73
            }

            top_sensors = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)[:10]
            s_names, s_vals = zip(*top_sensors)

            fig2, ax2 = plt.subplots(figsize=(6, 3.5))
            fig2.patch.set_facecolor('#0D1B2A')
            ax2.set_facecolor('#112233')

            colors_bar = ['#F59E0B' if v > 15 else '#0D9488' for v in s_vals]
            bars = ax2.barh(s_names, s_vals, color=colors_bar, edgecolor='none', height=0.6)

            ax2.set_xlabel('Importance Score', color='#94A3B8', fontsize=9)
            ax2.set_title('Top 10 Sensor Contributions', color='#E2E8F0', fontsize=10, pad=10)
            ax2.tick_params(colors='#64748B', labelsize=8)
            for spine in ax2.spines.values():
                spine.set_edgecolor('#1E3A5F')
            ax2.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        # ── Raw data preview ──
        with st.expander("View raw sensor data"):
            st.dataframe(df_input[SENSOR_COLS].tail(10).style.background_gradient(cmap='YlOrRd'), use_container_width=True)
