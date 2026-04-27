import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

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

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Space Mono', monospace; }
.stApp { background-color: #0D1B2A; color: #E2E8F0; }

.metric-card {
    background: #112233;
    border: 1px solid #1E3A5F;
    border-radius: 8px;
    padding: 20px;
    text-align: center;
}
.metric-value { font-family: 'Space Mono', monospace; font-size: 2.5rem; font-weight: 700; color: #14B8A6; }
.metric-label { font-size: 0.8rem; color: #64748B; text-transform: uppercase; letter-spacing: 2px; margin-top: 4px; }

.alert-red {
    background: linear-gradient(135deg, #7F1D1D, #991B1B);
    border: 2px solid #EF4444;
    border-radius: 8px; padding: 16px 24px;
    font-family: 'Space Mono', monospace; font-size: 1.1rem;
    color: #FEE2E2; text-align: center;
}
.alert-yellow {
    background: linear-gradient(135deg, #78350F, #92400E);
    border: 2px solid #F59E0B;
    border-radius: 8px; padding: 16px 24px;
    font-family: 'Space Mono', monospace; font-size: 1.1rem;
    color: #FEF3C7; text-align: center;
}
.alert-green {
    background: linear-gradient(135deg, #064E3B, #065F46);
    border: 2px solid #10B981;
    border-radius: 8px; padding: 16px 24px;
    font-family: 'Space Mono', monospace; font-size: 1.1rem;
    color: #D1FAE5; text-align: center;
}

.header-title { font-family: 'Space Mono', monospace; font-size: 2.8rem; font-weight: 700; color: #FFFFFF; }
.header-subtitle { font-size: 1rem; color: #14B8A6; font-style: italic; margin-top: -8px; }
.section-label { font-family: 'Space Mono', monospace; font-size: 0.7rem; color: #F59E0B; text-transform: uppercase; letter-spacing: 3px; margin-bottom: 8px; }

.confidence-box {
    background: #112233;
    border: 1px solid #1E3A5F;
    border-left: 4px solid #14B8A6;
    border-radius: 6px; padding: 12px 16px; margin: 8px 0;
}

div[data-testid="stFileUploader"] { background: #112233; border: 2px dashed #1E3A5F; border-radius: 8px; padding: 10px; }
.stButton > button { background: #0D9488; color: white; border: none; border-radius: 6px; font-family: 'Space Mono', monospace; font-weight: 700; letter-spacing: 1px; padding: 10px 24px; width: 100%; }
.stButton > button:hover { background: #14B8A6; }
</style>
""", unsafe_allow_html=True)

# ── Model definition ──
class RULTransformer(nn.Module):
    def __init__(self, n_sensors=15, model_dim=64, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(n_sensors, model_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, 30, model_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=n_heads,
            dim_feedforward=128, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.regressor = nn.Sequential(nn.Linear(model_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        x = self.input_projection(x)
        x = x + self.pos_encoding
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.regressor(x).squeeze(-1)

SENSOR_COLS = ['s2','s3','s4','s6','s7','s8','s9','s11','s12','s13','s14','s15','s17','s20','s21']
WINDOW_SIZE = 30

MODEL_INFO = {
    'FD001 — Single condition, HPC fault': {'file': 'jetlag_model.pth', 'rmse': 3.88, 'cz_rmse': 1.44},
    'FD002 — Multi-condition, HPC fault':  {'file': 'model_fd002.pth',  'rmse': 13.57, 'cz_rmse': None},
    'FD003 — Single condition, dual fault': {'file': 'model_fd003.pth', 'rmse': 3.79, 'cz_rmse': None},
    'FD004 — Multi-condition, dual fault':  {'file': 'model_fd004.pth', 'rmse': 12.67, 'cz_rmse': None},
    'Federated — Cross-fleet global model': {'file': 'federated_model.pth', 'rmse': None, 'cz_rmse': None},
}

@st.cache_resource
def load_model(model_file):
    model = RULTransformer()
    try:
        model.load_state_dict(torch.load(model_file, map_location='cpu'))
        model.eval()
        return model
    except:
        return None

@st.cache_resource
def load_scaler():
    with open('jetlag_scaler.pkl', 'rb') as f:
        return pickle.load(f)

scaler = load_scaler()

def predict_with_uncertainty(model, x_tensor, n_samples=50):
    """Monte Carlo Dropout for uncertainty estimation"""
    model.train()  # enable dropout
    predictions = []
    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(x_tensor).item()
            predictions.append(max(0, min(125, pred)))
    model.eval()
    mean_pred = np.mean(predictions)
    std_pred = np.std(predictions)
    ci_lower = max(0, mean_pred - 1.96 * std_pred)
    ci_upper = min(125, mean_pred + 1.96 * std_pred)
    return mean_pred, std_pred, ci_lower, ci_upper

def get_zone(rul):
    if rul < 30:
        return 'RED', '⚠ CRITICAL', '#EF4444'
    elif rul < 60:
        return 'YELLOW', '⚡ WARNING', '#F59E0B'
    else:
        return 'GREEN', '✓ NOMINAL', '#10B981'

# ── Header ──
st.markdown('<div class="header-title">✈ JetLag</div>', unsafe_allow_html=True)
st.markdown('<div class="header-subtitle">Turbofan Engine Remaining Useful Life Monitor — NASA C-MAPSS</div>', unsafe_allow_html=True)
st.markdown("---")

# ── Sidebar ──
with st.sidebar:
    st.markdown('<div class="section-label">Model Selection</div>', unsafe_allow_html=True)
    selected_model_name = st.selectbox("Select fleet model", list(MODEL_INFO.keys()))
    model_info = MODEL_INFO[selected_model_name]

    st.markdown("---")
    if model_info['rmse']:
        st.markdown(f"**Overall RMSE:** {model_info['rmse']}")
    if model_info['cz_rmse']:
        st.markdown(f"**Critical Zone RMSE:** {model_info['cz_rmse']}")

    st.markdown("---")
    st.markdown('<div class="section-label">Alert Zones</div>', unsafe_allow_html=True)
    st.markdown("🟢 **GREEN** — RUL > 60 cycles")
    st.markdown("🟡 **YELLOW** — RUL 30–60 cycles")
    st.markdown("🔴 **RED** — RUL < 30 cycles")

    st.markdown("---")
    st.markdown('<div class="section-label">How to Use</div>', unsafe_allow_html=True)
    st.markdown("Upload a CSV with columns: s2, s3, s4, s6, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21. Need at least 30 rows.")

    st.markdown("---")
    use_demo = st.button("Load Demo Engine")

# ── Load model ──
model = load_model(model_info['file'])
if model is None:
    st.warning(f"Model file {model_info['file']} not found. Using FD001 model.")
    model = load_model('jetlag_model.pth')

# ── Load data ──
def make_demo_data(n_cycles=180):
    np.random.seed(42)
    data = {}
    for i, s in enumerate(SENSOR_COLS):
        direction = 1 if s in ['s11', 's13'] else -1
        base = np.random.uniform(0.2, 0.8)
        trend = direction * np.linspace(0, np.random.uniform(1.5, 2.5), n_cycles)
        noise = np.random.normal(0, 0.08, n_cycles)
        data[s] = base + trend + noise
    return pd.DataFrame(data)

df_input = None
if use_demo:
    df_input = make_demo_data(180)
    st.success("Demo engine loaded — 180 cycles")
else:
    uploaded = st.file_uploader("Upload engine sensor CSV", type=['csv'], label_visibility="collapsed")
    if uploaded:
        df_input = pd.read_csv(uploaded)
        st.success(f"Loaded {len(df_input)} cycles")

# ── Main analysis ──
if df_input is not None:
    missing = [c for c in SENSOR_COLS if c not in df_input.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
    elif len(df_input) < WINDOW_SIZE:
        st.error(f"Need at least {WINDOW_SIZE} cycles.")
    else:
        sensor_data = df_input[SENSOR_COLS].copy()
        sensor_norm = scaler.transform(sensor_data)

        # Predict RUL + uncertainty at each cycle
        rul_preds, ci_lowers, ci_uppers, cycles = [], [], [], []
        for i in range(WINDOW_SIZE - 1, len(sensor_norm)):
            window = sensor_norm[i - WINDOW_SIZE + 1:i + 1]
            x = torch.FloatTensor(window).unsqueeze(0)
            mean_p, std_p, ci_l, ci_u = predict_with_uncertainty(model, x)
            rul_preds.append(mean_p)
            ci_lowers.append(ci_l)
            ci_uppers.append(ci_u)
            cycles.append(i + 1)

        latest_rul = rul_preds[-1]
        latest_ci_l = ci_lowers[-1]
        latest_ci_u = ci_uppers[-1]
        zone, zone_label, zone_color = get_zone(latest_rul)
        health_pct = max(0, min(100, (latest_rul / 125) * 100))

        # ── Metrics ──
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{latest_rul:.0f}</div><div class="metric-label">Predicted RUL</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><div class="metric-value" style="color:{zone_color}">{health_pct:.0f}%</div><div class="metric-label">Health Index</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card"><div class="metric-value" style="font-size:1.4rem">{latest_ci_l:.0f}–{latest_ci_u:.0f}</div><div class="metric-label">95% Confidence</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{len(df_input)}</div><div class="metric-label">Cycles Observed</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Alert ──
        if zone == 'RED':
            st.markdown(f'<div class="alert-red">⚠ CRITICAL ZONE — RUL = {latest_rul:.0f} cycles — MAINTENANCE REQUIRED IMMEDIATELY</div>', unsafe_allow_html=True)
        elif zone == 'YELLOW':
            st.markdown(f'<div class="alert-yellow">⚡ WARNING — RUL = {latest_rul:.0f} cycles — Schedule maintenance within {latest_rul:.0f} flights</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-green">✓ ENGINE NOMINAL — RUL = {latest_rul:.0f} cycles remaining — No action required</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Plots ──
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-label">RUL Degradation Curve with Confidence Band</div>', unsafe_allow_html=True)
            fig1, ax1 = plt.subplots(figsize=(6, 3.8))
            fig1.patch.set_facecolor('#0D1B2A')
            ax1.set_facecolor('#112233')

            rul_arr = np.array(rul_preds)
            ci_l_arr = np.array(ci_lowers)
            ci_u_arr = np.array(ci_uppers)

            # Color segments by zone
            for i in range(len(cycles)-1):
                c = cycles[i]
                r = rul_arr[i]
                color = '#EF4444' if r < 30 else '#F59E0B' if r < 60 else '#14B8A6'
                ax1.plot([cycles[i], cycles[i+1]], [rul_arr[i], rul_arr[i+1]], color=color, linewidth=2)

            # Confidence band
            ax1.fill_between(cycles, ci_l_arr, ci_u_arr, alpha=0.2, color='#14B8A6', label='95% CI')

            ax1.axhline(y=60, color='#F59E0B', linestyle='--', linewidth=1, alpha=0.6, label='Warning (60)')
            ax1.axhline(y=30, color='#EF4444', linestyle='--', linewidth=1, alpha=0.6, label='Critical (30)')
            ax1.fill_between(cycles, 0, 30, alpha=0.08, color='#EF4444')
            ax1.fill_between(cycles, 30, 60, alpha=0.05, color='#F59E0B')

            ax1.set_xlabel('Cycle', color='#94A3B8', fontsize=9)
            ax1.set_ylabel('Predicted RUL', color='#94A3B8', fontsize=9)
            ax1.set_title('RUL with 95% Confidence Band', color='#E2E8F0', fontsize=10)
            ax1.tick_params(colors='#64748B', labelsize=8)
            for spine in ax1.spines.values(): spine.set_edgecolor('#1E3A5F')
            ax1.legend(fontsize=7, facecolor='#112233', labelcolor='#E2E8F0', edgecolor='#1E3A5F')
            plt.tight_layout()
            st.pyplot(fig1)
            plt.close()

        with col2:
            st.markdown('<div class="section-label">Three-Zone Health Status</div>', unsafe_allow_html=True)
            fig2, ax2 = plt.subplots(figsize=(6, 3.8))
            fig2.patch.set_facecolor('#0D1B2A')
            ax2.set_facecolor('#112233')

            # Zone bands
            ax2.axhspan(60, 125, alpha=0.15, color='#10B981')
            ax2.axhspan(30, 60, alpha=0.15, color='#F59E0B')
            ax2.axhspan(0, 30, alpha=0.15, color='#EF4444')

            ax2.plot(cycles, rul_arr, color='white', linewidth=2)
            ax2.fill_between(cycles, ci_l_arr, ci_u_arr, alpha=0.15, color='white')

            ax2.axhline(y=60, color='#F59E0B', linestyle='--', linewidth=1, alpha=0.8)
            ax2.axhline(y=30, color='#EF4444', linestyle='--', linewidth=1, alpha=0.8)

            ax2.text(cycles[0]+2, 90, '🟢 GREEN — Nominal', color='#10B981', fontsize=8)
            ax2.text(cycles[0]+2, 44, '🟡 YELLOW — Warning', color='#F59E0B', fontsize=8)
            ax2.text(cycles[0]+2, 12, '🔴 RED — Critical', color='#EF4444', fontsize=8)

            ax2.set_xlabel('Cycle', color='#94A3B8', fontsize=9)
            ax2.set_ylabel('RUL', color='#94A3B8', fontsize=9)
            ax2.set_title('Three-Zone Maintenance Alert System', color='#E2E8F0', fontsize=10)
            ax2.tick_params(colors='#64748B', labelsize=8)
            for spine in ax2.spines.values(): spine.set_edgecolor('#1E3A5F')
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        # ── Sensor Importance ──
        st.markdown('<div class="section-label">Sensor Importance (Permutation — FD001)</div>', unsafe_allow_html=True)
        importance_scores = {
            's11': 19.77, 's13': 17.54, 's4': 14.38, 's14': 13.51, 's8': 13.30,
            's3': 12.09, 's15': 11.89, 's2': 11.72, 's7': 11.65, 's12': 10.93,
        }
        top = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        s_names, s_vals = zip(*top)

        fig3, ax3 = plt.subplots(figsize=(10, 2.8))
        fig3.patch.set_facecolor('#0D1B2A')
        ax3.set_facecolor('#112233')
        colors = ['#F59E0B' if v > 15 else '#0D9488' for v in s_vals]
        ax3.barh(s_names, s_vals, color=colors, height=0.6)
        ax3.set_xlabel('Importance (RMSE increase when shuffled)', color='#94A3B8', fontsize=9)
        ax3.set_title('s11 (fan speed) and s13 (corrected fan speed) dominate — consistent with HPC degradation physics', color='#E2E8F0', fontsize=9)
        ax3.tick_params(colors='#64748B', labelsize=8)
        ax3.invert_yaxis()
        for spine in ax3.spines.values(): spine.set_edgecolor('#1E3A5F')
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

        # ── Raw data ──
        with st.expander("View raw sensor data"):
            st.dataframe(df_input[SENSOR_COLS].tail(10).style.background_gradient(cmap='YlOrRd'), use_container_width=True)
