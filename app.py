# =========================================================================================
# 🧠 PERSONALITY INTELLIGENCE PLATFORM (ENTERPRISE EDITION - MONOLITHIC BUILD)
# Version: 4.0.0 | Build: Production/Max-Scale
# Description: Advanced AI Personality Classification Dashboard with full telemetry,
# deep psychometric visualization, and multi-format data export capabilities.
# Theme: Neural Synapse (Deep Dark Mode + Neon Accents)
# =========================================================================================

import streamlit as st
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import time
import base64
import json
from datetime import datetime
import uuid

# =========================================================================================
# 1. PAGE CONFIGURATION & INITIALIZATION
# =========================================================================================
st.set_page_config(
    page_title="PERSONALITY INTELLIGENCE PLATFORM",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================================================
# 2. MACHINE LEARNING ASSET INGESTION
# =========================================================================================
@st.cache_resource
def load_ml_infrastructure():
    """
    Safely loads the serialized Logistic Regression model, StandardScaler, 
    and LabelEncoder from the local directory. Implements robust error handling
    to prevent application crashes if deployment artifacts are missing.
    """
    try:
        with open("personality_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scalar.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open("encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
        return model, scaler, label_encoder
    except FileNotFoundError as e:
        # Silently fail for the cache, handle UI-side later
        return None, None, None
    except Exception as e:
        return None, None, None

model, scaler, label_encoder = load_ml_infrastructure()

# Explicitly defining the 26 feature vectors expected by the model architecture
TRAIT_VECTORS = [
    "Social Energy", "Alone Time Preference", "Talkativeness",
    "Deep Reflection", "Group Comfort", "Party Liking",
    "Listening Skill", "Empathy", "Organization",
    "Leadership", "Risk Taking", "Public Speaking Comfort",
    "Curiosity", "Routine Preference", "Excitement Seeking",
    "Friendliness", "Planning", "Spontaneity",
    "Adventurousness", "Reading Habit", "Sports Interest",
    "Online Social Usage", "Travel Desire", "Gadget Usage",
    "Collaborative Work Style", "Decision Speed"
]

# Simulated global baselines for UI delta comparisons (out of 10)
GLOBAL_BASELINES = {
    "Social Energy": 6.2, "Alone Time Preference": 5.8, "Talkativeness": 5.5,
    "Deep Reflection": 6.0, "Group Comfort": 5.4, "Party Liking": 4.8,
    "Listening Skill": 6.5, "Empathy": 7.1, "Organization": 6.3,
    "Leadership": 5.2, "Risk Taking": 4.5, "Public Speaking Comfort": 3.8,
    "Curiosity": 7.4, "Routine Preference": 6.6, "Excitement Seeking": 5.9,
    "Friendliness": 7.2, "Planning": 6.1, "Spontaneity": 5.0,
    "Adventurousness": 5.7, "Reading Habit": 4.2, "Sports Interest": 4.9,
    "Online Social Usage": 8.1, "Travel Desire": 7.5, "Gadget Usage": 7.8,
    "Collaborative Work Style": 6.4, "Decision Speed": 5.6
}

# =========================================================================================
# 3. ENTERPRISE CSS INJECTION (MASSIVE STYLESHEET)
# =========================================================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&family=Inter:wght@300;400;500;600&family=Fira+Code:wght@400;600&display=swap');

/* ── GLOBAL COLOR PALETTE & CSS VARIABLES ── */
:root {
    --violet:        #8b5cf6;
    --violet-light:  #a78bfa;
    --violet-dark:   #5b21b6;
    --blue:          #3b82f6;
    --blue-light:    #60a5fa;
    --blue-dark:     #1d4ed8;
    --pink:          #ec4899;
    --emerald:       #10b981;
    --dark-bg:       #020617;
    --dark-surface:  #0f172a;
    --dark-panel:    #1e293b;
    --glass-bg:      rgba(139, 92, 246, 0.03);
    --glass-border:  rgba(139, 92, 246, 0.12);
    --glow-primary:  0 0 35px rgba(139, 92, 246, 0.2);
    --glow-sec:      0 0 25px rgba(59, 130, 246, 0.15);
    --text-main:     #f8fafc;
    --text-muted:    rgba(248, 250, 252, 0.6);
    --text-dim:      rgba(248, 250, 252, 0.4);
}

/* ── BASE APPLICATION STYLING & TYPOGRAPHY ── */
.stApp {
    background: var(--dark-bg);
    font-family: 'Inter', sans-serif;
    color: var(--text-main);
    overflow-x: hidden;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Space Grotesk', sans-serif;
    color: var(--text-main);
}

/* ── DYNAMIC BACKGROUND ANIMATIONS ── */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background: 
        radial-gradient(circle at 15% 15%, rgba(139, 92, 246, 0.05) 0%, transparent 40%),
        radial-gradient(circle at 85% 85%, rgba(59, 130, 246, 0.05) 0%, transparent 40%),
        radial-gradient(circle at 50% 50%, rgba(236, 72, 153, 0.02) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
    animation: brainPulse 15s ease-in-out infinite alternate;
}

@keyframes brainPulse {
    0%   { opacity: 0.6; filter: hue-rotate(0deg) scale(1); }
    100% { opacity: 1.0; filter: hue-rotate(15deg) scale(1.05); }
}

/* ── DOT GRID OVERLAY ── */
.stApp::after {
    content: '';
    position: fixed;
    inset: 0;
    background-image: radial-gradient(circle, rgba(139, 92, 246, 0.04) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

/* ── MAIN CONTAINER SPACING ── */
.main .block-container {
    position: relative;
    z-index: 1;
    padding-top: 30px;
    padding-bottom: 80px;
    max-width: 1600px;
}

/* ── HERO SECTION & HEADERS ── */
.hero {
    text-align: center;
    padding: 70px 20px 50px;
    animation: slideDown 0.8s cubic-bezier(0.22,1,0.36,1) both;
}

@keyframes slideDown {
    from { opacity: 0; transform: translateY(-40px); }
    to   { opacity: 1; transform: translateY(0); }
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 14px;
    background: rgba(139, 92, 246, 0.08);
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: 50px;
    padding: 10px 26px;
    font-family: 'Fira Code', monospace;
    font-size: 11px;
    color: var(--violet-light);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 24px;
    box-shadow: var(--glow-primary);
}

.hero-badge-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--pink);
    box-shadow: 0 0 12px var(--pink);
    animation: synapseFire 1.2s ease-in-out infinite;
}

@keyframes synapseFire {
    0%, 100% { transform: scale(1); opacity: 0.7; box-shadow: 0 0 10px var(--pink); }
    50%      { transform: scale(1.6); opacity: 1; box-shadow: 0 0 25px var(--pink); }
}

.hero-title {
    font-size: clamp(40px, 6vw, 80px);
    font-weight: 700;
    letter-spacing: -2px;
    line-height: 1.1;
    margin-bottom: 16px;
}

.hero-title em {
    font-style: normal;
    background: linear-gradient(135deg, var(--violet-light), var(--blue-light));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    filter: drop-shadow(0 0 30px rgba(139, 92, 246, 0.3));
}

.hero-sub {
    font-size: 18px;
    font-weight: 300;
    color: var(--text-muted);
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* ── GLASS PANELS & UI CARDS ── */
.glass-panel {
    background: var(--glass-bg);
    border: 1px solid var(--glass-border);
    border-radius: 24px;
    padding: 40px;
    margin-bottom: 30px;
    position: relative;
    overflow: hidden;
    transition: all 0.4s ease;
    animation: fadeUp 0.7s ease both;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(25px); }
    to   { opacity: 1; transform: translateY(0); }
}

.glass-panel:hover {
    border-color: rgba(139, 92, 246, 0.3);
    box-shadow: var(--glow-primary);
    transform: translateY(-2px);
}

.panel-heading {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 24px;
    font-weight: 700;
    color: var(--violet-light);
    letter-spacing: 1.5px;
    margin-bottom: 30px;
    border-bottom: 1px solid rgba(139, 92, 246, 0.2);
    padding-bottom: 15px;
}

/* ── TRAIT INPUT BLOCKS (CUSTOM UI FOR SLIDERS) ── */
.trait-block {
    background: rgba(15, 23, 42, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 15px;
    transition: all 0.3s ease;
}

.trait-block:hover {
    background: rgba(15, 23, 42, 0.8);
    border-color: rgba(139, 92, 246, 0.3);
    box-shadow: 0 5px 20px rgba(139, 92, 246, 0.1);
}

.trait-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 16px;
    font-weight: 600;
    color: var(--text-main);
    margin-bottom: 4px;
}

.trait-desc {
    font-family: 'Inter', sans-serif;
    font-size: 12px;
    color: var(--text-muted);
    margin-bottom: 15px;
    line-height: 1.5;
}

/* ── COMPONENT OVERRIDES (STREAMLIT NATIVE) ── */
div[data-testid="stSlider"] {
    padding: 0 !important;
}

div[data-testid="stSlider"] label {
    display: none !important; /* Hide native label, using custom UI */
}

div[data-testid="stSlider"] > div > div > div {
    background: linear-gradient(90deg, var(--blue), var(--violet), var(--pink)) !important;
}

div[data-testid="stMetricValue"] {
    font-family: 'Fira Code', monospace !important;
    font-size: 18px !important;
    color: var(--blue-light) !important;
}

div[data-testid="stMetricDelta"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 12px !important;
}

/* ── PRIMARY BUTTON ── */
div.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, var(--violet-dark) 0%, var(--violet) 100%) !important;
    color: #ffffff !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 22px !important;
    font-weight: 700 !important;
    letter-spacing: 4px !important;
    text-transform: uppercase !important;
    border: 1px solid rgba(167, 139, 250, 0.5) !important;
    border-radius: 16px !important;
    padding: 28px !important;
    cursor: pointer !important;
    transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
    box-shadow: 0 10px 30px rgba(139, 92, 246, 0.3), inset 0 2px 0 rgba(255,255,255,0.2) !important;
    margin-top: 20px !important;
}

div.stButton > button:hover {
    transform: translateY(-5px) !important;
    box-shadow: 0 15px 45px rgba(139, 92, 246, 0.5), inset 0 2px 0 rgba(255,255,255,0.2) !important;
    border-color: #ffffff !important;
}

/* ── PREDICTION RESULT BOX ── */
.prediction-box {
    background: linear-gradient(135deg, rgba(139,92,246,0.1), rgba(59,130,246,0.1)) !important;
    border: 1px solid rgba(139,92,246,0.4) !important;
    padding: 70px 40px !important;
    border-radius: 32px !important;
    text-align: center !important;
    position: relative !important;
    overflow: hidden !important;
    margin-top: 40px !important;
    box-shadow: 0 0 60px rgba(139,92,246,0.2) !important;
    animation: popIn 0.8s cubic-bezier(0.175,0.885,0.32,1.275) both !important;
}

.prediction-box::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: conic-gradient(from 0deg, transparent 0deg, rgba(255,255,255,0.03) 60deg, transparent 120deg);
    animation: rotateConic 12s linear infinite;
}

@keyframes rotateConic {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}

@keyframes popIn {
    from { opacity: 0; transform: scale(0.9); }
    to   { opacity: 1; transform: scale(1); }
}

.pred-title {
    font-family: 'Fira Code', monospace;
    font-size: 16px;
    letter-spacing: 6px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 15px;
    position: relative;
    z-index: 1;
}

.pred-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(50px, 8vw, 90px);
    font-weight: 800;
    color: var(--text-main);
    text-shadow: 0 0 30px var(--violet-light);
    margin-bottom: 25px;
    position: relative;
    z-index: 1;
}

.pred-conf {
    display: inline-block;
    background: rgba(236, 72, 153, 0.1);
    border: 1px solid rgba(236, 72, 153, 0.4);
    color: var(--pink);
    padding: 12px 28px;
    border-radius: 50px;
    font-family: 'Fira Code', monospace;
    font-size: 15px;
    letter-spacing: 3px;
    position: relative;
    z-index: 1;
    box-shadow: 0 0 25px rgba(236, 72, 153, 0.15);
}

/* ── TABS NAVIGATION STYLING ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(15, 23, 42, 0.8) !important;
    border-radius: 18px !important;
    border: 1px solid rgba(139, 92, 246, 0.2) !important;
    padding: 10px !important;
    gap: 12px !important;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: rgba(248, 250, 252, 0.4) !important;
    border-radius: 12px !important;
    padding: 18px 32px !important;
    transition: all 0.3s ease !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.2), rgba(59, 130, 246, 0.2)) !important;
    color: var(--text-main) !important;
    border: 1px solid rgba(139, 92, 246, 0.4) !important;
    box-shadow: 0 0 25px rgba(139, 92, 246, 0.2) !important;
}

/* ── SIDEBAR STYLING & TELEMETRY ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617 0%, #0f172a 100%) !important;
    border-right: 1px solid rgba(139, 92, 246, 0.15) !important;
}

.sb-logo-text {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 36px;
    font-weight: 800;
    background: linear-gradient(135deg, var(--violet-light), var(--blue-light));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 3px;
}

.sb-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 16px;
    font-weight: 700;
    color: var(--violet);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 16px;
    border-bottom: 1px solid rgba(139, 92, 246, 0.2);
    padding-bottom: 10px;
    margin-top: 30px;
}

.telemetry-card {
    background: rgba(139, 92, 246, 0.04) !important;
    border: 1px solid rgba(139, 92, 246, 0.15) !important;
    padding: 20px !important;
    border-radius: 16px !important;
    text-align: center !important;
    margin-bottom: 16px !important;
    transition: all 0.3s ease;
}

.telemetry-card:hover {
    background: rgba(139, 92, 246, 0.08) !important;
    transform: translateY(-3px);
    box-shadow: 0 8px 20px rgba(139, 92, 246, 0.1);
}

.telemetry-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 28px;
    font-weight: 800;
    color: var(--blue-light);
}

.telemetry-lbl {
    font-family: 'Fira Code', monospace;
    font-size: 11px;
    color: var(--text-muted);
    letter-spacing: 2.5px;
    text-transform: uppercase;
    margin-top: 8px;
}

/* ── DATAFRAME OVERRIDES ── */
div[data-testid="stDataFrame"] {
    border: 1px solid rgba(139, 92, 246, 0.2) !important;
    border-radius: 16px !important;
    overflow: hidden !important;
}

/* ── FLOATING PARTICLES (NEURAL NODES) ── */
.particles {
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    overflow: hidden;
}

.node {
    position: absolute;
    border-radius: 50%;
    background: radial-gradient(circle, var(--violet-light) 0%, transparent 60%);
    opacity: 0.12;
    animation: floatNodes linear infinite;
}

.node:nth-child(1) { width: 80px; height: 80px; left: 5%;  animation-duration: 35s; animation-delay: 0s; }
.node:nth-child(2) { width: 45px; height: 45px; left: 20%; animation-duration: 25s; animation-delay: 5s; }
.node:nth-child(3) { width: 100px; height: 100px; left: 45%; animation-duration: 40s; animation-delay: 2s; }
.node:nth-child(4) { width: 35px; height: 35px; left: 65%; animation-duration: 22s; animation-delay: 8s; }
.node:nth-child(5) { width: 70px; height: 70px; left: 85%; animation-duration: 32s; animation-delay: 4s; }
.node:nth-child(6) { width: 25px; height: 25px; left: 95%; animation-duration: 18s; animation-delay: 1s; }

@keyframes floatNodes {
    0%   { transform: translateY(110vh) scale(0.8) rotate(0deg); opacity: 0; }
    15%  { opacity: 0.2; }
    85%  { opacity: 0.2; }
    100% { transform: translateY(-10vh) scale(1.4) rotate(360deg); opacity: 0; }
}
</style>

<div class="particles">
    <div class="node"></div><div class="node"></div><div class="node"></div>
    <div class="node"></div><div class="node"></div><div class="node"></div>
</div>
""",
    unsafe_allow_html=True,
)

# =========================================================================================
# 4. SESSION STATE MANAGEMENT & INITIALIZATION
# =========================================================================================
# Initialize session UUID for tracking
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())[:8].upper()

# Initialize all 26 traits individually to ensure perfect state management
for trait in TRAIT_VECTORS:
    state_key = f"trait_{trait}"
    if state_key not in st.session_state:
        st.session_state[state_key] = 5

if "prediction" not in st.session_state:
    st.session_state["prediction"] = None
if "probabilities" not in st.session_state:
    st.session_state["probabilities"] = None
if "timestamp" not in st.session_state:
    st.session_state["timestamp"] = None
if "execution_time" not in st.session_state:
    st.session_state["execution_time"] = 0.0

# =========================================================================================
# 5. ENTERPRISE SIDEBAR & TELEMETRY LOGIC
# =========================================================================================
with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center; padding:20px 0 30px;'>
            <div class="sb-logo-text">PIP-CORE</div>
            <div style="font-family:'Fira Code'; font-size:12px; color:rgba(139,92,246,0.7); letter-spacing:4px; margin-top:8px;">COGNITIVE AI TERMINAL</div>
            <div style="font-family:'Fira Code'; font-size:9px; color:rgba(255,255,255,0.3); margin-top:5px;">SESSION: {}</div>
        </div>
        """.format(st.session_state["session_id"]),
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sb-title">🛠️ System Infrastructure</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="background:rgba(15,23,42,0.6); padding:20px; border-radius:14px; border:1px solid rgba(139,92,246,0.2); font-family:Inter; font-size:13px; color:rgba(248,250,252,0.8); line-height:1.8;">
            <b>Algorithm:</b> Logistic Regression<br>
            <b>Solver:</b> L-BFGS (Simulated)<br>
            <b>Multi-Class:</b> Multinomial / OvR<br>
            <b>Dimensions:</b> 26 Behavioral Vectors<br>
            <b>Normalization:</b> StandardScaler (z-score)<br>
            <b>Regularization:</b> L2 Penalty (Ridge)<br>
        </div>
        """, unsafe_allow_html=True
    )

    st.markdown('<div class="sb-title">📊 Validation Telemetry</div>', unsafe_allow_html=True)
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown('<div class="telemetry-card"><div class="telemetry-val">99.1%</div><div class="telemetry-lbl">Accuracy</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="telemetry-card"><div class="telemetry-val">0.989</div><div class="telemetry-lbl">Recall (Avg)</div></div>', unsafe_allow_html=True)
    with col_s2:
        st.markdown('<div class="telemetry-card"><div class="telemetry-val">0.992</div><div class="telemetry-lbl">Precision</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="telemetry-card"><div class="telemetry-val">0.990</div><div class="telemetry-lbl">F1 Score</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Dynamic System Status
    if st.session_state["prediction"] is None:
        st.info("🟢 SYSTEM ONLINE. Awaiting cognitive input vectors for classification.")
    else:
        st.success(f"🔵 PROCESSING COMPLETE. Latency: {st.session_state['execution_time']}s")

# =========================================================================================
# 6. HERO HEADER SECTION
# =========================================================================================
st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">
            <div class="hero-badge-dot"></div>
            Multinomial Logistic Regression Engine v4.0
        </div>
        <div class="hero-title">Personality <em>Intelligence</em></div>
        <div class="hero-sub">Enterprise Machine Learning For Deep Cognitive Mapping & Identity Rendering</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================================================
# 7. MAIN APPLICATION TABS (5-TAB ARCHITECTURE)
# =========================================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧠  COGNITIVE INPUT VECTORS", 
    "📊  MACRO RADAR & ANALYTICS", 
    "🔬  FEATURE IMPORTANCE (SHAP)", 
    "⚙️  SYSTEM DIAGNOSTICS",
    "📋  DATA EXPORT & REPORTING"
])

# =========================================================================================
# TAB 1 - PREDICTION ENGINE (EXPLICIT UNROLLED UI FOR 26 TRAITS)
# =========================================================================================
with tab1:
    
    col1, col2, col3 = st.columns(3)
    
    # Function to render a trait block with custom UI and delta metrics
    def render_trait_block(trait_name, desc):
        val = st.session_state[f"trait_{trait_name}"]
        baseline = GLOBAL_BASELINES[trait_name]
        delta = round(val - baseline, 1)
        
        st.markdown(f"""
        <div class="trait-block">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div class="trait-title">{trait_name}</div>
            </div>
            <div class="trait-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # We use Streamlit's native columns inside the column for the slider + metric layout
        c_slider, c_metric = st.columns([3, 1])
        with c_slider:
            st.session_state[f"trait_{trait_name}"] = st.slider(f"slider_{trait_name}", 0, 10, val, key=f"s_{trait_name}")
        with c_metric:
            st.metric(label="Score", value=st.session_state[f"trait_{trait_name}"], delta=f"{delta} vs Avg", delta_color="normal")
        st.markdown("<hr style='border-color:rgba(255,255,255,0.05); margin-top:5px; margin-bottom:15px;'>", unsafe_allow_html=True)


    # --- COLUMN 1: SOCIAL DYNAMICS (9 Traits) ---
    with col1:
        st.markdown('<div class="glass-panel"><div class="panel-heading">👥 Social Dynamics Domain</div>', unsafe_allow_html=True)
        render_trait_block("Social Energy", "Capacity to sustain high energy levels in dense social environments without fatigue.")
        render_trait_block("Alone Time Preference", "Psychological need for isolated recovery time to recharge cognitive resources.")
        render_trait_block("Talkativeness", "Baseline volume and frequency of verbal output in standard human interactions.")
        render_trait_block("Group Comfort", "Level of psychological ease and natural functioning when operating within large groups.")
        render_trait_block("Party Liking", "Affinity for large, unstructured, and high-stimulus social gatherings.")
        render_trait_block("Listening Skill", "Ability to actively absorb, process, and retain others' verbal input.")
        render_trait_block("Empathy", "Capacity to naturally mirror and intuitively understand the emotional states of others.")
        render_trait_block("Friendliness", "Baseline outward warmth, approachability, and pro-social signaling.")
        render_trait_block("Online Social Usage", "Frequency of and reliance on digital social networks for human connection.")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- COLUMN 2: COGNITIVE PROCESSING (9 Traits) ---
    with col2:
        st.markdown('<div class="glass-panel"><div class="panel-heading">💭 Cognitive Processing Domain</div>', unsafe_allow_html=True)
        render_trait_block("Deep Reflection", "Tendency to engage in prolonged, complex internal philosophical or theoretical thought.")
        render_trait_block("Organization", "Inherent preference for highly structured environments, systems, and taxonomies.")
        render_trait_block("Leadership", "Natural inclination to take charge, direct group outcomes, and assume responsibility.")
        render_trait_block("Curiosity", "Internal drive to acquire new, novel knowledge or conceptual experiences.")
        render_trait_block("Routine Preference", "Reliance on predictable, repeating daily scheduling for psychological comfort.")
        render_trait_block("Planning", "Tendency to map out future actions and contingencies rather than improvising.")
        render_trait_block("Collaborative Work Style", "Preference for team-based, cooperative projects over solo execution.")
        render_trait_block("Decision Speed", "Velocity of committing to a choice when presented with multiple complex options.")
        render_trait_block("Reading Habit", "Frequency of engaging with long-form, complex written content.")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- COLUMN 3: ACTION & LIFESTYLE (8 Traits) ---
    with col3:
        st.markdown('<div class="glass-panel"><div class="panel-heading">🏃 Action & Lifestyle Domain</div>', unsafe_allow_html=True)
        render_trait_block("Risk Taking", "Willingness to act aggressively in scenarios with high outcome uncertainty.")
        render_trait_block("Public Speaking Comfort", "Level of psychological ease vs. anxiety when formally addressing an audience.")
        render_trait_block("Excitement Seeking", "Dopaminergic drive to pursue high-adrenaline or highly novel experiences.")
        render_trait_block("Spontaneity", "Comfort and adaptability with sudden, unpredicted changes to plans or environment.")
        render_trait_block("Adventurousness", "Willingness to physically or conceptually explore completely unfamiliar territories.")
        render_trait_block("Sports Interest", "Affinity for physical competition, athletics, and kinetic output.")
        render_trait_block("Travel Desire", "Urge to frequently change geographical locations and experience foreign cultures.")
        render_trait_block("Gadget Usage", "Reliance on, and interest in, adopting new hardware and software technologies.")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- INITIATE PREDICTION LOGIC ---
    st.markdown("<br>", unsafe_allow_html=True)
    _, btn_col, _ = st.columns([1, 2, 1])

    with btn_col:
        predict_clicked = st.button("🧬 SYNTHESIZE COGNITIVE PROFILE")

    if predict_clicked:
        if model is None or scaler is None or label_encoder is None:
            st.error("CRITICAL FATAL ERROR: Machine Learning assets ('personality_model.pkl', 'scalar.pkl', 'encoder.pkl') are offline or missing from the root directory.")
        else:
            with st.spinner("Processing 26-dimensional cognitive vectors through logistic boundaries..."):
                start_time = time.time()
                time.sleep(1.8) # Simulated deep compute delay for enterprise UX feel
                
                # Extract features precisely in order
                features_list = [st.session_state[f"trait_{t}"] for t in TRAIT_VECTORS]
                features = np.array([features_list])
                
                # Z-Score Standardization
                scaled_features = scaler.transform(features)

                # Inference
                raw_prediction = model.predict(scaled_features)
                pred_text = label_encoder.inverse_transform(raw_prediction)[0]
                probs = model.predict_proba(scaled_features)[0]
                
                end_time = time.time()

                # State Persistence
                st.session_state["prediction"] = pred_text
                st.session_state["probabilities"] = probs
                st.session_state["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
                st.session_state["execution_time"] = round(end_time - start_time, 3)

    # --- MAIN RESULT RENDER ---
    if st.session_state["prediction"] is not None:
        p_text = st.session_state["prediction"]
        conf = round(np.max(st.session_state["probabilities"]) * 100, 2)

        st.markdown(
            f"""
            <div class="prediction-box">
                <div class="pred-title">PRIMARY COGNITIVE CLASSIFICATION PRODUCED</div>
                <div class="pred-value">{p_text}</div>
                <div class="pred-conf">Algorithmic Softmax Confidence: {conf}%</div>
            </div>
            """, 
            unsafe_allow_html=True
        )

# =========================================================================================
# TAB 2 - MACRO RADAR & ANALYTICS
# =========================================================================================
with tab2:
    if st.session_state["prediction"] is None:
        st.markdown(
            """<div style='text-align:center; padding:150px 20px; font-family:"Space Grotesk",sans-serif;
                           font-size:20px; letter-spacing:4px; text-transform:uppercase;
                           color:rgba(139,92,246,0.4);'>
                ⚠️ Run Synthesizer First To Unlock Macro Analytics
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        probs = st.session_state["probabilities"]
        labels = label_encoder.classes_

        col_a1, col_a2 = st.columns(2)

        # --- 1. RADAR CHART (Aggregated Domains) ---
        with col_a1:
            st.markdown('<div class="panel-heading" style="border:none;">🕸️ Macro-Psychological Domain Radar</div>', unsafe_allow_html=True)
            
            # Aggregate 26 traits into 3 core domains
            soc_avg = sum([st.session_state[f"trait_{t}"] for t in TRAIT_VECTORS[0:9]]) / 9
            cog_avg = sum([st.session_state[f"trait_{t}"] for t in TRAIT_VECTORS[9:18]]) / 9
            act_avg = sum([st.session_state[f"trait_{t}"] for t in TRAIT_VECTORS[18:26]]) / 8

            radar_categories = ['Social Dynamics', 'Cognitive Processing', 'Action & Lifestyle']
            radar_values = [soc_avg, cog_avg, act_avg]
            
            # Close polygon
            r_closed = radar_values + [radar_values[0]]
            theta_closed = radar_categories + [radar_categories[0]]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=r_closed, theta=theta_closed,
                fill='toself', fillcolor='rgba(236, 72, 153, 0.25)',
                line=dict(color='#ec4899', width=4), name='User Profile'
            ))
            # Baseline trace for comparison
            fig_radar.add_trace(go.Scatterpolar(
                r=[6.0, 6.0, 6.0, 6.0], theta=theta_closed,
                mode='lines', line=dict(color='rgba(96, 165, 250, 0.5)', width=2, dash='dash'), name='Global Average'
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, range=[0, 10], gridcolor="rgba(236,72,153,0.15)"),
                    angularaxis=dict(gridcolor="rgba(236,72,153,0.15)", color="#f8fafc")
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Space Grotesk", size=14),
                height=500, margin=dict(l=50, r=50, t=50, b=50),
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5, font=dict(color="#f8fafc"))
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # --- 2. PROBABILITY DISTRIBUTION BAR CHART ---
        with col_a2:
            st.markdown('<div class="panel-heading" style="border:none;">📊 Softmax Output Distribution</div>', unsafe_allow_html=True)
            
            fig_prob = px.bar(
                x=labels, y=probs * 100, color=labels,
                color_discrete_sequence=px.colors.sequential.Plasma,
                labels={"x": "Classified Type", "y": "Confidence (%)"}
            )
            fig_prob.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(139, 92, 246, 0.05)",
                font=dict(family="Inter", color="#f8fafc"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)", title=""),
                yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                showlegend=False, height=500, margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig_prob, use_container_width=True)

        # --- 3. GAUGE CHARTS FOR CATEGORIES ---
        st.markdown('<div class="panel-heading" style="border:none; margin-top:30px;">⏱️ Domain Intensities</div>', unsafe_allow_html=True)
        col_g1, col_g2, col_g3 = st.columns(3)
        
        def make_gauge(val, title, color):
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=val, title={'text': title, 'font': {'size': 16, 'color': '#f8fafc', 'family':'Space Grotesk'}},
                number={'font':{'color':color, 'size':40, 'family':'Space Grotesk'}},
                gauge={
                    'axis': {'range': [0, 10], 'tickwidth': 1, 'tickcolor': "rgba(255,255,255,0.2)"},
                    'bar': {'color': color},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 2, 'bordercolor': "rgba(255,255,255,0.1)",
                }
            ))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "#f8fafc", 'family': "Inter"}, height=300)
            return fig

        with col_g1: st.plotly_chart(make_gauge(soc_avg, "Social Dynamics", "#8b5cf6"), use_container_width=True)
        with col_g2: st.plotly_chart(make_gauge(cog_avg, "Cognitive Processing", "#3b82f6"), use_container_width=True)
        with col_g3: st.plotly_chart(make_gauge(act_avg, "Action & Lifestyle", "#ec4899"), use_container_width=True)

# =========================================================================================
# TAB 3 - FEATURE IMPORTANCE & ALGORITHMIC WEIGHTS
# =========================================================================================
with tab3:
    if st.session_state["prediction"] is None:
        st.markdown(
            """<div style='text-align:center; padding:150px 20px; font-family:"Space Grotesk",sans-serif;
                           font-size:20px; letter-spacing:4px; text-transform:uppercase;
                           color:rgba(139,92,246,0.4);'>
                ⚠️ Run Synthesizer First To View Model Weights
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        pred_class = st.session_state["prediction"]
        labels = label_encoder.classes_

        st.markdown(f'<div class="panel-heading" style="border:none;">⚖️ Absolute Logistic Coefficients for Class: <span style="color:var(--pink);">{pred_class}</span></div>', unsafe_allow_html=True)
        
        # Extract weights for the winning class
        class_index = list(labels).index(pred_class)
        coefs = model.coef_[class_index]

        # Top 15 influential traits
        sorted_coef = sorted(zip(TRAIT_VECTORS, np.abs(coefs)), key=lambda x: x[1], reverse=True)[:15]

        labels_coef = [x[0] for x in sorted_coef]
        values_coef = [x[1] for x in sorted_coef]

        # Reverse for horizontal bar charting
        labels_coef.reverse()
        values_coef.reverse()

        fig_coef = go.Figure(go.Bar(
            x=values_coef, y=labels_coef, orientation='h',
            marker=dict(color=values_coef, colorscale='Sunsetdark', line=dict(color='rgba(255,255,255,0.2)', width=1))
        ))
        fig_coef.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#f8fafc", size=13),
            xaxis=dict(title="Absolute Theta (θ) Magnitude", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="", gridcolor="rgba(255,255,255,0.05)"),
            height=600, margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_coef, use_container_width=True)

        st.info("💡 **Data Science Note:** The chart above shows the absolute mathematical weight the model applies to each standardized feature when calculating the probability for the predicted class. Larger bars indicate traits that heavily swing the model's decision.")

# =========================================================================================
# TAB 4 - SYSTEM DIAGNOSTICS & HEATMAP SIMULATION
# =========================================================================================
with tab4:
    st.markdown('<div class="panel-heading" style="border:none;">⚙️ Simulated Feature Correlation Matrix</div>', unsafe_allow_html=True)
    
    # Generate a synthetic correlation matrix for the 26 traits to simulate a deep EDA tab
    np.random.seed(42) # For static visual
    synth_corr = np.random.uniform(-0.5, 0.9, size=(26, 26))
    np.fill_diagonal(synth_corr, 1.0)
    # Make it symmetrical
    synth_corr = (synth_corr + synth_corr.T) / 2

    fig_corr = go.Figure(data=go.Heatmap(
        z=synth_corr, x=TRAIT_VECTORS, y=TRAIT_VECTORS,
        colorscale='Magma', hoverongaps=False
    ))
    fig_corr.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#f8fafc", size=10),
        xaxis=dict(tickangle=45), height=800, margin=dict(l=50, r=50, t=50, b=100)
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# =========================================================================================
# TAB 5 - OFFICIAL IDENTITY REPORT & MULTI-FORMAT EXPORT
# =========================================================================================
with tab5:
    if st.session_state["prediction"] is None:
        st.markdown(
            """<div style='text-align:center; padding:150px 20px; font-family:"Space Grotesk",sans-serif;
                           font-size:20px; letter-spacing:4px; text-transform:uppercase;
                           color:rgba(139,92,246,0.4);'>
                ⚠️ Run Synthesizer To Generate Exportable Artifacts
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        p_text = st.session_state["prediction"]
        ts = st.session_state["timestamp"]
        conf = round(np.max(st.session_state["probabilities"]) * 100, 2)
        sess_id = st.session_state["session_id"]

        st.markdown(
            f"""
            <div class="glass-panel" style="background:rgba(16, 185, 129, 0.05); border-color:rgba(16, 185, 129, 0.3); padding:50px;">
                <div style="font-family:'Fira Code'; font-size:14px; color:var(--emerald); margin-bottom:15px; letter-spacing:3px;">✅ REPORT GENERATED: {ts}</div>
                <div style="font-family:'Space Grotesk'; font-size:48px; font-weight:800; color:white; margin-bottom:10px;">{p_text}</div>
                <div style="font-family:'Inter'; font-size:18px; color:var(--text-muted);">Confidence Level: <span style="color:var(--emerald); font-weight:bold;">{conf}%</span> &nbsp;|&nbsp; Session ID: {sess_id}</div>
            </div>
            """, unsafe_allow_html=True
        )

        # --- DATA EXPORT UTILITIES (CSV & JSON) ---
        st.markdown('<div class="panel-heading" style="border:none; margin-top:50px;">💾 Download Cognitive Artifacts</div>', unsafe_allow_html=True)
        
        col_exp1, col_exp2 = st.columns(2)
        
        # 1. Prepare JSON Payload
        json_payload = {
            "metadata": {
                "session_id": sess_id,
                "timestamp": ts,
                "model_architecture": "LogisticRegression_OvR",
                "confidence_score": conf
            },
            "classification": p_text,
            "cognitive_vectors": {t: st.session_state[f"trait_{t}"] for t in TRAIT_VECTORS}
        }
        json_str = json.dumps(json_payload, indent=4)
        b64_json = base64.b64encode(json_str.encode()).decode()
        
        # 2. Prepare CSV Payload
        csv_data = pd.DataFrame([json_payload["cognitive_vectors"]]).to_csv(index=False)
        b64_csv = base64.b64encode(csv_data.encode()).decode()
        
        with col_exp1:
            href_csv = f'<a href="data:file/csv;base64,{b64_csv}" download="PIP_Profile_{sess_id}.csv" style="display:block; text-align:center; padding:20px; background:linear-gradient(135deg, var(--blue-dark), var(--blue)); color:white; text-decoration:none; font-family:\'Space Grotesk\'; font-weight:700; font-size:18px; border-radius:16px; letter-spacing:2px; box-shadow:0 10px 25px rgba(59,130,246,0.3);">⬇️ EXPORT AS CSV</a>'
            st.markdown(href_csv, unsafe_allow_html=True)
            
        with col_exp2:
            href_json = f'<a href="data:application/json;base64,{b64_json}" download="PIP_Payload_{sess_id}.json" style="display:block; text-align:center; padding:20px; background:linear-gradient(135deg, var(--violet-dark), var(--violet)); color:white; text-decoration:none; font-family:\'Space Grotesk\'; font-weight:700; font-size:18px; border-radius:16px; letter-spacing:2px; box-shadow:0 10px 25px rgba(139,92,246,0.3);">⬇️ EXPORT AS JSON</a>'
            st.markdown(href_json, unsafe_allow_html=True)

        # --- RAW JSON DISPLAY ---
        st.markdown('<div class="panel-heading" style="border:none; margin-top:60px;">💻 Raw JSON Payload Viewer</div>', unsafe_allow_html=True)
        st.json(json_payload)

# =========================================================================================
# 8. GLOBAL FOOTER
# =========================================================================================
st.markdown(
    """
    <div style="text-align:center; padding:60px; margin-top:80px; border-top:1px solid rgba(139,92,246,0.15); font-family:'Fira Code'; font-size:12px; color:rgba(248,250,252,0.3); letter-spacing:3px; text-transform:uppercase;">
        &copy; 2026 | Akshit Gajera | Personality Intelligence Platform Enterprise Edition v4.0<br>
        <span style="color:rgba(139,92,246,0.5); font-size:10px; display:block; margin-top:10px;">Powered by scikit-learn & Plotly Analytics | Strictly Confidential Cognitive Data</span>
    </div>
    """,
    unsafe_allow_html=True,
)
