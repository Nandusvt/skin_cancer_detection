"""
SkinAI — Skin Cancer Detection & Classification
Redesigned Streamlit UI matching the DermVision HTML aesthetic.
"""

import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import base64
import os


# ─────────────────────────────────────────────
# AUTH GATE — must be FIRST thing in the file
# ─────────────────────────────────────────────
if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
    st.switch_page("pages/signin.py")
    st.stop()

# ...existing code...




# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SkinAI — Dermoscopy Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS  (mirrors your HTML exactly)
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Hide Streamlit chrome ── */
#MainMenu { visibility: hidden !important; }
[data-testid="stToolbar"] { display: none !important; }
header[data-testid="stHeader"] { display: none !important; }
.main .block-container { padding-top: 90px !important; }

/* ── Fixed Navbar ── */
.skinai-nav {
  position: fixed; top: 0; left: 0; right: 0; z-index: 9999;
  display: flex; justify-content: space-between; align-items: center;
  padding: 18px 50px;
  background: rgba(10, 20, 45, 0.6);
  backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px);
  border-bottom: 1px solid rgba(56, 189, 248, 0.12);
  box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
}
.nav-logo-brand { font-size: 26px; font-weight: 800; color: #e2e8f0; letter-spacing: 1px; }
.nav-logo-skin {
  font-weight: 900; font-size: 28px;
  background: linear-gradient(90deg, #38bdf8, #818cf8);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.nav-links-area { display: flex; gap: 0; }
.nav-links-area a {
  text-decoration: none; color: #94a3b8;
  font-weight: 600; font-size: 15px; margin-left: 32px;
  transition: color 0.3s; position: relative;
}
.nav-links-area a::after {
  content: ''; position: absolute; left: 0; bottom: -3px;
  width: 0; height: 2px;
  background: linear-gradient(90deg, #38bdf8, #818cf8);
  transition: width 0.3s ease; border-radius: 1px;
}
.nav-links-area a:hover { color: #e2e8f0; }
.nav-links-area a:hover::after { width: 100%; }

/* ── Hero two-column ── */
.hero-section {
  display: flex; min-height: calc(100vh - 80px);
  align-items: stretch; gap: 0;
}
.hero-left {
  flex: 1; display: flex; flex-direction: column; justify-content: center;
  padding: 60px 50px 60px 10px;
}
.hero-right {
  flex: 1; display: flex; align-items: center; justify-content: center;
  padding: 60px 0 60px 30px;
}
.image-frame {
  position: relative; width: 100%; min-height: 480px;
  border-radius: 32px; overflow: hidden;
  border: 1px solid rgba(56, 189, 248, 0.15);
  box-shadow: 0 0 60px rgba(14,165,233,0.08), 0 32px 80px rgba(0,0,0,0.5);
}
.image-frame img { width: 100%; height: 100%; object-fit: cover; display: block; }
.image-glow {
  position: absolute; inset: 0;
  background: linear-gradient(135deg, rgba(14,165,233,0.12) 0%, transparent 50%, rgba(99,102,241,0.08) 100%);
  pointer-events: none;
}

/* ── Hero text ── */
.hero-h1-new {
  font-family: 'Inter', sans-serif !important;
  font-size: clamp(2.4rem, 4vw, 3.8rem);
  font-weight: 900; line-height: 1.1;
  color: #f1f5f9; letter-spacing: -1px;
  margin-bottom: 24px;
}
.gradient-text {
  background: linear-gradient(90deg, #38bdf8 0%, #818cf8 60%, #c084fc 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-p { font-size: 17px; line-height: 1.8; color: #64748b; max-width: 500px; margin-bottom: 36px; }

/* ── CTA buttons ── */
.cta-row { display: flex; gap: 16px; margin-bottom: 48px; flex-wrap: wrap; }
.btn-primary-cta {
  text-decoration: none;
  background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);
  color: #ffffff; font-weight: 700; font-size: 15px;
  padding: 14px 32px; border-radius: 50px;
  box-shadow: 0 4px 24px rgba(14,165,233,0.35);
  transition: transform 0.2s, box-shadow 0.2s; letter-spacing: 0.3px; display: inline-block;
}
.btn-primary-cta:hover { transform: translateY(-3px); box-shadow: 0 8px 32px rgba(14,165,233,0.5); }
.btn-ghost-cta {
  text-decoration: none; background: rgba(56,189,248,0.08); color: #94a3b8;
  font-weight: 600; font-size: 15px; padding: 14px 32px;
  border-radius: 50px; border: 1px solid rgba(56,189,248,0.2);
  transition: background 0.2s, color 0.2s, border-color 0.2s; display: inline-block;
}
.btn-ghost-cta:hover { background: rgba(56,189,248,0.15); color: #e2e8f0; border-color: rgba(56,189,248,0.4); }

/* ── Root palette ── */
:root {
  --bg:        #07101f;
  --bg-card:   rgba(10, 20, 45, 0.72);
  --cyan:      #38bdf8;
  --indigo:    #818cf8;
  --violet:    #c084fc;
  --border:    rgba(56, 189, 248, 0.15);
  --text:      #e2e8f0;
  --muted:     #64748b;
  --danger:    #ef4444;
  --warning:   #f59e0b;
  --success:   #10b981;
}

/* ── Full-page background ── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
  background: var(--bg) !important;
  background-image:
    radial-gradient(ellipse at 12% 28%, rgba(0,180,255,0.07) 0%, transparent 55%),
    radial-gradient(ellipse at 85% 72%, rgba(130,60,255,0.07) 0%, transparent 55%) !important;
  font-family: 'Inter', sans-serif !important;
  color: var(--text) !important;
}

[data-testid="stSidebar"] {
  background: rgba(8, 16, 36, 0.95) !important;
  border-right: 1px solid var(--border) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #07101f; }
::-webkit-scrollbar-thumb { background: rgba(56,189,248,0.3); border-radius: 3px; }

/* ── Animated orbs (CSS only, injected via HTML) ── */
.orb {
  position: fixed; border-radius: 50%;
  filter: blur(90px); pointer-events: none; z-index: 0;
  animation: floatOrb 9s ease-in-out infinite;
}
.orb-1 { width:520px; height:520px; background:rgba(14,165,233,0.055); top:-120px; left:-120px; }
.orb-2 { width:420px; height:420px; background:rgba(99,102,241,0.055); bottom:-80px; right:-80px; animation-delay:-4.5s; }
.orb-3 { width:280px; height:280px; background:rgba(16,185,129,0.04);  top:45%;   left:48%;  animation-delay:-2.2s; }
@keyframes floatOrb {
  0%,100% { transform: translateY(0); }
  50%      { transform: translateY(22px); }
}

/* ── Logo / navbar strip ── */
.nav-strip {
  display: flex; align-items: center; gap: 14px;
  padding: 0 0 28px 0; margin-bottom: 4px;
}
.logo-mark {
  font-family: 'Inter', sans-serif; font-size: 28px; font-weight: 900;
  background: linear-gradient(90deg, #38bdf8, #818cf8);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  letter-spacing: -0.5px;
}
.logo-sub {
  font-size: 12px; font-weight: 600; color: var(--muted);
  text-transform: uppercase; letter-spacing: 2px; padding-top: 6px;
}

/* ── Hero heading ── */
.hero-h1 {
  font-family: 'Inter', sans-serif;
  font-size: clamp(2.4rem, 4vw, 3.8rem);
  font-weight: 900; line-height: 1.07;
  color: #f1f5f9; letter-spacing: -1.5px;
  margin-bottom: 16px;
}
.hero-h1 .grad {
  background: linear-gradient(90deg, #38bdf8 0%, #818cf8 55%, #c084fc 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-sub {
  font-size: 1.05rem; color: var(--muted);
  line-height: 1.8; max-width: 500px; margin-bottom: 28px;
}

/* ── Badge ── */
.badge {
  display: inline-block;
  background: rgba(14,165,233,0.12);
  border: 1px solid rgba(56,189,248,0.3);
  color: #38bdf8; font-size: 12px; font-weight: 700;
  padding: 5px 16px; border-radius: 50px;
  letter-spacing: 0.5px; margin-bottom: 20px; width: fit-content;
}

/* ── Stat strip ── */
.stats-row {
  display: flex; align-items: center; gap: 20px; flex-wrap: wrap;
  padding: 22px 28px;
  background: rgba(10,20,45,0.5);
  border: 1px solid var(--border);
  border-radius: 16px;
  backdrop-filter: blur(14px);
  margin-top: 8px; margin-bottom: 36px;
}
.stat-item { display: flex; flex-direction: column; gap: 2px; }
.stat-num {
  font-family: 'Inter', sans-serif; font-size: 20px; font-weight: 800;
  background: linear-gradient(90deg, #38bdf8, #818cf8);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.stat-label { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }
.stat-div { width:1px; height:32px; background: var(--border); }

/* ── Glassmorphic card ── */
.glass-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 28px 32px;
  backdrop-filter: blur(18px);
  box-shadow: 0 8px 40px rgba(0,0,0,0.35);
  margin-bottom: 20px;
  position: relative; overflow: hidden;
}
.glass-card::before {
  content: '';
  position: absolute; inset: 0;
  background: linear-gradient(135deg, rgba(56,189,248,0.04) 0%, transparent 60%);
  pointer-events: none;
}

/* ── Disclaimer ── */
.disclaimer {
  background: rgba(10,20,45,0.7);
  border-left: 4px solid var(--cyan);
  border-radius: 0 14px 14px 0;
  padding: 18px 24px;
  margin-bottom: 28px;
  font-size: 0.92rem; color: #94a3b8;
  line-height: 1.7;
  backdrop-filter: blur(10px);
}
.disclaimer h4 { color: var(--cyan); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 700; margin-bottom: 6px; }

/* ── Result banner ── */
.result-banner {
  border-radius: 18px; padding: 24px 30px;
  background: linear-gradient(135deg, rgba(10,20,45,0.9) 0%, rgba(30,41,59,0.9) 100%);
  border: 1px solid var(--border);
  position: relative; overflow: hidden;
  backdrop-filter: blur(16px);
  margin-bottom: 24px;
}
.result-banner::after {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 3px;
  background: linear-gradient(90deg, var(--cyan), var(--indigo), var(--violet));
}
.result-class {
  font-family: 'Inter', sans-serif; font-size: 1.8rem; font-weight: 800;
  background: linear-gradient(90deg, #38bdf8, #818cf8);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  margin: 6px 0;
}
.conf-pill {
  display: inline-block;
  background: rgba(56,189,248,0.12);
  border: 1px solid rgba(56,189,248,0.25);
  color: var(--cyan); font-size: 13px; font-weight: 700;
  padding: 4px 16px; border-radius: 50px;
}

/* ── Severity chip ── */
.sev-critical { background: rgba(239,68,68,0.15); border:1px solid rgba(239,68,68,0.4); color:#f87171; }
.sev-high     { background: rgba(245,158,11,0.15); border:1px solid rgba(245,158,11,0.4); color:#fbbf24; }
.sev-moderate { background: rgba(251,191,36,0.12); border:1px solid rgba(251,191,36,0.35);color:#fcd34d; }
.sev-low      { background: rgba(16,185,129,0.12); border:1px solid rgba(16,185,129,0.35);color:#34d399; }
.sev-chip {
  display: inline-block; font-size: 11px; font-weight: 700;
  padding: 3px 14px; border-radius: 50px;
  text-transform: uppercase; letter-spacing: 1px;
}

/* ── Info section cards ── */
.info-section {
  background: rgba(10,20,45,0.55);
  border: 1px solid var(--border);
  border-radius: 14px; padding: 20px 24px;
  margin-bottom: 14px;
  backdrop-filter: blur(10px);
}
.info-section h5 {
  font-family: 'Inter', sans-serif; font-size: 0.75rem;
  text-transform: uppercase; letter-spacing: 2px;
  color: var(--cyan); margin-bottom: 12px; font-weight: 700;
}
.info-section p, .info-section li { font-size: 0.93rem; color: #94a3b8; line-height: 1.7; }
.info-section ul { padding-left: 18px; }
.info-section li { margin-bottom: 4px; }

/* ── Recommendation bar ── */
.rec-bar {
  border-radius: 14px; padding: 18px 24px;
  background: rgba(10,20,45,0.7);
  border-left: 4px solid var(--cyan);
  font-size: 1.0rem; color: #cbd5e1; line-height: 1.7;
  margin-top: 8px;
}
.rec-bar strong { color: var(--text); }

/* ── Upload zone override ── */
[data-testid="stFileUploader"] {
  background: rgba(10,20,45,0.5) !important;
  border: 1.5px dashed rgba(56,189,248,0.3) !important;
  border-radius: 16px !important;
  padding: 8px !important;
}
[data-testid="stFileUploader"]:hover {
  border-color: rgba(56,189,248,0.6) !important;
}

/* ── Streamlit element resets ── */
h1,h2,h3,h4,h5 { font-family: 'Inter', sans-serif !important; color: var(--text) !important; }
p, li, label, div { color: inherit; }
[data-testid="stMarkdownContainer"] { color: var(--text); }

/* ── Button ── */
.stButton > button {
  background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%) !important;
  color: #fff !important; border: none !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 700 !important; font-size: 15px !important;
  padding: 12px 28px !important; border-radius: 50px !important;
  box-shadow: 0 4px 24px rgba(14,165,233,0.3) !important;
  transition: all 0.2s !important; letter-spacing: 0.3px !important;
}
.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 32px rgba(14,165,233,0.5) !important;
}

/* ── Success / error / info messages ── */
.stSuccess { background: rgba(16,185,129,0.1) !important; border: 1px solid rgba(16,185,129,0.3) !important; border-radius: 10px !important; }
.stError   { background: rgba(239,68,68,0.1) !important;  border: 1px solid rgba(239,68,68,0.3) !important;  border-radius: 10px !important; }
.stInfo    { background: rgba(56,189,248,0.08) !important; border: 1px solid rgba(56,189,248,0.25) !important; border-radius: 10px !important; }

/* ── Progress bar ── */
.stProgress > div > div { background: linear-gradient(90deg, #38bdf8, #818cf8) !important; }

/* ── Tab overrides ── */
[data-testid="stTabs"] button {
  font-family: 'Inter', sans-serif !important; font-weight: 600 !important;
  color: var(--muted) !important; border-radius: 8px 8px 0 0 !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
  color: var(--cyan) !important;
  border-bottom: 2px solid var(--cyan) !important;
}

/* ── Plotly transparent bg ── */
.js-plotly-plot .plotly { background: transparent !important; }

/* ── Sidebar text ── */
[data-testid="stSidebar"] * { color: #94a3b8 !important; }
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,[data-testid="stSidebar"] strong {
  color: var(--text) !important;
}

/* ── Download button ── */
.stDownloadButton > button {
  background: rgba(56,189,248,0.1) !important;
  border: 1px solid rgba(56,189,248,0.3) !important;
  color: var(--cyan) !important;
  font-family: 'Inter', sans-serif !important; font-weight: 700 !important;
  border-radius: 50px !important;
}
.stDownloadButton > button:hover {
  background: rgba(56,189,248,0.2) !important;
  border-color: rgba(56,189,248,0.5) !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Footer ── */
.footer {
  text-align: center; padding: 32px 0 16px;
  color: var(--muted); font-size: 0.82rem; line-height: 1.8;
}
.footer span {
  font-family: 'Inter', sans-serif; font-weight: 700;
  background: linear-gradient(90deg,#38bdf8,#818cf8);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
</style>

<!-- Animated orbs -->
<div class="orb orb-1"></div>
<div class="orb orb-2"></div>
<div class="orb orb-3"></div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _b64_img(path: str) -> str:
    """Return a base64 data-URI for a local image, or empty string if missing."""
    if os.path.exists(path):
        ext = os.path.splitext(path)[1].lstrip('.').lower()
        mime = 'jpeg' if ext == 'jpg' else ext
        with open(path, 'rb') as f:
            return f"data:image/{mime};base64,{base64.b64encode(f.read()).decode()}"
    return ''

DOC_IMG = _b64_img('doc.png')

# ─────────────────────────────────────────────
# DISEASE DATABASE
# ─────────────────────────────────────────────
DISEASE_INFO = {
    'akiec': {
        'name': 'Actinic Keratosis',
        'full_name': 'Actinic Keratosis & Intraepithelial Carcinoma',
        'severity': 'Moderate',
        'sev_class': 'sev-moderate',
        'accent': '#f59e0b',
        'description': 'A rough, scaly patch caused by years of UV exposure. Can progress to invasive squamous cell carcinoma if left untreated.',
        'symptoms': ['Rough, dry, or scaly patch of skin','Flat to slightly raised patch','Colour variations — pink, red, or brown','Itching or burning sensation in the affected area'],
        'risk_factors': ['Fair or light skin','Chronic sun exposure','Age over 40','Immunosuppressed individuals'],
        'treatment': 'Cryotherapy, topical fluorouracil / imiquimod, photodynamic therapy, or dermabrasion.',
        'recommendation': '⚠️ Consult a dermatologist promptly. Untreated lesions have a small but real chance of becoming cancerous.',
    },
    'bcc': {
        'name': 'Basal Cell Carcinoma',
        'full_name': 'Basal Cell Carcinoma',
        'severity': 'High',
        'sev_class': 'sev-high',
        'accent': '#ef4444',
        'description': 'The most common skin cancer. Grows slowly and rarely metastasises, but causes local tissue destruction if neglected.',
        'symptoms': ['Pearly or waxy bump','Flat, flesh-coloured lesion','Sore that heals and returns','Pink growth with raised, rolled edges'],
        'risk_factors': ['Chronic sun / UV exposure','Fair complexion','Family history of BCC','Prior radiation therapy'],
        'treatment': 'Mohs micrographic surgery, excision, electrodessication, radiation, or topical therapy.',
        'recommendation': '🚨 Book a dermatologist appointment as soon as possible for biopsy and treatment planning.',
    },
    'bkl': {
        'name': 'Benign Keratosis',
        'full_name': 'Benign Keratosis-like Lesions',
        'severity': 'Low',
        'sev_class': 'sev-low',
        'accent': '#10b981',
        'description': 'Non-cancerous growths including seborrhoeic keratoses and solar lentigines — a normal part of skin ageing.',
        'symptoms': ['Brown, black, or tan growths','Waxy "stuck-on" appearance','Slightly raised surface','Usually painless'],
        'risk_factors': ['Advancing age','Genetic predisposition','Cumulative sun exposure'],
        'treatment': 'No treatment required. Cryotherapy or laser if removal desired for cosmetic reasons.',
        'recommendation': '✅ Benign finding. Routine skin checks recommended to catch any future changes early.',
    },
    'df': {
        'name': 'Dermatofibroma',
        'full_name': 'Dermatofibroma',
        'severity': 'Low',
        'sev_class': 'sev-low',
        'accent': '#10b981',
        'description': 'A common benign fibrous nodule, typically triggered by minor trauma. Harmless and usually asymptomatic.',
        'symptoms': ['Small, firm subcutaneous bump','Red, brown, or purple pigmentation','Dimple sign when pinched','Occasionally itchy or tender'],
        'risk_factors': ['Minor skin injuries or insect bites','More prevalent in women 20–40 years','Unknown in most cases'],
        'treatment': 'No treatment needed. Excision if symptomatic or cosmetically bothersome.',
        'recommendation': '✅ Benign — removal only warranted if causing discomfort.',
    },
    'mel': {
        'name': 'Melanoma',
        'full_name': 'Melanoma',
        'severity': 'Critical',
        'sev_class': 'sev-critical',
        'accent': '#ef4444',
        'description': 'The deadliest form of skin cancer. Arises from melanocytes and can metastasise to lymph nodes and visceral organs if not detected early.',
        'symptoms': ['Asymmetric lesion','Irregular, poorly defined borders','Multiple or uneven colours','Diameter > 6 mm','Evolving in shape, size, or colour'],
        'risk_factors': ['Personal / family history of melanoma','High mole count or atypical naevi','Fair skin, light eyes, red/blonde hair','History of blistering sunburns','Immunosuppression'],
        'treatment': 'Wide local excision, sentinel node biopsy, immunotherapy (PD-1 inhibitors), targeted therapy (BRAF/MEK), or radiotherapy.',
        'recommendation': '🚨🚨 URGENT: See a dermatologist today. Thin melanomas caught early are highly curable — delay worsens prognosis dramatically.',
    },
    'nv': {
        'name': 'Melanocytic Nevus',
        'full_name': 'Melanocytic Nevus (Mole)',
        'severity': 'Low',
        'sev_class': 'sev-low',
        'accent': '#10b981',
        'description': 'A common benign mole composed of clusters of melanocytes. Stable naevi are almost never dangerous.',
        'symptoms': ['Round or oval, well-defined outline','Uniform tan, brown, or flesh colour','Flat or slightly dome-shaped','Stable over time'],
        'risk_factors': ['Genetics','Sun exposure in childhood','Fair complexion'],
        'treatment': 'No treatment required. Excision biopsy if clinical features raise concern.',
        'recommendation': '✅ Monitor using the ABCDE rule — Asymmetry, Border, Colour, Diameter, Evolving. Annual skin checks are wise.',
    },
    'vasc': {
        'name': 'Vascular Lesion',
        'full_name': 'Vascular Lesion',
        'severity': 'Low',
        'sev_class': 'sev-low',
        'accent': '#10b981',
        'description': 'Cutaneous vascular anomalies including haemangiomas, angiokeratomas, and pyogenic granulomas.',
        'symptoms': ['Bright red to deep purple discolouration','Flat (macular) or raised surface','Blanches under pressure (some subtypes)','Usually painless unless traumatised'],
        'risk_factors': ['Congenital vascular malformations','Hormonal fluctuations','Ageing','Trauma'],
        'treatment': 'Pulsed-dye or Nd:YAG laser, sclerotherapy, or surgical excision for problematic lesions.',
        'recommendation': '✅ Usually benign. Seek review if lesion bleeds repeatedly, grows rapidly, or changes character.',
    },
}

CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

# ─────────────────────────────────────────────
# MODEL LOADING & INFERENCE
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = keras.models.load_model('skin_cancer_model.h5')
        return model, None
    except Exception as e:
        return None, str(e)


def preprocess_image(image, target_size=(224, 224)):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size, Image.LANCZOS)
    img_array = np.array(image, dtype=np.float32)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)


def predict(model, image):
    processed = preprocess_image(image)
    preds = model.predict(processed, verbose=0)[0]
    top_idx = int(np.argmax(preds))
    return {
        'class': CLASS_NAMES[top_idx],
        'confidence': float(preds[top_idx]),
        'all_predictions': {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))},
    }

# ─────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter', color='#94a3b8'),
    margin=dict(l=16, r=16, t=40, b=16),
)

def probability_chart(predictions):
    df = pd.DataFrame([
        {'Label': DISEASE_INFO[k]['name'], 'Prob': v, 'Code': k}
        for k, v in predictions.items()
    ]).sort_values('Prob')

    bar_colors = []
    for p in df['Prob']:
        if   p >= 0.6: bar_colors.append('rgba(56,189,248,0.85)')
        elif p >= 0.25: bar_colors.append('rgba(129,140,248,0.6)')
        else:           bar_colors.append('rgba(71,85,105,0.5)')

    fig = go.Figure(go.Bar(
        x=df['Prob'], y=df['Label'], orientation='h',
        marker=dict(color=bar_colors, line=dict(width=0)),
        text=[f'{p:.1%}' for p in df['Prob']],
        textposition='outside',
        textfont=dict(color='#94a3b8', size=12),
        hovertemplate='<b>%{y}</b><br>%{x:.2%}<extra></extra>',
    ))
    fig.update_layout(
        **PLOT_LAYOUT,
        title=dict(text='Class Probabilities', font=dict(color='#e2e8f0', size=14), x=0.01),
        xaxis=dict(range=[0, 1.12], showgrid=False, zeroline=False, tickformat='.0%', color='#475569'),
        yaxis=dict(showgrid=False, color='#94a3b8'),
        height=360,
    )
    return fig


def gauge_chart(confidence):
    fig = go.Figure(go.Indicator(
        mode='gauge+number',
        value=confidence * 100,
        number=dict(suffix='%', font=dict(size=32, color='#e2e8f0')),
        title=dict(text='Confidence', font=dict(size=14, color='#94a3b8')),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor='#334155', tickfont=dict(color='#475569')),
            bar=dict(color='rgba(56,189,248,0.85)', thickness=0.22),
            bgcolor='rgba(0,0,0,0)',
            borderwidth=0,
            steps=[
                dict(range=[0,  50], color='rgba(239,68,68,0.08)'),
                dict(range=[50, 75], color='rgba(245,158,11,0.08)'),
                dict(range=[75,100], color='rgba(16,185,129,0.08)'),
            ],
            threshold=dict(line=dict(color='#38bdf8', width=3), thickness=0.8, value=confidence*100),
        ),
    ))
    fig.update_layout(**PLOT_LAYOUT, height=260)
    return fig

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 20px;">
        <div style="font-family:'Inter',sans-serif; font-size:22px; font-weight:900;
                    background:linear-gradient(90deg,#38bdf8,#818cf8);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            SkinAI
        </div>
        <div style="font-size:10px; color:#475569; letter-spacing:2px; text-transform:uppercase; margin-top:2px;">
            Dermoscopy Analysis v1.0
        </div>
    </div>
    <hr style="border-color:rgba(56,189,248,0.12); margin-bottom:20px;">
    """, unsafe_allow_html=True)

    st.markdown("**Model**")
    st.markdown("""
    <div style="font-size:13px; color:#64748b; line-height:1.7;">
        Architecture: <span style="color:#e2e8f0">EfficientNetB3</span><br>
        Dataset: <span style="color:#e2e8f0">HAM10000</span><br>
        Classes: <span style="color:#e2e8f0">7</span><br>
        Input: <span style="color:#e2e8f0">224 × 224 px</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>**Detectable Conditions**", unsafe_allow_html=True)

    sev_colors = {'Critical': '#f87171', 'High': '#fbbf24', 'Moderate': '#fcd34d', 'Low': '#34d399'}
    for code, info in DISEASE_INFO.items():
        dot = sev_colors.get(info['severity'], '#94a3b8')
        st.markdown(
            f'<div style="display:flex; align-items:center; gap:8px; margin:5px 0; font-size:13px; color:#94a3b8;">'
            f'<span style="width:7px;height:7px;border-radius:50%;background:{dot};flex-shrink:0;display:inline-block;"></span>'
            f'{info["name"]}</div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>**How to use**", unsafe_allow_html=True)
    for i, step in enumerate(['Upload a dermoscopy image','Click Analyse','Review results','Confirm with a dermatologist'], 1):
        st.markdown(
            f'<div style="display:flex;gap:10px;align-items:flex-start;margin:6px 0;font-size:13px;color:#64748b;">'
            f'<span style="color:#38bdf8;font-family:Inter,sans-serif;font-weight:800;flex-shrink:0;">{i:02d}</span>'
            f'{step}</div>',
            unsafe_allow_html=True
        )

# ─────────────────────────────────────────────
# MAIN — NAVBAR + HERO
# ─────────────────────────────────────────────
st.markdown("""
<!-- Fixed Navbar -->
<nav class="skinai-nav">
  <span class="nav-logo-brand"><span class="nav-logo-skin">Skin</span>AI</span>
  <div class="nav-links-area">
    <a href="#">Home</a>
    <a href="#">About</a>
  </div>
</nav>
""", unsafe_allow_html=True)

_doc_src = DOC_IMG or "https://placehold.co/600x480/0a142d/38bdf8?text=Doctor+Image"
st.markdown(f"""
<div class="hero-section">
  <!-- Left -->
  <div class="hero-left">
    <div class="badge">⚕️ AI-Powered Diagnosis</div>
    <h1 class="hero-h1-new">AI-Powered<br><span class="gradient-text">Skin Care</span></h1>
    <p class="hero-p">
      Our intelligent system helps detect and analyze skin conditions using
      advanced machine learning. Upload images, get instant insights, and
      understand possible risks with clarity and confidence. Designed for
      awareness, prevention, and smarter healthcare decisions, SkinAI brings
      technology and medical guidance together in one place.
    </p>
    <div class="cta-row">
      <a href="#" class="btn-primary-cta">Get Started →</a>
      <a href="#" class="btn-ghost-cta">Learn More</a>
    </div>
    <div class="stats-row">
      <div class="stat-item"><span class="stat-num">7</span><span class="stat-label">Conditions Detected</span></div>
      <div class="stat-div"></div>
      <div class="stat-item"><span class="stat-num">92%</span><span class="stat-label">Model Accuracy</span></div>
      <div class="stat-div"></div>
      <div class="stat-item"><span class="stat-num">HAM10K</span><span class="stat-label">Training Dataset</span></div>
    </div>
  </div>
  <!-- Right -->
  <div class="hero-right">
    <div class="image-frame">
      <img src="{_doc_src}" alt="Doctor" />
      <div class="image-glow"></div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="disclaimer">
    <h4>⚠️ Medical Disclaimer</h4>
    This tool is for <strong style="color:#e2e8f0">educational and screening purposes only</strong>.
    It is not a substitute for professional medical diagnosis.
    Always consult a qualified dermatologist before making clinical decisions.
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
model, error = load_model()

if error:
    st.markdown(f"""
    <div class="glass-card" style="border-left:4px solid #ef4444;">
        <h4 style="color:#f87171; margin-bottom:8px;">⚠️ Model Not Found</h4>
        <p style="color:#94a3b8; font-size:0.9rem; line-height:1.7;">
            Place <code style="color:#38bdf8">skin_cancer_model.h5</code> in the same directory as this script.<br>
            Error: <code style="color:#f87171">{error}</code>
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()
else:
    st.success("✅  Model loaded — ready to analyse")

st.markdown('<hr style="border-color:rgba(56,189,248,0.1);margin:28px 0;">', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# UPLOAD + ANALYSIS COLUMNS
# ─────────────────────────────────────────────

st.markdown('<div class="glass-card">', unsafe_allow_html=True)

st.markdown("#### 📤 Upload Lesion Image")

uploaded_file = st.file_uploader(
    "JPG · JPEG · PNG — Clear, well-lit dermoscopy photos work best",
    type=['jpg', 'jpeg', 'png'],
)

if uploaded_file:

    # Analyse button directly under upload
    run = st.button("🚀  Analyse Image", use_container_width=True)

    if run:
        image = Image.open(uploaded_file)

        # Fake loading animation
        import time
        prog = st.progress(0, text="Running inference…")
        for i in range(100):
            time.sleep(0.01)
            prog.progress(i + 1)
        prog.empty()

        results = predict(model, image)
        st.session_state['results'] = results
        st.session_state['image'] = image

        st.success("✅  Analysis complete")

# Display image AFTER analysis
if 'image' in st.session_state:
    st.markdown("<br>", unsafe_allow_html=True)
    st.image(
        st.session_state['image'],
        use_container_width=True
    )

st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────
if 'results' in st.session_state and uploaded_file:
    res   = st.session_state['results']
    cls   = res['class']
    conf  = res['confidence']
    info  = DISEASE_INFO[cls]

    st.markdown('<hr style="border-color:rgba(56,189,248,0.1);margin:32px 0 24px;">', unsafe_allow_html=True)

    # ── Result banner
    st.markdown(f"""
    <div class="result-banner">
        <div style="font-size:11px; color:#475569; text-transform:uppercase; letter-spacing:2px; margin-bottom:4px;">Top Prediction</div>
        <div class="result-class">{info['full_name']}</div>
        <div style="display:flex; gap:10px; align-items:center; margin-top:8px; flex-wrap:wrap;">
            <span class="conf-pill">Confidence {conf:.1%}</span>
            <span class="sev-chip {info['sev_class']}">{info['severity']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Charts
    ch1, ch2 = st.columns([3, 2], gap="large")
    with ch1:
        st.plotly_chart(probability_chart(res['all_predictions']), use_container_width=True, config={'displayModeBar': False})
    with ch2:
        st.plotly_chart(gauge_chart(conf), use_container_width=True, config={'displayModeBar': False})

    # ── Detail tabs
    st.markdown("#### 📋 Clinical Detail")
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Symptoms", "Risk Factors", "Treatment"])

    with tab1:
        st.markdown(f"""
        <div class="info-section">
            <h5>About this condition</h5>
            <p>{info['description']}</p>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        items = "".join(f"<li>{s}</li>" for s in info['symptoms'])
        st.markdown(f'<div class="info-section"><h5>Common Symptoms</h5><ul>{items}</ul></div>', unsafe_allow_html=True)

    with tab3:
        items = "".join(f"<li>{r}</li>" for r in info['risk_factors'])
        st.markdown(f'<div class="info-section"><h5>Risk Factors</h5><ul>{items}</ul></div>', unsafe_allow_html=True)

    with tab4:
        st.markdown(f"""
        <div class="info-section">
            <h5>Treatment Options</h5>
            <p>{info['treatment']}</p>
        </div>
        """, unsafe_allow_html=True)

    # ── Recommendation
    st.markdown(f'<div class="rec-bar">{info["recommendation"]}</div>', unsafe_allow_html=True)

    # ── Download
    st.markdown('<br>', unsafe_allow_html=True)
    report_lines = [
        "SKINAI — LESION ANALYSIS REPORT",
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "PREDICTION",
        f"  Diagnosis  : {info['full_name']}",
        f"  Confidence : {conf:.1%}",
        f"  Severity   : {info['severity']}",
        "",
        "CLASS PROBABILITIES",
        *[f"  {DISEASE_INFO[k]['name']:<30} {v:.1%}" for k, v in res['all_predictions'].items()],
        "",
        "DESCRIPTION",
        f"  {info['description']}",
        "",
        "RECOMMENDATION",
        f"  {info['recommendation']}",
        "",
        "─" * 60,
        "DISCLAIMER: Educational screening tool only.",
        "Not a substitute for professional medical diagnosis.",
        "Always consult a qualified dermatologist.",
    ]

    _, dc, _ = st.columns([1, 1, 1])
    with dc:
        st.download_button(
            label="📥  Download Report",
            data="\n".join(report_lines),
            file_name=f"skinai_{cls}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True,
        )

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown('<hr style="border-color:rgba(56,189,248,0.08); margin-top:48px;">', unsafe_allow_html=True)
st.markdown("""
<div class="footer">
    <span>SkinAI</span> &nbsp;·&nbsp; Powered by EfficientNetB3 &amp; TensorFlow<br>
    <span style="font-family:Inter; font-weight:400; font-size:11px; color:#334155;
                 -webkit-text-fill-color:#334155; background:none;">
        ⚕️ For educational and screening purposes only &nbsp;·&nbsp; Always consult a medical professional
    </span>
</div>
""", unsafe_allow_html=True)
