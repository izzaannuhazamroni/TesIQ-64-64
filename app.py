"""
RapIQ — Sistem Prediksi Kategori IQ
Berbasis Multilayer Perceptron (MLP)

Halaman:
  1. Prediksi Tunggal
  2. Pengujian Dataset (Bulk Prediction)
"""

import io
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="RapIQ — Prediksi Kategori IQ",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════
MODEL_PATH  = "model_mlp_iq.pkl"
SCALER_PATH = "scaler_iq.pkl"

REQUIRED_COLUMNS = [
    "education_mother",
    "education_father",
    "age_years",
    "gender",
]

EDU_MAP_ENCODE = {          # untuk CSV (bahasa Inggris)
    "primary or lower secondary": 0,
    "vocational": 1,
    "secondary": 2,
    "higher": 3,
}

EDU_MAP_UI = {              # untuk selectbox (bahasa Indonesia)
    "SD / SMP": 0,
    "Kejuruan": 1,
    "SMA": 2,
    "Perguruan Tinggi": 3,
}

GENDER_MAP = {"male": 1, "female": 0}
GENDER_MAP_UI = {"Laki-laki": 1, "Perempuan": 0}

IQ_LABELS = {
    0: "Moderate Intellectual Disability",
    1: "Mild Intellectual Disability",
    2: "Below Average",
    3: "Average",
    4: "Above Average",
}

IQ_RANGES = {
    0: "35 – 54",
    1: "55 – 69",
    2: "70 – 84",
    3: "85 – 114",
    4: "> 114",
}

IQ_COLORS = {
    0: "#DC2626",  # Red - Moderate Intellectual Disability
    1: "#F97316",  # Orange - Mild Intellectual Disability
    2: "#EAB308",  # Yellow - Below Average
    3: "#22C55E",  # Green - Average
    4: "#3B82F6",  # Blue - Above Average
}

# ══════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg: #F8FAFC;
    --text: #1E293B;
    --muted: #64748B;
    --card: #ffffff;
    --border: #E2E8F0;
    --primary: #3B82F6;
    --accent: #2563EB;
    --primary-soft: #EFF6FF;
    --header-start: #0F172A;
    --header-end: #334155;
    --info-bg: #EFF6FF;
    --info-border: #BFDBFE;
    --info-text: #0C4A6E;
    --error-bg: #FEE2E2;
    --error-border: #FECACA;
    --error-text: #7F1D1D;
    --success-bg: #F0FDF4;
    --success-border: #BBF7D0;
    --success-text: #166534;
    --shadow: 0 2px 12px rgba(15, 23, 42, 0.04);
    --shadow-strong: 0 15px 40px rgba(0, 0, 0, 0.15);
    --button-bg: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
    --button-text: #fff;
    --uploader-border: #BFDBFE;
    --uploader-bg: #F0F9FF;
}

[data-theme="dark"] {
    --bg: #020617;
    --text: #E2E8F0;
    --muted: #94A3B8;
    --card: #0F172A;
    --border: #334155;
    --primary: #60A5FA;
    --accent: #3B82F6;
    --primary-soft: #0F172A;
    --header-start: #111827;
    --header-end: #1E293B;
    --info-bg: #0F172A;
    --info-border: #334155;
    --info-text: #E0F2FE;
    --error-bg: rgba(248, 113, 113, 0.14);
    --error-border: #FCA5A5;
    --error-text: #F8B4B4;
    --success-bg: #164E63;
    --success-border: #0EA5E9;
    --success-text: #BAE6FD;
    --shadow: 0 2px 20px rgba(15, 23, 42, 0.5);
    --shadow-strong: 0 15px 40px rgba(0, 0, 0, 0.45);
    --button-bg: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
    --button-text: #fff;
    --uploader-border: #475569;
    --uploader-bg: #0F172A;
}

* { margin: 0; padding: 0; box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; color: var(--text); background: var(--bg); }

.stApp {
    background: var(--bg);
    color: var(--text);
}

#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 900px;
}

/* ── Main header ── */
.rapiq-header {
    background: linear-gradient(135deg, var(--header-start) 0%, var(--header-end) 50%, #334155 100%);
    border-radius: 24px;
    padding: 3rem 2.5rem;
    text-align: center;
    margin-bottom: 2.5rem;
    box-shadow: 0 20px 50px rgba(15, 23, 42, 0.15),
                0 0 1px rgba(15, 23, 42, 0.1);
    border: 1px solid rgba(148, 163, 184, 0.1);
    position: relative;
    overflow: hidden;
}

.rapiq-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(circle at 20% 50%, rgba(99, 102, 241, 0.08), transparent 50%),
                radial-gradient(circle at 80% 50%, rgba(59, 130, 246, 0.08), transparent 50%);
    pointer-events: none;
}

.rapiq-header-content {
    position: relative;
    z-index: 1;
}

.rapiq-header .logo {
    font-size: 4rem;
    line-height: 1;
    margin-bottom: .8rem;
    animation: float 3s ease-in-out infinite;
}

@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
}

.rapiq-header h1 {
    color: #fff;
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -0.8px;
    margin-bottom: .8rem;
    background: linear-gradient(135deg, #fff 0%, #E2E8F0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.rapiq-header p {
    color: rgba(255, 255, 255, 0.75);
    font-size: 0.95rem;
    font-weight: 400;
    line-height: 1.6;
    max-width: 600px;
    margin: 0 auto;
}

/* ── Tabs styling ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--card);
    border-radius: 16px;
    padding: 0.5rem;
    gap: 0.5rem;
    box-shadow: 0 4px 20px rgba(15, 23, 42, 0.06);
    border: 1px solid var(--border);
    margin-bottom: 2rem;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 12px;
    padding: 0.7rem 1.5rem;
    font-size: 0.9rem;
    font-weight: 500;
    color: #64748B;
    border: none;
    background: transparent;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.stTabs [data-baseweb="tab"]:hover {
    background: var(--primary-soft);
    color: var(--text);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #3B82F6 0%, #6366F1 100%) !important;
    color: #fff !important;
    font-weight: 600;
    box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
}

/* ── Section cards ── */
.section-card {
    background: var(--card);
    border-radius: 18px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.8rem;
    box-shadow: var(--shadow);
    border: 1px solid var(--border);
    transition: all 0.3s ease;
}

.section-card:hover {
    box-shadow: 0 8px 25px rgba(15, 23, 42, 0.08);
    border-color: #CBD5E1;
}

.section-title {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--primary);
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}

/* ── Primary button ── */
.stButton > button {
    width: 100%;
    background: var(--button-bg);
    color: var(--button-text);
    font-family: 'Inter', sans-serif;
    font-size: 0.95rem;
    font-weight: 600;
    border: none;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    cursor: pointer;
    letter-spacing: 0.3px;
    box-shadow: 0 4px 20px rgba(59, 130, 246, 0.3);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(59, 130, 246, 0.4);
}

.stButton > button:active {
    transform: translateY(0);
}

/* ── Download button ── */
.stDownloadButton > button {
    width: 100%;
    background: var(--card);
    color: var(--primary);
    font-family: 'Inter', sans-serif;
    font-size: 0.95rem;
    font-weight: 600;
    border: 2px solid var(--primary);
    border-radius: 12px;
    padding: 0.85rem 1.5rem;
    cursor: pointer;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.stDownloadButton > button:hover {
    background: var(--primary);
    color: var(--button-text);
    box-shadow: 0 4px 20px rgba(59, 130, 246, 0.3);
    transform: translateY(-2px);
}

/* ── Result card ── */
.result-card {
    border-radius: 20px;
    padding: 2.4rem 2rem;
    margin-top: 2rem;
    text-align: center;
    color: #fff;
    box-shadow: 0 15px 40px rgba(0, 0, 0, 0.15);
    border: none;
}

.result-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    opacity: 0.85;
    margin-bottom: 0.6rem;
}

.result-category {
    font-size: 1.8rem;
    font-weight: 800;
    margin-bottom: 0.4rem;
    letter-spacing: -0.5px;
}

.result-range {
    font-size: 0.95rem;
    font-weight: 500;
    margin-bottom: 1.2rem;
    opacity: 0.9;
}

.result-confidence {
    display: inline-block;
    background: rgba(255, 255, 255, 0.25);
    border-radius: 50px;
    padding: 0.5rem 1.4rem;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.4px;
    backdrop-filter: blur(10px);
}

/* ── Stat cards ── */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 1.2rem;
    margin-bottom: 2rem;
}

.stat-card {
    background: linear-gradient(135deg, var(--card) 0%, var(--primary-soft) 100%);
    border-radius: 16px;
    padding: 1.5rem 1.2rem;
    box-shadow: var(--shadow);
    border: 1px solid var(--border);
    text-align: center;
    transition: all 0.3s ease;
}

.stat-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 30px rgba(59, 130, 246, 0.1);
    border-color: #BFDBFE;
}

.stat-card .stat-val {
    font-size: 1.8rem;
    font-weight: 800;
    color: #3B82F6;
    line-height: 1.1;
}

.stat-card .stat-key {
    font-size: 0.75rem;
    color: #64748B;
    font-weight: 600;
    margin-top: 0.6rem;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* ── Probability table ── */
.prob-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
    margin-top: 1rem;
}

.prob-table th {
    text-align: left;
    color: var(--muted);
    font-weight: 600;
    padding: 0.8rem 0.8rem;
    border-bottom: 2px solid var(--border);
    background: var(--primary-soft);
}

.prob-table td {
    padding: 0.8rem 0.8rem;
    border-bottom: 1px solid var(--surface-bg);
    color: var(--text);
}

.prob-table tr:hover {
    background: var(--primary-soft);
}

.prob-table tr:last-child td {
    border-bottom: none;
}

.prob-bar-wrap {
    background: #E0E7FF;
    border-radius: 8px;
    height: 10px;
    width: 140px;
    display: inline-block;
    vertical-align: middle;
    margin-right: 8px;
    overflow: hidden;
}

.prob-bar-fill {
    border-radius: 8px;
    height: 10px;
    display: inline-block;
    background: linear-gradient(90deg, #3B82F6, #6366F1);
}

/* ── Info box ── */
.info-box {
    background: var(--info-bg);
    border: 1.5px solid var(--info-border);
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    font-size: 0.88rem;
    color: var(--info-text);
    margin-bottom: 1.2rem;
    line-height: 1.6;
}

.info-box code {
    background: #3B82F6;
    color: #fff;
    border-radius: 6px;
    padding: 0.2rem 0.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    font-weight: 500;
}

/* ── Error box ── */
.err-box {
    background: var(--error-bg);
    border: 1.5px solid var(--error-border);
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    color: var(--error-text);
    font-size: 0.9rem;
    margin-bottom: 1.2rem;
    font-weight: 500;
    line-height: 1.6;
}

/* ── Success box ── */
.ok-box {
    background: var(--success-bg);
    border: 1.5px solid var(--success-border);
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    color: var(--success-text);
    font-size: 0.9rem;
    margin-bottom: 1.2rem;
    font-weight: 500;
    line-height: 1.6;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 2px dashed var(--uploader-border) !important;
    border-radius: 16px !important;
    background: var(--uploader-bg) !important;
    padding: 2rem !important;
}

/* ── Input styling ── */
.stNumberInput input, .stSelectbox select {
    border-radius: 10px !important;
    border: 1.5px solid var(--border) !important;
    padding: 0.7rem 1rem !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
}

.stNumberInput input:focus, .stSelectbox select:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
}

/* ── Scrollbar styling ── */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: #F1F5F9;
}

::-webkit-scrollbar-thumb {
    background: #CBD5E1;
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: #94A3B8;
}

/* ── Responsive ── */
@media (max-width: 640px) {
    .rapiq-header {
        padding: 2rem 1.5rem;
    }
    
    .rapiq-header h1 {
        font-size: 2rem;
    }
    
    .stat-grid {
        grid-template-columns: repeat(2, 1fr);
    }
    
    .prob-table {
        font-size: 0.75rem;
    }
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# LOAD MODEL (cached)
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Memuat model…")
def load_artifacts(model_path: str, scaler_path: str):
    errors, model, scaler = [], None, None

    if not os.path.exists(model_path):
        errors.append(f"File model tidak ditemukan: <code>{model_path}</code>")
    else:
        try:
            model = joblib.load(model_path)
        except Exception as e:
            errors.append(f"Gagal memuat model: {e}")

    if not os.path.exists(scaler_path):
        errors.append(f"File scaler tidak ditemukan: <code>{scaler_path}</code>")
    else:
        try:
            scaler = joblib.load(scaler_path)
        except Exception as e:
            errors.append(f"Gagal memuat scaler: {e}")

    return model, scaler, errors


# ══════════════════════════════════════════════════════════════
# SESSION STATE HELPERS
# ══════════════════════════════════════════════════════════════
def ensure_state():
    if "active_page" not in st.session_state:
        st.session_state.active_page = "single"
    if "theme_mode" not in st.session_state:
        st.session_state.theme_mode = "light"
    if "single_result" not in st.session_state:
        st.session_state.single_result = None


def apply_theme():
    theme = st.session_state.get("theme_mode", "light")
    st.markdown(
        f"<script>document.documentElement.setAttribute('data-theme', '{theme}');</script>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════
# VALIDASI CSV
# ══════════════════════════════════════════════════════════════
def validasi_csv(df: pd.DataFrame) -> pd.DataFrame:
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        bullet = "\n".join(f"• {c}" for c in missing_cols)
        raise ValueError(
            f"Kolom berikut tidak ditemukan:\n\n{bullet}\n\n"
            "Kolom wajib: education_mother, education_father, age_years, gender"
        )
    return df


# ══════════════════════════════════════════════════════════════
# PREPROCESSING  (identik logika Gradio)
# ══════════════════════════════════════════════════════════════
def preprocessing(df: pd.DataFrame):
    data = df.copy()

    # ubah string kosong → NaN
    data = data.replace(r"^\s*$", np.nan, regex=True)

    jumlah_sebelum = len(data)
    data = data.dropna(subset=REQUIRED_COLUMNS)
    jumlah_sesudah = len(data)
    jumlah_terhapus = jumlah_sebelum - jumlah_sesudah

    if len(data) == 0:
        raise ValueError("Semua data terhapus karena missing value.")

    # bersihkan string
    for col in ["education_mother", "education_father"]:
        data[col] = (
            data[col].astype(str)
            .str.strip().str.lower()
            .str.replace(r"\s+", " ", regex=True)
        )
    data["gender"] = data["gender"].astype(str).str.strip().str.lower()

    # konversi age
    data["age_years"] = pd.to_numeric(data["age_years"], errors="coerce")

    # encoding
    data["education_mother"] = data["education_mother"].map(EDU_MAP_ENCODE)
    data["education_father"] = data["education_father"].map(EDU_MAP_ENCODE)
    data["gender"]            = data["gender"].map(GENDER_MAP)

    # validasi encoding
    errors = []
    if data["education_mother"].isnull().any():
        errors.append(
            "Kategori **education_mother** tidak valid.\n\n"
            "Nilai yang diperbolehkan:\n"
            "• primary or lower secondary\n• vocational\n• secondary\n• higher"
        )
    if data["education_father"].isnull().any():
        errors.append(
            "Kategori **education_father** tidak valid.\n\n"
            "Nilai yang diperbolehkan:\n"
            "• primary or lower secondary\n• vocational\n• secondary\n• higher"
        )
    if data["gender"].isnull().any():
        errors.append(
            "Kategori **gender** tidak valid.\n\n"
            "Nilai yang diperbolehkan:\n• male\n• female"
        )
    if errors:
        raise ValueError("\n\n---\n\n".join(errors))

    return data, jumlah_terhapus


# ══════════════════════════════════════════════════════════════
# BULK PREDICTION
# ══════════════════════════════════════════════════════════════
def bulk_prediction(model, scaler, df_raw: pd.DataFrame):
    """
    df_raw: DataFrame mentah dari CSV yang sudah divalidasi kolom.
    Returns: hasil_df, jumlah_terhapus
    """
    data, jumlah_terhapus = preprocessing(df_raw)

    X = data[REQUIRED_COLUMNS].copy()
    X_scaled = scaler.transform(X)

    pred  = model.predict(X_scaled)
    proba = model.predict_proba(X_scaled)
    confidence = np.max(proba, axis=1) * 100

    hasil_label = [IQ_LABELS[i] for i in pred]

    hasil_df = pd.DataFrame({
        "education_mother"     : df_raw.loc[data.index, "education_mother"],
        "education_father"     : df_raw.loc[data.index, "education_father"],
        "age_years"            : df_raw.loc[data.index, "age_years"],
        "gender"               : df_raw.loc[data.index, "gender"],
        "predicted_iq_category": hasil_label,
        "confidence_score"     : [f"{x:.2f}%" for x in confidence],
    })

    return hasil_df, jumlah_terhapus, confidence


# ══════════════════════════════════════════════════════════════
# CHART HELPERS
# ══════════════════════════════════════════════════════════════
def single_prob_chart(proba: np.ndarray):
    labels = [f"Kelas {i}" for i in range(5)]
    colors = [IQ_COLORS[i] for i in range(5)]
    fig = go.Figure(go.Bar(
        x=labels, y=proba * 100,
        marker_color=colors,
        text=[f"{v*100:.1f}%" for v in proba],
        textposition="outside",
        textfont=dict(family="Sora", size=11, color="#374151"),
    ))
    fig.update_layout(
        height=300, margin=dict(l=0, r=0, t=10, b=40),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        yaxis=dict(title="Probabilitas (%)", range=[0, max(proba)*120+5],
                   gridcolor="#F3F4F6", tickfont=dict(family="Sora", size=10)),
        xaxis=dict(tickfont=dict(family="Sora", size=10)),
        font=dict(family="Sora"), showlegend=False,
    )
    return fig


def bulk_dist_chart(hasil_df: pd.DataFrame):
    order  = [IQ_LABELS[i] for i in range(5)]
    counts = hasil_df["predicted_iq_category"].value_counts().reindex(order, fill_value=0)
    colors = [IQ_COLORS[i] for i in range(5)]

    fig = go.Figure(go.Bar(
        x=list(counts.index),
        y=list(counts.values),
        marker_color=colors,
        text=list(counts.values),
        textposition="outside",
        textfont=dict(family="Sora", size=12, color="#374151"),
    ))
    fig.update_layout(
        height=360, margin=dict(l=0, r=0, t=20, b=80),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        yaxis=dict(title="Jumlah Data", gridcolor="#F3F4F6",
                   tickfont=dict(family="Sora", size=10)),
        xaxis=dict(tickangle=-15, tickfont=dict(family="Sora", size=10)),
        font=dict(family="Sora"), showlegend=False,
    )
    return fig


# ══════════════════════════════════════════════════════════════
# TEMPLATE CSV
# ══════════════════════════════════════════════════════════════
TEMPLATE_CSV = (
    "education_mother,education_father,age_years,gender\n"
    "secondary,vocational,10,male\n"
    "higher,higher,15,female\n"
    "primary or lower secondary,secondary,8,male\n"
    "vocational,vocational,12,female\n"
)


# ══════════════════════════════════════════════════════════════
# HALAMAN 1 — PREDIKSI TUNGGAL
# ══════════════════════════════════════════════════════════════
def page_single(model, scaler):
    result_data = st.session_state.get("single_result")

    left, right = st.columns([1, 1.05])

    with left:
        st.markdown('<div class="section-card">'
                    '<div class="section-title">📋 Informasi Keluarga</div>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            edu_mother_lbl = st.selectbox("Pendidikan Ibu", list(EDU_MAP_UI), key="s_edu_m")
        with c2:
            edu_father_lbl = st.selectbox("Pendidikan Ayah", list(EDU_MAP_UI), key="s_edu_f")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-card">'
                    '<div class="section-title">👶 Informasi Anak</div>',
                    unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            age = st.number_input("Usia Anak (tahun)", min_value=1, max_value=18,
                                  value=10, step=1, key="s_age")
        with c4:
            gender_lbl = st.selectbox("Jenis Kelamin", list(GENDER_MAP_UI), key="s_gender")
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("🔍 Prediksi Kategori IQ", key="btn_single"):
            edu_mother = EDU_MAP_UI[edu_mother_lbl]
            edu_father = EDU_MAP_UI[edu_father_lbl]
            gender     = GENDER_MAP_UI[gender_lbl]

            try:
                X = pd.DataFrame(
                    [[edu_mother, edu_father, int(age), gender]],
                    columns=REQUIRED_COLUMNS,
                )
                X_scaled   = scaler.transform(X)
                pred_class = int(model.predict(X_scaled)[0])
                proba      = model.predict_proba(X_scaled)[0]
                confidence = float(proba.max()) * 100
            except Exception as e:
                st.markdown(f'<div class="err-box">⚠️ Prediksi gagal: {e}</div>',
                            unsafe_allow_html=True)
                return

            st.session_state.single_result = {
                "pred_class": pred_class,
                "category": IQ_LABELS[pred_class],
                "range": IQ_RANGES[pred_class],
                "confidence": confidence,
                "proba": proba.tolist(),
            }
            result_data = st.session_state.single_result

    with right:
        st.markdown('<div class="section-card">'
                    '<div class="section-title">🎯 Hasil Prediksi</div>',
                    unsafe_allow_html=True)

        if result_data:
            pred_class = result_data["pred_class"]
            cat = result_data["category"]
            rng = result_data["range"]
            confidence = result_data["confidence"]
            proba = np.array(result_data["proba"])
            bg = IQ_COLORS[pred_class]

            st.markdown(f"""
            <div class="result-card" style="background:{bg};color:#fff;">
                <div class="result-label">Hasil Prediksi</div>
                <div class="result-category">{cat}</div>
                <div class="result-range">Rentang IQ: {rng}</div>
                <span class="result-confidence">Confidence: {confidence:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div style="margin-top:1.4rem;">', unsafe_allow_html=True)
            st.plotly_chart(single_prob_chart(proba), width='stretch')
            st.markdown('</div>', unsafe_allow_html=True)

            rows = ""
            for i in range(5):
                pct = proba[i] * 100
                w = int(pct * 1.2)
                bold = "font-weight:700;color:var(--primary);" if i == pred_class else ""
                fill = "var(--primary)" if i == pred_class else "#A5B4FC"
                rows += f"""
                <tr>
                    <td style=\"{bold}\">Kelas {i}</td>
                    <td style=\"{bold}\">{IQ_LABELS[i]}</td>
                    <td style=\"{bold}\">{IQ_RANGES[i]}</td>
                    <td>
                        <span class=\"prob-bar-wrap\">
                            <span class=\"prob-bar-fill\"
                                  style=\"width:{w}px;background:{fill};\"></span>
                        </span>
                        <span style=\"{bold}\">{pct:.2f}%</span>
                    </td>
                </tr>"""

            st.markdown(f"""
            <table class="prob-table">
              <thead><tr>
                <th>Kelas</th><th>Kategori</th>
                <th>Rentang IQ</th><th>Probabilitas</th>
              </tr></thead>
              <tbody>{rows}</tbody>
            </table>
            """, unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="info-box">Masukkan data di kiri dan tekan tombol prediksi untuk melihat hasil di sini.</div>',
                unsafe_allow_html=True,
            )

        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# HALAMAN 2 — PENGUJIAN DATASET
# ══════════════════════════════════════════════════════════════
def page_bulk(model, scaler):

    # ── Template download ──
    st.markdown('<div class="section-card">'
                '<div class="section-title">📥 Download Template Dataset</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        Unduh template CSV terlebih dahulu, isi data sesuai format, lalu upload kembali.<br><br>
        Nilai yang diperbolehkan:<br>
        &nbsp;• <code>education_mother / education_father</code>:
        <code>primary or lower secondary</code> |
        <code>vocational</code> | <code>secondary</code> | <code>higher</code><br>
        &nbsp;• <code>gender</code>: <code>male</code> | <code>female</code><br>
        &nbsp;• <code>age_years</code>: angka (1 – 18)
    </div>
    """, unsafe_allow_html=True)
    st.download_button(
        label="⬇️ Download Template CSV",
        data=TEMPLATE_CSV,
        file_name="template_input_iq.csv",
        mime="text/csv",
        key="dl_template",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Upload ──
    st.markdown('<div class="section-card">'
                '<div class="section-title">📂 Upload Dataset CSV</div>',
                unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Pilih file CSV dataset",
        type=["csv"],
        key="bulk_upload",
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded is None:
        st.markdown(
            '<div class="info-box">ℹ️ Belum ada file yang diupload. '
            'Silakan upload file CSV di atas untuk memulai prediksi.</div>',
            unsafe_allow_html=True,
        )
        return

    if st.button("⚡ Jalankan Prediksi Dataset", key="btn_bulk"):
        try:
            # baca CSV
            try:
                df_raw = pd.read_csv(
                    io.StringIO(uploaded.getvalue().decode("utf-8")),
                    sep=None, engine="python", decimal=",",
                )
            except Exception as e:
                st.markdown(
                    f'<div class="err-box">⚠️ Gagal membaca file CSV: {e}</div>',
                    unsafe_allow_html=True,
                )
                return

            # validasi kolom
            try:
                validasi_csv(df_raw)
            except ValueError as e:
                st.markdown(
                    f'<div class="err-box">⚠️ {str(e)}</div>',
                    unsafe_allow_html=True,
                )
                return

            # prediksi
            try:
                hasil_df, jumlah_terhapus, confidence_arr = bulk_prediction(
                    model, scaler, df_raw
                )
            except ValueError as e:
                st.markdown(
                    f'<div class="err-box">⚠️ {str(e)}</div>',
                    unsafe_allow_html=True,
                )
                return

        except Exception as e:
            st.markdown(
                f'<div class="err-box">⚠️ Terjadi kesalahan: {e}</div>',
                unsafe_allow_html=True,
            )
            return

        # ── SUCCESS banner ──
        st.markdown(
            f'<div class="ok-box">✅ Prediksi berhasil untuk '
            f'<strong>{len(hasil_df)}</strong> data.</div>',
            unsafe_allow_html=True,
        )

        # ── Stat cards ──
        avg_conf    = float(confidence_arr.mean())
        top_cat     = hasil_df["predicted_iq_category"].value_counts().idxmax()
        top_cat_short = top_cat.replace("Intellectual Disability", "ID").replace("Intelligence", "")

        st.markdown(f"""
        <div class="stat-grid">
            <div class="stat-card">
                <div class="stat-val">{len(hasil_df)}</div>
                <div class="stat-key">Data Diproses</div>
            </div>
            <div class="stat-card">
                <div class="stat-val">{jumlah_terhapus}</div>
                <div class="stat-key">Data Dihapus</div>
            </div>
            <div class="stat-card">
                <div class="stat-val">{avg_conf:.1f}%</div>
                <div class="stat-key">Confidence Rata-rata</div>
            </div>
            <div class="stat-card">
                <div class="stat-val" style="font-size:1rem;">{top_cat_short.strip()}</div>
                <div class="stat-key">Kategori Terbanyak</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Preview tabel ──
        st.markdown('<div class="section-card">'
                    '<div class="section-title">📋 Preview Hasil (10 Data Pertama)</div>',
                    unsafe_allow_html=True)
        st.dataframe(hasil_df.head(10), width='stretch', hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Distribusi chart ──
        st.markdown('<div class="section-card">'
                    '<div class="section-title">📊 Distribusi Kategori IQ</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(bulk_dist_chart(hasil_df), width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Download hasil ──
        csv_out = hasil_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Hasil Prediksi (CSV)",
            data=csv_out,
            file_name="hasil_prediksi_iq.csv",
            mime="text/csv",
            key="dl_hasil",
        )


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    ensure_state()
    apply_theme()

    # header
    st.markdown("""
    <div class="rapiq-header">
        <div class="rapiq-header-content">
            <div class="logo">🧠</div>
            <h1>RapIQ</h1>
            <p>Sistem Prediksi Kategori IQ Berbasis Artificial Intelligence<br>
            Menggunakan Multilayer Perceptron (MLP)</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # menu buttons
    menu_col1, menu_col2, menu_col3 = st.columns([1.1, 1.1, 0.8])
    with menu_col1:
        if st.button("🔍 Prediksi Tunggal", key="menu_single"):
            st.session_state.active_page = "single"
    with menu_col2:
        if st.button("📂 Pengujian Dataset", key="menu_bulk"):
            st.session_state.active_page = "bulk"
    with menu_col3:
        theme_left, theme_right = st.columns(2)
        if theme_left.button("☀️ Light", key="theme_light"):
            st.session_state.theme_mode = "light"
        if theme_right.button("🌙 Dark", key="theme_dark"):
            st.session_state.theme_mode = "dark"

    # load model
    model, scaler, errors = load_artifacts(MODEL_PATH, SCALER_PATH)
    if errors:
        for err in errors:
            st.markdown(f'<div class="err-box">⚠️ {err}</div>',
                        unsafe_allow_html=True)
        return

    if st.session_state.active_page == "single":
        page_single(model, scaler)
    else:
        page_bulk(model, scaler)


if __name__ == "__main__":
    main()