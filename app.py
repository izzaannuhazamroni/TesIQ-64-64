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
    page_title="RapIQ — IQ Category Prediction System",
    page_icon="🧠",
    layout="wide",
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

EDU_MAP_ENCODE = {
    "primary or lower secondary": 0,
    "vocational": 1,
    "secondary": 2,
    "higher": 3,
}

EDU_MAP_UI = {
    "SD / SMP (Primary or Lower Secondary)": 0,
    "Kejuruan (Vocational)": 1,
    "SMA (Secondary)": 2,
    "Perguruan Tinggi (Higher)": 3,
}

GENDER_MAP    = {"male": 1, "female": 0}
GENDER_MAP_UI = {"Laki-laki (Male)": 1, "Perempuan (Female)": 0}

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
    0: "#EF4444",
    1: "#F97316",
    2: "#F59E0B",
    3: "#10B981",
    4: "#0F8B8D",
}

IQ_BG = {
    0: "#FEF2F2",
    1: "#FFF7ED",
    2: "#FFFBEB",
    3: "#F0FDF4",
    4: "#F0FDFA",
}

IQ_BORDER = {
    0: "#FECACA",
    1: "#FED7AA",
    2: "#FDE68A",
    3: "#BBF7D0",
    4: "#99F6E4",
}

# ══════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,300;0,14..32,400;0,14..32,500;0,14..32,600;0,14..32,700;0,14..32,800&display=swap');

/* ─── Base ─── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
.stApp { background: #F8FBFC; }
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 0 !important;
    padding-bottom: 3rem !important;
    max-width: 1300px !important;
}

/* ─── Top Header ─── */
.rq-header {
    background: #FFFFFF;
    border-bottom: 1px solid #E2E8F0;
    padding: 1.1rem 2.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1.5rem;
    flex-wrap: wrap;
}
.rq-brand {
    display: flex;
    align-items: center;
    gap: 0.85rem;
}
.rq-logomark {
    width: 44px; height: 44px;
    background: linear-gradient(135deg, #0F8B8D 0%, #5FD3BC 100%);
    border-radius: 11px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.35rem;
    box-shadow: 0 4px 12px rgba(15,139,141,0.22);
    flex-shrink: 0;
}
.rq-brand-text h1 {
    font-size: 1.4rem; font-weight: 800;
    color: #1E293B; margin: 0; letter-spacing: -0.4px; line-height: 1.2;
}
.rq-brand-text .sub {
    font-size: 0.72rem; font-weight: 600;
    color: #0F8B8D; margin: 0;
    letter-spacing: 0.5px; text-transform: uppercase;
}
.rq-disclaimer {
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    border-radius: 8px;
    padding: 0.55rem 0.9rem;
    font-size: 0.72rem; color: #92400E;
    line-height: 1.5; max-width: 440px;
    text-align: right;
}

/* ─── Tabs ─── */
.stTabs [data-baseweb="tab-list"] {
    background: #FFFFFF;
    border-bottom: 1px solid #E2E8F0;
    border-radius: 0;
    padding: 0 2.5rem;
    gap: 0;
    margin-bottom: 0;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 0;
    padding: 0.95rem 1.4rem;
    font-size: 0.875rem; font-weight: 600;
    color: #64748B;
    border: none;
    border-bottom: 2px solid transparent;
    background: transparent;
    margin-bottom: -1px;
    transition: color 0.15s;
}
.stTabs [aria-selected="true"] {
    background: transparent !important;
    color: #0F8B8D !important;
    font-weight: 700 !important;
    border-bottom: 2px solid #0F8B8D !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding: 2rem 2.5rem !important;
}

/* ─── Section Card ─── */
.rq-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 16px rgba(0,0,0,0.03);
}
.rq-card-header {
    display: flex; align-items: center; gap: 0.65rem;
    margin-bottom: 1.25rem;
    padding-bottom: 0.85rem;
    border-bottom: 1px solid #F1F5F9;
}
.rq-card-icon {
    width: 34px; height: 34px;
    border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; flex-shrink: 0;
}
.icon-teal  { background: #F0FDFA; }
.icon-blue  { background: #EFF6FF; }
.icon-amber { background: #FFFBEB; }
.icon-green { background: #F0FDF4; }
.icon-slate { background: #F8FAFC; }

.rq-card-title {
    font-size: 0.92rem; font-weight: 700;
    color: #1E293B; margin: 0; letter-spacing: -0.2px;
}
.rq-card-sub {
    font-size: 0.72rem; color: #94A3B8;
    margin: 0; font-weight: 400;
}

/* ─── Form inputs ─── */
.stSelectbox label, .stNumberInput label {
    font-size: 0.75rem !important; font-weight: 600 !important;
    color: #475569 !important; text-transform: uppercase !important;
    letter-spacing: 0.5px !important; margin-bottom: 0.3rem !important;
}
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    border: 1.5px solid #E2E8F0 !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.875rem !important;
    color: #1E293B !important;
    background: #FAFBFC !important;
}
.stSelectbox > div > div:focus-within,
.stNumberInput > div > div:focus-within {
    border-color: #0F8B8D !important;
    box-shadow: 0 0 0 3px rgba(15,139,141,0.1) !important;
    background: #FFFFFF !important;
}

/* ─── Buttons ─── */
.stButton > button {
    width: 100%;
    background: #0F8B8D; color: #ffffff;
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem; font-weight: 700;
    border: none; border-radius: 12px;
    padding: 0.9rem 1.5rem; cursor: pointer;
    letter-spacing: 0.2px;
    box-shadow: 0 4px 14px rgba(15,139,141,0.28);
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: #0d7a7c;
    box-shadow: 0 6px 20px rgba(15,139,141,0.38);
    transform: translateY(-1px);
}
.stButton > button:active { transform: translateY(0); }

/* ─── Download button ─── */
.stDownloadButton > button {
    width: 100%;
    background: #FFFFFF; color: #0F8B8D;
    font-family: 'Inter', sans-serif;
    font-size: 0.875rem; font-weight: 600;
    border: 1.5px solid #0F8B8D; border-radius: 12px;
    padding: 0.8rem 1.2rem; cursor: pointer;
    transition: all 0.2s ease;
}
.stDownloadButton > button:hover {
    background: #F0FDFA; border-color: #0d7a7c; color: #0d7a7c;
}

/* ─── Awaiting placeholder ─── */
.rq-placeholder {
    border: 1.5px dashed #CBD5E1;
    border-radius: 16px;
    padding: 3.5rem 2rem;
    text-align: center;
    background: #FAFBFC;
}

/* ─── Result card ─── */
.rq-result {
    border-radius: 16px;
    padding: 1.6rem 1.75rem;
    margin-bottom: 1.25rem;
    border: 1.5px solid;
    position: relative;
    overflow: hidden;
}
.rq-result-eyebrow {
    font-size: 0.68rem; font-weight: 700;
    letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 0.45rem;
}
.rq-result-cat {
    font-size: 1.7rem; font-weight: 800;
    letter-spacing: -0.5px; margin-bottom: 0.3rem; line-height: 1.2;
}
.rq-result-range { font-size: 0.88rem; font-weight: 500; opacity: 0.7; margin-bottom: 1rem; }
.rq-badge {
    display: inline-flex; align-items: center; gap: 0.35rem;
    border-radius: 999px; padding: 0.38rem 0.9rem;
    font-size: 0.8rem; font-weight: 700; border: 1.5px solid;
    margin-right: 0.5rem; margin-bottom: 0.4rem;
}
.rq-interp {
    font-size: 0.82rem; line-height: 1.65; margin-top: 0.9rem;
    padding-top: 0.9rem; border-top: 1px solid; opacity: 0.78; color: #475569;
}

/* ─── IQ Spectrum ─── */
.rq-spectrum {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 1.4rem 1.75rem;
    margin-bottom: 1.25rem;
}
.rq-spectrum-title {
    font-size: 0.68rem; font-weight: 700;
    letter-spacing: 1.5px; text-transform: uppercase;
    color: #64748B; margin-bottom: 1.2rem;
}
.rq-spectrum-track {
    position: relative; height: 10px; border-radius: 999px;
    background: linear-gradient(90deg,
        #EF4444 0%, #F97316 25%, #F59E0B 50%, #10B981 75%, #0F8B8D 100%);
    margin-bottom: 0.55rem;
}
.rq-spectrum-dot {
    position: absolute; top: -7px;
    width: 24px; height: 24px;
    border-radius: 50%; border: 3px solid #FFFFFF;
    box-shadow: 0 2px 8px rgba(0,0,0,0.22);
    transform: translateX(-50%);
    transition: left 0.4s ease;
}
.rq-spectrum-labels {
    display: flex; justify-content: space-between;
    font-size: 0.63rem; font-weight: 600;
    color: #94A3B8; text-transform: uppercase; letter-spacing: 0.2px;
    margin-bottom: 0.2rem;
}
.rq-spectrum-ranges {
    display: flex; justify-content: space-between;
    font-size: 0.6rem; color: #CBD5E1;
}

/* ─── Prob chart container ─── */
.rq-prob-chart-card {
    background: #FFFFFF; border: 1px solid #E2E8F0;
    border-radius: 16px; padding: 1.4rem 1.75rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

/* ─── Prob details table ─── */
.rq-prob-table {
    width: 100%; border-collapse: collapse;
    font-size: 0.84rem; margin-top: 0.5rem;
}
.rq-prob-table th {
    text-align: left; color: #64748B; font-size: 0.7rem;
    font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;
    padding: 0.6rem 0.75rem; border-bottom: 1px solid #E2E8F0;
}
.rq-prob-table td {
    padding: 0.65rem 0.75rem; border-bottom: 1px solid #F1F5F9;
    color: #334155; font-size: 0.84rem;
}
.rq-prob-table tr:last-child td { border-bottom: none; }
.rq-active-row td { background: #F0FDFA; font-weight: 700; }
.rq-dot {
    display: inline-block; width: 10px; height: 10px;
    border-radius: 50%; margin-right: 0.4rem; vertical-align: middle;
}

/* ─── Step indicator ─── */
.rq-steps {
    display: flex; align-items: flex-start; gap: 0;
    padding: 1.25rem 1.5rem;
    background: #FFFFFF; border: 1px solid #E2E8F0;
    border-radius: 16px; margin-bottom: 1.75rem;
    overflow-x: auto;
}
.rq-step {
    display: flex; flex-direction: column; align-items: center;
    flex: 1; min-width: 72px; position: relative;
}
.rq-step:not(:last-child)::after {
    content: ''; position: absolute; top: 13px;
    left: 58%; width: 84%; height: 1.5px;
    background: #E2E8F0; z-index: 0;
}
.rq-step.done:not(:last-child)::after { background: #0F8B8D; }
.rq-step-circle {
    width: 28px; height: 28px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.7rem; font-weight: 800; margin-bottom: 0.5rem;
    position: relative; z-index: 1;
    border: 2px solid #E2E8F0; background: #FFFFFF; color: #94A3B8;
}
.rq-step.done  .rq-step-circle { background:#0F8B8D; border-color:#0F8B8D; color:#fff; }
.rq-step.active .rq-step-circle { background:#F6AE2D; border-color:#F6AE2D; color:#fff; }
.rq-step-label {
    font-size: 0.6rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.3px;
    color: #94A3B8; text-align: center; line-height: 1.4;
}
.rq-step.done  .rq-step-label { color: #0F8B8D; }
.rq-step.active .rq-step-label { color: #D97706; }

/* ─── Stat cards ─── */
.rq-stat-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
.rq-stat {
    flex: 1; min-width: 120px;
    background: #FFFFFF; border: 1px solid #E2E8F0;
    border-radius: 14px; padding: 1.1rem 1.25rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.rq-stat-icon { font-size: 1.1rem; margin-bottom: 0.45rem; }
.rq-stat-val {
    font-size: 1.55rem; font-weight: 800; color: #1E293B;
    line-height: 1.1; margin-bottom: 0.2rem; letter-spacing: -0.5px;
}
.rq-stat-key {
    font-size: 0.7rem; font-weight: 600; color: #94A3B8;
    text-transform: uppercase; letter-spacing: 0.5px;
}

/* ─── Upload guidance ─── */
.rq-col-guide {
    background: #FAFBFC; border: 1px solid #E2E8F0;
    border-radius: 12px; padding: 1.1rem 1.25rem; margin-bottom: 1.25rem;
}
.rq-col-guide-title {
    font-size: 0.7rem; font-weight: 700; color: #475569;
    text-transform: uppercase; letter-spacing: 0.6px; margin-bottom: 0.75rem;
}
.rq-col-row {
    display: flex; align-items: flex-start; gap: 0.7rem;
    margin-bottom: 0.55rem; font-size: 0.8rem;
}
.rq-col-name {
    font-family: 'Inter', monospace; font-size: 0.72rem; font-weight: 700;
    color: #0F8B8D; background: #F0FDFA; border: 1px solid #99F6E4;
    border-radius: 6px; padding: 0.2rem 0.55rem;
    white-space: nowrap; flex-shrink: 0;
}
.rq-col-vals { font-size: 0.75rem; color: #64748B; line-height: 1.6; }

/* ─── Info / success / error boxes ─── */
.rq-info {
    background: #F0FDFA; border: 1px solid #99F6E4;
    border-radius: 12px; padding: 0.9rem 1.15rem;
    font-size: 0.84rem; color: #0F766E;
    margin-bottom: 1.25rem; line-height: 1.6;
    display: flex; align-items: flex-start; gap: 0.55rem;
}
.rq-err {
    background: #FEF2F2; border: 1px solid #FECACA;
    border-radius: 12px; padding: 0.9rem 1.15rem;
    color: #B91C1C; font-size: 0.875rem;
    margin-bottom: 1rem; line-height: 1.6;
    display: flex; align-items: flex-start; gap: 0.55rem;
}
.rq-ok {
    background: #F0FDF4; border: 1px solid #BBF7D0;
    border-radius: 12px; padding: 0.9rem 1.15rem;
    color: #166534; font-size: 0.875rem; font-weight: 500;
    margin-bottom: 1.25rem;
    display: flex; align-items: center; gap: 0.55rem;
}

/* ─── File uploader ─── */
[data-testid="stFileUploader"] {
    border: 2px dashed #CBD5E1 !important;
    border-radius: 14px !important;
    background: #FAFBFC !important;
    padding: 0.75rem !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #0F8B8D !important;
    background: #F0FDFA !important;
}

/* ─── Dataframe ─── */
[data-testid="stDataFrame"] {
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* ─── Divider ─── */
.rq-hr { height:1px; background:#F1F5F9; margin: 0.2rem 0 1.1rem; }

/* ─── Section label ─── */
.rq-section-label {
    font-size: 0.7rem; font-weight: 700; color: #94A3B8;
    text-transform: uppercase; letter-spacing: 0.8px;
    margin: 0.85rem 0 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# LOAD MODEL (cached)
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading model…")
def load_artifacts(model_path: str, scaler_path: str):
    errors, model, scaler = [], None, None

    if not os.path.exists(model_path):
        errors.append(f"Model file not found: <code>{model_path}</code>")
    else:
        try:
            model = joblib.load(model_path)
        except Exception as e:
            errors.append(f"Failed to load model: {e}")

    if not os.path.exists(scaler_path):
        errors.append(f"Scaler file not found: <code>{scaler_path}</code>")
    else:
        try:
            scaler = joblib.load(scaler_path)
        except Exception as e:
            errors.append(f"Failed to load scaler: {e}")

    return model, scaler, errors


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
# PREPROCESSING
# ══════════════════════════════════════════════════════════════
def preprocessing(df: pd.DataFrame):
    data = df.copy()

    data = data.replace(r"^\s*$", np.nan, regex=True)

    jumlah_sebelum = len(data)
    data = data.dropna(subset=REQUIRED_COLUMNS)
    jumlah_sesudah = len(data)
    jumlah_terhapus = jumlah_sebelum - jumlah_sesudah

    if len(data) == 0:
        raise ValueError("Semua data terhapus karena missing value.")

    for col in ["education_mother", "education_father"]:
        data[col] = (
            data[col].astype(str)
            .str.strip().str.lower()
            .str.replace(r"\s+", " ", regex=True)
        )
    data["gender"] = data["gender"].astype(str).str.strip().str.lower()

    data["age_years"] = pd.to_numeric(data["age_years"], errors="coerce")

    data["education_mother"] = data["education_mother"].map(EDU_MAP_ENCODE)
    data["education_father"] = data["education_father"].map(EDU_MAP_ENCODE)
    data["gender"]            = data["gender"].map(GENDER_MAP)

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
    """Original vertical Plotly bar chart for single prediction."""
    labels = [f"Class {i}" for i in range(5)]
    colors = [IQ_COLORS[i] for i in range(5)]

    fig = go.Figure(go.Bar(
        x=labels,
        y=proba * 100,
        marker_color=colors,
        marker_line_width=0,
        text=[f"{v*100:.1f}%" for v in proba],
        textposition="outside",
        textfont=dict(family="Inter", size=11, color="#374151"),
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=18, b=40),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        yaxis=dict(
            title="Probability (%)",
            range=[0, max(proba) * 120 + 5],
            gridcolor="#F1F5F9",
            tickfont=dict(family="Inter", size=10, color="#94A3B8"),
            title_font=dict(family="Inter", size=11, color="#64748B"),
            showgrid=True,
        ),
        xaxis=dict(tickfont=dict(family="Inter", size=10, color="#64748B")),
        font=dict(family="Inter"),
        showlegend=False,
        bargap=0.38,
    )
    return fig


def bulk_dist_chart(hasil_df: pd.DataFrame):
    """Original vertical Plotly bar chart for bulk distribution."""
    order  = [IQ_LABELS[i] for i in range(5)]
    counts = hasil_df["predicted_iq_category"].value_counts().reindex(order, fill_value=0)
    colors = [IQ_COLORS[i] for i in range(5)]

    fig = go.Figure(go.Bar(
        x=list(counts.index),
        y=list(counts.values),
        marker_color=colors,
        marker_line_width=0,
        text=list(counts.values),
        textposition="outside",
        textfont=dict(family="Inter", size=12, color="#374151"),
    ))
    fig.update_layout(
        height=360,
        margin=dict(l=0, r=0, t=20, b=90),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        yaxis=dict(
            title="Number of Records",
            gridcolor="#F1F5F9",
            tickfont=dict(family="Inter", size=10, color="#94A3B8"),
            title_font=dict(family="Inter", size=11, color="#64748B"),
        ),
        xaxis=dict(
            tickangle=-18,
            tickfont=dict(family="Inter", size=10, color="#64748B"),
        ),
        font=dict(family="Inter"),
        showlegend=False,
        bargap=0.35,
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
# IQ SPECTRUM COMPONENT
# ══════════════════════════════════════════════════════════════
def render_iq_spectrum(pred_class: int):
    positions = {0: 10, 1: 30, 2: 50, 3: 70, 4: 90}
    pos   = positions[pred_class]
    color = IQ_COLORS[pred_class]

    st.markdown(f"""
    <div class="rq-spectrum">
        <div class="rq-spectrum-title">⚡ IQ Spectrum — Predicted Position</div>
        <div style="position:relative;margin-bottom:0.6rem;">
            <div class="rq-spectrum-track">
                <div class="rq-spectrum-dot" style="left:{pos}%;background:{color};"></div>
            </div>
        </div>
        <div class="rq-spectrum-labels">
            <span>Moderate ID</span>
            <span>Mild ID</span>
            <span>Below Avg</span>
            <span>Average</span>
            <span>Above Avg</span>
        </div>
        <div class="rq-spectrum-ranges">
            <span>35–54</span><span>55–69</span><span>70–84</span>
            <span>85–114</span><span>114+</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# HALAMAN 1 — PREDIKSI TUNGGAL
# ══════════════════════════════════════════════════════════════
def page_single(model, scaler):
    col_left, col_right = st.columns([1, 1.2], gap="large")

    # ── LEFT PANEL ──
    with col_left:
        st.markdown("""
        <div class="rq-card">
            <div class="rq-card-header">
                <div class="rq-card-icon icon-teal">👨‍👩‍👦</div>
                <div>
                    <div class="rq-card-title">Patient / Child Information</div>
                    <div class="rq-card-sub">Assessment subject input data</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="rq-section-label">Parental Education</div>', unsafe_allow_html=True)
        edu_mother_lbl = st.selectbox("Mother's Education Level",  list(EDU_MAP_UI), key="s_edu_m")
        edu_father_lbl = st.selectbox("Father's Education Level",  list(EDU_MAP_UI), key="s_edu_f")

        st.markdown('<div class="rq-hr"></div><div class="rq-section-label">Child Information</div>', unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            age = st.number_input("Age (years)", min_value=1, max_value=18, value=10, step=1, key="s_age")
        with c4:
            gender_lbl = st.selectbox("Gender", list(GENDER_MAP_UI), key="s_gender")

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        predict_clicked = st.button("🔍  Predict IQ Category", key="btn_single")

    # ── RIGHT PANEL ──
    with col_right:
        if not predict_clicked:
            st.markdown("""
            <div class="rq-placeholder">
                <div style="font-size:2.8rem;margin-bottom:1rem;">🧠</div>
                <div style="font-size:1rem;font-weight:700;color:#1E293B;margin-bottom:0.5rem;">
                    Awaiting Assessment Input
                </div>
                <div style="font-size:0.82rem;color:#94A3B8;line-height:1.65;max-width:320px;margin:0 auto;">
                    Complete the child information form on the left, then click
                    <strong style="color:#0F8B8D;">Predict IQ Category</strong>
                    to generate the cognitive profile.
                </div>
            </div>
            """, unsafe_allow_html=True)
            return

        # ── Run prediction ──
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
            st.markdown(f'<div class="rq-err"><span>⚠️</span><span>Prediction failed: {e}</span></div>',
                        unsafe_allow_html=True)
            return

        cat    = IQ_LABELS[pred_class]
        rng    = IQ_RANGES[pred_class]
        color  = IQ_COLORS[pred_class]
        bg     = IQ_BG[pred_class]
        border = IQ_BORDER[pred_class]

        interpretations = {
            0: "The model indicates high probability that this individual falls within the Moderate Intellectual Disability range. Professional psychological evaluation is strongly recommended.",
            1: "The model suggests this individual may fall within the Mild Intellectual Disability range. A comprehensive clinical assessment is advised for confirmation.",
            2: "The model indicates this individual's cognitive profile is likely in the Below Average range. Targeted educational support strategies may be beneficial.",
            3: "The model indicates this individual falls within the Average IQ range, consistent with typical cognitive development for their age group.",
            4: "The model indicates a high probability that the individual belongs to the Above Average IQ category, suggesting strong cognitive capabilities.",
        }

        # Result card
        st.markdown(f"""
        <div class="rq-result" style="background:{bg};border-color:{border};">
            <div style="border-left:4px solid {color};padding-left:1rem;">
                <div class="rq-result-eyebrow" style="color:{color};">Prediction Result</div>
                <div class="rq-result-cat" style="color:#1E293B;">{cat}</div>
                <div class="rq-result-range">IQ Range: {rng}</div>
                <div style="margin-top:0.5rem;">
                    <span class="rq-badge" style="background:#FFFFFF;border-color:{border};color:{color};">
                        ✓ Confidence: {confidence:.1f}%
                    </span>
                    <span class="rq-badge" style="background:#FFFFFF;border-color:#E2E8F0;color:#64748B;">
                        MLP Model
                    </span>
                </div>
                <div class="rq-interp" style="border-color:{border};">
                    {interpretations[pred_class]}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # IQ Spectrum
        render_iq_spectrum(pred_class)

        # ── Probability Distribution (original Plotly bar chart) ──
        st.markdown("""
        <div class="rq-prob-chart-card">
            <div class="rq-card-header" style="margin-bottom:0.5rem;">
                <div class="rq-card-icon icon-teal">📊</div>
                <div>
                    <div class="rq-card-title">Probability Distribution</div>
                    <div class="rq-card-sub">MLP model output — all IQ classes</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(single_prob_chart(proba), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Probability Details Table ──
        st.markdown("""
        <div class="rq-card">
            <div class="rq-card-header">
                <div class="rq-card-icon icon-blue">📋</div>
                <div>
                    <div class="rq-card-title">Probability Details</div>
                    <div class="rq-card-sub">Full class probability breakdown</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        rows = ""
        for i in range(5):
            pct       = proba[i] * 100
            active    = "rq-active-row" if i == pred_class else ""
            dot_color = IQ_COLORS[i]
            rows += f"""
            <tr class="{active}">
                <td><span class="rq-dot" style="background:{dot_color};"></span>Class {i}</td>
                <td>{IQ_LABELS[i]}</td>
                <td style="font-family:'Inter',monospace;">{IQ_RANGES[i]}</td>
                <td style="font-weight:{'800' if i == pred_class else '400'};
                           color:{'#0F8B8D' if i == pred_class else '#334155'};">
                    {pct:.2f}%
                </td>
            </tr>"""

        st.markdown(f"""
        <table class="rq-prob-table">
            <thead>
                <tr>
                    <th>Class</th>
                    <th>Category</th>
                    <th>IQ Range</th>
                    <th>Probability</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# HALAMAN 2 — PENGUJIAN DATASET
# ══════════════════════════════════════════════════════════════
def page_bulk(model, scaler):

    # Step indicator
    st.markdown("""
    <div class="rq-steps">
        <div class="rq-step done">
            <div class="rq-step-circle">1</div>
            <div class="rq-step-label">Download<br>Template</div>
        </div>
        <div class="rq-step done">
            <div class="rq-step-circle">2</div>
            <div class="rq-step-label">Upload<br>Dataset</div>
        </div>
        <div class="rq-step active">
            <div class="rq-step-circle">3</div>
            <div class="rq-step-label">Run<br>Prediction</div>
        </div>
        <div class="rq-step">
            <div class="rq-step-circle">4</div>
            <div class="rq-step-label">Review<br>Results</div>
        </div>
        <div class="rq-step">
            <div class="rq-step-circle">5</div>
            <div class="rq-step-label">Download<br>Output</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 1], gap="large")

    # ── Template download ──
    with col_a:
        st.markdown("""
        <div class="rq-card">
            <div class="rq-card-header">
                <div class="rq-card-icon icon-amber">📥</div>
                <div>
                    <div class="rq-card-title">Step 1 — Download Template</div>
                    <div class="rq-card-sub">Get the required CSV format</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="rq-col-guide">
            <div class="rq-col-guide-title">Required Columns</div>
            <div class="rq-col-row">
                <span class="rq-col-name">education_mother</span>
                <span class="rq-col-vals">primary or lower secondary · vocational · secondary · higher</span>
            </div>
            <div class="rq-col-row">
                <span class="rq-col-name">education_father</span>
                <span class="rq-col-vals">primary or lower secondary · vocational · secondary · higher</span>
            </div>
            <div class="rq-col-row">
                <span class="rq-col-name">age_years</span>
                <span class="rq-col-vals">Numeric · range 1 to 18</span>
            </div>
            <div class="rq-col-row">
                <span class="rq-col-name">gender</span>
                <span class="rq-col-vals">male · female</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.download_button(
            label="⬇️  Download CSV Template",
            data=TEMPLATE_CSV,
            file_name="rapiq_template.csv",
            mime="text/csv",
            key="dl_template",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Dataset upload ──
    with col_b:
        st.markdown("""
        <div class="rq-card">
            <div class="rq-card-header">
                <div class="rq-card-icon icon-blue">📂</div>
                <div>
                    <div class="rq-card-title">Step 2 — Upload Dataset</div>
                    <div class="rq-card-sub">Drag & drop or click to browse</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Upload CSV dataset",
            type=["csv"],
            key="bulk_upload",
            label_visibility="collapsed",
        )

        if uploaded:
            st.markdown(f"""
            <div class="rq-ok" style="margin-top:0.75rem;">
                <span>✅</span>
                <span>File <strong>{uploaded.name}</strong> is ready for prediction.</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="rq-info" style="margin-top:0.75rem;">
                <span>ℹ️</span>
                <span>No file uploaded yet. Select a CSV file to begin bulk prediction.</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    if uploaded is None:
        return

    # ── Step 3: Run ──
    st.markdown("<br>", unsafe_allow_html=True)
    run_col, _ = st.columns([1, 2])
    with run_col:
        run_clicked = st.button("⚡  Run Prediction", key="btn_bulk")

    if not run_clicked:
        return

    # ── Process ──
    try:
        try:
            df_raw = pd.read_csv(
                io.StringIO(uploaded.getvalue().decode("utf-8")),
                sep=None, engine="python", decimal=",",
            )
        except Exception as e:
            st.markdown(f'<div class="rq-err"><span>⚠️</span><span>Failed to read CSV: {e}</span></div>', unsafe_allow_html=True)
            return

        try:
            validasi_csv(df_raw)
        except ValueError as e:
            st.markdown(f'<div class="rq-err"><span>⚠️</span><span>{str(e)}</span></div>', unsafe_allow_html=True)
            return

        try:
            hasil_df, jumlah_terhapus, confidence_arr = bulk_prediction(model, scaler, df_raw)
        except ValueError as e:
            st.markdown(f'<div class="rq-err"><span>⚠️</span><span>{str(e)}</span></div>', unsafe_allow_html=True)
            return

    except Exception as e:
        st.markdown(f'<div class="rq-err"><span>⚠️</span><span>An error occurred: {e}</span></div>', unsafe_allow_html=True)
        return

    # Success
    st.markdown(f"""
    <div class="rq-ok">
        <span>✅</span>
        <span>Prediction completed successfully for <strong>{len(hasil_df)}</strong> records.</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Step 4: Summary metrics ──
    avg_conf  = float(confidence_arr.mean())
    top_cat   = hasil_df["predicted_iq_category"].value_counts().idxmax()
    top_short = top_cat.replace("Intellectual Disability", "ID")

    st.markdown(f"""
    <div class="rq-stat-row">
        <div class="rq-stat">
            <div class="rq-stat-icon">📋</div>
            <div class="rq-stat-val">{len(hasil_df)}</div>
            <div class="rq-stat-key">Total Processed</div>
        </div>
        <div class="rq-stat">
            <div class="rq-stat-icon">🗑️</div>
            <div class="rq-stat-val">{jumlah_terhapus}</div>
            <div class="rq-stat-key">Removed Records</div>
        </div>
        <div class="rq-stat">
            <div class="rq-stat-icon">🎯</div>
            <div class="rq-stat-val">{avg_conf:.1f}%</div>
            <div class="rq-stat-key">Avg Confidence</div>
        </div>
        <div class="rq-stat">
            <div class="rq-stat-icon">🏆</div>
            <div class="rq-stat-val" style="font-size:0.88rem;letter-spacing:-0.1px;">{top_short}</div>
            <div class="rq-stat-key">Most Frequent</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Dataset distribution chart (original Plotly bar chart) ──
    st.markdown("""
    <div class="rq-card">
        <div class="rq-card-header">
            <div class="rq-card-icon icon-teal">📊</div>
            <div>
                <div class="rq-card-title">Prediction Category Distribution</div>
                <div class="rq-card-sub">Distribution across all IQ categories</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.plotly_chart(bulk_dist_chart(hasil_df), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Results preview ──
    st.markdown("""
    <div class="rq-card">
        <div class="rq-card-header">
            <div class="rq-card-icon icon-green">🔍</div>
            <div>
                <div class="rq-card-title">Results Preview</div>
                <div class="rq-card-sub">First 10 records from prediction output</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.dataframe(hasil_df.head(10), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Download ──
    st.markdown("""
    <div class="rq-card">
        <div class="rq-card-header">
            <div class="rq-card-icon icon-teal">⬇️</div>
            <div>
                <div class="rq-card-title">Step 5 — Download Prediction Results</div>
                <div class="rq-card-sub">Export full dataset with predicted categories</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    csv_out = hasil_df.to_csv(index=False).encode("utf-8")
    dl_col, _ = st.columns([1, 2])
    with dl_col:
        st.download_button(
            label="⬇️  Download Prediction Results (CSV)",
            data=csv_out,
            file_name="rapiq_prediction_results.csv",
            mime="text/csv",
            key="dl_hasil",
        )
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    # ── Compact professional header ──
    st.markdown("""
    <div class="rq-header">
        <div class="rq-brand">
            <div class="rq-logomark">🧠</div>
            <div class="rq-brand-text">
                <h1>RapIQ</h1>
                <p class="sub">IQ Category Prediction System &nbsp;·&nbsp; Multilayer Perceptron (MLP)</p>
            </div>
        </div>
        <div class="rq-disclaimer">
            ⚠️ This system supports educational and analytical purposes and
            <strong>does not replace professional psychological assessment.</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Load model ──
    model, scaler, errors = load_artifacts(MODEL_PATH, SCALER_PATH)
    if errors:
        st.markdown("<div style='padding:1.5rem 2.5rem;'>", unsafe_allow_html=True)
        for err in errors:
            st.markdown(f'<div class="rq-err"><span>⚠️</span><span>{err}</span></div>',
                        unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # ── Navigation tabs ──
    tab1, tab2 = st.tabs(["🔍  Single Prediction", "📂  Bulk Prediction — Dataset Testing"])

    with tab1:
        page_single(model, scaler)

    with tab2:
        page_bulk(model, scaler)


if __name__ == "__main__":
    main()