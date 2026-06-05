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

GENDER_MAP = {"male": 1, "female": 0}
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

IQ_BG_COLORS = {
    0: "#FEF2F2",
    1: "#FFF7ED",
    2: "#FFFBEB",
    3: "#F0FDF4",
    4: "#F0FDFA",
}

IQ_BORDER_COLORS = {
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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp {
    background: #F8FBFC;
}

#MainMenu, footer, header { visibility: hidden; }

.block-container {
    padding-top: 0 !important;
    padding-bottom: 3rem !important;
    max-width: 1280px !important;
}

/* ── TOP HEADER BANNER ── */
.rapiq-header {
    background: #FFFFFF;
    border-bottom: 1px solid #E2E8F0;
    padding: 1.5rem 2.5rem 1.2rem;
    margin-bottom: 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
}

.rapiq-header-left {
    display: flex;
    align-items: center;
    gap: 1rem;
}

.rapiq-logo-mark {
    width: 48px;
    height: 48px;
    background: linear-gradient(135deg, #0F8B8D 0%, #5FD3BC 100%);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    flex-shrink: 0;
    box-shadow: 0 4px 12px rgba(15,139,141,0.25);
}

.rapiq-header-titles h1 {
    font-size: 1.5rem;
    font-weight: 800;
    color: #1E293B;
    margin: 0;
    letter-spacing: -0.5px;
    line-height: 1.2;
}

.rapiq-header-titles .subtitle {
    font-size: 0.78rem;
    font-weight: 500;
    color: #0F8B8D;
    margin: 0;
    letter-spacing: 0.3px;
    text-transform: uppercase;
}

.rapiq-disclaimer {
    background: #F0FDFA;
    border: 1px solid #99F6E4;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    font-size: 0.72rem;
    color: #0F766E;
    line-height: 1.5;
    max-width: 480px;
    text-align: right;
}

/* ── CONTENT WRAPPER ── */
.content-wrapper {
    padding: 1.5rem 2rem;
}

/* ── SECTION CARD ── */
.section-card {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1.25rem;
    border: 1px solid #E2E8F0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 16px rgba(0,0,0,0.04);
}

.card-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 1.25rem;
    padding-bottom: 0.85rem;
    border-bottom: 1px solid #F1F5F9;
}

.card-icon {
    width: 32px;
    height: 32px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
}

.card-icon-teal { background: #F0FDFA; }
.card-icon-blue { background: #EFF6FF; }
.card-icon-amber { background: #FFFBEB; }
.card-icon-green { background: #F0FDF4; }

.card-title {
    font-size: 0.9rem;
    font-weight: 700;
    color: #1E293B;
    margin: 0;
    letter-spacing: -0.2px;
}

.card-subtitle {
    font-size: 0.72rem;
    color: #94A3B8;
    margin: 0;
    font-weight: 400;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: #FFFFFF;
    border-bottom: 1px solid #E2E8F0;
    border-radius: 0;
    padding: 0 2rem;
    gap: 0;
    margin-bottom: 0;
    box-shadow: none;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 0;
    padding: 1rem 1.5rem;
    font-size: 0.875rem;
    font-weight: 600;
    color: #64748B;
    border: none;
    border-bottom: 2px solid transparent;
    background: transparent;
    margin-bottom: -1px;
}

.stTabs [aria-selected="true"] {
    background: transparent !important;
    color: #0F8B8D !important;
    font-weight: 700 !important;
    border-bottom: 2px solid #0F8B8D !important;
}

.stTabs [data-baseweb="tab-panel"] {
    padding: 1.75rem 2rem;
}

/* ── FORM INPUTS ── */
.stSelectbox label,
.stNumberInput label {
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    color: #475569 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    margin-bottom: 0.35rem !important;
}

.stSelectbox > div > div,
.stNumberInput > div > div > input {
    border: 1.5px solid #E2E8F0 !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.875rem !important;
    color: #1E293B !important;
    background: #FAFBFC !important;
    transition: border-color 0.15s ease !important;
}

.stSelectbox > div > div:focus-within,
.stNumberInput > div > div:focus-within {
    border-color: #0F8B8D !important;
    box-shadow: 0 0 0 3px rgba(15,139,141,0.1) !important;
    background: #FFFFFF !important;
}

/* ── PREDICT BUTTON ── */
.stButton > button {
    width: 100%;
    background: #0F8B8D;
    color: #ffffff;
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem;
    font-weight: 700;
    border: none;
    border-radius: 12px;
    padding: 0.9rem 1.5rem;
    cursor: pointer;
    letter-spacing: 0.2px;
    box-shadow: 0 4px 14px rgba(15,139,141,0.3);
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background: #0d7a7c;
    box-shadow: 0 6px 20px rgba(15,139,141,0.4);
    transform: translateY(-1px);
}

.stButton > button:active {
    transform: translateY(0px);
}

/* ── DOWNLOAD BUTTON ── */
.stDownloadButton > button {
    width: 100%;
    background: #FFFFFF;
    color: #0F8B8D;
    font-family: 'Inter', sans-serif;
    font-size: 0.875rem;
    font-weight: 600;
    border: 1.5px solid #0F8B8D;
    border-radius: 12px;
    padding: 0.8rem 1.2rem;
    cursor: pointer;
    transition: all 0.2s ease;
}

.stDownloadButton > button:hover {
    background: #F0FDFA;
    border-color: #0d7a7c;
    color: #0d7a7c;
}

/* ── RESULT SECTION ── */
.result-main-card {
    border-radius: 16px;
    padding: 1.75rem 2rem;
    margin-bottom: 1.25rem;
    border: 1.5px solid;
    position: relative;
    overflow: hidden;
}

.result-main-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 4px; height: 100%;
}

.result-eyebrow {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

.result-category-name {
    font-size: 1.75rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin-bottom: 0.35rem;
    line-height: 1.2;
}

.result-iq-range {
    font-size: 0.9rem;
    font-weight: 500;
    margin-bottom: 1rem;
    opacity: 0.75;
}

.result-confidence-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    border-radius: 999px;
    padding: 0.4rem 1rem;
    font-size: 0.82rem;
    font-weight: 700;
    border: 1.5px solid;
    margin-right: 0.6rem;
    margin-bottom: 0.5rem;
}

.result-interpretation {
    font-size: 0.82rem;
    font-weight: 400;
    line-height: 1.6;
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid;
    opacity: 0.8;
}

/* ── IQ SPECTRUM ── */
.iq-spectrum-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

.spectrum-title {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #64748B;
    margin-bottom: 1.25rem;
}

.spectrum-track {
    position: relative;
    height: 8px;
    border-radius: 999px;
    background: linear-gradient(90deg, #EF4444 0%, #F97316 25%, #F59E0B 50%, #10B981 75%, #0F8B8D 100%);
    margin-bottom: 0.6rem;
}

.spectrum-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.65rem;
    font-weight: 600;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.3px;
    margin-bottom: 0.3rem;
}

.spectrum-ranges {
    display: flex;
    justify-content: space-between;
    font-size: 0.6rem;
    color: #CBD5E1;
    font-family: 'Inter', monospace;
}

.spectrum-indicator {
    position: absolute;
    top: -6px;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    border: 3px solid #FFFFFF;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    transform: translateX(-50%);
    transition: left 0.4s ease;
}

/* ── PROBABILITY BARS ── */
.prob-section {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1.25rem;
}

.prob-item {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.85rem;
    padding: 0.75rem 1rem;
    border-radius: 10px;
    transition: background 0.15s;
}

.prob-item:last-child { margin-bottom: 0; }
.prob-item.active-prob { background: #F0FDFA; }

.prob-label-col {
    width: 180px;
    flex-shrink: 0;
}

.prob-label-name {
    font-size: 0.78rem;
    font-weight: 600;
    color: #1E293B;
    line-height: 1.3;
}

.prob-label-range {
    font-size: 0.67rem;
    color: #94A3B8;
    font-family: 'Inter', monospace;
}

.prob-bar-outer {
    flex: 1;
    height: 8px;
    background: #F1F5F9;
    border-radius: 999px;
    overflow: hidden;
}

.prob-bar-inner {
    height: 100%;
    border-radius: 999px;
    transition: width 0.6s ease;
}

.prob-pct {
    width: 48px;
    text-align: right;
    font-size: 0.8rem;
    font-weight: 700;
    flex-shrink: 0;
}

/* ── STAT CARDS ── */
.stat-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
    flex-wrap: wrap;
}

.stat-card {
    flex: 1;
    min-width: 120px;
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 1.1rem 1.25rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

.stat-icon {
    font-size: 1.1rem;
    margin-bottom: 0.5rem;
}

.stat-value {
    font-size: 1.6rem;
    font-weight: 800;
    color: #1E293B;
    line-height: 1.1;
    margin-bottom: 0.25rem;
    letter-spacing: -0.5px;
}

.stat-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ── STEP INDICATOR ── */
.step-indicator {
    display: flex;
    align-items: flex-start;
    gap: 0;
    margin-bottom: 1.75rem;
    padding: 1.25rem 1.5rem;
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    overflow-x: auto;
}

.step-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    flex: 1;
    min-width: 80px;
    position: relative;
}

.step-item:not(:last-child)::after {
    content: '';
    position: absolute;
    top: 14px;
    left: 60%;
    width: 80%;
    height: 1.5px;
    background: #E2E8F0;
    z-index: 0;
}

.step-item.step-done:not(:last-child)::after {
    background: #0F8B8D;
}

.step-circle {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.72rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
    position: relative;
    z-index: 1;
    border: 2px solid #E2E8F0;
    background: #FFFFFF;
    color: #94A3B8;
}

.step-item.step-done .step-circle {
    background: #0F8B8D;
    border-color: #0F8B8D;
    color: #FFFFFF;
}

.step-item.step-active .step-circle {
    background: #F6AE2D;
    border-color: #F6AE2D;
    color: #FFFFFF;
}

.step-text {
    font-size: 0.62rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.3px;
    color: #94A3B8;
    text-align: center;
    line-height: 1.3;
}

.step-item.step-done .step-text { color: #0F8B8D; }
.step-item.step-active .step-text { color: #D97706; }

/* ── INFO / ERR / SUCCESS BOXES ── */
.info-box {
    background: #F0FDFA;
    border: 1px solid #99F6E4;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    font-size: 0.85rem;
    color: #0F766E;
    margin-bottom: 1.25rem;
    line-height: 1.6;
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
}

.err-box {
    background: #FEF2F2;
    border: 1px solid #FECACA;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    color: #B91C1C;
    font-size: 0.875rem;
    margin-bottom: 1rem;
    line-height: 1.6;
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
}

.ok-box {
    background: #F0FDF4;
    border: 1px solid #BBF7D0;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    color: #166534;
    font-size: 0.875rem;
    margin-bottom: 1.25rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-weight: 500;
}

/* ── TEMPLATE COLUMNS INFO ── */
.col-pill {
    display: inline-block;
    background: #F1F5F9;
    border: 1px solid #E2E8F0;
    border-radius: 6px;
    padding: 0.25rem 0.6rem;
    font-size: 0.72rem;
    font-family: 'Inter', monospace;
    font-weight: 600;
    color: #475569;
    margin: 0.2rem;
}

.col-required {
    background: #FFF7ED;
    border-color: #FED7AA;
    color: #92400E;
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
    border: 2px dashed #CBD5E1 !important;
    border-radius: 14px !important;
    background: #FAFBFC !important;
    padding: 1rem !important;
    transition: border-color 0.2s !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: #0F8B8D !important;
    background: #F0FDFA !important;
}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] {
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* ── DIVIDER ── */
.rapiq-divider {
    height: 1px;
    background: #F1F5F9;
    margin: 0.25rem 0 1.25rem;
}

/* ── UPLOAD GUIDANCE ── */
.upload-guidance {
    background: #FAFBFC;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 1.1rem 1.25rem;
    margin-bottom: 1.25rem;
}

.upload-guidance-title {
    font-size: 0.75rem;
    font-weight: 700;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 0.75rem;
}

.col-row {
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    margin-bottom: 0.6rem;
    font-size: 0.8rem;
}

.col-name {
    font-family: 'Inter', monospace;
    font-size: 0.75rem;
    font-weight: 700;
    color: #0F8B8D;
    background: #F0FDFA;
    border: 1px solid #99F6E4;
    border-radius: 6px;
    padding: 0.2rem 0.55rem;
    white-space: nowrap;
    flex-shrink: 0;
}

.col-values {
    font-size: 0.75rem;
    color: #64748B;
    line-height: 1.6;
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
def bulk_dist_chart(hasil_df: pd.DataFrame):
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
        height=340,
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
            tickangle=-20,
            tickfont=dict(family="Inter", size=10, color="#64748B"),
        ),
        font=dict(family="Inter"),
        showlegend=False,
        bargap=0.35,
    )
    return fig


def donut_chart(hasil_df: pd.DataFrame):
    order  = [IQ_LABELS[i] for i in range(5)]
    counts = hasil_df["predicted_iq_category"].value_counts().reindex(order, fill_value=0)
    colors = [IQ_COLORS[i] for i in range(5)]

    labels_short = ["Moderate ID", "Mild ID", "Below Avg", "Average", "Above Avg"]

    fig = go.Figure(go.Pie(
        labels=labels_short,
        values=list(counts.values),
        marker_colors=colors,
        hole=0.55,
        textinfo="percent",
        textfont=dict(family="Inter", size=11),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=0, r=0, t=10, b=10),
        paper_bgcolor="#FFFFFF",
        showlegend=True,
        legend=dict(
            font=dict(family="Inter", size=10, color="#64748B"),
            orientation="v",
            x=1.02, y=0.5,
        ),
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
    colors_spectrum = {0: "#EF4444", 1: "#F97316", 2: "#F59E0B", 3: "#10B981", 4: "#0F8B8D"}
    pos = positions[pred_class]
    color = colors_spectrum[pred_class]

    st.markdown(f"""
    <div class="iq-spectrum-card">
        <div class="spectrum-title">⚡ IQ Spectrum — Position Indicator</div>
        <div style="position: relative; margin-bottom: 0.5rem;">
            <div class="spectrum-track">
                <div class="spectrum-indicator" style="left: {pos}%; background: {color};"></div>
            </div>
        </div>
        <div class="spectrum-labels">
            <span>Moderate ID</span>
            <span>Mild ID</span>
            <span>Below Avg</span>
            <span>Average</span>
            <span>Above Avg</span>
        </div>
        <div class="spectrum-ranges">
            <span>35–54</span>
            <span>55–69</span>
            <span>70–84</span>
            <span>85–114</span>
            <span>114+</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# HALAMAN 1 — PREDIKSI TUNGGAL
# ══════════════════════════════════════════════════════════════
def page_single(model, scaler):
    col_left, col_right = st.columns([1, 1.15], gap="large")

    with col_left:
        # Patient Information Card
        st.markdown("""
        <div class="section-card">
            <div class="card-header">
                <div class="card-icon card-icon-teal">👨‍👩‍👦</div>
                <div>
                    <div class="card-title">Patient / Child Information</div>
                    <div class="card-subtitle">Enter assessment subject data</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown('<p style="font-size:0.72rem;font-weight:700;color:#94A3B8;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:0.5rem;">Parental Education</p>', unsafe_allow_html=True)

        edu_mother_lbl = st.selectbox(
            "Mother's Education Level",
            list(EDU_MAP_UI),
            key="s_edu_m",
        )
        edu_father_lbl = st.selectbox(
            "Father's Education Level",
            list(EDU_MAP_UI),
            key="s_edu_f",
        )

        st.markdown('<div class="rapiq-divider"></div>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:0.72rem;font-weight:700;color:#94A3B8;text-transform:uppercase;letter-spacing:0.8px;margin:0.75rem 0 0.5rem;">Child Information</p>', unsafe_allow_html=True)

        c3, c4 = st.columns(2)
        with c3:
            age = st.number_input(
                "Age (years)",
                min_value=1, max_value=18,
                value=10, step=1, key="s_age"
            )
        with c4:
            gender_lbl = st.selectbox("Gender", list(GENDER_MAP_UI), key="s_gender")

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        predict_clicked = st.button("🔍  Predict IQ Category", key="btn_single")

    with col_right:
        if not predict_clicked:
            # Placeholder state
            st.markdown("""
            <div class="section-card" style="text-align:center;padding:3rem 2rem;border-style:dashed;border-color:#CBD5E1;">
                <div style="font-size:2.5rem;margin-bottom:1rem;">🧠</div>
                <div style="font-size:1rem;font-weight:700;color:#1E293B;margin-bottom:0.5rem;">Awaiting Assessment</div>
                <div style="font-size:0.83rem;color:#94A3B8;line-height:1.6;">
                    Fill in the patient information on the left and click <strong>Predict IQ Category</strong> to generate the cognitive assessment result.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
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
                st.markdown(f'<div class="err-box">⚠️ Prediction failed: {e}</div>',
                            unsafe_allow_html=True)
                return

            cat   = IQ_LABELS[pred_class]
            rng   = IQ_RANGES[pred_class]
            color = IQ_COLORS[pred_class]
            bg    = IQ_BG_COLORS[pred_class]
            border = IQ_BORDER_COLORS[pred_class]

            interpretations = {
                0: "The model indicates high probability that this individual falls within the Moderate Intellectual Disability range. Professional psychological evaluation is strongly recommended.",
                1: "The model suggests this individual may fall within the Mild Intellectual Disability range. A comprehensive clinical assessment is advised for confirmation.",
                2: "The model indicates this individual's cognitive profile is likely in the Below Average range. Educational support strategies may be beneficial.",
                3: "The model indicates this individual falls within the Average IQ range, consistent with typical cognitive development for their age group.",
                4: "The model indicates high probability that this individual belongs to the Above Average IQ category, suggesting strong cognitive capabilities.",
            }

            # Main result card
            st.markdown(f"""
            <div class="result-main-card" style="background:{bg};border-color:{border};">
                <div style="border-left:4px solid {color};padding-left:1rem;">
                    <div class="result-eyebrow" style="color:{color};">Prediction Result</div>
                    <div class="result-category-name" style="color:#1E293B;">{cat}</div>
                    <div class="result-iq-range" style="color:#475569;">IQ Range: {rng}</div>
                    <div style="display:flex;align-items:center;flex-wrap:wrap;gap:0.5rem;margin-top:0.75rem;">
                        <div class="result-confidence-badge" style="background:#FFFFFF;border-color:{border};color:{color};">
                            <span>✓</span> Confidence: {confidence:.1f}%
                        </div>
                        <div class="result-confidence-badge" style="background:#FFFFFF;border-color:#E2E8F0;color:#64748B;">
                            MLP Model
                        </div>
                    </div>
                    <div class="result-interpretation" style="border-color:{border};color:#475569;">
                        {interpretations[pred_class]}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # IQ Spectrum
            render_iq_spectrum(pred_class)

            # Probability Distribution
            st.markdown("""
            <div class="prob-section">
                <div class="card-header" style="margin-bottom:0.75rem;">
                    <div class="card-icon card-icon-teal">📊</div>
                    <div>
                        <div class="card-title">Probability Distribution</div>
                        <div class="card-subtitle">Confidence across all IQ categories</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            for i in range(5):
                pct = proba[i] * 100
                bar_color = IQ_COLORS[i]
                is_active = (i == pred_class)
                active_cls = "active-prob" if is_active else ""
                pct_style = f"color:{bar_color};font-weight:800;" if is_active else "color:#94A3B8;"
                label_style = "font-weight:700;" if is_active else ""

                st.markdown(f"""
                <div class="prob-item {active_cls}">
                    <div class="prob-label-col">
                        <div class="prob-label-name" style="{label_style}">{IQ_LABELS[i]}</div>
                        <div class="prob-label-range">IQ {IQ_RANGES[i]}</div>
                    </div>
                    <div class="prob-bar-outer">
                        <div class="prob-bar-inner" style="width:{pct:.1f}%;background:{bar_color};"></div>
                    </div>
                    <div class="prob-pct" style="{pct_style}">{pct:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# HALAMAN 2 — PENGUJIAN DATASET
# ══════════════════════════════════════════════════════════════
def page_bulk(model, scaler):

    # Step Indicator
    st.markdown("""
    <div class="step-indicator">
        <div class="step-item step-done">
            <div class="step-circle">1</div>
            <div class="step-text">Download<br>Template</div>
        </div>
        <div class="step-item step-done">
            <div class="step-circle">2</div>
            <div class="step-text">Upload<br>Dataset</div>
        </div>
        <div class="step-item step-active">
            <div class="step-circle">3</div>
            <div class="step-text">Run<br>Prediction</div>
        </div>
        <div class="step-item">
            <div class="step-circle">4</div>
            <div class="step-text">Review<br>Results</div>
        </div>
        <div class="step-item">
            <div class="step-circle">5</div>
            <div class="step-text">Download<br>Output</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 1], gap="large")

    with col_a:
        # Template download card
        st.markdown("""
        <div class="section-card">
            <div class="card-header">
                <div class="card-icon card-icon-amber">📥</div>
                <div>
                    <div class="card-title">Step 1 — Download Template</div>
                    <div class="card-subtitle">Get the required CSV format</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="upload-guidance">
            <div class="upload-guidance-title">Required Columns</div>
            <div class="col-row">
                <span class="col-name">education_mother</span>
                <span class="col-values">primary or lower secondary · vocational · secondary · higher</span>
            </div>
            <div class="col-row">
                <span class="col-name">education_father</span>
                <span class="col-values">primary or lower secondary · vocational · secondary · higher</span>
            </div>
            <div class="col-row">
                <span class="col-name">age_years</span>
                <span class="col-values">Numeric value, 1–18</span>
            </div>
            <div class="col-row">
                <span class="col-name">gender</span>
                <span class="col-values">male · female</span>
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

    with col_b:
        # Upload card
        st.markdown("""
        <div class="section-card">
            <div class="card-header">
                <div class="card-icon card-icon-blue">📂</div>
                <div>
                    <div class="card-title">Step 2 — Upload Dataset</div>
                    <div class="card-subtitle">Drag & drop or click to browse</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Upload your CSV file",
            type=["csv"],
            key="bulk_upload",
            label_visibility="collapsed",
        )

        if uploaded:
            st.markdown(f"""
            <div class="ok-box" style="margin-top:0.75rem;">
                <span>✅</span>
                <span>File <strong>{uploaded.name}</strong> ready for prediction.</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box" style="margin-top:0.75rem;">
                <span>ℹ️</span>
                <span>No file uploaded yet. Please select a CSV file above to begin.</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    if uploaded is None:
        return

    # Run prediction button
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
            st.markdown(f'<div class="err-box"><span>⚠️</span><span>Failed to read CSV: {e}</span></div>', unsafe_allow_html=True)
            return

        try:
            validasi_csv(df_raw)
        except ValueError as e:
            st.markdown(f'<div class="err-box"><span>⚠️</span><span>{str(e)}</span></div>', unsafe_allow_html=True)
            return

        try:
            hasil_df, jumlah_terhapus, confidence_arr = bulk_prediction(model, scaler, df_raw)
        except ValueError as e:
            st.markdown(f'<div class="err-box"><span>⚠️</span><span>{str(e)}</span></div>', unsafe_allow_html=True)
            return

    except Exception as e:
        st.markdown(f'<div class="err-box"><span>⚠️</span><span>An error occurred: {e}</span></div>', unsafe_allow_html=True)
        return

    # ── Success banner ──
    st.markdown(f"""
    <div class="ok-box">
        <span>✅</span>
        <span>Prediction completed for <strong>{len(hasil_df)}</strong> records successfully.</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Step 4: Summary Statistics ──
    avg_conf = float(confidence_arr.mean())
    top_cat  = hasil_df["predicted_iq_category"].value_counts().idxmax()
    top_short = top_cat.replace("Intellectual Disability", "ID")

    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card">
            <div class="stat-icon">📋</div>
            <div class="stat-value">{len(hasil_df)}</div>
            <div class="stat-label">Total Processed</div>
        </div>
        <div class="stat-card">
            <div class="stat-icon">🗑️</div>
            <div class="stat-value">{jumlah_terhapus}</div>
            <div class="stat-label">Removed Records</div>
        </div>
        <div class="stat-card">
            <div class="stat-icon">🎯</div>
            <div class="stat-value">{avg_conf:.1f}%</div>
            <div class="stat-label">Avg Confidence</div>
        </div>
        <div class="stat-card">
            <div class="stat-icon">🏆</div>
            <div class="stat-value" style="font-size:0.9rem;letter-spacing:-0.2px;">{top_short}</div>
            <div class="stat-label">Most Frequent</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Dataset Analysis ──
    st.markdown("""
    <div class="section-card">
        <div class="card-header">
            <div class="card-icon card-icon-teal">📊</div>
            <div>
                <div class="card-title">Step 4 — Dataset Analysis</div>
                <div class="card-subtitle">Prediction category distribution</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    ch1, ch2 = st.columns([1.2, 1])
    with ch1:
        st.plotly_chart(bulk_dist_chart(hasil_df), use_container_width=True)
    with ch2:
        st.plotly_chart(donut_chart(hasil_df), use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Results Preview ──
    st.markdown("""
    <div class="section-card">
        <div class="card-header">
            <div class="card-icon card-icon-green">🔍</div>
            <div>
                <div class="card-title">Results Preview</div>
                <div class="card-subtitle">First 10 records</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.dataframe(
        hasil_df.head(10),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Download ──
    st.markdown("""
    <div class="section-card">
        <div class="card-header">
            <div class="card-icon card-icon-teal">⬇️</div>
            <div>
                <div class="card-title">Step 5 — Download Results</div>
                <div class="card-subtitle">Export full prediction dataset as CSV</div>
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
    # ── Top Header ──
    st.markdown("""
    <div class="rapiq-header">
        <div class="rapiq-header-left">
            <div class="rapiq-logo-mark">🧠</div>
            <div class="rapiq-header-titles">
                <h1>RapIQ</h1>
                <p class="subtitle">IQ Category Prediction System &nbsp;·&nbsp; Multilayer Perceptron (MLP)</p>
            </div>
        </div>
        <div class="rapiq-disclaimer">
            This system predicts IQ category based on trained machine learning patterns and
            <strong>does not replace professional psychological assessment.</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Load model ──
    model, scaler, errors = load_artifacts(MODEL_PATH, SCALER_PATH)
    if errors:
        st.markdown("<div style='padding:1.5rem 2rem;'>", unsafe_allow_html=True)
        for err in errors:
            st.markdown(f'<div class="err-box"><span>⚠️</span><span>{err}</span></div>',
                        unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # ── Tabs ──
    tab1, tab2 = st.tabs(["🔍  Single Prediction", "📂  Bulk Prediction — Dataset Testing"])

    with tab1:
        page_single(model, scaler)

    with tab2:
        page_bulk(model, scaler)


if __name__ == "__main__":
    main()