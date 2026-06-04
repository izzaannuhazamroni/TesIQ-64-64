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
    0: "#EF4444",
    1: "#F97316",
    2: "#EAB308",
    3: "#22C55E",
    4: "#4F46E5",
}

# ══════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Sora', sans-serif; }

.stApp { background: #F4F6FB; }

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.8rem; padding-bottom: 3rem; }

/* ── header ── */
.rapiq-header {
    background: linear-gradient(135deg, #4F46E5 0%, #6366F1 60%, #818CF8 100%);
    border-radius: 20px;
    padding: 2.4rem 2rem 2rem;
    text-align: center;
    margin-bottom: 1.6rem;
    box-shadow: 0 8px 32px rgba(79,70,229,0.22);
}
.rapiq-header .logo { font-size: 3rem; line-height: 1; margin-bottom: .4rem; }
.rapiq-header h1 {
    color: #fff; font-size: 2rem; font-weight: 700;
    letter-spacing: -.5px; margin: 0 0 .3rem;
}
.rapiq-header p {
    color: rgba(255,255,255,.82); font-size: .85rem;
    font-weight: 300; line-height: 1.5; margin: 0;
}

/* ── tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #fff;
    border-radius: 14px;
    padding: .35rem .4rem;
    gap: .3rem;
    box-shadow: 0 2px 10px rgba(79,70,229,.08);
    border: 1px solid #E8EAFF;
    margin-bottom: 1.4rem;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    padding: .55rem 1.3rem;
    font-size: .85rem;
    font-weight: 500;
    color: #6B7280;
    border: none;
    background: transparent;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg,#4F46E5,#6366F1) !important;
    color: #fff !important;
    font-weight: 600;
}

/* ── section card ── */
.section-card {
    background: #fff;
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.4rem;
    box-shadow: 0 2px 12px rgba(79,70,229,.07);
    border: 1px solid #E8EAFF;
}
.section-title {
    font-size: .7rem; font-weight: 600;
    letter-spacing: 1.5px; text-transform: uppercase;
    color: #4F46E5; margin-bottom: 1rem;
}

/* ── buttons ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg,#4F46E5,#6366F1);
    color: #fff; font-family: 'Sora',sans-serif;
    font-size: .95rem; font-weight: 600;
    border: none; border-radius: 12px;
    padding: .9rem 1.5rem; cursor: pointer;
    letter-spacing: .3px;
    box-shadow: 0 4px 16px rgba(79,70,229,.30);
    transition: opacity .2s;
}
.stButton > button:hover { opacity: .88; }

/* ── download button ── */
.stDownloadButton > button {
    width: 100%;
    background: #fff;
    color: #4F46E5; font-family: 'Sora',sans-serif;
    font-size: .88rem; font-weight: 600;
    border: 2px solid #4F46E5; border-radius: 12px;
    padding: .7rem 1.2rem; cursor: pointer;
    transition: all .2s;
}
.stDownloadButton > button:hover {
    background: #4F46E5; color: #fff;
}

/* ── result card ── */
.result-card {
    border-radius: 16px; padding: 1.8rem 2rem;
    margin-top: 1.4rem; text-align: center;
}
.result-label {
    font-size: .68rem; font-weight: 600;
    letter-spacing: 1.6px; text-transform: uppercase;
    opacity: .7; margin-bottom: .3rem;
}
.result-category {
    font-size: 1.45rem; font-weight: 700;
    margin-bottom: .15rem; letter-spacing: -.3px;
}
.result-range {
    font-size: .85rem; font-weight: 400;
    margin-bottom: 1rem; opacity: .75;
}
.result-confidence {
    display: inline-block;
    background: rgba(255,255,255,.35);
    border-radius: 50px; padding: .35rem 1.1rem;
    font-size: .82rem; font-weight: 600; letter-spacing: .3px;
}

/* ── stat cards ── */
.stat-grid { display: flex; gap: .9rem; margin-bottom: 1.4rem; flex-wrap: wrap; }
.stat-card {
    flex: 1; min-width: 120px;
    background: #fff; border-radius: 14px;
    padding: 1rem 1.1rem;
    box-shadow: 0 2px 10px rgba(79,70,229,.07);
    border: 1px solid #E8EAFF; text-align: center;
}
.stat-card .stat-val {
    font-size: 1.6rem; font-weight: 700; color: #4F46E5; line-height: 1.1;
}
.stat-card .stat-key {
    font-size: .72rem; color: #6B7280; font-weight: 500;
    margin-top: .2rem; letter-spacing: .3px;
}

/* ── prob table ── */
.prob-table { width:100%; border-collapse:collapse; font-size:.82rem; margin-top:.5rem; }
.prob-table th {
    text-align:left; color:#6B7280; font-weight:500;
    padding:.45rem .6rem; border-bottom:1px solid #E5E7EB;
}
.prob-table td { padding:.45rem .6rem; border-bottom:1px solid #F3F4F6; color:#374151; }
.prob-table tr:last-child td { border-bottom:none; }
.prob-bar-wrap {
    background:#E8EAFF; border-radius:6px; height:8px;
    width:120px; display:inline-block; vertical-align:middle; margin-right:6px;
}
.prob-bar-fill { border-radius:6px; height:8px; display:inline-block; }

/* ── template info box ── */
.info-box {
    background: #EEF2FF; border: 1px solid #C7D2FE;
    border-radius: 12px; padding: 1rem 1.2rem;
    font-size: .84rem; color: #3730A3; margin-bottom: 1rem;
}
.info-box code {
    background: #C7D2FE; border-radius: 4px;
    padding: .1rem .35rem; font-family: 'DM Mono', monospace;
    font-size: .8rem;
}

/* ── error box ── */
.err-box {
    background:#FEF2F2; border:1px solid #FECACA;
    border-radius:12px; padding:1rem 1.2rem;
    color:#B91C1C; font-size:.85rem; margin-bottom:.8rem;
}

/* ── success box ── */
.ok-box {
    background:#F0FDF4; border:1px solid #BBF7D0;
    border-radius:12px; padding:1rem 1.2rem;
    color:#166534; font-size:.85rem; margin-bottom:.8rem;
}

/* ── file uploader ── */
[data-testid="stFileUploader"] {
    border: 2px dashed #C7D2FE !important;
    border-radius: 14px !important;
    background: #F8F9FF !important;
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
    st.markdown('<div class="section-card">'
                '<div class="section-title">📋 Informasi Keluarga</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        edu_mother_lbl = st.selectbox("Pendidikan Ibu",  list(EDU_MAP_UI), key="s_edu_m")
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

        bg    = IQ_COLORS[pred_class]
        cat   = IQ_LABELS[pred_class]
        rng   = IQ_RANGES[pred_class]

        st.markdown(f"""
        <div class="result-card" style="background:{bg};color:#fff;">
            <div class="result-label">Hasil Prediksi</div>
            <div class="result-category">{cat}</div>
            <div class="result-range">Rentang IQ: {rng}</div>
            <span class="result-confidence">Confidence: {confidence:.1f}%</span>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-card" style="margin-top:1.4rem;">'
                    '<div class="section-title">📊 Probabilitas Seluruh Kelas</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(single_prob_chart(proba), width='stretch')

        rows = ""
        for i in range(5):
            pct   = proba[i] * 100
            w     = int(pct * 1.2)
            bold  = "font-weight:700;color:#4F46E5;" if i == pred_class else ""
            fill  = "#4F46E5" if i == pred_class else "#A5B4FC"
            rows += f"""
            <tr>
                <td style="{bold}">Kelas {i}</td>
                <td style="{bold}">{IQ_LABELS[i]}</td>
                <td style="{bold}">{IQ_RANGES[i]}</td>
                <td>
                    <span class="prob-bar-wrap">
                        <span class="prob-bar-fill"
                              style="width:{w}px;background:{fill};"></span>
                    </span>
                    <span style="{bold}">{pct:.2f}%</span>
                </td>
            </tr>"""

        st.markdown(f"""
        <table class="prob-table">
          <thead><tr>
            <th>Kelas</th><th>Kategori</th>
            <th>Rentang IQ</th><th>Probabilitas</th>
          </tr></thead>
          <tbody>{rows}</tbody>
        </table></div>""", unsafe_allow_html=True)


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
    # header
    st.markdown("""
    <div class="rapiq-header">
        <div class="logo">🧠</div>
        <h1>RapIQ</h1>
        <p>Sistem Prediksi Kategori IQ Berbasis Artificial Intelligence<br>
        Menggunakan Multilayer Perceptron (MLP)</p>
    </div>
    """, unsafe_allow_html=True)

    # load model
    model, scaler, errors = load_artifacts(MODEL_PATH, SCALER_PATH)
    if errors:
        for err in errors:
            st.markdown(f'<div class="err-box">⚠️ {err}</div>',
                        unsafe_allow_html=True)
        return

    # navigasi tab
    tab1, tab2 = st.tabs(["🔍 Prediksi Tunggal", "📂 Pengujian Dataset"])

    with tab1:
        page_single(model, scaler)

    with tab2:
        page_bulk(model, scaler)


if __name__ == "__main__":
    main()