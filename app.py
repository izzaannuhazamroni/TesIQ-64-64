"""
RapIQ — Sistem Prediksi Kategori IQ
Berbasis Multilayer Perceptron (MLP)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="RapIQ — Prediksi Kategori IQ",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
MODEL_PATH  = "model_mlp_iq.pkl"
SCALER_PATH = "scaler_iq.pkl"

IQ_LABELS = {
    0: "Moderate Intellectual Disability",
    1: "Mild Intellectual Disability",
    2: "Below Average Intelligence",
    3: "Average Intelligence",
    4: "Above Average Intelligence",
}

IQ_RANGES = {
    0: "35 – 54",
    1: "55 – 69",
    2: "70 – 84",
    3: "85 – 114",
    4: "> 114",
}

IQ_COLORS = {
    0: "#EF4444",   # red
    1: "#F97316",   # orange
    2: "#EAB308",   # yellow
    3: "#22C55E",   # green
    4: "#4F46E5",   # indigo
}

EDU_MAP = {
    "SD / SMP"          : 0,
    "Kejuruan"          : 1,
    "SMA"               : 2,
    "Perguruan Tinggi"  : 3,
}

GENDER_MAP = {
    "Laki-laki"  : 1,
    "Perempuan"  : 0,
}

CLASS_NAMES = [IQ_LABELS[i] for i in range(5)]

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

/* ── background ── */
.stApp {
    background: #F4F6FB;
}

/* ── hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; }

/* ── header card ── */
.rapiq-header {
    background: linear-gradient(135deg, #4F46E5 0%, #6366F1 60%, #818CF8 100%);
    border-radius: 20px;
    padding: 2.4rem 2rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(79,70,229,0.22);
}
.rapiq-header .logo { font-size: 3rem; line-height: 1; margin-bottom: .4rem; }
.rapiq-header h1 {
    color: #fff;
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.5px;
    margin: 0 0 .3rem;
}
.rapiq-header p {
    color: rgba(255,255,255,0.82);
    font-size: .85rem;
    font-weight: 300;
    line-height: 1.5;
    margin: 0;
}

/* ── section card ── */
.section-card {
    background: #fff;
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.4rem;
    box-shadow: 0 2px 12px rgba(79,70,229,0.07);
    border: 1px solid #E8EAFF;
}
.section-title {
    font-size: .7rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #4F46E5;
    margin-bottom: 1rem;
}

/* ── selectbox & number label tweaks ── */
.stSelectbox label, .stNumberInput label {
    font-size: .83rem !important;
    font-weight: 500 !important;
    color: #374151 !important;
}

/* ── predict button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #4F46E5, #6366F1);
    color: #fff;
    font-family: 'Sora', sans-serif;
    font-size: .95rem;
    font-weight: 600;
    border: none;
    border-radius: 12px;
    padding: .9rem 1.5rem;
    cursor: pointer;
    letter-spacing: 0.3px;
    box-shadow: 0 4px 16px rgba(79,70,229,0.30);
    transition: opacity .2s;
}
.stButton > button:hover { opacity: .88; }

/* ── result card ── */
.result-card {
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin-top: 1.4rem;
    text-align: center;
}
.result-label {
    font-size: .68rem;
    font-weight: 600;
    letter-spacing: 1.6px;
    text-transform: uppercase;
    opacity: .7;
    margin-bottom: .3rem;
}
.result-category {
    font-size: 1.45rem;
    font-weight: 700;
    margin-bottom: .15rem;
    letter-spacing: -0.3px;
}
.result-range {
    font-size: .85rem;
    font-weight: 400;
    margin-bottom: 1rem;
    opacity: .75;
}
.result-confidence {
    display: inline-block;
    background: rgba(255,255,255,0.35);
    border-radius: 50px;
    padding: .35rem 1.1rem;
    font-size: .82rem;
    font-weight: 600;
    letter-spacing: 0.3px;
}

/* ── prob table ── */
.prob-table {
    width: 100%;
    border-collapse: collapse;
    font-size: .82rem;
    margin-top: .5rem;
}
.prob-table th {
    text-align: left;
    color: #6B7280;
    font-weight: 500;
    padding: .45rem .6rem;
    border-bottom: 1px solid #E5E7EB;
}
.prob-table td {
    padding: .45rem .6rem;
    border-bottom: 1px solid #F3F4F6;
    color: #374151;
}
.prob-table tr:last-child td { border-bottom: none; }
.prob-bar-wrap {
    background: #E8EAFF;
    border-radius: 6px;
    height: 8px;
    width: 120px;
    display: inline-block;
    vertical-align: middle;
    margin-right: 6px;
}
.prob-bar-fill {
    background: #4F46E5;
    border-radius: 6px;
    height: 8px;
    display: inline-block;
}

/* ── error box ── */
.err-box {
    background: #FEF2F2;
    border: 1px solid #FECACA;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    color: #B91C1C;
    font-size: .85rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD MODEL (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Memuat model…")
def load_artifacts(model_path: str, scaler_path: str):
    errors = []
    model, scaler = None, None

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


# ─────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────
def predict(model, scaler, edu_mother: int, edu_father: int,
            age: int, gender: int):
    X = pd.DataFrame([[edu_mother, edu_father, age, gender]],
                     columns=["education_mother", "education_father",
                              "age_years", "gender"])
    X_scaled = scaler.transform(X)
    pred_class  = int(model.predict(X_scaled)[0])
    proba       = model.predict_proba(X_scaled)[0]
    confidence  = float(proba.max()) * 100
    return pred_class, proba, confidence


# ─────────────────────────────────────────────
# PLOTLY BAR CHART
# ─────────────────────────────────────────────
def probability_chart(proba: np.ndarray):
    labels = [f"Kelas {i}" for i in range(5)]
    colors = [IQ_COLORS[i] for i in range(5)]

    fig = go.Figure(go.Bar(
        x=labels,
        y=proba * 100,
        marker_color=colors,
        text=[f"{v*100:.1f}%" for v in proba],
        textposition="outside",
        textfont=dict(family="Sora", size=11, color="#374151"),
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=10, b=40),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        yaxis=dict(
            title="Probabilitas (%)",
            range=[0, max(proba) * 120 + 5],
            gridcolor="#F3F4F6",
            tickfont=dict(family="Sora", size=10),
        ),
        xaxis=dict(tickfont=dict(family="Sora", size=10)),
        font=dict(family="Sora"),
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
def main():
    # ── Header ──
    st.markdown("""
    <div class="rapiq-header">
        <div class="logo">🧠</div>
        <h1>RapIQ</h1>
        <p>Sistem Prediksi Kategori IQ Berbasis Artificial Intelligence<br>
        Menggunakan Multilayer Perceptron (MLP)</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load model ──
    model, scaler, errors = load_artifacts(MODEL_PATH, SCALER_PATH)

    if errors:
        for err in errors:
            st.markdown(f'<div class="err-box">⚠️ {err}</div>',
                        unsafe_allow_html=True)
        return

    # ── Form: Informasi Keluarga ──
    st.markdown('<div class="section-card">'
                '<div class="section-title">📋 Informasi Keluarga</div>',
                unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        edu_mother_label = st.selectbox(
            "Pendidikan Ibu",
            options=list(EDU_MAP.keys()),
            index=0,
        )
    with col2:
        edu_father_label = st.selectbox(
            "Pendidikan Ayah",
            options=list(EDU_MAP.keys()),
            index=0,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Form: Informasi Anak ──
    st.markdown('<div class="section-card">'
                '<div class="section-title">👶 Informasi Anak</div>',
                unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        age = st.number_input(
            "Usia Anak (tahun)",
            min_value=1,
            max_value=18,
            value=10,
            step=1,
        )
    with col4:
        gender_label = st.selectbox(
            "Jenis Kelamin",
            options=list(GENDER_MAP.keys()),
            index=0,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Predict button ──
    predict_btn = st.button("🔍 Prediksi Kategori IQ")

    if predict_btn:
        edu_mother = EDU_MAP[edu_mother_label]
        edu_father = EDU_MAP[edu_father_label]
        gender     = GENDER_MAP[gender_label]

        try:
            pred_class, proba, confidence = predict(
                model, scaler, edu_mother, edu_father, int(age), gender
            )
        except Exception as e:
            st.markdown(f'<div class="err-box">⚠️ Prediksi gagal: {e}</div>',
                        unsafe_allow_html=True)
            return

        bg_color  = IQ_COLORS[pred_class]
        category  = IQ_LABELS[pred_class]
        iq_range  = IQ_RANGES[pred_class]

        # ── Result card ──
        st.markdown(f"""
        <div class="result-card" style="background:{bg_color};color:#fff;">
            <div class="result-label">Hasil Prediksi</div>
            <div class="result-category">{category}</div>
            <div class="result-range">Rentang IQ: {iq_range}</div>
            <span class="result-confidence">Confidence: {confidence:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)

        # ── Probability chart ──
        st.markdown('<div class="section-card" style="margin-top:1.4rem;">'
                    '<div class="section-title">📊 Probabilitas Seluruh Kelas</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(probability_chart(proba), use_container_width=True)

        # ── Probability table ──
        rows_html = ""
        for i in range(5):
            pct   = proba[i] * 100
            width = int(pct * 1.2)   # scale to max 120px
            bold  = "font-weight:700;color:#4F46E5;" if i == pred_class else ""
            rows_html += f"""
            <tr>
                <td style="{bold}">Kelas {i}</td>
                <td style="{bold}">{IQ_LABELS[i]}</td>
                <td style="{bold}">{IQ_RANGES[i]}</td>
                <td>
                    <span class="prob-bar-wrap">
                        <span class="prob-bar-fill" style="width:{width}px;
                              background:{'#4F46E5' if i==pred_class else '#A5B4FC'};"></span>
                    </span>
                    <span style="{bold}">{pct:.2f}%</span>
                </td>
            </tr>"""

        st.markdown(f"""
        <table class="prob-table">
            <thead>
                <tr>
                    <th>Kelas</th>
                    <th>Kategori</th>
                    <th>Rentang IQ</th>
                    <th>Probabilitas</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()