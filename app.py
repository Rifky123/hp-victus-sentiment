"""
Analisis Sentimen Ulasan Laptop HP Victus
==========================================
Titik masuk aplikasi Streamlit. File ini hanya bertugas mengatur alur
halaman (page config, training model, sidebar, tab-tab), sedangkan
seluruh logika (preprocessing, model, komponen UI) ada di paket `src/`.

Jalankan dengan:
    streamlit run app.py
"""

import warnings

import streamlit as st

from src.components.sidebar import render_sidebar
from src.components.tab_dataset import render_dataset_tab
from src.components.tab_evaluation import render_evaluation_tab
from src.components.tab_history import render_history_tab
from src.components.tab_predict import render_predict_tab
from src.components.tab_visualization import render_visualization_tab
from src.config import APP_SUBTITLE, APP_TITLE, FOOTER_TEXT, PAGE_CONFIG
from src.model import train_model
from src.state import init_session_state
from src.styles import apply_custom_css

warnings.filterwarnings("ignore")

# ─── PAGE SETUP ────────────────────────────────────────────────────────
st.set_page_config(**PAGE_CONFIG)
apply_custom_css()

# ─── TRAIN / LOAD MODEL (CACHED) ────────────────────────────────────────
with st.spinner("⚙️ Melatih model Naive Bayes..."):
    artifacts = train_model()

# ─── SESSION STATE ───────────────────────────────────────────────────────
init_session_state()

# ─── SIDEBAR ─────────────────────────────────────────────────────────────
render_sidebar(artifacts)

# ─── HEADER ──────────────────────────────────────────────────────────────
st.markdown(f'<h1 class="main-title">{APP_TITLE}</h1>', unsafe_allow_html=True)
st.markdown(f'<p class="sub-title">{APP_SUBTITLE}</p>', unsafe_allow_html=True)

# ─── TABS ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🔍 Prediksi", "📜 Riwayat", "📊 Evaluasi Model", "📈 Visualisasi", "📋 Dataset"]
)

with tab1:
    render_predict_tab(artifacts)

with tab2:
    render_history_tab()

with tab3:
    render_evaluation_tab(artifacts)

with tab4:
    render_visualization_tab(artifacts)

with tab5:
    render_dataset_tab(artifacts)

# ─── FOOTER ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"""
    <div style="text-align:center;color:#aaa;font-size:0.82rem;padding:0.8rem 0;">
        {FOOTER_TEXT}
    </div>
    """,
    unsafe_allow_html=True,
)
