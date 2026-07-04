"""Komponen sidebar: identitas tim, skor performa model, dan info dataset."""

import streamlit as st

from src.config import METRIC_COLORS, TIM_PENELITI
from src.model import ModelArtifacts


def render_sidebar(artifacts: ModelArtifacts) -> None:
    df = artifacts.df
    metrics = artifacts.metrics

    with st.sidebar:
        st.markdown("## 💻 HP Victus Sentiment")
        st.markdown("---")

        st.markdown("### 👥 Tim Peneliti")
        for anggota in TIM_PENELITI:
            st.markdown(
                f"**{anggota['peran']}**  \n{anggota['nama']}  \n`{anggota['nim']}`"
            )
            st.markdown("")

        st.markdown("---")
        st.markdown("### 📊 Performa Model")
        for label, key in [
            ("🎯 Accuracy", "accuracy"),
            ("🔎 Precision", "precision"),
            ("📡 Recall", "recall"),
            ("⚖️ F1-Score", "f1"),
        ]:
            val = metrics[key]
            color = METRIC_COLORS[key]
            st.markdown(
                f"""
                <div class="sidebar-stat">
                    <h3 style="color:{color}">{val * 100:.1f}%</h3>
                    <p>{label}</p>
                </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("### 📦 Info Dataset")
        info = artifacts.dataset_info
        dup_note = (
            f"- **Duplikat dihapus:** {info['duplikat_dihapus']} "
            f"(dari {info['total_mentah']} baris mentah)\n"
            if info.get("duplikat_dihapus", 0) > 0
            else ""
        )
        st.markdown(
            f"""
            - **Total (bersih):** {len(df)} ulasan
            - **Positif:** {len(df[df.sentimen == 'positif'])} ulasan
            - **Negatif:** {len(df[df.sentimen == 'negatif'])} ulasan
            - **Netral:** {len(df[df.sentimen == 'netral'])} ulasan
            {dup_note}- **Split:** 80% train / 20% test
            """
        )
        st.markdown("---")
        st.caption("Universitas Dian Nuswantoro | Fakultas Ilmu Komputer | 2025")
