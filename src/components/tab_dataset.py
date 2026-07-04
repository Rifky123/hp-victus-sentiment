"""Tab 5 — Penjelajah dataset (filter & pencarian)."""

import streamlit as st

from src.model import ModelArtifacts


def render_dataset_tab(artifacts: ModelArtifacts) -> None:
    df = artifacts.df

    st.markdown('<p class="section-header">📋 Dataset Ulasan HP Victus</p>', unsafe_allow_html=True)

    info = artifacts.dataset_info
    if info.get("duplikat_dihapus", 0) > 0:
        st.caption(
            f"ℹ️ Dataset di bawah adalah versi yang **sudah dibersihkan** "
            f"({info['duplikat_dihapus']} ulasan duplikat dihapus dari "
            f"{info['total_mentah']} baris mentah)."
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📦 Total", len(df))
    c2.metric("😊 Positif", len(df[df.sentimen == "positif"]))
    c3.metric("😞 Negatif", len(df[df.sentimen == "negatif"]))
    c4.metric("😐 Netral", len(df[df.sentimen == "netral"]))

    st.markdown("---")

    col_f1, col_f2 = st.columns([1, 2])
    with col_f1:
        filter_s = st.selectbox("Filter sentimen:", ["Semua", "Positif", "Negatif", "Netral"])
    with col_f2:
        search_q = st.text_input("🔍 Cari kata dalam ulasan:", placeholder="Ketik kata kunci...")

    df_show = df.copy()
    if filter_s != "Semua":
        df_show = df_show[df_show["sentimen"] == filter_s.lower()]
    if search_q.strip():
        df_show = df_show[df_show["ulasan"].str.contains(search_q, case=False, na=False)]

    st.dataframe(
        df_show.reset_index(drop=True),
        use_container_width=True,
        column_config={
            "ulasan": st.column_config.TextColumn("Ulasan", width="large"),
            "sentimen": st.column_config.TextColumn("Sentimen", width="small"),
        },
    )
    st.caption(f"Menampilkan **{len(df_show)}** dari **{len(df)}** ulasan")
