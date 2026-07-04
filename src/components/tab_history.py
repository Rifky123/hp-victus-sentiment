"""Tab 2 — Riwayat prediksi pada sesi berjalan."""

import pandas as pd
import streamlit as st

from src.config import BADGE_CSS_CLASS, EMOJI
from src.state import clear_history


def render_history_tab() -> None:
    st.markdown('<p class="section-header">📜 Riwayat Prediksi Sesi Ini</p>', unsafe_allow_html=True)

    riwayat = st.session_state["riwayat"]

    if not riwayat:
        st.markdown(
            """
            <div style="text-align:center;padding:3rem;color:#bbb;border:2px dashed #eee;border-radius:12px;">
                <div style="font-size:2.5rem;">📭</div>
                <p>Belum ada riwayat prediksi.<br>Coba prediksi ulasan di tab <b>Prediksi</b>!</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    col_r1, col_r2 = st.columns([3, 1])
    with col_r1:
        st.markdown(f"**Total prediksi: {len(riwayat)}**")
    with col_r2:
        if st.button("🗑️ Hapus Riwayat", use_container_width=True):
            clear_history()
            st.rerun()

    df_riwayat = pd.DataFrame(riwayat)
    c1, c2, c3 = st.columns(3)
    c1.metric("😊 Positif", len(df_riwayat[df_riwayat.sentimen == "positif"]))
    c2.metric("😞 Negatif", len(df_riwayat[df_riwayat.sentimen == "negatif"]))
    c3.metric("😐 Netral", len(df_riwayat[df_riwayat.sentimen == "netral"]))

    st.markdown("---")

    for item in riwayat:
        badge_cls = BADGE_CSS_CLASS[item["sentimen"]]
        emoji = EMOJI[item["sentimen"]]
        st.markdown(
            f"""
            <div class="riwayat-card">
                <div class="riwayat-teks">
                    {item['ulasan']}
                    <div class="riwayat-meta">🕐 {item['waktu']}</div>
                </div>
                <div style="text-align:right;min-width:110px;">
                    <span class="badge {badge_cls}">{emoji} {item['sentimen'].upper()}</span>
                    <div class="riwayat-meta" style="margin-top:4px;">
                        {item['confidence']:.1f}% confidence
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.download_button(
        "⬇️ Download Riwayat CSV",
        data=df_riwayat[["waktu", "ulasan", "sentimen", "confidence"]]
        .to_csv(index=False)
        .encode("utf-8"),
        file_name="riwayat_prediksi.csv",
        mime="text/csv",
    )
