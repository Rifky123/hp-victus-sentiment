"""Tab 1 — Prediksi sentimen (single input & batch upload CSV)."""

import pandas as pd
import streamlit as st

from src.config import (
    BADGE_CSS_CLASS,
    CONTOH_ULASAN,
    EMOJI,
    RESULT_CSS_CLASS,
)
from src.model import ModelArtifacts, get_top_contributing_words, predict_sentiment
from src.preprocessing import highlight_kata
from src.state import add_to_history, clear_prediction_result, save_prediction_result


def _render_input_panel(artifacts: ModelArtifacts) -> None:
    st.markdown("**✏️ Masukkan ulasan:**")
    ulasan_input = st.text_area(
        label="",
        placeholder="Ketik ulasan laptop HP Victus di sini...",
        height=150,
        key="single_input",
        value=st.session_state["ulasan_input"],
    )

    st.caption(f"{'🔴' if len(ulasan_input) > 500 else '🟢'} {len(ulasan_input)} karakter")

    c1, c2 = st.columns(2)
    with c1:
        predict_btn = st.button("🔮 Prediksi", use_container_width=True, type="primary")
    with c2:
        if st.button("🗑️ Bersihkan", use_container_width=True):
            clear_prediction_result()
            st.rerun()

    st.markdown("---")
    st.markdown("**💡 Coba contoh ulasan:**")
    for label, teks in CONTOH_ULASAN.items():
        if st.button(label, use_container_width=True, key=f"ex_{label}"):
            pred, proba, clean = predict_sentiment(teks, artifacts)
            add_to_history(teks, pred, proba)
            save_prediction_result(teks, pred, proba, clean)
            st.rerun()

    if predict_btn:
        if ulasan_input.strip():
            pred, proba, clean = predict_sentiment(ulasan_input, artifacts)
            add_to_history(ulasan_input, pred, proba)
            save_prediction_result(ulasan_input, pred, proba, clean)
            st.rerun()
        else:
            st.warning("⚠️ Masukkan teks ulasan terlebih dahulu.")


def _render_output_panel(artifacts: ModelArtifacts) -> None:
    st.markdown("**📋 Hasil Prediksi:**")

    if st.session_state["hasil_prediksi"]:
        pred, proba, clean, teks_asli = st.session_state["hasil_prediksi"]
        confidence = max(proba.values()) * 100

        highlighted = highlight_kata(teks_asli)
        st.markdown(
            f"""
            <div class="result-box {RESULT_CSS_CLASS[pred]}">
                <div style="margin-bottom:0.6rem;">
                    <span style="font-size:1.8rem">{EMOJI[pred]}</span>
                    &nbsp;
                    <span class="badge {BADGE_CSS_CLASS[pred]}">{pred.upper()}</span>
                    &nbsp;
                    <span style="color:#888;font-size:0.85rem;">({confidence:.1f}% confidence)</span>
                </div>
                <p style="color:#444;font-size:0.9rem;margin:0;line-height:1.5;">{highlighted}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("**📊 Probabilitas per kelas:**")
        order = sorted(proba.items(), key=lambda x: x[1], reverse=True)
        for kelas, prob in order:
            col_k, col_b = st.columns([1, 3])
            col_k.markdown(f"{EMOJI[kelas]} **{kelas.capitalize()}**")
            col_b.progress(prob, text=f"{prob * 100:.1f}%")

        st.markdown(
            """
            <div style="margin-top:0.8rem;font-size:0.8rem;color:#666;">
                <span class="kata-positif">kata positif</span>&nbsp;&nbsp;
                <span class="kata-negatif">kata negatif</span>
                &nbsp; ← highlight berbasis kamus kata kunci manual
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            "⚠️ Highlight di atas hanya ilustrasi berdasarkan daftar kata "
            "buatan tangan, **bukan** cerminan langsung dari perhitungan "
            "model. Lihat bagian di bawah untuk kata yang benar-benar "
            "dipakai model dalam mengambil keputusan."
        )

        st.markdown("**🧠 Kata yang benar-benar berpengaruh menurut model:**")
        top_words = get_top_contributing_words(teks_asli, artifacts, top_n=5)
        if top_words:
            for kata, skor in top_words:
                st.markdown(f"- `{kata}`")
        else:
            st.caption("Tidak ada kata dari input yang dikenali model (di luar kosakata TF-IDF).")
        st.caption(
            "Dihitung dari bobot TF-IDF × log-probabilitas Naive Bayes "
            "untuk kelas yang diprediksi — ini fitur yang sesungguhnya "
            "dipakai model, bukan kamus kata kunci."
        )

        with st.expander("🔬 Teks setelah preprocessing"):
            st.code(clean, language=None)
    else:
        st.markdown(
            """
            <div style="text-align:center;padding:3rem 1rem;color:#aaa;border:2px dashed #ddd;border-radius:12px;">
                <div style="font-size:3rem;">🔍</div>
                <p style="margin-top:0.5rem;">Hasil prediksi akan muncul di sini</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_batch_panel(artifacts: ModelArtifacts) -> None:
    st.markdown("---")
    st.markdown('<p class="section-header">📂 Prediksi Batch (Upload CSV)</p>', unsafe_allow_html=True)

    col_up, col_info = st.columns([2, 1])
    with col_up:
        uploaded = st.file_uploader("Upload CSV dengan kolom 'ulasan'", type=["csv"])
    with col_info:
        st.info("💡 Format CSV: pastikan ada kolom **ulasan** berisi teks ulasan yang ingin diprediksi.")

    if not uploaded:
        return

    try:
        df_up = pd.read_csv(uploaded)
        if "ulasan" not in df_up.columns:
            st.error("❌ Kolom 'ulasan' tidak ditemukan.")
            return

        prog = st.progress(0, text="Memproses ulasan...")
        results = []
        for i, row in df_up.iterrows():
            p, pb, _ = predict_sentiment(str(row["ulasan"]), artifacts)
            results.append(
                {"sentimen_prediksi": p, "confidence": f"{max(pb.values()) * 100:.1f}%"}
            )
            prog.progress((i + 1) / len(df_up), text=f"Memproses {i + 1}/{len(df_up)}...")
        prog.empty()

        df_up = pd.concat([df_up, pd.DataFrame(results)], axis=1)
        st.success(f"✅ {len(df_up)} ulasan berhasil diproses!")

        counts_pred = df_up["sentimen_prediksi"].value_counts()
        c1, c2, c3 = st.columns(3)
        c1.metric("😊 Positif", counts_pred.get("positif", 0))
        c2.metric("😞 Negatif", counts_pred.get("negatif", 0))
        c3.metric("😐 Netral", counts_pred.get("netral", 0))

        st.dataframe(df_up, use_container_width=True)
        st.download_button(
            "⬇️ Download Hasil CSV",
            data=df_up.to_csv(index=False).encode("utf-8"),
            file_name="hasil_prediksi_batch.csv",
            mime="text/csv",
            use_container_width=True,
        )
    except Exception as e:  # noqa: BLE001 - tampilkan error apa pun ke pengguna
        st.error(f"Gagal: {e}")


def render_predict_tab(artifacts: ModelArtifacts) -> None:
    st.markdown('<p class="section-header">🔍 Prediksi Sentimen Ulasan</p>', unsafe_allow_html=True)

    col_in, col_out = st.columns([1, 1], gap="large")
    with col_in:
        _render_input_panel(artifacts)
    with col_out:
        _render_output_panel(artifacts)

    _render_batch_panel(artifacts)
