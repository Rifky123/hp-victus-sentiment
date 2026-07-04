"""Tab 4 — Visualisasi data & performa model."""

import matplotlib.pyplot as plt
import streamlit as st
from wordcloud import WordCloud

from src.model import ModelArtifacts

_DIST_COLORS = ["#28a745", "#dc3545", "#ffc107"]


def _render_distribution_charts(df) -> None:
    col_pie, col_bar = st.columns(2)
    counts = df["sentimen"].value_counts()

    with col_pie:
        st.markdown("**Distribusi Sentimen Dataset**")
        fig, ax = plt.subplots(figsize=(5, 4))
        wedges, texts, autotexts = ax.pie(
            counts.values,
            labels=[l.capitalize() for l in counts.index],
            autopct="%1.1f%%",
            colors=_DIST_COLORS,
            startangle=90,
            wedgeprops=dict(edgecolor="white", linewidth=2.5),
            pctdistance=0.75,
        )
        for at in autotexts:
            at.set_fontsize(11)
            at.set_fontweight("bold")
            at.set_color("white")
        ax.set_title("Distribusi Sentimen", fontsize=12, fontweight="bold", pad=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_bar:
        st.markdown("**Jumlah Ulasan per Kategori**")
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(
            [l.capitalize() for l in counts.index],
            counts.values,
            color=_DIST_COLORS,
            edgecolor="white",
            linewidth=2,
            width=0.5,
            zorder=3,
        )
        ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
        for bar, val in zip(bars, counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                str(val), ha="center", va="bottom", fontweight="bold", fontsize=13,
            )
        ax.set_ylabel("Jumlah", fontsize=11)
        ax.set_title("Jumlah Ulasan per Sentimen", fontsize=12, fontweight="bold")
        ax.set_ylim(0, max(counts.values) * 1.18)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


def _render_wordclouds(df) -> None:
    st.markdown("**☁️ Word Cloud per Sentimen**")
    wc_cols = st.columns(3)
    for col, label, cmap, emoji in zip(
        wc_cols, ["positif", "negatif", "netral"], ["Greens", "Reds", "YlOrBr"], ["😊", "😞", "😐"]
    ):
        with col:
            st.markdown(f"**{emoji} {label.capitalize()}**")
            subset = df[df["sentimen"] == label]["ulasan"].str.cat(sep=" ")
            if subset.strip():
                wc = WordCloud(
                    width=400, height=260, background_color="white",
                    colormap=cmap, max_words=60, collocations=False,
                ).generate(subset)
                fig, ax = plt.subplots(figsize=(4, 2.6))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)


def _render_metric_comparison(metrics: dict) -> None:
    st.markdown("**📉 Perbandingan Metrik Evaluasi**")
    fig, ax = plt.subplots(figsize=(7, 3))
    m_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    m_values = [metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"]]
    m_colors = ["#667eea", "#f093fb", "#4facfe", "#43e97b"]
    bars = ax.barh(
        m_labels, [v * 100 for v in m_values], color=m_colors,
        edgecolor="white", height=0.45, zorder=3,
    )
    ax.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    for bar, val in zip(bars, m_values):
        ax.text(
            val * 100 + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{val * 100:.1f}%", va="center", fontweight="bold", fontsize=11,
        )
    ax.set_xlim(0, 115)
    ax.set_xlabel("Nilai (%)", fontsize=11)
    ax.set_title("Perbandingan Metrik Evaluasi Model", fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_visualization_tab(artifacts: ModelArtifacts) -> None:
    df = artifacts.df
    metrics = artifacts.metrics

    st.markdown('<p class="section-header">📈 Visualisasi Data & Model</p>', unsafe_allow_html=True)

    _render_distribution_charts(df)
    st.markdown("---")
    _render_wordclouds(df)
    st.markdown("---")
    _render_metric_comparison(metrics)
