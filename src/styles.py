"""Custom CSS untuk aplikasi Streamlit."""

import streamlit as st

CUSTOM_CSS = """
<style>
    /* Global */
    .block-container { padding-top: 1.5rem; }

    /* Header */
    .main-title {
        text-align: center;
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        text-align: center;
        color: #777;
        margin-bottom: 1.5rem;
        font-size: 0.95rem;
        letter-spacing: 0.3px;
    }

    /* Badge sentimen */
    .badge {
        display: inline-block;
        padding: 5px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1rem;
        letter-spacing: 0.5px;
    }
    .badge-positif { background:#28a745; color:white; }
    .badge-negatif { background:#dc3545; color:white; }
    .badge-netral  { background:#ffc107; color:#333;  }

    /* Result box */
    .result-box {
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        margin: 0.8rem 0;
        border-left: 6px solid;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    }
    .result-positif { background:#f0fff4; border-color:#28a745; }
    .result-negatif { background:#fff5f5; border-color:#dc3545; }
    .result-netral  { background:#fffdf0; border-color:#ffc107; }

    /* Section header */
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #2d2d2d;
        margin: 1.2rem 0 0.6rem 0;
        padding-bottom: 0.3rem;
        border-bottom: 3px solid #667eea33;
    }

    /* Riwayat card */
    .riwayat-card {
        background: #f8f9ff;
        border: 1px solid #e0e4ff;
        border-radius: 10px;
        padding: 0.9rem 1.2rem;
        margin-bottom: 0.7rem;
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 1rem;
    }
    .riwayat-teks { color: #333; font-size: 0.9rem; flex: 1; }
    .riwayat-meta { color: #999; font-size: 0.75rem; margin-top: 3px; }

    /* Sidebar */
    .sidebar-stat {
        background: linear-gradient(135deg, #667eea22, #764ba222);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        text-align: center;
    }
    .sidebar-stat h3 { margin: 0; font-size: 1.5rem; color: #667eea; }
    .sidebar-stat p  { margin: 0; font-size: 0.8rem; color: #555; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 18px;
        font-weight: 600;
        font-size: 0.88rem;
    }

    /* Highlight kata */
    .kata-positif { background:#d4edda; border-radius:3px; padding:1px 3px; color:#155724; font-weight:600; }
    .kata-negatif { background:#f8d7da; border-radius:3px; padding:1px 3px; color:#721c24; font-weight:600; }

    /* Metric angka besar */
    .big-metric {
        border-radius: 12px;
        padding: 1.1rem;
        text-align: center;
        box-shadow: 0 3px 12px rgba(0,0,0,0.08);
        margin-bottom: 0.5rem;
    }
    .big-metric h2 { font-size: 1.9rem; margin: 0; font-weight: 800; }
    .big-metric p  { margin: 0; font-size: 0.82rem; opacity: 0.85; }
</style>
"""


def apply_custom_css() -> None:
    """Suntikkan CSS kustom ke halaman Streamlit."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
