"""Manajemen `st.session_state` aplikasi."""

from __future__ import annotations

import datetime
from typing import Dict

import streamlit as st

_DEFAULTS = {
    "ulasan_input": "",
    "hasil_prediksi": None,
    "riwayat": [],
}


def init_session_state() -> None:
    """Inisialisasi semua key session_state dengan nilai default."""
    for key, default in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default


def save_prediction_result(teks: str, pred: str, proba: dict, clean: str) -> None:
    """Simpan hasil prediksi terkini ke session_state."""
    st.session_state["ulasan_input"] = teks
    st.session_state["hasil_prediksi"] = (pred, proba, clean, teks)


def clear_prediction_result() -> None:
    st.session_state["ulasan_input"] = ""
    st.session_state["hasil_prediksi"] = None


def add_to_history(teks: str, pred: str, proba: Dict[str, float]) -> None:
    """Tambahkan satu entri riwayat prediksi (terbaru di posisi awal)."""
    st.session_state["riwayat"].insert(
        0,
        {
            "waktu": datetime.datetime.now().strftime("%H:%M:%S"),
            "ulasan": teks,
            "sentimen": pred,
            "confidence": max(proba.values()) * 100,
        },
    )


def clear_history() -> None:
    st.session_state["riwayat"] = []
