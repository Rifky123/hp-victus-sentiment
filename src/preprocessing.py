"""
Modul preprocessing teks bahasa Indonesia.

Berisi pipeline pembersihan teks (case folding, filtering karakter,
stopword removal, stemming) menggunakan PySastrawi, serta fungsi
highlight kata kunci sentimen untuk tampilan UI.
"""

from __future__ import annotations

import re
from typing import Tuple

import streamlit as st
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from src.config import KATA_NEGATIF, KATA_POSITIF

_DIGIT_PATTERN = re.compile(r"\d+")
_NON_ALPHA_PATTERN = re.compile(r"[^a-z\s]")
_MULTI_SPACE_PATTERN = re.compile(r"\s+")

# Token pengganti untuk semua rangkaian digit (mis. "16GB" -> "angka gb").
# Versi awal kode ini MEMBUANG angka sepenuhnya lewat regex
# `[^a-z\s]`, sehingga info seperti "RAM 16GB" / "harga 15 juta" hilang
# total dan tidak bisa jadi sinyal bagi model (padahal ulasan kelas
# netral sering berisi spesifikasi berupa angka). Dengan menggantinya
# jadi token "angka", model masih tahu "di sini ada nilai numerik"
# tanpa membuat fitur terlalu spesifik/sparse akibat angka yang variatif.
_NUMBER_TOKEN = " angka "


@st.cache_resource(show_spinner=False)
def load_nlp_tools() -> Tuple[object, object]:
    """Muat & cache stopword remover dan stemmer PySastrawi.

    Objek-objek ini relatif berat untuk diinisialisasi sehingga
    di-cache oleh Streamlit agar hanya dibuat sekali per sesi server.
    """
    stop_factory = StopWordRemoverFactory()
    stemmer_factory = StemmerFactory()
    return stop_factory.create_stop_word_remover(), stemmer_factory.create_stemmer()


def clean_text(text: str, stopword_remover, stemmer) -> str:
    """Bersihkan satu string teks ulasan.

    Tahapan: case folding -> hapus karakter non-alfabet -> normalisasi
    spasi -> stopword removal -> stemming.
    """
    text = text.lower()
    text = _DIGIT_PATTERN.sub(_NUMBER_TOKEN, text)
    text = _NON_ALPHA_PATTERN.sub(" ", text)
    text = _MULTI_SPACE_PATTERN.sub(" ", text).strip()
    text = stopword_remover.remove(text)
    text = stemmer.stem(text)
    return text


def make_preprocessor():
    """Buat fungsi preprocessing siap pakai (tanpa perlu passing tools)."""
    stopword_remover, stemmer = load_nlp_tools()

    def _preprocess(text: str) -> str:
        return clean_text(text, stopword_remover, stemmer)

    return _preprocess


def highlight_kata(teks: str) -> str:
    """Bungkus kata positif/negatif dalam `teks` dengan span HTML berwarna."""
    words = teks.split()
    result = []
    for w in words:
        wl = w.lower().strip(".,!?")
        if wl in KATA_POSITIF:
            result.append(f'<span class="kata-positif">{w}</span>')
        elif wl in KATA_NEGATIF:
            result.append(f'<span class="kata-negatif">{w}</span>')
        else:
            result.append(w)
    return " ".join(result)
