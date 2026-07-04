"""
Konfigurasi terpusat untuk aplikasi.

Semua nilai statis (path, palet warna, kamus kata kunci, hyperparameter
model, dan info tim) didefinisikan di sini agar tidak tersebar di
berbagai file dan mudah diubah tanpa menyentuh logika aplikasi.
"""

from __future__ import annotations

from pathlib import Path

# ─── PATH ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATASET_PATH = DATA_DIR / "dataset.csv"
EXTERNAL_VALIDATION_PATH = DATA_DIR / "external_validation.csv"
EXTERNAL_VALIDATION_CITATION = (
    "Sutoyo, R., Achmad, S., Chowanda, A., Andangsari, E. W., & Isa, S. M. "
    "(2022). PRDECT-ID: Indonesian product reviews dataset for emotions "
    "classification tasks. Data in Brief, 44, 108554. "
    "(subset kategori 'Computers and Laptops', 200 ulasan asli Tokopedia, "
    "lisensi MIT)"
)

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────
PAGE_CONFIG = {
    "page_title": "Analisis Sentimen HP Victus",
    "page_icon": "💻",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

APP_TITLE = "💻 Analisis Sentimen Ulasan Laptop HP Victus"
APP_SUBTITLE = (
    "Metode Naive Bayes + TF-IDF &nbsp;|&nbsp; "
    "Universitas Dian Nuswantoro Semarang &nbsp;|&nbsp; 2025"
)
FOOTER_TEXT = (
    "💻 Analisis Sentimen HP Victus &nbsp;·&nbsp; Naive Bayes + TF-IDF "
    "&nbsp;·&nbsp; Universitas Dian Nuswantoro Semarang &nbsp;·&nbsp; 2025"
)

# ─── PALET WARNA ─────────────────────────────────────────────────────────
COLORS = {
    "primary": "#667eea",
    "secondary": "#764ba2",
    "positif": "#28a745",
    "negatif": "#dc3545",
    "netral": "#ffc107",
}

METRIC_COLORS = {
    "accuracy": "#667eea",
    "precision": "#f093fb",
    "recall": "#4facfe",
    "f1": "#43e97b",
}

SENTIMENT_LABELS = ["positif", "negatif", "netral"]
EMOJI = {"positif": "😊", "negatif": "😞", "netral": "😐"}
RESULT_CSS_CLASS = {
    "positif": "result-positif",
    "negatif": "result-negatif",
    "netral": "result-netral",
}
BADGE_CSS_CLASS = {
    "positif": "badge-positif",
    "negatif": "badge-negatif",
    "netral": "badge-netral",
}

# ─── KAMUS KATA KUNCI SENTIMEN (untuk highlight, bukan untuk model) ──────
KATA_POSITIF = [
    "bagus", "kencang", "cepat", "puas", "mantap", "recommended", "oke",
    "baik", "nyaman", "worth", "murah", "canggih", "keren", "jos", "tahan",
    "awet", "dingin", "responsif", "jernih", "tajam", "ringan", "mulus",
    "enak", "memuaskan", "terbaik",
]
KATA_NEGATIF = [
    "lambat", "panas", "berisik", "boros", "rusak", "kecewa", "jelek",
    "buruk", "lemot", "lag", "overheat", "baterai", "mahal", "berat",
    "lama", "macet", "masalah", "kacau", "gagal", "mati", "tidak",
    "kurang", "susah", "ribet", "parah",
]

# ─── HYPERPARAMETER MODEL ────────────────────────────────────────────────
MODEL_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "tfidf_max_features": 3000,
    "tfidf_ngram_range": (1, 2),
    "naive_bayes_alpha": 0.5,
    "logreg_max_iter": 1000,
}

# Jumlah fold untuk stratified k-fold cross-validation. Dipakai untuk
# mendapatkan estimasi performa yang lebih stabil daripada satu kali
# train/test split saja (yang rawan bias pada dataset kecil).
CV_FOLDS = 5

# Model yang dibandingkan pada tahap cross-validation. Naive Bayes tetap
# menjadi model utama yang dipakai aplikasi (sesuai tujuan tugas), tapi
# Logistic Regression disertakan sebagai baseline pembanding di tab
# Evaluasi supaya performa NB punya konteks.
BASELINE_MODEL_NAME = "Naive Bayes (utama)"
COMPARISON_MODEL_NAME = "Logistic Regression (pembanding)"

# ─── ULASAN UJI KETAHANAN (robustness check, tab Evaluasi) ──────────────
# Kalimat-kalimat ini SENGAJA ditulis dengan gaya lebih natural & ambigu
# (bukan berasal dari proses pembuatan dataset asli), untuk menunjukkan
# bahwa performa ~100% pada data holdout/CV adalah karena kosakata
# dataset yang sangat "bersih" per kelas, bukan karena model benar-benar
# sempurna menggeneralisasi ke gaya bahasa ulasan dunia nyata.
UJI_KETAHANAN = [
    {
        "teks": "Lumayan sih buat kerja sehari-hari, walau kadang suka nge-lag pas buka banyak tab",
        "catatan": "campuran/ambigu",
    },
    {
        "teks": "Awalnya kecewa krn dus penyok pas nyampe, tapi laptopnya sendiri oke aja",
        "catatan": "campuran/ambigu",
    },
    {
        "teks": "Ya standar aja lah, sesuai harga, ga ada yang spesial tapi juga ga jelek",
        "catatan": "netral-ambigu",
    },
    {
        "teks": "Fan noise nya lumayan ganggu pas malem2 kerja, tapi grafiknya emang joss buat AAA games",
        "catatan": "campuran",
    },
    {
        "teks": "worth it kok buat yang budget segini, meski chargernya agak gede",
        "catatan": "positif-ambigu",
    },
]

# ─── CONTOH ULASAN (tab prediksi) ────────────────────────────────────────
CONTOH_ULASAN = {
    "😊 Positif": (
        "Laptop ini sangat kencang dan layarnya tajam, "
        "sangat puas dengan pembelian ini"
    ),
    "😞 Negatif": (
        "Kipas berisik dan baterai cepat habis, tidak sesuai harapan saya"
    ),
    "😐 Netral": (
        "Laptop ini memiliki RAM 16GB dan SSD 512GB sesuai spesifikasi"
    ),
}

# ─── TIM PENELITI ────────────────────────────────────────────────────────
TIM_PENELITI = [
    {"nama": "Maulana Tsaqif M.", "nim": "A11.2024.15938", "peran": "👑 Ketua"},
    {"nama": "Rifky Maulana", "nim": "A11.2024.15950", "peran": "👤 Anggota"},
    {"nama": "M. Juan T. P.", "nim": "A11.2024.15937", "peran": "👤 Anggota"},
    {"nama": "Putra Karisma", "nim": "A11.2024.15628", "peran": "👤 Anggota"},
    {"nama": "Teduh Firmansyah", "nim": "11.2024.16021", "peran": "👤 Anggota"},
]
