# 💻 Analisis Sentimen Ulasan Laptop HP Victus

**Naive Bayes + TF-IDF | Aplikasi Web Streamlit**
Tugas Akhir Semester — Mata Kuliah Pemrograman Mesin
Universitas Dian Nuswantoro Semarang | 2025

---

## 📖 Deskripsi

Aplikasi web interaktif untuk menganalisis sentimen (**positif / negatif /
netral**) dari ulasan laptop HP Victus berbahasa Indonesia. Model
diklasifikasi menggunakan **Multinomial Naive Bayes** dengan representasi
teks **TF-IDF**, dan preprocessing bahasa Indonesia (stopword removal +
stemming) menggunakan **PySastrawi**.

## ✨ Fitur

| Fitur | Deskripsi |
|---|---|
| 🔍 **Prediksi Sentimen** | Input teks ulasan → hasil positif/negatif/netral + probabilitas per kelas |
| 📂 **Prediksi Batch** | Upload CSV berisi banyak ulasan, diproses & diunduh sekaligus |
| 📜 **Riwayat Prediksi** | Riwayat prediksi selama sesi berjalan, bisa diunduh sebagai CSV |
| 📊 **Evaluasi Model** | Accuracy, Precision, Recall, F1-Score, Confusion Matrix, Classification Report |
| 📈 **Visualisasi** | Pie chart & bar chart distribusi sentimen, word cloud per kelas, grafik perbandingan metrik |
| 📋 **Dataset Viewer** | Jelajahi dan filter dataset ulasan |

## 🗂️ Struktur Proyek

```
Project_uas/
├── app.py                        # Entry point Streamlit (orkestrasi halaman)
├── src/
│   ├── config.py                 # Konfigurasi terpusat (path, warna, kamus kata, hyperparameter)
│   ├── styles.py                 # CSS kustom
│   ├── preprocessing.py          # Pipeline NLP (stopword removal, stemming, highlight kata)
│   ├── model.py                  # Training, evaluasi, dan prediksi model
│   ├── state.py                  # Manajemen st.session_state
│   └── components/               # Komponen UI per tab
│       ├── sidebar.py
│       ├── tab_predict.py
│       ├── tab_history.py
│       ├── tab_evaluation.py
│       ├── tab_visualization.py
│       └── tab_dataset.py
├── data/
│   ├── dataset.csv               # Dataset training (500 baris, 431 setelah dedup)
│   └── external_validation.csv   # 200 ulasan ASLI Tokopedia (PRDECT-ID) untuk validasi eksternal
├── tests/
│   ├── test_preprocessing.py     # Unit test preprocessing
│   └── test_model.py             # Test integrasi training & prediksi
├── .streamlit/
│   └── config.toml               # Tema warna aplikasi
├── requirements.txt              # Dependency produksi
├── requirements-dev.txt          # Dependency pengembangan (pytest, black, flake8)
├── pyproject.toml                # Metadata proyek & konfigurasi tools
├── Makefile                      # Shortcut perintah umum
├── LICENSE
└── README.md
```

## 🚀 Cara Menjalankan

### 1. Buat virtual environment (disarankan)

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
# atau, untuk kebutuhan pengembangan (testing, linting):
pip install -r requirements-dev.txt
```

### 3. Jalankan aplikasi

```bash
streamlit run app.py
# atau:
make run
```

Aplikasi akan terbuka di `http://localhost:8501`.

## 🧪 Menjalankan Test

```bash
pytest tests/ -v
# atau:
make test
```

## 🧰 Stack Teknologi

- **Python + Streamlit** — antarmuka web
- **Scikit-learn** — Multinomial Naive Bayes + TF-IDF Vectorizer
- **PySastrawi** — preprocessing NLP bahasa Indonesia (stopword removal + stemming)
- **Matplotlib + WordCloud** — visualisasi data
- **Pandas / NumPy** — manipulasi data

## 🔄 Alur Sistem

```
Input Ulasan
    → Preprocessing (case folding, filtering, stopword removal, stemming)
    → TF-IDF Vectorization
    → Multinomial Naive Bayes
    → Output Sentimen (positif / negatif / netral) + Probabilitas
```

## ⚙️ Konfigurasi Model

| Parameter | Nilai |
|---|---|
| Algoritma | Multinomial Naive Bayes (alpha = 0.5) |
| Representasi teks | TF-IDF, max_features = 3000, ngram_range = (1, 2) |
| Split data | 80% training / 20% testing (stratified) |
| Validasi | Stratified 5-fold cross-validation (Naive Bayes vs Logistic Regression) |

## 🔍 Perbaikan Metodologi (v2.1)

Review awal menemukan beberapa potensi masalah metodologis. Berikut yang
sudah diperbaiki, didokumentasikan secara transparan (bukan disembunyikan):

| Temuan | Perbaikan |
|---|---|
| **69 dari 500 baris dataset adalah duplikat persis**, menyebabkan potensi kebocoran data antara train/test (25 dari ~97 data test ternyata juga ada di data train) | Deduplikasi dilakukan di `load_dataset()` sebelum split. Info jumlah duplikat yang dihapus ditampilkan transparan di UI (sidebar, tab Dataset, tab Evaluasi) |
| Preprocessing lama **membuang semua angka** (`RAM 16GB` → `ram gb`, info numerik hilang total) | Angka sekarang diganti token `angka` (`RAM 16GB` → `ram angka gb`), mempertahankan sinyal "ada spesifikasi numerik" tanpa membuat fitur terlalu spesifik |
| **Hanya satu kali train/test split** — metrik rawan bias karena kebetulan pembagian data | Ditambahkan **5-fold stratified cross-validation**, dilaporkan sebagai rata-rata ± standar deviasi, lebih dapat dipercaya daripada satu angka tunggal |
| **Tidak ada model pembanding** — sulit menilai apakah performa Naive Bayes "biasa saja" atau memang bagus | Logistic Regression ditambahkan sebagai baseline, dievaluasi dengan cross-validation yang sama, ditampilkan berdampingan di tab Evaluasi |
| Fitur *highlight kata* di UI memakai **kamus kata kunci manual**, tapi terkesan seperti mencerminkan cara kerja model | Label & disclaimer diperjelas ("ilustrasi berbasis kamus, bukan model"), ditambahkan fitur baru yang menampilkan kata-kata yang **benar-benar** berkontribusi ke prediksi (dihitung dari bobot TF-IDF × log-probabilitas Naive Bayes) |

### ⚠️ Catatan jujur: kenapa akurasi masih ~100% setelah semua perbaikan?

Setelah dedup dan cross-validation yang benar, akurasi model **tetap** di
kisaran 99–100% pada data holdout. Ini **bukan** bug. Aplikasi sekarang
membuktikan penyebabnya secara konkret lewat tiga diagnostik di tab
**Evaluasi**:

1. **🌍 Validasi eksternal pakai 200 ulasan ASLI (paling penting).**
   Model diuji pada 200 ulasan produk komputer/laptop **asli dari
   Tokopedia** (subset dataset publik **PRDECT-ID**, Sutoyo et al. 2022,
   *Data in Brief* — bukan data buatan sendiri, dan model belum pernah
   melihat data ini saat training). Hasilnya: **akurasi anjlok dari 100%
   (holdout) menjadi ~41.5%** pada data asli tersebut. Gap sebesar ini
   adalah bukti kuantitatif paling meyakinkan bahwa performa 100% di
   holdout adalah artefak dataset training, bukan cerminan kemampuan
   model yang sesungguhnya di dunia nyata.

2. **Diagnostik kemurnian kosakata.** **83.1%** dari seluruh kata akar
   (hasil stemming) dalam dataset training ternyata **hanya pernah
   muncul di satu kelas sentimen saja** — mis. kata *"game"* selalu di
   ulasan positif, *"bobot"* selalu di negatif, *"inci"* selalu di
   netral, tidak pernah tertukar. Kosakata sesempit dan sebersih ini
   membuat tugas klasifikasi nyaris trivial bagi model bag-of-words apa
   pun, terlepas dari algoritmanya — indikasi kuat dataset training
   bersifat sintetis/template, bukan ulasan bebas dari pengguna nyata.

3. **Uji ketahanan (robustness check) informal.** Lima kalimat ditulis
   manual dengan gaya lebih natural, campur aduk, dan ambigu, lalu diuji
   ke model. Hasilnya: confidence turun signifikan (44–94%, dibanding
   ~100% pada data holdout) dan sebagian prediksi meleset — pelengkap
   kualitatif untuk bukti kuantitatif di poin 1.

**Untuk laporan tugas akhir, kami merekomendasikan mencantumkan ketiga
bukti ini secara eksplisit** (screenshot tab Evaluasi sudah cukup),
sekaligus menyarankan pengumpulan/pelabelan data riil yang lebih banyak
sebagai pengembangan lanjutan. Kejujuran + bukti kuantitatif nyata soal
keterbatasan ini jauh lebih bernilai di mata penguji dibanding mengklaim
performa 100% sebagai bukti model yang sempurna — dan langsung menjawab
kecurigaan "kok bisa 100%?" sebelum ditanyakan.

## 👥 Tim Peneliti

| Peran | Nama | NIM |
|---|---|---|
| 👑 Ketua | Maulana Tsaqif M. | A11.2024.15938 |
| 👤 Anggota | Rifky Maulana | A11.2024.15950 |
| 👤 Anggota | M. Juan T. P. | A11.2024.15937 |
| 👤 Anggota | Putra Karisma | A11.2024.15628 |
| 👤 Anggota | Teduh Firmansyah | 11.2024.16021 |

## 📚 Sumber Data Eksternal

Dataset `data/external_validation.csv` (200 ulasan) diambil dari subset
kategori *"Computers and Laptops"* pada dataset publik **PRDECT-ID**
(lisensi MIT), dan digunakan murni untuk keperluan validasi/evaluasi
akademis:

> Sutoyo, R., Achmad, S., Chowanda, A., Andangsari, E. W., & Isa, S. M.
> (2022). PRDECT-ID: Indonesian product reviews dataset for emotions
> classification tasks. *Data in Brief*, *44*, 108554.
> https://doi.org/10.1016/j.dib.2022.108554
> Repositori: https://github.com/rhiosutoyo/PRDECT-ID-Indonesian-Product-Reviews-Dataset

## 📄 Lisensi

Proyek ini dilisensikan di bawah [MIT License](LICENSE).

---
*Universitas Dian Nuswantoro Semarang · Fakultas Ilmu Komputer · 2025*
