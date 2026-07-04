"""
Modul model machine learning: pembersihan data, cross-validation,
training model final, dan prediksi.

Model utama yang dipakai aplikasi adalah Multinomial Naive Bayes dengan
representasi teks TF-IDF. Logistic Regression disertakan sebagai model
pembanding (baseline) agar performa Naive Bayes punya konteks, dan
evaluasi dilakukan dengan dua cara:

1. **Stratified K-Fold Cross-Validation** -- estimasi performa yang lebih
   stabil karena dihitung dari beberapa kali split, bukan cuma sekali.
   TF-IDF di-*fit* ulang di dalam setiap fold (lewat `Pipeline`) supaya
   tidak ada kebocoran informasi dari data validasi ke proses training.
2. **Holdout train/test split (80/20)** -- dipakai untuk menghasilkan
   model final yang aktif dipakai aplikasi untuk prediksi interaktif,
   beserta confusion matrix & classification report yang mudah dibaca.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.config import (
    CV_FOLDS,
    DATASET_PATH,
    EXTERNAL_VALIDATION_PATH,
    MODEL_CONFIG,
    SENTIMENT_LABELS,
)
from src.preprocessing import make_preprocessor


@dataclass
class ModelArtifacts:
    """Bungkusan seluruh objek hasil training yang dipakai UI."""

    model: MultinomialNB
    tfidf: TfidfVectorizer
    preprocess_fn: Callable[[str], str]
    df: pd.DataFrame
    metrics: Dict
    cv_results: Dict = field(default_factory=dict)
    dataset_info: Dict = field(default_factory=dict)


def load_dataset(path=DATASET_PATH) -> Tuple[pd.DataFrame, Dict]:
    """Muat dataset ulasan dari CSV, buang baris kosong & duplikat.

    Dataset mentah ternyata mengandung banyak ulasan yang duplikat
    persis (>10% dari total baris). Jika dibiarkan, `train_test_split`
    bisa menempatkan salinan ulasan yang sama di data *train* maupun
    *test* sekaligus -- model jadi "menghafal" alih-alih belajar
    generalisasi, dan metrik evaluasi (akurasi dsb.) jadi menggembung
    secara tidak realistis. Karena itu deduplikasi dilakukan di sini,
    sebelum data displit, dan informasinya dicatat agar transparan
    (ditampilkan di UI, bukan disembunyikan).
    """
    df_raw = pd.read_csv(path)
    df_raw.dropna(inplace=True)

    total_mentah = len(df_raw)
    df_clean = df_raw.drop_duplicates(subset=["ulasan"]).reset_index(drop=True)
    total_bersih = len(df_clean)

    info = {
        "total_mentah": total_mentah,
        "total_bersih": total_bersih,
        "duplikat_dihapus": total_mentah - total_bersih,
    }
    return df_clean, info


def _compute_metrics(y_test, y_pred) -> Dict:
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "cm": confusion_matrix(y_test, y_pred, labels=SENTIMENT_LABELS),
        "report": classification_report(y_test, y_pred, output_dict=True),
    }


def _build_candidate_models() -> Dict[str, object]:
    """Model yang dievaluasi lewat cross-validation."""
    return {
        "Naive Bayes": MultinomialNB(alpha=MODEL_CONFIG["naive_bayes_alpha"]),
        "Logistic Regression": LogisticRegression(
            max_iter=MODEL_CONFIG["logreg_max_iter"],
            random_state=MODEL_CONFIG["random_state"],
        ),
    }


def run_cross_validation(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Evaluasi tiap model kandidat dengan Stratified K-Fold CV.

    Setiap model dibungkus dalam `Pipeline(TfidfVectorizer -> classifier)`
    sehingga vectorizer di-*fit* ulang di setiap fold hanya dari data
    training fold tersebut -- mencegah kebocoran informasi dari data
    validasi ke proses fitting TF-IDF.

    Returns
    -------
    dict
        `{nama_model: {"accuracy_mean", "accuracy_std", "precision_mean",
        ..., "f1_mean", "f1_std"}}`
    """
    X = df["ulasan_bersih"]
    y = df["sentimen"]

    cv = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=MODEL_CONFIG["random_state"]
    )
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision_weighted",
        "recall": "recall_weighted",
        "f1": "f1_weighted",
    }

    results: Dict[str, Dict[str, float]] = {}
    for name, clf in _build_candidate_models().items():
        pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=MODEL_CONFIG["tfidf_max_features"],
                        ngram_range=MODEL_CONFIG["tfidf_ngram_range"],
                    ),
                ),
                ("clf", clf),
            ]
        )
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring)

        model_result = {}
        for metric in scoring:
            key = f"test_{metric}"
            model_result[f"{metric}_mean"] = float(np.mean(scores[key]))
            model_result[f"{metric}_std"] = float(np.std(scores[key]))
        results[name] = model_result

    return results


@st.cache_resource(show_spinner=False)
def train_model(data_path=DATASET_PATH) -> ModelArtifacts:
    """Latih & evaluasi pipeline TF-IDF + Multinomial Naive Bayes.

    Alur:
    1. Muat dataset + deduplikasi.
    2. Preprocessing seluruh teks (case folding, stopword, stemming).
    3. Cross-validation (5-fold) untuk Naive Bayes vs Logistic Regression
       -- estimasi performa yang lebih dapat dipercaya.
    4. Holdout split 80/20 untuk melatih model final yang dipakai
       aplikasi secara interaktif, lengkap dengan confusion matrix &
       classification report.

    Hasil di-cache oleh Streamlit (`st.cache_resource`) sehingga seluruh
    proses ini hanya berjalan sekali per sesi server.
    """
    df, dataset_info = load_dataset(data_path)
    preprocess_fn = make_preprocessor()

    df = df.copy()
    df["ulasan_bersih"] = df["ulasan"].apply(preprocess_fn)

    # -- Cross-validation (estimasi performa yang lebih stabil) ---------
    cv_results = run_cross_validation(df)

    # -- Holdout split untuk model final yang dipakai aplikasi ----------
    X_train, X_test, y_train, y_test = train_test_split(
        df["ulasan_bersih"],
        df["sentimen"],
        test_size=MODEL_CONFIG["test_size"],
        random_state=MODEL_CONFIG["random_state"],
        stratify=df["sentimen"],
    )

    tfidf = TfidfVectorizer(
        max_features=MODEL_CONFIG["tfidf_max_features"],
        ngram_range=MODEL_CONFIG["tfidf_ngram_range"],
    )
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    model = MultinomialNB(alpha=MODEL_CONFIG["naive_bayes_alpha"])
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    metrics = _compute_metrics(y_test, y_pred)
    metrics["n_train"] = len(X_train)
    metrics["n_test"] = len(X_test)

    dataset_info = dict(dataset_info)
    dataset_info["kosakata"] = compute_vocabulary_exclusivity(df)

    return ModelArtifacts(
        model=model,
        tfidf=tfidf,
        preprocess_fn=preprocess_fn,
        df=df,
        metrics=metrics,
        cv_results=cv_results,
        dataset_info=dataset_info,
    )


def compute_vocabulary_exclusivity(df: pd.DataFrame) -> Dict:
    """Diagnostik: seberapa besar kosakata dataset eksklusif per kelas.

    Ini menjelaskan MENGAPA akurasi model bisa mendekati 100% bahkan
    setelah dedup + cross-validation yang benar: jika sebagian besar
    kata akar (hasil stemming) hanya pernah muncul pada satu kelas
    sentimen saja, klasifikasi menjadi hampir trivial bagi model
    berbasis bag-of-words apa pun -- bukan karena Naive Bayes "sempurna"
    menggeneralisasi, melainkan karena kosakata dataset yang sangat
    sempit & bersih per kelas (indikasi data sintetis/formulaic).
    """
    kata_ke_kelas: Dict[str, set] = {}
    for _, row in df.iterrows():
        for w in set(row["ulasan_bersih"].split()):
            kata_ke_kelas.setdefault(w, set()).add(row["sentimen"])

    total = len(kata_ke_kelas)
    eksklusif = [(w, next(iter(s))) for w, s in kata_ke_kelas.items() if len(s) == 1]
    n_eksklusif = len(eksklusif)

    return {
        "total_kata_unik": total,
        "n_eksklusif": n_eksklusif,
        "pct_eksklusif": (n_eksklusif / total * 100) if total else 0.0,
        "contoh_eksklusif": eksklusif[:12],
    }


def predict_sentiment(text: str, artifacts: ModelArtifacts):
    """Prediksi sentimen sebuah teks ulasan.

    Returns
    -------
    pred : str
        Label sentimen hasil prediksi ('positif' | 'negatif' | 'netral').
    proba : dict[str, float]
        Probabilitas per kelas.
    clean : str
        Teks setelah melalui tahap preprocessing.
    """
    clean = artifacts.preprocess_fn(text)
    vec = artifacts.tfidf.transform([clean])
    pred = artifacts.model.predict(vec)[0]
    proba = dict(zip(artifacts.model.classes_, artifacts.model.predict_proba(vec)[0]))
    return pred, proba, clean


def get_top_contributing_words(
    text: str, artifacts: ModelArtifacts, top_n: int = 5
) -> List[Tuple[str, float]]:
    """Kata-kata yang benar-benar mendorong prediksi model (bukan kamus manual).

    Cara kerja: untuk kelas yang diprediksi, Naive Bayes punya
    `feature_log_prob_` (log-probabilitas tiap term muncul pada kelas
    tersebut). Term-term dalam teks input diberi skor
    `tfidf_weight x log_prob`, lalu diurutkan -- ini mendekati kontribusi
    riil tiap kata terhadap keputusan model, berbeda dari fitur
    "highlight kata" di UI yang memakai kamus kata kunci buatan tangan
    dan hanya untuk ilustrasi, bukan cerminan langsung dari model.
    """
    clean = artifacts.preprocess_fn(text)
    vec = artifacts.tfidf.transform([clean])
    pred = artifacts.model.predict(vec)[0]
    class_idx = list(artifacts.model.classes_).index(pred)

    feature_names = artifacts.tfidf.get_feature_names_out()
    log_probs = artifacts.model.feature_log_prob_[class_idx]

    nonzero_idx = vec.nonzero()[1]
    if len(nonzero_idx) == 0:
        return []

    scored = [
        (feature_names[i], float(vec[0, i] * log_probs[i])) for i in nonzero_idx
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_n]


@st.cache_resource(show_spinner=False)
def evaluate_external_dataset(
    _artifacts: ModelArtifacts, path=EXTERNAL_VALIDATION_PATH
) -> Dict:
    """Uji model pada data ulasan ASLI dari luar proses training (out-of-distribution).

    Berbeda dari metrik holdout/CV di atas (yang dihitung dari data yang
    "berasal dari keluarga distribusi" yang sama dengan data training),
    fungsi ini mengukur performa pada 200 ulasan produk komputer/laptop
    ASLI dari Tokopedia (dataset publik PRDECT-ID, bukan buatan sendiri).
    Skor di sini jauh lebih mencerminkan performa dunia nyata, dan lebih
    kredibel untuk menjawab pertanyaan "kenapa akurasi 100%?" daripada
    metrik in-distribution semata.

    Catatan: dataset eksternal ini hanya berlabel positif/negatif (tanpa
    netral), jadi metrik dihitung hanya dari baris berlabel tersebut.
    Prediksi model yang keliru menjadi 'netral' pada data ini otomatis
    terhitung sebagai salah (bukan dikecualikan).
    """
    df_ext = pd.read_csv(path)
    df_ext.dropna(inplace=True)

    y_true = df_ext["sentimen"]
    y_pred = [
        predict_sentiment(teks, _artifacts)[0] for teks in df_ext["ulasan"]
    ]

    labels_evaluasi = ["positif", "negatif"]
    return {
        "n": len(df_ext),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, labels=labels_evaluasi, average="weighted", zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, labels=labels_evaluasi, average="weighted", zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, labels=labels_evaluasi, average="weighted", zero_division=0
        ),
        "cm": confusion_matrix(y_true, y_pred, labels=SENTIMENT_LABELS),
        "n_diprediksi_netral": sum(1 for p in y_pred if p == "netral"),
    }
