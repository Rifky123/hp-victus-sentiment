"""
Test integrasi untuk modul `src.model`.

Catatan: test ini melatih pipeline TF-IDF + Naive Bayes sungguhan
(termasuk cross-validation 5-fold) di atas dataset asli, sehingga lebih
lambat daripada unit test biasa. Jalankan dengan:

    pytest tests/test_model.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from src.config import CV_FOLDS, SENTIMENT_LABELS  # noqa: E402
from src.model import (  # noqa: E402
    compute_vocabulary_exclusivity,
    evaluate_external_dataset,
    get_top_contributing_words,
    load_dataset,
    predict_sentiment,
    train_model,
)


@pytest.fixture(scope="module")
def artifacts():
    # st.cache_resource di luar konteks Streamlit runtime akan berjalan
    # sebagai fungsi biasa (tanpa caching), sehingga aman dipanggil di test.
    return train_model()


def test_load_dataset_menghapus_duplikat():
    df, info = load_dataset()
    assert info["duplikat_dihapus"] >= 0
    assert info["total_bersih"] == info["total_mentah"] - info["duplikat_dihapus"]
    assert len(df) == info["total_bersih"]
    # tidak boleh ada ulasan yang persis sama tersisa
    assert df["ulasan"].duplicated().sum() == 0


def test_model_menghasilkan_metrik_valid(artifacts):
    metrics = artifacts.metrics
    for key in ("accuracy", "precision", "recall", "f1"):
        assert 0.0 <= metrics[key] <= 1.0


def test_confusion_matrix_berukuran_3x3(artifacts):
    cm = artifacts.metrics["cm"]
    assert cm.shape == (3, 3)


def test_tidak_ada_kebocoran_train_test(artifacts):
    """Regresi: dataset lama punya banyak duplikat sehingga ulasan yang
    sama bisa muncul di train & test set sekaligus. Setelah dedup di
    `load_dataset`, seharusnya tidak ada lagi ulasan identik yang
    tersisa di dataset final."""
    df = artifacts.df
    assert df["ulasan"].duplicated().sum() == 0


def test_cross_validation_menghasilkan_dua_model(artifacts):
    cv = artifacts.cv_results
    assert "Naive Bayes" in cv
    assert "Logistic Regression" in cv
    for hasil in cv.values():
        assert 0.0 <= hasil["accuracy_mean"] <= 1.0
        assert hasil["accuracy_std"] >= 0.0


def test_predict_sentiment_mengembalikan_label_valid(artifacts):
    pred, proba, clean = predict_sentiment(
        "Laptop ini sangat kencang dan responsif, puas sekali", artifacts
    )
    assert pred in SENTIMENT_LABELS
    assert set(proba.keys()) == set(SENTIMENT_LABELS)
    assert abs(sum(proba.values()) - 1.0) < 1e-6
    assert isinstance(clean, str)


def test_vocabulary_exclusivity_diagnostic(artifacts):
    """Diagnostik ini menjelaskan kenapa akurasi bisa ~100%: sebagian
    besar kosakata dataset eksklusif ke satu kelas saja."""
    kosakata = artifacts.dataset_info["kosakata"]
    assert kosakata["total_kata_unik"] > 0
    assert 0 <= kosakata["n_eksklusif"] <= kosakata["total_kata_unik"]
    assert 0.0 <= kosakata["pct_eksklusif"] <= 100.0
    assert isinstance(kosakata["contoh_eksklusif"], list)


def test_evaluate_external_dataset_menghasilkan_metrik_valid(artifacts):
    """Validasi eksternal pakai 200 ulasan asli (PRDECT-ID) di luar training."""
    hasil = evaluate_external_dataset(artifacts)
    assert hasil["n"] == 200
    for key in ("accuracy", "precision", "recall", "f1"):
        assert 0.0 <= hasil[key] <= 1.0
    assert hasil["cm"].shape == (3, 3)
    assert hasil["n_diprediksi_netral"] >= 0


def test_top_contributing_words_relevan(artifacts):
    kata_kontributor = get_top_contributing_words(
        "Laptop ini sangat kencang, kipasnya juga tidak berisik", artifacts, top_n=5
    )
    assert isinstance(kata_kontributor, list)
    assert len(kata_kontributor) <= 5
    for kata, skor in kata_kontributor:
        assert isinstance(kata, str)
        assert isinstance(skor, float)
