"""Unit test untuk modul `src.preprocessing`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import highlight_kata, make_preprocessor  # noqa: E402


def test_highlight_kata_positif():
    hasil = highlight_kata("Laptop ini bagus dan kencang")
    assert '<span class="kata-positif">bagus</span>' in hasil
    assert '<span class="kata-positif">kencang</span>' in hasil


def test_highlight_kata_negatif():
    hasil = highlight_kata("Laptop ini lambat dan panas")
    assert '<span class="kata-negatif">lambat</span>' in hasil
    assert '<span class="kata-negatif">panas</span>' in hasil


def test_highlight_kata_netral_tidak_berubah():
    hasil = highlight_kata("Laptop ini punya RAM 16GB")
    assert hasil == "Laptop ini punya RAM 16GB"


def test_highlight_kata_mengabaikan_tanda_baca():
    hasil = highlight_kata("Sangat bagus, saya puas!")
    assert '<span class="kata-positif">bagus,</span>' in hasil
    assert '<span class="kata-positif">puas!</span>' in hasil


def test_preprocessing_tidak_membuang_info_angka():
    """Regresi: versi awal kode menghapus semua digit lewat regex,
    sehingga 'RAM 16GB' -> 'ram gb' (info angka hilang total). Sekarang
    angka diganti token 'angka' supaya sinyal keberadaan spesifikasi
    numerik tetap ada."""
    preprocess = make_preprocessor()
    hasil = preprocess("RAM 16GB dan SSD 512GB")
    assert "angka" in hasil
    assert "16" not in hasil  # nilai literal tetap tidak dipertahankan
    assert "gb" in hasil


def test_preprocessing_lowercase_dan_bersih():
    preprocess = make_preprocessor()
    hasil = preprocess("LAPTOP Ini SANGAT Bagus!!!")
    assert hasil == hasil.lower()
    assert "!" not in hasil
