# Aplikasi Web Deteksi Spam

Aplikasi web berbasis Flask untuk mendeteksi pesan spam menggunakan model pembelajaran mesin. Proyek ini menggunakan Naive Bayes classifier dan CountVectorizer dengan n-gram untuk memprediksi apakah sebuah pesan adalah spam atau bukan.

---

## Daftar Isi
- [Fitur](#fitur)
- [Struktur Proyek](#struktur-proyek)
- [Kebutuhan](#kebutuhan)
- [Instalasi](#instalasi)
- [Penggunaan](#penggunaan)
- [Dataset](#dataset)

---

## Fitur
- Praproses teks dengan:
  - Mengubah teks menjadi huruf kecil.
  - Menghapus tanda baca dan angka.
  - Menghapus kata-kata umum (stop words).
- Ekstraksi fitur menggunakan `CountVectorizer` dengan bigram.
- Melatih model Naive Bayes untuk deteksi spam.
- Antarmuka web untuk input pengguna dan prediksi.

---

## Struktur Proyek
```plaintext
.
├── app.py                 # File aplikasi Flask
├── spam_detector.py       # File untuk pelatihan model
├── templates/             # Folder untuk menyimpan file HTML
│   └── index.html         # Halaman utama aplikasi
├── static/                # Folder untuk menyimpan file statis
│   └── style.css          # File CSS untuk tampilan
├── spam_model.pkl         # File model terlatih
├── vectorizer.pkl         # File vectorizer terlatih
└── spam.csv               # Dataset untuk melatih model
```

---

## Kebutuhan
- Python 3.8 atau lebih baru
- Library Python berikut:
  - Flask
  - joblib
  - nltk
  - scikit-learn
  - pandas

---

## Instalasi
1. **Clone repositori ini**:
   ```bash
   git clone https://github.com/Jezz76/EmailClassifier.git
   cd spam-detector
   ```

2. **Install dependensi**:
   Gunakan `pip` untuk menginstal semua dependensi.
   ```bash
   pip install -r requirements.txt
   ```

3. **Jalankan aplikasi**:
   ```bash
   python app.py
   ```

4. **Aplikasi akan berjalan di**.
```bash
 http://127.0.0.1:5000
   ```

---

## Penggunaan
- Masukkan pesan yang ingin Anda deteksi pada form di halaman utama.
- Klik tombol **Deteksi**.
- Hasil akan ditampilkan, apakah pesan tersebut **spam** atau **bukan spam**.

---

## Dataset
Dataset yang digunakan adalah file `spam.csv`, yang berisi pesan-pesan teks dengan label **spam** dan **ham** (bukan spam). Dataset ini telah dipraproses menggunakan fungsi di file `spam_detector.py`.

---

