# Data Mining Template - Naive Bayes (untuk Mahasiswa)

Template ini dibuat untuk praktikum/tugas data mining berbasis **FastAPI**.
Fokus utamanya: mahasiswa mengerjakan **logika backend** (training, preprocessing, prediksi, evaluasi), karena **UI/frontend sudah disediakan**.

## Tujuan Template

- Memberikan starter project yang siap jalan untuk eksperimen algoritma data mining.
- Mengurangi waktu setup (auth, halaman admin/user, upload dataset, penyimpanan model sudah ada).
- Membantu mahasiswa fokus ke implementasi fitur inti machine learning, terutama **Naive Bayes**.

## Yang Sudah Disediakan

- Struktur project FastAPI + Jinja template.
- Sistem login sederhana (role `admin` dan `user`).
- Halaman admin untuk:
  - upload dataset CSV/Excel,
  - training model,
  - melihat daftar model dan metrik,
  - manajemen user.
- Halaman user untuk:
  - melakukan prediksi,
  - melihat riwayat hasil prediksi.
- Starter service untuk algoritma di `app/services/ml/algorithms.py`:
  - `naive_bayes`
  - `knn`
  - `kmeans`
  - `fpgrowth`
- Penyimpanan model artifact menggunakan `joblib`.

## Fokus Pengerjaan Mahasiswa

Mahasiswa diharapkan mengembangkan/menyempurnakan logika backend, contohnya:

- Validasi input training/prediksi yang lebih kuat.
- Perbaikan preprocessing (handling missing value, encoding, scaling, feature selection).
- Evaluasi model lebih lengkap (mis. per-class metrics, cross-validation, dll).
- Penyesuaian pipeline Naive Bayes sesuai karakteristik dataset.
- Error handling dan pesan error yang lebih informatif.
- Eksperimen perbandingan algoritma.

## Struktur Folder Penting

```text
app/
  core/                 # config, db, dependency
  models/               # ORM models (user, dataset, model_artifact, prediction)
  routers/              # route auth/admin/user/pages
  services/
    ml/algorithms.py    # starter implementasi algoritma
    training.py         # proses train + simpan model
    predict.py          # proses prediksi + simpan history
  templates/            # frontend (sudah disediakan)
  static/               # asset CSS
data/
  datasets/             # file dataset upload
  models/               # model hasil training (.joblib)
scripts/
  init_db.py            # inisialisasi DB + seed user default
```

## Cara Menjalankan

1. Buat virtual environment dan install dependency:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Inisialisasi database + user awal:

```bash
python scripts/init_db.py
```

3. Jalankan aplikasi:

```bash
uvicorn app.main:app --reload
```

4. Buka di browser:

```text
http://127.0.0.1:8000
```

## Akun Default

- Admin: `admin` / `admin123`
- User: `user` / `user123`

> Disarankan ganti password default sebelum dipakai di kelas/lab.

## Alur Penggunaan Singkat

1. Login sebagai **admin**.
2. Upload dataset.
3. Train model (minimal Naive Bayes).
4. Login sebagai **user**.
5. Pilih model lalu lakukan prediksi.
6. Lihat hasil dan riwayat prediksi.

## Catatan Pengembangan

- File utama pengembangan algoritma: `app/services/ml/algorithms.py`.
- Integrasi training ada di: `app/services/training.py`.
- Integrasi prediksi ada di: `app/services/predict.py`.
- Jika menambah algoritma baru, pastikan alur train dan predict ikut diperbarui.

Silakan gunakan template ini sebagai pondasi. Frontend sudah siap, jadi energi utama bisa difokuskan ke **backend logic** dan pemahaman konsep data mining.
