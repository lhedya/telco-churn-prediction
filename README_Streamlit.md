# 📡 Telco Churn Prediction — Streamlit Deployment

Aplikasi ini men-deploy model **Random Forest** dari `Telco_Churn_Final_Combined.ipynb` ke web app interaktif.

## ✅ Validasi yang Sudah Dilakukan

Sebelum diserahkan, saya sudah:
1. **Replikasi pipeline persis** sesuai notebook kamu (load data → cleaning → feature engineering → encoding → SMOTE → scaling → training)
2. **Validasi metrik model** — hasil saya (AUC 0.834, Accuracy 76.72%, Precision 54.96%, Recall 68.18%, F1 60.86%) **cocok dengan slide PPT kamu** (AUC 0.834, Accuracy 76.70%, Precision 55.0%, Recall 68.20%, F1 60.90%) — selisih hanya pembulatan
3. **Test logic prediksi** dengan 3 skenario: profil High Risk (94.3% churn probability), Low Risk (1.2%), dan edge case tanpa internet service (3.4%) — semua berjalan tanpa error
4. **Jalankan aplikasi Streamlit** secara nyata — HTTP 200, tidak ada error di log

## 📁 File yang Diserahkan

| File | Kegunaan |
|---|---|
| `streamlit_app.py` | Aplikasi utama |
| `rf_model.pkl` | Model Random Forest (manual params, sesuai notebook) |
| `scaler.pkl` | StandardScaler yang sama dipakai saat training |
| `feature_cols.pkl` | Urutan 26 kolom fitur — wajib agar prediksi konsisten |
| `label_encoders.pkl` | Mapping LabelEncoder untuk 12 kolom biner/multi-kategori |
| `requirements.txt` | Dependency untuk Streamlit Cloud |

## 🚀 Cara Deploy

### Opsi 1: Streamlit Community Cloud (gratis, paling mudah)
1. Buat repository baru di GitHub, upload kelima file di atas
2. Buka [share.streamlit.io](https://share.streamlit.io) → Sign in dengan GitHub
3. Klik **New app** → pilih repo → pilih file `streamlit_app.py` → Deploy
4. Tunggu 1-2 menit, app akan online dengan URL publik (bisa dimasukkan ke CV/LinkedIn)

### Opsi 2: Jalankan Lokal Dulu (untuk testing)
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## 🎯 Fitur Aplikasi

- **Form input lengkap** — 19 field sesuai semua kolom dataset asli (demografi, layanan, kontrak, billing)
- **Prediksi real-time** — probabilitas churn + gauge chart visual
- **Rekomendasi otomatis** — saran retensi spesifik berdasarkan profil pelanggan (kontrak month-to-month → migrasi tahunan, electronic check → auto-pay, dst), sesuai insight dari notebook
- **Transparansi model** — expander untuk lihat detail 26 fitur yang dikirim ke model (bagus untuk demo ke assessor yang ingin lihat "di balik layar")

## 📌 Untuk Ditambahkan ke PPT

Setelah app online, screenshot tampilan form dan hasil prediksinya, masukkan sebagai slide baru setelah slide "Business Recommendation" — misalnya berjudul **"Model Deployment: Real-Time Churn Prediction Tool"**. Sertakan juga URL Streamlit Cloud kamu di slide tersebut atau di slide "Terima Kasih" bersama kontak.
