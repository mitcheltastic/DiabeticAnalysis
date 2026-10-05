# 🩺 Defensible Machine Learning for Early Diabetes Screening
### *Benchmarking Imputation Paradigms & Exposing Class-Conditional Target Leakage in the PIMA Cohort*

> **Academic Research & Journal Preparation**  
> **Peneliti / Mahasiswa**: Mitchel Mohamad Affandi  
> **Dosen Pembimbing**: Dr. R. Yunendah Nur Fu'adah, S.T., M.T., Ph.D.  
> **Repository Status**: Final & Synchronized with v2 Defensibility Audit (October 2026)  
> **Core Mandate**: Maximum Defensibility — Every metric, test, and claim is mathematically audited, leak-free, and publication-ready.

---

## ⚡ Quick Navigation / Dokumen Utama untuk Ibu Pembimbing

| Dokumen | Format | Deskripsi & Tujuan |
|---|:---:|---|
| **[EXECUTIVE_SUMMARY_REPORT.pdf](EXECUTIVE_SUMMARY_REPORT.pdf)** | 📄 PDF | **Laporan Resmi 4 Halaman (Bahasa Indonesia)** lengkap dengan tabel, temuan forensik, dan lampiran visual utama. Sangat disarankan untuk dibaca langsung oleh dosen. |
| **[EXECUTIVE_SUMMARY_REPORT.md](EXECUTIVE_SUMMARY_REPORT.md)** | 📝 Markdown | Kloning versi teks markdown dari laporan ringkasan eksekutif di atas. |
| **[PRESENTATION_GUIDE.md](PRESENTATION_GUIDE.md)** | 📊 Slide Deck Guide | Panduan 12 slide lengkap untuk sidang / presentasi bimbingan, dilengkapi prompt Gemini AI Pro dan speaker notes dwi-bahasa. |
| **[v2/RESULTS.md](v2/RESULTS.md)** | 🔬 Technical Audit | Laporan teknis lengkap seluruh 6 fase audit, pembuktian matematis kebocoran data, dan tabel statistik detail. |

---

## 🧭 Executive Summary: Empat Temuan Ilmiah Utama

Dalam penelitian *machine learning* medis, **angka yang lebih rendah namun jujur dan metodologinya benar bernilai seribu kali lebih tinggi daripada angka tinggi semu yang cacat metodologis**. Repositori ini merangkum evolusi dari eksplorasi awal hingga audit forensik menyeluruh:

```
[Phase 1: Baseline Replication]     --> Replicated exact upperclassmen paper baseline: 0.8506 (90/10/13/41 matrix).
[Phase 2: Data Hygiene Audit]       --> Discovered & rectified 5 non-physiological Glucose=0 mg/dL across all files.
[Phase 3: Forensic Leakage Proof]   --> Mathematically proved class-conditional target leakage in RLTR (t=26.77, r=0.79).
[Phase 4: Leak-Free Benchmark]      --> 50-fold repeated CV under Nadeau-Bengio test (Median imputation is champion).
[Phase 5: Medical SHAP Restoration] --> Glucose restored to #1, BMI to #3, Insulin drops to true physiological rank.
[Phase 6: Clinical Screening Opt]   --> Calibrated threshold=0.37 boosts screening sensitivity from 59% to 79.85%.
[Phase 7: Frozen Untouched Holdout] --> Leak-free holdout: 74.68% (0.8135 AUC), McNemar p=0.0004 consistent with leakage.
```

---

## 🔬 Rangkuman 4 Pilar Kontribusi Ilmiah

### 1. Replikasi Presisi Baseline Kakak Kelas & Klarifikasi Mitos 0.88
- **Replikasi 100% Presisi**: Menggunakan default XGBoost pada data RLTR (seed 42, 80:20 split), script kami mereplikasi angka paper kakak kelas persis hingga 1 pasien:
  $$\text{Akurasi} = \mathbf{0.8506} \quad (\mathbf{TN = 90}, \mathbf{FP = 10}, \mathbf{FN = 13}, \mathbf{TP = 41})$$
- **Klarifikasi Angka 0.88**: Di paper resmi tidak ada angka 0.88. Angka $0.8831$ (atau $0.8961$ pada model consensus) adalah hasil eksplorasi pada split acak tertentu (`seed=12`), dengan margin of error split tunggal mencapai $\pm 5.5\%$. Uji McNemar membuktikan tidak ada perbedaan signifikan antara model paper dan model consensus pada split yang sama ($p = 0.7539$).

### 2. Temuan Kritis: Bug 5 Glukosa Nol
- Pada seluruh 5 file imputasi yang beredar (`LTR`, `NSSR`, `RLTR`, `SIM`, `TR`), pihak pembuat mengimputasi insulin dan skinfold, namun **melewatkan 5 pasien dengan glukosa bernilai 0 mg/dL** (Rows 75, 182, 342, 349, 502).
- Karena glukosa adalah biomarker utama diabetes ($F = 245.67$), angka nol memaksa pasien diabetes masuk ke kelompok sehat pada decision tree (*false negative*). Kami memperbaikinya dengan imputasi median ($117\text{ mg/dL}$).

### 3. Pembuktian Matematis Target Leakage pada File RLTR
Mengapa file RLTR sempat melonjak ke akurasi $\approx 85\%$ sendirian? Audit kami membuktikan bahwa **label target Outcome terinjeksi ke dalam nilai insulin yang hilang**:
- **Regresi OLS**: Koefisien `Outcome` pada insulin di data asli tidak signifikan ($\beta = -5.72, t = -0.45, p = 0.653$). Namun di file RLTR melonjak menjadi **$\beta = +28.68\text{ mg/dL}$ ($t = 26.77, p < 10^{-75}$)**.
- **Korelasi Parsial**: $r(\text{Insulin}, \text{Outcome} \mid \text{Glucose}, \text{BMI})$ di data asli adalah $-0.0168$ ($p = 0.740$), sedangkan di file RLTR melompat ke **$+0.8208$ ($p < 10^{-90}$)**.
- **Eksperimen Rekonstruksi**: Kami merekonstruksi imputasi dengan membatasi donor pool per kelas target, dan berhasil menghasilkan korelasi **$r = 0.7908$**, cocok persis dengan file aslinya ($r = 0.7895$).

### 4. Benchmark Murni Leak-Free & Uji Statistik Nadeau-Bengio
Pada pengujian 50-fold cross-validation murni bebas kebocoran:
- Evaluasi menggunakan **Nadeau-Bengio corrected resampled t-test** dengan koreksi **Holm-Bonferroni** membuktikan bahwa **tidak ada metode imputasi kompleks yang secara signifikan mengungguli imputasi median sederhana** (MissForest $p=0.8751$, MICE $p=1.0000$; semua CI 95% mencakup angka 0).
- Imputasi median adalah baseline yang sangat tangguh, defensibel, dan hemat komputasi untuk kohort PIMA.
- Pada model leak-free, analisis SHAP mengembalikan **Glukosa ke peringkat #1** dan **BMI ke peringkat #3**, memulihkan kebenaran klinis.

---

## 📊 Visualisasi Publikasi Utama (Key Exhibits)

Seluruh gambar resolusi tinggi (300 DPI) telah dihasilkan dan tersimpan di repositori:

| Gambar | Lokasi File | Deskripsi & Peran dalam Naskah |
|---|---|---|
| **Exhibit A** | `v2/figures_v2/A1_imputer_x_model.png` | Matrix 50-Fold CV: Metode Imputasi × Model Klasifikasi (Leak-Free) |
| **Exhibit B** | `figures/09_multi_model_roc_curves_with_consensus.png` | Kurva ROC-AUC Multi-Model & Consensus Ensemble *(Permintaan Dosen Pembimbing)* |
| **Exhibit C** | `figures/12_consensus_model_confusion_matrix.png` | Confusion Matrix Model Consensus pada Holdout Split *(Permintaan Dosen Pembimbing)* |
| **Exhibit D** | `v2/figures_v2/A3_leak_free_shap_ranking.png` | Ranking Fitur Global SHAP Murni Leak-Free (Glukosa #1) |
| **Exhibit E** | `v2/figures_v2/A2_threshold_tradeoff_curve.png` | Kurva Tradeoff Threshold Skrining Klinis (Sensitivitas $79.85\%$ pada ambang $0.37$) |
| **Exhibit F** | `v2/figures_v2/A4_partial_dependence_profiles.png` | Profil Partial Dependence untuk Glukosa, BMI, Age, dan Insulin |

---

## 📁 Struktur Repositori

```text
DiabeticAnalysis/
├── EXECUTIVE_SUMMARY_REPORT.pdf    # Laporan Eksekutif PDF 4 Halaman (Resmi & Siap Cetak/Kirim)
├── EXECUTIVE_SUMMARY_REPORT.md     # Kloning Laporan Eksekutif Format Markdown
├── PRESENTATION_GUIDE.md           # Panduan 12 Slide Presentasi + Prompt Gemini AI Pro
├── README.md                       # Dokumentasi Utama Repositori (File ini)
├── Dataset Diabetes.csv            # Data mentah PIMA Indians (semicolon-separated)
├── LTR_Imputed.csv                 # Linear Trend Imputation
├── NSSR_Imputed.csv                # Non-linear Spline Regression Imputation
├── RLTR_Imputed.csv                # Robust Linear Trend Imputation (Terbukti Target Leakage)
├── SIM_Imputed.csv                 # Simple / Single Imputation Method
├── TR_Imputed.csv                  # Trend Regression Imputation
├── figures/                        # 12 Gambar Analisis Eksploratori Awal (Fig 01 - 12)
├── v2/                             # Pipeline & Hasil Audit Forensik Defensibilitas (v2)
│   ├── RESULTS.md                  # Dokumentasi Teknis Hasil Audit v2 Lengkap
│   ├── figures_v2/                 # 4 Gambar Publikasi v2 (A1, A2, A3, A4)
│   ├── results/                    # Hasil CSV Numerik Deterministik (Phase 1 s.d. Phase 6)
│   └── scripts/                    # Skrip Python Deterministik (run_phase1.py s.d. run_phase6.py)
└── venv/                           # Dedicated Virtual Environment Python 3.12
```

---

## 🚀 Panduan Menjalankan Kode (Deterministik & Reproducible)

Seluruh skrip berjalan langsung menggunakan virtual environment yang tersedia:

```powershell
# Phase 1: Replikasi baseline paper (0.8506), uji McNemar, dan bootstrap 95% CI
.\venv\Scripts\python.exe v2/scripts/run_phase1.py

# Phase 2: Audit matematis target leakage, korelasi parsial, dan rekonstruksi class-conditional
.\venv\Scripts\python.exe v2/scripts/run_phase2.py

# Phase 3: Benchmark 50-fold leak-free imputasi & uji Nadeau-Bengio corrected resampled t-test
.\venv\Scripts\python.exe v2/scripts/run_phase3.py

# Phase 4: Uji ablasi fitur nested CV, benchmarking 10 model, dan optimasi threshold skrining
.\venv\Scripts\python.exe v2/scripts/run_phase4.py

# Phase 5: Analisis interpretability SHAP murni leak-free dan profil Partial Dependence (PDP)
.\venv\Scripts\python.exe v2/scripts/run_phase5.py

# Phase 6: Evaluasi akhir pada untouched holdout test set (seed 42)
.\venv\Scripts\python.exe v2/scripts/run_phase6.py
```

---

## 🎓 Kesiapan untuk Sidang & Publikasi Jurnal

Repositori ini siap diajukan untuk bimbingan skripsi / tugas akhir bersama **Ibu Dr. R. Yunendah Nur Fu'adah, S.T., M.T., Ph.D.** maupun penyusunan naskah jurnal internasional (Q1/Q2 Medical Informatics). Temuan audit target leakage dan perbaikan higienitas data merupakan kontribusi langka yang memberikan nilai akademis tinggi bagi karya ilmiah ini.
