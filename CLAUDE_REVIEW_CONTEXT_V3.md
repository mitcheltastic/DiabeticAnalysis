# 📋 DOSIR KOMPREHENSIF RISET v3: AUDIT METODOLOGIS, REPLIKASI BASELINE, & STANDARISASI BENCHMARK KLINIS KOHORT DIABETES PIMA

> **Dokumen Konteks untuk Review Akademik & Penyusunan Slide Presentasi (Edisi Revisi Pasca-Audit)**  
> **Peneliti Utama / Mahasiswa**: Mitchel Mohamad Affandi  
> **Dosen Pembimbing / Senior Co-Author**: Dr. R. Yunendah Nur Fu'adah, S.T., M.T., Ph.D.  
> **Target Output**: Naskah Konferensi Internasional Baru (IEEE / Scopus Indexed Proceedings) & Bahan Diskusi Tatap Muka  
> **Repositori**: `mitcheltastic/DiabeticAnalysis` (Branch: `main`)  
> **Siklus Data**: Retraining v3 (Leak-Free, 100-Seed Monte Carlo, 50-Fold Repeated Nested CV)

---

## 🧭 1. LATAR BELAKANG AKADEMIK & SITUASI RISET

### 1.1 Hubungan dengan Studi Terdahulu (Prior Work)
Penelitian ini melanjutkan dan menyempurnakan dua publikasi terdahulu yang melibatkan Ibu Dr. R. Yunendah Nur Fu'adah sebagai **Penulis ke-4 (Senior Author / Co-Author)**:
1. **Paper 1 (Saputra et al., IEEE, 2023)**:
   - *Judul*: "Application of Rough Set Theory for Missing Value Imputation and Classification of Diabetes Dataset Using XGBoost"
   - *Hasil yang dilaporkan*: Menggunakan imputasi RLTR (*Relative Limited Tolerance Relation*) dan algoritma XGBoost, dilaporkan mencapai akurasi **85.06%** pada data uji ($80:20$ holdout split).
2. **Paper 2 (Pratama et al., OPLIB Telkom, 2024)**:
   - *Judul*: "Penerapan Rough Set dan XGBoost dengan Optimasi Hyperparameter Menggunakan Optuna untuk Klasifikasi Penyakit Diabetes Mellitus"
   - *Hasil yang dilaporkan*: Menggunakan RLTR + XGBoost + optimasi Optuna, dilaporkan mencapai akurasi **87.01%** pada data uji.

### 1.2 Anomali Metodologis & Misi Edisi v3
Ketika dilakukan pengujian ulang yang ketat (*honest leak-free cross-validation* dengan pemisahan partisi latih-uji sebelum imputasi dan penskalaan), performa berbagai model machine learning standar (Logistic Regression, Random Forest, SVM, XGBoost, CatBoost) **selalu stabil dan konsisten berada di kisaran 76% – 78% (ROC-AUC ~0.835)**. 
- Di literatur global (dimulai dari paper pionir pendiri dataset PIMA: *Smith et al., 1988* yang mendapatkan akurasi 76%), angka 76%–78% adalah batas optimal alami untuk kohort PIMA jika dievaluasi tanpa kebocoran informasi (*leak-free*).
- **Pertanyaan Ilmiah Bersama**: *"Mengapa hasil pada studi terdahulu mencapai 85%–87%? Apa mekanisme metodologis di baliknya?"*
- **Tujuan Edisi v3**: Mengembangkan naskah konferensi baru dengan menyajikan **audit forensik metodologis yang konstruktif**, mereplikasi hasil terdahulu secara deterministik, menjelaskan fenomena pengkondisian label pada imputasi RLTR, dan menegakkan **benchmark klinis murni yang defensibel, sahih, dan siap lolos peer review jurnal/konferensi internasional bereputasi tinggi**.

---

## 🔬 2. DATASET PIMA: ANATOMI & KARAKTERISTIK DATA HILANG

Dataset Pima Indians Diabetes terdiri dari **768 pasien wanita** keturunan Indian Pima berusia $\ge 21$ tahun (500 non-diabetes / `Outcome=0`, 268 diabetes / `Outcome=1`):
- **Nilai 0 Biologis (Data Hilang / Missing Values)**:
  - `Glucose`: 5 pasien ($0.65\%$) — *Catatan Klinis: Glukosa di PIMA diukur melalui 2-hour Oral Glucose Tolerance Test (OGTT) plasma concentration melalui venipuncture darah*.
  - `BloodPressure`: 35 pasien ($4.56\%$) — Tekanan darah diastolik (mmHg).
  - `BMI`: 11 pasien ($1.43\%$) — Indeks massa tubuh ($\text{kg/m}^2$).
  - `SkinThickness`: 227 pasien ($29.56\%$) — Ketebalan lipatan kulit trisep (skinfold caliper).
  - `Insulin`: **374 pasien (48.70% — hampir separuh kohort tidak memiliki catatan insulin!)**
- **Data Biologis Lengkap**: Hanya **392 pasien (51.04%)** yang memiliki rekam 8 parameter fisiologis lengkap. Pasien dengan insulin tercatat di lab berjumlah **394 pasien**.
- **Koreksi Taksonomi Imputasi**: Lima metode imputasi studi terdahulu (`TR`, `LTR`, `RLTR`, `NSSR`, `SIM`) diklarifikasi secara ilmiah **bukanlah regresi parametrik linier/spline**, melainkan metode **Rough-Set Similarity & Tolerance Relations** (Relasi Toleransi dan Keserupaan Himpunan Kasar).

---

## 📊 3. DELAPAN EKSPERIMEN INTI (E1 HINGGA E8) & BUKTI EMPIRIS

### [E1] Audit Metodologis 3-Tier Kebocoran Pengkondisian Target (Target Leakage Audit)
- **Data Biologis Murni ($N=392$)**: Pada pasien nyata, setelah mengontrol 7 kovariat klinis lainnya (glukosa, umur, BMI, dll.) melalui regresi linier OLS multivariat, label `Outcome` **sama sekali tidak memiliki korelasi parsial terhadap insulin**:  
  $$\beta = -6.16\text{ mg/dL}, \quad t = -0.484, \quad p = 0.6285 \quad (\text{Independen & Bebas Kebocoran Target})$$
- **File RLTR Terpublikasi ($N=768$)**: Terjadi anomali matematis ekstrem di mana label target `Outcome` terkondisikan secara masif ke dalam nilai imputasi insulin:  
  $$\beta = +28.68\text{ mg/dL}, \quad \mathbf{t = 26.77}, \quad \mathbf{p = 3.97 \times 10^{-88}} \quad (\Delta\beta = +34.84\text{ mg/dL})$$
- **Uji Dua Sampel Independen Fisher $z$ (Sampel Saling Lepas / Disjoint Partition)**:  
  Membandingkan pasien dengan insulin asli terukur lab ($n=394$) vs pasien yang insulinnya diisi melalui RLTR ($n=374$):
  - Pasien Insulin Terukur Asli ($n=394$): Korelasi point-biserial $r = +0.3035$
  - Pasien Insulin Imputasi RLTR ($n=374$): Korelasi point-biserial melompat menjadi **$r = +0.7895$**
  - **Uji Fisher $z$ Sampel Independen**: $\mathbf{z = 10.4422}, \quad \mathbf{p = 1.62 \times 10^{-25}}$ (Terbukti secara definitif bahwa korelasi pada baris imputasi jauh melampaui distribusi biologis asli).
- **Konfirmasi Mekanisme pada Honest RLTR**: Ketika algoritma RLTR dijalankan secara murni tanpa pengkondisian label target (`HonestRLTRImputer` dari E3), koefisien regresi multivariat OLS kembali normal dan tidak signifikan:  
  $$\beta = -2.72\text{ mg/dL}, \quad t = -0.4149, \quad p = 0.6783 \quad (\text{Membuktikan target leakage pada file RLTR lama})$$

### [E2 & E2b] Tangga Inflasi Metodologis (100 Seeds) & Replikasi Presisi
Dekomposisi bertahap bagaimana akurasi bergerak dari baseline murni hingga angka studi terdahulu:
1. **R0 (Murni Leak-Free Baseline)**: **75.25% ± 3.06%** (AUC $0.8163 \pm 0.0296$ dihitung sebagai rata-rata 100 holdout splits 80:20).
2. **R1 (Pre-split Scaling)**: 75.23% (tidak ada dampak inflasi).
3. **R2 (Pre-split MICE)**: 74.88% (tidak ada dampak inflasi).
4. **R3a_RLTR (File RLTR As-Is dengan XGBoost Reference Configuration `lr=0.1, depth=6`)**:  
   - Rata-rata 100 split: **83.69% ± 2.65%** (AUC $0.8998 \pm 0.0217$), naik $+8.45\text{ pp}$ ($p_{\text{NB}} = 3.90 \times 10^{-7}$).
   - Pada partisi standar **Seed 42**, akurasinya menghasilkan tepat **85.0649%** dengan matriks konfusi:  
     $$\mathbf{TN=90, \ FP=10, \ FN=13, \ TP=41}$$  
     👉 **Akurasi dan matriks konfusi identik dengan Paper 1 (Saputra et al., IEEE, 2023, Tabel II)**.
5. **R4-50 (Direct Test-Set Objective Tuning, 50 trials Optuna)**:  
   - Rata-rata 100 split: **86.42% ± 2.21%**, naik $+10.61\text{ pp}$ ($p_{\text{NB}} = 8.16 \times 10^{-10}$).
   - Pada **Seed 42**, akurasinya menghasilkan tepat **87.0130%** dengan matriks konfusi:  
     $$\mathbf{TN=92, \ FP=8, \ FN=12, \ TP=42}$$  
     👉 **Konsisten dengan hasil optimasi Optuna pada Paper 2 (Pratama et al., OPLIB, 2024)**.
6. **R5 (Batas Atas Variabilitas Stokastik / Stochastic Upper Bound)**:  
   - Akurasi maksimum yang dapat dicapai di antara 100 split acak adalah **89.61% (138/154)** pada R3a (Seed 12), dan **90.91% (140/154)** pada R4 (pencarian 30 seed).  
   - *Catatan Metodologis*: Replikasi membuktikan bahwa studi terdahulu berakar pada Seed 42, bukan hasil cherry-picking seed. R5 dicantumkan murni sebagai demonstrasi batas atas variabilitas stokastik pada data uji berukuran kecil ($N_{\text{test}}=154$).

### [E3] Pembuktian Kesetaraan Imputasi (50-Fold Repeated CV & Uji TOST)
- Menguji 6 strategi imputasi leak-free: Median, Mean, MICE (BayesianRidge), MissForest (ExtraTrees), Honest RLTR ($\epsilon=0.15$), Honest RLTR ($\epsilon=0.25$).
- **Hasil**: Semua metode menghasilkan akurasi yang setara di kisaran **77.2% – 77.6%** ($p \ge 0.442$).
- **Uji Ekuivalensi Dua Arah (Two One-Sided Tests / TOST)**: Dengan batas ketidakpedulian klinis apriori $\pm 1.5\text{ pp}$ ($\pm 0.015$), seluruh perbandingan membuktikan kesetaraan statistik yang sangat meyakinkan:  
  $$\mathbf{p_{\text{TOST}} < 0.0001}$$
- **Kesimpulan Ilmiah**: Algoritma imputasi kompleks yang mahal secara komputasi tidak memberikan keunggulan di atas imputasi median sederhana. Imputasi median adalah pilihan paling parsimonius, cepat, dan bebas risiko overfitting.

### [E4] Seleksi Model Akhir & Asas Occam's Razor
- Diuji pada 50 outer-folds independen di bawah protokol apriori (Protokol Amendemen 2):
  - **LogReg-L2 Default ($C=1.0$)**: **AUC $0.8357 \pm 0.0381$**, Akurasi $77.45\% \pm 0.0366$, Brier Score $0.1601$ *(Protokol: 50 outer folds nested CV)*.
  - LogReg-ElasticNet Tuned: AUC $0.8369$ ($p_{\text{Holm}} = 1.0000$, TOST $p < 0.0001$).
  - Tuned XGBoost: AUC $0.8379$ ($p_{\text{Holm}} = 1.0000$, TOST $p < 0.0001$).
  - Tuned CatBoost: AUC $0.8417$ ($p_{\text{Holm}} = 1.0000$, TOST $p = 0.0014$).
- *Saturasi Batas Bawah*: Optuna memilih `max_depth=2` (batas bawah ruang pencarian) pada **78.0% fold XGBoost**, menunjukkan bahwa pada sampel kecil ($N=768$), model pohon memangkas kompleksitasnya mendekati estimator linier.
- **Model Terpilih Resmi**: **Logistic Regression L2 Default** (sederhana, kokoh, transparan secara klinis).

### [E5] Kebijakan Skrining Klinis, Kalibrasi Probabilitas, & Decision Curve Analysis (DCA)
- **Nested Threshold Policy B ($\tau \approx 0.280$)**: Dirancang untuk skenario skrining:
  - Sensitivitas: **80.43% ± 8.88%**
  - Spesifisitas: **68.48% ± 7.16%**
  - Evaluasi pada Frozen Holdout Seed 42: Akurasi **72.73%**, Sensitivitas **83.33%**, Spesifisitas **67.00%**, **ROC-AUC = 0.8130** *(Protokol: Single frozen holdout seed 42)*.
- **Kalibrasi Probabilitas**: Model menunjukkan kalibrasi yang baik dengan *Calibration Slope* $0.7608$ dan *CITL* $-0.0062$.
- **Tafsiran Decision Curve Analysis (DCA)**: Memberikan keuntungan klinis bersih (*net clinical benefit*) positif pada seluruh rentang ambang probabilitas $p_t \in [0.10, 0.55]$. Pada ambang $p_t = 0.20$, *net benefit* adalah **$0.2458$**, yang berarti model secara bersih mendeteksi setara dengan **25 pasien diabetes positif benar per 100 pasien**, setelah memperhitungkan dan mengurangi dampak negatif positif palsu yang dibobot oleh ambang keputusan $\left(\frac{p_t}{1 - p_t}\right)$.

### [E6] Uji Subgrup Klinis: Discrepancy Performa RLTR pada Pasien Riil (Headline Novelty)
- Membedah data uji 100 split menjadi 2 subgrup fisiologis:
  1. **Pasien yang Nilai Insulinnya Diimputasi ($n \approx 84$)**:
     - Model RLTR Terdahulu: Akurasi **92.00% ± 3.31%** (AUC $0.9700$).
  2. **Pasien dengan Nilai Insulin Asli Laboratorium ($n \approx 70$)**:
     - Model RLTR Terdahulu: **Turun drastis menjadi 75.75% ± 3.87% (AUC $0.8060$)** $\rightarrow$ **Penurunan sebesar -16.25 pp ($p_{\text{NB}} = 2.25 \times 10^{-9}$)**!
     - Model Murni Kita (LogReg-L2): **Mencapai 77.68% ± 3.63% (AUC $0.8446$)**!
- **Dua Bukti Statistik Penentu**:
  1. Pada pasien biologis riil, **Model Murni Kita mengungguli Model RLTR Terdahulu sebesar +0.0386 AUC ($p_{\text{NB}} = 0.0295$)**!
  2. Interaksi *Difference-in-Differences* ($\Delta\Delta$): $\Delta\Delta\text{Acc} = -18.51\text{ pp} \ (p_{\text{NB}} = 8.98 \times 10^{-11})$.
- **Makna Ilmiah**: Model terdahulu menunjukkan performa tinggi terutama pada baris data yang nilai insulinnya disintesis secara artifisial, namun mengalami penurunan daya diskriminasi saat berhadapan dengan data pasien riil berinsulin lengkap.

### [E7] Stabilitas Kepentingan Fitur SHAP (200 Bootstrap Refits)
- Evaluasi 200 kali bootstrap resample pada data latih:
  1. **Glucose**: Mean |SHAP| = 1.0535 (Konsisten 100% di Ranking #1)
  2. **BMI**: Mean |SHAP| = 0.5809 (Konsisten 100% di Ranking #2)
  3. **Pregnancies**: Mean |SHAP| = 0.3298 (Ranking #3 pada LogReg-L2)
  4. **DiabetesPedigreeFunction**: Mean |SHAP| = 0.1800 (Ranking #4 pada LogReg-L2)
  5. **Age**: Mean |SHAP| = 0.1417 (Ranking #5 pada LogReg-L2, Ranking #3 pada XGBoost)
  6. **Insulin**: Mean |SHAP| = **0.0902 (Median Rank: 6.0 dari 8 fitur)**
  7. **BloodPressure**: Mean |SHAP| = 0.0727
  8. **SkinThickness**: Mean |SHAP| = 0.0615
- *Penjelasan Klinis Koefisien Insulin yang Negatif ($\beta = -0.069$)*: Pada model multivariat terkontrol, setelah mengontrol glukosa plasma yang sangat tinggi dan BMI obesitas, kadar insulin puasa yang relatif lebih rendah mencerminkan dekompensasi atau kelelahan sel beta pankreas (*beta-cell exhaustion*) pada diabetes tipe 2 lanjut, di mana kapasitas sekresi insulin mulai mengalami kegagalan.

### [E8] Kurva Ablasi Kumulatif Top-K (Temuan Parsimoni Parameter Klinis)
Eksperimen penambahan fitur secara bertahap mengikuti urutan konsensus SHAP Model Akhir LogReg-L2 (100 seeds Monte Carlo):
- **Top 1 (Glucose)**: Akurasi 73.99% (AUC 0.7863)
- **Top 2 (+ BMI)**: Akurasi 76.10% (AUC 0.8179)
- **Top 3 (+ Pregnancies)**: Akurasi 76.38% (AUC 0.8269)
- **Top 4 (+ DiabetesPedigreeFunction)**: Akurasi **76.55% – 76.68% (AUC 0.8335 – 0.8400)** $\rightarrow$ **PLATO STATISTIK**
- **Top 8 (Semua 8 Fitur Lengkap)**: Akurasi **76.60% (AUC 0.8369)**
- **Temuan Klinis**: Menambahkan 4 fitur sisanya (terutama Insulin yang 49% missing dan SkinThickness yang 30% missing) hanya memberikan fluktuasi akurasi marginal ($\pm 0.3\%$). Skrining awal diabetes dapat dioptimalkan menggunakan parameter klinis rutin tanpa memerlukan tes laboratorium insulin puasa yang invasif dan mahal.

---

## 🖼️ 4. DAFTAR GAMBAR ILMIAH (FIGURES) UNTUK DIUNGGAH KE CLAUDE

Semua file gambar beresolusi tinggi (300 DPI) berada di folder `v3/figures/`:

| No | Nama File Gambar | Deskripsi Visual & Perbaikan yang Diterapkan |
| :---: | :--- | :--- |
| 1 | `E2_inflation_ladder_waterfall.png` | **Barchart Tangga Inflasi**: Dekomposisi kenaikan performa dari baseline murni 75.25% $\rightarrow$ RLTR leak (+8.45 pp) $\rightarrow$ Test-tuned (+2.16 pp) $\rightarrow$ Batas atas stokastik R5 (90.91%). |
| 2 | `E5_decision_curve_analysis.png` | **Decision Curve Analysis (DCA)**: Kurva keuntungan klinis bersih (*net benefit*) model vs strategi *treat all* / *treat none*. |
| 3 | `E5_reliability_diagram_seed42.png` | **Diagram Reliabilitas Kalibrasi**: Kurva probabilitas prediksi vs frekuensi observasi nyata (Brier score 0.1754, Slope 0.7608). |
| 4 | `E7_feature_importance_ranks.png` | **Boxplot Stabilitas Peringkat Fitur**: Memperlihatkan kestabilan Glukosa (#1) dan BMI (#2) di 100% bootstrap refits (label sumbu x sudah dirotasi sehingga tidak bertumpuk). |
| 5 | `E7_shap_importance_comparison.png` | **Bar Chart Perbandingan SHAP Global**: Nilai mean absolute SHAP dengan 95% CI (label panel LogReg dan XGBoost telah diperbaiki dan disinkronkan, `sharey=False`). |
| 6 | `E8_shap_beeswarm_lr.png` | **SHAP Beeswarm Plot (Logistic Regression)**: Arah pengaruh nilai tinggi/rendah masing-masing fitur terhadap risiko diabetes pada model linier akhir. |
| 7 | `E8_shap_beeswarm_xgb.png` | **SHAP Beeswarm Plot (Tuned XGBoost)**: Visualisasi dampak non-linier fitur terhadap log-odds risiko diabetes pada model pohon boosting. |
| 8 | `E8_topk_feature_ablation_curve.png` | **Kurva Ablasi Kumulatif Top-K (2 Panel)**: Akurasi dan ROC-AUC vs jumlah fitur ($k=1 \dots 8$) yang memperlihatkan plato pada Top-4 (skala y AUC diperluas dan label sumbu x dirotasi rapi). |

---

## 🎯 5. AGENDA & KEPUTUSAN BERSAMA DENGAN BU YUNEN

1. **Memvalidasi Standar Ilmiah 76%–78%**: Menyepakati bahwa kisaran performa ini adalah batas akurasi murni yang sahih di literatur internasional untuk dataset PIMA, dan mencerminkan metodologi yang jujur.
2. **Posisi Naskah Baru**: Memposisikan paper baru bukan sebagai pembatalan studi sebelumnya, melainkan sebagai **analisis audit metodologis lanjutan, konfirmasi replikasi presisi, dan penetapan protokol benchmark klinis yang defensibel**.
3. **Keputusan Keterlibatan Penulis Lama**: Mengonsultasikan kepada Ibu Yunen mengenai format sitasi yang paling etis dan bijak terhadap publikasi tahun 2023 dan 2024.
