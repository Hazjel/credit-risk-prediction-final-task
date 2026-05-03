# Credit Risk Prediction - Final Project ID/X Partners

## Project Overview
Proyek ini bertujuan untuk membangun model Machine Learning yang dapat memprediksi risiko kredit (*Credit Risk*) dari calon peminjam, mengklasifikasikan mereka ke dalam kategori **Good Debt** (kemungkinan besar akan melunasi pinjaman) atau **Bad Debt** (kemungkinan gagal bayar/default). Model ini dikembangkan sebagai alat bantu keputusan (*decision support tool*) bagi *underwriter* untuk meminimalkan potensi kerugian finansial akibat *False Negative* (meloloskan pinjaman yang akhirnya gagal bayar).

Dataset yang digunakan berasal dari **Lending Club Loan Data (2007-2014)**.

## Key Insights (Exploratory Data Analysis)
1. **Target Imbalance:** Hanya ~19% dari total pinjaman yang tercatat sebagai Bad Debt.
2. **Grade Risk:** Sistem grade dari Lending Club cukup akurat, di mana pinjaman dengan grade F dan G memiliki *default rate* tertinggi (melebihi 40%).
3. **Interest Rate Correlation:** Peminjam yang berakhir gagal bayar umumnya sudah dibebankan tingkat suku bunga (*interest rate*) rata-rata yang lebih tinggi sejak awal, menandakan mereka memang memiliki profil risiko tinggi sejak di-*appraise*.
4. **Income Skewness:** Distribusi pendapatan tahunan sangat *right-skewed*, sehingga membutuhkan teknik penanganan *outlier* pada tahap *preprocessing*.

## Model Architecture & Performance
Tiga arsitektur model dikembangkan dan diuji untuk klasifikasi ini:
- **Logistic Regression** (Baseline)
- **XGBoost** (Gradient Boosting)
- **Stacking Ensemble** (Base: LR, XGBoost, RF | Meta: LR)

Seluruh model dioptimasi tidak menggunakan default threshold 0.5, melainkan menggunakan **Threshold Tuning** via *Precision-Recall Curve* untuk memaksimalkan deteksi *Bad Debt*.

**Best Model: XGBoost**
- **ROC-AUC:** 0.714
- **F1-Score:** 0.422
- **Recall:** 60.5%
- *Overfit Gap sangat rendah (0.028), menunjukkan model mampu melakukan generalisasi dengan baik pada data yang belum pernah dilihat.*

## Business Recommendations
Berdasarkan hasil penemuan Exploratory Data Analysis (EDA) dan pemodelan, berikut adalah rekomendasi strategis untuk implementasi bisnis:

- Deploy **XGBoost sebagai decision-support layer** — bukan *verdict* final, melainkan peringatan dini (*pre-screening*).
- Hentikan **auto-approval untuk Grade E-G** — alihkan ke *manual review* oleh *Underwriter* senior.
- Terapkan **batas DTI lebih ketat bagi Penyewa (RENT)** — mitigasi risiko kerentanan kas akibat beban sewa tak tercatat.
- Ubah **pencairan Debt Consolidation langsung ke kreditur asal** — cegah penyalahgunaan dana tunai baru oleh nasabah.
- Gunakan **ML scoring sebagai prescreening awal** — berhenti menutupi risiko tinggi hanya dengan menaikkan bunga (*adverse selection*).

## Repository Structure
- `credit_risk.ipynb`: Jupyter Notebook berisi *End-to-End Pipeline* (EDA, Data Prep, Modeling, Evaluation).
- `credit_risk_presentation.pdf`: Slide presentasi hasil analisis (*Business Deck*).
- `loan_data_2007_2014.csv`: Dataset asli Lending Club (dikelola via Git LFS).
- `LCDataDictionary.xlsx`: Kamus data berisi penjelasan masing-masing fitur.
