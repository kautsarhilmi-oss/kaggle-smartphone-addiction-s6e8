# 📱 Predicting Smartphone Addiction — Kaggle PS S6E8

[![Kaggle](https://img.shields.io/badge/Kaggle-Playground%20S6E8-blue)](https://www.kaggle.com/competitions/playground-series-s6e8)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green)](https://python.org)
[![Score](https://img.shields.io/badge/CV%20AUC-0.9561-brightgreen)]()

## 🎯 Overview

Binary classification untuk memprediksi apakah seorang pengguna **kecanduan smartphone** berdasarkan pola perilaku digital — screen time, social media, gaming, sleep habits, stress level, dan notification frequency.

- **Competition:** Kaggle Playground Series Season 6, Episode 8
- **Metric:** Area Under the ROC Curve (AUC-ROC)
- **CV Score:** ~0.9561 (3-Fold Stratified CV)
- **Model:** HistGradientBoostingClassifier (scikit-learn)

---

## 📂 Repository Structure


---

## 📊 Dataset

| Split | Rows    | Features | Notes                      |
|-------|---------|----------|----------------------------|
| Train | 691,369 | 13       | ~71% positive (addicted)   |
| Test  | 296,302 | 12       | No target label            |

**Key challenges:**
- Missing values 10–20% di hampir semua kolom
- Class imbalance: 71% addicted vs 29% not addicted

---

## 🔧 Pendekatan

### 1. Feature Engineering (9 fitur baru)

| Feature | Keterangan |
|---------|-----------|
| `leisure_to_productive` | Rasio screen time leisure vs produktif |
| `weekend_to_daily_ratio` | Weekend binge signal |
| `sleep_deficit` | Kekurangan jam tidur dari target 7 jam |
| `notif_per_app_open` | Rasio notifikasi per buka app |
| `screen_per_sleep` | Screen time per jam tidur |
| `social_media_ratio` | Dominasi sosmed dari total screen time |
| `app_opens_log` | Log-transform untuk distribusi skewed |

### 2. Missing Value Strategy
`HistGradientBoostingClassifier` menangani NaN secara native — **tanpa imputation**.

### 3. Class Imbalance
Menggunakan `class_weight='balanced'` untuk menangani imbalance 71%/29%.

### 4. Cross-Validation
3-Fold Stratified CV pada 150K subsample — cepat dan representatif.

---

## 📈 Hasil

| Fold | AUC     |
|------|---------|
| 1    | 0.95588 |
| 2    | 0.95698 |
| 3    | 0.95545 |
| **Overall OOF** | **0.95608** |

---

## 🚀 Cara Menjalankan

```bash
pip install scikit-learn pandas numpy

# Letakkan train.csv, test.csv, sample_submission.csv di folder yang sama
python smartphone_addiction_prediction.py
```

---


## 📥 Data

Dataset tidak disertakan di repo ini karena ukuran dan ketentuan Kaggle.
Download langsung dari halaman kompetisi:

👉 [Kaggle — Playground Series S6E8](https://www.kaggle.com/competitions/playground-series-s6e8/data)

---

## 💡 Potential Improvements

- Full 5-fold CV pada seluruh dataset
- Target encoding untuk fitur kategorikal
- Stacking ensemble (LightGBM + XGBoost + HistGBM)
- Kalibrasi probabilitas (Platt Scaling)

---

## 👤 Author

**Kautsar Hilmi** — Magister Teknik Informatika, Universitas Pamulang  
*Data Science | Machine Learning | Analytics*

[![GitHub](https://img.shields.io/badge/GitHub-kautsarhilmi--oss-black)](https://github.com/kautsarhilmi-oss)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-kautsarhilmi-blue)](https://www.linkedin.com/in/kautsarhilmi/)
