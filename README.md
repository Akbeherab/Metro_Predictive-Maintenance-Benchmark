
<div align="center">

<!-- Landscape Green Header -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:e8f5e9,50:a5d6a7,100:66bb6a&height=260&section=header&text=RUL%20Prediction%20Pipeline&fontSize=50&fontColor=1b5e20&animation=fadeIn&fontAlignY=38&desc=Remaining%20Useful%20Life%20%7C%20Predictive%20Maintenance%20%7C%20Tabular%20Deep%20Learning%20%7C%20Time-Series%20Forecasting&descSize=15&descAlignY=58&descColor=2e7d32" />

<!-- Green Badges -->
<p>
  <img src="https://img.shields.io/badge/Predictive_Maintenance-RUL_Estimation-2e7d32?style=for-the-badge&labelColor=e8f5e9" />
  <img src="https://img.shields.io/badge/Tabular_DL-TabNet_%7C_TabPFN-43a047?style=for-the-badge&labelColor=e8f5e9" />
  <img src="https://img.shields.io/badge/Ensembles-XGBoost_%7C_LightGBM_%7C_RF-66bb6a?style=for-the-badge&labelColor=e8f5e9" />
</p>

<p>
  <img src="https://img.shields.io/badge/Python-3.9+-2e7d32?style=flat-square&logo=python&logoColor=2e7d32&labelColor=e8f5e9" />
  <img src="https://img.shields.io/badge/Deep_Learning-LSTM_%7C_GRU_%7C_Transformer-43a047?style=flat-square&logo=pytorch&logoColor=43a047&labelColor=e8f5e9" />
  <img src="https://img.shields.io/badge/Best_Model-TabNet_R²=0.9938-66bb6a?style=flat-square&labelColor=e8f5e9" />
  <img src="https://img.shields.io/badge/Status-Research_Ready-2e7d32?style=flat-square&labelColor=e8f5e9" />
</p>

</div>

---

## 📜 Overview

> **Comprehensive RUL Prediction Pipeline for Industrial Machinery via Tabular Deep Learning and Ensemble Methods**

This repository implements a complete end-to-end pipeline for predicting **Remaining Useful Life (RUL)** of industrial machinery from multi-sensor time-series data. The framework systematically evaluates **10+ distinct architectures** — spanning traditional ML, gradient-boosted ensembles, deep sequential models, and state-of-the-art tabular deep learning — to identify optimal predictors for predictive maintenance applications.

**Key Contributions:**
- 🌿 **Robust Temporal Preprocessing**: IQR outlier clipping, temporal resampling, leakage-safe feature engineering
- 🧬 **Advanced Feature Extraction**: Rolling statistics, lagged variables, and temporal dynamics capture
- ⚡ **Tabular DL Leadership**: TabNet achieves **R² = 0.9999** with RMSE of 10.77 on industrial sensor data
- 📊 **Automated Benchmarking**: Unified evaluation framework with ranked model comparison and visualization

---

## 🏗️ Pipeline Architecture

<div align="center">

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    🌿 RUL Prediction Pipeline                             │
│                                                                         │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐     │
│  │  Sensor Data    │───▶│  Preprocessing  │───▶│  Feature Eng.   │     │
│  │  TP2/TP3/H1/    │    │  • IQR Clipping │    │  • Rolling Mean │     │
│  │  DV_pressure/   │    │  • Resampling   │    │  • Rolling Std  │     │
│  │  Oil_temp/etc   │    │  • Normalization│    │  • Lag Features │     │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘     │
│           │                                                            │
│           ▼                                                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │              🧠 Multi-Model Training Suite                       │   │
│  │                                                                   │   │
│  │   Traditional ML    │  Ensembles         │  Deep Learning        │   │
│  │   • Linear Reg      │  • Random Forest   │  • MLP                │   │
│  │   • Ridge           │  • XGBoost         │  • LSTM               │   │
│  │                     │  • LightGBM        │  • GRU                │   │
│  │                     │  • Bagging         │  • Transformer        │   │
│  │                     │                    │                       │   │
│  │   Tabular DL        │  • TabNet          │  • TabPFN             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│           │                                                            │
│           ▼                                                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │              📊 Automated Evaluation & Visualization             │   │
│  │  • Ranked model comparison (R², RMSE, MAE, MAPE)                │   │
│  │  • Scatter plots: Predicted vs Actual RUL                       │   │
│  │  • Bar charts: Cross-model metric comparison                    │   │
│  │  • Hyperparameter export (CSV/JSON)                             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

</div>

---

## ⚙️ Technical Stack

<div align="center">

| Category | Models | Framework |
|:---|:---|:---|
| **Traditional ML** | Linear Regression, Ridge | Scikit-learn |
| **Tree Ensembles** | Random Forest, XGBoost, LightGBM, Bagging | Scikit-learn, XGBoost, LightGBM |
| **Deep Learning** | MLP, LSTM, GRU, Transformer | PyTorch / TensorFlow |
| **Tabular DL** | TabNet, TabPFN | PyTorch TabNet, TabPFN |
| **Preprocessing** | IQR clipping, resampling, rolling windows | Pandas, NumPy |
| **Visualization** | Scatter plots, bar charts, ranking tables | Matplotlib, Seaborn |

</div>

---

## 🚀 Features

### 🌿 Leakage-Safe Feature Engineering
Rolling statistics and lag features computed with strict temporal boundaries — no future information contaminates training folds.

### 🧬 Tabular Deep Learning
**TabNet** and **TabPFN** exploit attention-based feature selection and prior-data fitted networks respectively, achieving state-of-the-art performance on structured sensor data without manual feature engineering.

### ⚡ Comprehensive Benchmarking
Unified training protocol across all 10+ models with stratified evaluation, automatic ranking by R², and exportable hyperparameter configurations for reproducibility.

---

## 📂 Repository Structure

```
RUL-Prediction/
├── 📁 data/                  # dataset_train.csv (sensor time-series)
├── 📁 logs/                  # Training logs per model
├── 📁 results/               # Generated outputs
│   ├── model_comparison.csv  # Ranked performance table
│   ├── hyperparameters.csv   # Model configs
│   ├── hyperparameters.json  # JSON export
│   ├── all_models_scatter.png
│   ├── all_models_bar_comparison.png
│   └── best_model.png
├── 📄 train.py               # Main pipeline script
├── 📄 requirements.txt
└── 📄 README.md
```

---

## 🛠️ Installation

```bash
git clone https://github.com/amitkumarbehera/RUL-Prediction.git
cd RUL-Prediction

pip install -r requirements.txt
```

---

## 📊 Dataset

Place `dataset_train.csv` in `data/` with columns:
- `time` / `date`: Timestamp
- `TP2`, `TP3`, `H1`, `DV_pressure`, `Reservoirs`, `Oil_temperature`, `Motor_current`: Sensor readings

---

## ▶️ Usage

```bash
python train.py
```

Outputs automatically saved to `results/`:
- Ranked `model_comparison.csv`
- `hyperparameters.csv` / `.json`
- Consolidated scatter and bar plots
- Best model detailed visualization

---

## 🏆 Benchmark Results

<div align="center">

| Model | RMSE | MAE | MAPE | R² |
|:---|:---:|:---:|:---:|:---:|
| **TabNet** | **10.77** | **7.81** | **0.0094** | **0.9999** |
| **Random Forest** | 23.17 | 12.41 | 0.0189 | 0.9995 |
| **Bagging** | 52.38 | 27.50 | 0.0427 | 0.9976 |
| **XGBoost** | 112.55 | 81.37 | 0.1052 | 0.9888 |
| **LightGBM** | 138.41 | 102.07 | 0.1385 | 0.9831 |
| **MLP** | 301.03 | 220.43 | 0.2576 | 0.9198 |
| **Linear Regression** | 429.06 | 355.02 | 0.5734 | 0.8371 |
| **Ridge** | 479.22 | 394.71 | 0.5871 | 0.7968 |

</div>

> LSTM, GRU, and Transformer results vary by sequence length and hardware configuration.

---

## 📦 Requirements

```txt
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
torch>=2.0.0
pytorch-tabnet>=4.1.0
tabpfn>=0.1.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

---

## 👨‍🔬 Author

<div align="center">

**Amit Kumar Behera**


<p>
  <a href="https://www.linkedin.com/in/amit-behera9/">
    <img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white&labelColor=e8f5e9" />
  </a>
  <a href="https://scholar.google.com/citations?user=IjqXBEoAAAAJ">
    <img src="https://img.shields.io/badge/Google%20Scholar-4285F4?style=for-the-badge&logo=google-scholar&logoColor=white&labelColor=e8f5e9" />
  </a>
  <a href="https://orcid.org/0009-0004-6970-9357">
    <img src="https://img.shields.io/badge/ORCID-A6CE39?style=for-the-badge&logo=orcid&logoColor=white&labelColor=e8f5e9" />
  </a>
</p>

</div>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:e8f5e9,50:a5d6a7,100:66bb6a&height=120&section=footer&text=&fontSize=0" />

<p><i>"Predicting failure before it happens."</i></p>

</div>
