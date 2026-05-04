# JetLag - Turbofan Engine RUL Predictor

**Detecting Engine Fatigue Before It Fails**

Live demo: https://jetlag-anu.streamlit.app/

Paper: JetLag: Turbofan RUL Prediction with Critical Zone Loss and Sensor Attribution — Anu Singh, GWU CSCI 6366, Spring 2026

---

## What it does

Predicts Remaining Useful Life (RUL) of turbofan jet engines from multivariate sensor readings using a pure Transformer neural network trained on NASA C-MAPSS data. Includes a critical zone loss function that applies a 3× penalty for errors when true RUL < 30 cycles, statistically validated via Mann-Whitney U test (p < 0.0001).

---

## Results (FD001 Validation Set)

| Model | Overall RMSE | CZ RMSE (RUL < 30) |
|---|---|---|
| Autoencoder Baseline | 41.37 | — |
| Transformer (MSE Loss) | 3.86 | 2.14 |
| Transformer (Critical Zone Loss) | 3.69 | 1.59 |

CZ loss achieves a 26% reduction in critical zone RMSE, confirmed by Mann-Whitney U test (p < 0.0001 on 587 critical zone samples).

### Official NASA Test Set

| Dataset | RMSE | NASA Score |
|---|---|---|
| FD001 | 19.74 | 1,709.09 |
| FD002 | 27.56 | 16,439.63 |
| FD003 | 15.76 | 1,345.61 |
| FD004 | 29.42 | 12,871.02 |

---

## Repository Contents

| File | Description |
|---|---|
| `NASA_CMAPSS.ipynb` | Main notebook — all experiments reproducible |
| `app.py` | Streamlit dashboard |
| `requirements.txt` | Python dependencies |
| `jetlag_model.pth` | FD001 CZ Loss model weights |
| `model_fd002.pth` | FD002 model weights |
| `model_fd003.pth` | FD003 model weights |
| `model_fd004.pth` | FD004 model weights |
| `federated_model.pth` | Federated model weights |
| `scaler_fd001.pkl` | FD001 StandardScaler |
| `scaler_fd002.pkl` | FD002 StandardScaler |
| `scaler_fd003.pkl` | FD003 StandardScaler |
| `scaler_fd004.pkl` | FD004 StandardScaler |

---

## How to Run the Dashboard

```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload a CSV with sensor columns: s2, s3, s4, s6, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21. Minimum 30 rows required.

---

## How to Reproduce Experiments

1. Open `NASA_CMAPSS.ipynb` in Google Colab
2. Run all cells in order
3. NASA C-MAPSS dataset is downloaded automatically from the public repository

All experiments (autoencoder, Transformer MSE, critical zone loss, ablation studies, permutation importance, federated learning, official test set evaluation) are in the notebook.

---

## Dataset

NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) — Saxena et al., 2008. Publicly available from NASA's Prognostics Center of Excellence.

---

## Course

CSCI 6366 Neural Networks & Deep Learning — GWU Spring 2026
