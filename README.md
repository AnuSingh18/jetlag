# JetLag — Turbofan Engine RUL Predictor

**Detecting Engine Fatigue Before It Fails**

Live demo: [Deploy on Streamlit Cloud]

## What it does

Predicts Remaining Useful Life (RUL) of turbofan jet engines from sensor readings using a Transformer neural network trained on NASA C-MAPSS data.

## Results

| Model | Overall RMSE | Critical Zone RMSE |
|---|---|---|
| Autoencoder Baseline | 41.37 | — |
| Transformer (MSE Loss) | 3.90 | 2.17 |
| Transformer (Critical Zone Loss) | **3.56** | **1.21** |
| State of Art (Literature) | ~12.19 | — |

## How to run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files needed
- `app.py` — Streamlit app
- `jetlag_model.pth` — Trained Transformer weights
- `jetlag_scaler.pkl` — Fitted StandardScaler

## Course
CSCI 6366 Neural Networks & Deep Learning - GWU Spring 2026
