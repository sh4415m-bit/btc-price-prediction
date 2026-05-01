# Predicting Bitcoin Honestly

Deep learning forecasting of daily Bitcoin closing prices using four neural architectures (DNN, LSTM with attention, CNN-Transformer, TCN) and two ensemble strategies, evaluated under a leak-free walk-forward protocol.

**MAI 604 — Deep Learning** · School of Engineering, Applied Sciences and Technology · April 2026

## Authors

- Shamsa Alsuwaidi (20250007177)
- Saeeda Amer (20250006821)
- Amna Aljaroodi (20250007933)

## Headline result

| Metric | Value |
|---|---|
| Walk-forward MAPE (20 folds, 2019–2025) | **2.23%** |
| Static-split Day+1 RMSE (best: TCN) | $2,144 |
| Statistically significant model winner | None (DM p > 0.05) |

## Repository structure

- `notebooks/` — main Jupyter notebook (full pipeline + experiments)
- `report/` — LaTeX source and compiled PDF of the project report
- `presentation/` — LaTeX source and compiled PDF of the slide deck
- `diagrams_source/` — TikZ source for the six pipeline diagrams

## How to run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/Deep_learning_GP_predicting_bitcoin_prices_v4.ipynb
```

## Method overview

The pipeline applies stationary feature engineering (log returns, scale-invariant technical indicators, halving-cycle encoding) before fitting a leak-free StandardScaler on training rows only. Four deep learning architectures are trained under an identical protocol (AdamW + Huber loss + cosine annealing) and combined through two ensemble strategies. Evaluation includes a comprehensive metric suite, walk-forward validation across 20 folds, Diebold–Mariano significance testing, feature-group ablation, and Monte Carlo Dropout uncertainty quantification.

## References

- Wu et al. (2025) — Bitcoin Price Prediction using ML and Combinatorial Fusion Analysis (IEEE CAI)
- Jin et al. (2025) — Deep Temporal Convolutional Networks for High-Frequency Cryptocurrency Price Forecasting (CSAI)
