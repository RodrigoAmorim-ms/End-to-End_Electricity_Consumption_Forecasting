# German Electricity Consumption Forecasting – MLOps & Time-Series Project

## Overview
This project focuses on time-series forecasting of the 24h (daily) electricity consumption of Germany using real-world power grid data from the [Open Power System Data (OPSD) platform](https://open-power-system-data.org).
The primary objective was to design an end-to-end ML project that combines classical ML, deep learning, and MLOps best practices to make a time-series forecasting.

We forecast electricity consumption 24 hours ahead using:

A tree-based ML model (XGBoost)

A state-of-the-art deep learning model (Temporal Fusion Transformer – TFT)

The project emphasizes experiment tracking, reproducibility, fair model comparison (offline A/B testing), and interpretability, rather than just raw predictive performance.

All experiments are tracked using MLflow, making results transparent and reproducible.

## Results

## Project Details

-**Language:** Python (Jupyter Notebooks)

-**Dataset:** [time_series_60min_singleindex.csv](https://data.open-power-system-data.org/time_series/2020-10-06)

-**Forecast Horizon:** 24 hours ahead 

### Models:

**Naïve baselines** (daily & weekly seasonality)

**XGBoost** (feature-based time-series regression - Machine Learning)

**Temporal Fusion Transformer** (Deep learning)

### Experiment Tracking: 
-MLflow

### Hyperparameter Optimization: 
-Optuna

### Environment Management:
-Poetry

### Level of Maturity: 0
[See MLOps Maturity Model](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/mlops-maturity-model)

## Key Concepts Demonstrated

-Time-series forecasting fundamentals

-Supervised learning formulation for time-series

-Rolling-origin (walk-forward) evaluation

-Offline A/B testing for model comparison

-Hyperparameter tuning with Optuna

-Experiment tracking with MLflow

-Reproducible ML experiments

-Model interpretability (feature importance & attention mechanisms)

## Project Structure
.
├── Notebooks/
│   ├── 00_EDA.ipynb
│   ├── 04_AB_testing.ipynb
│
├── Models/
│   ├── 02_Naïve.ipynb
│   ├── 03_XGBoost.ipynb
│   ├── 04_TFT.ipynb
│ 
├── Data/
│   └── Processed/
│       └── GermanEnergyConsumption.parquet
│   └── Raw/
│       └── time_series_60min_singleindex.csv
│
├── pyproject.toml           # Poetry dependencies
├── README.md

## Methodology
**Train / Validation / Test Split**

-Training set: Historical data (2015-2017) 

-Validation set: Used for hyperparameter tuning (2018) *After the hyperparameter tuning and during the consequent training of the final model, this validation set would be incorporated into the training set to increase the available training data 

-Test set: Last full year of data (2019) 

Evaluation is done using rolling 24-hour forecasts across the test year, simulating real-world deployment conditions.

**Models**
1. Naïve Baselines

Daily persistence (same hour, previous day)

Weekly persistence (same hour, previous week)

These serve as strong sanity checks and performance baselines.

2. XGBoost

-Feature-based time-series regression

-Lag features (hours, days, weeks)

-Rolling statistics

-Calendar features (hour, weekday)

-Hyperparameter tuning with Optuna

-Evaluated using rolling 24-hour forecasting

3. Temporal Fusion Transformer (TFT)

Sequence-to-sequence deep learning model

Learns temporal dependencies automatically

Uses encoder–decoder architecture

Supports:

Attention-based interpretability

Probabilistic forecasting (quantile loss)

Hyperparameter tuning with Optuna

**A/B Offline Testing**

To ensure a fair comparison between models:

Both models predict the same rolling 24-hour windows and metrics are computed over the entire test year 

**Evaluation metrics:**

- MAE, RMSE, MAE 

Results were logged using MLflow and compared in the last notebok **04_AB_Testing.ipynb**

## Results 

| Model         | MAE ↓   | RMSE ↓  | MAPE ↓ |
|---------------|---------|---------|--------|
| Naïve (Daily) | 4557.51 | 6943.97 | 0.0833 |
| Naïve (Weekly)| 2558.00 | 4464.94 | 0.0474 |
| XGBoost       | 450.49  | 598.24  | 0.0082 |
| TFT           | 2327.10 | 3232.66 | 0.0426 |


## Conclusion 
- Use XGBoost for german electricity forecasting (Considering a context of low computational resources)


## Future Improvements

- Add probabilistic metrics (Pinball Loss, CRPS)

- Add multivariate forecasting (weather, renewables)

- Deploy best model as a REST API

- CI/CD pipeline for automated retraining

- Model monitoring & drift detection
