# ML Housing Prediction

This repository contains machine learning experiments and feature engineering for housing price data analysis using Python and scikit-learn.

## Project Overview

The goal of this project is to analyze housing data (such as median house value, median income, room counts, etc.), apply data preprocessing and feature engineering, and build regression models to predict housing prices. This work is primarily based on the California housing dataset.

## Data Preparation

- Loaded data from CSV and handled missing values (e.g., median imputation for `total_bedrooms`).
- Removed outliers using thresholds on features like `total_rooms`, `total_bedrooms`, `households`, `population`, and `median_income` (using boxplots for visualization).
- Created new features:
  - `rooms_per_household`
  - `bedrooms_per_room`
  - `population_per_household`
- One-hot encoded the categorical `ocean_proximity` feature.

## Modeling Approaches Tried

### Linear Regression

- Standardized features using `StandardScaler`.
- Tried both simple linear regression and a pipeline with feature scaling.
- Achieved R² score on test set: ~0.66

### Ridge Regression

- Used `Ridge` regression with regularization (alpha tuning).
- Also integrated into a pipeline with scaling and polynomial features.
- R² score: ~0.56 (with basic features); improved with more feature engineering.

### Polynomial & Pipeline

- Used `PolynomialFeatures` to capture non-linear relationships.
- Combined with `StandardScaler` and `Ridge` in a `Pipeline`.
- Example pipeline:
  ```
  PolynomialFeatures(degree=3) → StandardScaler → Ridge(alpha=1)
  ```
- R² score: ~0.68

### LightGBM Regressor

- Applied `LightGBM` for gradient boosting regression.
- Used transformed and scaled features.
- Achieved R² score: ~0.68

### Feature Engineering

- Explored the impact of new features (room/household ratios, population ratios).
- Used boxplots to visualize and remove outliers.
- One-hot encoding for categorical data.

## Metrics

- Common metrics calculated: Mean Absolute Error (MAE), Mean Squared Error (MSE), R² score.
- Example:
  - `MAE: ~47000`
  - `MSE: ~4.2e9`
  - `Best R²: ~0.68` (Polynomial Ridge & LightGBM)

## Notebooks & Scripts

- The core workflow is in `ML.ipynb`.
- All data preparation, modeling, evaluation, and visualizations are demonstrated in this notebook.

## Dependencies

- Python 3.x
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- lightgbm

## How to Run

1. Clone the repo and install dependencies.
2. Place your housing CSV data in the project directory.
3. Run `ML.ipynb` for the full workflow and analysis.

---

Feel free to explore the notebook and try your own feature engineering or model tuning!
