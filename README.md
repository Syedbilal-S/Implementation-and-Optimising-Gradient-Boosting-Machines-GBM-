# Implementing and Optimizing Gradient Boosting Machines (GBM) from Scratch

This project implements the core logic of a **Gradient Boosting Machine (GBM)** for regression **from scratch** using only NumPy and basic Python – no high-level ML libraries for the custom model.  

It also benchmarks the custom implementation against `sklearn.ensemble.GradientBoostingRegressor` on a **housing dataset** (`housing.csv`) or a **synthetic regression dataset**.

---

## 1. Project Goals

- Understand the internal mechanics of Gradient Boosting:
  - Residual fitting (negative gradients)
  - Additive model construction
  - Learning rate and number of estimators
- Implement a **CART-style regression tree** base learner from scratch.
- Build a **Gradient Boosting Regressor from scratch** on top of these trees.
- Perform **hyperparameter tuning (grid search)**.
- **Benchmark**:
  - Custom GBM vs `sklearn` GradientBoostingRegressor
  - Compare MSE, R², and training time.

---

## 2. Repository Structure

- `gbm_from_scratch.py`  
  Main script containing:
  - `RegressionTree` – simple regression tree implementation.
  - `GradientBoostingRegressorScratch` – custom GBM implementation.
  - Utility functions for:
    - Loading housing data (`housing.csv`)
    - Generating synthetic regression data
    - Hyperparameter grid search
    - Benchmarking vs scikit-learn.

- `housing.csv`  
  Tabular housing dataset with numeric features and a target column (e.g., house price).

> ⚠️ Make sure `housing.csv` is in the **same folder** as `gbm_from_scratch.py`.

---

## 3. Requirements

- Python 3.8+
- Recommended packages:

```bash
pip install numpy pandas scikit-learn
