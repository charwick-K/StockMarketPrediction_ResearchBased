# Risk-Aware & Interpretable Deep Temporal Modeling for Financial Forecasting

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Status](https://img.shields.io/badge/Focus-Research%20Grade-success)

---

## 🔬 Overview

This repository presents a **research-driven framework for financial time-series forecasting** that prioritizes **generalization, interpretability, and evaluation integrity** over raw metric optimization.

Instead of proposing a single predictive model, this work investigates **how different temporal inductive biases affect learning behavior, risk sensitivity, and decision trustworthiness** under strict leak-free conditions.

Each model in this repository represents a **controlled research hypothesis**, not a tutorial or benchmark replication.

---

## 🎯 Key Research Objectives

- Study the **capacity limits of recurrent models** for financial data  
- Separate **temporal memory extraction** from **decision modeling**
- Capture **short-term micro-patterns** alongside **long-term temporal dependencies**
- Introduce **temporal interpretability** via attribution-oriented attention
- Enforce **leak-free evaluation** that mirrors real-world deployment constraints

---

## 🧠 Core Contributions

### 1. Risk-Aware Learning Formulation
- Models are trained in **log-return space** for numerical stability
- Evaluation is performed in **original price space** for financial realism
- Avoids misleading performance gains caused by scale mismatch

### 2. Architecture-as-Hypothesis Design
Each model is designed to answer a **specific research question**, rather than competing blindly on accuracy.

### 3. Leak-Free Evaluation Protocol
- Forward-chaining `TimeSeriesSplit`
- Strict temporal separation of Train / Validation / Final Test
- No random shuffling, no look-ahead bias

### 4. Interpretability by Design
- Attention mechanisms are treated as **temporal attribution tools**
- Enables post-hoc analysis of *which historical periods influenced decisions*

---

## 🏗️ Model Suite & Research Intent

| File | Research Purpose |
|---|---|
| `temporal_baseline_lstm.py` | Establishes lower-bound performance of pure recurrence |
| `structured_temporal_regressor.py` | Separates memory extraction from decision regression |
| `local_global_temporal_fusion.py` | Models short-term micro-patterns + long-term trends |
| `interpretable_temporal_attention.py` | Attribution-aware forecasting with explainability |

Each file is **self-contained, reproducible, and leak-free**.

---

## ⚙️ Methodology

### Data Transformation
Raw closing prices are converted to log-returns:

\[
r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)
\]

This stabilizes variance and improves temporal learning dynamics.

### Sliding Window Supervision
A fixed-length window transforms the time series into supervised sequences while preserving temporal order.

### Evaluation Metrics
All models are evaluated on **unseen data** using:
- RMSE (Price Scale)
- MAE
- R² Score

---

## 📁 Repository Structure

```

.
├── data/
│   └── stock.csv
│
├── temporal_baseline_lstm.py
├── structured_temporal_regressor.py
├── local_global_temporal_fusion.py
├── interpretable_temporal_attention.py
│
└── README.md

````

---

## ▶️ How to Run

1. Place your dataset in `data/stock.csv`
   - Required columns: `date`, `close`
2. Choose a model file
3. Run directly:
```bash
python interpretable_temporal_attention.py
````

Each script performs:

* Cross-validation
* Final training
* Evaluation on fully unseen test data

---

## 📈 Why This Matters (2025 Context)

Most deep-learning-based financial models:

* Optimize metrics without interpretability
* Ignore temporal leakage
* Fail under real deployment constraints

This framework explicitly addresses these gaps by combining:

* **Risk-aware learning**
* **Transparent temporal attribution**
* **Evaluation discipline**

---

## 🔮 Future Extensions

* Probabilistic forecasting (quantile / Bayesian heads)
* Volatility-aware or risk-sensitive loss functions
* Multi-asset joint learning
* Regulatory-grade explainability analysis
* Integration with portfolio decision systems

---

## 📌 Intended Use

* Research experimentation
* Advanced academic projects
* Internship / R&D portfolio
* Basis for IEEE-style conference or journal submission

---

## 📜 Disclaimer

This repository is intended for **research and educational purposes only**.
It does not constitute financial or investment advice.
