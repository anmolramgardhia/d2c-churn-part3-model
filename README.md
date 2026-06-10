# Part 3 — Churn Prediction Model & Model Card

## Overview

This part builds a machine-learning churn-prediction model to identify customers likely to make no purchase in the 60 days following the snapshot date (`2025-09-30`).

**Target:** `churn_next_60d` (1 = churned, 0 = retained)  
**Train/Val/Test split:** Pre-assigned in `churn_labels.csv`

---

## File Structure

```
Part3-Churn-Model/
├── churn_model.ipynb     ← Full modelling notebook
├── model.pkl             ← Saved trained model (generated after run)
├── metrics.json          ← Evaluation metrics (generated after run)
├── error_analysis.md     ← FP/FN analysis with 10 customer examples
├── model_card.md         ← Model card: use, data, performance, ethics
├── requirements.txt
└── README.md
```

---

## Python Environment

This project was developed and tested with the following environment:

| Property | Value |
|---|---|
| **Python Version** | Python 3.14.4 |
| **Python Executable** | `~\AppData\Local\Python\pythoncore-3.14-64\python.exe` |
| **OS** | Windows 11 (64-bit) |

> **Note:** To avoid compatibility issues, please use **Python 3.14.x (64-bit)**. You can download it from [python.org](https://www.python.org/downloads/). On Windows, run commands using `python` (not `python3`).

---

## Setup & Run Instructions

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd Part3-Churn-Model
```

### 2. Create a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Place the dataset files
```
data/
├── rfm_modeling_snapshot.csv   ← Primary feature table
├── churn_labels.csv            ← Target + split
└── (optional raw CSVs for custom feature engineering)
```

### 5. Run the notebook
```bash
jupyter notebook churn_model.ipynb
```
Run all cells top to bottom (`Kernel > Restart & Run All`).  
The notebook will save `model.pkl` and `metrics.json` in the working directory.

---

## Outputs

| File | Description |
|---|---|
| `churn_model.ipynb` | Full modelling notebook |
| `model.pkl` | Serialised final model (scikit-learn Pipeline) |
| `metrics.json` | Evaluation metrics dict (AUC, F1, precision, recall, etc.) |
| `error_analysis.md` | Analysis of false positives and false negatives |
| `model_card.md` | Structured model card |

---

## Model Summary

| Model | Validation AUC | Validation Avg Precision | Notes |
|---|---|---|---|
| Logistic Regression (baseline) | **0.8847** | **0.8690** | Higher on both validation metrics |
| LightGBM | 0.8754 | 0.8551 | ✅ **Selected as final model** |

> **Why LightGBM when LR scores higher?**
> The ~1pp validation gap is within the expected statistical noise for a single ~400-row validation split and is not a reliable signal of better generalisation. LightGBM was chosen for its ability to capture **non-linear feature interactions** (e.g., high recency × zero sessions × discount-dependency), native **SHAP / feature importance** for CRM explainability, robustness to skewed outlier features, and better expected performance as the dataset grows with retraining cycles. See `model_card.md §5` for the full six-factor rationale.

Decision threshold: **0.5795** (F1-optimal, tuned on the validation set in `churn_model.ipynb`)

> **Deployment note:** The Part 4 FastAPI service uses a threshold of **0.40** — deliberately lower to favour recall, because the cost of missing a churner exceeds the cost of a wasted retention offer. See `model_card.md §5` for the full rationale.

---

## Notes on Leakage Prevention

- Only features from `rfm_modeling_snapshot.csv` (all safe, pre-snapshot) are used.
- `churn_next_60d` and `split` columns are excluded from feature inputs.
- Post-snapshot order data is never used.
