# Model Card — D2C Customer Churn Prediction Model

**Version:** 1.0  
**Date:** 2026-05-30  
**Authors:** [Anmoldeep Singh]  
**Contact:** [anmolramgardhia@gmail.com]

---

## 1. Model Overview

| Field | Detail |
|---|---|
| **Model Name** | D2C Churn Predictor v1 |
| **Model Type** | LightGBM Binary Classifier (Pipeline) |
| **Task** | Binary classification — predict whether a customer will make no purchase in the 60 days following `2025-09-30` |
| **Output** | `churn_probability` ∈ [0, 1]; `churn_class` ∈ {0, 1} |
| **Notebook Threshold (F1-optimal)** | **0.5795** — chosen by maximising F1 on the validation set during `churn_model.ipynb` |
| **Deployment Threshold (Part 4)** | **0.40** — deliberately lowered for production to favour recall (see §5 note below) |
| **Serialisation** | `model.pkl` (scikit-learn Pipeline via joblib) |

---

## 2. Intended Use

### ✅ Appropriate Uses

- **Internal CRM prioritisation:** Identifying which customers to contact first in a retention campaign
- **Budget allocation:** Directing limited campaign spend toward high-risk, high-value customers
- **Proactive outreach:** Triggering automated win-back emails or loyalty tier offers for customers crossing the risk threshold
- **Segment enrichment:** Combining churn probability with RFM segments to create richer customer profiles for the retention team

### ❌ Inappropriate Uses

- **Automatic denial of service or features** based on predicted churn risk — this model is for proactive retention, not punitive restrictions
- **Legal or contractual decisions** — churn probability is a probabilistic estimate, not a fact about customer intent
- **External customer communication** — customers should not be told they were flagged as churn-risk; this is an internal signal only
- **Real-time, per-transaction scoring** — this model is designed for batch, snapshot-based scoring, not live inference per event

---

## 3. Training Data

| Field | Detail |
|---|---|
| **Dataset** | D2C Customer Churn Capstone dataset |
| **Snapshot Date** | `2025-09-30` |
| **Target Definition** | `churn_next_60d = 1` if no purchase between `2025-10-01` and `2025-11-29`; else `0` |
| **Training Rows** | 1,728 |
| **Train Churn Rate** | 47.0% |

### Feature Sources

| Source Table | Features Used |
|---|---|
| `rfm_modeling_snapshot.csv` | All pre-built RFM features, web engagement, support signal aggregates |
| `customers.csv` (via rfm_snapshot) | `city_tier`, `age_group`, `acquisition_channel`, `loyalty_tier`, `preferred_category`, `marketing_consent` |

### Data Leakage Controls

- Post-snapshot order data (`order_date > 2025-09-30`) was **never used** as model input
- Target variable `churn_next_60d` was excluded from all feature sets
- `split` column was excluded from all feature sets
- All features were verified to represent state as of or before `2025-09-30`

---

## 4. Model Architecture

```
Input Features (numeric + categorical)
         │
         ▼
ColumnTransformer
  ├── Numeric:      MedianImputer → StandardScaler
  └── Categorical:  ConstantImputer('missing') → OrdinalEncoder
         │
         ▼
LightGBM Classifier
  ├── n_estimators: 300
  ├── learning_rate: 0.05
  ├── max_depth: 6
  ├── num_leaves: 31
  └── class_weight: 'balanced'
         │
         ▼
churn_probability → threshold → churn_class
```

**Baseline comparison:** Logistic Regression (same preprocessing, `class_weight='balanced'`)

---

## 5. Performance

### Validation Set

| Model | AUC-ROC | Avg Precision |
|---|---|---|
| Logistic Regression (baseline) | 0.8847 | 0.8690 |
| LightGBM (final) | 0.8754 | 0.8551 |

### ⚠️ Model Selection Rationale — Why LightGBM Despite Lower Validation Scores

> The table above shows Logistic Regression outperforming LightGBM on **both** validation metrics (AUC-ROC: +0.93pp; Avg Precision: +1.39pp). LightGBM was still selected as the final model. This is a deliberate, reasoned decision — not an oversight.

| Factor | Explanation |
|---|---|
| **Statistical noise at this dataset size** | The validation split contains approximately 400 customers. At this scale, a gap of ~1pp in AUC-ROC is within the expected variance of a single train/val split. It is not a reliable signal that LR generalises better — it may simply reflect a validation split that happens to favour LR's linear decision boundary. |
| **Non-linear feature interactions** | Churn is driven by combinations of signals, not individual features in isolation (e.g., *high recency AND low sessions AND discount-dependent*). LightGBM captures these multiplicative interactions natively via its tree structure. Logistic Regression cannot model them without explicit manual feature engineering. |
| **Richer interpretability for the CRM team** | LightGBM provides native feature importance scores and SHAP values, which power the error analysis (`error_analysis.md`) and the `risk_explanation` field in the API response. Logistic Regression coefficients are interpretable in isolation but do not generalise to interaction effects in the same way. |
| **Expected better generalisation at scale** | The small dataset (2,400 rows) limits LightGBM's ability to fully express its capacity advantage. As more customer data accumulates and retraining occurs (see `monitoring_plan.md`), LightGBM's performance is expected to improve relative to a linear model, while LR's improvement will plateau. |
| **Robustness to outliers and skew** | Several features (`monetary_180d`, `recency_days`, `avg_resolution_hours_90d`) have heavy right-skewed distributions with extreme outliers. Tree-based models are inherently robust to these without additional transformation. LR with StandardScaler partially mitigates this, but outliers still influence coefficient estimation. |
| **Production scalability** | LightGBM inference is faster on large batch requests and handles missing values natively, making it more suitable for the `/batch_predict` endpoint in Part 4. |

> **Conclusion:** LR's narrow validation edge (~1pp) is unlikely to be meaningful at this sample size and does not outweigh LightGBM's structural advantages for this use case. If the gap were larger (e.g., >3pp) or consistent across multiple cross-validation folds, it would warrant choosing LR. A future improvement would be to add **stratified k-fold cross-validation** to confirm which model generalises better with less variance from any single split.

### Test Set — Notebook Threshold (0.5795, F1-optimal)

The following metrics were computed at the **notebook threshold of 0.5795**, which maximises F1 on the validation set. These are the numbers recorded in `Part3/metrics.json`.

| Metric | Value |
|---|---|
| AUC-ROC | 0.8474 |
| Average Precision | 0.8090 |
| F1 Score | 0.7818 |
| Precision | 0.7963 |
| Recall | 0.7679 |
| Decision Threshold | **0.5795** |

### Confusion Matrix (Test Set @ threshold 0.5795)

|  | Predicted: Retained | Predicted: Churned |
|---|---|---|
| **Actual: Retained** | 135 | 33 |
| **Actual: Churned** | 39 | 129 |

---

### ⚠️ Threshold Change for Deployment (Part 4)

> **The deployment API (Part 4) uses a threshold of `0.40`, not `0.5795`.**
>
> This is a **deliberate, business-driven decision**, not an error:
>
> | Consideration | Reasoning |
> |---|---|
> | **Asymmetric cost** | A missed churner (False Negative) costs more than a wasted retention offer (False Positive). Lowering the threshold from 0.5795 → 0.40 increases recall at the cost of precision. |
> | **Campaign budget headroom** | Retention offers (₹20–₹35 per customer) are inexpensive relative to the lifetime value of a recovered churner, so over-flagging is acceptable. |
> | **Operational conservatism** | A round-number threshold (0.40) is easier to communicate to CRM and operations teams than a 4-decimal figure. |
> | **Configurable at runtime** | The deployment threshold is set via the `CHURN_THRESHOLD` environment variable in `Part4/app/main.py` so it can be adjusted without redeploying the model. |
>
> The F1 score reported in `Part4/metrics.json` (0.7899) reflects test-set performance recalculated at the 0.40 deployment threshold.

---

## 6. Top Predictive Features

Based on LightGBM feature importance and SHAP analysis:

| Rank | Feature | Direction | Interpretation |
|---|---|---|---|
| 1 | `recency_days` | ↑ higher → more churn | Customers who haven't ordered recently are at higher risk |
| 2 | `monetary_180d` | ↓ lower → more churn | Low historical spend predicts exit |
| 3 | `days_since_signup` | ↓ lower → more churn | Newer customers are more likely to churn than established ones |
| 4 | `last_visit_days_ago` | ↑ higher → more churn | Long absence from the app/site is a strong churn signal |
| 5 | `product_views_30d` | ↓ lower → more churn | Low recent engagement and browsing signals disengagement |
| 6 | `avg_discount_pct_180d` | ↑ higher → more churn | Highly discount-dependent customers churn when promotions end |
| 7 | `preferred_category` | Varies by category | Different categories have distinct natural replenishment cycles |
| 8 | `email_opens_30d` | ↓ lower → more churn | Tuning out of marketing communications indicates lost interest |

---

## 7. Limitations

1. **Snapshot-based, not real-time:** The model scores customers as of `2025-09-30`. It should be retrained regularly (see Monitoring Plan) — customer behaviour evolves.

2. **Label definition sensitivity:** "Churn" is defined as zero purchases in 60 days. A customer who placed one small, uncharacteristic order avoids the churn label even if they are effectively lost. This creates near-churn false negatives.

3. **Campaign effect not modelled:** The model does not know which customers received retention campaigns before or during the scoring window. Customers who were retained because of a campaign are labelled "not churned" — but the model may not generalise to uncontacted customers.

4. **Small dataset:** 2,400 customers is a relatively small training set. Model performance estimates may have higher variance than in production-scale deployments.

5. **Geographic / seasonal bias:** The training window covers a specific calendar period. Seasonal purchase patterns (e.g., festival buying) may cause the model to underperform if deployed outside this season.

6. **Imbalanced classes:** Churn rate is approximately 47.0%. The model uses `class_weight='balanced'` and a tuned threshold to compensate, but performance on the minority class may still be limited.

---

## 8. Ethical Risks

| Risk | Description | Mitigation |
|---|---|---|
| **Demographic bias** | If churn risk correlates with `city_tier` or `age_group` and these correlate with protected characteristics, the model may disproportionately flag or deprioritise certain demographic groups | Audit model performance across `city_tier` and `age_group` subgroups; ensure retention offers are equally accessible |
| **Disparate campaign access** | High-risk customers receive proactive outreach and offers; low-risk customers do not — but if low-risk = high RFM = wealthier customers, this may not be inequitable | Ensure Champions also receive relationship-building outreach, not only at-risk customers |
| **Self-fulfilling prophecy** | Customers labelled high-risk who receive no outreach may actually churn as a result of being ignored | Use the model to drive *more* outreach to high-risk customers, not to exclude them |
| **Stigmatisation** | Customers should never be told they were flagged as churn-risk internally | This is an internal tool only — do not expose predictions to customers |
| **Over-discounting** | If the retention team applies discounts to all flagged customers, it may train customers to expect discounts (see Part 2 analysis) | Follow retention strategy guidelines: match offer type to segment, not just risk score |

---

## 9. Monitoring Needs

See `monitoring_plan.md` in Part 4 for the full monitoring and retraining plan.

**Key metrics to track post-deployment:**

- **Prediction score drift:** Compare the distribution of `churn_probability` scores against the **47%** training baseline.
- **Label drift:** Monitor actual churn rate over time — if it diverges from the **47%** training baseline by more than ±5pp, the model may be stale.
- **Performance decay:** Re-evaluate on rolling test windows. Track AUC-ROC against the **0.8474** baseline and F1 Score against the **0.7899** baseline.
- **Feature distribution shift:** Track mean and variance of `recency_days`, `sessions_30d`, `monetary_180d` — sudden shifts indicate product or market changes.

**Recommended retraining trigger:** Initiate a retraining cycle if validation AUC-ROC drops > 3pp below the **0.8474** baseline, or if the actual churn rate shifts by > ±5pp from the **47%** baseline for two consecutive months.

---

## 10. Responsible Use Note

This model is an **assistance tool for the retention team**, not a decision-making authority.

**Do:**
- Use churn probability as one input among several when deciding who to contact
- Cross-reference with support tickets, manual priority flags, and segment context
- Apply human judgment for edge cases and high-value customers

**Do not:**
- Automatically deny benefits or downgrade service based on predicted churn risk
- Use the model output without reviewing individual customer context
- Trust the model blindly for customers where the signal is weak (churn_probability between 0.40 and 0.60)
- Share individual churn probability scores externally or with the customer
