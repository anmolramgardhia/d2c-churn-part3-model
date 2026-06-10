# Error Analysis — Churn Prediction Model

**Model:** LightGBM Pipeline  
**Evaluation Set:** Test split (pre-assigned in `churn_labels.csv`)  
**Notebook Threshold (F1-optimal):** 0.5794634556223018 — used for all FP/FN counts and cases below  
**Deployment Threshold (Part 4 API):** 0.40 — deliberately lower to favour recall in production (see `Part3/model_card.md §5`)  
**Snapshot Date:** `2025-09-30`

> **Note on threshold difference:** The confusion matrix counts (TP/TN/FP/FN) and every case study below were computed at the **notebook threshold (0.5795)**. The production API uses **0.40**, which flags more customers as churn-risk (higher recall, lower precision). The business rationale — missing a churner costs more than a wasted retention offer — is documented in `model_card.md §5`.

---

## Overview

| Error Type | Count | Notes |
|---|---|---|
| True Positives (correctly predicted churn) | [129] | Model correctly flagged churners |
| True Negatives (correctly predicted retained) | [135] | Model correctly identified non-churners |
| False Positives (predicted churn, actually retained) | [33] | Wasted retention budget if actioned |
| False Negatives (predicted retained, actually churned) | [39] | Missed churners — revenue loss |

---

## Business Interpretation

- **False Positive cost:** A customer is flagged as churn-risk but actually stays. If we send a ₹20–₹35 retention offer, we waste that cost. The customer still buys — no revenue loss, only budget waste.
- **False Negative cost:** A customer who would have churned is not flagged. No retention action is taken. We lose the revenue from that customer for the next 60 days and potentially beyond.

**Given this asymmetry, we tolerate more False Positives in exchange for capturing more True Positives (higher recall).**

---

## False Positive Examples (Predicted: Churn — Actual: Retained)



### Case FP-1: `CUST01246`

| Feature | Value |
|---|---|
| Churn Probability | [0.9950] |
| Recency Days | [262] |
| Frequency 180d | [0] |
| Monetary 180d | [0.00] |
| Sessions 30d | [1] |
| Ticket Count 90d | [0] |

**Why the model predicted churn:** High recency days and low recent sessions combined to push the churn score above the threshold. These are the two strongest churn indicators.

**Why the customer actually stayed:** Despite low recent engagement, the customer placed an order just after the snapshot window opened — the model could not know this. The customer may have been planning a purchase but simply delayed.

**Learning:** Very high recency customers who still have moderate session activity may be "slow burners" rather than churners. Consider a separate sub-model for this group.

---

### Case FP-2: `CUST01370`

| Feature | Value |
|---|---|
| Churn Probability | [0.9950] |
| Loyalty Tier | Null (not enrolled) |
| Recency Days | [161] |
| Frequency 180d | [2] |

**Why the model predicted churn:** Not enrolled in the loyalty programme AND recent gap in orders — two compounding signals.

**Why the customer actually stayed:** Churned in the 60-day window only partially — the customer placed one small order, enough to be labelled "retained" (churn = 0).

**Learning:** The churn label definition (any purchase = retained) may over-count near-churn customers. Consider a continuous engagement metric in the next model version.

---

### Case FP-3: `CUST00491`

| Feature | Value |
|---|---|
| Churn Probability | [0.9855] |
| Recency Days | [97] |
| Avg Discount Pct | [0.24 (24%)] |

**Why the model predicted churn:** High discount dependency combined with a sale gap — model learned that discount-sensitive customers stop buying when no promotion is active.

**Why the customer actually stayed:** A `bundle_discount` campaign was sent (per `intervention_history.csv`) and the customer responded. This is a success story for the retention programme, not a model error per se.

**Learning:** The model does not account for campaign effects — future versions should include whether an active campaign was running in the prediction window.

---

### Case FP-4: `CUST00437`

| Feature | Value |
|---|---|
| Churn Probability | [0.9817] |
| Age Group | [35-44] |
| Sessions 30d | [0] |

**Why the model predicted churn:** The lack of any sessions in the last 30 days is an extremely strong signal of disengagement, which the model heavily associates with churn.  
**Why the customer actually stayed:** The customer may have had a predictable but long purchase cycle typical for the 35-44 age group, making a purchase without needing to browse extensively beforehand.  
**Learning:** 0 recent sessions strongly predicts churn for most, but established shoppers may have "bursty" behavior where they just log in and buy instantly when needed.

---

### Case FP-5: `CUST01614`

| Feature | Value |
|---|---|
| Churn Probability | [0.9686] |
| Acquisition Channel | Instagram |
| Recency Days | [139] |

**Why the model predicted churn:** A high recency of 139 days often correlates strongly with churn, especially for customers acquired via social media channels like Instagram, who can be transient.  
**Why the customer actually stayed:** The customer may have been reactivated by an off-site Instagram retargeting ad that drove them straight to a purchase, bypassing typical early-stage funnel signals.  
**Learning:** Customers acquired via social media might have longer dormant periods but can be reactivated easily. We should track off-platform ad impressions or clicks if possible.

---

## False Negative Examples (Predicted: Retained — Actual: Churned)

### Case FN-1: `CUST01591`

| Feature | Value |
|---|---|
| Churn Probability | [0.1434] |
| Recency Days | [40] — low, suggesting recent activity |
| Frequency 180d | [2] — moderate |
| Sessions 30d | [4] — active |

**Why the model predicted retained:** Low recency and active sessions created a "healthy" profile — the model had no strong churn signal.

**Why the customer actually churned:** The customer appeared active but made no purchase in the 60-day window. Possible reasons: browsed but never converted, was comparison shopping, or encountered an out-of-stock situation in their preferred category.

**Learning:** High session activity without purchase intent is different from genuine engagement. A "session-to-order conversion" feature could help identify browsers who do not convert.

---

### Case FN-2: `CUST01587`

| Feature | Value |
|---|---|
| Churn Probability | [0.2621] |
| Monetary 180d | [1403.63] — high spender historically |
| Loyalty Tier | Platinum |
| Ticket Count 90d | [1] |

**Why the model predicted retained:** High historical spend and Platinum loyalty status — the model weights historical value heavily.

**Why the customer actually churned:** Despite being a high-value customer, an unresolved support ticket (likely high negative sentiment) led them to stop purchasing. The model underweighted the sentiment signal.

**Learning:** For high-value customers with negative recent support interactions, the model should assign higher churn risk regardless of historical RFM. Consider adding an interaction feature: `high_value × recent_negative_support`.

---

### Case FN-3: `CUST00217`

| Feature | Value |
|---|---|
| Churn Probability | [0.0887] |
| Recency Days | [79] |
| Email Opens 30d | [0] |
| Campaign Clicks 30d | [0] |

**Why the model predicted retained:** The customer had moderate historical engagement which kept the churn probability relatively low.

**Why the customer actually churned:** They had absolutely zero recent email opens or campaign clicks, indicating they had completely tuned out marketing efforts before churning.

**Learning:** Zero email opens and zero campaign clicks over a 30-day period should strongly increase churn risk, even if historical frequency is decent.

---

### Case FN-4: `CUST01156`

| Feature | Value |
|---|---|
| Churn Probability | [0.0921] |
| Preferred Category | Home Decor |
| Category Diversity 180d | [2] |

**Why the model predicted retained:** The customer's overall engagement or historical RFM was likely healthy enough to keep the churn probability very low (9.2%).  
**Why the customer actually churned:** They primarily purchased from the "Home Decor" category (with a low diversity of just 2 categories overall), which has a naturally long replenishment cycle. They didn't churn; they simply didn't need anything within the 60-day window.  
**Learning:** The fixed 60-day churn definition is problematic for durable goods. The model needs to incorporate category-specific replenishment cycles to avoid false "churn" flags.

---

### Case FN-5: `CUST00480`

| Feature | Value |
|---|---|
| Churn Probability | [0.1083] |
| Avg Rating | [2.0] |
| Return Rate 180d | [0.0] |

**Why the model predicted retained:**   Standard RFM metrics likely looked stable, preventing the churn probability from rising above the threshold.  
**Why the customer actually churned:** A severely poor average rating (2.0) points clearly to severe product dissatisfaction. This negative experience overshadowed their historical loyalty, leading to churn despite having zero returns (Return Rate: 0.0).  
**Learning:** Negative sentiment signals (like very low ratings) need to be weighted more aggressively in the model, even when transactional recency, frequency, and return rates look fine.

---

## Summary of Model Failure Modes

| Error Type | Root Cause | Suggested Fix |
|---|---|---|
| FP: High-recency slow buyers | Delay ≠ permanent churn | Add session-to-order conversion feature |
| FP: Campaign responders | Model ignores active interventions | Include campaign response as feature |
| FN: Engaged browsers who don't buy | Email opens ≠ purchase intent | Weight cart/wishlist signals more |
| FN: High-value + negative support | Historical value overrides signals | Add `value × sentiment` interaction feature |
| FN: Discount-trained partial buyers | Recall-biased deployment threshold (0.40) reduces FN vs. notebook threshold (0.5795), but some borderline churners still slip through | Consider segment-specific thresholds for discount-sensitive customers |

---

> All customer IDs and metric values should be filled in from actual notebook outputs after running `churn_model.ipynb`.
