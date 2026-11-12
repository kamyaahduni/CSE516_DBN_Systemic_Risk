# Results Documentation

## Overview

This directory contains all experimental results from the Bayesian Network credit default prediction system (M4 Final Submission).

**Experiment:** Structure learning (PC Algorithm) + Parameter learning (MLE) + Inference (Belief Propagation)  
**Dataset:** Synthetic credit data (500 samples, 11 features, balanced classes)  
**Train/Val/Test Split:** 70% / 15% / 15%  
**Seed:** 42  
**Execution Date:** November 12, 2026

---

## File Contents

### 1. `initial_kpi_results.csv`

**Purpose:** Twelve Key Performance Indicators (KPIs) across three evaluation dimensions

**Format:** Single row, 12 columns

**Columns:**

| KPI | Value | Target | Status | Interpretation |
|-----|-------|--------|--------|---|
| `stability` | 0.70 | ≥0.65 | ✓ PASS | Network structure stable across data subsamples |
| `sota_alignment` | 0.75 | ≥0.70 | ✓ PASS | Structure aligns with state-of-the-art methods |
| `significance` | 0.0 | ≥0.50 | ✗ FAIL | 0% of edges statistically significant (synthetic data limitation) |
| `domain_fit` | 1.0 | ≥0.80 | ✓ PASS | Structure perfect match to financial domain |
| `ci_width` | 0.193 | ≤0.30 | ✓ PASS | Parameter confidence intervals reasonably narrow |
| `param_significance` | 0.0 | ≥0.50 | ✗ FAIL | 0% of parameters statistically significant (poor learning) |
| `train_r2` | 0.132 | ≥0.15 | ≈ NEAR | Training R² slightly below target |
| `val_r2` | 0.091 | ≥0.10 | ≈ NEAR | Validation R² near target, minimal overfitting |
| `auc_roc` | 0.8107 | ≥0.75 | ✓ PASS | Good discrimination ability |
| `vs_baseline` | -4.5 | ≥0 | ✗ FAIL | **Model performs 4.5% WORSE than baseline** |
| `calibration_error` | 0.258 | ≤0.20 | ✗ FAIL | **Calibration error EXCEEDS target** |
| `interpretability` | 0.85 | ≥0.70 | ✓ PASS | 5 discovered edges provide good interpretability |

**Summary:**
- **Total KPIs:** 12
- **Pass:** 5 (42%)
- **Near Target:** 2 (17%)
- **Fail:** 5 (42%)

**Critical Findings:**
1. Baseline comparison is NEGATIVE (-4.5%)
2. Calibration error exceeds target (0.258 vs 0.20)
3. Parameter learning is weak (0% significant)

---

### 2. `parameters.csv`

**Purpose:** Learned parameters for each node in the Bayesian Network

**Format:** 11 rows (one per feature), 9 columns

**Columns:**

| Column | Meaning | Example | Interpretation |
|--------|---------|---------|---|
| `node` | Feature index | Feature_1 | Which node this row describes |
| `coefficient` | β₀ (intercept) or mean | 0.0027 | Baseline prediction when parents are zero |
| `parents_coef` | Parent regression weights | [0.891] | How much each parent influences this node |
| `ci_lower` | 95% CI lower bound | -0.0445 | Lower confidence limit |
| `ci_upper` | 95% CI upper bound | 0.0499 | Upper confidence limit |
| `ci_width` | CI width (precision) | 0.0944 | Narrower = more precise |
| `p_value` | Statistical significance | 0.911 | >0.05 = not significant |
| `r_squared` | Variance explained | 0.812 | % of variance explained by parents |
| `mse` | Mean squared error | 0.200 | Prediction error magnitude |

**Key Observations:**

**Features with SIGNIFICANT learning (R² > 0.5):**
- Feature_1: R² = 0.812, p-value < 0.05 ✓
- Feature_2: R² = 0.603, p-value = 0.328 ≈

**Features with NO significant learning (R² ≈ 0):**
- Feature_0: R² ≈ 0, p-value = 0.112 ✗
- Feature_3: R² ≈ 0, p-value = 0.704 ✗
- Feature_4: R² ≈ 0, p-value = 0.435 ✗
- Feature_5: R² ≈ 0, p-value = 0.652 ✗
- Feature_6: R² ≈ 0, p-value = 0.568 ✗
- Feature_7: R² ≈ 0, p-value = 0.254 ✗
- Features 8-10: R² ≈ 0.01-0.02, marginal learning

**Interpretation:** Only 2 out of 11 features learned meaningful relationships. The other 9 features are predicted as constants, which severely limits model utility.

---

### 3. `predictions_test_set.csv`

**Purpose:** Predicted and actual outcomes for all 75 test samples

**Format:** 75 rows (one per test sample), 2 columns

**Columns:**

| Column | Range | Meaning |
|--------|-------|---------|
| `prediction` | [0, 1] | Predicted default probability from Bayesian Network |
| `y_true` | {0, 1} | Actual default status (binary) |

**Usage Example:**
```python
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix

# Load predictions
df = pd.read_csv('predictions_test_set.csv')

# Calculate AUC-ROC
auc = roc_auc_score(df['y_true'], df['prediction'])
print(f"AUC-ROC: {auc:.4f}")  # Output: 0.8107

# Confusion matrix (threshold 0.5)
cm = confusion_matrix(df['y_true'], (df['prediction'] > 0.5).astype(int))
# [[tn, fp],
#  [fn, tp]]
```

**Summary Statistics:**
- Min prediction: 0.087
- Max prediction: 0.923
- Mean prediction: 0.50
- Std prediction: 0.23
- Actual defaults: 38/75 (50.7%)
- Predicted > 0.5: 40/75 (53.3%)

---

### 4. `learned_dag.json`

**Purpose:** Learned Directed Acyclic Graph (DAG) structure

**Format:** JSON with edges and metadata

**Contents:**
```json
{
  "edges": [
    [0, 1], [1, 2], [4, 10], [5, 8], [6, 9]
  ],
  "n_edges": 5,
  "description": "Directed acyclic graph learned via PC Algorithm"
}
```

**Interpretation:**
- Feature_0 → Feature_1
- Feature_1 → Feature_2
- Feature_4 → Feature_10
- Feature_5 → Feature_8
- Feature_6 → Feature_9

**Properties:**
- **Nodes:** 11 (all features)
- **Edges:** 5 (sparse structure)
- **Edge density:** 5/110 = 4.5% (9% if considering direction)
- **Maximum in-degree:** 1 (most nodes have ≤ 1 parent)
- **Acyclicity:** Enforced by PC Algorithm

---

### 5. `auc_curve.png`

**Purpose:** ROC curve visualization

**Content:** ROC curve plot showing true positive rate vs false positive rate

**Interpretation:**
- Curve above diagonal = model better than random
- AUC = 0.8107 = Area under curve
- Larger AUC = better discrimination

---

## Missing Analyses

**IMPORTANT:** The following analyses were mentioned in initial documentation but were NOT PERFORMED:

###  Cross-Seed Validation

**Claimed:** Mean AUC 0.8110 ± 0.0110 across 5 seeds (42, 123, 456, 789, 999)

**Actual:** Only seed 42 results present

**Missing files:**
- predictions_test_seed123.csv
- predictions_test_seed456.csv
- predictions_test_seed789.csv
- predictions_test_seed999.csv
- kpi_results_seeds_aggregated.csv

**Why this matters:** Cannot verify robustness across different random initializations

###  Sample Size Sensitivity Analysis

**Claimed:** AUC performance at different training set sizes:
- 50% data: AUC 0.7834
- 70% data: AUC 0.8107
- 90% data: AUC 0.8245
- 100% data: AUC 0.8312

**Actual:** Only 70% training data tested

**Missing files:**
- predictions_test_set_50pct.csv
- predictions_test_set_90pct.csv
- predictions_test_set_100pct.csv

**Why this matters:** Cannot verify monotonic improvement or scaling behavior

###  Multiple Baseline Comparisons

**Claimed:** Comparison against 5 baselines (LR, RF, DT, GB, NN)

**Actual:** Only Logistic Regression baseline computed

**Missing baselines:**
- Random Forest results
- Decision Tree results
- Gradient Boosting results
- Neural Network results

**Why this matters:** Cannot claim competitive performance vs state-of-the-art

###  Bootstrap Confidence Intervals for AUC

**Claimed:** 95% CI via 1000 bootstrap resamples

**Actual:** Not computed

**Missing:** auc_bootstrap_ci.csv

**Why this matters:** Cannot quantify uncertainty in AUC estimate

---

## Performance Summary

### Overall Assessment

| Dimension | Status | Evidence |
|-----------|--------|----------|
| **Discrimination** | ✓ GOOD | AUC 0.8107 exceeds 0.75 target |
| **Baseline Comparison** | ✗ NEGATIVE | -4.5% worse than logistic regression |
| **Calibration** | ✗ POOR | 0.258 exceeds 0.20 target |
| **Robustness** |  UNKNOWN | Only 1 seed tested, not 5 |
| **Sensitivity** |  UNKNOWN | Only 70% training data, not multiple sizes |
| **Parameter Learning** | ✗ WEAK | Only 2/11 features significantly learned |
| **Structure Quality** | ✓ GOOD | 5 edges, stable, domain-aligned |

### Conclusion

The Bayesian Network demonstrates **good discrimination ability (AUC 0.8107)** and **interpretable structure (5 edges)**, but suffers from:

1. **Poor calibration** - Probabilities not reliable for risk quantification
2. **Negative baseline comparison** - Underperforms simpler logistic regression
3. **Weak parameter learning** - 9/11 features learned as constants
4. **Unverified robustness** - No cross-seed or sensitivity validation

**Production Readiness:** NOT READY. Further improvements required before real-world deployment.

---

## Reproducibility

To reproduce these results:

```bash
cd Code/src
python3 main.py --seed 42
```

**Requirements:** See requirements.txt

**Output Directory:** Results/

**Current Limitations:**
- Seed is hardcoded; changing requires source modification
- Only 70% training data; different fractions require code changes
- No baseline model trained within this directory; external comparison needed

---

## References

Results interpretation based on:

1. Fawcett, T. (2006). "An Introduction to ROC Analysis." *Pattern Recognition Letters*, 27(8), 861-874.
   - ROC curve and AUC interpretation

2. Niculescu-Mizil, A., & Caruana, R. (2005). "Predicting Good Probabilities with Supervised Learning." *ICML*, 625-632.
   - Probability calibration methods and metrics

---

**Document Version:** M4 Final - CORRECTED  
**Last Updated:** November 12, 2026, 21:10 IST
