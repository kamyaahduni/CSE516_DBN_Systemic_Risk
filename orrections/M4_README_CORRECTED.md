# M4: Bayesian Networks for Credit Default Prediction - Final Report

**Group:** G15  
**Date:** November 2026  
**Course:** CSE 516 - Probabilistic Graphical Models  
**Institution:** Stony Brook University  

---

## Executive Summary

This milestone presents the complete implementation and evaluation of a Bayesian Network for credit default prediction. The model uses structure learning (PC Algorithm), parameter learning (OLS regression), and inference (Belief Propagation) on a 500-sample synthetic credit dataset.

**Key Result:** The Bayesian Network achieves AUC-ROC of 0.8107 but performs 4.5% WORSE than the baseline logistic regression model. This indicates that the BN structure learned by the PC Algorithm does not effectively capture predictive relationships in this dataset.

---

## Methodology Overview

### 1. Data Preparation
- **Dataset:** 500 synthetic credit samples, 11 continuous features
- **Target:** Binary default status (balanced: 249 defaults, 251 non-defaults)
- **Preprocessing:** StandardScaler normalization (μ=0, σ=1)
- **Split:** 70% training (350), 15% validation (75), 15% test (75)
- **Reproducibility:** Fixed seed = 42

### 2. Structure Learning: PC Algorithm
- **Method:** Constraint-based causal discovery
- **Significance Threshold:** α = 0.05
- **Result:** Learned DAG with 5 edges
  - Feature_0 → Feature_1
  - Feature_1 → Feature_2
  - Feature_4 → Feature_10
  - Feature_5 → Feature_8
  - Feature_6 → Feature_9

**Interpretation:** The algorithm discovered moderate conditional dependencies. The sparse structure suggests most features are conditionally independent given others.

### 3. Parameter Learning: OLS Regression
- **Method:** Ordinary Least Squares on each conditional distribution
- **Confidence Intervals:** 95% via Fisher information matrix
- **Result:** Mixed parameter quality

| Parameter Quality | Count | Interpretation |
|-------------------|-------|-----------------|
| Strong (R² > 0.60) | 1 | Only Feature_1 learns meaningful patterns |
| Moderate (0.01 < R² < 0.60) | 2 | Features 2, 8, 9 learn weakly |
| Weak (R² ≈ 0) | 8 | Features 0,3,4,5,6,7,10 predict constants |

**Critical Issue:** 91% of the learned network (8 out of 11 features) has R² ≈ 0, meaning parameter learning failed for most features. These features simply predict the mean value with no real pattern capture.

### 4. Inference: Belief Propagation
- **Method:** Message passing on learned DAG
- **Output:** P(default | evidence) for each test sample
- **Evaluation:** 12 KPIs across 3 pillars

---

## Results & Evaluation

### Primary Metrics (Test Set, Seed 42)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **AUC-ROC** | 0.8107 | ≥ 0.80 | ✅ PASS |
| **vs Baseline (LR)** | -4.5% | ≥ 0% | ❌ FAIL |
| **Calibration Error** | 0.258 | ≤ 0.20 | ❌ FAIL |

**Key Finding:** While the AUC passes the 0.80 target, the model performs worse than the simple Logistic Regression baseline (-4.5%) and fails calibration quality (0.258 vs 0.20 target).

### Representation Pillar

| KPI | Value | Target | Status |
|-----|-------|--------|--------|
| Stability | 0.70 | ≥ 0.65 | ✅ PASS |
| SOTA Alignment | 0.75 | ≥ 0.70 | ✅ PASS |
| Significance | 0.0% | > 50% | ❌ FAIL |
| Domain Fit | 1.0 | ≥ 0.80 | ✅ PASS |

**Interpretation:** Stability and domain fit are good. However, 0% parameter significance indicates the learned parameters lack statistical significance - most features don't meaningfully predict the target.

### Learning Pillar

| KPI | Value | Target | Status |
|-----|-------|--------|--------|
| CI Width | 0.193 | ≤ 0.25 | ✅ PASS |
| Param Significance | 0.0% | > 50% | ❌ FAIL |
| Train R² | 0.132 | ≥ 0.40 | ❌ FAIL |
| Val R² | 0.091 | ≥ 0.35 | ❌ FAIL |

**Critical Issue:** Parameter significance is 0%, and both R² values are far below targets. This confirms the fundamental weakness: the model doesn't learn meaningful patterns from data.

### Inference Pillar

| KPI | Value | Target | Status |
|-----|-------|--------|--------|
| AUC-ROC | 0.8107 | ≥ 0.80 | ✅ PASS |
| vs Baseline | -4.5% | ≥ 0% | ❌ FAIL |
| Calibration Error | 0.258 | ≤ 0.20 | ❌ FAIL |
| Interpretability | 0.85 | ≥ 0.80 | ✅ PASS |

**Summary:** Inference quality is mixed. AUC is acceptable but the BN underperforms the baseline LR model, and calibration (probability reliability) is poor.

---

## Analysis & Insights

### Why the BN Underperforms Logistic Regression

The Bayesian Network achieves -4.5% compared to Logistic Regression baseline for several reasons:

1. **Weak Parameter Learning (91% R² ≈ 0)**
   - Most features learned nothing meaningful
   - The model essentially ignores ~8 out of 11 features
   - Logistic Regression learns direct coefficients for all features

2. **Sparse DAG Structure**
   - PC Algorithm discovered only 5 conditional independence edges
   - This sparsity may oversimplify the actual data relationships
   - Logistic Regression has no such structural constraints

3. **Limited Training Data**
   - 500 total samples, 350 training samples
   - PC Algorithm struggles with small sample sizes
   - Structure learning needs more data for reliable discovery

### Why Calibration Error is High (0.258 vs 0.20 target)

- The model's predicted probabilities don't match actual default rates
- Example: When model predicts 50% default probability, actual defaults ≈ 40%
- This discrepancy means the model is **overconfident** in its predictions
- Calibration can be improved with post-hoc methods (Platt scaling), but this was not attempted

### About Cross-Seed Validation (UNPERFORMED)

Documentation initially claimed cross-seed validation with seeds [42, 123, 456, 789, 999]. **This analysis was not actually performed.** Only seed 42 was tested. A complete cross-seed validation would require:

```bash
for seed in 42 123 456 789 999; do
    python3 src/main.py --seed $seed --train_fraction 0.7
done
```

This was not executed due to time constraints. Robustness claims cannot be made without this validation.

### About Baseline Implementations

The project originally planned to compare against 5 baselines:
1. ✅ Logistic Regression (implemented)
2. ❌ Random Forest (NOT implemented)
3. ❌ Decision Tree (NOT implemented)
4. ❌ Gradient Boosting (NOT implemented)
5. ❌ Neural Network (NOT implemented)

Only Logistic Regression was coded. A complete baseline comparison would require implementing the missing 4 models. Current comparison is therefore incomplete.

---

## Data Characteristics

### Dataset Composition
- **Total Samples:** 500
- **Features:** 11 continuous financial metrics
- **Target:** Binary default (249 defaults, 251 non-defaults)
- **Preprocessing:** StandardScaler (μ=0, σ=1)
- **Train/Val/Test Split:** 350/75/75

### Feature List
All features are continuous financial indicators after standardization:
- Feature_0: Debt-to-Income Ratio
- Feature_1: Credit Utilization Rate
- Feature_2: Credit Score
- Feature_3: Payment Delinquency
- Feature_4: Available Credit
- Feature_5: Interest Rate
- Feature_6: Loan Term
- Feature_7: Employment Stability
- Feature_8: Recent Inquiries
- Feature_9: Payment History Score
- Feature_10: Income Volatility

### Synthetic Data Limitations

This is a **synthetic dataset for educational purposes.** Real-world credit default prediction would require:
- Real historical data from a financial institution
- Time-series features capturing payment behavior over time
- Macroeconomic indicators
- Much larger feature sets (50-200+ variables vs. 11 here)
- Longer observation periods

**Implications:** The AUC of 0.8107 may not transfer to real credit data. Validation on actual loan portfolios would be required before any production deployment.

---

## Code Implementation

Complete implementation in `Code/` folder:
- `src/main.py` - Main pipeline and experiment runner
- `src/pc_algorithm.py` - Structure learning (PC Algorithm)
- `src/parameter_learning.py` - Parameter estimation (OLS)
- `src/belief_propagation.py` - Inference engine
- `src/evaluation.py` - 12 KPI computation
- `src/utils.py` - Helper functions and plotting

**To reproduce results:**
```bash
cd Code/src
python3 main.py --seed 42 --train_fraction 0.7
```

Output saved to `Results/` folder with predictions, KPIs, and statistical summaries.

---

## Conclusions

### What Worked Well
1. ✅ PC Algorithm successfully learned a 5-edge DAG structure
2. ✅ AUC of 0.8107 meets the baseline target (≥0.80)
3. ✅ Model shows good interpretability and domain fit
4. ✅ Confidence intervals computed successfully for parameters
5. ✅ Reproducible results with fixed seed

### What Did Not Work
1. ❌ BN performs 4.5% worse than simple Logistic Regression
2. ❌ 91% of learned parameters have R² ≈ 0 (no learning)
3. ❌ Calibration error 0.258 exceeds target 0.20 (poor probability reliability)
4. ❌ Cross-seed validation not performed (robustness unverified)
5. ❌ Only 1 of 5 promised baselines implemented (incomplete comparison)

### Root Causes
1. **Data Insufficiency:** 500 samples too small for reliable PC Algorithm structure discovery
2. **Parameter Learning Failure:** OLS regression with weak features leads to R² ≈ 0 for most parameters
3. **Model Mismatch:** Bayesian Network structure doesn't capture patterns that Logistic Regression easily finds

### Recommendations for Improvement
1. Collect more training data (1000+ samples) for better structure learning
2. Implement feature engineering to create more predictive features
3. Use causal validation techniques to verify learned structure
4. Implement missing baselines for complete comparison
5. Apply post-hoc calibration (Platt scaling) to improve probability estimates
6. Perform cross-seed validation to test robustness
7. Consider hybrid approaches (BN for structure + LR for calibration)

---

## Files in This Submission

- `M4_G15/README.md` - This file (Main methodology and results)
- `M4_G15/Data_README.md` - Complete data documentation
- `M4_G15/Code_README.md` - Code implementation guide
- `M4_G15/Results_README.md` - Detailed results analysis
- `Code/` - Full source code implementation
- `Data/` - Synthetic dataset
- `Results/` - Experimental outputs (predictions, KPIs, statistics)
- `Corrections/` - Formal error corrections and analysis

---

## References

1. Spirtes, Glymour, Scheines (2000). *Causation, Prediction, and Search*. MIT Press.
2. Efron & Tibshirani (1986). Bootstrap methods for standard errors, confidence intervals, and other measures of statistical accuracy.
3. Hosmer & Lemeshow (2000). Applied Logistic Regression, 2nd Edition. Wiley.

---

**Report Date:** November 12, 2026  
**Group:** G15  
**Members:** Kamya Shah (Individual Submission)

---

**END OF M4 REPORT**
