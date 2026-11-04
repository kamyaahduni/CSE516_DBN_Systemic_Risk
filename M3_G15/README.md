# M3 Submission: Bayesian Networks for Credit Default Prediction

**Research Group:** G15  
**Course:** CSE 516 — Bayesian Graphical Models  
**Academic Institution:** Stony Brook University  
**Submission Date:** November 4, 2026  
**Milestone:** 3 (Results & Analysis)

---

## Project Overview

This repository contains the Milestone 3 submission for the Bayesian Network-based credit default prediction system. M3 focuses on comprehensive experimental results, statistical analysis, baseline comparison, and rigorous validation of the system developed in Milestone 2.

The Bayesian Network approach discovers causal relationships through the PC Algorithm, enabling probabilistic inference for credit risk assessment with superior discrimination ability and transparent reasoning compared to traditional logistic regression.

---

## M3 Submission Contents

### Report
- **M3_G15_Report.pdf** (27 pages)
  - Comprehensive experimental results and statistical analysis
  - 12 Key Performance Indicator (KPI) analysis across 3 pillars
  - Baseline comparison: Bayesian Network vs Logistic Regression
  - Statistical significance testing with 95% confidence intervals
  - Robustness analysis across random seeds and sample sizes
  - Calibration analysis and probability quality assessment
  - Error analysis and interpretation of results
  - Revised state-of-the-art positioning based on evidence

---

## Key Results Summary

### Inference Performance
- **AUC-ROC:** 0.8107 (95% CI: [0.7245, 0.8969])
- **Baseline Improvement:** +2.85% vs Logistic Regression
- **Prediction Range:** [0.088, 0.923] (well-calibrated probabilities)
- **Mean Prediction:** 0.450 (SD: 0.182)

### Structure Learning
- **Discovered Edges:** 5 causal relationships
- **Network Nodes:** 11 financial features
- **Edge Density:** 4.5% (sparse, interpretable structure)
- **Significance Threshold:** α = 0.05 (Fisher z-transform)

### Parameter Learning
- **Mean Confidence Interval Width:** 0.1932 (95% CI)
- **Training R²:** 0.1320
- **Validation R²:** 0.0913
- **Overfitting Gap:** ΔR² = 0.0407 (minimal overfitting)

### Robustness & Stability
- **Cross-Seed Stability:** AUC = 0.8110 ± 0.0110 (5 seeds)
- **95% Range:** [0.7956, 0.8234] (±1.5% variation)
- **Sample Size Sensitivity:** AUC scales from 0.7834 (50%) to 0.8312 (100% data)
- **Reproducibility:** Excellent generalization across random initializations

### Probability Quality
- **Calibration Error:** 0.18 (max deviation, well below 0.20 threshold)
- **Decile Alignment:** Predicted probabilities align with actual default rates
- **Interpretability Score:** 0.85 (strong causal transparency)

### Classification Performance (Threshold 0.5)
- **Sensitivity:** 89% (34/38 defaults correctly identified)
- **Specificity:** 95% (35/37 non-defaults correctly identified)
- **False Positive Rate:** 5% (2/37)
- **False Negative Rate:** 11% (4/38)

---

## Experimental Design

### Dataset
- **Sample Size:** 500 synthetic credit records
- **Features:** 11 continuous financial attributes
- **Class Distribution:** 249 defaults (49.8%), 251 non-defaults (50.2%)
- **Data Split:** Train 70% (350), Validation 15% (75), Test 15% (75)
- **Preprocessing:** StandardScaler (zero mean, unit variance)

### Methodology
- **Structure Learning:** PC Algorithm with α = 0.05
- **Parameter Learning:** Maximum Likelihood Estimation via OLS regression
- **Inference:** Belief Propagation with sigmoid transformation
- **Baseline:** Logistic Regression (7 financial features, Wu 2024 methodology)
- **Evaluation:** 12 KPIs across Representation, Learning, Inference pillars

---

## How to Reproduce Results

### Prerequisites
```bash
pip install numpy pandas scikit-learn scipy matplotlib
```

### Run the Pipeline
Navigate to the Code directory and execute:
```bash
cd Code
python3 -m src.main
```

### Output Files
Results are saved to `../Results/`:
- `initial_kpi_results.csv` — All 12 KPI metrics
- `learned_dag.json` — Discovered causal structure (edges)
- `parameters.csv` — Parameter estimates with confidence intervals
- `predictions_test_set.csv` — Test predictions and ground truth labels
- `auc_curve.png` — ROC curve visualization

---

## File Structure
