# M3: Experimental Validation & Robustness

## Overview
Comprehensive experimental validation with 200+ runs across multiple dimensions: multi-seed robustness, baseline comparison, sample size scaling, and adversarial testing.

## Contents
- **Experimental_Results/**: CSV files with detailed results
  - Multi-seed results (10 seeds × 6 methods)
  - Sample size scaling analysis (7 sizes)
  - Noise robustness testing (4 levels)
  - Missing data robustness (4 levels)
- **Reports:** Statistical analysis and findings
- **Presentations:** 21-slide results presentation

## Key Results
- Logistic Regression: AUC 0.9005 ± 0.0353
- Bayesian Network: AUC ~0.85 (interpretable)
- Paired t-tests with Bonferroni correction
- Cohen's d effect sizes
- Wilcoxon signed-rank test validation

## Data Files
- `m3_multi_seed_results.csv` - Main results
- `m3_sample_size_results.csv` - Scaling analysis
- `m3_noise_results.csv` - Noise robustness
- `m3_missing_data_results.csv` - Missing data handling
- `m3_statistics.csv` - Summary statistics

## Team
Group G15 - CSE 516 - Probabilistic Graphical Models
Ahmedabad University, November 2026
