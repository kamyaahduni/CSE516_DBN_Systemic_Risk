# Data Documentation

## Dataset Overview

**Name:** Synthetic Credit Default Dataset  
**Samples:** 500  
**Features:** 11 continuous financial attributes  
**Target:** Binary default status (0 = non-default, 1 = default)  
**Class Distribution:** Balanced (249 defaults 49.8%, 251 non-defaults 50.2%)  
**Missing Values:** None  
**Generation Method:** Synthetic (randomly generated for educational purposes)  
**Seed:** 42 (for reproducibility)

---

## Data Composition

| Property | Value |
|----------|-------|
| **Total Records** | 500 |
| **Default Cases** | 249 (49.8%) |
| **Non-Default Cases** | 251 (50.2%) |
| **Feature Dimensionality** | 11 |
| **Data Type** | Continuous (financial metrics) |
| **Preprocessing** | StandardScaler normalization (μ=0, σ=1) |

---

## Feature Descriptions

All features are continuous financial metrics, standardized to zero mean and unit variance.

| Feature | Index | Type | Original Range | Interpretation |
|---------|-------|------|---|---|
| Feature_0 | X₀ | Continuous | [0.5, 2.5] | Debt-to-Income Ratio |
| Feature_1 | X₁ | Continuous | [0.1, 0.9] | Credit Utilization Rate |
| Feature_2 | X₂ | Continuous | [10, 800] | Credit Score (scaled) |
| Feature_3 | X₃ | Continuous | [0.0, 5.0] | Payment Delinquency (months) |
| Feature_4 | X₄ | Continuous | [1000, 100000] | Available Credit (normalized) |
| Feature_5 | X₅ | Continuous | [0.02, 0.25] | Interest Rate |
| Feature_6 | X₆ | Continuous | [6, 360] | Loan Term (months, normalized) |
| Feature_7 | X₇ | Continuous | [0.0, 1.0] | Employment Stability Index |
| Feature_8 | X₈ | Continuous | [0.1, 5.0] | Recent Inquiries (count) |
| Feature_9 | X₉ | Continuous | [0.0, 10.0] | Payment History Score |
| Feature_10 | X₁₀ | Continuous | [0.5, 2.0] | Income Volatility Index |

**Target:** Default Status (D)
- 1 = Default occurred
- 0 = No default (fully repaid or current)

---

## Data Limitations

**IMPORTANT:** This is **synthetic data** generated for educational purposes.

**Key Limitations:**
1. **Non-realistic distributions:** Real credit data exhibits different statistical properties
2. **Balanced classes:** Real default rates are typically 2-5% (not 50%)
3. **Limited scope:** Only 11 features (real models use 50-200+ variables)
4. **No temporal dynamics:** Single-period snapshot; real credit is multi-period
5. **No hidden confounders:** Assumes data generation process is complete

**Implication:** Results obtained on this dataset may NOT transfer to real-world credit data.

---

## Data Partitioning

**Method:** Stratified random split (preserves class distribution)

| Partition | Records | Defaults | Non-Defaults | Purpose |
|-----------|---------|----------|--|--|
| **Training** | 350 (70%) | 174 | 176 | Structure + parameter learning |
| **Validation** | 75 (15%) | 37 | 38 | Hyperparameter tuning (unused in M4) |
| **Test** | 75 (15%) | 38 | 37 | Unbiased performance evaluation |

**Reproducibility:** Stratified split with seed=42 ensures identical partitions across runs

---

## Data Preprocessing

### Normalization

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_normalized = scaler.fit_transform(X_raw)
```

**Effect:**
- Mean: μ = 0 (shifted to origin)
- Std Dev: σ = 1 (scaled to unit variance)
- Preserves distribution shape (symmetric features remain symmetric)
- Enables fair comparison of regression coefficients

### Missing Value Handling

**Status:** No missing values in dataset (0 NaN values)

If real-world data contained missing values:
- Continuous features: Mean/median imputation
- Categorical features: Mode imputation  
- Deletion: Only if < 5% missing

### Outlier Detection

**Q-Q Plot:** Features approximately normal after standardization  
**Outlier Threshold:** |z-score| > 3 (identifies < 1% samples)  
**Treatment:** Retained for robustness validation

---

## Loading Data

```python
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('synthetic_data.csv')

# Separate features and target
X = df.iloc[:, :-1].values       # Features (500 × 11)
y = df.iloc[:, -1].values        # Target (500,)

# Verify shape
print(f"Features shape: {X.shape}")  # (500, 11)
print(f"Target shape: {y.shape}")    # (500,)

# Check class distribution
print(f"Default rate: {y.mean():.1%}")  # ~49.8%
```

---

## Data Quality Assurance

### Validation Checklist

-  No missing values (NaN/None)
-  Consistent data types (all numeric)
-  Balanced class distribution (49.8% / 50.2%)
-  No duplicate records (all unique)
-  Reasonable feature ranges
-  Stratified split maintains class balance
-  Deterministic reproducibility via seed=42

---

## File Format

**Format:** CSV (comma-separated values)  
**Encoding:** UTF-8  
**Location:** To be determined (not in repository)  
**File Size:** ~44 KB uncompressed  

---

## Citation

If using this dataset in publications or further research:

```bibtex
@misc{cse516_credit_data,
  title={Synthetic Credit Default Dataset},
  author={Shah, Kamya},
  institution={Ahmedabad University},
  year={2026},
  note={Educational dataset for CSE 516 - Bayesian Graphical Models}
}
```

---

**Document Version:** M4 Final  
**Last Updated:** November 12, 2026
