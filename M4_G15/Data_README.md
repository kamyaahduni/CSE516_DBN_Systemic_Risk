# Data Documentation

## Dataset Overview

**Name:** Synthetic Credit Dataset  
**Samples:** 500  
**Features:** 11 continuous financial attributes  
**Target:** Binary default status (0 = non-default, 1 = default)  
**Class Distribution:** Balanced (249 defaults 49.8%, 251 non-defaults 50.2%)  
**Missing Values:** None  

---

## Data Composition

| Property | Value |
|----------|-------|
| **Total Records** | 500 |
| **Default Cases** | 249 (49.8%) |
| **Non-Default Cases** | 251 (50.2%) |
| **Feature Dimensionality** | 11 |
| **Data Type** | Continuous (financial ratios) |
| **Preprocessing** | StandardScaler normalization (μ=0, σ=1) |

---

## Feature Descriptions

All features are **continuous financial metrics**, standardized to zero mean and unit variance.

| Feature | Index | Type | Range (Original) | Interpretation |
|---------|-------|------|------------------|-----------------|
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
- 1 = Default occurred within observation period
- 0 = No default (fully repaid or current)

---

## Data Statistics

### Summary Statistics

```
             Mean    Std    Min    25%    50%    75%    Max
Feature_0   1.50   0.50   0.50   1.15   1.50   1.85   2.50
Feature_1   0.50   0.20   0.10   0.35   0.50   0.65   0.90
Feature_2 405.00  150.00  10.00 285.00 405.00 525.00 800.00
...
Default     0.498  0.500   0.00   0.00   0.00   1.00   1.00
```

### Correlation Structure

Features exhibit meaningful correlations (discovered by PC Algorithm):
- Feature_0 ↔ Feature_2: Debt-to-income vs credit score (expected negative)
- Feature_1 ↔ Feature_4: Credit utilization vs available credit (expected negative)
- Feature_3 ↔ Feature_9: Delinquency vs payment history (expected negative)

PC Algorithm discovers 5 conditional independence edges capturing these relationships.

---

## Data Partitioning

**Method:** Stratified random split preserving class distribution

| Partition | Records | Defaults | Non-Defaults | Use Case |
|-----------|---------|----------|--------------|----------|
| **Training** | 350 (70%) | 174 | 176 | Structure + Parameter Learning |
| **Validation** | 75 (15%) | 37 | 38 | Hyperparameter tuning (not used in M4) |
| **Test** | 75 (15%) | 38 | 37 | Unbiased Performance Evaluation |

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

**No missing values in this dataset** - All 500 samples complete across all 11 features

If real-world data contains missing values:
- Imputation: Mean/median for continuous features
- Multiple imputation: For robustness with MCAR/MAR assumptions
- Deletion: Only if < 5% missing (not recommended)

### Outlier Detection

**Q-Q Plot:** Features approximately normal after standardization  
**Outlier Threshold:** |z-score| > 3 (< 1% samples)  
**Treatment:** Retained for robustness validation (M4 robustness shows stable performance)

---

## Data Reproducibility

### Seed Specification

```python
np.random.seed(42)  # Primary results
# or
np.random.seed(seed)  # Cross-seed validation
```

**Deterministic Data Generation:**
1. Random number generator initialized with seed
2. Train/val/test split produces identical samples
3. PC Algorithm operates deterministically on feature data
4. Results fully reproducible across computing environments

### Synthetic Nature & Limitations

**Note:** This dataset is synthetic for educational purposes.

**Limitations:**
1. Real credit data exhibits different distributions
2. Missing important time-series features
3. No temporal dynamics (multi-period behavior)
4. Simplified feature space (real models use 50-200+ variables)
5. Balanced classes (real default rates ~2-5%)

**Implications:**
- AUC 0.8107 may not directly transfer to real data
- Parameter estimates (modest R²) reflect synthetic simplicity
- Real-world validation required before production deployment

---

## Data Access & Licensing

**Format:** CSV (comma-separated values)  
**Size:** ~44 KB uncompressed  
**Encoding:** UTF-8  
**License:** Educational (CSE 516)  

**File Location:**
```
M4_G15/Data/synthetic_data.csv
```

**Load Instructions:**
```python
import pandas as pd
df = pd.read_csv('../Data/synthetic_data.csv')
X = df.iloc[:, :-1].values       # Features (500 × 11)
y = df.iloc[:, -1].values        # Target (500,)
```

---

## Feature Engineering Notes

### Current Features

11 features represent key credit risk factors:
- **Leverage:** Debt-to-income ratio (Feature_0)
- **Credit Utilization:** Current vs available (Feature_1)
- **Credit History:** Score and payment history (Feature_2, Feature_9)
- **Default Risk:** Payment delinquency (Feature_3)
- **Capacity:** Available credit and income stability (Feature_4, Feature_7)
- **Cost:** Interest rate (Feature_5)
- **Commitment:** Loan term and inquiries (Feature_6, Feature_8)
- **Stability:** Income and payment volatility (Feature_10)

### Potential Enhancements (Real-World Data)

1. **Temporal Features:**
   - Multi-period payment history
   - Trend indicators (improving vs deteriorating)
   - Seasonal patterns

2. **Economic Indicators:**
   - Macroeconomic variables (unemployment, GDP growth)
   - Industry-specific factors
   - Credit market conditions

3. **Behavioral Features:**
   - Employment tenure and changes
   - Previous default history
   - Account management patterns

4. **Network Features:**
   - Portfolio concentration
   - Correlated defaults
   - Systemic risk indicators

---

## Quality Assurance

### Data Validation Checklist

- ✅ No missing values (NaN/None)
- ✅ Consistent data types (all numeric)
- ✅ Balanced class distribution (49.8% / 50.2%)
- ✅ No duplicate records (all unique combinations)
- ✅ Reasonable feature ranges
- ✅ Stratified split maintains class balance
- ✅ Deterministic reproducibility via seed

### Validation Script

```python
def validate_dataset(df):
    # Check dimensions
    assert df.shape == (500, 12), "Expected 500 samples, 12 columns"
    
    # Check for missing values
    assert df.isnull().sum().sum() == 0, "Contains missing values"
    
    # Check class balance
    assert abs(df.iloc[:, -1].value_counts()[1]/500 - 0.498) < 0.01, "Unbalanced classes"
    
    # Check data types
    assert all(df.dtypes == 'float64'), "Non-numeric data detected"
    
    print("✓ Dataset validation passed")
```

---

## Data Citation

If using this dataset in publications or further research:

```bibtex
@misc{cse516_credit_data,
  title={Synthetic Credit Default Dataset},
  author={CSE 516 Course},
  institution={Stony Brook University},
  year={2026},
  note={Educational dataset for Bayesian Graphical Models}
}
```

---

## Contact & Support

For data-related questions:
1. Review this README and feature descriptions
2. Check Code/README for loading/preprocessing instructions
3. See main README.md for experimental context
4. Refer to M4_G15_Report.pdf for methodology details

---

**END OF DATA DOCUMENTATION**