# Code Documentation: Bayesian Network for Credit Default Prediction

## Overview

Complete source code implementation for structure learning (PC Algorithm), parameter learning (MLE), and inference (Belief Propagation) for credit default prediction using Bayesian networks.

**Language:** Python 3.8+  
**Dependencies:** scikit-learn, numpy, pandas, scipy, matplotlib, seaborn  
**Total Lines:** ~2000 (documented, production-quality)  
**License:** Educational (CSE 516)

---

## Environment Setup

### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python3 -c "import numpy, scipy, sklearn, pandas; print('All dependencies OK')"
```

---

## File Structure

```
Code/
├── README.md                    # This file
├── requirements.txt             # Python package dependencies
└── src/
    ├── main.py                  # Main pipeline entry point
    ├── pc_algorithm.py          # Structure learning (PC Algorithm)
    ├── parameter_learning.py    # Parameter estimation (OLS regression)
    ├── belief_propagation.py    # Inference engine
    ├── evaluation.py            # 12 KPI computation
    ├── utils.py                 # Utilities and helpers
    └── __init__.py              # Package initialization
```

---

## Main Pipeline: main.py

### Function Signature

```python
def run_experiment(
    seed: int = 42,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    alpha: float = 0.05,
    output_dir: str = '../Results'
) -> dict
```

### Command-Line Usage

**Basic (Primary Results):**
```bash
cd src
python3 main.py --seed 42 --train_fraction 0.7
```

**With All Options:**
```bash
python3 main.py \
    --seed 42 \
    --train_fraction 0.7 \
    --val_fraction 0.15 \
    --alpha 0.05 \
    --output_dir ../Results
```

**Cross-Seed Stability (5 seeds):**
```bash
for seed in 42 123 456 789 999; do
    python3 main.py --seed $seed --train_fraction 0.7
done
```

**Sample Size Sensitivity:**
```bash
for fraction in 0.5 0.7 0.9 1.0; do
    python3 main.py --seed 42 --train_fraction $fraction
done
```

### Output

All results saved to `../Results/`:

```
experiment_log_seed42_fraction0.7.txt          # Experiment summary
predictions_test_seed42.csv                     # Test predictions
kpi_results_seed42.csv                          # 12 KPIs
statistical_summary_seed42.txt                  # Stats summary
```

### Pipeline Steps (Automatic)

1. **Load Data:** 500 synthetic samples, 11 features, binary target
2. **Split Data:** 70% train, 15% val, 15% test (stratified)
3. **Normalize:** StandardScaler (μ=0, σ=1)
4. **Structure Learning:** PC Algorithm, α=0.05
5. **Parameter Learning:** OLS regression + CIs
6. **Inference:** Belief propagation on test set
7. **Evaluation:** 12 KPIs computation
8. **Output:** Predictions, metrics, statistical analysis

---

## Module APIs

### pc_algorithm.py

**Main Function:**
```python
def pc_algorithm(
    data: np.ndarray,
    alpha: float = 0.05,
    verbose: bool = True
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Discover conditional independence structure via PC Algorithm.
    
    Args:
        data: (n_samples, n_features) array
        alpha: Significance threshold for independence tests
        verbose: Print phase progress
    
    Returns:
        skeleton: (n_features, n_features) adjacency matrix
        edges: List of (parent, child) tuples
    """
```

**Key Methods:**
- `unconditional_independence_test()`: Fisher z-transform test
- `conditional_independence_test()`: Partial correlation test
- `identify_colliders()`: Phase 4 edge orientation
- `propagate_constraints()`: Consistency rules

**Example Usage:**
```python
from pc_algorithm import pc_algorithm
skeleton, edges = pc_algorithm(data, alpha=0.05)
print(f"Discovered {len(edges)} edges")
```

### parameter_learning.py

**Main Function:**
```python
def learn_parameters(
    data: np.ndarray,
    skeleton: np.ndarray,
    target_col: int = -1
) -> Dict[int, Dict]:
    """
    Estimate CPD parameters via OLS regression.
    
    Args:
        data: Training data
        skeleton: DAG adjacency matrix
        target_col: Column index for default label
    
    Returns:
        parameters: Dict with coefficients, CIs, R² for each node
    """
```

**Output Structure:**
```python
{
    0: {
        'coefficients': [...],           # β values
        'confidence_intervals': [...],   # 95% CIs
        'p_values': [...],              # Significance tests
        'train_r2': 0.132,              # Training R²
        'val_r2': 0.091                 # Validation R²
    },
    ...
}
```

**Example Usage:**
```python
from parameter_learning import learn_parameters
params = learn_parameters(train_data, skeleton)
print(f"Parameters: {params[0]['coefficients']}")
print(f"95% CI: {params[0]['confidence_intervals']}")
```

### belief_propagation.py

**Main Function:**
```python
def belief_propagation(
    evidence: np.ndarray,
    parameters: Dict[int, Dict],
    skeleton: np.ndarray
) -> np.ndarray:
    """
    Compute P(default|evidence) via message passing.
    
    Args:
        evidence: (n_samples, n_features) array
        parameters: Learned parameters from parameter_learning
        skeleton: DAG structure
    
    Returns:
        probabilities: (n_samples,) array of P(default|evidence)
    """
```

**Example Usage:**
```python
from belief_propagation import belief_propagation
probs = belief_propagation(test_data, params, skeleton)
print(f"Predicted probabilities: {probs[:5]}")
print(f"Mean: {probs.mean():.3f}, Std: {probs.std():.3f}")
```

### evaluation.py

**Main Function:**
```python
def compute_kpis(
    predictions: np.ndarray,
    actuals: np.ndarray,
    parameters: Dict[int, Dict],
    skeleton: np.ndarray
) -> Dict[str, Dict]:
    """
    Compute 12 KPIs across 3 pillars.
    
    Returns:
        kpis: {
            'representation': {'stability': 0.70, ...},
            'learning': {'ci_width': 0.193, ...},
            'inference': {'auc': 0.8107, ...}
        }
    """
```

**KPI Categories:**
- **Representation:** Stability, SOTA Alignment, Significance, Domain Fit
- **Learning:** CI Width, Param Significance, Train R², Val R²
- **Inference:** AUC-ROC, vs Baseline, Calibration Error, Interpretability

**Example Usage:**
```python
from evaluation import compute_kpis, compute_auc_ci, \
    compute_calibration, paired_t_test

# All KPIs
kpis = compute_kpis(predictions, actuals, params, skeleton)

# Individual metrics
auc = compute_auc_ci(predictions, actuals, n_bootstrap=1000)
calibration = compute_calibration(predictions, actuals)
p_value = paired_t_test(predictions_bn, predictions_lr)

print(f"AUC: {auc['auc']:.4f}, 95% CI: {auc['ci']}")
print(f"Calibration Error: {calibration['max_error']:.4f}")
print(f"BN vs LR p-value: {p_value:.4f}")
```

### utils.py

**Key Functions:**

```python
# Data Loading
def load_data(dataset_path: str = '../Data/synthetic_data.csv') -> Tuple[np.ndarray, np.ndarray]:
    """Load 500 samples × 11 features + binary target"""

# Normalization
def normalize_data(X: np.ndarray) -> Tuple[np.ndarray, Any]:
    """StandardScaler normalization (μ=0, σ=1)"""

# Train/Val/Test Split
def split_data(
    X: np.ndarray, y: np.ndarray,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42
) -> Dict[str, np.ndarray]:
    """Stratified split with reproducibility"""

# Plotting
def plot_auc_roc(predictions: np.ndarray, actuals: np.ndarray):
    """Plot ROC curve with AUC annotation"""

def plot_calibration(predictions: np.ndarray, actuals: np.ndarray, n_bins: int = 10):
    """Decile calibration plot"""

def plot_feature_importance(parameters: Dict[int, Dict]):
    """Visualize parameter magnitudes"""
```

---

## Random Seeds & Reproducibility

### Fixed Seeds for Reproducibility

```python
np.random.seed(seed)  # NumPy operations
random.seed(seed)     # Python random
```

**Primary Results:** seed = 42

**Cross-Seed Validation:** seeds = [42, 123, 456, 789, 999]

### Data Reproducibility

Seed controls:
1. Data generation
2. Train/val/test split
3. PC Algorithm tie-breaking
4. Any stochastic operations

**Result:** Same seed → identical results

---

## Experiment Configuration

### Configuration File (Optional)

Create `config.yaml` for reproducible experiments:

```yaml
experiment:
  name: "M4_G15_Primary"
  seed: 42
  train_fraction: 0.7
  val_fraction: 0.15

structure_learning:
  method: "pc_algorithm"
  alpha: 0.05
  verbose: true

parameter_learning:
  method: "ols_regression"
  ci_level: 0.95

inference:
  method: "belief_propagation"

evaluation:
  bootstrap_samples: 1000
  n_bins_calibration: 10
  threshold_classification: 0.5
```

**Usage:**
```python
import yaml
with open('config.yaml') as f:
    config = yaml.safe_load(f)
run_experiment(**config['experiment'])
```

---

## Statistical Methods

### Confidence Intervals: Fisher Information

```python
def compute_parameter_ci(
    X: np.ndarray,
    y: np.ndarray,
    confidence_level: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 95% CI from Fisher information matrix.
    Var(β̂) = σ̂² (Xᵀ X)⁻¹
    """
```

### Bootstrap: AUC CI

```python
def bootstrap_auc_ci(
    predictions: np.ndarray,
    actuals: np.ndarray,
    n_resamples: int = 1000,
    ci_level: float = 0.95
) -> Dict:
    """
    Compute 95% AUC CI via 1000 bootstrap resamples.
    Method: DeLong
    """
```

### Significance Testing: Paired t-test

```python
def paired_t_test(
    predictions_bn: np.ndarray,
    predictions_lr: np.ndarray,
    alternative: str = 'two-sided'
) -> float:
    """Paired t-test: BN vs Logistic Regression"""
```

---

## Output File Formats

### predictions_test_seed42.csv

```
prediction,actual,probability_default
0.823,1,0.8234
0.156,0,0.1562
...
```

### kpi_results_seed42.csv

```
pillar,kpi,value,target,status
representation,stability,0.70,0.65,PASS
representation,sota_alignment,0.75,0.70,PASS
...
```

### statistical_summary_seed42.txt

```
=== STATISTICAL ANALYSIS SUMMARY ===
AUC-ROC: 0.8107
95% CI: [0.7245, 0.8969]
Bootstrap N: 1000
Cross-seed mean: 0.8110 ± 0.0110
...
```

---

## Performance Optimization

### Memory Usage

Typical memory footprint:
- Data (500 × 11): ~44 KB
- Structure learning: ~1 MB
- Parameter storage: ~100 KB
- Inference: ~50 KB

**Total:** < 2 MB for all operations

### Computational Time

Typical runtime:
- Data loading: < 100 ms
- Structure learning (PC Algorithm): 500-1000 ms
- Parameter learning (OLS): 50-100 ms
- Inference (1000 samples): 10-20 ms
- Evaluation (12 KPIs): 200-500 ms

**Total:** ~2-3 seconds per experiment

---

## Troubleshooting

### Common Issues

**Issue:** `ImportError: No module named 'sklearn'`
```bash
pip install scikit-learn
```

**Issue:** `MemoryError` during bootstrap (1000 resamples)
```python
# Reduce bootstrap samples
bootstrap_samples = 500  # Default: 1000
```

**Issue:** Singular matrix in OLS regression
```python
# Add regularization (not in primary pipeline, but available)
from sklearn.linear_model import Ridge
Ridge(alpha=0.1).fit(X, y)
```

**Issue:** Different results on different runs
```python
# Check seed is set before running
np.random.seed(42)
run_experiment(seed=42)
```

---

## Logging & Debugging

### Enable Verbose Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from main import run_experiment
results = run_experiment(verbose=True)
```

### Debug PC Algorithm

```python
from pc_algorithm import pc_algorithm
skeleton, edges = pc_algorithm(
    data,
    alpha=0.05,
    verbose=True  # Prints phase progress
)
```

### Validate Parameter Estimates

```python
from parameter_learning import learn_parameters
params = learn_parameters(train_data, skeleton, validate=True)
# Outputs: R², residual plots, assumption checks
```

---

## Testing

### Unit Tests

```bash
cd Code
python3 -m pytest tests/
```

### Integration Test

```bash
# Verify full pipeline
python3 src/main.py --seed 42 --train_fraction 0.7

# Check outputs exist
ls -la ../Results/
```

---

## References for Implementation

1. **PC Algorithm:**
   - Spirtes, Glymour, Scheines (2000) - Algorithm 5.4 in text
   - Original paper: Algorithm PC with FCI improvements

2. **OLS Regression with CI:**
   - Fisher information computation: standard statistical theory
   - Implementation: scipy.stats t-distribution for CI bounds

3. **Belief Propagation:**
   - Message passing on DAGs: standard inference procedure
   - Probability calibration via sigmoid: logistic model

4. **Bootstrap & Significance Testing:**
   - Efron & Tibshirani (1986) for bootstrap methodology
   - DeLong method for AUC CI computation

---

## Contact

For questions about code implementation:
- Check docstrings: `python3 -c "import main; help(main.run_experiment)"`
- Review inline comments for algorithm details
- See main README.md for overall methodology

---

**END OF CODE DOCUMENTATION**