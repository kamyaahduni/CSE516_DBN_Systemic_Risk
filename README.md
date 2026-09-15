# CSE 516 M1 Submission: Credit Default Prediction using Bayesian Networks

##  Project Overview

**Course:** CSE 516 - Probabilistic Graphical Models (Monsoon 2026)  
**Project Category:** Applied AI  
**Project Title:** Credit Default Prediction using Causal Bayesian Networks  
**Group Number:** 15
**Institution:** Ahmedabad University 
**Submission Date:** September 15, 2026  

---

##  Team Members

| Name | Roll Number | Email | Role | 
|------|-------------|-------|------|--------------|
| Kamya  | AU2400014 | kamya.s1@ahduni.edu.in |
---

##  Problem Statement

### What We're Solving

Credit default prediction is a critical challenge in financial services. Traditional credit scoring models (logistic regression, random forests) can predict *whether* a borrower defaults, but they cannot explain *why*—they miss causal pathways and violate regulatory requirements for explainability.

**Our approach:** Use **Causal Bayesian Networks** with **PC Algorithm** to discover the actual causal structure driving default risk.

### Why It Matters

1. **Regulatory Requirement:** Fair Lending laws require explainable credit decisions
2. **Risk Management:** Understanding causality enables targeted interventions (e.g., "increase income" vs. "increase credit score requirement")
3. **Fairness & Ethics:** Causal models prevent discriminatory decisions based on spurious correlations
4. **Publication Potential:** Bridges machine learning + finance with principled causal inference

### Key Assumptions

- **Markov Property:** Conditional independencies in our DAG reflect data independencies
- **Causal Sufficiency:** No hidden confounders affecting multiple variables
- **No Reverse Causality:** Causality flows from borrower attributes → default
- **Linear Relationships:** Gaussian Bayesian Network (relationships are approximately linear)
- **Stationarity:** Causal structure is stable over time

---

##  PGM Framework Mapping

### 1. REPRESENTATION: Bayesian Network (Directed Acyclic Graph)

**Why Bayesian Networks for credit?**
- Explicitly model causal structure (not just correlations)
- Enable interpretable decision-making
- Support both forward reasoning (prediction) and backward reasoning (diagnosis)
- Bridge domain knowledge with data-driven learning

**Our Model Structure:**

**12 Random Variables (Nodes):**

| Variable | Type | Description | Range/Values | Domain Meaning |
|----------|------|-------------|--------------|---|
| **Age** | Continuous | Borrower's age | 18–75 years | Life stage affects employment & debt patterns |
| **Employment_Status** | Categorical | Type of employment | {Employed, Self-employed, Unemployed, Retired} | Income stability indicator |
| **Years_Employed** | Continuous | Duration in current job | 0–50 years | Employment stability |
| **Annual_Income** | Continuous | Gross annual income | $20K–$500K | Core ability to repay |
| **Existing_Debt** | Continuous | Total outstanding debt | $0–$500K | Existing financial burden |
| **Debt_to_Income_Ratio** | Continuous | Debt / Income | 0–10 | Financial health metric |
| **Credit_Score** | Continuous | Credit score | 300–850 | Historical payment behavior |
| **Credit_History_Length** | Continuous | Years since credit opened | 0–70 years | Experience with credit |
| **Loan_Amount** | Continuous | Loan amount applied for | $1K–$1M | Size of financial commitment |
| **Loan_Purpose** | Categorical | Why borrowing? | {Home, Auto, Personal, Education, Business} | Loan type risk profile |
| **Payment_Ability** | Continuous | Derived: capacity to repay | 0–1 | Synthetic metric = f(Income, Debt, Years_Employed) |
| **Default** | Binary (Target) | Loan default status | {0 = Non-Default, 1 = Default} | What we predict |

**~16 Causal Edges (Relationships):**

```
Age → Existing_Debt              [Age influences accumulated debt]
Age → Credit_History_Length      [Older → longer credit history]
Age → Employment_Status          [Age affects employment type]

Employment_Status → Annual_Income        [Job type determines income level]
Employment_Status → Years_Employed      [Employment type affects job tenure]

Annual_Income → Existing_Debt           [Higher income → can accumulate more debt]
Annual_Income → Loan_Amount             [Income determines loan capacity]
Annual_Income → Payment_Ability         [Income is key to repayment capacity]

Years_Employed → Payment_Ability        [Employment stability increases repayment ability]
Years_Employed → Credit_Score           [Stable employment → better credit score]

Existing_Debt → Debt_to_Income_Ratio    [Debt is numerator in D/I ratio]
Existing_Debt → Payment_Ability         [More debt reduces ability to repay new loan]

Credit_Score → Credit_History_Length    [Good score requires long history]
Credit_Score → Payment_Ability          [High score indicates repayment ability]

Loan_Amount → Debt_to_Income_Ratio      [Loan adds to numerator]

[Final Impact on Default:]
Payment_Ability → Default               [Lower ability → higher default risk]
Debt_to_Income_Ratio → Default          [Higher ratio → higher default risk]
Credit_Score → Default                  [Lower score → higher default risk]
Credit_History_Length → Default         [Longer history → lower default risk]
```

**Conditional Independence Assumptions:**

From this DAG structure, we derive key conditional independencies:
- Age ⊥ Annual_Income (no direct edge; income flows through Employment_Status)
- Age ⊥ Loan_Purpose (no causal link)
- Employment_Status ⊥ Credit_History_Length | Age (Age is confounder)
- Loan_Amount ⊥ Credit_Score (independent causal pathways)

---

### 2. LEARNING: Structure + Parameter Learning

#### A) Structure Learning: PC Algorithm

**What:** Discover the causal DAG from observational data  
**Method:** PC (Peter-Clark) Constraint-Based Algorithm  
**Why:** Principled, statistically rigorous, reproducible

**How PC Algorithm Works:**

```
Phase 1: Start with Complete Graph
  ├─ Begin with all 12 nodes connected (undirected)

Phase 2: Remove Independent Edges (Test Pairwise Independence)
  ├─ For each pair of variables (X, Y):
  │  ├─ Compute correlation: r(X, Y)
  │  ├─ Perform significance test: H₀: ρ = 0 (vs. HA: ρ ≠ 0)
  │  ├─ p-value = P(|r| > observed | no correlation)
  │  └─ If p-value > α (0.05): REMOVE edge X—Y (they're independent)

Phase 3: Test Conditional Independence (Condition on 1 variable)
  ├─ For each remaining pair (X, Y):
  │  ├─ For each other variable Z:
  │  │  ├─ Compute partial correlation: r(X, Y | Z)
  │  │  ├─ Test: H₀: ρ(X,Y|Z) = 0
  │  │  └─ If p > α: REMOVE edge (X and Y are cond. indep. given Z)

Phase 4: Repeat (Higher-Order Conditioning)
  ├─ Condition on sets of 2, 3, ... variables
  ├─ Keep removing edges until no more can be removed

Phase 5: Orient Edges (Determine Causality Direction)
  ├─ Rule R1: If A—C—B and A ⊥ B, then A → C ← B (v-structure)
  ├─ Rule R2-4: Propagate orientations to avoid cycles
  └─ Output: Directed Acyclic Graph (DAG)
```

**Key Parameters:**
- **Significance level (α):** 0.05 (p-value threshold for independence test)
- **Multiple testing correction:** Benjamini-Hochberg FDR control
- **Partial correlation method:** Gaussian (assumes linear relationships)

**Output:** Learned DAG with ~16 edges (should match domain theory closely)

**Validation:**
- Edge stability: Do edges appear in >80% of bootstrap samples?
- SOTA alignment: Do discovered edges match Senyk et al. structure?
- Interpretability: Do edges make economic sense?

#### B) Parameter Learning: Maximum Likelihood Estimation (MLE)

**What:** Estimate strength of each causal relationship  
**Method:** Linear regression for each variable given its parents

**For each node with parents, fit regression:**

```
Example: Default = β₀ + β₁·Payment_Ability + β₂·Credit_Score + β₃·Debt_to_Income + ε

Where:
  β₀ = intercept
  β₁, β₂, β₃ = causal effect strengths (weights)
  ε = residual error (unexplained variance)
```

**Implementation Details:**

- **Continuous Variables:** Gaussian CPD with mean = β₀ + Σβᵢ·parentᵢ, variance = σ²
- **Binary Variables:** Logistic CPD (sigmoid function for probability)
- **Categorical Variables:** Multinomial CPD (discrete probabilities)

**Optimization:** Ordinary Least Squares (OLS) minimizes sum of squared residuals

**Validation Metrics:**
- R² (R-squared): % of variance explained (target: >0.40)
- p-values: Are β coefficients statistically significant? (target: >85% significant)
- Confidence intervals: 95% CI width < 0.2 (narrow = precise estimates)
- Cross-validation: Does model generalize to test data?

---

### 3. INFERENCE: Answering Questions About Default

#### Query Type 1: Credit Scoring (Forward Prediction)

**Question:** Given this borrower's profile, what's the probability they'll default?

**Mathematically:** P(Default = Yes | Age=35, Income=$60K, CreditScore=720, ...)

**How We Answer:** 
- Feed observed values to Belief Propagation
- Algorithm computes marginal probability of Default
- Output: Probability in range [0, 1]
- Decision: If P(Default) > 0.5 → Reject loan, else Approve

**Application:** Credit decision (Accept/Reject/Special Review)

---

#### Query Type 2: Causal Effect Estimation (Intervention Analysis)

**Question:** If this borrower's income increases by $10K, how much does default risk decrease?

**Mathematically:** ∂P(Default) / ∂Income = ?

**How We Answer:**
- Use learned parameters (β coefficients from MLE)
- Compute marginal effect: How much does outcome change per unit change in predictor?
- Report: Causal effect size with 95% confidence interval

**Example Result:** "A $10K income increase reduces default probability by 2.3% (95% CI: 1.8–2.8%)"

**Application:** Policy analysis, intervention targeting (e.g., "Which factors should we target to reduce default?")

---

#### Query Type 3: Diagnosis (Backward Reasoning)

**Question:** This borrower defaulted. What attributes most likely contributed?

**Mathematically:** P(Low_Income | Default=Yes)? P(High_Debt | Default=Yes)?

**How We Answer:**
- Observe: Default=Yes
- Compute: Posterior probability of each variable given default
- Identify: Which attributes are most predictive of observed default?

**Application:** Risk profiling, understanding default drivers

---

#### Query Type 4: Counterfactuals (What-If Scenarios)

**Question:** "If this borrower's employment changed from Unemployed to Employed, would they default?"

**Mathematically:** P(Default | Employment=Employed, all else unchanged)

**How We Answer:**
- Start with observed borrower profile
- Intervene: Set Employment=Employed (hypothetically)
- Compute: What would Default probability be?
- Compare: Hypothetical vs. actual

**Application:** Personnel decisions, policy simulation, fairness auditing

---

##  State-of-the-Art Position

### Our SOTA Anchor: Senyk et al. (2024)

**Citation:** A. P. Senyk, O. S. Manziy, P. E. Ohloblin, and V. V. Petrovych. "Application of the Bayesian Approach to Modeling Credit Risks." *Mathematical Modeling and Computing*, vol. 11, no. 4, pp. 1–15, 2024.

**What They Did:**
- Applied Bayesian Networks to credit risk modeling
- Used domain expert guidance to structure the DAG
- Estimated parameters via Bayesian inference
- Evaluated on Eastern European lending dataset
- Reported 75–85% accuracy on default prediction

**Key Findings:**
- Bayesian Networks capture interpretable causal pathways
- Employment → Income → Creditworthiness chain identified
- Outperformed logistic regression on explainability

**Limitations We Address:**

| Limitation | Senyk et al. | Our Approach |
|---|---|---|
| **Structure Learning** | Heuristic (expert pruning + ad-hoc tests) | Rigorous PC Algorithm with formal hypothesis testing |
| **Statistical Rigor** | No formal testing, p-values, confidence intervals | Formal independence tests, 95% CIs, significance levels |
| **Robustness** | No stability analysis or sensitivity testing | Jackknife stability (>80% threshold), bootstrap, sensitivity to α |
| **Baselines** | Only logistic regression | 5 baselines: Logistic Reg, RF, DT, GB, Neural Network |
| **Causal Quantification** | Qualitative insights only | Quantified causal effects with confidence intervals |
| **Reproducibility** | Method unclear, hard to replicate | Fully documented, code released |

### How We Extend SOTA: 5 Key Contributions

**Contribution 1: Rigorous Structure Learning**
- Replace heuristic pruning with PC Algorithm
- Principled statistical independence testing
- Reproducible, auditable causal discovery
- Formal hypothesis testing on every edge

**Contribution 2: Formal Statistical Validation**
- p-values and significance levels on all edges
- 95% confidence intervals on causal effects
- Multiple testing correction (Benjamini-Hochberg FDR)
- Statistical power analysis

**Contribution 3: Comprehensive Robustness Analysis**
- Jackknife stability: Leave-one-year-out, check edge stability
- Bootstrap confidence intervals on parameters
- Sensitivity analysis: How does structure change as α varies?
- Subgroup analysis: Does structure hold across demographics?

**Contribution 4: Multiple Baseline Comparisons**
- 5 strong baselines (not just logistic regression)
- Compare on predictive accuracy (AUC, F1) AND interpretability
- Demonstrate that rigorous BN achieves competitive accuracy with superior explainability
- Fair comparison: Same dataset, tuned hyperparameters

**Contribution 5: Causal Effect Quantification**
- Standardized causal effects (β coefficients) with CIs
- Enable policy analysis: "How much does $10K income increase reduce default?"
- Counterfactual predictions: "If employment status changed..."
- Decision support: "Which interventions are most effective?"

### Publication Strategy (After M4)

**Target Venues (by likelihood):**
1. **Primary:** Research in International Business and Finance (RIBAF) — same journal as Senyk et al.
2. **Secondary:** CLeaR Conference (Causal Learning and Reasoning)
3. **Ambitious:** ICML, ICLR, NeurIPS (if results are strong enough)
4. **Safety:** FMA Annual Meeting, EFA Annual Meeting

**Expected Timeline:** A* publication by Summer/Fall 2027

---

##  Baselines & Key Performance Indicators

### Baseline Methods (5 Strong Comparisons)

We compare our Bayesian Network against state-of-the-art methods:

| Baseline | Type | Library | Why Selected | Expected Performance |
|----------|------|---------|--------------|-----|
| **Logistic Regression** | Linear classifier | statsmodels | Industry standard for credit scoring | Baseline: ~70% AUC |
| **Random Forest** | Ensemble, tree-based | scikit-learn | Modern ML, feature importance | Strong: ~75–78% AUC |
| **Decision Tree** | Rule-based | scikit-learn | Interpretable alternative | Moderate: ~68–72% AUC |
| **Gradient Boosting** | Ensemble, sequential | XGBoost | State-of-the-art ML performance | Very Strong: ~78–80% AUC |
| **Neural Network** | Deep learning | TensorFlow | High-capacity black-box | Strong: ~76–79% AUC |

**Baseline Strategy:**
- Train all on same dataset (70% train, 15% val, 15% test)
- Tune hyperparameters on validation set
- Evaluate on held-out test set
- Compare: BN AUC vs. Best Baseline AUC
- Target: BN within 2% of best (competitive accuracy with superior interpretability)

---

### Key Performance Indicators (3 Dimensions)

#### DIMENSION 1: REPRESENTATION (Is the learned DAG correct?)

| KPI | Metric | Target | Why It Matters |
|-----|--------|--------|---|
| **Structure Stability** | % edges stable in ≥80% of jackknife samples | >80% | Robust edges appear consistently across data samples |
| **SOTA Alignment** | % discovered edges matching Senyk et al. | >70% | Should recover known causal pathways |
| **Edge Significance** | % edges with p-value < 0.05 | >90% | Only include statistically significant edges |
| **Domain Alignment** | Expert review: Do edges make economic sense? | 100% | Causal structure must align with lending theory |

---

#### DIMENSION 2: LEARNING (Are parameters learned correctly?)

| KPI | Metric | Target | Why It Matters |
|-----|--------|--------|---|
| **Parameter Precision** | Average width of 95% CI for β coefficients | <0.2 | Narrow CIs = precise, confident estimates |
| **Parameter Significance** | % parameters with p-value < 0.05 | >85% | Most causal effects should be statistically significant |
| **Model Fit (Training)** | R² on training data | >0.40 | Learned CPDs explain ≥40% of default variance |
| **Generalization** | R² on test data (gap < 0.02 from training) | >0.38 | No overfitting; generalizes to unseen data |

---

#### DIMENSION 3: INFERENCE (Do predictions work well?)

| KPI | Metric | Target | Why It Matters |
|-----|--------|--------|---|
| **Predictive Accuracy** | AUC-ROC on test set | >0.75 | Discriminates default vs. non-default well |
| **vs. Baselines** | BN AUC - Best Baseline AUC | ≥ -0.02 (within 2%) | Competitive with modern ML (may not be best, but close) |
| **Calibration** | Brier score on test set | <0.20 | Predicted probabilities match observed frequencies |
| **Interpretability** | SHAP feature importance vs. causal edges alignment | >80% | Explanation mechanism aligns with learned causality |

---

##  Dataset

### Dataset Selection: Kaggle Loan Default

**Source:** https://www.kaggle.com/datasets/ (search "loan default" or "credit default")

**Alternative Options:**
- UCI Credit Approval Dataset: https://archive.ics.uci.edu/ml/
- German Credit Dataset: https://archive.ics.uci.edu/ml/

### Dataset Overview

| Aspect | Details |
|--------|---------|
| **Records** | 100,000–500,000 individual loans |
| **Time Period** | Historical data spanning 5–10 years |
| **Geographic Scope** | Single country/market (e.g., USA) |
| **Features** | 15–20 variables (demographics, financial, loan details) |
| **Target Variable** | Loan default status (binary: 0 = Non-Default, 1 = Default) |
| **Class Distribution** | ~85% non-default, ~15% default (imbalanced) |
| **Feature Types** | Mix of continuous and categorical variables |
| **Data Quality** | Minimal missing values (<2%), realistic class imbalance |
| **File Format** | CSV |
| **File Size** | ~50–100 MB |

### Data Preprocessing Pipeline

**Step 1: Data Loading & Exploration**
```python
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('loan_default.csv')

# Exploratory Data Analysis (EDA)
print(df.head())           # First few rows
print(df.info())           # Data types, missing values
print(df.describe())       # Summary statistics
print(df['Default'].value_counts())  # Class distribution
```

**Step 2: Handle Missing Values**
```python
# Identify missing values
print(df.isnull().sum())

# If <2% missing: Impute via mean (continuous) or mode (categorical)
df['income'].fillna(df['income'].mean(), inplace=True)
df['employment'].fillna(df['employment'].mode()[0], inplace=True)

# If >5% missing: Remove feature or records
df = df.dropna(subset=['critical_feature'])
```

**Step 3: Outlier Detection & Treatment**
```python
# Identify outliers (beyond 3 standard deviations)
for col in df.select_dtypes(include=[np.number]).columns:
    mean = df[col].mean()
    std = df[col].std()
    outliers = (df[col] < mean - 3*std) | (df[col] > mean + 3*std)
    
    # Winsorize (cap at 3σ boundary) instead of removing
    df[col] = np.clip(df[col], mean - 3*std, mean + 3*std)
```

**Step 4: Feature Engineering**
```python
# Create derived features (domain knowledge)
df['debt_to_income'] = df['existing_debt'] / df['annual_income']
df['payment_ability'] = (df['annual_income'] - df['existing_debt']) / df['annual_income']

# Encode categorical variables
df = pd.get_dummies(df, columns=['employment_status', 'loan_purpose'])
```

**Step 5: Scaling & Normalization**
```python
from sklearn.preprocessing import StandardScaler

# Standardize continuous features (mean=0, std=1)
scaler = StandardScaler()
continuous_cols = ['age', 'income', 'debt', 'credit_score', ...]
df[continuous_cols] = scaler.fit_transform(df[continuous_cols])
```

**Step 6: Train/Validation/Test Split**
```python
from sklearn.model_selection import train_test_split

# Split 1: Train + Temp (70/30)
train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df['Default'])

# Split 2: Val + Test (15/15 of original)
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df['Default'])

print(f"Train: {len(train_df)} records ({len(train_df)/len(df)*100:.1f}%)")
print(f"Val:   {len(val_df)} records ({len(val_df)/len(df)*100:.1f}%)")
print(f"Test:  {len(test_df)} records ({len(test_df)/len(df)*100:.1f}%)")
```

---

##  GitHub Repository Structure

**Repository Name:** `CSE516_Credit_Default_BN`  
**Access:** PUBLIC (not private)  
**Link:** `https://github.com/[YourHandle]/CSE516_Credit_Default_BN`

### Repository Organization

```
CSE516_Credit_Default_BN/
├── README.md                                  ← Main project documentation (this file)
├── requirements.txt                           ← Python dependencies (pip install -r requirements.txt)
├── setup.py                                   ← Package setup
├── .gitignore                                 ← Git ignore patterns
│
├── docs/
│   ├── METHODOLOGY.md                         ← Technical approach & mathematical details
│   └── REFERENCES.md                          ← Full citation list (BibTeX format)
│
├── data/
│   ├── raw/
│   │   └── README.md                          ← Data download/source instructions
│   └── processed/
│       └── README.md                          ← Preprocessing pipeline documentation
│
├── src/                                       ← Source code (filled in M2)
│   ├── preprocessing.py                       ← Data preprocessing pipeline
│   ├── pc_algorithm.py                        ← PC Algorithm implementation
│   ├── parameter_learning.py                  ← MLE parameter estimation
│   ├── inference.py                           ← Belief Propagation inference
│   ├── baselines.py                           ← Baseline method implementations
│   └── evaluation.py                          ← Evaluation metrics & comparison
│
├── notebooks/                                 ← Jupyter notebooks (filled in M2+)
│   ├── 01_data_exploration.ipynb
│   ├── 02_pc_algorithm.ipynb
│   ├── 03_parameter_learning.ipynb
│   ├── 04_baseline_comparison.ipynb
│   └── 05_results.ipynb
│
├── results/                                   ← Experimental results (filled in M3)
│   ├── learned_dag.npy                        ← Learned DAG structure
│   ├── parameters.pickle                      ← Learned parameters (CPDs)
│   ├── metrics.json                           ← Evaluation metrics
│   └── visualizations/
│       ├── dag_structure.png                  ← Graphical model visualization
│       ├── baseline_comparison.pdf            ← Performance comparison plots
│       └── causal_effects.png                 ← Causal effect coefficients
│
└── M1_<group_name>/                           ←  M1 SUBMISSION FOLDER
    ├── README.md                              ← M1 folder documentation
    ├── Report/
    │   └── M1_<group_name>_Report.pdf         ← Official M1 proposal (8-10 pages)
    ├── Video/
    │   └── M1_<group_name>_Video_Link.txt     ← Video link (or .mp4 file)
    ├── Code/ (optional)
    │   ├── src/
    │   │   └── [Draft code if any]
    │   └── README.md
    ├── Data/ (optional)
    │   └── README.md                          ← Data preparation notes
    └── Results/ (optional)
        └── [Draft plots, models, etc.]
```

---

##  Video Deliverable

**Duration:** 20 minutes exactly  
**Format:** MP4 (1080p, 30 fps)  
**Location:** `M1_<group_name>/Video/M1_<group_name>_Video_Link.txt` (or video file)

**Content Breakdown:** 6 Segments (See VIDEO_SCRIPT_COMPLETE.md for full script)

1. **Introduction & Problem (3 min)** — Why we need causal models
2. **Graphical Model (3 min)** — Show the Bayesian Network structure
3. **PGM Framework (4 min)** — Representation, Learning, Inference
4. **SOTA & Extension (4 min)** — Senyk et al. + our 5 contributions
5. **Baselines & KPIs (3 min)** — How we measure success
6. **Dataset & Timeline (3 min)** — Data + project schedule

See `M1_<group_name>/Video/VIDEO_SCRIPT_COMPLETE.md` for word-for-word script.

---

##  M1 Contents Summary

### What This M1 Folder Contains

**REQUIRED:**
-  `README.md` — This documentation (project overview, framework, team info)
-  `Report/M1_<group_name>_Report.pdf` — Official 8-10 page proposal
-  `Video/M1_<group_name>_Video_Link.txt` — Video link or MP4 file

 Google Form link

---

## 🎯 Key Questions Each Team Member Should Be Able to Answer

Before the evaluation, ensure each team member can discuss:

1. **Graphical Model**
   - "Show me the Bayesian Network. What do the nodes represent?"
   - "Why is there an edge from Income to Debt?"
   - "What conditional independencies does the DAG imply?"

2. **PGM Components**
   - "How do we learn the structure? (PC Algorithm)"
   - "How do we learn the parameters? (MLE via regression)"
   - "What inferences can we make? (credit scoring, causal effects)"

3. **SOTA Position**
   - "What did Senyk et al. (2024) do?"
   - "How do we extend their work? (5 contributions)"
   - "Why is this publication-ready?"

4. **Baselines & KPIs**
   - "What are your 5 baselines?"
   - "How do you measure success? (3 KPI dimensions)"
   - "What are your target metrics?"

5. **Dataset & Preprocessing**
   - "Where does the data come from?"
   - "How large is the dataset?"
   - "How do you preprocess it?"

6. **Individual Contribution**
   - "What did YOU specifically do for M1?"
   - "Which sections did YOU write?"
   - "What GitHub commits did YOU make?"

---

## 📅 Project Timeline

| Milestone | Date | Deliverables | Status |
|-----------|------|--------------|--------|
| **M1** | Sept 15, 2026 | Proposal + Video + Folder | 🔴 **CURRENT** |
| **M2** | Oct 11, 2026 | PC Algorithm + Data + Parameters | Upcoming |
| **M3** | Nov 1, 2026 | Baselines + Robustness + Causal Effects | Upcoming |
| **M4** | Nov 22, 2026 | Final Paper + Code + Results | Upcoming |

---

## 📧 Contact & Support

**Instructor:** Dhaval Patel 
**Course:** CSE 516 - Probabilistic Graphical Models  
**Institution:** Ahmedabad University 
**Semester:** Monsoon 2026  


**GitHub Repository:** https://github.com/kamyaahduni/CSE516_Credit_Default_BN  

---



##  Appendix: Quick Reference

### The 3 PGM Pillars 

**Representation:** Bayesian Network with 12 nodes, ~16 edges representing causal structure

**Learning:** PC Algorithm for structure discovery + MLE for parameter estimation

**Inference:** Four query types — credit scoring, causal effects, diagnosis, counterfactuals

### The SOTA Anchor 

**Senyk et al. (2024):** "Application of the Bayesian Approach to Modeling Credit Risks"

### The 5 Contributions 

1. Rigorous PC Algorithm
2. Formal statistical validation
3. Comprehensive robustness analysis
4. Multiple baseline comparisons
5. Causal effect quantification

### The 5 Baselines 

1. Logistic Regression
2. Random Forest
3. Decision Tree
4. Gradient Boosting
5. Neural Network

### The 3 KPI Dimensions

1. **Representation:** Structure stability, alignment, significance
2. **Learning:** Parameter precision, significance, model fit
3. **Inference:** Predictive accuracy, competitiveness, interpretability

---

**Document Version:** M1 Final  
**Last Updated:** September 15, 2026  
