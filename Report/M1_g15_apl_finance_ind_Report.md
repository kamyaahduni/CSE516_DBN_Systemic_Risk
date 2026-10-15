# CSE 516: Probabilistic Graphical Models
## M1 Proposal and Scoping Document

**Project:** Credit Default Prediction using Causal Bayesian Networks  
**Category:** Applied AI  
**Group:** 12 
**Team Members:** Kamya 
**Institution:** Ahmedabad University 
**Submission Date:** September 15, 2026

---

## 1. Problem Statement

### 1.1 Problem Definition

Credit default prediction is a critical challenge in financial services and risk management. Banks and lending institutions must determine which borrowers are likely to default on loans to minimize financial losses while maintaining lending capacity. While traditional credit scoring models (logistic regression, random forests, neural networks) provide predictions, they fail to answer fundamental causal questions: "Does low credit score *cause* default, or is it a symptom of underlying income instability?" "Which factors are true drivers of default vs. spurious correlations?" This lack of interpretability violates regulatory requirements and undermines trust in lending decisions.

### 1.2 Motivation and Significance

**Motivation:**
- **Regulatory Requirement:** Lending institutions are increasingly required to explain credit decisions (e.g., Fair Lending laws). Black-box models cannot provide causal justifications.
- **Risk Management:** Understanding causal pathways to default enables targeted interventions (e.g., "stabilize borrower income" rather than "increase credit score requirements").
- **Fairness & Explainability:** Causal models provide interpretable, auditable decision-making essential for ethical AI in finance.

**Significance:**
- Extends causal inference to credit risk—an understudied but high-impact application domain
- Bridges machine learning and financial services via Bayesian Networks and causal discovery
- Publication-ready contribution to finance + AI intersection

### 1.3 Scope and Assumptions

**Scope:**
- Domain: Credit default prediction (binary classification: default vs. non-default)
- Dataset: 100,000+ individual loan records with 15-20 features
- Time Horizon: Single-period prediction (cross-sectional, not time-series)
- Geographic Focus: Single country/market (historical lending data)

**Assumptions:**
- **Causal Markov Assumption:** The graphical model captures all conditional independencies in the data; unobserved confounders are negligible
- **Causal Sufficiency:** No unobserved common causes affecting multiple observed variables
- **No Reverse Causality:** Causality flows from borrower attributes → default, not vice versa
- **Linear Relationships:** Gaussian Bayesian Network assumes linear relationships (validated via residual analysis)
- **Stationarity:** Causal structure remains constant across the data collection period

---

## 2. Mapping onto PGM Framework

### 2.1 Representation

**Graphical Model Type:** Bayesian Network (Directed Acyclic Graph)

**Model Specification:**

**Nodes (Random Variables):**
| Variable | Type | Description | Range/Values |
|----------|------|-------------|--------------|
| Age | Continuous | Borrower's age | 18–75 years |
| Employment_Status | Categorical | Employment type | {Employed, Self-employed, Unemployed, Retired} |
| Years_Employed | Continuous | Duration in current job | 0–50 years |
| Annual_Income | Continuous | Gross annual income | $20K–$500K |
| Existing_Debt | Continuous | Total outstanding debt | $0–$500K |
| Debt_to_Income_Ratio | Continuous | Debt / Income | 0–10 |
| Credit_Score | Continuous | Credit score | 300–850 |
| Credit_History_Length | Continuous | Years since credit account opened | 0–70 years |
| Loan_Amount | Continuous | Amount of loan applied for | $1K–$1M |
| Loan_Purpose | Categorical | Reason for loan | {Home, Auto, Personal, Education, Business} |
| Payment_Ability | Continuous | Capacity to repay (derived feature) | 0–1 |
| Default | Binary (Target) | Loan default status | {0 = Non-Default, 1 = Default} |

**Total Nodes:** 12 (11 input, 1 target)

**Edges (Causal Relationships):**

Preliminary causal structure based on financial domain theory and lending practice:

```
Age → Existing_Debt
Age → Credit_History_Length
Age → Employment_Status

Employment_Status → Annual_Income
Employment_Status → Years_Employed

Annual_Income → Existing_Debt
Annual_Income → Loan_Amount
Annual_Income → Payment_Ability

Years_Employed → Payment_Ability
Years_Employed → Credit_Score

Existing_Debt → Debt_to_Income_Ratio
Existing_Debt → Payment_Ability

Credit_Score → Credit_History_Length
Credit_Score → Payment_Ability

Loan_Amount → Debt_to_Income_Ratio

Payment_Ability → Default
Debt_to_Income_Ratio → Default
Credit_Score → Default
Credit_History_Length → Default
```

**Total Edges:** ~16 causal edges (refined during structure learning)

**Conditional Independence Assumptions:**

Based on the DAG, the following conditional independencies hold:
- Age ⊥ Annual_Income (no direct edge; income influenced only by employment)
- Age ⊥ Loan_Purpose (no causal relationship)
- Employment_Status ⊥ Credit_History_Length | Age (age confounds relationship)
- Loan_Amount ⊥ Credit_Score (independent mechanisms)

**Why This Representation is Appropriate:**

1. **Captures Causality:** DAG structure explicitly models causal pathways to default, distinguishing root causes from symptoms.
2. **Domain Alignment:** Structure aligns with lending domain knowledge (income influences debt, employment affects creditworthiness, etc.).
3. **Interpretability:** Each edge has economic meaning; decision-makers can understand why default is predicted.
4. **Modularity:** Conditional probability tables (CPTs) can be updated as lending policies change.
5. **Inference-Ready:** Supports both predictive queries (P(Default | attributes)) and counterfactual reasoning ("If income increases, default probability decreases?").

---

### 2.2 Inference

**Inference Tasks:**

**Task 1: Credit Scoring (Primary Inference)**
- **Query:** P(Default = Yes | Age, Employment_Status, Annual_Income, Credit_Score, ...)
- **Meaning:** Predict probability of default given borrower profile
- **Application:** Accept/reject loan, set interest rate

**Task 2: Causal Effect Estimation**
- **Query:** What is the causal effect of Income on Default? (Holds other variables constant)
- **Meaning:** How much does default probability change if income increases by $10K?
- **Application:** Policy analysis ("increase income requirements" vs. "reduce credit score thresholds")

**Task 3: Diagnosis (Backward Reasoning)**
- **Query:** Given Default = Yes, what is P(Low_Income | Default)?
- **Meaning:** Among defaulters, what proportion have low income?
- **Application:** Risk profiling, targeted interventions

**Task 4: Counterfactual Reasoning**
- **Query:** "If this borrower's employment status changed to 'Employed', would they default?"
- **Meaning:** Estimate outcome under hypothetical intervention
- **Application:** What-if analysis for policy decisions

**Inference Method:** Belief Propagation (Pearl's algorithm) for exact inference on Bayesian Networks
- Handles discrete and continuous variables via Gaussian belief propagation
- Scalable to moderate-sized networks (12 nodes)
- Provides exact marginal and conditional probabilities

---

### 2.3 Learning

**Structure Learning:**

**Goal:** Discover the causal DAG from data (not just use domain knowledge)

**Method:** PC Algorithm (Constraint-Based Structure Learning)
- **Phase 1:** Start with complete undirected graph (all variables connected)
- **Phase 2:** Test pairwise independence (remove edges between independent variables)
  - Statistical test: Partial correlation with significance level α = 0.05
  - Edge removed if: |correlation| < threshold
- **Phase 3:** Condition on larger sets (test conditional independence)
  - Example: If Age and Income independent given Employment_Status, remove edge
- **Phase 4:** Orient edges using v-structure rules (derive causal direction)
  - If A → C ← B and A, B independent, then A causes both (v-structure)

**Why PC Algorithm?**
- Principled: Uses statistical independence tests, not heuristics
- Rigorous: Comes with theoretical guarantees under causal assumptions
- Interpretable: Each edge removal justified by p-value
- SOTA: State-of-the-art for causal discovery in observational data

**Training Data:** 100,000 loan records split into:
- 70% training (70K records) → Structure learning + parameter learning
- 15% validation (15K records) → Hyperparameter tuning, initial evaluation
- 15% test (15K records) → Final evaluation, robustness testing

---

**Parameter Learning:**

**Goal:** Estimate CPT values (conditional probabilities) given learned structure

**Method:** Maximum Likelihood Estimation (MLE) via Linear Regression
- For each node with parents, fit regression: Node = f(Parents)
- Example: Default = β₀ + β₁·Payment_Ability + β₂·Credit_Score + ε
- Parameters (β) estimated via OLS (Ordinary Least Squares)

**Implementation:**
- Continuous variables: Gaussian CPTs (mean, variance)
- Discrete variables: Multinomial CPTs (probability distributions)
- Mixed: Conditional linear Gaussian CPTs

**Validation:**
- R² score: % variance explained by learned CPDs
- Log-likelihood: Model fit quality
- Cross-validation: Ensure parameters generalize

---

## 3. State-of-the-Art (SOTA) Position

### 3.1 SOTA Reference Paper

**Base Paper:**

A. P. Senyk, O. S. Manziy, P. E. Ohloblin, and V. V. Petrovych, "Application of the Bayesian Approach to Modeling Credit Risks," *Mathematical Modeling and Computing*, vol. 11, no. 4, pp. 1–15, 2024.

### 3.2 SOTA Summary

**What Senyk et al. (2024) Does:**
- Models credit risk using Bayesian Networks
- Develops a BN with ~10 credit-related variables
- Uses heuristic structure learning (domain expert input + statistical testing)
- Estimates parameters via Bayesian inference
- Evaluates on real credit dataset (Eastern European lending data)
- Reports 75-85% prediction accuracy

**Key Findings:**
- Bayesian Networks outperform logistic regression on interpretability
- Causal structure reveals employment → income → creditworthiness chain
- Identifies critical risk pathways for policy intervention

**Limitations of Senyk et al.:**
1. **Structure Learning:** Uses heuristic, ad-hoc approach (manual pruning + statistical tests)
   - Not principled or reproducible
   - Relies heavily on domain expert judgment
   - Cannot systematically discover surprising causal paths

2. **No Formal Statistical Testing:** Missing formal hypothesis testing on edges
   - No p-values or confidence intervals on discovered edges
   - Cannot assess statistical significance of structure

3. **Limited Robustness Analysis:** No evaluation of:
   - Structure stability across data subsamples
   - Sensitivity to hyperparameters or missing data
   - Generalization to new populations

4. **Baseline Comparisons:** Only compares to logistic regression
   - Missing comparisons to Random Forest, Neural Networks, Gradient Boosting
   - Cannot claim superiority over modern ML approaches

### 3.3 How Our Work Extends SOTA

**Our Contribution:**

1. **Rigorous Structure Learning (PC Algorithm)**
   - Principled constraint-based method replaces heuristic pruning
   - Formal independence testing with p-values and significance levels
   - Reproducible, auditable causal discovery

2. **Formal Statistical Validation**
   - Confidence intervals (95% CI) on all causal effects
   - Hypothesis testing framework for edge inclusion/exclusion
   - Multiple testing correction (Benjamini-Hochberg FDR control)

3. **Comprehensive Robustness Analysis**
   - Jackknife stability: Leave-one-year-out cross-validation
   - Bootstrap confidence intervals on learned parameters
   - Sensitivity analysis: How structure changes with α ∈ {0.01, 0.05, 0.10}
   - Subgroup analysis: Does structure hold across borrower demographics?

4. **Multiple Baseline Comparisons**
   - Logistic Regression, Random Forest, Decision Tree, Neural Network, Gradient Boosting
   - Compare on predictive accuracy (AUC, F1) AND interpretability
   - Show that rigorous BN achieves competitive accuracy with superior explainability

5. **Causal Effect Quantification**
   - Report standardized causal effects (β coefficients) with confidence intervals
   - Enable policy analysis: "How much does $10K income increase reduce default?"
   - Counterfactual predictions: "If employment status changed..."

6. **Publication-Ready Methodology**
   - Methodologically rigorous for top-tier venues (A* journals/conferences)
   - Clear alignment with CSE 516 course (Representation + Learning + Inference)
   - Replicable across different datasets/populations

---

## 4. Baselines and KPIs

### 4.1 Baseline Methods

We compare our Bayesian Network against five standard baselines:

| Baseline | Type | Implementation | Justification |
|----------|------|---|---|
| **Logistic Regression** | Linear classifier | statsmodels.GLM | Industry standard for credit scoring |
| **Random Forest** | Ensemble, non-parametric | sklearn.ensemble.RandomForestClassifier | Modern ML baseline, interpretable feature importance |
| **Decision Tree** | Rule-based, interpretable | sklearn.tree.DecisionTreeClassifier | Simple, interpretable alternative |
| **Gradient Boosting** | Ensemble, sequential | sklearn.ensemble.GradientBoostingClassifier | State-of-the-art ML performance |
| **Neural Network** | Deep learning | TensorFlow.keras | Flexible, high-capacity black-box |

**Baseline Rationale:**
- Cover spectrum: Statistical (Logistic), Tree-based (RF, DT, GB), Deep Learning (NN)
- Include industry standard (Logistic Regression used in real lending)
- Enable fair comparison: All trained on same dataset, hyperparameters tuned

---

### 4.2 Key Performance Indicators (KPIs)

KPIs evaluated across three PGM dimensions:

**DIMENSION 1: REPRESENTATION (Does the learned DAG make sense?)**

| KPI | Metric | Target | Rationale |
|-----|--------|--------|-----------|
| **Structure Stability** | % edges stable across jackknife samples | > 80% | Edges appearing in >80% of subsamples are robust |
| **SOTA Alignment** | % edges matching Senyk et al. structure | > 70% | Should recover known causal pathways |
| **Edge Significance** | % edges with p-value < 0.05 | > 90% | Edges should be statistically significant |
| **Domain Alignment** | Expert review: Do edges make economic sense? | 100% | Causal structure should match lending domain theory |

**DIMENSION 2: LEARNING (Are parameters learned correctly?)**

| KPI | Metric | Target | Rationale |
|-----|--------|--------|-----------|
| **Parameter Precision** | 95% CI width for β coefficients | < 0.2 | Narrow confidence intervals = precise estimates |
| **Parameter Significance** | % parameters with p-value < 0.05 | > 85% | Most parameters should be statistically significant |
| **Model Fit (Training)** | R² on training data | > 0.40 | Learned CPDs explain ≥40% of variance |
| **Generalization** | R² on test data | > 0.38 | Test R² close to training (no overfitting) |

**DIMENSION 3: INFERENCE (Do predictions work well?)**

| KPI | Metric | Target | Rationale |
|-----|--------|--------|-----------|
| **Predictive Accuracy** | AUC-ROC on test set | > 0.75 | Discriminates default vs. non-default well |
| **vs. Baselines** | BN AUC - Best Baseline AUC | ≥ -0.02 | Competitive with modern ML (within 2%) |
| **Calibration** | Brier score | < 0.20 | Predicted probabilities match observed frequencies |
| **Interpretability** | SHAP/feature importance alignment with causal edges | > 80% | Explanation mechanism aligns with learned causality |

---

## 5. Dataset(s)

### 5.1 Dataset Selection

**Primary Dataset:** Kaggle Loan Default Dataset / UCI Credit Default Dataset

**Source & Access:**
- **Option A:** https://www.kaggle.com/datasets/ (search "loan default")
  - Public, free download, CSV format
  - 100K–500K records
  - Requires Kaggle account (free signup)

- **Option B:** https://archive.ics.uci.edu/ml/
  - UCI Repository: German Credit, Credit Approval datasets
  - Smaller (~1K records) but well-studied

### 5.2 Dataset Description

| Aspect | Details |
|--------|---------|
| **Records** | 100,000–500,000 loans |
| **Time Period** | Historical data, 5–10 years of lending |
| **Geographic Scope** | [Single country/market, e.g., USA] |
| **Features** | 15–20 variables (demographics, financial, loan details) |
| **Target Variable** | Loan default status (binary: Default vs. Non-Default) |
| **Class Distribution** | ~85% non-default, ~15% default (imbalanced) |
| **Feature Types** | Mix: Continuous (income, debt), Categorical (employment, purpose) |
| **Data Quality** | Minimal missing values (<2%), no major quality issues |
| **Size** | ~50–100 MB (CSV) |

### 5.3 Data Preprocessing Pipeline

**Step 1: Data Loading & Exploration**
- Load CSV into pandas DataFrame
- Exploratory Data Analysis (EDA): distributions, missing values, outliers

**Step 2: Handling Missing Values**
- Missing < 2%: Impute via mean (continuous) or mode (categorical)
- Missing > 5%: Remove feature or record

**Step 3: Outlier Detection & Treatment**
- Identify: Values beyond 3 standard deviations
- Treat: Cap at 3σ boundary (Winsorization) or remove records

**Step 4: Feature Engineering**
- Create derived features: Debt-to-Income Ratio, Payment Ability Index
- Encode categorical variables: One-hot encoding (for inference), ordinal (for learning)

**Step 5: Scaling & Normalization**
- Standardize continuous features: z-score normalization (mean=0, std=1)
- Needed for: PC Algorithm (independence tests), parameter learning stability

**Step 6: Train/Val/Test Split**
- Train: 70% (70K records) → Structure learning, parameter learning
- Validation: 15% (15K records) → Hyperparameter tuning
- Test: 15% (15K records) → Final evaluation

---

## 6. GitHub Repository

**Repository Name:** `CSE516_Credit_Default_BN`

**Link:** https://github.com/kamyaahduni/CSE516_DBN_Systemic_Risk.git

**Repository Structure:**
```
CSE516_Credit_Default_BN/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── docs/
│   ├── METHODOLOGY.md
│   └── REFERENCES.md
├── data/
│   ├── raw/
│   │   └── README.md
│   └── processed/
│       └── README.md
├── src/
│   ├── preprocessing.py
│   ├── pc_algorithm.py
│   ├── parameter_learning.py
│   ├── inference.py
│   ├── baselines.py
│   └── evaluation.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_pc_algorithm.ipynb
│   ├── 03_parameter_learning.ipynb
│   ├── 04_baseline_comparison.ipynb
│   └── 05_results.ipynb
├── results/
│   ├── learned_dag.npy
│   ├── parameters.pickle
│   ├── metrics.json
│   └── visualizations/
└── M1_GroupName/
    ├── README.md
    ├── Report/
    │   └── M1_GroupName_Report.pdf
    ├── Video/
    │   └── M1_GroupName_Video_Link.txt
    └── Code/ (M1-stage materials)
```

---

## 7. References

### Base Paper (SOTA Anchor)

[1] A. P. Senyk, O. S. Manziy, P. E. Ohloblin, and V. V. Petrovych, "Application of the Bayesian Approach to Modeling Credit Risks," *Mathematical Modeling and Computing*, vol. 11, no. 4, pp. 1–15, 2024.

### Foundational PGM

[2] D. Koller and N. Friedman, *Probabilistic Graphical Models: Principles and Techniques*. MIT Press, 2009.

[3] K. P. Murphy, *Machine Learning: A Probabilistic Perspective*. MIT Press, 2012.

[4] J. Pearl, *Causality: Models, Reasoning, and Inference*, 2nd ed. Cambridge University Press, 2009.

### Structure Learning & Causal Discovery

[5] P. Spirtes and C. Glymour, "An Algorithm for Fast Recovery of Sparse Causal Graphs," *Social Science Research*, vol. 62, no. 1, pp. 9–28, 1991.

[6] P. Spirtes, C. N. Glymour, and R. Scheines, *Causation, Prediction, and Search*, 2nd ed. MIT Press, 2000.

### Credit Risk & Default Prediction

[7] Z. Li, Y. Qu, and H. Zhang, "Machine Learning Methods for Default Prediction: A Survey," *Computational Economics*, vol. 57, no. 1, pp. 123–156, 2021.

[8] T. Hastie, R. Tibshirani, and J. Friedman, *The Elements of Statistical Learning*, 2nd ed. Springer, 2009.

### Causal Inference & Explainability

[9] A. Sharma, A. G. Reddy, A. Kumar, and B. B. Sharma, "Causal Inference in Finance: A Survey," *ACM Computing Surveys*, vol. 56, no. 3, pp. 1–38, 2024.

[10] J. Pearl and D. Mackenzie, *The Book of Why: The New Science of Cause and Effect*. Basic Books, 2018.

### Statistical Testing & Model Evaluation

[11] Y. Benjamini and Y. Hochberg, "Controlling the False Discovery Rate," *Journal of the Royal Statistical Society*, vol. 57, no. 1, pp. 289–300, 1995.

[12] T. Fawcett, "An Introduction to ROC Analysis," *Pattern Recognition Letters*, vol. 27, no. 8, pp. 861–874, 2006.

### Software & Implementation

[13] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.

[14] T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," in *Proceedings of KDD 2016*, pp. 785–794, 2016.

[15] M. Abadi et al., "TensorFlow: A System for Large-Scale Machine Learning," in *Proceedings of OSDI 2016*, pp. 265–283, 2016.

---

## Summary & Next Steps

This proposal establishes a rigorous, publication-quality project applying causal Bayesian Networks to credit default prediction. By combining principled structure learning (PC Algorithm), formal statistical validation, and comprehensive robustness analysis, we extend Senyk et al. (2024) with methodological rigor unmatched in current credit risk literature.


**Expected Impact:**
- A*, publication-ready research contribution to credit risk + causal inference
- Actionable causal insights for lending policy
- Open-source code and reproducible methodology for the community
