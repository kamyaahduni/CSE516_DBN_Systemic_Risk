# M2 REPORT - FORMAL CORRECTIONS

**Milestone:** M2 - Implementation  
**Document Type:** REPORT  
**Status:** INCOMPLETE IMPLEMENTATION

---

## CORRECTION #1: INCOMPLETE BASELINE IMPLEMENTATION

### ERROR LOCATION
**File:** README.md (Main), M1 Report (promised)  
**Claims:** "5 strong baselines: Logistic Regression, Random Forest, Decision Tree, Gradient Boosting, Neural Network"

### ISSUE STATEMENT
M1 proposal and M2 implementation promised evaluation against 5 baseline methods but only logistic regression was actually implemented. The other 4 baselines (RF, DT, GB, NN) are missing from M2 entirely.

### EVIDENCE
**What Was Promised:**
- baseline_lr.py ✓ IMPLEMENTED
- baseline_rf.py ✗ MISSING
- baseline_dt.py ✗ MISSING
- baseline_gb.py ✗ MISSING
- baseline_nn.py ✗ MISSING

**Code Status:**
- Only `Code/src/baseline_lr.py` exists
- No Random Forest baseline implementation
- No Decision Tree baseline implementation
- No Gradient Boosting baseline implementation
- No Neural Network baseline implementation

### CORRECT STATEMENT
Update M2 documentation to reflect actual completion:

**OPTION A: Update to completed work**
```
## Baselines Implemented (M2)

We implemented the following baseline method:
- ✓ Logistic Regression (baseline_lr.py)

Planned baselines (not implemented in M2):
- ⏳ Random Forest (deferred to M3)
- ⏳ Decision Tree (deferred to M3)
- ⏳ Gradient Boosting (deferred to M3)
- ⏳ Neural Network (deferred to M3)

Note: Full baseline comparison planned for M3/M4 evaluation.
```

**OPTION B: Honest statement**
```
## Baseline Status

M2 implements initial structure and parameter learning. Baseline comparisons 
deferred to M3 due to project scope constraints. Only Logistic Regression 
baseline completed in M2.
```

### SEVERITY
 **CRITICAL** - 80% of promised baselines missing  
Affects ability to make comparative claims in M3-M4.

### IMPACT ON M3-M4
This incomplete implementation in M2 cascades to M3-M4 where baselines are claimed but not delivered, leading to false comparative claims.

---

## SUMMARY OF M2 CORRECTIONS

| Error | Type | Severity | Required Action |
|-------|------|----------|---|
| 4 missing baselines (RF, DT, GB, NN) | Implementation | CRITICAL | Update docs or implement baselines |

---

**Correction Document Version:** 1.0  
**Date:** November 12, 2026
