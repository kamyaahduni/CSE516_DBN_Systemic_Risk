# M1 REPORT - FORMAL CORRECTIONS

**Milestone:** M1 - Project Proposal  
**Document Type:** REPORT  
**Submission Date (Original):** September 15, 2026  
**Correction Date:** November 12, 2026  
**Status:** REQUIRES CORRECTION BEFORE USE

---

## CORRECTION #1: WRONG GROUP NUMBER

### ERROR LOCATION
**File:** Report/M1_g15_apl_finance_ind_Report.md  
**Line 6:**
```
Group: 12
```

### ISSUE STATEMENT
The M1 proposal document lists the group number as "12" but all subsequent milestones (M2, M3, M4) and the GitHub repository reference the correct group number as "G15" or "Group 15".

### EVIDENCE
- GitHub repository: `kamyaahduni/CSE516_DBN_Systemic_Risk`
- M4 document header: "Group: G15"
- M2, M3 documents: Reference "G15"
- Consistent usage in all final submissions: "G15"

### CORRECT VERSION
```
Group: G15
```

### CORRECTION ACTION REQUIRED
**Change Line 6 from:**
```
Group: 12
```

**To:**
```
Group: G15
```

### SEVERITY
🟡 **MINOR** - Inconsistency between M1 and later milestones  
Does not affect technical content, only administrative metadata.

---

## CORRECTION #2: DATASET SIZE DISCREPANCY

### ERROR LOCATION
**File:** Report/M1_g15_apl_finance_ind_Report.md  
**Line 36:**
```
Dataset: 100,000+ individual loan records with 15-20 features
```

### ISSUE STATEMENT
The M1 proposal specifies a dataset of "100,000+ individual loan records" but the actual implementation (M2-M4) uses a synthetic dataset of only "500 samples with 11 features". This significant discrepancy is never explicitly explained or justified in the documentation.

### EVIDENCE
- **M1 Scope Statement:** "100,000–500,000 individual loans" (Line 413 in main README)
- **M4 Actual Dataset:** "500 samples, 11 features" (Data_README.md)
- **Actual Implementation:** Uses synthetic data, not 100K+ real loan records

### ROOT CAUSE
The project switched from planned use of large real-world dataset to synthetic dataset for:
- Computational efficiency (smaller dataset for M2-M4 course timeline)
- Reproducibility (synthetic data with fixed seed)
- Privacy (no real credit data needed for educational work)

### CORRECTION ACTION REQUIRED

**OPTION A: Update M1 to match actual implementation**
```
Dataset: Synthetic credit dataset with 500 samples and 11 financial features
(Note: Original proposal planned 100,000+ records; reduced scope for 
computational feasibility within course timeline)
```

**OPTION B: Add explanation section to M1**
Add new section explaining scope change:
```
## Note on Dataset Scope Change

Original proposal specified 100,000+ real loan records. The final implementation 
uses a synthetic dataset of 500 samples for the following reasons:

1. **Timeline Constraints:** Course milestones (M1-M4) required rapid iteration
2. **Reproducibility:** Synthetic data with fixed seed (42) ensures reproducible results
3. **Privacy:** No real credit data required for educational demonstration
4. **Computational Efficiency:** 500-sample experiments run in minutes for feasibility analysis

This change is noted in M2-M4 documentation but should be clarified in M1 proposal.
```

### SEVERITY
 **MODERATE** - Significant scope change not explicitly documented  
Affects understanding of project evolution but does not affect technical validity.

---

## SUMMARY OF M1 CORRECTIONS

| Error | Type | Severity | Required Action |
|-------|------|----------|--|
| Wrong group number (12 vs G15) | Metadata | MINOR | Change Line 6 |
| Dataset size discrepancy (100K vs 500) | Scope | MODERATE | Add explanation or update scope |

---

## IMPLEMENTATION CHECKLIST

- [ ] Update group number from "12" to "G15"
- [ ] Add explanation for dataset scope change
- [ ] Verify consistency with M2-M4 documents
- [ ] Commit corrections with message: "Correct M1 metadata and scope documentation"

---

**Correction Document Version:** 1.0  
**Date:** November 12, 2026
