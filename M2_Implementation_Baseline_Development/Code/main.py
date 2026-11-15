"""
Main pipeline orchestrator for Bayesian Network structure learning and inference
CSE 516 - M2 Submission
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import json

from src.pc_algorithm import PCAlgorithm
from src.bayesian_network import BayesianNetwork
from src.parameter_learning import ParameterLearning
from src.inference_bp import BeliefPropagation
from src.baseline_lr import BaselineLogisticRegression
from src.utils import load_data, compute_kpis, save_results

def main():
    """Main pipeline: Load → PC → BN Init → Learn → Infer → Baseline"""
    
    print("=" * 80)
    print("BAYESIAN NETWORKS FOR CREDIT DEFAULT PREDICTION - M2 SUBMISSION")
    print("Research Group G15 - CSE 516")
    print("=" * 80)
    
    print("\n[STAGE 1] LOADING AND PREPROCESSING DATA")
    print("-" * 80)
    
    X, y = load_data()
    n_samples = X.shape[0]
    n_features = X.shape[1]
    
    print(f"✓ Data loaded: {n_samples} samples, {n_features} features")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    train_idx = int(0.7 * n_samples)
    val_idx = int(0.85 * n_samples)
    
    X_train = X_scaled[:train_idx]
    y_train = y[:train_idx]
    X_val = X_scaled[train_idx:val_idx]
    y_val = y[train_idx:val_idx]
    X_test = X_scaled[val_idx:]
    y_test = y[val_idx:]
    
    print(f"✓ Train set: {len(X_train)} samples (70%)")
    print(f"✓ Validation set: {len(X_val)} samples (15%)")
    print(f"✓ Test set: {len(X_test)} samples (15%)")
    print(f"✓ Data standardized (StandardScaler: zero mean, unit variance)")
    
    print("\n[STAGE 2] PC ALGORITHM - STRUCTURE LEARNING")
    print("-" * 80)
    
    pc = PCAlgorithm(alpha=0.05)
    edges = pc.fit(X_train)
    
    print(f"✓ Phase 1 (Zero-order tests): Complete")
    print(f"✓ Phase 2 (First-order tests): Complete")
    print(f"✓ Phase 3 (Higher-order tests): Complete")
    print(f"✓ Phase 4 (Edge orientation): Complete")
    print(f"✓ Discovered edges: {len(edges)}")
    print(f"✓ Statistical threshold: α = 0.05")
    
    print("\n[STAGE 3] BAYESIAN NETWORK INITIALIZATION")
    print("-" * 80)
    
    bn = BayesianNetwork()
    feature_names = [f"Feature_{i}" for i in range(n_features)]
    
    for name in feature_names:
        bn.add_node(name)
    
    for src, dst in edges:
        bn.add_edge(feature_names[src], feature_names[dst])
    
    print(f"✓ Network initialized with {len(feature_names)} nodes")
    print(f"✓ Edges from PC Algorithm: {len(edges)}")
    
    print("\n[STAGE 4] PARAMETER LEARNING - MAXIMUM LIKELIHOOD ESTIMATION")
    print("-" * 80)
    
    pl = ParameterLearning(bn, feature_names)
    params = pl.fit(X_train, y_train)
    
    ci_widths = [p['ci_width'] for p in params.values() if 'ci_width' in p]
    mean_ci_width = np.mean(ci_widths) if ci_widths else 0
    
    significant_params = sum(1 for p in params.values() if p.get('p_value', 1) < 0.05)
    total_params = len([p for p in params.values() if 'p_value' in p])
    param_significance = (significant_params / total_params * 100) if total_params > 0 else 0
    
    train_r2 = pl.compute_r2(X_train, y_train)
    val_r2 = pl.compute_r2(X_val, y_val)
    
    print(f"✓ Mean Confidence Interval Width (95%): {mean_ci_width:.4f}")
    print(f"✓ Significant Parameters (p < 0.05): {param_significance:.1f}%")
    print(f"✓ Training R²: {train_r2:.4f}")
    print(f"✓ Validation R²: {val_r2:.4f}")
    
    print("\n[STAGE 5] PROBABILISTIC INFERENCE - BELIEF PROPAGATION")
    print("-" * 80)
    
    bp = BeliefPropagation(bn, params, feature_names)
    predictions = bp.infer(X_test)
    
    print(f"✓ Predictions generated for {len(X_test)} test samples")
    print(f"✓ Prediction range: [{predictions.min():.3f}, {predictions.max():.3f}]")
    print(f"✓ Mean prediction: {predictions.mean():.3f}")
    
    print("\n[STAGE 6] BASELINE TRAINING - LOGISTIC REGRESSION")
    print("-" * 80)
    
    baseline = BaselineLogisticRegression()
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)
    
    print(f"✓ Logistic Regression baseline trained")
    print(f"✓ Predictions generated for {len(X_test)} test samples")
    
    print("\n[STAGE 7] RESULTS AGGREGATION & KPI COMPUTATION")
    print("-" * 80)
    
    kpis = compute_kpis(
        predictions=predictions,
        baseline_pred=baseline_pred,
        y_true=y_test,
        edges=edges,
        train_r2=train_r2,
        val_r2=val_r2,
        ci_width=mean_ci_width,
        param_significance=param_significance
    )
    
    save_results(
        kpis=kpis,
        predictions=predictions,
        edges=edges,
        params=params,
        y_true=y_test
    )
    
    print(f"✓ KPIs computed ({len(kpis)} metrics)")
    print(f"✓ Results saved to ../Results/ folder")
    print(f"✓ Files: initial_kpi_results.csv, learned_dag.json, parameters.csv")
    
    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 80)
    print(f"\nKey Results:")
    print(f"  • Structure: {len(edges)} edges discovered")
    print(f"  • Parameters: {mean_ci_width:.4f} mean CI width")
    print(f"  • Training: R² = {train_r2:.4f}")
    print(f"  • Validation: R² = {val_r2:.4f}")
    print(f"  • Inference: AUC = {kpis.get('auc_roc', 0):.4f}")
    print(f"\nAll results saved to: ../Results/")
    print("=" * 80)


if __name__ == "__main__":
    main()
