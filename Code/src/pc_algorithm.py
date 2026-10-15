"""
PC Algorithm for constraint-based structure learning
Phase 1-4: Conditional independence testing with Fisher z-transform
"""

import numpy as np
from scipy.stats import norm
from itertools import combinations


class PCAlgorithm:
    """Peter-Clark (PC) Algorithm for Bayesian Network structure learning"""
    
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.n_features = None
        self.edges = []
    
    def fit(self, X):
        self.n_features = X.shape[1]
        adjacency = {i: set(range(self.n_features)) - {i} for i in range(self.n_features)}
        
        self._phase_1(X, adjacency)
        self._phase_2(X, adjacency)
        self._phase_3(X, adjacency)
        self._phase_4(adjacency)
        
        return self.edges
    
    def _phase_1(self, X, adjacency):
        to_remove = []
        
        for i in range(self.n_features):
            for j in range(i + 1, self.n_features):
                corr = np.corrcoef(X[:, i], X[:, j])[0, 1]
                z_stat = 0.5 * np.log((1 + corr) / (1 - corr + 1e-10))
                z_stat *= np.sqrt(X.shape[0] - 3)
                p_value = 2 * (1 - norm.cdf(np.abs(z_stat)))
                
                if p_value > self.alpha:
                    to_remove.append((i, j))
        
        for i, j in to_remove:
            if j in adjacency[i]:
                adjacency[i].discard(j)
                adjacency[j].discard(i)
    
    def _phase_2(self, X, adjacency):
        to_remove = []
        
        for i in range(self.n_features):
            for j in list(adjacency[i]):
                for z in list(adjacency[i] - {j}):
                    if self._is_independent(X, i, j, {z}):
                        to_remove.append((i, j))
                        break
        
        for i, j in to_remove:
            if j in adjacency[i]:
                adjacency[i].discard(j)
                adjacency[j].discard(i)
    
    def _phase_3(self, X, adjacency):
        for order in range(2, self.n_features):
            to_remove = []
            
            for i in range(self.n_features):
                for j in list(adjacency[i]):
                    candidates = adjacency[i] - {j}
                    
                    for cond_set in combinations(candidates, min(order, len(candidates))):
                        if self._is_independent(X, i, j, set(cond_set)):
                            to_remove.append((i, j))
                            break
            
            if not to_remove:
                break
            
            for i, j in to_remove:
                if j in adjacency[i]:
                    adjacency[i].discard(j)
                    adjacency[j].discard(i)
    
    def _phase_4(self, adjacency):
        edges = []
        for i in range(self.n_features):
            for j in adjacency[i]:
                if i < j:
                    edges.append((i, j))
        self.edges = edges
    
    def _is_independent(self, X, i, j, cond_set):
        partial_corr = self._partial_correlation(X, i, j, cond_set)
        z_stat = 0.5 * np.log((1 + partial_corr) / (1 - partial_corr + 1e-10))
        z_stat *= np.sqrt(X.shape[0] - len(cond_set) - 3)
        p_value = 2 * (1 - norm.cdf(np.abs(z_stat)))
        return p_value > self.alpha
    
    def _partial_correlation(self, X, i, j, cond_set):
        if len(cond_set) == 0:
            return np.corrcoef(X[:, i], X[:, j])[0, 1]
        
        X_i_resid = X[:, i] - self._regress_out(X, i, cond_set)
        X_j_resid = X[:, j] - self._regress_out(X, j, cond_set)
        
        partial_corr = np.corrcoef(X_i_resid, X_j_resid)[0, 1]
        return np.clip(partial_corr, -1, 1)
    
    def _regress_out(self, X, target, features):
        X_features = X[:, list(features)]
        X_features = np.column_stack([np.ones(X.shape[0]), X_features])
        
        try:
            coef = np.linalg.lstsq(X_features, X[:, target], rcond=None)[0]
            return X_features @ coef
        except:
            return np.zeros(X.shape[0])
