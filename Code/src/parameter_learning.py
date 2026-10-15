"""
Parameter Learning via Maximum Likelihood Estimation (OLS Regression)
"""

import numpy as np
from scipy import stats


class ParameterLearning:
    
    def __init__(self, bn, feature_names):
        self.bn = bn
        self.feature_names = feature_names
        self.params = {}
        self.models = {}
    
    def fit(self, X, y):
        for i, node in enumerate(self.feature_names):
            parents = self.bn.get_parents(node)
            parent_indices = [self.feature_names.index(p) for p in parents]
            
            if len(parent_indices) == 0:
                target = X[:, i]
                X_design = np.ones((X.shape[0], 1))
            else:
                target = X[:, i]
                X_design = np.column_stack([np.ones(X.shape[0]), X[:, parent_indices]])
            
            try:
                coef = np.linalg.lstsq(X_design, target, rcond=None)[0]
                predictions = X_design @ coef
                residuals = target - predictions
                
                n = X.shape[0]
                p = X_design.shape[1]
                sse = np.sum(residuals ** 2)
                mse = sse / (n - p)
                r2 = 1 - (sse / np.sum((target - target.mean()) ** 2) + 1e-10)
                
                var_covar = mse * np.linalg.inv(X_design.T @ X_design)
                se = np.sqrt(np.diag(var_covar))
                t_crit = stats.t.ppf(0.975, n - p)
                ci_lower = coef - t_crit * se
                ci_upper = coef + t_crit * se
                ci_width = ci_upper[0] - ci_lower[0]
                
                t_stats = coef / (se + 1e-10)
                p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - p))
                
                self.params[node] = {
                    'coefficient': coef[0],
                    'parents_coef': coef[1:] if len(parent_indices) > 0 else [],
                    'ci_lower': ci_lower[0],
                    'ci_upper': ci_upper[0],
                    'ci_width': ci_width,
                    'p_value': p_values[0],
                    'r_squared': r2,
                    'mse': mse
                }
                self.models[node] = {'coef': coef, 'se': se}
                
            except:
                self.params[node] = {
                    'coefficient': 0,
                    'parents_coef': [0] * len(parent_indices),
                    'ci_lower': 0,
                    'ci_upper': 0,
                    'ci_width': 0,
                    'p_value': 1.0,
                    'r_squared': 0,
                    'mse': 0
                }
        
        return self.params
    
    def compute_r2(self, X, y):
        r2_values = []
        
        for i, node in enumerate(self.feature_names):
            parents = self.bn.get_parents(node)
            parent_indices = [self.feature_names.index(p) for p in parents]
            
            target = X[:, i]
            
            if len(parent_indices) == 0:
                X_design = np.ones((X.shape[0], 1))
            else:
                X_design = np.column_stack([np.ones(X.shape[0]), X[:, parent_indices]])
            
            if node in self.models:
                coef = self.models[node]['coef']
                predictions = X_design @ coef
                residuals = target - predictions
                sse = np.sum(residuals ** 2)
                sst = np.sum((target - target.mean()) ** 2)
                r2 = 1 - (sse / (sst + 1e-10))
                r2_values.append(max(0, r2))
        
        return np.mean(r2_values) if r2_values else 0.0
