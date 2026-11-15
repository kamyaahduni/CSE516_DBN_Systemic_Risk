"""
Baseline: Logistic Regression for comparative evaluation
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np


class BaselineLogisticRegression:
    
    def __init__(self, selected_features=None):
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.selected_features = selected_features or list(range(7))
        self.is_fitted = False
    
    def fit(self, X, y):
        X_selected = X[:, self.selected_features]
        self.model.fit(X_selected, y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_selected = X[:, self.selected_features]
        return self.model.predict_proba(X_selected)[:, 1]
    
    def predict_class(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_selected = X[:, self.selected_features]
        return self.model.predict(X_selected)
    
    def evaluate(self, X, y):
        predictions = self.predict(X)
        auc = roc_auc_score(y, predictions)
        return auc
    
    def get_coefficients(self):
        return {
            'intercept': self.model.intercept_[0],
            'coefficients': self.model.coef_[0]
        }
