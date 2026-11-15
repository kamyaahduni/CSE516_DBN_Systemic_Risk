"""
Belief Propagation inference algorithm
"""

import numpy as np
from scipy.special import expit


class BeliefPropagation:
    
    def __init__(self, bn, params, feature_names):
        self.bn = bn
        self.params = params
        self.feature_names = feature_names
    
    def infer(self, X, query_idx=0, evidence=None):
        predictions = []
        
        for sample_idx in range(X.shape[0]):
            sample = X[sample_idx]
            messages = self._forward_propagation(sample)
            query_node = self.feature_names[query_idx] if query_idx < len(self.feature_names) else self.feature_names[0]
            posterior = self._compute_posterior(query_node, messages)
            prob = expit(posterior)
            predictions.append(prob)
        
        return np.array(predictions)
    
    def _forward_propagation(self, sample):
        messages = {}
        
        for node in self.feature_names:
            parents = self.bn.get_parents(node)
            
            if len(parents) == 0:
                node_idx = self.feature_names.index(node)
                messages[node] = sample[node_idx]
            else:
                node_idx = self.feature_names.index(node)
                parent_indices = [self.feature_names.index(p) for p in parents]
                
                params = self.params.get(node, {})
                coef = params.get('coefficient', 0)
                parent_coef = params.get('parents_coef', [0] * len(parents))
                
                weighted_sum = coef + sum(parent_coef[i] * sample[parent_indices[i]] 
                                         for i in range(len(parents)))
                messages[node] = weighted_sum
        
        return messages
    
    def _compute_posterior(self, query_node, messages):
        if query_node in messages:
            return messages[query_node]
        else:
            return np.mean(list(messages.values())) if messages else 0.0
    
    def predict_proba(self, X):
        return self.infer(X)
