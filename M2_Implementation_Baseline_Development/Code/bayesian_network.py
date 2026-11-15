"""
Bayesian Network graph structure and CPT storage
"""

import json
import numpy as np


class BayesianNetwork:
    """Directed acyclic graph (DAG) for Bayesian Network"""
    
    def __init__(self):
        self.nodes = set()
        self.edges = []
        self.cpts = {}
        self.adjacency = {}
    
    def add_node(self, node_name):
        self.nodes.add(node_name)
        self.adjacency[node_name] = {'parents': set(), 'children': set()}
    
    def add_edge(self, parent, child):
        if parent not in self.nodes:
            self.add_node(parent)
        if child not in self.nodes:
            self.add_node(child)
        
        self.edges.append((parent, child))
        self.adjacency[child]['parents'].add(parent)
        self.adjacency[parent]['children'].add(child)
    
    def get_parents(self, node):
        return list(self.adjacency[node]['parents'])
    
    def get_children(self, node):
        return list(self.adjacency[node]['children'])
    
    def set_cpt(self, node, cpt_dict):
        self.cpts[node] = cpt_dict
    
    def get_cpt(self, node):
        return self.cpts.get(node, None)
    
    def to_dict(self):
        return {
            'nodes': list(self.nodes),
            'edges': self.edges,
            'cpts': {k: str(v) for k, v in self.cpts.items()}
        }
    
    def to_json(self):
        return json.dumps(self.to_dict(), indent=2)
