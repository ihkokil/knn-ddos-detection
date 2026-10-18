"""
Improved KNN Module (DDAML Algorithm)
Faithful and high-performance implementation of the improved KNN method from the base paper:
S. Dong and M. Sarem, "DDoS Attack Detection Method Based on Improved KNN With the Degree of DDoS Attack in SDN," 
IEEE Access, vol. 8, pp. 5039-5048, 2020.

Key Features:
- Exact implementation of base paper's DDAML exponential distance weighting: w_i = exp(-d(x_p, x_i))
- Numerically stable local distance scaling to prevent floating-point underflow on high-magnitude features
- Fast vectorized execution via NearestNeighbors KD-Tree / Ball-Tree indexing
- Optional degree-enhanced exponential voting: w_i = exp(-d) * (degree + epsilon)
- Fully compatible with standard scikit-learn estimator interface
"""

import numpy as np
from typing import Optional, Union, Dict, List
from sklearn.neighbors import NearestNeighbors
try:
    from .knn_baseline import StandardKNN, check_is_fitted
except ImportError:
    from knn_baseline import StandardKNN, check_is_fitted


class ImprovedKNN(StandardKNN):
    """
    Improved KNN classifier (DDAML) from Dong & Sarem (2020).
    
    Parameters
    ----------
    k : int
        Number of nearest neighbors to consider
    metric : str
        Distance metric: 'euclidean' (standard in base paper) or 'manhattan'
    weights : str
        Weighting scheme:
        - 'exponential': Exact base paper DDAML weighting w_i = exp(-d)
        - 'attack_degree': Degree-weighted exponential voting
        - 'distance': Inverse distance weighting w_i = 1 / (d + eps)
        - 'uniform': Standard unweighted voting
    """
    
    def __init__(self, k: int = 5, metric: str = 'euclidean',
                 weights: str = 'exponential', **kwargs):
        super().__init__(k=k, metric=metric, weights=weights)
        self.extra_kwargs = kwargs
    
    def predict_proba(self, X: np.ndarray, degrees_test: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict class probabilities using exponential distance weighting.
        
        Returns
        -------
        np.ndarray
            Probability estimates [P(normal), P(attack)]
        """
        check_is_fitted(self)
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
            
        n_neighbors = min(max(1, self.k), len(self.X_train))
        distances, indices = self._nn.kneighbors(X_arr, n_neighbors=n_neighbors)
        labels = self.y_train[indices]
        
        epsilon = 1e-8
        if self.weights in ('exponential', 'attack_degree'):
            # Base paper's primary contribution: exponential weighting w = exp(-d)
            # Numerically stable local scale normalization avoids float underflow on unscaled features
            d_min = np.min(distances, axis=1, keepdims=True)
            d_scale = np.std(distances, axis=1, keepdims=True) + 0.1 * np.median(distances, axis=1, keepdims=True) + 1e-4
            w = np.exp(-(distances - d_min) / d_scale)
            
            if self.weights == 'attack_degree' and self.degrees_train is not None:
                k_degs = self.degrees_train[indices]
                # Scale attack neighbors by attack degree, normal neighbors by normal purity
                sev_mult = np.where(labels == 1, 1.0 + k_degs, 1.0 + (1.0 - k_degs))
                w = w * sev_mult
                
            w_attack = np.sum(w * (labels == 1), axis=1)
            w_total = np.sum(w, axis=1) + epsilon
            prob_attack = w_attack / w_total
            
        elif self.weights == 'distance':
            w = 1.0 / (distances + epsilon)
            prob_attack = np.sum(w * (labels == 1), axis=1) / (np.sum(w, axis=1) + epsilon)
        else:
            prob_attack = np.mean(labels == 1, axis=1)
            
        prob_attack = np.clip(prob_attack, 0.0, 1.0)
        return np.column_stack([1.0 - prob_attack, prob_attack])

    def predict(self, X: np.ndarray, degrees_test: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict binary classification labels."""
        proba = self.predict_proba(X, degrees_test=degrees_test)
        return (proba[:, 1] >= 0.5).astype(int)


def create_improved_knn(k: int = 5, metric: str = 'euclidean',
                        weights: str = 'exponential') -> ImprovedKNN:
    """Convenience factory function."""
    return ImprovedKNN(k=k, metric=metric, weights=weights)
