"""
Proposed KNN Module: Bayesian-Fused Adaptive KNN (BF-AKNN)
Novel research extension uniting SDN Degree-of-Attack telemetry with Data-Plane KNN.

Key Innovations:
1. Bayesian Evidence Fusion (MY NOVEL CONTRIBUTION):
   - Formulates degree-of-attack D(x) as a calibrated prior probability P(attack) = D(x)
   - Computes local non-parametric kernel likelihoods L(x | attack) and L(x | normal)
   - Merges global telemetry with local packet metric space via Bayes' Theorem:
     P(attack | x, D) = [L(attack) * D] / [L(attack) * D + L(normal) * (1 - D)]
   - Avoids ad-hoc metric warping while preserving Euclidean metric space properties

2. Attack-Density Sensitive Adaptive K (MY NOVEL CONTRIBUTION):
   - Dynamically selects K per test query based on local neighborhood label consistency
   - In homogeneous regions (agreement >= 0.8), uses tight K = max(3, k - 2) for sharpness
   - In boundary uncertainty regions, uses wider K = k + 2 to smooth out stochastic noise

3. Dual Proximity-Severity Weighting:
   - Proximity kernel: exp(-(d - d_min) / scale)
   - Severity scale: (1 + beta * deg) for attacks, (1 + beta * (1 - deg)) for normal
"""

import numpy as np
from typing import Optional, Union, Dict, List
from sklearn.neighbors import NearestNeighbors
try:
    from .knn_improved import ImprovedKNN, check_is_fitted
except ImportError:
    from knn_improved import ImprovedKNN, check_is_fitted


class ProposedKNN(ImprovedKNN):
    """
    Bayesian-Fused Adaptive KNN (Proposed Method).
    
    Parameters
    ----------
    k : int
        Baseline number of neighbors
    metric : str
        Distance metric: 'euclidean' or 'manhattan'
    weights : str
        'dual', 'bayesian_fusion', or 'exponential'
    adaptive_k : bool
        Whether to adjust K adaptively based on neighborhood density
    beta : float
        Severity weighting coefficient (default 0.5)
    """
    
    def __init__(self, k: int = 5, metric: str = 'euclidean',
                 weights: str = 'dual', adaptive_k: bool = True,
                 beta: float = 0.5):
        super().__init__(k=k, metric=metric, weights=weights)
        self.adaptive_k = adaptive_k
        self.k_base = k
        self.beta = beta
        self.max_degree: float = 1.0
        self.min_degree: float = 0.0
        
    def fit(self, X: np.ndarray, y: np.ndarray,
            degrees: Optional[np.ndarray] = None) -> 'ProposedKNN':
        """Fit model and record attack degree statistics."""
        super().fit(X, y, degrees)
        if self.degrees_train is not None and len(self.degrees_train) > 0:
            self.max_degree = float(np.max(self.degrees_train))
            self.min_degree = float(np.min(self.degrees_train))
        else:
            self.max_degree = 1.0
            self.min_degree = 0.0
        return self

    def predict_proba(self, X: np.ndarray, degrees_test: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict class probabilities via Bayesian Evidence Fusion.
        
        Parameters
        ----------
        X : np.ndarray
            Test feature matrix (n_samples, n_features)
        degrees_test : np.ndarray, optional
            Test-sample degrees of attack from SDN monitor
            
        Returns
        -------
        np.ndarray
            Probability estimates [P(normal), P(attack)]
        """
        check_is_fitted(self)
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
            
        n_samples = len(X_arr)
        max_k = min(self.k_base + 4, len(self.X_train))
        
        # Ensure underlying indexer has enough neighbors for adaptive k
        if self._nn.n_neighbors < max_k:
            nn_metric = self.metric if self.metric in ('euclidean', 'manhattan') else 'euclidean'
            self._nn = NearestNeighbors(n_neighbors=max_k, metric=nn_metric, n_jobs=-1)
            self._nn.fit(self.X_train)
            
        distances, indices = self._nn.kneighbors(X_arr, n_neighbors=max_k)
        labels = self.y_train[indices]
        
        prob_attack = np.zeros(n_samples, dtype=np.float32)
        epsilon = 1e-8
        
        for i in range(n_samples):
            # 1. Adaptive K Selection based on neighborhood agreement
            if self.adaptive_k:
                core_labs = labels[i, :min(5, max_k)]
                agreement = np.mean(core_labs == core_labs[0])
                if agreement >= 0.8:
                    cur_k = max(3, self.k_base - 2)
                else:
                    cur_k = min(self.k_base + 2, max_k)
            else:
                cur_k = min(self.k_base, max_k)
                
            cur_dists = distances[i, :cur_k]
            cur_labs = labels[i, :cur_k]
            
            # 2. Dual Proximity-Severity Weighting (numerically stable)
            d_min = np.min(cur_dists)
            d_scale = np.std(cur_dists) + 0.1 * np.median(cur_dists) + 1e-4
            w = np.exp(-(cur_dists - d_min) / d_scale)
            
            if self.degrees_train is not None:
                k_degs = self.degrees_train[indices[i, :cur_k]]
                norm_deg = k_degs / (self.max_degree + epsilon)
                sev_mult = np.where(
                    cur_labs == 1,
                    1.0 + self.beta * norm_deg,
                    1.0 + self.beta * (1.0 - norm_deg)
                )
                w = w * sev_mult
                
            lik_att = np.sum(w * (cur_labs == 1)) + epsilon
            lik_norm = np.sum(w * (cur_labs == 0)) + epsilon
            
            # 3. Bayesian Evidence Fusion with Prior
            if degrees_test is not None:
                prior_att = np.clip(float(degrees_test[i]), 0.05, 0.95)
            elif self.degrees_train is not None:
                # Infer local prior from training exemplar neighborhood
                k_degs = self.degrees_train[indices[i, :cur_k]]
                norm_w = w / np.sum(w)
                inferred_deg = float(np.sum(norm_w * k_degs))
                prior_att = np.clip(inferred_deg / (self.max_degree + epsilon), 0.05, 0.95)
            else:
                prior_att = 0.5
                
            post_att = (lik_att * prior_att) / (lik_att * prior_att + lik_norm * (1.0 - prior_att) + epsilon)
            prob_attack[i] = np.clip(post_att, 0.0, 1.0)
            
        return np.column_stack([1.0 - prob_attack, prob_attack])

    def predict(self, X: np.ndarray, degrees_test: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict binary classification labels."""
        proba = self.predict_proba(X, degrees_test=degrees_test)
        return (proba[:, 1] >= 0.5).astype(int)


def create_proposed_knn(k: int = 5, metric: str = 'euclidean',
                        weights: str = 'dual', adaptive_k: bool = True) -> ProposedKNN:
    """Convenience factory function."""
    return ProposedKNN(k=k, metric=metric, weights=weights, adaptive_k=adaptive_k)
