"""
Improved KNN Module
Implementation of the improved KNN from the base paper "DDoS Attack Detection 
Method Based on Improved KNN With the Degree of DDoS Attack in Software-Defined Networks".

The key improvement incorporates the degree-of-DDoS-attack metric into the 
KNN classification logic through:
1. Degree-weighted distance metric: d' = d / (degree + ε)
2. Degree-weighted voting: higher attack degree → more influence on classification

This module bridges the standard KNN and the proposed method.
"""

import numpy as np
from collections import Counter
from typing import Optional, Dict, List
try:
    from .knn_baseline import StandardKNN, check_is_fitted
except ImportError:
    from knn_baseline import StandardKNN, check_is_fitted


class ImprovedKNN(StandardKNN):
    """
    Improved KNN from the base paper incorporating degree-of-attack metric.
    
    The improvement over standard KNN:
    - Distance modification: divide distance by (degree + ε), so flows with
      higher attack degree have SHORTER effective distance (more influential)
    - Weighted voting: combine inverse distance with attack degree weighting
    
    Formally:
    - For each test sample x, compute Euclidean/Manhattan distance to all
      training samples x_i: d(x, x_i)
    - Modify distance: d'_i = d(x, x_i) / (degree_i + ε)
      where degree_i is the degree-of-attack for training sample i
    - Select k neighbors with smallest d'_i (modified distances)
    - Weighted voting: vote_weight = (1 / (d'_i + ε)) × degree_i
      Higher degree + closer distance = stronger voting influence
    """
    
    def __init__(self, k: int = 5, metric: str = 'euclidean',
                 weights: str = 'attack_degree'):
        """
        Initialize the improved KNN.
        
        Parameters
        ----------
        k : int
            Number of neighbors
        metric : str
            Distance metric: 'euclidean' or 'manhattan'
        weights : str
            Voting scheme: 'uniform', 'distance', or 'attack_degree'
        """
        super().__init__(k=k, metric=metric, weights=weights)
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            degrees: np.ndarray) -> 'ImprovedKNN':
        """
        Fit the improved KNN model.
        
        Parameters
        ----------
        X : np.ndarray
            Training feature matrix
        y : np.ndarray
            Training labels (0=normal, 1=attack)
        degrees : np.ndarray
            Degree-of-attack per training sample, normalized to [0, 1]
        
        Returns
        -------
        self : ImprovedKNN
            Fitted classifier
        """
        # Call parent fit to store data
        super().fit(X, y, degrees)
        return self
    
    def _compute_distances(self, sample: np.ndarray) -> np.ndarray:
        """
        Compute degree-modified distances from sample to all training examples.
        
        The base paper's key improvement: d'_i = d(x, x_i) / (degree_i + ε)
        
        This means: training samples with higher attack degree get SHORTER
        modified distance, making them more likely to be among the k nearest
        neighbors and thus more influential in classification.
        """
        # Compute standard distances
        if self.metric == 'euclidean':
            std_distances = np.sqrt(np.sum((self.X_train - sample) ** 2, axis=1))
        elif self.metric == 'manhattan':
            std_distances = np.sum(np.abs(self.X_train - sample), axis=1)
        else:
            std_distances = np.sqrt(np.sum((self.X_train - sample) ** 2, axis=1))
        
        # Apply degree modification: d' = d / (degree + ε)
        # Higher degree → smaller d' → more influential
        epsilon = 1e-8
        if self.degrees_train is not None:
            modified_distances = std_distances / (self.degrees_train + epsilon)
        else:
            # Fall back to standard distances if no degrees provided
            modified_distances = std_distances
        
        return modified_distances
    
    def _weighted_vote(self, k_labels: np.ndarray, k_degrees: Optional[np.ndarray],
                       k_distances: np.ndarray) -> int:
        """
        Improved voting: combine inverse modified distance with attack degree.
        
        The base paper's weighting scheme:
        - Modified distance d'_i = d_i / (degree_i + ε) already encodes the
          degree effect in neighbor selection
        - Additional voting weight: (1 / (d'_i + ε)) × degree_i
          = (degree_i + ε) / d_i × degree_i ≈ degree_i² / d_i
        
        This dual weighting ensures that neighbors that are both close (small d)
        AND have high attack degree exert the strongest influence.
        """
        if self.weights == 'uniform':
            vote_counts = Counter(k_labels)
            most_common = vote_counts.most_common(1)[0]
            return most_common[0]
        
        elif self.weights == 'distance':
            epsilon = 1e-8
            # Standard inverse-distance weighting
            weights = 1.0 / (k_distances + epsilon)
            vote_sum = {}
            for label, weight in zip(k_labels, weights):
                vote_sum[label] = vote_sum.get(label, 0.0) + weight
            return max(vote_sum, key=vote_sum.get)
        
        elif self.weights == 'attack_degree':
            # Base paper's improved voting scheme
            if k_degrees is None:
                # Fall back to uniform
                vote_counts = Counter(k_labels)
                most_common = vote_counts.most_common(1)[0]
                return most_common[0]
            
            epsilon = 1e-8
            # Get modified distances (already computed by parent's _compute_distances)
            # k_distances here are the MODIFIED distances d' = d / (degree + ε)
            # Voting weight: combine proximity and degree
            # weight = (1 / (d' + ε)) × degree = (degree + ε) / (d / (degree + ε)) × degree
            # Simplified: degree² / (d + ε°) approximately
            
            # Since k_distances are already modified (d'), we use:
            # weight = degree² / (d' + ε) = degree² / (d/(degree+ε) + ε)
            # For simplicity and the base paper's intent: weight ∝ degree / d'
            
            # Actually, the modified distances already incorporate degree.
            # The voting should just use the modified distances for proximity,
            # and degree as an additional multiplier.
            
            # Base paper scheme: weight = degree × (1 / modified_distance)
            modified_dists = k_distances  # Already d' = d/(degree+ε)
            degree_vals = k_degrees + epsilon
            
            # Combined weight: degree proximity × degree influence
            # Smaller modified distance + higher degree = higher weight
            weights = degree_vals / (modified_dists + epsilon)
            
            vote_sum = {}
            for label, weight in zip(k_labels, weights):
                vote_sum[label] = vote_sum.get(label, 0.0) + weight
            return max(vote_sum, key=vote_sum.get)
        
        else:
            # Uniform fallback
            vote_counts = Counter(k_labels)
            most_common = vote_counts.most_common(1)[0]
            return most_common[0]
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using the improved KNN.
        
        Returns probability estimates incorporating the degree-of-attack
        weighting scheme from the base paper.
        """
        check_is_fitted(self)
        n_samples = X.shape[0]
        proba = np.zeros((n_samples, 2))
        
        for i in range(n_samples):
            proba[i] = self._predict_sample_proba(X[i:i+1])
        
        return proba
    
    def _predict_sample_proba(self, sample: np.ndarray) -> np.ndarray:
        """Predict probability for a single sample with degree weighting."""
        # Compute modified distances
        distances = self._compute_distances(sample)
        
        # Get k nearest neighbor indices
        k_indices = np.argsort(distances)[:self.k]
        
        # Get labels, degrees, and distances of k nearest neighbors
        k_labels = self.y_train[k_indices]
        k_degrees = self.degrees_train[k_indices] if self.degrees_train is not None else None
        k_distances = distances[k_indices]
        
        # Weighted voting with degree
        votes = self._weighted_vote(k_labels, k_degrees, k_distances)
        
        # Return as probability: [P(normal), P(attack)]
        # If attack is predicted class, attack_probability = vote score for attack
        # We compute scores for both classes
        normal_score = 0.0
        attack_score = 0.0
        
        if self.weights == 'attack_degree' and k_degrees is not None:
            epsilon = 1e-8
            modified_dists = distances[k_indices]
            degree_vals = k_degrees + epsilon
            
            for label, md, dw in zip(k_labels, modified_dists, degree_vals):
                if label == 0:
                    normal_score += 1.0 / (md + epsilon)  # inverse modified distance
                else:
                    attack_score += 1.0 / (md + epsilon)
        else:
            # Count based on weighted voting
            # Re-compute the vote scores
            if self.weights == 'uniform':
                normal_score = int(np.sum(k_labels == 0))
                attack_score = int(np.sum(k_labels == 1))
            elif self.weights == 'distance':
                epsilon = 1e-8
                weights = 1.0 / (k_distances + epsilon)
                for label, weight in zip(k_labels, weights):
                    if label == 0:
                        normal_score += weight
                    else:
                        attack_score += weight
            elif self.weights == 'attack_degree':
                epsilon = 1e-8
                # Use modified distances and degrees
                for label, md, dd in zip(k_labels, k_distances, k_degrees):
                    cw = dd / (md + epsilon)  # combined weight
                    if label == 0:
                        normal_score += cw
                    else:
                        attack_score += cw
            else:
                normal_score = int(np.sum(k_labels == 0))
                attack_score = int(np.sum(k_labels == 1))
        
        total = normal_score + attack_score
        if total == 0:
            return np.array([0.5, 0.5])
        return np.array([normal_score / total, attack_score / total])


# Convenience function for quick setup
def create_improved_knn(k: int = 5, metric: str = 'euclidean',
                        weights: str = 'attack_degree') -> ImprovedKNN:
    """
    Create and configure an ImprovedKNN instance.
    
    Parameters
    ----------
    k : int
        Number of neighbors
    metric : str
        Distance metric
    weights : str
        Voting scheme
    
    Returns
    -------
    ImprovedKNN
        Configured classifier instance
    """
    return ImprovedKNN(k=k, metric=metric, weights=weights)


if __name__ == "__main__":
    # Quick test: improved KNN with degree-of-attack
    print("Testing improved KNN (base paper method)...")
    
    np.random.seed(42)
    n_train = 500
    n_test = 100
    n_features = 10
    
    # Training data
    X_train = np.random.randn(n_train, n_features)
    X_train = np.abs(X_train)
    # Attack class (last 20%) has higher feature values
    X_train[200:, :] *= 3
    
    y_train = np.array([0] * 400 + [1] * 100)
    
    # Degree-of-attack per training sample (normalized to [0, 1])
    # Normal flows: low degree; Attack flows: high degree
    degrees_train = np.zeros(n_train)
    degrees_train[200:] = np.random.uniform(0.5, 1.0, 100)  # Attack samples have high degree
    
    # Test data
    X_test = np.random.randn(n_test, n_features)
    X_test = np.abs(X_test)
    X_test[:80, :] *= 3
    X_test[80:, :] *= 1
    
    y_test = np.array([1] * 80 + [0] * 20)
    
    # Test degrees for test data (for degree-modified computation)
    degrees_test = np.zeros(n_test)
    degrees_test[:80] = np.random.uniform(0.5, 1.0, 80)
    degrees_test[80:] = np.random.uniform(0.1, 0.4, 20)
    
    # Train improved KNN
    knn_imp = ImprovedKNN(k=5, metric='euclidean', weights='attack_degree')
    knn_imp.fit(X_train, y_train, degrees_train)
    
    # Predict
    y_pred = knn_imp.predict(X_test)
    y_proba = knn_imp.predict_proba(X_test)
    
    # Evaluate
    accuracy = np.sum(y_pred == y_test) / len(y_test)
    print(f"✓ Improved KNN predictions: accuracy={accuracy:.3f}")
    print(f"  Predicted: {dict(Counter(y_pred))}")
    print(f"  True: {dict(Counter(y_test))}")
    print(f"  Probabilities shape: {y_proba.shape}")
    print(f"  Sample prob: {y_proba[0]}")
    print(f"  Degree range in train: [{degrees_train.min():.3f}, {degrees_train.max():.3f}]")