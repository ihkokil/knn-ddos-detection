"""
KNN Baseline Module
Clean implementation of standard K-Nearest Neighbors for DDoS detection.

Provides:
- fit/predict methods
- Configurable K value and distance metric
- Prediction probability/confidence output
- Standard Euclidean and Manhattan distances
- Weighted distance variant (inverse distance weighting)
"""

import numpy as np
from collections import Counter
from typing import Optional, Union, Dict, List


class StandardKNN:
    """
    Standard K-Nearest Neighbors classifier for DDoS detection.
    
    Parameters
    ----------
    k : int
        Number of nearest neighbors to consider (1 <= k <= n_train)
    metric : {'euclidean', 'manhattan', 'weighted'}
        Distance metric to use for neighbor selection
    weights : {'uniform', 'distance', 'attack_degree'}
        Weighting scheme for voting:
        - 'uniform': all neighbors vote equally
        - 'distance': closer neighbors have higher voting weight
        - 'attack_degree': neighbors with higher attack degree have more influence
    """
    
    def __init__(self, k: int = 5, metric: str = 'euclidean', 
                 weights: str = 'uniform'):
        """
        Initialize the KNN classifier.
        
        Parameters
        ----------
        k : int
            Number of neighbors
        metric : str
            Distance metric
        weights : str
            Voting weight scheme
        """
        self.k = k
        self.metric = metric
        self.weights = weights
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.degrees_train: Optional[np.ndarray] = None  # For improved/variant versions
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            degrees: Optional[np.ndarray] = None) -> 'StandardKNN':
        """
        Fit the KNN model using training data.
        
        Parameters
        ----------
        X : np.ndarray
            Training feature matrix (n_samples, n_features)
        y : np.ndarray
            Training labels (0=normal, 1=attack)
        degrees : np.ndarray, optional
            Degree-of-attack values per training sample (for improved variants)
        
        Returns
        -------
        self : StandardKNN
            Fitted classifier
        """
        self.X_train = X
        self.y_train = y
        if degrees is not None:
            self.degrees_train = degrees
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.
        
        Parameters
        ----------
        X : np.ndarray
            Test feature matrix (n_samples, n_features)
        
        Returns
        -------
        np.ndarray
            Predicted labels (0=normal, 1=attack)
        """
        check_is_fitted(self)
        return np.asarray([self._predict_sample(row) for row in X], dtype=int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples in X.
        
        Parameters
        ----------
        X : np.ndarray
            Test feature matrix (n_samples, n_features)
        
        Returns
        -------
        np.ndarray
            Probability estimates for each class [P(normal), P(attack)]
        """
        check_is_fitted(self)
        n_samples = X.shape[0]
        proba = np.zeros((n_samples, 2))
        
        for i in range(n_samples):
            proba[i] = self._predict_sample_proba(X[i:i+1])
        
        return proba
    
    def _predict_sample(self, sample: np.ndarray) -> int:
        """Predict label for a single sample."""
        # Compute distances to all training samples
        distances = self._compute_distances(sample)
        
        # Get indices of k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # Get labels and optionally degrees of k nearest neighbors
        k_labels = self.y_train[k_indices]
        k_degrees = None
        if self.degrees_train is not None:
            k_degrees = self.degrees_train[k_indices]
        
        # Perform weighted voting
        predicted_label = self._weighted_vote(k_labels, k_degrees, distances[k_indices])
        return predicted_label
    
    def _predict_sample_proba(self, sample: np.ndarray) -> np.ndarray:
        """Predict probability for a single sample."""
        distances = self._compute_distances(sample)
        k_indices = np.argsort(distances)[:self.k]
        
        k_labels = self.y_train[k_indices]
        k_degrees = None
        if self.degrees_train is not None:
            k_degrees = self.degrees_train[k_indices]
        
        # Get vote counts with weighting
        votes = self._weighted_vote_with_proba(k_labels, k_degrees, distances)
        return votes
    
    def _compute_distances(self, sample: np.ndarray) -> np.ndarray:
        """Compute distances from sample to all training examples."""
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((self.X_train - sample) ** 2, axis=1))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(self.X_train - sample), axis=1)
        elif self.metric == 'weighted':
            # Distance weighted by feature magnitude (inverse of feature values)
            # Simple: use inverse Euclidean distance
            euclidean = np.sqrt(np.sum((self.X_train - sample) ** 2, axis=1))
            # Avoid division by zero
            epsilon = 1e-8
            return euclidean / (epsilon + np.maximum(1, np.abs(self.X_train).sum(axis=1)))
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def _weighted_vote(self, k_labels: np.ndarray, k_degrees: Optional[np.ndarray],
                       k_distances: np.ndarray) -> int:
        """
        Perform weighted voting among k nearest neighbors.
        
        Parameters
        ----------
        k_labels : np.ndarray
            Labels of k neighbors
        k_degrees : np.ndarray, optional
            Degrees of k neighbors (for attack-degree weighting)
        k_distances : np.ndarray
            Distances to k neighbors
        
        Returns
        -------
        int
            Predicted label (0=normal, 1=attack)
        """
        if self.weights == 'uniform':
            # Simple majority vote
            vote_counts = Counter(k_labels)
            most_common = vote_counts.most_common(1)[0]
            return most_common[0]
        
        elif self.weights == 'distance':
            # Weight by inverse distance: closer = more weight
            # Add epsilon to avoid division by zero
            epsilon = 1e-8
            weights = 1.0 / (k_distances + epsilon)
            # Weighted vote: sum of weights per class
            vote_sum = {}
            for label, weight in zip(k_labels, weights):
                vote_sum[label] = vote_sum.get(label, 0.0) + weight
            return max(vote_sum, key=vote_sum.get)
        
        elif self.weights == 'attack_degree':
            # Weight by attack degree: higher degree = more influential
            if k_degrees is None:
                # Fall back to uniform voting
                vote_counts = Counter(k_labels)
                most_common = vote_counts.most_common(1)[0]
                return most_common[0]
            
            # Combined weight: inverse distance × attack degree
            epsilon = 1e-8
            distance_weights = 1.0 / (k_distances + epsilon)
            degree_weights = k_degrees + epsilon  # Ensure positive
            combined_weights = distance_weights * degree_weights
            
            vote_sum = {}
            for label, cw in zip(k_labels, combined_weights):
                vote_sum[label] = vote_sum.get(label, 0.0) + cw
            return max(vote_sum, key=vote_sum.get)
        
        else:
            # Uniform fallback
            vote_counts = Counter(k_labels)
            most_common = vote_counts.most_common(1)[0]
            return most_common[0]
    
    def _weighted_vote_with_proba(self, k_labels: np.ndarray,
                                  k_degrees: Optional[np.ndarray],
                                  k_distances: np.ndarray) -> np.ndarray:
        """
        Weighted voting that returns probability estimates.
        
        Returns [P(normal), P(attack)] based on weighted vote scores.
        """
        if self.weights == 'uniform':
            # Count votes for each class
            normal_votes = int(np.sum(k_labels == 0))
            attack_votes = int(np.sum(k_labels == 1))
            total = normal_votes + attack_votes
            if total == 0:
                return np.array([0.5, 0.5])
            return np.array([normal_votes / total, attack_votes / total])
        
        elif self.weights == 'distance':
            epsilon = 1e-8
            weights = 1.0 / (k_distances + epsilon)
            normal_weight = 0.0
            attack_weight = 0.0
            for label, weight in zip(k_labels, weights):
                if label == 0:
                    normal_weight += weight
                else:
                    attack_weight += weight
            total = normal_weight + attack_weight
            if total == 0:
                return np.array([0.5, 0.5])
            return np.array([normal_weight / total, attack_weight / total])
        
        elif self.weights == 'attack_degree':
            if k_degrees is None:
                normal_votes = int(np.sum(k_labels == 0))
                attack_votes = int(np.sum(k_labels == 1))
                total = normal_votes + attack_votes
                if total == 0:
                    return np.array([0.5, 0.5])
                return np.array([normal_votes / total, attack_votes / total])
            
            epsilon = 1e-8
            distance_weights = 1.0 / (k_distances + epsilon)
            degree_weights = k_degrees + epsilon
            combined_weights = distance_weights * degree_weights
            
            normal_weight = 0.0
            attack_weight = 0.0
            for label, cw in zip(k_labels, combined_weights):
                if label == 0:
                    normal_weight += cw
                else:
                    attack_weight += cw
            total = normal_weight + attack_weight
            if total == 0:
                return np.array([0.5, 0.5])
            return np.array([normal_weight / total, attack_weight / total])
        
        else:
            normal_votes = int(np.sum(k_labels == 0))
            attack_votes = int(np.sum(k_labels == 1))
            total = normal_votes + attack_votes
            if total == 0:
                return np.array([0.5, 0.5])
            return np.array([normal_votes / total, attack_votes / total])


def check_is_fitted(obj) -> None:
    """Check if the object has been fitted, raise ValueError if not."""
    if obj.X_train is None or obj.y_train is None:
        raise ValueError("KNN model must be fitted before calling predict/predict_proba")


# Example usage and quick test
if __name__ == "__main__":
    print("Testing standard KNN baseline...")
    
    # Create synthetic dataset
    np.random.seed(42)
    n_train = 500
    n_test = 100
    n_features = 10
    
    # Training data: 80% normal, 20% attack
    X_train = np.random.randn(n_train, n_features)
    X_train = np.abs(X_train)
    # Add some separation: attack features have higher values
    X_train[200:, :] *= 3  # Attack class has higher features
    
    y_train = np.array([0] * 400 + [1] * 100)  # 80% normal, 20% attack
    
    # Test data
    X_test = np.random.randn(n_test, n_features)
    X_test = np.abs(X_test)
    X_test[:80, :] *= 3  # 80% attack in test
    X_test[80:, :] *= 1  # normal
    y_test = np.array([1] * 80 + [0] * 20)
    
    # Train KNN
    knn = StandardKNN(k=5, metric='euclidean', weights='uniform')
    knn.fit(X_train, y_train)
    
    # Predict
    y_pred = knn.predict(X_test)
    y_proba = knn.predict_proba(X_test)
    
    # Evaluate
    accuracy = np.sum(y_pred == y_test) / len(y_test)
    print(f"✓ KNN predictions: accuracy={accuracy:.3f}")
    print(f"  Predicted: {dict(Counter(y_pred))}")
    print(f"  True: {dict(Counter(y_test))}")
    print(f"  Probability shape: {y_proba.shape}")
    print(f"  Sample probabilities: {y_proba[0]}")