"""
Classifier Module
Unified classifier interface wrapping all three KNN variants:
- Standard KNN (baseline)
- Improved KNN (base paper)
- Proposed KNN (my contribution)

Provides method selection by config, prediction output with labels and confidence,
and consistent API across all variants.
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple
from src.data_loader import load_dataset
from src.knn_baseline import StandardKNN, check_is_fitted
from src.knn_improved import ImprovedKNN
from src.knn_proposed import ProposedKNN


class KNNClassifier:
    """
    Unified classifier interface for DDoS detection using KNN variants.
    
    Supports three classification methods:
    1. 'baseline' - Standard KNN (no degree-of-attack integration)
    2. 'improved' - Improved KNN from base paper (degree-weighted)
    3. 'proposed' - My proposed KNN variant (adaptive K + dual weighting)
    
    Provides consistent fit/predict/proba interface, method selection
    via configuration, and prediction output with labels and confidence scores.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the KNN classifier with configuration.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary with keys:
            - 'method': 'baseline', 'improved', or 'proposed'
            - 'k': int, number of neighbors
            - 'metric': 'euclidean' or 'manhattan'
            - 'weights': voting scheme (variant-specific)
            - 'adaptive_k': bool (proposed method only)
            - 'degree_method': 'base_paper' or 'proposed' (for degree computation)
        """
        if config is None:
            config = {
                'method': 'proposed',
                'k': 5,
                'metric': 'euclidean',
                'weights': 'dual',
                'adaptive_k': True,
                'degree_method': 'proposed'
            }
        self.config = config
        self.method = config.get('method', 'proposed')
        self.trained = False
        
        # Initialize the appropriate KNN variant
        self._init_knn_model()
    
    def _init_knn_model(self) -> None:
        """Initialize the KNN model based on the configured method."""
        self.method = self.config.get('method', 'proposed')
        k = self.config.get('k', 5)
        metric = self.config.get('metric', 'euclidean')
        weights = self.config.get('weights', 'dual')
        adaptive_k = self.config.get('adaptive_k', True)
        degree_method = self.config.get('degree_method', 'proposed')
        
        if self.method == 'baseline':
            self.knn_model = StandardKNN(k=k, metric=metric, weights=weights)
            self.model_name = 'StandardKNN'
        elif self.method == 'improved':
            self.knn_model = ImprovedKNN(k=k, metric=metric, weights=weights)
            self.model_name = 'ImprovedKNN'
        elif self.method == 'proposed':
            self.knn_model = ProposedKNN(k=k, metric=metric, weights=weights,
                                         adaptive_k=adaptive_k)
            self.model_name = 'ProposedKNN'
        else:
            raise ValueError(f"Unknown KNN method: {self.method}")
        
        # Degree-of-attack computation settings
        self.degree_method = degree_method
        self.trained = False
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            degrees: Optional[np.ndarray] = None) -> 'KNNClassifier':
        """
        Fit the KNN classifier on training data.
        
        Parameters
        ----------
        X : np.ndarray
            Training feature matrix (n_samples, n_features)
        y : np.ndarray
            Training labels (0=normal, 1=attack)
        degrees : np.ndarray, optional
            Degree-of-attack per training sample.
            Required for 'improved' and 'proposed' methods.
        
        Returns
        -------
        self : KNNClassifier
            Fitted classifier
        """
        self.method = self.config.get('method', 'proposed')
        
        if self.method == 'baseline':
            # Baseline doesn't need degrees
            self.knn_model.fit(X, y)
        elif self.method in ('improved', 'proposed'):
            if degrees is None:
                raise ValueError(
                    f"{self.method} KNN requires degree-of-attack values. "
                    "Compute degrees using DegreeOfAttackComputer or provide them explicitly."
                )
            # Call appropriate fit method
            if self.method == 'improved':
                self.knn_model.fit(X, y, degrees)
            elif self.method == 'proposed':
                self.knn_model.fit(X, y, degrees)
        
        self.trained = True
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
        check_is_trained(self)
        
        if self.method == 'baseline':
            return self.knn_model.predict(X)
        elif self.method in ('improved', 'proposed'):
            return self.knn_model.predict(X)
    
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
            Probability estimates, shape (n_samples, 2)
            Columns: [P(normal), P(attack)]
        """
        check_is_trained(self)
        
        if self.method == 'baseline':
            return self.knn_model.predict_proba(X)
        elif self.method in ('improved', 'proposed'):
            return self.knn_model.predict_proba(X)
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute raw decision scores for classification.
        
        Returns the margin between the winning and runner-up class scores.
        Higher positive values indicate more confidence in the predicted class.
        
        Parameters
        ----------
        X : np.ndarray
            Test feature matrix
        
        Returns
        -------
        np.ndarray
            Decision scores, shape (n_samples,)
        """
        check_is_trained(self)
        proba = self.predict_proba(X)
        # Margin = |P(attack) - P(normal)| = |2*P(attack) - 1|
        # Positive = attack predicted, negative = normal predicted
        margin = 2 * proba[:, 1] - 1
        return margin
    
    def get_degree_values(self, X: np.ndarray, 
                          feature_names: Optional[List[str]] = None,
                          X_df: Optional[Any] = None) -> np.ndarray:
        """
        Compute degree-of-attack values for test data.
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Test data features
        feature_names : list, optional
            Feature names if X is numpy array
        X_df : pd.DataFrame, optional
            DataFrame if X is already a DataFrame
        
        Returns
        -------
        np.ndarray
            Degree-of-attack values in [0, 1]
        """
        if X_df is not None:
            from src.degree_of_attack import DegreeOfAttackComputer
            computer = DegreeOfAttackComputer(
                method=self.degree_method,
                threshold_type='fixed',
                fixed_threshold=0.5
            )
            return computer.compute_degree(X_df, feature_names)
        elif isinstance(X, np.ndarray) and feature_names is not None:
            from src.degree_of_attack import DegreeOfAttackComputer
            computer = DegreeOfAttackComputer(
                method=self.degree_method,
                threshold_type='fixed',
                fixed_threshold=0.5
            )
            return computer.compute_degree(X, feature_names)
        else:
            # Return zeros if cannot compute
            return np.zeros(len(X))
    
    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """
        Evaluate classifier performance.
        
        Parameters
        ----------
        X : np.ndarray
            Test features
        y_true : np.ndarray
            True labels
        
        Returns
        -------
        dict
            Evaluation metrics: accuracy, precision, recall, f1, auc_roc approximation
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        # Accuracy
        accuracy = np.sum(y_pred == y_true) / len(y_true)
        
        # Precision (attack class = 1)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        precision = tp / (tp + fp + 1e-10) if (tp + fp) > 0 else 0.0
        
        # Recall (attack class = 1)
        fn = np.sum((y_pred == 0) & (y_true == 1))
        recall = tp / (tp + fn + 1e-10) if (tp + fn) > 0 else 0.0
        
        # F1-score
        f1 = 2 * precision * recall / (precision + recall + 1e-10) if (precision + recall) > 0 else 0.0
        
        # AUC-ROC approximation (using decision function and true labels)
        # Simple trapezoidal approximation with single operating point
        decision = self.decision_function(X)
        # Sort by decision score and compute ROC AUC
        # For simplicity, use the probability of attack as proxy
        auc_proxy = float(np.mean(y_proba[y_true == 1, 1]) - np.mean(y_proba[y_true == 0, 1])) / 2 + 0.5
        auc_roc = np.clip(auc_proxy, 0.0, 1.0)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc_roc': auc_roc,
            'decision_margin_mean': float(np.mean(np.abs(self.decision_function(X)))),
            'predicted_attack_rate': float(np.mean(y_pred))
        }
    
    def predict_with_degree(self, X: np.ndarray, 
                            feature_names: Optional[List[str]] = None,
                            X_df: Optional[Any] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict labels, probabilities, and degree-of-attack values.
        
        Returns three arrays for downstream analysis:
        1. Predicted labels
        2. Prediction probabilities (P(attack))
        3. Degree-of-attack values per sample
        
        Parameters
        ----------
        X : np.ndarray
            Test features
        feature_names : list, optional
            Feature names for degree computation
        X_df : pd.DataFrame, optional
            DataFrame for degree computation
        
        Returns
        -------
        tuple
            (y_pred, y_proba, degrees)
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        # Compute degrees
        if X_df is not None:
            degrees = self.get_degree_values(X_df, feature_names)
        elif isinstance(X, np.ndarray) and feature_names is not None:
            degrees = self.get_degree_values(X, feature_names)
        else:
            degrees = self.knn_model.degrees_train[:len(X)] if hasattr(self.knn_model, 'degrees_train') and self.knn_model.degrees_train is not None else \
                np.zeros(len(X))
        
        return y_pred, y_proba, degrees


def check_is_trained(obj) -> None:
    """Check if classifier has been fitted."""
    if not obj.trained or obj.knn_model is None or obj.knn_model.X_train is None:
        raise ValueError("KNN classifier must be fitted before calling predict/predict_proba")


# Convenience function for creating classifier from config
def create_classifier_from_config(config: Dict[str, Any] = None) -> KNNClassifier:
    """
    Create a KNNClassifier instance from configuration.
    
    Parameters
    ----------
    config : dict, optional
        Classifier configuration
    
    Returns
    -------
    KNNClassifier
        Configured classifier instance
    """
    if config is None:
        config = {
            'method': 'proposed',
            'k': 5,
            'metric': 'euclidean',
            'weights': 'dual',
            'adaptive_k': True,
            'degree_method': 'proposed'
        }
    return KNNClassifier(config=config)


if __name__ == "__main__":
    # Quick test: unified classifier interface
    print("Testing unified KNN classifier interface...")
    
    np.random.seed(42)
    n_train = 500
    n_test = 100
    n_features = 10
    
    # Training data
    X_train = np.random.randn(n_train, n_features)
    X_train = np.abs(X_train)
    X_train[200:, :] *= 3
    
    y_train = np.array([0] * 400 + [1] * 100)
    
    # Degrees for improved/proposed methods
    degrees_train = np.zeros(n_train)
    degrees_train[200:] = np.random.uniform(0.5, 1.0, 100)
    
    # Test data
    X_test = np.random.randn(n_test, n_features)
    X_test = np.abs(X_test)
    X_test[:80, :] *= 3
    
    y_test = np.array([1] * 80 + [0] * 20)
    
    # Test all three methods
    for method in ['baseline', 'improved', 'proposed']:
        config = {
            'method': method,
            'k': 5,
            'metric': 'euclidean',
            'weights': 'uniform' if method == 'baseline' else 'attack_degree' 
                       if method == 'improved' else 'dual',
            'adaptive_k': True if method == 'proposed' else False,
            'degree_method': 'proposed'
        }
        
        classifier = KNNClassifier(config=config)
        classifier.fit(X_train, y_train, degrees_train if method != 'baseline' else None)
        
        y_pred = classifier.predict(X_test)
        y_proba = classifier.predict_proba(X_test)
        
        accuracy = np.sum(y_pred == y_test) / len(y_test)
        print(f"✓ {method:12s}: accuracy={accuracy:.3f}, "
              f"P(attack) mean={np.mean(y_proba[:, 1]):.3f}")
    
    # Test evaluate method
    print("\n--- Evaluation ---")
    config = {'method': 'proposed', 'k': 5, 'metric': 'euclidean',
              'weights': 'dual', 'adaptive_k': True, 'degree_method': 'proposed'}
    classifier = KNNClassifier(config=config)
    classifier.fit(X_train, y_train, degrees_train)
    
    metrics = classifier.evaluate(X_test, y_test)
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.3f}")