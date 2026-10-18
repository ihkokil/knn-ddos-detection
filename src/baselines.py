"""
Baseline Comparison Module
Implements and evaluates baseline classifiers for comparison with KNN methods:
- Random Forest
- SVM (Support Vector Machine)
- XGBoost / LightGBM
- Neural Network (MLP)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix,
                            precision_recall_curve, auc)
import time
import warnings
warnings.filterwarnings('ignore')

# Optional gradient boosting
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available, will use sklearn GradientBoostingClassifier as fallback")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available")

# Optional calibration
try:
    from sklearn.calibration import CalibratedClassifierCV
    CALIBRATION_AVAILABLE = True
except ImportError:
    CALIBRATION_AVAILABLE = False


class BaselineClassifier:
    """Base class for baseline classifiers with unified interface."""
    
    def __init__(self, name: str, random_state: int = 42):
        self.name = name
        self.random_state = random_state
        self.model = None
        self.is_fitted = False
        self.training_time = 0.0
        self.inference_times = []
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'BaselineClassifier':
        """Fit the classifier."""
        start = time.perf_counter()
        self.model.fit(X, y, **kwargs)
        self.training_time = time.perf_counter() - start
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted:
            raise ValueError(f"{self.name} must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError(f"{self.name} must be fitted before prediction")
        
        start = time.perf_counter()
        proba = self.model.predict_proba(X)
        self.inference_times.append(time.perf_counter() - start)
        return proba
    
    def get_params(self) -> Dict:
        """Get model parameters."""
        return self.model.get_params() if self.model else {}
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance if available."""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_).mean(axis=0)
        return None


class RandomForestBaseline(BaselineClassifier):
    """Random Forest classifier with optimized defaults for DDoS detection."""
    
    def __init__(self, random_state: int = 42, n_estimators: int = 200,
                 max_depth: int = None, min_samples_split: int = 5,
                 min_samples_leaf: int = 2, class_weight: str = 'balanced',
                 n_jobs: int = -1):
        super().__init__("RandomForest", random_state)
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs
        )


class SVMClassifier(BaselineClassifier):
    """Support Vector Machine with RBF kernel optimized for DDoS detection."""
    
    def __init__(self, random_state: int = 42, C: float = 1.0,
                 kernel: str = 'rbf', gamma: str = 'scale',
                 class_weight: str = 'balanced', probability: bool = True):
        super().__init__("SVM", random_state)
        self.model = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            class_weight=class_weight,
            probability=probability,
            random_state=random_state,
            cache_size=1000
        )


class XGBoostBaseline(BaselineClassifier):
    """XGBoost classifier optimized for DDoS detection."""
    
    def __init__(self, random_state: int = 42, n_estimators: int = 200,
                 max_depth: int = 6, learning_rate: float = 0.1,
                 subsample: float = 0.8, colsample_bytree: float = 0.8,
                 scale_pos_weight: float = 1.0, n_jobs: int = -1):
        super().__init__("XGBoost", random_state)
        
        if XGBOOST_AVAILABLE:
            self.model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                scale_pos_weight=scale_pos_weight,
                random_state=random_state,
                n_jobs=n_jobs,
                eval_metric='logloss',
                verbosity=0
            )
        else:
            # Fallback to sklearn GradientBoosting
            from sklearn.ensemble import GradientBoostingClassifier
            self.model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                random_state=random_state
            )
            print("  Using GradientBoostingClassifier as XGBoost fallback")


class LightGBMBaseline(BaselineClassifier):
    """LightGBM classifier optimized for DDoS detection."""
    
    def __init__(self, random_state: int = 42, n_estimators: int = 200,
                 max_depth: int = -1, learning_rate: float = 0.1,
                 num_leaves: int = 31, class_weight: str = 'balanced',
                 n_jobs: int = -1, verbose: int = -1):
        super().__init__("LightGBM", random_state)
        
        if LIGHTGBM_AVAILABLE:
            self.model = lgb.LGBMClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                num_leaves=num_leaves,
                class_weight=class_weight,
                random_state=random_state,
                n_jobs=n_jobs,
                verbose=verbose
            )
        else:
            # Fallback to sklearn GradientBoosting
            from sklearn.ensemble import GradientBoostingClassifier
            self.model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=6 if max_depth == -1 else max_depth,
                learning_rate=learning_rate,
                random_state=random_state
            )
            print("  Using GradientBoostingClassifier as LightGBM fallback")


class MLPBaseline(BaselineClassifier):
    """Multi-Layer Perceptron (Neural Network) baseline."""
    
    def __init__(self, random_state: int = 42, hidden_layer_sizes: Tuple = (100, 50),
                 activation: str = 'relu', solver: str = 'adam',
                 alpha: float = 0.0001, learning_rate: str = 'adaptive',
                 max_iter: int = 500, early_stopping: bool = True):
        super().__init__("MLP", random_state)
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            learning_rate=learning_rate,
            max_iter=max_iter,
            early_stopping=early_stopping,
            random_state=random_state
        )


class LogisticRegressionBaseline(BaselineClassifier):
    """Logistic Regression baseline (linear model)."""
    
    def __init__(self, random_state: int = 42, C: float = 1.0,
                 penalty: str = 'l2', class_weight: str = 'balanced',
                 max_iter: int = 1000, solver: str = 'lbfgs'):
        super().__init__("LogisticRegression", random_state)
        self.model = LogisticRegression(
            C=C,
            penalty=penalty,
            class_weight=class_weight,
            max_iter=max_iter,
            solver=solver,
            random_state=random_state
        )


def create_all_baselines(random_state: int = 42, 
                         class_weights: Dict = None) -> Dict[str, BaselineClassifier]:
    """
    Create all baseline classifiers with reasonable defaults.
    
    Parameters
    ----------
    random_state : int
        Random seed
    class_weights : dict, optional
        Class weights for imbalanced data
        
    Returns
    -------
    dict
        Dictionary of classifier name -> classifier instance
    """
    # Compute scale_pos_weight for XGBoost if class_weights provided
    scale_pos_weight = 1.0
    if class_weights and 0 in class_weights and 1 in class_weights:
        scale_pos_weight = class_weights[0] / class_weights[1]
    
    baselines = {
        "RandomForest": RandomForestBaseline(
            random_state=random_state,
            n_estimators=200,
            class_weight=class_weights or 'balanced'
        ),
        "SVM": SVMClassifier(
            random_state=random_state,
            C=1.0,
            class_weight=class_weights or 'balanced'
        ),
        "XGBoost": XGBoostBaseline(
            random_state=random_state,
            n_estimators=200,
            scale_pos_weight=scale_pos_weight
        ),
        "LightGBM": LightGBMBaseline(
            random_state=random_state,
            n_estimators=200,
            class_weight=class_weights or 'balanced'
        ),
        "MLP": MLPBaseline(
            random_state=random_state,
            hidden_layer_sizes=(128, 64, 32),
            max_iter=300
        ),
        "LogisticRegression": LogisticRegressionBaseline(
            random_state=random_state,
            class_weight=class_weights or 'balanced'
        )
    }
    
    return baselines


def evaluate_classifier(clf: BaselineClassifier, 
                        X_test: np.ndarray, y_test: np.ndarray,
                        X_train: np.ndarray = None, y_train: np.ndarray = None,
                        compute_curves: bool = True) -> Dict[str, Any]:
    """
    Comprehensive evaluation of a classifier.
    
    Parameters
    ----------
    clf : BaselineClassifier
        Fitted classifier
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test labels
    X_train, y_train : optional
        Training data (for additional metrics)
    compute_curves : bool
        Whether to compute ROC/PR curves
        
    Returns
    -------
    dict
        Comprehensive evaluation metrics
    """
    # Predictions
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    
    # Ensure proba is 2D
    if y_proba.ndim == 1:
        y_proba = np.column_stack([1 - y_proba, y_proba])
    
    # Basic metrics
    metrics = {
        'method': clf.name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'training_time': clf.training_time,
        'mean_inference_time': np.mean(clf.inference_times) if clf.inference_times else 0,
        'std_inference_time': np.std(clf.inference_times) if clf.inference_times else 0,
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        metrics['detection_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # AUC-ROC
    if y_proba.shape[1] == 2:
        metrics['auc_roc'] = roc_auc_score(y_test, y_proba[:, 1])
        
        # PR-AUC
        if compute_curves:
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba[:, 1])
            metrics['pr_auc'] = auc(recall_curve, precision_curve)
    
    # Feature importance
    importance = clf.get_feature_importance()
    if importance is not None:
        metrics['feature_importance'] = importance.tolist()
    
    return metrics


def run_baseline_comparison(X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray,
                           baselines: Dict[str, BaselineClassifier] = None,
                           random_state: int = 42) -> Dict[str, Dict[str, Any]]:
    """
    Run complete baseline comparison.
    
    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training data
    X_test, y_test : np.ndarray
        Test data
    baselines : dict, optional
        Pre-created baselines (created if not provided)
    random_state : int
        Random seed
        
    Returns
    -------
    dict
        Results for each baseline
    """
    # Compute class weights
    from collections import Counter
    class_counts = Counter(y_train)
    n_total = len(y_train)
    class_weights = {cls: n_total / (len(class_counts) * count) 
                     for cls, count in class_counts.items()}
    
    if baselines is None:
        baselines = create_all_baselines(random_state, class_weights)
    
    results = {}
    
    print(f"\nRunning baseline comparison on {len(X_train)} train / {len(X_test)} test samples...")
    print(f"Class distribution - Train: {dict(Counter(y_train))}, Test: {dict(Counter(y_test))}")
    
    for name, clf in baselines.items():
        print(f"\n  Training {name}...")
        try:
            clf.fit(X_train, y_train)
            metrics = evaluate_classifier(clf, X_test, y_test, X_train, y_train)
            results[name] = metrics
            print(f"    Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, "
                  f"AUC: {metrics.get('auc_roc', 0):.4f}")
            print(f"    Train time: {metrics['training_time']:.3f}s, "
                  f"Inference: {metrics['mean_inference_time']*1000:.2f}ms avg")
        except Exception as e:
            print(f"    FAIL Failed: {e}")
            results[name] = {'method': name, 'error': str(e)}
    
    return results


def hyperparameter_tune(clf: BaselineClassifier, X: np.ndarray, y: np.ndarray,
                        param_grid: Dict, cv: int = 3, scoring: str = 'f1') -> BaselineClassifier:
    """
    Perform hyperparameter tuning using GridSearchCV.
    
    Parameters
    ----------
    clf : BaselineClassifier
        Classifier to tune
    X, y : np.ndarray
        Training data
    param_grid : dict
        Parameter grid
    cv : int
        CV folds
    scoring : str
        Scoring metric
        
    Returns
    -------
    BaselineClassifier
        Best fitted classifier
    """
    grid_search = GridSearchCV(
        clf.model, param_grid, cv=cv, scoring=scoring, 
        n_jobs=-1, verbose=0
    )
    grid_search.fit(X, y)
    
    # Create new classifier with best params
    best_clf = type(clf)(random_state=clf.random_state)
    best_clf.model = grid_search.best_estimator_
    best_clf.is_fitted = True
    best_clf.training_time = 0  # Not measured here
    
    print(f"  Best params for {clf.name}: {grid_search.best_params_}")
    print(f"  Best CV {scoring}: {grid_search.best_score_:.4f}")
    
    return best_clf


def get_param_grids() -> Dict[str, Dict]:
    """Get hyperparameter grids for each baseline."""
    return {
        "RandomForest": {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        "SVM": {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.1],
            'kernel': ['rbf', 'linear']
        },
        "XGBoost": {
            'n_estimators': [100, 200, 300],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0]
        },
        "LightGBM": {
            'n_estimators': [100, 200, 300],
            'num_leaves': [31, 63, 127],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [-1, 10, 20]
        },
        "MLP": {
            'hidden_layer_sizes': [(64,), (128, 64), (128, 64, 32), (256, 128, 64)],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['adaptive', 'constant']
        },
        "LogisticRegression": {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'liblinear']
        }
    }


def calibrate_predictions(clf: BaselineClassifier, X_cal: np.ndarray, y_cal: np.ndarray,
                          method: str = 'isotonic') -> BaselineClassifier:
    """
    Calibrate classifier probabilities using held-out calibration data.
    
    Parameters
    ----------
    clf : BaselineClassifier
        Fitted classifier
    X_cal, y_cal : np.ndarray
        Calibration data
    method : str
        'isotonic' or 'sigmoid' (Platt scaling)
        
    Returns
    -------
    BaselineClassifier
        Calibrated classifier
    """
    if not CALIBRATION_AVAILABLE:
        print("  Calibration not available (sklearn version)")
        return clf
    
    calibrated = CalibratedClassifierCV(clf.model, method=method, cv='prefit')
    calibrated.fit(X_cal, y_cal)
    
    # Wrap in baseline interface
    calibrated_clf = type(clf)(random_state=clf.random_state)
    calibrated_clf.model = calibrated
    calibrated_clf.is_fitted = True
    calibrated_clf.name = clf.name + "_calibrated"
    
    return calibrated_clf


def compute_calibration_metrics(y_true: np.ndarray, y_proba: np.ndarray,
                                 n_bins: int = 10) -> Dict[str, float]:
    """
    Compute calibration metrics: ECE, MCE, reliability diagram data.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_proba : np.ndarray
        Predicted probabilities for positive class
    n_bins : int
        Number of bins for reliability diagram
        
    Returns
    -------
    dict
        Calibration metrics
    """
    # Ensure 1D probabilities
    if y_proba.ndim == 2:
        y_proba = y_proba[:, 1]
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_proba, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    ece = 0.0  # Expected Calibration Error
    mce = 0.0  # Maximum Calibration Error
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_proba[mask])
            bin_count = np.sum(mask)
            
            bin_accuracies.append(bin_acc)
            bin_confidences.append(bin_conf)
            bin_counts.append(bin_count)
            
            ece += (bin_count / len(y_true)) * abs(bin_acc - bin_conf)
            mce = max(mce, abs(bin_acc - bin_conf))
    
    return {
        'ece': float(ece),
        'mce': float(mce),
        'bin_accuracies': bin_accuracies,
        'bin_confidences': bin_confidences,
        'bin_counts': bin_counts,
        'bin_edges': bin_edges.tolist()
    }


def compare_all_methods(results: Dict[str, Dict]) -> pd.DataFrame:
    """Create comparison DataFrame from results."""
    rows = []
    for method, metrics in results.items():
        if 'error' not in metrics:
            row = {'Method': method}
            for key in ['accuracy', 'precision', 'recall', 'f1', 'fpr', 'fnr', 
                       'detection_rate', 'auc_roc', 'pr_auc', 'training_time', 
                       'mean_inference_time']:
                row[key] = metrics.get(key, 0)
            rows.append(row)
    
    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values('f1', ascending=False)
    return df


if __name__ == "__main__":
    print("Testing baseline classifiers...")
    
    # Create synthetic data
    np.random.seed(42)
    n_train = 1000
    n_test = 300
    n_features = 20
    
    X_train = np.random.randn(n_train, n_features)
    X_train[500:] *= 2  # Make second class separable
    y_train = np.array([0] * 500 + [1] * 500)
    
    X_test = np.random.randn(n_test, n_features)
    X_test[150:] *= 2
    y_test = np.array([0] * 150 + [1] * 150)
    
    # Run comparison
    baselines = create_all_baselines()
    results = run_baseline_comparison(X_train, y_train, X_test, y_test, baselines)
    
    # Print comparison table
    comparison_df = compare_all_methods(results)
    print("\n=== Baseline Comparison ===")
    print(comparison_df[['Method', 'accuracy', 'precision', 'recall', 'f1', 'auc_roc', 
                         'training_time', 'mean_inference_time']].to_string(index=False))
    
    # Test calibration
    print("\n=== Calibration Test ===")
    for name, clf in baselines.items():
        if hasattr(clf, 'is_fitted') and clf.is_fitted:
            y_proba = clf.predict_proba(X_test)
            if y_proba.ndim == 2:
                cal_metrics = compute_calibration_metrics(y_test, y_proba[:, 1])
                print(f"  {name}: ECE={cal_metrics['ece']:.4f}, MCE={cal_metrics['mce']:.4f}")
    
    print("\n[PASS] Baseline comparison complete!")