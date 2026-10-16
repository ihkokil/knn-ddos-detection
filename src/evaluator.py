"""
Evaluation and Metrics Module
Comprehensive evaluation of DDoS detection results with standard and custom metrics.

Computes:
- Classification accuracy, precision, recall, F1-score
- False positive/negative rates
- Confusion matrix
- ROC curve and AUC
- Detection latency and computation time
- Degree-of-attack measurement quality
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve, auc as auc_func
import warnings


class DDoSEvaluator:
    """
    Comprehensive evaluator for DDoS detection systems.
    
    Provides metrics for assessing detection performance including:
    - Standard classification metrics (accuracy, precision, recall, F1)
    - Rate-based metrics (FPR, FNR, detection rate)
    - Ranking metrics (AUC-ROC)
    - Time-based metrics (computation latency)
    - Degree-of-attack specific metrics
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the evaluator.
        
        Parameters
        ----------
        config : dict
            Evaluation configuration:
            - 'metrics': list of metrics to compute
            - 'average': 'macro', 'micro', 'weighted' for multi-class
            - 'return_confusion': bool
        """
        if config is None:
            config = {
                'metrics': ['accuracy', 'precision', 'recall', 'f1', 'fpr', 'fnr', 
                           'detection_rate', 'auc_roc', 'computation_time'],
                'return_confusion': True,
                'positive_label': 1  # Attack class is positive
            }
        self.config = config
    
    def evaluate_classification(self, y_true: np.ndarray, y_pred: np.ndarray,
                                y_proba: Optional[np.ndarray] = None,
                                degrees: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Evaluate classification performance with comprehensive metrics.
        
        Parameters
        ----------
        y_true : np.ndarray
            Ground truth labels (0=normal, 1=attack)
        y_pred : np.ndarray
            Predicted labels
        y_proba : np.ndarray, optional
            Prediction probabilities for positive class, shape (n_samples,)
            or (n_samples, 2) for [P(normal), P(attack)]
        degrees : np.ndarray, optional
            Degree-of-attack values per sample (for degree-specific analysis)
        
        Returns
        -------
        dict
            Dictionary of evaluation metrics
        """
        n_samples = len(y_true)
        
        # Basic counts
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))  # True Positives
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))  # False Positives
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))  # True Negatives
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))  # False Negatives
        
        # Compute metrics
        accuracy = float(np.sum(y_pred == y_true) / n_samples) if n_samples > 0 else 0.0
        
        precision = float(tp / (tp + fp + 1e-10)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn + 1e-10)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp + 1e-10)) if (tn + fp) > 0 else 0.0
        
        # False Positive Rate = FP / (FP + TN)
        fpr = float(fp / (fp + tn + 1e-10)) if (fp + tn) > 0 else 0.0
        
        # False Negative Rate = FN / (FN + TP)
        fnr = float(fn / (fn + tp + 1e-10)) if (fn + tp) > 0 else 0.0
        
        # Detection Rate = TP / (TP + FN) = Recall
        detection_rate = recall
        
        # F1-score
        f1 = float(2 * precision * recall / (precision + recall + 1e-10)) if (precision + recall) > 0 else 0.0
        
        # False Negative Rate complement (True Positive Rate = Recall)
        tpr = recall  # True Positive Rate = Recall
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1': f1,
            'fpr': fpr,
            'fnr': fnr,
            'detection_rate': detection_rate,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
            'total_samples': n_samples,
        }
        
        # Add probability-based metrics if probabilities provided
        if y_proba is not None:
            # Handle both (n,) and (n, 2) formats
            if y_proba.ndim == 2:
                if y_proba.shape[1] == 2:
                    proba_attack = y_proba[:, 1]  # Probability of attack class
                else:
                    proba_attack = y_proba[:, 0]
            else:
                proba_attack = y_proba
            
            # AUC-ROC (if we have proper scores and binary labels)
            try:
                if len(np.unique(y_true)) > 1 and len(proba_attack) > 1:
                    auc = float(roc_auc_score(y_true, proba_attack))
                else:
                    auc = 0.5  # Random classifier
            except Exception:
                auc = 0.5
            
            # Precision-Recauc AUC (PR-AUC)
            try:
                precision_curve, recall_curve, _ = precision_recall_curve(y_true, proba_attack)
                pr_auc = float(auc_func(recall_curve, precision_curve))
            except Exception:
                pr_auc = float(f1) if f1 > 0 else 0.0
            
            results['auc_roc'] = auc
            results['pr_auc'] = pr_auc
            results['mean_attack_probability'] = float(np.mean(proba_attack))
            results['max_attack_probability'] = float(np.max(proba_attack))
        
        # Degree-of-attack specific metrics if degrees provided
        if degrees is not None:
            # Separate degrees by true class
            deg_normal = degrees[y_true == 0]
            deg_attack = degrees[y_true == 1]
            
            results.update({
                'normal_degree_mean': float(np.mean(deg_normal)) if len(deg_normal) > 0 else 0.0,
                'normal_degree_std': float(np.std(deg_normal)) if len(deg_normal) > 0 else 0.0,
                'attack_degree_mean': float(np.mean(deg_attack)) if len(deg_attack) > 0 else 0.0,
                'attack_degree_std': float(np.std(deg_attack)) if len(deg_attack) > 0 else 0.0,
                'degree_gap': float(np.mean(deg_attack) - np.mean(deg_normal)) 
                               if len(deg_attack) > 0 and len(deg_normal) > 0 else 0.0,
                'degree_overlap': float(
                    len(np.intersect1d(
                        (deg_normal > np.percentile(deg_normal, 75)),
                        (deg_attack < np.percentile(deg_attack, 25))
                    )) / min(len(deg_normal), len(deg_attack))
                ) if len(deg_normal) > 0 and len(deg_attack) > 0 else 0.0,
                'degree_threshold_0.5_accuracy': float(
                    np.sum((degrees > 0.5) == y_true) / len(degrees)
                ) if len(degrees) > 0 else 0.0,
            })
        
        return results
    
    def compute_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  labels: Optional[List] = None) -> np.ndarray:
        """
        Compute confusion matrix.
        
        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        labels : list, optional
            Label order [negative, positive]
        
        Returns
        -------
        np.ndarray
            Confusion matrix [[TN, FP], [FN, TP]]
        """
        if labels is None:
            labels = [0, 1]  # [normal, attack]
        
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        return cm
    
    def evaluate_degree_performance(self, degrees: np.ndarray, 
                                    y_true: np.ndarray,
                                    thresholds: Optional[List[float]] = None) -> Dict[str, float]:
        """
        Evaluate degree-of-attack metric performance.
        
        Parameters
        ----------
        degrees : np.ndarray
            Degree-of-attack values in [0, 1]
        y_true : np.ndarray
            True binary labels
        thresholds : list, optional
            Thresholds to evaluate; if None, uses [0.3, 0.5, 0.7]
        
        Returns
        -------
        dict
            Degree-based performance at each threshold
        """
        if thresholds is None:
            thresholds = [0.3, 0.5, 0.7]
        
        results = {}
        
        for t in thresholds:
            # Classify based on threshold
            y_pred_thresh = (degrees >= t).astype(int)
            
            # Compute classification metrics at this threshold
            tp = int(np.sum((y_pred_thresh == 1) & (y_true == 1)))
            fp = int(np.sum((y_pred_thresh == 1) & (y_true == 0)))
            fn = int(np.sum((y_pred_thresh == 0) & (y_true == 1)))
            tn = int(np.sum((y_pred_thresh == 0) & (y_true == 0)))
            
            accuracy = float(np.sum(y_pred_thresh == y_true) / len(degrees)) if len(degrees) > 0 else 0.0
            precision = float(tp / (tp + fp + 1e-10)) if (tp + fp) > 0 else 0.0
            recall = float(tp / (tp + fn + 1e-10)) if (tp + fn) > 0 else 0.0
            fpr = float(fp / (fp + tn + 1e-10)) if (fp + tn) > 0 else 0.0
            fnr = float(fn / (fn + tp + 1e-10)) if (fn + tp) > 0 else 0.0
            f1 = float(2 * precision * recall / (precision + recall + 1e-10)) if (precision + recall) > 0 else 0.0
            
            results[f'threshold_{t}'] = {
                'threshold': t,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'fpr': fpr,
                'fnr': fnr,
                'f1': f1,
                'true_positives': tp,
                'false_positives': fp,
            }
        
        # Find optimal threshold (Youden's index = max TPR - FPR)
        youden_indices = []
        for t in np.linspace(0, 1, 101):
            y_pred_t = (degrees >= t).astype(int)
            tpr = np.sum((y_pred_t == 1) & (y_true == 1)) / (np.sum(y_true == 1) + 1e-10)
            fpr_t = np.sum((y_pred_t == 1) & (y_true == 0)) / (np.sum(y_true == 0) + 1e-10)
            youden = tpr - fpr_t
            youden_indices.append((t, youden))
        
        # Best threshold
        best_idx = max(range(len(youden_indices)), key=lambda i: youden_indices[i][1])
        best_threshold, best_youden = youden_indices[best_idx]
        
        results['optimal_threshold'] = best_threshold
        results['youden_index'] = best_youden
        results['optimal_threshold_details'] = results.get(f'threshold_{best_threshold}', {})
        
        return results
    
    def compute_detection_latency(self, 
                                  start_times: np.ndarray,
                                  end_times: np.ndarray,
                                  description: str = "detection") -> Dict[str, float]:
        """
        Compute detection latency metrics.
        
        Parameters
        ----------
        start_times : np.ndarray
            Timestamps when detection started (per flow or per window)
        end_times : np.ndarray
            Timestamps when detection completed
        description : str
            Description for display purposes
        
        Returns
        -------
        dict
            Latency metrics
        """
        latencies = end_times - start_times
        # Ensure non-negative (handle any negative due to timing edge cases)
        latencies = np.maximum(latencies, 0)
        
        return {
            f'{description}_latency_mean': float(np.mean(latencies)),
            f'{description}_latency_median': float(np.median(latencies)),
            f'{description}_latency_min': float(np.min(latencies)),
            f'{description}_latency_max': float(np.max(latencies)),
            f'{description}_latency_std': float(np.std(latencies)),
            f'{description}_latencies': latencies.tolist(),
        }
    
    def generate_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                       y_proba: Optional[np.ndarray] = None,
                                       degrees: Optional[np.ndarray] = None,
                                       target_names: Optional[List[str]] = None) -> str:
        """
        Generate a human-readable classification report.
        
        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        y_proba : np.ndarray, optional
            Prediction probabilities
        degrees : np.ndarray, optional
            Degree-of-attack values
        target_names : list, optional
            Names for [normal, attack]
        
        Returns
        -------
        str
            Formatted classification report
        """
        if target_names is None:
            target_names = ['Normal', 'Attack']
        
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("DDoS DETECTION CLASSIFICATION REPORT")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # Basic metrics
        eval_results = self.evaluate_classification(y_true, y_pred, y_proba, degrees)
        
        report_lines.append("Overall Performance:")
        report_lines.append(f"  Accuracy:  {eval_results['accuracy']:.4f} ({eval_results['accuracy']*100:.2f}%)")
        report_lines.append(f"  Precision: {eval_results['precision']:.4f} ({eval_results['precision']*100:.2f}%)")
        report_lines.append(f"  Recall:    {eval_results['recall']:.4f} ({eval_results['recall']*100:.2f}%)")
        report_lines.append(f"  F1-score:  {eval_results['f1']:.4f} ({eval_results['f1']*100:.2f}%)")
        report_lines.append(f"  FPR:       {eval_results['fpr']:.4f} ({eval_results['fpr']*100:.2f}%)")
        report_lines.append(f"  FNR:       {eval_results['fnr']:.4f} ({eval_results['fnr']*100:.2f}%)")
        report_lines.append(f"  Detection Rate: {eval_results['detection_rate']:.4f} "
                          f"({eval_results['detection_rate']*100:.2f}%)")
        if 'auc_roc' in eval_results:
            report_lines.append(f"  AUC-ROC:   {eval_results['auc_roc']:.4f}")
        report_lines.append("")
        
        # Confusion matrix counts
        tp = eval_results['true_positives']
        fp = eval_results['false_positives']
        tn = eval_results['true_negatives'] if 'true_negatives' in eval_results else 0
        fn = eval_results['false_negatives']
        
        report_lines.append("Confusion Matrix:")
        report_lines.append(f"                  | Predicted Normal | Predicted Attack")
        report_lines.append(f"  Actual Normal  |        {tn:5d}        |        {fp:5d}")
        report_lines.append(f"  Actual Attack  |        {fn:5d}        |        {tp:5d}")
        report_lines.append("")
        
        # Degree-specific info if provided
        if degrees is not None:
            deg_normal = degrees[y_true == 0]
            deg_attack = degrees[y_true == 1]
            report_lines.append("Degree-of-Attack Analysis:")
            report_lines.append(f"  Normal flows: mean degree = {np.mean(deg_normal):.4f} "
                              f"(std: {np.std(deg_normal):.4f}), count = {len(deg_normal)}")
            report_lines.append(f"  Attack flows: mean degree = {np.mean(deg_attack):.4f} "
                              f"(std: {np.std(deg_attack):.4f}), count = {len(deg_attack)}")
            report_lines.append(f"  Degree gap (attack - normal): {np.mean(deg_attack) - np.mean(deg_normal):.4f}")
            report_lines.append("")
        
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)
    
    def compare_methods(self, results_dict: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Compare multiple methods' evaluation results.
        
        Parameters
        ----------
        results_dict : dict
            Dictionary mapping method names to their evaluation result dicts
            e.g., {'baseline': {...}, 'improved': {...}, 'proposed': {...}}
        
        Returns
        -------
        pd.DataFrame
            DataFrame with methods as index and metrics as columns
        """
        comparison_data = {}
        
        for method_name, method_results in results_dict.items():
            for metric_name, metric_value in method_results.items():
                if metric_name not in comparison_data:
                    comparison_data[metric_name] = {}
                comparison_data[metric_name][method_name] = metric_value
        
        df = pd.DataFrame(comparison_data).T
        df = df.sort_index()  # Sort by metric name
        
        return df


# Convenience function for quick evaluation
def quick_evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None,
                   degrees: np.ndarray = None) -> Dict[str, float]:
    """
    Quick evaluation of classification results.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_proba : np.ndarray, optional
        Prediction probabilities
    degrees : np.ndarray, optional
        Degree-of-attack values
    
    Returns
    -------
    dict
        Evaluation metrics
    """
    from src.evaluator import DDoSEvaluator
    evaluator = DDoSEvaluator()
    return evaluator.evaluate_classification(y_true, y_pred, y_proba, degrees)


if __name__ == "__main__":
    # Quick test: evaluator metrics
    print("Testing evaluator module...")
    
    np.random.seed(42)
    n = 200
    
    # True labels: 70% normal, 30% attack
    y_true = np.array([0] * 140 + [1] * 60)
    
    # Predictions: ~85% accuracy
    y_pred = np.array([0] * 120 + [1] * 20 + [1] * 40 + [0] * 20)  # Some errors
    # Fix to have correct distribution
    y_pred = np.random.choice([0, 1], size=n, p=[0.75, 0.25])  # Random for demo
    # Actually let's make it more structured
    # 80% of normal predicted normal, 20% false attack
    # 70% of attack predicted attack, 30% false normal
    y_pred = np.zeros(n)
    # Normal samples (first 140)
    n_normal_pred = int(140 * 0.8)  # 80% correctly predicted normal
    y_pred[:n_normal_pred] = 0  # True negatives
    y_pred[n_normal_pred:140] = 1  # False positives
    # Attack samples (last 60)
    n_attack_pred = int(60 * 0.7)  # 70% correctly predicted attack
    y_pred[140:n_normal_pred + n_attack_pred] = 1  # True positives
    y_pred[n_normal_pred + n_attack_pred:200] = 0  # False negatives
    
    # Probabilities
    y_proba = np.random.rand(n)  # Random probs for demo
    y_proba = np.column_stack([1 - y_proba, y_proba])  # [P(normal), P(attack)]
    
    # Degrees
    degrees = np.random.rand(n)
    degrees[y_true == 1] = np.random.uniform(0.5, 1.0, np.sum(y_true == 1))
    degrees[y_true == 0] = np.random.uniform(0.0, 0.5, np.sum(y_true == 0))
    
    # Evaluate
    results = quick_evaluate(y_true, y_pred, y_proba, degrees)
    print("\n=== Evaluation Results ===")
    for key, value in sorted(results.items()):
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        elif isinstance(value, np.ndarray):
            print(f"  {key}: array of {len(value)} elements")
        else:
            print(f"  {key}: {value}")