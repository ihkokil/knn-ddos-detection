"""
Unit tests for statistical significance tests and metric evaluation.
"""

import pytest
import numpy as np
from src.statistical_tests import (
    paired_wilcoxon,
    paired_ttest,
    cohens_d_paired,
    friedman_test,
    multiple_comparison_correction
)
from src.evaluator import DDoSEvaluator


def test_wilcoxon_and_effect_size():
    # Model B consistently outperforms Model A
    scores_a = np.array([0.75, 0.76, 0.74, 0.75, 0.77, 0.76, 0.75, 0.74, 0.76, 0.75])
    scores_b = np.array([0.80, 0.82, 0.81, 0.79, 0.83, 0.82, 0.81, 0.80, 0.82, 0.81])
    
    w_res = paired_wilcoxon(scores_b, scores_a)
    assert w_res['p_value'] < 0.05
    assert w_res['p_value'] > 0.0
    
    d_val = cohens_d_paired(scores_b, scores_a)
    assert d_val > 1.0  # Large positive effect size


def test_multiple_comparison_correction():
    p_vals = [0.001, 0.02, 0.04, 0.20]
    res = multiple_comparison_correction(p_vals, method='holm', alpha=0.05)
    assert 'corrected_p' in res
    assert 'reject' in res


def test_ddos_evaluator():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 1, 1])
    y_proba = np.array([
        [0.9, 0.1],
        [0.8, 0.2],
        [0.4, 0.6],
        [0.2, 0.8],
        [0.1, 0.9],
        [0.1, 0.9]
    ])
    
    evaluator = DDoSEvaluator()
    metrics = evaluator.evaluate_classification(y_true, y_pred, y_proba)
    
    assert 0.0 <= metrics['accuracy'] <= 1.0
    assert 0.0 <= metrics['auc_roc'] <= 1.0
    assert 'f1' in metrics
    assert 'fpr' in metrics
