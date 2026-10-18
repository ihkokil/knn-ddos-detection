"""
Unit tests for DegreeOfAttackComputer and thresholding methods.
"""

import pytest
import numpy as np
import pandas as pd
from src.degree_of_attack import DegreeOfAttackComputer, compute_degree_with_threshold, degree_analysis_report


def test_degree_range_and_finite():
    X = np.random.uniform(0, 100, (50, 6))
    comp_base = DegreeOfAttackComputer(method='base_paper')
    deg_base = comp_base.compute_degree_from_array(X)
    
    assert len(deg_base) == 50
    assert np.all(deg_base >= 0.0)
    assert np.all(deg_base <= 1.0)
    assert not np.isnan(deg_base).any()
    
    comp_prop = DegreeOfAttackComputer(method='proposed')
    deg_prop = comp_prop.compute_degree_from_array(X)
    assert len(deg_prop) == 50
    assert np.all(deg_prop >= 0.0)
    assert np.all(deg_prop <= 1.0)
    assert not np.isnan(deg_prop).any()


def test_named_dataframe_degree():
    df = pd.DataFrame({
        'packet_rate': [5.0, 120.0, 500.0, 2.0],
        'byte_rate': [200.0, 5000.0, 20000.0, 100.0],
        'duration': [60.0, 2.0, 0.5, 100.0],
        'serror_rate': [0.0, 0.9, 1.0, 0.0],
        'diff_srv_rate': [0.0, 0.5, 0.8, 0.0]
    })
    comp = DegreeOfAttackComputer(method='proposed')
    deg = comp.compute_degree(df.values, feature_names=list(df.columns))
    
    assert len(deg) == 4
    # The flood attack rows (index 1, 2) should have strictly higher degree than normal (index 0, 3)
    assert deg[2] > deg[0]
    assert deg[1] > deg[3]


def test_thresholding_and_analysis():
    degrees = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    binary, th = compute_degree_with_threshold(degrees, threshold=0.5, method='fixed')
    assert np.array_equal(binary, [0, 0, 0, 1, 1, 1])
    assert th == 0.5
    
    # Test supervised Youden optimization
    y_true = np.array([0, 0, 0, 1, 1, 1])
    b_youd, th_youd = compute_degree_with_threshold(degrees, method='youden', y_true=y_true)
    assert np.array_equal(b_youd, y_true)
    assert 0.3 < th_youd <= 0.7
    
    # Test unsupervised Otsu thresholding
    b_otsu, th_otsu = compute_degree_with_threshold(degrees, method='otsu')
    assert np.array_equal(b_otsu, [0, 0, 0, 1, 1, 1])
    assert 0.3 < th_otsu <= 0.7
    
    report = degree_analysis_report(degrees)
    assert report['n_samples'] == 6
    assert 0.0 <= report['degree_stats']['mean'] <= 1.0

