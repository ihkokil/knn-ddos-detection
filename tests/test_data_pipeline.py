"""
Unit tests for data_loader and preprocessor modules.
"""

import pytest
import numpy as np
from src.data_loader import load_dataset, load_synthetic_traffic
from src.preprocessor import FeaturePreprocessor


def test_synthetic_loading():
    data = load_dataset({'dataset': 'synthetic', 'random_seed': 42})
    assert data['X_train'] is not None
    assert data['y_train'] is not None
    assert set(np.unique(data['y_train'])).issubset({0, 1})
    assert len(data['feature_names']) == 10


def test_nsl_kdd_subsample():
    data = load_dataset({
        'dataset': 'NSL-KDD',
        'subsample_size': 500,
        'random_seed': 42
    })
    if data['X_train'] is not None:
        assert data['X_train'].shape[0] == 500
        assert data['X_train'].shape[1] == 41
        assert data['X_train'].dtype == np.float64
        assert 0 in data['class_distributions']['train']
        assert 1 in data['class_distributions']['train']


def test_preprocessor_scaling(synthetic_data):
    X = synthetic_data['X_train']
    y = synthetic_data['y_train']
    prep = FeaturePreprocessor(config={'scaling': 'minmax', 'encode_categorical': False})
    X_tr_s, X_te_s, y_tr, y_te = prep.fit_transform(X, y)
    assert np.all(X_tr_s >= 0.0)
    assert np.all(X_tr_s <= 1.0 + 1e-6)


def test_insdn_subsample():
    data = load_dataset({
        'dataset': 'InSDN',
        'subsample_size': 200,
        'random_seed': 42
    })
    if data['X_train'] is not None:
        assert data['X_train'].shape[0] == 200
        assert data['X_train'].shape[1] == 77
        assert data['X_train'].dtype == np.float64
        assert 0 in data['class_distributions']['train']
        assert 1 in data['class_distributions']['train']
        assert data['X_test'].shape[0] >= 200
        assert data['X_test'].shape[1] == 77
