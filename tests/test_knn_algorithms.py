"""
Unit tests for StandardKNN, ImprovedKNN, and ProposedKNN algorithms.
"""

import pytest
import numpy as np
from src.knn_baseline import StandardKNN
from src.knn_improved import ImprovedKNN
from src.knn_proposed import ProposedKNN


class TestStandardKNN:
    def test_fit_and_predict(self, synthetic_data):
        X_tr, y_tr = synthetic_data['X_train'], synthetic_data['y_train']
        X_te, y_te = synthetic_data['X_test'], synthetic_data['y_test']
        
        clf = StandardKNN(k=5, metric='euclidean', weights='uniform')
        clf.fit(X_tr, y_tr)
        
        preds = clf.predict(X_te)
        proba = clf.predict_proba(X_te)
        
        assert len(preds) == len(X_te)
        assert set(np.unique(preds)).issubset({0, 1})
        assert proba.shape == (len(X_te), 2)
        np.testing.assert_allclose(np.sum(proba, axis=1), 1.0, atol=1e-5)
        assert not np.isnan(preds).any()
        assert not np.isnan(proba).any()

    def test_edge_case_k1(self, synthetic_data):
        X_tr, y_tr = synthetic_data['X_train'][:20], synthetic_data['y_train'][:20]
        clf = StandardKNN(k=1)
        clf.fit(X_tr, y_tr)
        # On training data itself, k=1 should have 100% accuracy
        preds = clf.predict(X_tr)
        np.testing.assert_array_equal(preds, y_tr)


class TestImprovedKNN:
    def test_fit_and_predict_with_degrees(self, synthetic_data, sample_degrees):
        X_tr, y_tr = synthetic_data['X_train'], synthetic_data['y_train']
        X_te = synthetic_data['X_test']
        d_tr = sample_degrees['deg_train']
        
        clf = ImprovedKNN(k=5, metric='euclidean', weights='distance')
        clf.fit(X_tr, y_tr, d_tr)
        
        preds = clf.predict(X_te)
        proba = clf.predict_proba(X_te)
        
        assert len(preds) == len(X_te)
        assert proba.shape == (len(X_te), 2)
        np.testing.assert_allclose(np.sum(proba, axis=1), 1.0, atol=1e-5)
        assert not np.isnan(preds).any()
        assert not np.isnan(proba).any()

    def test_zero_distance_stability(self):
        # Test numerical stability when query points are identical to training points
        X = np.ones((10, 4))
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        deg = np.linspace(0.1, 0.9, 10)
        
        clf = ImprovedKNN(k=3)
        clf.fit(X, y, deg)
        proba = clf.predict_proba(X)
        assert not np.isnan(proba).any()
        np.testing.assert_allclose(np.sum(proba, axis=1), 1.0, atol=1e-5)


class TestProposedKNN:
    def test_bayesian_fusion_predict(self, synthetic_data, sample_degrees):
        X_tr, y_tr = synthetic_data['X_train'], synthetic_data['y_train']
        X_te = synthetic_data['X_test']
        d_tr = sample_degrees['deg_train']
        d_te = sample_degrees['deg_test']
        
        clf = ProposedKNN(k=5, adaptive_k=True)
        clf.fit(X_tr, y_tr, d_tr)
        
        preds = clf.predict(X_te, degrees_test=d_te)
        proba = clf.predict_proba(X_te, degrees_test=d_te)
        
        assert len(preds) == len(X_te)
        assert proba.shape == (len(X_te), 2)
        np.testing.assert_allclose(np.sum(proba, axis=1), 1.0, atol=1e-5)
        assert not np.isnan(preds).any()
        assert not np.isnan(proba).any()

    def test_fallback_without_test_degrees(self, synthetic_data, sample_degrees):
        X_tr, y_tr = synthetic_data['X_train'], synthetic_data['y_train']
        X_te = synthetic_data['X_test']
        d_tr = sample_degrees['deg_train']
        
        clf = ProposedKNN(k=5, adaptive_k=False)
        clf.fit(X_tr, y_tr, d_tr)
        # Call without degrees_test
        preds = clf.predict(X_te)
        proba = clf.predict_proba(X_te)
        assert len(preds) == len(X_te)
        assert not np.isnan(proba).any()
