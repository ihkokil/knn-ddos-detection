"""
Pytest fixtures and configuration for KNN-based DDoS detection test suite.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_synthetic_traffic
from src.knn_baseline import StandardKNN
from src.knn_improved import ImprovedKNN
from src.knn_proposed import ProposedKNN
from src.degree_of_attack import DegreeOfAttackComputer


@pytest.fixture(scope="session")
def random_seed():
    return 42


@pytest.fixture(scope="session")
def synthetic_data(random_seed):
    """Generate reproducible synthetic flow traffic for fast unit testing."""
    X_train, y_train, X_test, y_test, feature_names = load_synthetic_traffic({
        'num_normal': 400,
        'num_attack': 200,
        'attack_intensity': 8.0,
        'num_features': 8,
        'random_seed': random_seed
    })
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return {
        'X_train': X_train_scaled,
        'y_train': y_train,
        'X_test': X_test_scaled,
        'y_test': y_test,
        'feature_names': feature_names
    }


@pytest.fixture(scope="session")
def sample_degrees(synthetic_data):
    """Compute degrees of attack for the synthetic test data."""
    comp = DegreeOfAttackComputer(method='proposed')
    deg_train = comp.compute_degree_from_array(synthetic_data['X_train'], synthetic_data['feature_names'])
    deg_test = comp.compute_degree_from_array(synthetic_data['X_test'], synthetic_data['feature_names'])
    return {
        'deg_train': deg_train,
        'deg_test': deg_test
    }
