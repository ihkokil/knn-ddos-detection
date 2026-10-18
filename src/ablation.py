"""
Ablation Study Module
Systematic ablation experiments for the proposed KNN components:
1. Degree weighting
2. Adaptive K
3. Bounded distance compression
4. Dual voting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from itertools import product
import time
from collections import Counter

from src.knn_baseline import StandardKNN, check_is_fitted as check_bl
from src.knn_improved import ImprovedKNN, check_is_fitted as check_imp
from src.knn_proposed import ProposedKNN, check_is_fitted as check_prop
from src.degree_of_attack import DegreeOfAttackComputer
from src.evaluator import DDoSEvaluator, quick_evaluate
from src.utils import set_seed


class AblationConfig:
    """Configuration for a specific ablation variant."""
    
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.params = kwargs
    
    def __repr__(self):
        return f"AblationConfig({self.name}, {self.params})"


# Define all ablation variants
ABLATION_VARIANTS = [
    # Base 1: Standard uniform KNN
    AblationConfig("baseline_knn", 
                   method="baseline", weights="uniform", adaptive_k=False),
    
    # Base 2: Dong & Sarem (2020) Improved KNN (exponential distance weighting)
    AblationConfig("base_paper", 
                   method="improved", weights="exponential", adaptive_k=False),
    
    # Variant 1: Improved KNN + Exemplar Degree Weighting
    AblationConfig("degree_weighted", 
                   method="improved", weights="attack_degree", adaptive_k=False),
    
    # Variant 2: Adaptive K only (exponential distance, no severity multiplier)
    AblationConfig("adaptive_k_only", 
                   method="proposed", weights="exponential", adaptive_k=True),
    
    # Variant 3: Dual severity voting only (fixed k=5, dual proximity-severity)
    AblationConfig("dual_voting_only", 
                   method="proposed", weights="dual", adaptive_k=False),
    
    # Full proposed: Adaptive K + Dual proximity-severity voting
    AblationConfig("full_proposed", 
                   method="proposed", weights="dual", adaptive_k=True),
]


class AblationStudy:
    """
    Comprehensive ablation study for proposed KNN components.
    
    Tests the contribution of each component:
    1. Degree weighting (base paper contribution)
    2. Bounded distance compression (my contribution)
    3. Adaptive K selection (my contribution)
    4. Dual voting (my contribution)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize ablation study.
        
        Parameters
        ----------
        config : dict
            Configuration with keys:
            - 'k': baseline K value
            - 'metric': distance metric
            - 'degree_method': 'base_paper' or 'proposed'
            - 'num_trials': number of trials per variant
            - 'random_seed': base random seed
        """
        if config is None:
            config = {
                'k': 5,
                'metric': 'euclidean',
                'degree_method': 'proposed',
                'num_trials': 10,
                'random_seed': 42
            }
        self.config = config
        self.results = {}
        
    def _create_model(self, variant: AblationConfig) -> Any:
        """Create model instance for a variant."""
        k = self.config.get('k', 5)
        metric = self.config.get('metric', 'euclidean')
        weights = variant.params.get('weights', 'uniform')
        adaptive_k = variant.params.get('adaptive_k', False)
        method = variant.params.get('method', 'baseline')
        
        if method == 'baseline':
            return StandardKNN(k=k, metric=metric, weights=weights)
        elif method == 'improved':
            return ImprovedKNN(k=k, metric=metric, weights=weights)
        elif method == 'proposed':
            return ProposedKNN(k=k, metric=metric, weights=weights, adaptive_k=adaptive_k)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _compute_degrees(self, X: np.ndarray, feature_names: List[str], 
                         method: str = 'proposed') -> np.ndarray:
        """Compute degree of attack for training data."""
        deg_computer = DegreeOfAttackComputer(method=method)
        return deg_computer.compute_degree(X, feature_names=feature_names)
    
    def run_single_variant(self, variant: AblationConfig,
                           X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray,
                           feature_names: List[str],
                           trial_seed: int = None) -> Dict[str, Any]:
        """
        Run a single ablation variant.
        
        Returns
        -------
        dict
            Evaluation metrics for this variant
        """
        if trial_seed is not None:
            set_seed(trial_seed)
        
        # Compute degrees
        degree_method = self.config.get('degree_method', 'proposed')
        train_degrees = self._compute_degrees(X_train, feature_names, degree_method)
        test_degrees = self._compute_degrees(X_test, feature_names, degree_method)
        
        # Create and train model
        model = self._create_model(variant)
        
        start_time = time.perf_counter()
        model.fit(X_train, y_train, train_degrees)
        train_time = time.perf_counter() - start_time
        
        # Predict
        start_time = time.perf_counter()
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        inference_time = time.perf_counter() - start_time
        
        # Ensure proba shape
        if y_proba.ndim == 1:
            y_proba = np.column_stack([1 - y_proba, y_proba])
        
        # Evaluate
        evaluator = DDoSEvaluator()
        metrics = evaluator.evaluate_classification(y_test, y_pred, y_proba, test_degrees)
        
        # Add metadata
        metrics['variant'] = variant.name
        metrics['training_time'] = train_time
        metrics['inference_time'] = inference_time
        metrics['inference_time_per_sample'] = inference_time / len(X_test)
        
        return metrics
    
    def run_multiple_trials(self, variant: AblationConfig,
                           X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray,
                           feature_names: List[str]) -> Dict[str, Any]:
        """
        Run multiple trials for a variant and aggregate statistics.
        """
        num_trials = self.config.get('num_trials', 10)
        base_seed = self.config.get('random_seed', 42)
        
        trial_results = []
        
        for trial in range(num_trials):
            trial_seed = base_seed + trial * 1000 + hash(variant.name) % 1000
            result = self.run_single_variant(variant, X_train, y_train, 
                                             X_test, y_test, feature_names,
                                             trial_seed)
            trial_results.append(result)
        
        # Aggregate
        agg = {
            'variant': variant.name,
            'num_trials': num_trials,
            'trials': trial_results
        }
        
        # Compute statistics for key metrics
        metric_keys = ['accuracy', 'precision', 'recall', 'f1', 'fpr', 'fnr', 
                       'detection_rate', 'auc_roc', 'training_time', 'inference_time']
        
        for key in metric_keys:
            values = [r.get(key, 0) for r in trial_results if key in r]
            if values:
                agg[f'{key}_mean'] = float(np.mean(values))
                agg[f'{key}_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                agg[f'{key}_min'] = float(np.min(values))
                agg[f'{key}_max'] = float(np.max(values))
                # 95% CI
                agg[f'{key}_ci95'] = float(1.96 * np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
        
        return agg
    
    def run_full_ablation(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray,
                          feature_names: List[str],
                          variants: List[AblationConfig] = None) -> Dict[str, Any]:
        """
        Run complete ablation study across all variants.
        """
        if variants is None:
            variants = ABLATION_VARIANTS
        
        print(f"\nRunning ablation study with {len(variants)} variants...")
        print(f"Trials per variant: {self.config.get('num_trials', 10)}")
        
        all_results = {}
        
        for variant in variants:
            print(f"\n  Variant: {variant.name}")
            print(f"    Params: {variant.params}")
            
            try:
                result = self.run_multiple_trials(variant, X_train, y_train, 
                                                  X_test, y_test, feature_names)
                all_results[variant.name] = result
                print(f"    F1: {result.get('f1_mean', 0):.4f} +/- {result.get('f1_std', 0):.4f}")
                print(f"    AUC: {result.get('auc_roc_mean', 0):.4f} +/- {result.get('auc_roc_std', 0):.4f}")
            except Exception as e:
                print(f"    FAIL Failed: {e}")
                all_results[variant.name] = {'variant': variant.name, 'error': str(e)}
        
        self.results = all_results
        return all_results
    
    def compute_ablation_effects(self) -> pd.DataFrame:
        """
        Compute effect sizes comparing each ablated variant to full proposed.
        """
        if not self.results:
            return pd.DataFrame()
        
        # Get full proposed as reference
        full_key = 'full_proposed'
        if full_key not in self.results:
            # Find the most complete variant
            full_key = max(self.results.keys(), key=lambda k: len(self.results[k].get('trials', [])))
        
        full_metrics = self.results.get(full_key, {})
        
        effects = []
        
        for variant_name, result in self.results.items():
            if variant_name == full_key or 'error' in result:
                continue
            
            effect_row = {'variant': variant_name, 'reference': full_key}
            
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']:
                full_val = full_metrics.get(f'{metric}_mean', 0)
                variant_val = result.get(f'{metric}_mean', 0)
                full_std = full_metrics.get(f'{metric}_std', 0)
                variant_std = result.get(f'{metric}_std', 0)
                
                # Absolute difference
                effect_row[f'{metric}_diff'] = variant_val - full_val
                
                # Cohen's d (effect size)
                pooled_std = np.sqrt((full_std**2 + variant_std**2) / 2)
                if pooled_std > 0:
                    effect_row[f'{metric}_cohens_d'] = (variant_val - full_val) / pooled_std
                else:
                    effect_row[f'{metric}_cohens_d'] = 0.0
            
            effects.append(effect_row)
        
        return pd.DataFrame(effects)
    
    def get_summary_table(self) -> pd.DataFrame:
        """Get summary table of all variants."""
        rows = []
        for variant_name, result in self.results.items():
            if 'error' in result:
                continue
            row = {'Variant': variant_name}
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'fpr', 'fnr', 
                          'detection_rate', 'auc_roc']:
                row[metric] = result.get(f'{metric}_mean', 0)
                row[f'{metric}_std'] = result.get(f'{metric}_std', 0)
            row['training_time'] = result.get('training_time_mean', 0)
            row['inference_time_ms'] = result.get('inference_time_mean', 0) * 1000
            rows.append(row)
        
        df = pd.DataFrame(rows)
        if len(df) > 0:
            df = df.sort_values('f1', ascending=False)
        return df


# =============================================================================
# Component-Level Ablation (More Granular)
# =============================================================================

def run_component_ablation(X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray,
                           feature_names: List[str],
                           config: Dict = None) -> Dict[str, Any]:
    """
    Run fine-grained component ablation.
    
    Tests each component independently by toggling on/off:
    - use_degree_weighting: whether to use degree in distance/voting
    - use_distance_compression: whether to use bounded distance compression
    - use_adaptive_k: whether to use adaptive K
    - use_dual_voting: whether to use dual voting (proximity * degree)
    """
    if config is None:
        config = {'k': 5, 'metric': 'euclidean', 'num_trials': 5, 'random_seed': 42}
    
    # Component combinations (2^4 = 16, but we test key ones)
    components = {
        'use_degree_weighting': [False, True],
        'use_distance_compression': [False, True],
        'use_adaptive_k': [False, True],
        'use_dual_voting': [False, True]
    }
    
    # Key combinations to test
    test_configs = [
        # Baseline: nothing
        {'use_degree_weighting': False, 'use_distance_compression': False,
         'use_adaptive_k': False, 'use_dual_voting': False},
        # Base paper: degree weighting only
        {'use_degree_weighting': True, 'use_distance_compression': False,
         'use_adaptive_k': False, 'use_dual_voting': False},
        # Proposed distance only
        {'use_degree_weighting': True, 'use_distance_compression': True,
         'use_adaptive_k': False, 'use_dual_voting': False},
        # + Adaptive K
        {'use_degree_weighting': True, 'use_distance_compression': True,
         'use_adaptive_k': True, 'use_dual_voting': False},
        # + Dual voting
        {'use_degree_weighting': True, 'use_distance_compression': True,
         'use_adaptive_k': False, 'use_dual_voting': True},
        # Full proposed
        {'use_degree_weighting': True, 'use_distance_compression': True,
         'use_adaptive_k': True, 'use_dual_voting': True},
    ]
    
    results = {}
    
    for i, comp_config in enumerate(test_configs):
        name = f"comp_{i}"
        for k, v in comp_config.items():
            if v:
                name += f"_{k}"
        name = name or "none"
        
        print(f"\n  Testing: {name}")
        print(f"    Config: {comp_config}")
        
        # We'd need a custom KNN that accepts these flags
        # For now, map to existing variants
        if not comp_config['use_degree_weighting']:
            variant = AblationConfig(name, method="baseline", weights="uniform", adaptive_k=False)
        elif comp_config['use_degree_weighting'] and not comp_config['use_distance_compression'] and not comp_config['use_adaptive_k'] and not comp_config['use_dual_voting']:
            variant = AblationConfig(name, method="improved", weights="attack_degree", adaptive_k=False)
        elif comp_config['use_degree_weighting'] and comp_config['use_distance_compression'] and not comp_config['use_adaptive_k'] and not comp_config['use_dual_voting']:
            variant = AblationConfig(name, method="proposed", weights="attack_degree", adaptive_k=False)
        elif comp_config['use_degree_weighting'] and comp_config['use_distance_compression'] and comp_config['use_adaptive_k'] and not comp_config['use_dual_voting']:
            variant = AblationConfig(name, method="proposed", weights="attack_degree", adaptive_k=True)
        elif comp_config['use_degree_weighting'] and comp_config['use_distance_compression'] and not comp_config['use_adaptive_k'] and comp_config['use_dual_voting']:
            variant = AblationConfig(name, method="proposed", weights="dual", adaptive_k=False)
        else:
            variant = AblationConfig(name, method="proposed", weights="dual", adaptive_k=True)
        
        study = AblationStudy(config)
        result = study.run_multiple_trials(variant, X_train, y_train, X_test, y_test, feature_names)
        results[name] = result
    
    return results


if __name__ == "__main__":
    print("Testing ablation study module...")
    
    # Create synthetic data
    np.random.seed(42)
    n_train = 500
    n_test = 200
    n_features = 10
    
    X_train = np.random.randn(n_train, n_features)
    X_train = np.abs(X_train)
    X_train[250:] *= 3
    y_train = np.array([0] * 250 + [1] * 250)
    
    X_test = np.random.randn(n_test, n_features)
    X_test = np.abs(X_test)
    X_test[:120] *= 3
    y_test = np.array([1] * 120 + [0] * 80)
    
    feature_names = [f"f_{i}" for i in range(n_features)]
    
    # Run quick ablation (fewer trials for testing)
    config = {'k': 5, 'metric': 'euclidean', 'num_trials': 3, 'random_seed': 42}
    study = AblationStudy(config)
    
    # Test a few variants
    test_variants = [
        AblationConfig("baseline_knn", method="baseline", weights="uniform", adaptive_k=False),
        AblationConfig("base_paper", method="improved", weights="attack_degree", adaptive_k=False),
        AblationConfig("full_proposed", method="proposed", weights="dual", adaptive_k=True),
    ]
    
    results = study.run_full_ablation(X_train, y_train, X_test, y_test, feature_names, test_variants)
    
    # Print summary
    summary = study.get_summary_table()
    print("\n=== Ablation Summary ===")
    print(summary[['Variant', 'accuracy', 'f1', 'auc_roc']].to_string(index=False))
    
    # Compute effects
    effects = study.compute_ablation_effects()
    print("\n=== Ablation Effects (vs Full Proposed) ===")
    print(effects[['variant', 'f1_diff', 'f1_cohens_d', 'auc_roc_diff', 'auc_roc_cohens_d']].to_string(index=False))
    
    print("\n[PASS] Ablation study test complete!")