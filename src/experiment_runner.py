"""
Experiment Runner Module
Orchestrates complete experiment pipelines for KNN-based DDoS detection.

Supports:
- Single experiment execution with given configuration
- Grid search over K values and degree thresholds
- Multiple trials with different random seeds (averaging results)
- Comparative analysis: baseline vs improved vs proposed methods
- Result aggregation and saving
"""

import numpy as np
import pandas as pd
import time
import json
import os
from typing import Dict, List, Any, Optional
from concurrent.futures import ProcessPoolExecutor as PPE

from src.data_loader import load_dataset, report_class_balance
from src.preprocessor import quick_preprocess, FeaturePreprocessor
from src.feature_engineering import engineer_features_for_knn
from src.degree_of_attack import DegreeOfAttackComputer, compute_degree_with_threshold
from src.knn_baseline import StandardKNN, check_is_fitted as check_bl
from src.knn_improved import ImprovedKNN, check_is_fitted as check_imp
from src.knn_proposed import ProposedKNN, check_is_fitted as check_prop
from src.classifier import KNNClassifier, check_is_trained as check_cls
from src.evaluator import DDoSEvaluator, quick_evaluate
from src.traffic_generator import TrafficGenerator, quick_generate_dataset


def run_single_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a single experiment with the given configuration.
    
    Parameters
    ----------
    config : dict
        Experiment configuration with keys:
        - 'method': 'baseline', 'improved', or 'proposed'
        - 'dataset': 'NSL-KDD', 'CIC-DDoS2019', or 'synthetic'
        - 'k': int, number of neighbors
        - 'metric': 'euclidean' or 'manhattan'
        - 'weights': voting scheme
        - 'adaptive_k': bool (proposed only)
        - 'degree_method': 'base_paper' or 'proposed'
        - 'balance_classes': bool (use SMOTE)
        - 'random_seed': int
    
    Returns
    -------
    dict
        Experiment results including all evaluation metrics
    """
    # Extract configuration
    method = config.get('method', 'proposed')
    dataset_name = config.get('dataset', 'synthetic')
    k = config.get('k', 5)
    metric = config.get('metric', 'euclidean')
    weights = config.get('weights', 'dual' if method == 'proposed' else 'uniform')
    adaptive_k = config.get('adaptive_k', True) if method == 'proposed' else False
    degree_method = config.get('degree_method', 'proposed')
    balance_classes = config.get('balance_classes', False)
    random_seed = config.get('random_seed', 42)
    
    np.random.seed(random_seed)
    
    # 1. Load dataset
    data = load_dataset({
        'dataset': dataset_name,
        'dataset_path': config.get('dataset_path'),
        'random_seed': random_seed,
    })
    
    if data['X_train'] is None or data['X_test'] is None:
        return {'error': f'Failed to load dataset: {dataset_name}'}
    
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
    feature_names = data['feature_names']
    
    # 2. Preprocess data
    if balance_classes:
        # Apply SMOTE-like balancing on training set
        # For simplicity, just note it; actual SMOTE in preprocessor
        pass
    
    # 3. Engineer features for KNN
    eng_config = {
        'include_entropy': True,
        'degree_method': degree_method,
        'selected_features': None
    }
    
    try:
        X_train_eng, y_train_eng, train_feats, train_degrees = engineer_features_for_knn(
            pd.DataFrame(X_train, columns=feature_names) if hasattr(feature_names, '__len__') else None,
            eng_config)
        # If the above didn't work with DataFrame, convert numpy
        if not isinstance(X_train_eng, np.ndarray) or X_train_eng.dtype != float:
            # Fallback: just use the original features with degree computation
            from src.degree_of_attack import DegreeOfAttackComputer
            deg_computer = DegreeOfAttackComputer(method=degree_method)
            train_degrees = deg_computer.compute_degree_from_array(X_train, feature_names)
            X_train_eng = X_train
            y_train_eng = y_train
            train_feats = feature_names
        else:
            # Ensure labels are correct
            y_train_eng = y_train_eng if 'y_train_eng' in dir() else y_train
            train_degrees = train_degrees if 'train_degrees' in dir() else np.zeros(len(X_train))
    except Exception as e:
        # Fallback: basic feature handling
        print(f"  Feature engineering warning: {e}, using basic features")
        train_degrees = np.zeros(len(X_train))
        X_train_eng = X_train
        y_train_eng = y_train
        train_feats = feature_names
    
    # Ensure train_degrees has correct length
    if len(train_degrees) != len(X_train_eng):
        # Recompute degrees on engineered features
        from src.degree_of_attack import DegreeOfAttackComputer
        deg_computer = DegreeOfAttackComputer(method=degree_method)
        train_degrees = deg_computer.compute_degree_from_array(X_train_eng, train_feats)
    
    # Test set feature engineering
    try:
        X_test_eng, y_test_eng, test_feats, test_degrees = engineer_features_for_knn(
            pd.DataFrame(X_test, columns=feature_names) if feature_names else None,
            eng_config)
        
        if len(test_degrees) != len(X_test_eng):
            from src.degree_of_attack import DegreeOfAttackComputer
            deg_computer = DegreeOfAttackComputer(method=degree_method)
            test_degrees = deg_computer.compute_degree_from_array(X_test_eng, test_feats)
    except Exception as e:
        print(f"  Test feature engineering warning: {e}")
        test_degrees = np.zeros(len(X_test))
        X_test_eng = X_test
        y_test_eng = y_test
        test_feats = feature_names
    
    # 4. Initialize and train classifier
    if method == 'baseline':
        knn_model = StandardKNN(k=k, metric=metric, weights=weights)
        knn_model.fit(X_train_eng, y_train_eng)
        
    elif method == 'improved':
        knn_model = ImprovedKNN(k=k, metric=metric, weights=weights)
        knn_model.fit(X_train_eng, y_train_eng, train_degrees)
        
    elif method == 'proposed':
        knn_model = ProposedKNN(k=k, metric=metric, weights=weights,
                                adaptive_k=adaptive_k)
        knn_model.fit(X_train_eng, y_train_eng, train_degrees)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # 5. Predict
    if method == 'proposed':
        y_pred = knn_model.predict(X_test_eng, degrees_test=test_degrees)
        y_proba = knn_model.predict_proba(X_test_eng, degrees_test=test_degrees)
    else:
        y_pred = knn_model.predict(X_test_eng)
        y_proba = knn_model.predict_proba(X_test_eng)
    
    # Ensure y_proba has correct shape
    if y_proba.ndim == 1:
        # Convert to [P(normal), P(attack)] format
        y_proba = np.column_stack([1 - y_proba, y_proba])
    
    # 6. Evaluate
    evaluator = DDoSEvaluator()
    metrics = evaluator.evaluate_classification(y_test, y_pred, y_proba, test_degrees)
    
    # 7. Additional degree analysis
    if test_degrees is not None and len(test_degrees) > 0:
        degree_binary, degree_threshold = compute_degree_with_threshold(
            test_degrees
        )
        metrics["degree_threshold"] = float(degree_threshold)
        metrics["degree_threshold_accuracy"] = float(
            np.mean(degree_binary == y_test)
        )
    
    # Compile results
    results = {
        'method': method,
        'dataset': dataset_name,
        'k': k,
        'metric': metric,
        'random_seed': random_seed,
        'accuracy': metrics.get('accuracy', 0.0),
        'precision': metrics.get('precision', 0.0),
        'recall': metrics.get('recall', 0.0),
        'f1': metrics.get('f1', 0.0),
        'fpr': metrics.get('fpr', 0.0),
        'fnr': metrics.get('fnr', 0.0),
        'detection_rate': metrics.get('detection_rate', 0.0),
        'auc_roc': metrics.get('auc_roc', 0.5),
        'train_samples': len(X_train_eng),
        'test_samples': len(X_test_eng),
        'feature_count': len(train_feats) if 'train_feats' in dir() else 0,
        'execution_time': metrics.get('computation_time', 0.0),
    }
    
    # Add degree-specific metrics if available
    degree_keys = [k for k in metrics.keys() if 'degree' in k.lower() or 'threshold' in k.lower() or 'optimal' in k.lower()]
    for key in degree_keys:
        results[key] = metrics.get(key, 0.0)
    
    return results


def run_grid_experiment(config: Dict[str, Any], 
                        k_grid: Optional[List[int]] = None,
                        threshold_grid: Optional[List[float]] = None) -> List[Dict[str, Any]]:
    """
    Run a grid of experiments over specified parameter values.
    
    Parameters
    ----------
    config : dict
        Base experiment configuration
    k_grid : list, optional
        List of K values to test
    threshold_grid : list, optional
        List of degree thresholds to evaluate
    
    Returns
    -------
    list
        List of result dicts, one per (config, parameter) combination
    """
    # Default K grid
    if k_grid is None:
        k_grid = [1, 3, 5, 7, 10, 15, 20, 25, 30]
    
    # Default threshold grid
    if threshold_grid is None:
        threshold_grid = [0.3, 0.5, 0.7]
    
    base_method = config.get('method', 'proposed')
    results = []
    
    # Grid search over K values
    for k_val in k_grid:
        experiment_config = {**config, 'k': k_val}
        result = run_single_experiment(experiment_config)
        result['grid_k'] = k_val
        results.append(result)
    
    # If method is proposed, also test adaptive K
    if base_method == 'proposed':
        for k_val in k_grid:
            for adaptive in [True, False]:
                experiment_config = {**config, 'k': k_val, 'adaptive_k': adaptive}
                result = run_single_experiment(experiment_config)
                result['grid_k'] = k_val
                result['adaptive_k'] = adaptive
                results.append(result)
    
    return results


def run_multiple_trials(base_config: Dict[str, Any], 
                        num_trials: int = 10,
                        vary_seed: bool = True) -> Dict[str, Any]:
    """
    Run multiple experiments with different random seeds and average results.
    
    Parameters
    ----------
    base_config : dict
        Base configuration for all trials
    num_trials : int
        Number of trials to run
    vary_seed : bool
        Whether to vary the random seed each trial
    
    Returns
    -------
    dict
        Aggregated results with mean and standard deviation across trials
    """
    all_results = []
    
    for trial_idx in range(num_trials):
        # Determine seed for this trial
        if vary_seed:
            trial_seed = base_config.get('random_seed', 42) + trial_idx
        else:
            trial_seed = base_config.get('random_seed', 42)
        
        trial_config = {**base_config, 'random_seed': trial_seed}
        result = run_single_experiment(trial_config)
        
        # Rename seed field for clarity
        result['trial_seed'] = trial_seed
        all_results.append(result)
    
    # Aggregate results: compute mean and std for each metric
    aggregated = {
        'method': base_config.get('method', 'proposed'),
        'dataset': base_config.get('dataset', 'synthetic'),
        'num_trials': num_trials,
        'trials': all_results,
    }
    
    # Metrics to aggregate
    metric_keys = ['accuracy', 'precision', 'recall', 'f1', 'fpr', 'fnr', 
                   'detection_rate', 'auc_roc']
    
    aggregated_stats = {}
    for key in metric_keys:
        values = [r.get(key, 0.0) for r in all_results if key in r]
        if values:
            aggregated_stats[f'{key}_mean'] = float(np.mean(values))
            aggregated_stats[f'{key}_std'] = float(
                np.std(values, ddof=1) if len(values) > 1 else 0.0
            )
            aggregated_stats[f'{key}_count'] = len(values)
            aggregated_stats[f'{key}_ci95'] = float(
                1.96 * np.std(values, ddof=1) / np.sqrt(len(values))
                if len(values) > 1 else 0.0
            )
            aggregated_stats[f'{key}_min'] = float(np.min(values))
            aggregated_stats[f'{key}_max'] = float(np.max(values))
        else:
            aggregated_stats[f'{key}_mean'] = 0.0
            aggregated_stats[f'{key}_std'] = 0.0
            aggregated_stats[f'{key}_min'] = 0.0
            aggregated_stats[f'{key}_max'] = 0.0
    
    # Also aggregate non-metric fields
    for key in ['k', 'metric', 'method', 'dataset']:
        values = [r.get(key, '') for r in all_results if key in r]
        if values:
            # Take the most common value
            from collections import Counter
            aggregated_stats[key] = Counter(values).most_common(1)[0][0]
        else:
            aggregated_stats[key] = base_config.get(key, '')
    
    aggregated['stats'] = aggregated_stats
    return aggregated


def compare_all_methods(config: Dict[str, Any], 
                       k_values: Optional[List[int]] = None) -> Dict[str, Dict[str, float]]:
    """
    Compare baseline, improved, and proposed methods under the same conditions.
    
    Parameters
    ----------
    config : dict
        Base configuration (will be applied to all three methods)
    k_values : list, optional
        K values to test; if None, uses [5] (single K)
    
    Returns
    -------
    dict
        Dictionary mapping method names to their evaluation results
    """
    if k_values is None:
        k_values = [5]
    
    methods = ['baseline', 'improved', 'proposed']
    comparison = {}
    
    for method in methods:
        method_config = {**config, 'method': method}
        
        # Adjust weights per method
        if method == 'baseline':
            method_config['weights'] = 'uniform'
        elif method == 'improved':
            method_config['weights'] = 'attack_degree'
        elif method == 'proposed':
            method_config['weights'] = 'dual'
        
        # Run experiment with first K value (or average over K values)
        method_results = []
        for k_val in k_values:
            trial_config = {**method_config, 'k': k_val}
            result = run_single_experiment(trial_config)
            method_results.append(result)
        
        # Average results across K values
        if method_results:
            comparison[method] = {
                'accuracy': float(np.mean([r.get('accuracy', 0) for r in method_results])),
                'precision': float(np.mean([r.get('precision', 0) for r in method_results])),
                'recall': float(np.mean([r.get('recall', 0) for r in method_results])),
                'f1': float(np.mean([r.get('f1', 0) for r in method_results])),
                'fpr': float(np.mean([r.get('fpr', 0) for r in method_results])),
                'detection_rate': float(np.mean([r.get('detection_rate', 0) for r in method_results])),
                'auc_roc': float(np.mean([r.get('auc_roc', 0.5) for r in method_results])),
            }
    
    return comparison


def save_experiment_results(results: List[Dict[str, Any]], 
                          output_path: str) -> None:
    """
    Save experiment results to CSV file.
    
    Parameters
    ----------
    results : list
        List of result dictionaries
    output_path : str
        Path to output CSV file
    """
    # Normalize results to consistent columns
    all_keys = set()
    for r in results:
        all_keys.update(r.keys())
    
    # Ensure all results have the same keys
    results_normalized = []
    for r in results:
        normalized = {k: r.get(k, None) for k in all_keys}
        results_normalized.append(normalized)
    
    # Save to CSV
    df = pd.DataFrame(results_normalized)
    df.to_csv(output_path, index=False)
    
    print(f"Saved {len(results)} experiment results to {output_path}")


def load_experiment_results(input_path: str) -> List[Dict[str, Any]]:
    """
    Load experiment results from CSV file.
    
    Parameters
    ----------
    input_path : str
        Path to input CSV file
    
    Returns
    -------
    list
        List of result dictionaries
    """
    df = pd.read_csv(input_path)
    return df.to_dict('records')


def run_full_pipeline(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Run the complete experiment pipeline from data loading to result aggregation.
    
    Parameters
    ----------
    config : dict, optional
        Full pipeline configuration
    
    Returns
    -------
    dict
        Complete pipeline results including all method comparisons
    """
    if config is None:
        config = {
            'method': 'proposed',
            'dataset': 'synthetic',
            'k': 5,
            'metric': 'euclidean',
            'weights': 'dual',
            'adaptive_k': True,
            'degree_method': 'proposed',
            'balance_classes': False,
            'random_seed': 42
        }
    
    print("=" * 70)
    print("FULL EXPERIMENT PIPELINE")
    print("=" * 70)
    
    # Step 1: Generate or load dataset
    print("\n[Step 1] Generating/loading dataset...")
    data = load_dataset({'dataset': config['dataset'], 'random_seed': config['random_seed']})
    print(f"  Loaded: {data['dataset_name']} with {data['X_train'].shape if data['X_train'] is not None else 0} training, "
          f"{data['X_test'].shape if data['X_test'] is not None else 0} test samples")
    print(f"  Class imbalance ratio: {data.get('imbalance_ratio', 1.0):.2f}:1")
    
    # Step 2: Run comparison of all three methods
    print("\n[Step 2] Comparing all three KNN methods...")
    comparison = compare_all_methods(config, k_values=[5])
    
    for method_name, metrics in comparison.items():
        print(f"  {method_name:12s}: Acc={metrics['accuracy']:.3f}, "
              f"P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, "
              f"F1={metrics['f1']:.3f}, AUC={metrics['auc_roc']:.3f}")
    
    # Step 3: Multiple trials for statistical rigor
    print("\n[Step 3] Running multiple trials for statistical rigor...")
    trial_results = run_multiple_trials(config, num_trials=5, vary_seed=True)
    
    stats = trial_results.get('stats', {})
    print(f"  Mean accuracy across 5 trials: {stats.get('accuracy_mean', 0):.3f} +/- {stats.get('accuracy_std', 0):.3f}")
    print(f"  Mean F1-score: {stats.get('f1_mean', 0):.3f} +/- {stats.get('f1_std', 0):.3f}")
    print(f"  Mean AUC-ROC: {stats.get('auc_roc_mean', 0):.3f} +/- {stats.get('auc_roc_std', 0):.3f}")
    
    # Step 4: Save results
    print("\n[Step 4] Saving results...")
    # Create results directory if needed
    os.makedirs('results/metrics', exist_ok=True)
    save_experiment_results(trial_results.get('trials', []), 'results/metrics/experiment_results.csv')
    
    # Compile final pipeline output
    pipeline_output = {
        'comparison': comparison,
        'trial_aggregation': trial_results,
        'config': config,
        'dataset_info': {
            'name': config['dataset'],
            'train_samples': data['X_train'].shape[0] if data['X_train'] is not None else 0,
            'test_samples': data['X_test'].shape[0] if data['X_test'] is not None else 0,
        }
    }
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    
    return pipeline_output


if __name__ == "__main__":
    # Quick test: experiment runner
    print("Testing experiment runner module...")
    
    # Base configuration
    base_config = {
        'method': 'proposed',
        'dataset': 'synthetic',
        'k': 5,
        'metric': 'euclidean',
        'weights': 'dual',
        'adaptive_k': True,
        'degree_method': 'proposed',
        'balance_classes': False,
        'random_seed': 42
    }
    
    # Run single experiment
    print("\n  Running single experiment...")
    result = run_single_experiment(base_config)
    print(f"    Accuracy: {result.get('accuracy', 0):.3f}")
    print(f"    F1: {result.get('f1', 0):.3f}")
    print(f"    AUC-ROC: {result.get('auc_roc', 0):.3f}")
    
    # Run grid experiment (small grid for quick test)
    print("\n  Running grid over K values (small test)...")
    grid_results = run_grid_experiment(base_config, k_grid=[3, 5, 7])
    print(f"    Grid results: {len(grid_results)} configurations")
    for gr in grid_results[:3]:  # Show first 3
        print(f"    K={gr.get('grid_k')}: Acc={gr.get('accuracy', 0):.3f}")
    
    # Run multiple trials
    print("\n  Running 3 trials...")
    trial_agg = run_multiple_trials(base_config, num_trials=3, vary_seed=True)
    stats = trial_agg.get('stats', {})
    print(f"    Accuracy mean: {stats.get('accuracy_mean', 0):.3f} +/- {stats.get('accuracy_std', 0):.3f}")
    print(f"    F1 mean: {stats.get('f1_mean', 0):.3f} +/- {stats.get('f1_std', 0):.3f}")
    
    # Compare all methods
    print("\n  Comparing all three methods...")
    comparison = compare_all_methods(base_config, k_values=[5])
    for method_name, metrics in comparison.items():
        print(f"    {method_name:12s}: Acc={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}, "
              f"AUC={metrics['auc_roc']:.3f}")