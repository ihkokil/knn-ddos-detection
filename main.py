"""
Main Entry Point
Executes the complete DDoS detection research pipeline.

Usage:
    python main.py --config config/config.yaml
    python main.py --method proposed --dataset synthetic --k 5
    python main.py --phase full_eval --datasets NSL-KDD CIC-DDoS2019 --num-trials 30

Phases:
1. Data loading and preprocessing
2. Feature engineering
3. KNN classification (baseline/improved/proposed)
4. Evaluation and metrics computation
5. Experiment execution (single/multiple/grid)
6. Result visualization
7. Result saving
8. Comprehensive Evaluation (full empirical study)
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd
from typing import Any, Dict

# Add project root and src to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from src.data_loader import load_dataset, report_class_balance
from src.preprocessor import quick_preprocess
from src.feature_engineering import engineer_features_for_knn, create_synthetic_ddos_dataset
from src.degree_of_attack import DegreeOfAttackComputer, compute_degree_with_threshold
from src.knn_baseline import StandardKNN
from src.knn_improved import ImprovedKNN
from src.knn_proposed import ProposedKNN
from src.classifier import KNNClassifier, check_is_trained as check_cls
from src.evaluator import DDoSEvaluator, quick_evaluate
from src.traffic_generator import TrafficGenerator, quick_generate_dataset
from src.experiment_runner import (
    run_single_experiment, run_grid_experiment, 
    run_multiple_trials, compare_all_methods, 
    save_experiment_results, load_experiment_results,
    run_full_pipeline
)
from src.visualizer import ExperimentVisualizer, create_visualizer
from src.utils import (
    Timer, ConfigLoader, set_seed, ensure_dir,
    generate_experiment_id, format_metrics_report, timer_decorator,
    safe_divide
)

# Comprehensive evaluation module
try:
    from src.full_evaluation import run_full_evaluation, FullEvaluationConfig, quick_test
    EVAL_AVAILABLE = True
    Q1_AVAILABLE = True
except ImportError:
    EVAL_AVAILABLE = False
    Q1_AVAILABLE = False
    print("Note: Comprehensive evaluation module not available")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='DDoS Detection Research Pipeline'
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/config.yaml',
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--phase', 
        type=str, 
        choices=['data', 'train', 'evaluate', 'experiment', 'visualize', 'full', 'full_eval', 'q1'],
        default='full',
        help='Execution phase (use full_eval for comprehensive study)'
    )
    
    parser.add_argument(
        '--method', 
        type=str, 
        choices=['baseline', 'improved', 'proposed'],
        default=None,
        help='KNN method: baseline, improved, or proposed'
    )
    
    parser.add_argument(
        '--dataset', 
        type=str, 
        choices=['NSL-KDD', 'CIC-DDoS2019', 'InSDN', 'synthetic'],
        default=None,
        help='Dataset to use (legacy, use --datasets for full_eval)'
    )
    parser.add_argument(
        '--datasets', 
        nargs='+',
        choices=['NSL-KDD', 'CIC-DDoS2019', 'InSDN', 'both', 'all'],
        default=None,
        help='Datasets for comprehensive evaluation (can specify multiple)'
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        default=None,
        help='Dataset directory, resolved relative to this project'
    )
    
    parser.add_argument(
        '--k', 
        type=int, 
        default=None,
        help='K value for KNN'
    )
    
    parser.add_argument(
        '--trials', 
        type=int, 
        default=1,
        help='Number of experiment trials (legacy, use --num-trials for full_eval)'
    )
    parser.add_argument(
        '--num-trials', 
        type=int, 
        default=30,
        help='Number of independent trials for comprehensive evaluation'
    )
    
    parser.add_argument(
        '--grid_k', 
        action='store_true',
        help='Run grid search over K values'
    )
    
    parser.add_argument(
        '--visualize', 
        action='store_true',
        help='Generate plots from saved results'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        default='results',
        help='Output directory base path'
    )
    
    # Comprehensive evaluation specific options
    parser.add_argument(
        '--no-baselines', 
        action='store_true',
        help='Disable baseline comparisons (RF, SVM, XGBoost, LightGBM, MLP)'
    )
    parser.add_argument(
        '--no-ablations', 
        action='store_true',
        help='Disable ablation studies'
    )
    parser.add_argument(
        '--no-stats', 
        action='store_true',
        help='Disable statistical tests'
    )
    parser.add_argument(
        '--sdn', 
        action='store_true',
        help='Enable SDN evaluation (requires Mininet/Ryu, root)'
    )
    parser.add_argument(
        '--quick-test', 
        action='store_true',
        help='Run quick test with synthetic data'
    )
    parser.add_argument(
        '--download-data', 
        action='store_true',
        help='Download datasets if not present'
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)
    from src.utils import ConfigLoader
    config = ConfigLoader.load_yaml(config_path)
    print(f"Loaded configuration from {config_path}")
    # Convert the structured YAML schema into the flat schema used by the
    # experiment runner while retaining explicit command-line overrides.
    dataset_cfg = config.get("dataset", {})
    knn_cfg = config.get("knn", {})
    degree_cfg = config.get("degree_of_attack", {})
    experiment_cfg = config.get("experiment", {})
    raw_path = dataset_cfg.get("raw_path", "data/raw")
    if not os.path.isabs(raw_path):
        raw_path = os.path.join(project_root, raw_path)
    return {
        **config,
        "dataset": dataset_cfg.get("name", "synthetic"),
        "dataset_path": os.path.join(
            raw_path,
            "cicddos2019" if dataset_cfg.get("name") == "CIC-DDoS2019"
            else "nsl_kdd" if dataset_cfg.get("name") == "NSL-KDD"
            else ""
        ),
        "random_seed": dataset_cfg.get("random_seed", 42),
        "k": knn_cfg.get("default_k", 5),
        "metric": knn_cfg.get("default_metric", "euclidean"),
        "adaptive_k": knn_cfg.get("adaptive_k", True),
        "degree_method": degree_cfg.get("method", "proposed"),
        "num_trials": experiment_cfg.get("num_trials", 10),
    }


def phase_data(config: Dict[str, Any]):
    """Phase 1: Data loading and exploration."""
    print("\n" + "="*60)
    print("PHASE 1: DATA LOADING AND EXPLORATION")
    print("="*60)
    
    dataset_name = config.get('dataset', 'synthetic')
    
    # Load dataset
    data = load_dataset({'dataset': dataset_name, 'random_seed': config.get('random_seed', 42)})
    
    if data['X_train'] is None:
        print(f"! Using synthetic data generation...")
        # Generate synthetic dataset
        data = load_dataset({'dataset': 'synthetic', 'random_seed': 42})
    
    # Report class balance
    report_class_balance(data['y_train'], data['y_test'], dataset_name)
    
    # Basic statistics
    print(f"\n  Dataset: {data['dataset_name']}")
    print(f"  Train samples: {data['X_train'].shape[0]}")
    print(f"  Test samples: {data['X_test'].shape[0]}")
    print(f"  Feature names: {len(data['feature_names'])} features")
    print(f"  Imbalance ratio (train): {data.get('imbalance_ratio', 1.0):.2f}:1")
    
    return data


def phase_train(data: Any, config: Dict[str, Any]):
    """Phase 2: Train KNN classifiers."""
    print("\n" + "="*60)
    print("PHASE 2: TRAINING KNN CLASSIFIERS")
    print("="*60)
    
    method = config.get('method', 'proposed')
    k = config.get('k', 5)
    metric = config.get('metric', 'euclidean')
    degree_method = config.get('degree_method', 'proposed')
    balance_classes = config.get('balance_classes', False)
    random_seed = config.get('random_seed', 42)
    
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
    feature_names = data['feature_names']
    
    # Feature engineering
    print(f"\n  Feature engineering (method: {degree_method} degree)...")
    eng_config = {
        'include_entropy': True,
        'degree_method': degree_method,
        'selected_features': None
    }
    
    try:
        X_train_eng, y_train_eng, train_feats, train_degrees = engineer_features_for_knn(
            None, eng_config)  # None means use DataFrame internally
        
        # Ensure consistent lengths
        if len(train_degrees) != len(X_train_eng):
            from src.degree_of_attack import DegreeOfAttackComputer
            deg_computer = DegreeOfAttackComputer(method=degree_method)
            train_degrees = deg_computer.compute_degree_from_array(X_train_eng, train_feats)
        
        # Test set
        # Try engineering test features
        try:
            X_test_eng, y_test_eng, test_feats, test_degrees = engineer_features_for_knn(
                None, eng_config)
            
            if len(test_degrees) != len(X_test_eng):
                from src.degree_of_attack import DegreeOfAttackComputer
                deg_computer = DegreeOfAttackComputer(method=degree_method)
                test_degrees = deg_computer.compute_degree_from_array(X_test_eng, test_feats)
        except Exception as e:
            print(f"  Test feature warning: {e}")
            X_test_eng = X_test
            y_test_eng = y_test
            test_degrees = np.zeros(len(X_test))
            test_feats = feature_names
    except Exception as e:
        print(f"  Feature engineering fallback: {e}")
        from src.degree_of_attack import DegreeOfAttackComputer
        deg_computer = DegreeOfAttackComputer(method=degree_method)
        train_degrees = deg_computer.compute_degree_from_array(X_train, feature_names)
        X_train_eng = X_train
        y_train_eng = y_train
        train_degrees = train_degrees
        X_test_eng = X_test
        y_test_eng = y_test
        test_degrees = deg_computer.compute_degree_from_array(X_test, feature_names)
        test_feats = feature_names
    
    # Print degree statistics
    print(f"  Train degree range: [{train_degrees.min():.3f}, {train_degrees.max():.3f}]")
    print(f"  Test degree range:  [{test_degrees.min():.3f}, {test_degrees.max():.3f}]")
    print(f"  Train degrees > 0.5: {np.sum(train_degrees > 0.5)} / {len(train_degrees)}")
    print(f"  Test degrees > 0.5:  {np.sum(test_degrees > 0.5)} / {len(test_degrees)}")
    
    # Train three KNN variants
    print(f"\n  Training KNN variants (k={k}, metric={metric})...")
    
    # Baseline KNN
    print(f"  [Baseline] Standard KNN...")
    baseline = StandardKNN(k=k, metric=metric, weights='uniform')
    baseline.fit(X_train_eng, y_train_eng)
    baseline_pred = baseline.predict(X_test_eng)
    baseline_proba = baseline.predict_proba(X_test_eng)
    
    # Improved KNN (base paper)
    print(f"  [Improved] Improved KNN (degree-weighted)...")
    improved = ImprovedKNN(k=k, metric=metric, weights='attack_degree')
    improved.fit(X_train_eng, y_train_eng, train_degrees)
    improved_pred = improved.predict(X_test_eng)
    improved_proba = improved.predict_proba(X_test_eng)
    
    # Proposed KNN
    print(f"  [Proposed] Proposed KNN (adaptive + dual weighting)...")
    proposed = ProposedKNN(k=k, metric=metric, weights='dual', adaptive_k=True)
    proposed.fit(X_train_eng, y_train_eng, train_degrees)
    proposed_pred = proposed.predict(X_test_eng)
    proposed_proba = proposed.predict_proba(X_test_eng)
    
    # Evaluate all three
    evaluator = DDoSEvaluator()
    
    print(f"\n  Evaluation results:")
    baseline_metrics = evaluator.evaluate_classification(y_test, baseline_pred, baseline_proba, train_degrees[:len(test_degrees)] if len(train_degrees) >= len(test_degrees) else train_degrees)
    improved_metrics = evaluator.evaluate_classification(y_test, improved_pred, improved_proba, train_degrees[:len(test_degrees)] if len(train_degrees) >= len(test_degrees) else train_degrees)
    proposed_metrics = evaluator.evaluate_classification(y_test, proposed_pred, proposed_proba, test_degrees)
    
    # Display results
    results_table = {
        'Method': ['Baseline', 'Improved (base paper)', 'Proposed (my contribution)'],
        'Accuracy': [
            baseline_metrics.get('accuracy', 0),
            improved_metrics.get('accuracy', 0),
            proposed_metrics.get('accuracy', 0)
        ],
        'Precision': [
            baseline_metrics.get('precision', 0),
            improved_metrics.get('precision', 0),
            proposed_metrics.get('precision', 0)
        ],
        'Recall': [
            baseline_metrics.get('recall', 0),
            improved_metrics.get('recall', 0),
            proposed_metrics.get('recall', 0)
        ],
        'F1': [
            baseline_metrics.get('f1', 0),
            improved_metrics.get('f1', 0),
            proposed_metrics.get('f1', 0)
        ],
        'FPR': [
            baseline_metrics.get('fpr', 0),
            improved_metrics.get('fpr', 0),
            proposed_metrics.get('fpr', 0)
        ],
        'Detection Rate': [
            baseline_metrics.get('detection_rate', 0),
            improved_metrics.get('detection_rate', 0),
            proposed_metrics.get('detection_rate', 0)
        ],
    }
    
    results_df = pd.DataFrame(results_table)
    print(results_df.to_string(index=False))
    
    # Degree-specific analysis
    print(f"\n  Degree-of-Attack Analysis:")
    degree_labels, degree_threshold = compute_degree_with_threshold(test_degrees)
    degree_metrics = evaluator.evaluate_classification(
        y_test, degree_labels
    )
    print(f"  Fixed degree threshold: {degree_threshold:.3f}")
    print(f"  Degree-threshold accuracy: {degree_metrics['accuracy']:.3f}")
    
    return {
        'baseline': baseline_metrics,
        'improved': improved_metrics,
        'proposed': proposed_metrics,
        'train_degrees': train_degrees,
        'test_degrees': test_degrees,
        'results_table': results_df
    }


def phase_evaluate(results: Dict[str, Any], config: Dict[str, Any]):
    """Phase 3: Evaluation and analysis."""
    print("\n" + "="*60)
    print("PHASE 3: EVALUATION AND ANALYSIS")
    print("="*60)
    
    # Generate visualization
    visualizer = create_visualizer(config)
    
    # Plot method comparison
    comparison = {
        'Baseline': {
            'accuracy': results['baseline'].get('accuracy', 0),
            'precision': results['baseline'].get('precision', 0),
            'recall': results['baseline'].get('recall', 0),
            'f1': results['baseline'].get('f1', 0),
            'auc_roc': results['baseline'].get('auc_roc', 0.5)
        },
        'Improved (base paper)': {
            'accuracy': results['improved'].get('accuracy', 0),
            'precision': results['improved'].get('precision', 0),
            'recall': results['improved'].get('recall', 0),
            'f1': results['improved'].get('f1', 0),
            'auc_roc': results['improved'].get('auc_roc', 0.5)
        },
        'Proposed (my contribution)': {
            'accuracy': results['proposed'].get('accuracy', 0),
            'precision': results['proposed'].get('precision', 0),
            'recall': results['proposed'].get('recall', 0),
            'f1': results['proposed'].get('f1', 0),
            'auc_roc': results['proposed'].get('auc_roc', 0.5)
        }
    }
    
    visualizer.plot_method_comparison_bar(
        comparison, 
        save_name=f"{config.get('method', 'comparison')}_comparison.png"
    )
    
    # Plot K-vs-accuracy if we have K values
    if config.get('grid_search_k', False):
        # This would need K vs accuracy data from grid search
        pass
    
    # Plot degree distribution
    test_degrees = results.get('test_degrees', np.array([]))
    y_test = None  # Would need true test labels
    
    if len(test_degrees) > 0 and y_test is not None:
        # Separate by true class
        normal_deg = test_degrees[y_test == 0]
        attack_deg = test_degrees[y_test == 1]
        
        visualizer.plot_degree_distribution(
            normal_deg, attack_deg,
            save_name="degree_distribution.png"
        )
    
    # Print formatted report
    print("\n" + format_metrics_report({
        'Baseline Accuracy': results['baseline'].get('accuracy', 0),
        'Improved Accuracy': results['improved'].get('accuracy', 0),
        'Proposed Accuracy': results['proposed'].get('accuracy', 0),
        'Baseline F1': results['baseline'].get('f1', 0),
        'Improved F1': results['improved'].get('f1', 0),
        'Proposed F1': results['proposed'].get('f1', 0),
        'Baseline AUC': results['baseline'].get('auc_roc', 0.5),
        'Improved AUC': results['improved'].get('auc_roc', 0.5),
        'Proposed AUC': results['proposed'].get('auc_roc', 0.5),
    }))


def phase_experiment(config: Dict[str, Any]):
    """Phase 4: Run experiment pipeline."""
    print("\n" + "="*60)
    print("PHASE 4: EXPERIMENT PIPELINE")
    print("="*60)
    
    phase = config.get('experiment_phase', 'single')
    
    if phase == 'single':
        result = run_single_experiment(config)
        print(f"\n  Single experiment result:")
        print(f"    Accuracy: {result.get('accuracy', 0):.3f}")
        print(f"    F1: {result.get('f1', 0):.3f}")
        print(f"    AUC-ROC: {result.get('auc_roc', 0):.3f}")
        
    elif phase == 'grid':
        k_grid = config.get('grid_k_values', [1, 3, 5, 7, 10, 15, 20, 25, 30])
        results = run_grid_experiment(config, k_grid=k_grid)
        print(f"\n  Grid experiment: {len(results)} configurations")
        
        # Save results
        save_experiment_results(results, 'results/metrics/grid_results.csv')
        print(f"  Results saved to results/metrics/grid_results.csv")
        
    elif phase == 'trials':
        num_trials = config.get('num_trials', 10)
        trial_results = run_multiple_trials(config, num_trials=num_trials, vary_seed=True)
        stats = trial_results.get('stats', {})
        
        print(f"\n  Multiple trials ({num_trials}):")
        print(f"    Accuracy mean: {stats.get('accuracy_mean', 0):.3f} ± {stats.get('accuracy_std', 0):.3f}")
        print(f"    F1 mean: {stats.get('f1_mean', 0):.3f} ± {stats.get('f1_std', 0):.3f}")
        print(f"    AUC-ROC mean: {stats.get('auc_roc_mean', 0):.3f} ± {stats.get('auc_roc_std', 0):.3f}")
        
        # Save trial results
        save_experiment_results(trial_results.get('trials', []), 'results/metrics/trial_results.csv')
        
    elif phase == 'comparison':
        comparison = compare_all_methods(config, k_values=config.get('k_values', [5]))
        print(f"\n  Method comparison:")
        for method_name, metrics in comparison.items():
            print(f"    {method_name}:")
            for metric_name, metric_value in metrics.items():
                print(f"      {metric_name}: {metric_value:.3f}")
        
        # Save comparison
        save_experiment_results([comparison], 'results/metrics/comparison_results.csv')
    
    return config


def phase_visualize(config: Dict[str, Any]):
    """Phase 5: Visualize saved results."""
    print("\n" + "="*60)
    print("PHASE 5: VISUALIZATION FROM SAVED RESULTS")
    print("="*60)
    
    visualizer = create_visualizer(config)
    
    # Load experiment results if available
    results_path = os.path.join(config.get('output', 'results'), 'metrics', 'experiment_results.csv')
    
    if os.path.exists(results_path):
        results = load_experiment_results(results_path)
        print(f"Loaded {len(results)} experiment results from {results_path}")
        
        # Plot summary
        visualizer.plot_experiment_results_summary(
            results, 
            metric='accuracy',
            save_name="experiment_summary_accuracy.png"
        )
        
        # Plot per-method comparison
        comparison = {}
        for r in results:
            method = r.get('method', 'unknown')
            if method not in comparison:
                comparison[method] = {}
            for key in ['accuracy', 'f1', 'auc_roc']:
                if key in r:
                    comparison[method][key] = r[key]
        
        if comparison:
            visualizer.plot_method_comparison_bar(comparison,
                                                 save_name="saved_method_comparison.png")
    else:
        print("! No saved results found. Run experiments first.")


def phase_full_eval(config: Dict[str, Any], args):
    """Comprehensive Evaluation Phase."""
    if not EVAL_AVAILABLE:
        print("! Comprehensive evaluation module not available")
        return
    
    print("\n" + "="*70)
    print("COMPREHENSIVE EMPIRICAL EVALUATION")
    print("="*70)
    
    # Resolve datasets
    datasets = args.datasets
    if datasets is None and args.dataset:
        datasets = [args.dataset]
    if datasets is None:
        datasets = ["NSL-KDD"]
    if "both" in datasets:
        datasets = ["NSL-KDD", "CIC-DDoS2019"]
    if "all" in datasets:
        datasets = ["NSL-KDD", "InSDN"]
    
    # Create evaluation config - use base data root without dataset subdirectory
    base_data_root = args.dataset_path or config.get('dataset_path', 'data/raw')
    # Remove dataset subdirectory if present
    if base_data_root.endswith('cicddos2019') or base_data_root.endswith('nsl_kdd'):
        base_data_root = os.path.dirname(base_data_root)
    
    eval_config = FullEvaluationConfig(
        datasets=datasets,
        data_root=base_data_root,
        download_datasets=args.download_data,
        num_trials=args.num_trials,
        run_baselines=not args.no_baselines,
        run_ablations=not args.no_ablations,
        run_statistical_tests=not args.no_stats,
        run_sdn_evaluation=args.sdn,
        output_dir=args.output or 'results/evaluation'
    )
    
    # Run quick test if requested
    if args.quick_test:
        quick_test()
        return
    
    # Run full evaluation
    results = run_full_evaluation(eval_config)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)


phase_q1 = phase_full_eval


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command-line arguments
    if args.method is not None:
        config['method'] = args.method
    if args.dataset is not None:
        config['dataset'] = args.dataset
    if args.dataset_path is not None:
        config['dataset_path'] = args.dataset_path
    if args.k is not None:
        config['k'] = args.k
    
    # Set seed for reproducibility
    set_seed(config.get('random_seed', 42))
    
    print("="*70)
    print("DDoS DETECTION RESEARCH PIPELINE")
    print("="*70)
    print(f"Phase: {args.phase}")
    print(f"Method: {config.get('method', 'proposed')}")
    print(f"Dataset: {config.get('dataset', 'synthetic')}")
    print(f"K value: {config.get('k', 5)}")
    print(f"Seed: {config.get('random_seed', 42)}")
    print("="*70)
    
    # Execute requested phase
    if args.phase == 'data':
        # Just load and report data
        data = phase_data(config)
        
    elif args.phase == 'train':
        # Data + training
        data = phase_data(config)
        results = phase_train(data, config)
        
    elif args.phase == 'evaluate':
        # Data + training + evaluation
        data = phase_data(config)
        train_results = phase_train(data, config)
        phase_evaluate(train_results, config)
        
    elif args.phase == 'experiment':
        phase_experiment(config)
        
    elif args.phase == 'visualize':
        phase_visualize(config)
        
    elif args.phase in ['full_eval', 'q1']:
        phase_full_eval(config, args)
        
    elif args.phase == 'full':
        # Full pipeline: data + train + evaluate + experiment
        print("\n[Running full pipeline...]")
        data = phase_data(config)
        train_results = phase_train(data, config)
        phase_evaluate(train_results, config)
        phase_experiment(config)
        phase_visualize(config)
        
        print("\n" + "="*70)
        print("FULL PIPELINE COMPLETE")
        print("="*70)
    
    else:
        print(f"! Unknown phase: {args.phase}")
        print("Available phases: data, train, evaluate, experiment, visualize, full_eval, full")


if __name__ == "__main__":
    main()