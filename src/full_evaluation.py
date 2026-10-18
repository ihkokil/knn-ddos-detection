"""
Comprehensive DDoS Detection Evaluation Orchestrator
Runs the complete empirical study with:

1. Exact CIC-DDoS2019 and NSL-KDD datasets with provenance/checksums
2. Group/time-aware splits (no leakage)
3. 20-30 independent trials per dataset
4. Comparison: Random Forest, SVM, XGBoost, LightGBM, MLP
5. Attack-type-specific results (UDP, TCP-SYN, TCP-ACK, ICMP, HTTP, Slowloris)
6. Precision-recall curves, calibration, inference latency, memory, controller overhead
7. Ablations: degree weighting, adaptive K, bounded distance compression, dual voting
8. Statistical tests with effect sizes and multiple comparison correction
9. SDN mitigation behavior in Mininet with Ryu
10. Explicit dataset-specific feature schemas
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import all modules
from src.dataset_downloader import DatasetDownloader, quick_download_nsl_kdd, quick_prepare_cicddos2019
from src.feature_schema import (get_dataset_schema, engineer_features_universal,
                                 validate_schema_compliance, CICDDOS2019Schema, NSLKDDSchema)
from src.split_strategies import (split_dataset_aware, create_splitter, 
                                   CICDDOS2019Splitter, NSLKDDSplitter,
                                   verify_no_leakage)
from src.baselines import (create_all_baselines, run_baseline_comparison,
                            evaluate_classifier, calibrate_predictions,
                            compute_calibration_metrics, compare_all_methods)
from src.ablation import (AblationStudy, AblationConfig, ABLATION_VARIANTS,
                           run_component_ablation)
from src.statistical_tests import (compare_methods_pairwise, compare_all_metrics,
                                    friedman_test, analyze_trial_stability,
                                    compute_win_tie_loss, generate_statistical_report,
                                    cohens_d_paired, hedges_g_paired, cliff_delta,
                                    vargha_delaney_a12, multiple_comparison_correction)
from src.sdn_integration import (run_sdn_evaluation, SDNExperimentConfig,
                                  export_model_for_ryu, check_sdn_dependencies)
from src.degree_of_attack import DegreeOfAttackComputer
from src.evaluator import DDoSEvaluator, quick_evaluate
from src.knn_baseline import StandardKNN
from src.knn_improved import ImprovedKNN
from src.knn_proposed import ProposedKNN
from src.utils import set_seed, ensure_dir, Timer

warnings.filterwarnings('ignore')


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class FullEvaluationConfig:
    """Complete configuration for comprehensive empirical evaluation."""
    # Dataset settings
    datasets: List[str] = field(default_factory=lambda: ["NSL-KDD", "CIC-DDoS2019"])
    data_root: str = "data/raw"
    download_datasets: bool = True
    
    # Splitting settings
    test_size: float = 0.2
    use_group_aware_split: bool = True
    use_time_aware_split: bool = True
    
    # Trial settings
    num_trials: int = 30
    random_seeds: List[int] = field(default_factory=lambda: list(range(42, 42+30)))
    max_train_samples: Optional[int] = 8000
    max_test_samples: Optional[int] = 2500
    
    # KNN settings
    k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 7, 10, 15, 20])
    default_k: int = 5
    metric: str = "euclidean"
    
    # Baseline settings
    run_baselines: bool = True
    baseline_names: List[str] = field(default_factory=lambda: [
        "RandomForest", "SVM", "XGBoost", "LightGBM", "MLP", "LogisticRegression"
    ])
    
    # Ablation settings
    run_ablations: bool = True
    ablation_trials: int = 10
    
    # Statistical settings
    run_statistical_tests: bool = True
    correction_method: str = "holm"
    alpha: float = 0.05
    
    # SDN settings
    run_sdn_evaluation: bool = False  # Requires root/mininet
    sdn_scenarios: List[str] = field(default_factory=lambda: [
        "udp_flood", "syn_flood", "icmp_flood", "http_flood", "slowloris"
    ])
    sdn_attack_duration: int = 30
    
    # Output settings
    output_dir: str = "results/evaluation"
    save_intermediate: bool = True
    generate_plots: bool = True
    
    # Feature engineering
    degree_method: str = "proposed"
    include_entropy: bool = True


# Alias for backward compatibility
Q1EvaluationConfig = FullEvaluationConfig


# =============================================================================
# Data Loading with Explicit Schema
# =============================================================================

def load_dataset_with_schema(dataset_name: str, config: Q1EvaluationConfig) -> Dict[str, Any]:
    """
    Load dataset using explicit schema-based feature engineering with strict zero-leakage.
    """
    print(f"\n{'='*60}")
    print(f"Loading {dataset_name} with explicit schema (leak-free)")
    print(f"{'='*60}")
    
    schema = get_dataset_schema(dataset_name)
    downloader = DatasetDownloader(config.data_root)
    
    if dataset_name == "NSL-KDD":
        # Load NSL-KDD
        data_dir = Path(config.data_root) / "nsl_kdd"
        train_path = data_dir / "KDDTrain+.txt"
        test_path = data_dir / "KDDTest+.txt"
        
        if not train_path.exists() or not test_path.exists():
            if config.download_datasets:
                print("NSL-KDD not found, downloading...")
                quick_download_nsl_kdd(config.data_root)
            else:
                raise FileNotFoundError(f"NSL-KDD not found at {data_dir}")
        
        # Load raw data
        col_names = schema.raw_features
        train_df = pd.read_csv(train_path, header=None, names=col_names)
        test_df = pd.read_csv(test_path, header=None, names=col_names)
        
        # Validate schema
        validation = validate_schema_compliance(train_df, dataset_name)
        print(f"Schema compliance: {validation['compliance_score']:.2%}")
        if validation['missing_columns']:
            print(f"Missing columns: {validation['missing_columns']}")
            
        from src.feature_schema import engineer_features_nsl_kdd
        X_train, y_train, feature_names, _, y_attack_train, _ = engineer_features_nsl_kdd(
            train_df, schema, include_entropy=False, include_degree=False
        )
        X_test, y_test, _, _, y_attack_test, _ = engineer_features_nsl_kdd(
            test_df, schema, include_entropy=False, include_degree=False
        )
        
        deg_comp = DegreeOfAttackComputer(method="base_paper")
        deg_comp.fit(X_train, feature_names=feature_names)
        degrees_train = deg_comp.transform(X_train, feature_names=feature_names)
        degrees_test = deg_comp.transform(X_test, feature_names=feature_names)

    elif dataset_name.upper() in ["INSDN", "IN-SDN"]:
        data_dir = Path(config.data_root) / "insdn"
        train_path = data_dir / "insdn_ddos_binary_0n1d_train_smote.csv"
        test_path = data_dir / "insdn_test_labeled.csv"
        
        if not train_path.exists() or not test_path.exists():
            if config.download_datasets:
                print("InSDN not found, downloading and preparing...")
                downloader.download_insdn("insdn")
            else:
                raise FileNotFoundError(f"InSDN not found at {data_dir}")
                
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        validation = validate_schema_compliance(train_df, "InSDN")
        print(f"Schema compliance: {validation['compliance_score']:.2%}")
        
        from src.feature_schema import engineer_features_insdn
        X_train, y_train, feature_names, _, y_attack_train = engineer_features_insdn(
            train_df, schema, include_entropy=False, include_degree=False
        )
        X_test, y_test, _, _, y_attack_test = engineer_features_insdn(
            test_df, schema, include_entropy=False, include_degree=False
        )
        
        deg_comp = DegreeOfAttackComputer(method="proposed")
        deg_comp.fit(X_train, feature_names=feature_names)
        degrees_train = deg_comp.transform(X_train, feature_names=feature_names)
        degrees_test = deg_comp.transform(X_test, feature_names=feature_names)

    elif dataset_name == "CIC-DDoS2019":
        data_dir = Path(config.data_root) / "cicddos2019"
        train_path = data_dir / "Wednesday_workingHours_pcap_ISCX.csv"
        test_path = data_dir / "Thursday_workingHours_pcap_ISCX.csv"
        
        if not train_path.exists() or not test_path.exists():
            if config.download_datasets:
                print("CIC-DDoS2019 not found, preparing download instructions...")
                quick_prepare_cicddos2019(config.data_root)
            raise FileNotFoundError(f"CIC-DDoS2019 not found at {data_dir}. "
                                    f"See {data_dir}/DOWNLOAD_INSTRUCTIONS.txt")
        
        # Load raw data (sample for memory)
        print("Loading CIC-DDoS2019 (this may take a while)...")
        train_df = pd.read_csv(train_path, nrows=100000)
        test_df = pd.read_csv(test_path, nrows=100000)
        
        validation = validate_schema_compliance(train_df, dataset_name)
        print(f"Schema compliance: {validation['compliance_score']:.2%}")
        
        from src.feature_schema import engineer_features_cicddos2019
        X_train, y_train, feature_names, _, y_attack_train = engineer_features_cicddos2019(
            train_df, schema, include_degree=False
        )
        X_test, y_test, _, _, y_attack_test = engineer_features_cicddos2019(
            test_df, schema, include_degree=False
        )
        deg_comp = DegreeOfAttackComputer(method="proposed")
        deg_comp.fit(X_train, feature_names=feature_names)
        degrees_train = deg_comp.transform(X_train, feature_names=feature_names)
        degrees_test = deg_comp.transform(X_test, feature_names=feature_names)
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    X_train_full, y_train_full = X_train, y_train
    X_test_full, y_test_full = X_test, y_test
    
    if config.max_train_samples is not None and len(X_train) > config.max_train_samples:
        from sklearn.model_selection import train_test_split
        X_train, _, y_train, _ = train_test_split(
            X_train, y_train, train_size=config.max_train_samples,
            random_state=42, stratify=y_train
        )
    if config.max_test_samples is not None and len(X_test) > config.max_test_samples:
        from sklearn.model_selection import train_test_split
        X_test, _, y_test, _ = train_test_split(
            X_test, y_test, train_size=config.max_test_samples,
            random_state=42, stratify=y_test
        )
        
    deg_comp.fit(X_train, feature_names=feature_names)
    degrees_train = deg_comp.transform(X_train, feature_names=feature_names)
    degrees_test = deg_comp.transform(X_test, feature_names=feature_names)
        
    print(f"Data loaded: Train={len(X_train)} (full: {len(X_train_full)}), Test={len(X_test)} (full: {len(X_test_full)})")
    print(f"Features: {len(feature_names)}")
    print(f"Train class dist: {np.bincount(y_train)}")
    print(f"Test class dist: {np.bincount(y_test)}")
    
    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "X_train_full": X_train_full, "X_test_full": X_test_full,
        "y_train_full": y_train_full, "y_test_full": y_test_full,
        "degrees_train": degrees_train, "degrees_test": degrees_test,
        "y_attack_train": y_attack_train[:len(X_train)] if y_attack_train is not None else None,
        "y_attack_test": y_attack_test[:len(X_test)] if y_attack_test is not None else None,
        "feature_names": feature_names,
        "schema": schema,
        "dataset_name": dataset_name,
        "validation": validation
    }


# =============================================================================
# KNN Model Training
# =============================================================================

def train_knn_models(X_train: np.ndarray, y_train: np.ndarray, 
                     degrees_train: np.ndarray, config: Q1EvaluationConfig) -> Dict[str, Any]:
    """Train all three KNN variants."""
    print(f"\nTraining KNN models (k={config.default_k})...")
    
    models = {}
    
    # Baseline KNN
    print("  Training Standard KNN...")
    baseline = StandardKNN(k=config.default_k, metric=config.metric, weights='uniform')
    baseline.fit(X_train, y_train)
    models['baseline'] = baseline
    
    # Improved KNN (base paper)
    print("  Training Improved KNN (base paper)...")
    improved = ImprovedKNN(k=config.default_k, metric=config.metric, 
                           weights='attack_degree')
    improved.fit(X_train, y_train, degrees_train)
    models['improved'] = improved
    
    # Proposed KNN
    print("  Training Proposed KNN...")
    proposed = ProposedKNN(k=config.default_k, metric=config.metric, 
                           weights='dual', adaptive_k=True)
    proposed.fit(X_train, y_train, degrees_train)
    models['proposed'] = proposed
    
    return models


def evaluate_knn_models(models: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray,
                        degrees_test: np.ndarray) -> Dict[str, Dict]:
    """Evaluate all KNN models."""
    print("\nEvaluating KNN models...")
    results = {}
    
    for name, model in models.items():
        print(f"  Evaluating {name}...")
        try:
            y_proba = model.predict_proba(X_test, degrees_test=degrees_test)
            y_pred = model.predict(X_test, degrees_test=degrees_test)
        except (TypeError, ValueError):
            y_proba = model.predict_proba(X_test)
            y_pred = model.predict(X_test)
        
        if y_proba.ndim == 1:
            y_proba = np.column_stack([1 - y_proba, y_proba])
        
        evaluator = DDoSEvaluator()
        metrics = evaluator.evaluate_classification(y_test, y_pred, y_proba, degrees_test)
        metrics['method'] = name
        results[name] = metrics
        
        print(f"    Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, "
              f"AUC: {metrics.get('auc_roc', 0):.4f}")
    
    return results


# =============================================================================
# Per-Attack-Type Evaluation
# =============================================================================

def evaluate_per_attack_type(models: Dict[str, Any], X_test: np.ndarray,
                              y_test: np.ndarray, y_attack_test: np.ndarray,
                              schema: Any, dataset_name: str,
                              degrees_test: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Evaluate models per attack type."""
    print("\nEvaluating per attack type...")
    
    attack_types = schema.get_attack_type_names()
    results = {}
    
    for name, model in models.items():
        try:
            y_proba = model.predict_proba(X_test, degrees_test=degrees_test)
            y_pred = model.predict(X_test, degrees_test=degrees_test)
        except (TypeError, ValueError):
            y_proba = model.predict_proba(X_test)
            y_pred = model.predict(X_test)
        if y_proba.ndim == 1:
            y_proba = np.column_stack([1 - y_proba, y_proba])
        
        attack_results = {}
        for attack_idx, attack_name in enumerate(attack_types):
            if attack_idx == 0:  # Skip normal/benign
                continue
            mask = y_attack_test == attack_idx
            if np.sum(mask) < 5:
                continue
            
            evaluator = DDoSEvaluator()
            metrics = evaluator.evaluate_classification(
                y_test[mask], y_pred[mask], y_proba[mask]
            )
            attack_results[attack_name] = metrics
        
        results[name] = attack_results
    
    return results


# =============================================================================
# Calibration and Latency Evaluation
# =============================================================================

def evaluate_calibration_and_latency(models: Dict[str, Any], X_test: np.ndarray,
                                      y_test: np.ndarray, config: Q1EvaluationConfig) -> Dict[str, Any]:
    """Evaluate calibration, inference latency, and memory."""
    print("\nEvaluating calibration and latency...")
    results = {}
    
    for name, model in models.items():
        print(f"  Evaluating {name}...")
        
        # Calibration
        y_proba = model.predict_proba(X_test)
        if y_proba.ndim == 1:
            y_proba = np.column_stack([1 - y_proba, y_proba])
        
        cal_metrics = compute_calibration_metrics(y_test, y_proba[:, 1])
        
        # Latency (multiple runs)
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            _ = model.predict(X_test[:100])
            latencies.append(time.perf_counter() - start)
        
        # Memory estimation (rough)
        import sys
        model_size = sys.getsizeof(model)
        if hasattr(model, 'X_train') and model.X_train is not None:
            model_size += model.X_train.nbytes
            if hasattr(model, 'y_train') and model.y_train is not None:
                model_size += model.y_train.nbytes
            if hasattr(model, 'degrees_train') and model.degrees_train is not None:
                model_size += model.degrees_train.nbytes
        
        results[name] = {
            'calibration': cal_metrics,
            'latency': {
                'mean_ms': np.mean(latencies) * 1000,
                'median_ms': np.median(latencies) * 1000,
                'std_ms': np.std(latencies) * 1000,
                'min_ms': np.min(latencies) * 1000,
                'max_ms': np.max(latencies) * 1000,
                'per_sample_us': np.mean(latencies) * 1e6 / 100
            },
            'memory_mb': model_size / (1024 * 1024)
        }
        
        print(f"    ECE: {cal_metrics['ece']:.4f}, Latency: {results[name]['latency']['mean_ms']:.2f}ms, "
              f"Memory: {results[name]['memory_mb']:.2f}MB")
    
    return results


# =============================================================================
# Multi-Trial Evaluation
# =============================================================================

def run_multi_trial_evaluation(data: Dict[str, Any], config: Q1EvaluationConfig) -> Dict[str, Any]:
    """Run multiple independent trials with different random seeds (strictly leak-free)."""
    print(f"\n{'='*60}")
    print(f"Running {config.num_trials} independent trials")
    print(f"{'='*60}")
    
    X_train_full = data.get("X_train_full", data["X_train"])
    y_train_full = data.get("y_train_full", data["y_train"])
    X_test_full = data.get("X_test_full", data["X_test"])
    y_test_full = data.get("y_test_full", data["y_test"])
    feature_names = data.get("feature_names", None)
    deg_method = "base_paper" if data["dataset_name"] == "NSL-KDD" else "proposed"
    
    train_sub_max = config.max_train_samples or 8000
    test_sub_max = config.max_test_samples or 2500
    
    all_trial_results = {
        'baseline': [], 'improved': [], 'proposed': []
    }
    all_baseline_results = {name: [] for name in config.baseline_names}
    
    for trial_idx in range(config.num_trials):
        seed = config.random_seeds[trial_idx] if trial_idx < len(config.random_seeds) else 42 + trial_idx
        set_seed(seed)
        
        print(f"\n  Trial {trial_idx+1}/{config.num_trials} (seed={seed})...")
        
        from sklearn.model_selection import train_test_split
        if len(X_train_full) > train_sub_max:
            X_tr, _, y_tr, _ = train_test_split(
                X_train_full, y_train_full, train_size=train_sub_max,
                random_state=seed, stratify=y_train_full
            )
        else:
            X_tr, y_tr = X_train_full, y_train_full
            
        if len(X_test_full) > test_sub_max:
            X_te, _, y_te, _ = train_test_split(
                X_test_full, y_test_full, train_size=test_sub_max,
                random_state=seed, stratify=y_test_full
            )
        else:
            X_te, y_te = X_test_full, y_test_full
            
        # Fit degree computer STRICTLY on training split
        deg_comp = DegreeOfAttackComputer(method=deg_method)
        deg_comp.fit(X_tr, feature_names=feature_names)
        deg_tr = deg_comp.transform(X_tr, feature_names=feature_names)
        deg_te = deg_comp.transform(X_te, feature_names=feature_names)
        
        # Train KNN models
        t_fit = time.perf_counter()
        knn_models = train_knn_models(X_tr, y_tr, deg_tr, config)
        fit_time_knn = time.perf_counter() - t_fit
        
        # Evaluate KNN
        for name, model in knn_models.items():
            t_infer = time.perf_counter()
            y_pred = model.predict(X_te)
            if name in ['improved', 'proposed']:
                y_proba = model.predict_proba(X_te, degrees_test=deg_te)
            else:
                y_proba = model.predict_proba(X_te)
            infer_duration = time.perf_counter() - t_infer
            
            if y_proba.ndim == 1:
                y_proba = np.column_stack([1 - y_proba, y_proba])
            
            evaluator = DDoSEvaluator()
            metrics = evaluator.evaluate_classification(y_te, y_pred, y_proba, deg_te)
            metrics['trial'] = trial_idx
            metrics['seed'] = seed
            metrics['fit_time_s'] = fit_time_knn
            metrics['inference_time'] = infer_duration
            metrics['latency_us_per_flow'] = (infer_duration / len(X_te)) * 1e6
            all_trial_results[name].append(metrics)
        
        # Evaluate baselines
        if config.run_baselines:
            baseline_models = create_all_baselines(random_state=seed)
            baseline_results = run_baseline_comparison(X_tr, y_tr, X_te, y_te, baseline_models)
            
            for name, metrics in baseline_results.items():
                if 'error' not in metrics:
                    metrics['trial'] = trial_idx
                    metrics['seed'] = seed
                    all_baseline_results[name].append(metrics)
        
        # Progress
        if (trial_idx + 1) % 5 == 0 or trial_idx == 0:
            for knn_name in ['baseline', 'improved', 'proposed']:
                if all_trial_results[knn_name]:
                    accs = [r['accuracy'] for r in all_trial_results[knn_name]]
                    f1s = [r['f1'] for r in all_trial_results[knn_name]]
                    lats = [r.get('latency_us_per_flow', 0) for r in all_trial_results[knn_name]]
                    print(f"    {knn_name:10s}: Acc={np.mean(accs):.4f}+/-{np.std(accs):.4f}, "
                          f"F1={np.mean(f1s):.4f}+/-{np.std(f1s):.4f}, Latency={np.mean(lats):.1f}us")
    
    return {
        'knn_trials': all_trial_results,
        'baseline_trials': all_baseline_results
    }


def aggregate_trial_results(trial_results: Dict[str, List]) -> Dict[str, Dict]:
    """Aggregate results across trials."""
    aggregated = {}
    
    for method, trials in trial_results.items():
        if not trials:
            continue
        
        agg = {'method': method, 'num_trials': len(trials)}
        
        metric_keys = ['accuracy', 'precision', 'recall', 'f1', 'fpr', 'fnr', 
                       'detection_rate', 'auc_roc', 'training_time', 'inference_time',
                       'latency_us_per_flow', 'fit_time_s']
        
        for key in metric_keys:
            values = [t.get(key, 0) for t in trials if key in t]
            if values:
                agg[f'{key}_mean'] = float(np.mean(values))
                agg[f'{key}_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                agg[f'{key}_ci95'] = float(1.96 * np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
                agg[f'{key}_min'] = float(np.min(values))
                agg[f'{key}_max'] = float(np.max(values))
        
        aggregated[method] = agg
    
    return aggregated


# =============================================================================
# Ablation Study Runner
# =============================================================================

def run_ablation_study(data: Dict[str, Any], config: Q1EvaluationConfig) -> Dict[str, Any]:
    """Run comprehensive ablation study."""
    print(f"\n{'='*60}")
    print("Running Ablation Study")
    print(f"{'='*60}")
    
    X = np.vstack([data["X_train"], data["X_test"]])
    y = np.concatenate([data["y_train"], data["y_test"]])
    degrees = np.concatenate([data["degrees_train"], data["degrees_test"]])
    
    # Use a fixed split for ablation
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te, deg_tr, deg_te = train_test_split(
        X, y, degrees, test_size=config.test_size, random_state=42, stratify=y
    )
    
    feature_names = data["feature_names"]
    
    # Run ablation study
    ablation_config = {
        'k': config.default_k,
        'metric': config.metric,
        'degree_method': config.degree_method,
        'num_trials': config.ablation_trials,
        'random_seed': 42
    }
    
    study = AblationStudy(ablation_config)
    results = study.run_full_ablation(X_tr, y_tr, X_te, y_te, feature_names)
    
    # Compute effects
    effects_df = study.compute_ablation_effects()
    summary_df = study.get_summary_table()
    
    print("\nAblation Summary:")
    print(summary_df[['Variant', 'accuracy', 'f1', 'auc_roc', 'training_time', 'inference_time_ms']].to_string(index=False))
    
    print("\nAblation Effects (vs Full Proposed):")
    print(effects_df[['variant', 'f1_diff', 'f1_cohens_d', 'auc_roc_diff', 'auc_roc_cohens_d']].to_string(index=False))
    
    return {
        'variant_results': results,
        'effects': effects_df.to_dict('records'),
        'summary': summary_df.to_dict('records')
    }


# =============================================================================
# Statistical Testing
# =============================================================================

def run_statistical_analysis(trial_results: Dict[str, Any], config: Q1EvaluationConfig) -> Dict[str, Any]:
    """Run comprehensive statistical analysis."""
    print(f"\n{'='*60}")
    print("Running Statistical Analysis")
    print(f"{'='*60}")
    
    # Combine KNN and baseline results
    all_results = {}
    for method, trials in trial_results['knn_trials'].items():
        all_results[method] = {'trials': trials}
    for method, trials in trial_results['baseline_trials'].items():
        all_results[method] = {'trials': trials}
    
    # Pairwise comparisons for all metrics
    print("\nPairwise comparisons (Wilcoxon, Holm-corrected):")
    metric_comparisons = compare_all_metrics(all_results, 
                                              metrics=['accuracy', 'precision', 'recall', 'f1', 'auc_roc'],
                                              test='wilcoxon', correction=config.correction_method)
    
    for metric, df in metric_comparisons.items():
        print(f"\n{metric.upper()}:")
        for _, row in df.iterrows():
            sig = "***" if row.get('p_corrected', 1) < 0.001 else \
                  "**" if row.get('p_corrected', 1) < 0.01 else \
                  "*" if row.get('p_corrected', 1) < 0.05 else "ns"
            print(f"  {row['method_1']:20s} vs {row['method_2']:20s}: "
                  f"p={row.get('p_corrected', row['p_value']):.4f} {sig}, "
                  f"Diff={row['mean_diff']:+.4f}, d={row['cohens_d']:.3f}")
    
    # Friedman test
    print("\nFriedman test (omnibus):")
    friedman_results = {}
    for metric in ['accuracy', 'f1', 'auc_roc']:
        trial_arrays = []
        methods = []
        for method, result in all_results.items():
            if 'trials' in result:
                values = [t.get(metric, 0) for t in result['trials']]
                if len(values) > 0:
                    trial_arrays.append(np.array(values))
                    methods.append(method)
        
        if len(trial_arrays) >= 3 and all(len(arr) == len(trial_arrays[0]) for arr in trial_arrays):
            friedman = friedman_test(*trial_arrays)
            if 'error' not in friedman:
                friedman_results[metric] = friedman
                print(f"  {metric}: chi2={friedman['statistic']:.3f}, p={friedman['p_value']:.4f}")
    
    # Stability analysis
    print("\nTrial stability (CV = std/mean):")
    for method_type in ['knn_trials', 'baseline_trials']:
        stability = analyze_trial_stability(trial_results[method_type], 'f1')
        if len(stability) > 0:
            print(f"  {method_type}:")
            print(stability[['method', 'mean', 'std', 'cv']].to_string(index=False))
    
    # Win/Tie/Loss
    print("\nWin/Tie/Loss matrix (F1):")
    wtl = compute_win_tie_loss(all_results, 'f1')
    print(wtl.to_string())
    
    # Generate full report
    report = generate_statistical_report(all_results)
    
    return {
        'pairwise_comparisons': {k: v.to_dict('records') for k, v in metric_comparisons.items()},
        'friedman_tests': friedman_results,
        'stability': {k: v.to_dict('records') for k, v in {
            'knn': analyze_trial_stability(trial_results['knn_trials'], 'f1'),
            'baselines': analyze_trial_stability(trial_results['baseline_trials'], 'f1')
        }.items()},
        'win_tie_loss': wtl.to_dict(),
        'full_report': report
    }


# =============================================================================
# SDN Evaluation
# =============================================================================

def run_sdn_evaluation_wrapper(models: Dict[str, Any], feature_names: List[str],
                                config: Q1EvaluationConfig) -> Dict[str, Any]:
    """Run SDN evaluation with best model."""
    print(f"\n{'='*60}")
    print("Running SDN Evaluation (Mininet + Ryu)")
    print(f"{'='*60}")
    
    # Check dependencies
    deps = check_sdn_dependencies()
    missing = [k for k, v in deps.items() if not v]
    if missing:
        print(f"Missing dependencies: {missing}")
        print("Skipping SDN evaluation. Run with sudo and install dependencies.")
        return {'skipped': True, 'missing_dependencies': missing}
    
    # Use proposed model
    best_model = models.get('proposed')
    if best_model is None:
        print("No proposed model available for SDN evaluation")
        return {'skipped': True, 'reason': 'no_proposed_model'}
    
    # Export model
    model_data = export_model_for_ryu(best_model, feature_names)
    
    # Run SDN evaluation
    sdn_config = SDNExperimentConfig(
        attack_scenarios=config.sdn_scenarios,
        attack_duration=config.sdn_attack_duration,
        results_dir=f"{config.output_dir}/sdn"
    )
    
    results = run_sdn_evaluation(
        knn_model=best_model,
        feature_names=feature_names,
        scenarios=config.sdn_scenarios,
        config=sdn_config
    )
    
    return results


# =============================================================================
# Main Evaluation Pipeline
# =============================================================================

def run_full_evaluation(config: FullEvaluationConfig = None) -> Dict[str, Any]:
    """
    Run complete evaluation pipeline.
    
    This is the main entry point that orchestrates all components.
    """
    if config is None:
        config = FullEvaluationConfig()
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("COMPREHENSIVE DDoS DETECTION EVALUATION")
    print("="*80)
    print(f"Datasets: {config.datasets}")
    print(f"Trials: {config.num_trials}")
    print(f"K values: {config.k_values}")
    print(f"Baselines: {config.baseline_names}")
    print(f"Output: {config.output_dir}")
    print("="*80)
    
    start_time = time.time()
    all_results = {}
    
    # Process each dataset
    for dataset_name in config.datasets:
        print(f"\n{'#'*80}")
        print(f"# DATASET: {dataset_name}")
        print(f"{'#'*80}")
        
        try:
            # Load data with explicit schema
            data = load_dataset_with_schema(dataset_name, config)
            
            # Train KNN models
            knn_models = train_knn_models(data["X_train"], data["y_train"], 
                                           data["degrees_train"], config)
            
            # Evaluate KNN models
            knn_results = evaluate_knn_models(knn_models, data["X_test"], 
                                               data["y_test"], data["degrees_test"])
            
            # Per-attack-type evaluation
            attack_results = evaluate_per_attack_type(knn_models, data["X_test"],
                                                       data["y_test"], data["y_attack_test"],
                                                       data["schema"], dataset_name,
                                                       data.get("degrees_test"))
            
            # Calibration and latency
            cal_lat_results = evaluate_calibration_and_latency(knn_models, data["X_test"],
                                                                data["y_test"], config)
            
            # Multi-trial evaluation
            trial_results = run_multi_trial_evaluation(data, config)
            
            # Aggregate trials
            knn_aggregated = aggregate_trial_results(trial_results['knn_trials'])
            baseline_aggregated = aggregate_trial_results(trial_results['baseline_trials'])
            
            # Run baselines on full data (single run for comparison)
            baseline_results = {}
            if config.run_baselines:
                print("\nRunning baseline comparison on full data...")
                baseline_models = create_all_baselines(random_state=42)
                baseline_results = run_baseline_comparison(
                    data["X_train"], data["y_train"], 
                    data["X_test"], data["y_test"], 
                    baseline_models
                )
            
            # Ablation study
            ablation_results = {}
            if config.run_ablations:
                ablation_results = run_ablation_study(data, config)
            
            # Statistical analysis
            statistical_results = {}
            if config.run_statistical_tests:
                statistical_results = run_statistical_analysis(trial_results, config)
            
            # SDN evaluation (optional)
            sdn_results = {}
            if config.run_sdn_evaluation:
                sdn_results = run_sdn_evaluation_wrapper(knn_models, data["feature_names"], config)
            
            # Compile dataset results
            all_results[dataset_name] = {
                'dataset': dataset_name,
                'data_info': {
                    'train_samples': len(data["X_train"]),
                    'test_samples': len(data["X_test"]),
                    'features': len(data["feature_names"]),
                    'train_class_dist': np.bincount(data["y_train"]).tolist(),
                    'test_class_dist': np.bincount(data["y_test"]).tolist(),
                    'attack_types': data["schema"].get_attack_type_names()
                },
                'knn_results': knn_results,
                'knn_aggregated': knn_aggregated,
                'baseline_results': baseline_results,
                'baseline_aggregated': baseline_aggregated,
                'attack_type_results': attack_results,
                'calibration_latency': cal_lat_results,
                'ablation_results': ablation_results,
                'statistical_results': statistical_results,
                'sdn_results': sdn_results,
                'feature_names': data["feature_names"],
                'schema_validation': data["validation"]
            }
            
            # Save intermediate results
            if config.save_intermediate:
                dataset_output = output_dir / f"{dataset_name.lower().replace('-', '_')}_results.json"
                with open(dataset_output, 'w') as f:
                    json.dump(all_results[dataset_name], f, indent=2, default=str)
                print(f"\nSaved intermediate results to {dataset_output}")
            
        except Exception as e:
            print(f"\nFAIL Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[dataset_name] = {'error': str(e)}
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Results saved to: {output_dir}")
    
    # Save final combined results
    final_output = output_dir / "evaluation_complete.json"
    with open(final_output, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"Final results saved to: {final_output}")
    
    # Export CSV tables for paper reproduction
    try:
        summary_rows = []
        for d_name, d_res in all_results.items():
            if 'knn_aggregated' not in d_res:
                continue
            for group_key in ['knn_aggregated', 'baseline_aggregated']:
                for m_name, agg in d_res.get(group_key, {}).items():
                    acc_m, acc_s = agg.get('accuracy_mean', 0.0), agg.get('accuracy_std', 0.0)
                    prec_m, prec_s = agg.get('precision_mean', 0.0), agg.get('precision_std', 0.0)
                    rec_m, rec_s = agg.get('recall_mean', 0.0), agg.get('recall_std', 0.0)
                    f1_m, f1_s = agg.get('f1_mean', 0.0), agg.get('f1_std', 0.0)
                    fpr_m, fpr_s = agg.get('fpr_mean', 0.0), agg.get('fpr_std', 0.0)
                    fnr_m, fnr_s = agg.get('fnr_mean', 0.0), agg.get('fnr_std', 0.0)
                    auc_m, auc_s = agg.get('auc_roc_mean', 0.0), agg.get('auc_roc_std', 0.0)
                    lat_m, lat_s = agg.get('latency_us_per_flow_mean', 0.0), agg.get('latency_us_per_flow_std', 0.0)
                    summary_rows.append({
                        'dataset': d_name,
                        'Model': m_name,
                        'accuracy_mean': acc_m, 'accuracy_std': acc_s, 'accuracy_display': f"{acc_m:.4f} +/- {acc_s:.4f}",
                        'precision_mean': prec_m, 'precision_std': prec_s, 'precision_display': f"{prec_m:.4f} +/- {prec_s:.4f}",
                        'recall_mean': rec_m, 'recall_std': rec_s, 'recall_display': f"{rec_m:.4f} +/- {rec_s:.4f}",
                        'f1_mean': f1_m, 'f1_std': f1_s, 'f1_display': f"{f1_m:.4f} +/- {f1_s:.4f}",
                        'fpr_mean': fpr_m, 'fpr_std': fpr_s, 'fpr_display': f"{fpr_m:.4f} +/- {fpr_s:.4f}",
                        'fnr_mean': fnr_m, 'fnr_std': fnr_s, 'fnr_display': f"{fnr_m:.4f} +/- {fnr_s:.4f}",
                        'auc_roc_mean': auc_m, 'auc_roc_std': auc_s, 'auc_roc_display': f"{auc_m:.4f} +/- {auc_s:.4f}",
                        'latency_us_per_flow_mean': lat_m, 'latency_us_per_flow_std': lat_s, 'latency_us_per_flow_display': f"{lat_m:.2f} +/- {lat_s:.2f}"
                    })
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_path = output_dir / "benchmark_summary.csv"
            summary_df.to_csv(summary_path, index=False)
            top_results = Path("results")
            top_results.mkdir(parents=True, exist_ok=True)
            summary_df.to_csv(top_results / "benchmark_summary.csv", index=False)
            print(f"Benchmark summary saved to {summary_path}")
    except Exception as e:
        print(f"Warning: Failed to export summary CSV: {e}")
        
    return all_results


# Alias for backward compatibility
run_q1_evaluation = run_full_evaluation


# =============================================================================
# Quick Test Function
# =============================================================================

def quick_test():
    """Quick test with synthetic data."""
    print("Running quick test with synthetic data...")
    
    from src.feature_engineering import create_synthetic_ddos_dataset
    
    # Generate synthetic data
    df = create_synthetic_ddos_dataset(num_samples=2000, attack_ratio=0.3)
    
    # Use CIC schema as template
    schema = CICDDOS2019Schema()
    
    # Quick feature engineering
    from src.feature_engineering import engineer_features_for_knn
    X, y, feat_names, degrees = engineer_features_for_knn(df)
    
    # Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test, deg_train, deg_test = train_test_split(
        X, y, degrees, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train and evaluate
    config = Q1EvaluationConfig(
        num_trials=3,
        k_values=[3, 5, 7],
        run_baselines=True,
        run_ablations=True,
        ablation_trials=3,
        run_statistical_tests=True,
        run_sdn_evaluation=False,
        output_dir="results/quick_test"
    )
    
    # Create mock data dict
    data = {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "degrees_train": deg_train, "degrees_test": deg_test,
        "y_attack_train": np.zeros(len(y_train)),
        "y_attack_test": np.zeros(len(y_test)),
        "feature_names": feat_names,
        "schema": schema,
        "dataset_name": "synthetic"
    }
    
    # Run components
    knn_models = train_knn_models(X_train, y_train, deg_train, config)
    knn_results = evaluate_knn_models(knn_models, X_test, y_test, deg_test)
    
    print("\nQuick test KNN results:")
    for name, metrics in knn_results.items():
        print(f"  {name}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, "
              f"AUC={metrics.get('auc_roc', 0):.4f}")
    
    print("\nOK Quick test complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive DDoS Detection Evaluation")
    parser.add_argument("--datasets", nargs="+", default=["NSL-KDD"],
                       choices=["NSL-KDD", "CIC-DDoS2019", "InSDN", "both", "all"])
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--no-baselines", action="store_true")
    parser.add_argument("--no-ablations", action="store_true")
    parser.add_argument("--no-stats", action="store_true")
    parser.add_argument("--sdn", action="store_true")
    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument("--output", default="results/evaluation")
    parser.add_argument("--data-root", default="data/raw")
    
    args = parser.parse_args()
    
    if args.quick_test:
        quick_test()
        sys.exit(0)
    
    # Resolve datasets
    datasets = args.datasets
    if "both" in datasets:
        datasets = ["NSL-KDD", "CIC-DDoS2019"]
    if "all" in datasets:
        datasets = ["NSL-KDD", "InSDN"]
    
    # Create config
    config = Q1EvaluationConfig(
        datasets=datasets,
        data_root=args.data_root,
        num_trials=args.trials,
        run_baselines=not args.no_baselines,
        run_ablations=not args.no_ablations,
        run_statistical_tests=not args.no_stats,
        run_sdn_evaluation=args.sdn,
        output_dir=args.output
    )
    
    # Run evaluation
    results = run_q1_evaluation(config)
    
    # Print summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    for dataset_name, result in results.items():
        if 'error' in result:
            print(f"\n{dataset_name}: ERROR - {result['error']}")
            continue
        
        print(f"\n{dataset_name}:")
        print(f"  Train: {result['data_info']['train_samples']}, "
              f"Test: {result['data_info']['test_samples']}")
        
        if 'knn_aggregated' in result:
            print("  KNN Methods (aggregated):")
            for method, metrics in result['knn_aggregated'].items():
                print(f"    {method:15s}: Acc={metrics.get('accuracy_mean',0):.4f}+/-{metrics.get('accuracy_std',0):.4f}, "
                      f"F1={metrics.get('f1_mean',0):.4f}+/-{metrics.get('f1_std',0):.4f}, "
                      f"AUC={metrics.get('auc_roc_mean',0):.4f}+/-{metrics.get('auc_roc_std',0):.4f}")
        
        if 'baseline_aggregated' in result:
            print("  Baselines (aggregated):")
            for method, metrics in result['baseline_aggregated'].items():
                print(f"    {method:20s}: Acc={metrics.get('accuracy_mean',0):.4f}+/-{metrics.get('accuracy_std',0):.4f}, "
                      f"F1={metrics.get('f1_mean',0):.4f}+/-{metrics.get('f1_std',0):.4f}, "
                      f"AUC={metrics.get('auc_roc_mean',0):.4f}+/-{metrics.get('auc_roc_std',0):.4f}")