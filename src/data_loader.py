"""
Data Loader Module
Loads datasets (NSL-KDD, CIC-DDoS2019, synthetic traffic) and performs
train/test splitting with class balance reporting and stratified subsampling.

Based on: Base paper "DDoS Attack Detection Method Based on Improved KNN With 
          the Degree of DDoS Attack in Software-Defined Networks"
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from pathlib import Path
import os
from typing import Optional, Tuple, List, Dict, Any

from src.feature_schema import NSL_KDD_RAW_FEATURES, NSL_KDD_CATEGORICAL_ENCODINGS

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_dataset_path(dataset_path: str | os.PathLike | None,
                           default_name: str) -> Path:
    """Resolve dataset paths relative to the project root."""
    if dataset_path is None:
        return PROJECT_ROOT / "data" / "raw" / default_name
    path = Path(dataset_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_nsl_kdd(dataset_path: str = None,
                 use_20percent: bool = False,
                 subsample_size: Optional[int] = None,
                 encode_categorical: bool = True,
                 random_seed: int = 42) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]]:
    """
    Load the NSL-KDD dataset with correct 43-column alignment and verified labels.
    
    Parameters
    ----------
    dataset_path : str, optional
        Path to the NSL-KDD dataset directory.
    use_20percent : bool, default=False
        Whether to load KDDTrain+_20Percent.txt if available.
    subsample_size : int, optional
        Stratified subsample size for rapid multi-trial benchmarks.
    encode_categorical : bool, default=True
        Whether to encode categorical features (protocol, service, flag) as floats.
    random_seed : int, default=42
        Random state for subsampling.
        
    Returns
    -------
    tuple
        (X_train, y_train, X_test, y_test, feature_names)
    """
    if dataset_path is None:
        dataset_path = _resolve_dataset_path(None, "nsl_kdd")
    else:
        dataset_path = _resolve_dataset_path(dataset_path, "nsl_kdd")
    
    train_filename = "KDDTrain+_20Percent.txt" if use_20percent else "KDDTrain+.txt"
    train_path = os.path.join(dataset_path, train_filename)
    if not os.path.exists(train_path):
        train_path = os.path.join(dataset_path, "KDDTrain+.txt")
    test_path = os.path.join(dataset_path, "KDDTest+.txt")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"NSL-KDD dataset not found at {dataset_path}")
        return None
    
    col_names = NSL_KDD_RAW_FEATURES
    
    try:
        train_df = pd.read_csv(train_path, header=None, names=col_names)
        test_df = pd.read_csv(test_path, header=None, names=col_names)
    except Exception as e:
        print(f"Error loading NSL-KDD: {e}")
        return None
    
    # Accurate label mapping: strip trailing dot, normal = 0, any attack = 1
    y_train = (train_df["label"].astype(str).str.strip().str.lower().str.rstrip(".") != "normal").astype(int).values
    y_test = (test_df["label"].astype(str).str.strip().str.lower().str.rstrip(".") != "normal").astype(int).values
    
    # Feature extraction (exclude label and difficulty)
    feature_cols = [c for c in col_names if c not in ["label", "difficulty"]]
    
    if encode_categorical:
        for col, mapping in NSL_KDD_CATEGORICAL_ENCODINGS.items():
            if col in train_df.columns:
                train_df[col] = train_df[col].map(mapping).fillna(0)
                test_df[col] = test_df[col].map(mapping).fillna(0)
                
    X_train = train_df[feature_cols].values.astype(float)
    X_test = test_df[feature_cols].values.astype(float)
    
    # Stratified subsampling if requested
    if subsample_size is not None and subsample_size < len(X_train):
        test_sub_size = min(len(X_test), max(1000, int(subsample_size * len(X_test) / len(X_train))))
        X_train, _, y_train, _ = train_test_split(
            X_train, y_train, train_size=subsample_size, stratify=y_train, random_state=random_seed
        )
        if test_sub_size < len(X_test):
            X_test, _, y_test, _ = train_test_split(
                X_test, y_test, train_size=test_sub_size, stratify=y_test, random_state=random_seed
            )
            
    print(f"NSL-KDD loaded: {len(X_train)} training samples, {len(X_test)} test samples")
    print(f"  Training class distribution: {Counter(y_train)}")
    print(f"  Test class distribution: {Counter(y_test)}")
    
    return X_train, y_train, X_test, y_test, feature_cols


def load_insdn(dataset_path: str = None,
               subsample_size: Optional[int] = None,
               random_seed: int = 42) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]]:
    """
    Load the InSDN dataset (dedicated Software-Defined Network intrusion dataset).
    
    Parameters
    ----------
    dataset_path : str, optional
        Path to the InSDN dataset directory.
    subsample_size : int, optional
        Stratified subsample size for rapid multi-trial benchmarks.
    random_seed : int, default=42
        Random state for subsampling.
        
    Returns
    -------
    tuple
        (X_train, y_train, X_test, y_test, feature_names)
    """
    if dataset_path is None:
        dataset_path = _resolve_dataset_path(None, "insdn")
    else:
        dataset_path = _resolve_dataset_path(dataset_path, "insdn")
    clean_tr_path = os.path.join(dataset_path, "insdn_clean_train.csv")
    clean_te_path = os.path.join(dataset_path, "insdn_clean_test.csv")
    if os.path.exists(clean_tr_path) and os.path.exists(clean_te_path):
        train_path = clean_tr_path
        test_path = clean_te_path
    else:
        train_path = os.path.join(dataset_path, "insdn_ddos_binary_0n1d_train_smote.csv")
        test_path = os.path.join(dataset_path, "insdn_test_labeled.csv")
    raw_test_path = os.path.join(dataset_path, "insdn_ddos_binary_0n1d_test.csv")
    raw_label_path = os.path.join(dataset_path, "insdn_ddos_label.csv")
    
    # Auto-reconstruct test labels if insdn_test_labeled.csv does not exist yet
    if not os.path.exists(test_path) and os.path.exists(raw_test_path) and os.path.exists(raw_label_path):
        try:
            print("InSDN: Reconstructing ground-truth test labels...")
            test_df = pd.read_csv(raw_test_path)
            raw_df = pd.read_csv(raw_label_path)
            keys = ['Protocol', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 
                    'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Flow Byts/s', 'Flow Pkts/s']
            raw_map = raw_df[keys + ['Label']].drop_duplicates(subset=keys)
            merged = test_df[keys].merge(raw_map, on=keys, how='left')
            test_df['Label'] = (merged['Label'].astype(str).str.strip().str.lower() != 'normal').astype(int)
            if 'Target' in test_df.columns:
                test_df = test_df.drop(columns=['Target'])
            test_df.to_csv(test_path, index=False)
            print(f"InSDN: Saved verified test labels to {test_path}")
        except Exception as e:
            print(f"Error reconstructing InSDN test labels: {e}")
            
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        if os.path.exists(raw_label_path):
            print(f"InSDN: Loading from {raw_label_path} with 80/20 train/test split...")
            df = pd.read_csv(raw_label_path)
            y = (df['Label'].astype(str).str.strip().str.lower() != 'normal').astype(int).values
            feature_cols = [c for c in df.columns if c not in ['Label', 'Target']]
            X = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).values
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=random_seed
            )
            return X_train, y_train, X_test, y_test, feature_cols
            
        print(f"InSDN dataset not found at {dataset_path}")
        return None
        
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
    except Exception as e:
        print(f"Error loading InSDN CSVs: {e}")
        return None
        
    feature_cols = [c for c in train_df.columns if c not in ['Label', 'Target'] and c in test_df.columns]
    
    y_train = train_df['Label'].astype(int).values
    y_test = test_df['Label'].astype(int).values
    
    numeric_tr = train_df[feature_cols].apply(pd.to_numeric, errors='coerce')
    numeric_te = test_df[feature_cols].apply(pd.to_numeric, errors='coerce')
    
    X_train = np.nan_to_num(numeric_tr.values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(numeric_te.values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    
    # Stratified subsampling if requested
    if subsample_size is not None and subsample_size < len(X_train):
        test_sub_size = min(len(X_test), max(1000, int(subsample_size * len(X_test) / len(X_train))))
        X_train, _, y_train, _ = train_test_split(
            X_train, y_train, train_size=subsample_size, stratify=y_train, random_state=random_seed
        )
        if test_sub_size < len(X_test):
            X_test, _, y_test, _ = train_test_split(
                X_test, y_test, train_size=test_sub_size, stratify=y_test, random_state=random_seed
            )
            
    print(f"InSDN loaded: {len(X_train)} training samples, {len(X_test)} test samples")
    print(f"  Training class distribution: {Counter(y_train)}")
    print(f"  Test class distribution: {Counter(y_test)}")
    
    return X_train, y_train, X_test, y_test, feature_cols


def load_cicddos2019(dataset_path: str = None) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]]:
    """
    Load the CIC-DDoS2019 dataset.
    """
    if dataset_path is None:
        dataset_path = _resolve_dataset_path(None, "cicddos2019")
    else:
        dataset_path = _resolve_dataset_path(dataset_path, "cicddos2019")
    
    # Check for available CSV files
    candidate_files = [
        os.path.join(dataset_path, "01-12", "DrDoS_DNS.csv"),
        os.path.join(dataset_path, "01-12", "Syn.csv"),
        os.path.join(dataset_path, "DrDoS_DNS.csv"),
        os.path.join(dataset_path, "Syn.csv"),
        os.path.join(dataset_path, "Wednesday_workingHours_pcap_ISCX.csv"),
        os.path.join(dataset_path, "Thursday_workingHours_pcap_ISCX.csv")
    ]
    existing = [f for f in candidate_files if os.path.exists(f)]
    
    if not existing:
        print(f"CIC-DDoS2019 dataset not found at {dataset_path}")
        return None
    
    train_path = existing[0]
    test_path = existing[1] if len(existing) > 1 else existing[0]
    
    try:
        train_df = pd.read_csv(train_path, nrows=50000)
        test_df = pd.read_csv(test_path, nrows=20000)
    except Exception as e:
        print(f"Error loading CIC-DDoS2019: {e}")
        return None
    
    lbl_col = [c for c in train_df.columns if c.strip().lower() == "label"]
    if not lbl_col:
        print("Label column not found in CIC-DDoS2019")
        return None
    lbl = lbl_col[0]
    
    y_train = (train_df[lbl].astype(str).str.strip().str.lower() != "benign").astype(int).values
    y_test = (test_df[lbl].astype(str).str.strip().str.lower() != "benign").astype(int).values
    
    drop_cols = {lbl, "Flow ID", "Source IP", "Src IP", "Destination IP", "Dst IP", "Timestamp", "SimillarHTTP"}
    feature_cols = [c for c in train_df.columns if c not in drop_cols and c in test_df.columns]
    
    numeric_tr = train_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    numeric_te = test_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    valid_cols = [c for c in feature_cols if numeric_tr[c].notna().any()]
    
    X_train = np.nan_to_num(numeric_tr[valid_cols].values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(numeric_te[valid_cols].values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    
    return X_train, y_train, X_test, y_test, valid_cols


def load_synthetic_traffic(generator_config: dict = None) -> tuple:
    """
    Generate synthetic SDN traffic data for DDoS detection.
    """
    if generator_config is None:
        generator_config = {
            'num_normal': 5000,
            'num_attack': 2000,
            'attack_intensity': 10.0,
            'num_features': 10,
            'attack_types': ['udp_flood', 'syn_flood', 'icmp_flood'],
            'random_seed': 42
        }
    
    seed = generator_config.get('random_seed', 42)
    np.random.seed(seed)
    
    num_normal = generator_config.get('num_normal', 5000)
    num_attack = generator_config.get('num_attack', 2000)
    num_features = generator_config.get('num_features', 10)
    attack_intensity = generator_config.get('attack_intensity', 10.0)
    
    X_normal = np.abs(np.random.randn(num_normal, num_features))
    X_normal[:, 1] = X_normal[:, 1] * 10 + 1
    X_normal[:, 2] = X_normal[:, 2] * 100 + 10
    X_normal[:, 0] = X_normal[:, 0] * 10 + 1
    
    X_attack = np.abs(np.random.randn(num_attack, num_features))
    X_attack[:, 1] = X_attack[:, 1] * attack_intensity + 1
    X_attack[:, 2] = X_attack[:, 2] * attack_intensity + 10
    X_attack[:, 0] = X_attack[:, 0] * 5 + 1
    
    y_normal = np.zeros(num_normal, dtype=int)
    y_attack = np.ones(num_attack, dtype=int)
    
    X = np.vstack([X_normal, X_attack])
    y = np.concatenate([y_normal, y_attack])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    feature_names = [f"feature_{i}" for i in range(num_features)]
    return X_train, y_train, X_test, y_test, feature_names


def load_dataset(config: dict = None) -> dict:
    """
    Load dataset based on configuration dictionary.
    """
    config = config or {}
    dataset_name = config.get('dataset', 'NSL-KDD')
    seed = config.get('random_seed', 42)
    subsample_size = config.get('subsample_size', None)
    use_20percent = config.get('use_20percent', False)
    dataset_path = config.get('dataset_path', None)
    
    result = {
        'dataset_name': dataset_name,
        'X_train': None,
        'X_test': None,
        'y_train': None,
        'y_test': None,
        'feature_names': None,
        'class_distributions': {},
        'imbalance_ratio': None
    }
    
    if dataset_name == 'NSL-KDD':
        data = load_nsl_kdd(
            dataset_path=dataset_path,
            use_20percent=use_20percent,
            subsample_size=subsample_size,
            encode_categorical=True,
            random_seed=seed
        )
        if data is not None:
            result['X_train'], result['y_train'], result['X_test'], result['y_test'], result['feature_names'] = data
            
    elif dataset_name.upper() in ['INSDN', 'IN-SDN']:
        data = load_insdn(
            dataset_path=dataset_path,
            subsample_size=subsample_size,
            random_seed=seed
        )
        if data is not None:
            result['X_train'], result['y_train'], result['X_test'], result['y_test'], result['feature_names'] = data
            
    elif dataset_name == 'CIC-DDoS2019':
        data = load_cicddos2019(dataset_path)
        if data is not None:
            result['X_train'], result['y_train'], result['X_test'], result['y_test'], result['feature_names'] = data
            
    elif dataset_name == 'synthetic':
        data = load_synthetic_traffic({
            'num_normal': 2000,
            'num_attack': 800,
            'attack_intensity': 5.0,
            'num_features': 10,
            'random_seed': seed
        })
        if data is not None:
            result['X_train'], result['y_train'], result['X_test'], result['y_test'], result['feature_names'] = data
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    if result['y_train'] is not None and result['y_test'] is not None:
        train_counts = Counter(result['y_train'])
        test_counts = Counter(result['y_test'])
        result['class_distributions'] = {'train': dict(train_counts), 'test': dict(test_counts)}
        majority = max(train_counts.values()) if train_counts else 1
        minority = min(train_counts.values()) if train_counts else 1
        result['imbalance_ratio'] = majority / (minority + 1e-9)
        
    return result


def report_class_balance(y_train: np.ndarray, y_test: np.ndarray, 
                         dataset_name: str = "dataset") -> dict:
    train_counts = Counter(y_train)
    test_counts = Counter(y_test)
    return {
        'dataset': dataset_name,
        'train_samples': len(y_train),
        'test_samples': len(y_test),
        'train_distribution': dict(train_counts),
        'test_distribution': dict(test_counts)
    }
