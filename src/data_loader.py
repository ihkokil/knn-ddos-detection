"""
Data Loader Module
Loads datasets (NSL-KDD, CIC-DDoS2019, synthetic traffic) and performs
basic train/test splitting with class balance reporting.

Author: Research Engineer
Based on: Base paper "DDoS Attack Detection Method Based on Improved KNN With 
          the Degree of DDoS Attack in Software-Defined Networks"
          and extended research direction from Paper_2.md
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_dataset_path(dataset_path: str | os.PathLike | None,
                           default_name: str) -> Path:
    """Resolve dataset paths relative to the project, not the caller's cwd."""
    if dataset_path is None:
        return PROJECT_ROOT / "data" / "raw" / default_name
    path = Path(dataset_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_nsl_kdd(dataset_path: str = None) -> tuple:
    """
    Load the NSL-KDD dataset.
    
    The NSL-KDD dataset is an improved version of KDD99, containing 41 features
    and a binary label (normal vs. attack).
    
    Parameters
    ----------
    dataset_path : str, optional
        Path to the NSL-KDD dataset files. If None, uses default paths.
    
    Returns
    -------
    tuple
        (train_features, train_labels, test_features, test_labels, feature_names)
    """
    # NSL-KDD file paths (standard locations)
    if dataset_path is None:
        dataset_path = _resolve_dataset_path(None, "nsl_kdd")
    else:
        dataset_path = _resolve_dataset_path(dataset_path, "nsl_kdd")
    
    # Try to load from common locations
    train_path = os.path.join(dataset_path, "KDDTrain+.txt")
    test_path = os.path.join(dataset_path, "KDDTest+.txt")
    
    # If files don't exist, return with placeholder info
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"NSL-KDD dataset not found at {dataset_path}")
        print("Please download NSL-KDD or specify correct path.")
        return None
    
    # Load training data
    col_names = [
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
        "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
        "num_compromised", "root_shell", "su_attempted", "num_file_creates",
        "num_shells", "num_access_files", "num_outbound_cmds", "count",
        "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
        "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"
    ]
    
    try:
        train_df = pd.read_csv(train_path, header=None, names=col_names)
        test_df = pd.read_csv(test_path, header=None, names=col_names)
    except Exception as e:
        print(f"Error loading NSL-KDD: {e}")
        return None
    
    # Convert label: "normal." -> 0, attack types -> 1
    train_df["label"] = (train_df["label"].astype(str).str.lower()
                         .ne("normal.")).astype(int)
    test_df["label"] = (test_df["label"].astype(str).str.lower()
                        .ne("normal.")).astype(int)
    
    # Separate features and labels
    feature_cols = [c for c in col_names if c not in ["label", "difficulty"]]
    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values
    
    print(f"NSL-KDD loaded: {len(train_df)} training samples, {len(test_df)} test samples")
    print(f"  Training class distribution: {Counter(y_train)}")
    print(f"  Test class distribution: {Counter(y_test)}")
    
    return X_train, y_train, X_test, y_test, feature_cols


def load_cicddos2019(dataset_path: str = None) -> tuple:
    """
    Load the CIC-DDoS2019 dataset.
    
    CIC-DDoS2019 is a more recent dataset containing DDoS attack flows with
    higher fidelity and more realistic traffic patterns than NSL-KDD.
    
    Parameters
    ----------
    dataset_path : str, optional
        Path to the CIC-DDoS2019 dataset.
    
    Returns
    -------
    tuple
        (X_train, y_train, X_test, y_test, feature_cols)
    """
    if dataset_path is None:
        dataset_path = _resolve_dataset_path(None, "cicddos2019")
    else:
        dataset_path = _resolve_dataset_path(dataset_path, "cicddos2019")
    
    # CIC-DDoS2019 CSV files
    train_path = os.path.join(dataset_path, "Wednesday_workingHours_pcap_ISCX.csv")
    test_path = os.path.join(dataset_path, "Thursday_workingHours_pcap_ISCX.csv")
    
    if not os.path.exists(train_path) and not os.path.exists(test_path):
        print(f"CIC-DDoS2019 dataset not found at {dataset_path}")
        print("Please download CIC-DDoS2019 or specify correct path.")
        return None
    
    try:
        # Load with sample for efficiency; full dataset may be large
        train_df = pd.read_csv(train_path, nrows=100000)
        test_df = pd.read_csv(test_path, nrows=100000)
    except Exception as e:
        print(f"Error loading CIC-DDoS2019: {e}")
        return None
    
    if "Label" not in train_df or "Label" not in test_df:
        raise ValueError("CIC-DDoS2019 files must contain a 'Label' column")

    def encode_labels(frame: pd.DataFrame) -> pd.Series:
        labels = frame["Label"].astype(str).str.strip().str.lower()
        encoded = pd.Series(
            np.where(labels.eq("benign"), 0,
                     labels.str.contains("ddos|attack|flood|dos", regex=True)),
            index=frame.index,
            dtype="int64",
        )
        unknown = ~labels.eq("benign") & ~labels.str.contains(
            "ddos|attack|flood|dos", regex=True
        )
        if unknown.any():
            values = sorted(labels[unknown].unique())
            raise ValueError(f"Unsupported CIC-DDoS2019 labels: {values[:10]}")
        return encoded

    train_df["Label"] = encode_labels(train_df)
    test_df["Label"] = encode_labels(test_df)
    
    # Separate features and labels
    # Drop non-feature columns
    drop_cols = {"Label", "Src IP", "Dst IP", "Timestamp", "Flow ID"}
    feature_cols = [
        c for c in train_df.columns
        if c not in drop_cols and c in test_df.columns
    ]
    numeric_cols = train_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    numeric_test = test_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    feature_cols = [c for c in feature_cols if numeric_cols[c].notna().any()]
    X_train = numeric_cols[feature_cols].values.astype(float)
    y_train = train_df["Label"].values.astype(int)
    X_test = numeric_test[feature_cols].values.astype(float)
    y_test = test_df["Label"].values.astype(int)
    
    # Handle any remaining NaN/Inf
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"CIC-DDoS2019 loaded: {len(train_df)} training samples, {len(test_df)} test samples")
    print(f"  Training class distribution: {Counter(y_train)}")
    print(f"  Test class distribution: {Counter(y_test)}")
    
    return X_train, y_train, X_test, y_test, feature_cols


def load_synthetic_traffic(generator_config: dict = None) -> tuple:
    """
    Load or generate synthetic SDN traffic data for DDoS detection.
    
    Generates synthetic normal and DDoS flow data with controllable parameters.
    Useful for experiments where real datasets are unavailable or for 
    controlled variable analysis.
    
    Parameters
    ----------
    generator_config : dict, optional
        Configuration dict with parameters for synthetic data generation.
        Keys: 'num_normal', 'num_attack', 'attack_intensity', 'num_features',
              'attack_types' (list), 'random_seed'.
    
    Returns
    -------
    tuple
        (X_train, y_train, X_test, y_test, feature_names)
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
    
    np.random.seed(generator_config['random_seed'])
    
    num_normal = generator_config['num_normal']
    num_attack = generator_config['num_attack']
    num_features = generator_config['num_features']
    attack_intensity = generator_config['attack_intensity']
    attack_types = generator_config['attack_types']
    
    # Generate normal traffic features
    # Features: [duration, packet_rate, byte_rate, src_entropy, dst_entropy, 
    #            port_entropy, flag_SYN, flag_ACK, flag_FIN, protocol]
    X_normal = np.random.randn(num_normal, num_features)
    # Ensure non-negative for rate/features
    X_normal = np.abs(X_normal)
    # Scale features to realistic ranges
    X_normal[:, 1] = X_normal[:, 1] * 10 + 1  # packet_rate: 1-11
    X_normal[:, 2] = X_normal[:, 2] * 100 + 10  # byte_rate: 10-110
    X_normal[:, 0] = X_normal[:, 0] * 10 + 1  # duration: 1-11
    # Entropy features: 0-1 range
    X_normal[:, 3:6] = np.abs(X_normal[:, 3:6])
    
    # Generate attack traffic features
    # Attack flows have higher packet/byte rates and different entropy patterns
    X_attack = np.random.randn(num_attack, num_features)
    X_attack = np.abs(X_attack)
    # Scale attack features to be higher than normal
    X_attack[:, 1] = X_attack[:, 1] * attack_intensity + 1  # higher packet rate
    X_attack[:, 2] = X_attack[:, 2] * attack_intensity + 10  # higher byte rate
    X_attack[:, 0] = X_attack[:, 0] * 5 + 1  # duration
    X_attack[:, 3:6] = np.abs(X_attack[:, 3:6])
    
    # Create labels: 0 = normal, 1 = attack
    y_normal = np.zeros(num_normal)
    y_attack = np.ones(num_attack)
    
    # Combine dataset
    X = np.vstack([X_normal, X_attack])
    y = np.concatenate([y_normal, y_attack])
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(X))
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=generator_config['random_seed'],
        stratify=y
    )
    
    feature_names = [f"feature_{i}" for i in range(num_features)]
    
    print(f"Synthetic traffic generated: {len(X)} total samples")
    print(f"  Train: {len(X_train)} samples, Class dist: {Counter(y_train)}")
    print(f"  Test: {len(X_test)} samples, Class dist: {Counter(y_test)}")
    print(f"  Attack intensity: {attack_intensity}")
    print(f"  Attack types: {attack_types}")
    
    return X_train, y_train, X_test, y_test, feature_names


def load_dataset(config: dict = None) -> dict:
    """
    Load dataset based on configuration.
    
    Parameters
    ----------
    config : dict, optional
        Configuration dictionary. If None, uses default config.
    
    Returns
    -------
    dict
        Dictionary containing loaded data and metadata:
        - 'X_train', 'X_test', 'y_train', 'y_test': numpy arrays
        - 'feature_names': list of feature names
        - 'dataset_name': str
        - 'class_distributions': dict with train/test counts
        - 'imbalance_ratio': ratio of majority/minority class
    """
    if config is None:
        config = {
            'dataset': 'CIC-DDoS2019',
            'random_seed': 42
        }
    
    config = {**{'dataset': 'CIC-DDoS2019', 'random_seed': 42}, **(config or {})}
    dataset_name = config['dataset']
    seed = config['random_seed']
    
    np.random.seed(seed)
    
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
    
    dataset_path = config.get("dataset_path")
    if dataset_name == 'NSL-KDD':
        data = load_nsl_kdd(dataset_path)
        if data is not None:
            X_train, y_train, X_test, y_test, feature_cols = data
            result['X_train'] = X_train
            result['X_test'] = X_test
            result['y_train'] = y_train
            result['y_test'] = y_test
            result['feature_names'] = feature_cols
    
    elif dataset_name == 'CIC-DDoS2019':
        data = load_cicddos2019(dataset_path)
        if data is not None:
            X_train, y_train, X_test, y_test, feature_cols = data
            result['X_train'] = X_train
            result['X_test'] = X_test
            result['y_train'] = y_train
            result['y_test'] = y_test
            result['feature_names'] = feature_cols
    
    elif dataset_name == 'synthetic':
        # Use a small synthetic dataset for quick testing
        data = load_synthetic_traffic({
            'num_normal': 2000,
            'num_attack': 800,
            'attack_intensity': 5.0,
            'num_features': 10,
            'attack_types': ['udp_flood', 'syn_flood', 'icmp_flood'],
            'random_seed': seed
        })
        if data is not None:
            X_train, y_train, X_test, y_test, feature_names = data
            result['X_train'] = X_train
            result['X_test'] = X_test
            result['y_train'] = y_train
            result['y_test'] = y_test
            result['feature_names'] = feature_names
    
    else:
        print(f"Unknown dataset: {dataset_name}")
        print(f"Available options: NSL-KDD, CIC-DDoS2019, synthetic")
    
    # Compute class distributions and imbalance ratio
    if result['y_train'] is not None and result['y_test'] is not None:
        train_counts = Counter(result['y_train'])
        test_counts = Counter(result['y_test'])
        result['class_distributions'] = {
            'train': dict(train_counts),
            'test': dict(test_counts)
        }
        
        # Imbalance ratio: majority/minority in training set
        if 0 in train_counts and 1 in train_counts:
            majority = max(train_counts.values())
            minority = min(train_counts.values())
            result['imbalance_ratio'] = majority / minority
        elif 0 in train_counts:
            result['imbalance_ratio'] = train_counts[0] / 1  # only normal
        elif 1 in train_counts:
            result['imbalance_ratio'] = train_counts[1] / 1  # only attack
        else:
            result['imbalance_ratio'] = 1.0
    
    print(f"Dataset: {dataset_name} loaded successfully")
    imbalance = result.get("imbalance_ratio", 1.0)
    print(f"  Imbalance ratio (train): {imbalance:.2f}:1")
    print(f"  Feature names: {len(result['feature_names'])} features")
    
    return result


def report_class_balance(y_train: np.ndarray, y_test: np.ndarray, 
                         dataset_name: str = "dataset") -> dict:
    """
    Report class balance statistics for train and test sets.
    
    Parameters
    ----------
    y_train : np.ndarray
        Training set labels (0 = normal, 1 = attack)
    y_test : np.ndarray
        Test set labels (0 = normal, 1 = attack)
    dataset_name : str
        Name of the dataset for display purposes
    
    Returns
    -------
    dict
        Class balance statistics
    """
    train_counts = Counter(y_train)
    test_counts = Counter(y_test)
    
    total_train = len(y_train)
    total_test = len(y_test)
    
    balance_report = {
        'dataset': dataset_name,
        'train_total': total_train,
        'test_total': total_test,
        'train_normal': train_counts.get(0, 0),
        'train_attack': train_counts.get(1, 0),
        'test_normal': test_counts.get(0, 0),
        'test_attack': test_counts.get(1, 0),
        'train_imbalance_ratio': train_counts.get(0, 0) / max(train_counts.get(1, 1), 1),
        'test_imbalance_ratio': test_counts.get(0, 0) / max(test_counts.get(1, 1), 1),
        'train_percent_attack': train_counts.get(1, 0) / total_train * 100,
        'test_percent_attack': test_counts.get(1, 0) / total_test * 100
    }
    
    print(f"\n=== Class Balance Report: {dataset_name} ===")
    print(f"  Training set: {total_train} samples")
    print(f"    Normal: {train_counts.get(0, 0)} ({train_counts.get(1, 0) / total_train * 100:.1f}% attack)")
    print(f"    Attack: {train_counts.get(1, 0)} ({100 - train_counts.get(1, 0) / total_train * 100:.1f}% normal)")
    print(f"  Test set: {total_test} samples")
    print(f"    Normal: {test_counts.get(0, 0)} ({test_counts.get(1, 0) / total_test * 100:.1f}% attack)")
    print(f"    Attack: {test_counts.get(1, 0)} ({100 - test_counts.get(1, 0) / total_test * 100:.1f}% normal)")
    print(f"  Imbalance ratio (train): {balance_report['train_imbalance_ratio']:.2f}:1")
    
    return balance_report


if __name__ == "__main__":
    # Quick test: load configuration and dataset
    print("Testing data_loader module...")
    config = {
        'dataset': 'synthetic',
        'random_seed': 42
    }
    data = load_dataset(config)
    if data['X_train'] is not None:
        print(f"✓ Data loaded: {data['X_train'].shape} training features")
        print(f"✓ Labels shape: {data['y_train'].shape}")
        print(f"✓ Feature names: {len(data['feature_names'])} features")
    else:
        print("! Data loading returned None - check dataset paths")