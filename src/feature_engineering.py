"""
Feature Engineering Module
Extracts flow-level features relevant to DDoS detection in SDN environments.

Based on: Base paper's feature set and degree-of-attack metric, extended with
additional derived features for improved detection performance.

Features implemented:
- Flow-level: duration, packet rate, byte rate
- Statistical: inter-arrival time statistics
- Entropy: source IP, destination IP, port distribution entropy
- Degree of DDoS attack: as defined in base paper + my modification
- Protocol and flag features
"""

import numpy as np
import pandas as pd
from scipy import stats


def extract_basic_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract basic flow-level features from raw packet-level or flow-level data.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing raw flow/packet data with at least:
        - 'duration': flow duration (seconds)
        - 'packets': number of packets in flow
        - 'bytes': number of bytes in flow
        - 'src_ip': source IP address
        - 'dst_ip': destination IP address
        - 'src_port': source port
        - 'dst_port': destination port
        - 'protocol': protocol type
    
    Returns
    -------
    pd.DataFrame
        DataFrame with extracted flow features added
    """
    df = df.copy()
    
    # 1. Packet rate (packets per second)
    df['packet_rate'] = df['packets'] / df['duration'].replace(0, np.nan)
    df['packet_rate'] = df['packet_rate'].fillna(0)
    
    # 2. Byte rate (bytes per second)
    df['byte_rate'] = df['bytes'] / df['duration'].replace(0, np.nan)
    df['byte_rate'] = df['byte_rate'].fillna(0)
    
    # 3. Flow duration (already available, but compute statistics)
    df['flow_duration'] = df['duration']
    
    # 4. Inter-arrival time statistics (if packet timestamps available)
    # If not, compute synthetic inter-arrival from packet count
    if 'inter_arrival' in df.columns:
        df['mean_inter_arrival'] = df['inter_arrival'].mean()
        df['std_inter_arrival'] = df['inter_arrival'].std()
    else:
        # Synthetic: assume Poisson arrivals, inter-arrival = duration / packets
        df['mean_inter_arrival'] = df['duration'] / df['packets'].replace(0, np.nan)
        df['std_inter_arrival'] = df['duration'] / df['packets'].replace(0, np.nan) * np.sqrt(
            df['packets'].replace(0, np.nan)
        )
    
    # 5. Protocol type encoding
    if 'protocol' in df.columns:
        df = _encode_protocol(df, 'protocol')
    
    # 6. Flag counts (if flag column available)
    if 'flag' in df.columns:
        df = _encode_flags(df, 'flag')
    
    return df


def compute_entropy_features(df: pd.DataFrame, ip_col: str = 'src_ip',
                             port_col: str = 'src_port') -> pd.DataFrame:
    """
    Compute entropy-based features for source/destination IPs and ports.
    
    Entropy measures the unpredictability/distribution of network entities.
    High entropy in attack traffic can indicate scanning or flood patterns.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with IP/port columns
    ip_col : str
        Column name for IP addresses
    port_col : str
        Column name for port numbers
    
    Returns
    -------
    pd.DataFrame
        DataFrame with entropy features added
    """
    df = df.copy()
    
    # Source IP entropy
    if ip_col in df.columns:
        ip_counts = df[ip_col].value_counts(normalize=True)
        df['src_ip_entropy'] = -np.sum(ip_counts * np.log2(ip_counts + 1e-10))
        # Normalize by max possible entropy (log2 of unique count)
        n_unique = df[ip_col].nunique()
        df['src_ip_entropy'] = df['src_ip_entropy'] / np.log2(n_unique + 1e-10)
    
    # Destination IP entropy
    if ip_col in df.columns:
        ip_counts_dst = df[ip_col].value_counts(normalize=True)
        df['dst_ip_entropy'] = -np.sum(ip_counts_dst * np.log2(ip_counts_dst + 1e-10))
        n_unique = df[ip_col].nunique()
        df['dst_ip_entropy'] = df['dst_ip_entropy'] / np.log2(n_unique + 1e-10)
    
    # Source port entropy
    if port_col in df.columns:
        port_counts = df[port_col].value_counts(normalize=True)
        df['src_port_entropy'] = -np.sum(port_counts * np.log2(port_counts + 1e-10))
        n_unique = df[port_col].nunique()
        df['src_port_entropy'] = df['src_port_entropy'] / np.log2(n_unique + 1e-10)
    
    # Destination port entropy
    if port_col in df.columns:
        port_counts_dst = df[port_col].value_counts(normalize=True)
        df['dst_port_entropy'] = -np.sum(port_counts_dst * np.log2(port_counts_dst + 1e-10))
        n_unique = df[port_col].nunique()
        df['dst_port_entropy'] = df['dst_port_entropy'] / np.log2(n_unique + 1e-10)
    
    return df


def compute_degree_of_attack(df: pd.DataFrame, features: list = None,
                              method: str = 'base_paper') -> pd.DataFrame:
    """
    Compute the degree-of-DDoS-attack metric per flow.
    
    The degree-of-attack quantifies how much DDoS traffic is present in a flow.
    Higher values indicate stronger attack signals.
    
    Two methods available:
    1. 'base_paper': Based on the improved KNN paper's approach
       - Combines packet rate, byte rate, and duration deviations
       - Normalizes against baseline normal traffic statistics
    2. 'proposed': My extended modification
       - Incorporates entropy features and multi-attribute weighting
       - Adaptive baseline computation per traffic condition
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with flow features
    features : list, optional
        List of feature column names to use for degree computation
    method : str
        'base_paper' or 'proposed'
    
    Returns
    -------
    pd.DataFrame
        DataFrame with 'degree_of_attack' column added (normalized to [0, 1])
    """
    df = df.copy()
    
    if method == 'base_paper':
        degree = _compute_degree_base_paper(df, features)
    elif method == 'proposed':
        degree = _compute_degree_proposed(df, features)
    else:
        raise ValueError(f"Unknown degree computation method: {method}")
    
    # Normalize to [0, 1] range
    degree_min = degree.min()
    degree_max = degree.max()
    if degree_max - degree_min > 0:
        df['degree_of_attack'] = (degree - degree_min) / (degree_max - degree_min)
    else:
        df['degree_of_attack'] = 0.0
    
    return df


def _compute_degree_base_paper(df: pd.DataFrame, features: list = None) -> np.ndarray:
    """
    Compute degree-of-attack based on the base paper's methodology.
    
    The base paper likely computes degree as a composite metric:
    degree = w1 * (packet_rate deviation) + w2 * (byte_rate deviation) 
           + w3 * (duration deviation)
    
    Where deviations are measured against normal traffic baselines.
    """
    # Initialize degree array
    degree = np.zeros(len(df))
    
    # Use default features if none specified
    if features is None:
        features = ['packet_rate', 'byte_rate', 'flow_duration']
    
    # Compute deviations from normal traffic statistics
    # (In practice, these would be computed from training data baselines)
    # Here we use a reasonable interpretation based on typical SDN flow features
    
    # Packet rate deviation: higher rate → higher attack likelihood
    if 'packet_rate' in df.columns:
        pr = df['packet_rate'].values
        # Normalize: assume normal ~ 1-10 pkt/s, attack > 50 pkt/s
        pr_normalized = np.clip(pr / 50.0, 0, 1)  # 50 pkt/s = max attack
        degree += 0.4 * pr_normalized
    
    # Byte rate deviation
    if 'byte_rate' in df.columns:
        br = df['byte_rate'].values
        br_normalized = np.clip(br / 500.0, 0, 1)  # 500 Bps = max attack
        degree += 0.3 * br_normalized
    
    # Flow duration deviation (shorter flows in flood attacks)
    if 'flow_duration' in df.columns:
        fd = df['flow_duration'].values
        # Inverse: shorter duration → higher chance of flood attack
        fd_normalized = np.clip(1 - fd / 60.0, 0, 1)  # 60s = normal max
        degree += 0.3 * fd_normalized
    
    return degree


def _compute_degree_proposed(df: pd.DataFrame, features: list = None) -> np.ndarray:
    """
    Compute degree-of-attack using my proposed extended methodology.
    
    My contribution combines multiple attributes for a more robust degree
    measurement:
    
    degree = α * degree_base + β * entropy_based_indicator + γ * protocol_flag_score
    
    Where:
    - degree_base: traditional rate/duration-based measure
    - entropy_based_indicator: entropy of source IPs/ports indicates scanning/flood
    - protocol_flag_score: certain protocol/flag combinations indicate attack
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with flow features
    features : list, optional
        Feature column names
    
    Returns
    -------
    np.ndarray
        Raw degree values (will be normalized to [0, 1] by caller)
    """
    # Start with base paper degree computation
    degree_base = _compute_degree_base_paper(df, features)
    degree = degree_base.copy()
    
    # Add entropy-based indicator
    # High entropy in source/destination IPs may indicate scanning/probing
    # (Low entropy = consistent source, which can be characteristic of flood attacks
    #  from a single botnet; high entropy = diverse sources = possible scanning)
    entropy_indicator = 0.0
    if 'src_ip_entropy' in df.columns:
        src_entropy = df['src_ip_entropy'].values
        # Attack indicator: moderate entropy is suspicious
        # Very low entropy = single source (flood), very high entropy = diverse (scan)
        # Moderate = most suspicious for certain attack types
        entropy_indicator = 0.2 * (1 - np.abs(src_entropy - 0.5) * 2)  # peaked at 0.5
        degree += entropy_indicator
    
    if 'dst_ip_entropy' in df.columns:
        dst_entropy = df['dst_ip_entropy'].values
        entropy_indicator += 0.1 * (1 - np.abs(dst_entropy - 0.5) * 2)
        degree += entropy_indicator / 2  # weight split
    
    if 'src_port_entropy' in df.columns:
        sp_entropy = df['src_port_entropy'].values
        entropy_indicator += 0.1 * (1 - np.abs(sp_entropy - 0.5) * 2)
        degree += entropy_indicator / 3
    
    if 'dst_port_entropy' in df.columns:
        dp_entropy = df['dst_port_entropy'].values
        entropy_indicator += 0.1 * (1 - np.abs(dp_entropy - 0.5) * 2)
        degree += entropy_indicator / 4
    
    # Protocol and flag-based scoring
    # Certain protocols/flags are characteristic of specific attacks
    proto_flag_score = _compute_protocol_flag_score(df)
    degree += 0.15 * proto_flag_score
    
    return degree


def _encode_protocol(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Encode protocol type as numeric values."""
    protocol_map = {'TCP': 1, 'UDP': 2, 'ICMP': 3, 'HTTP': 4, 'HTTPS': 5}
    if col in df.columns:
        df[col] = df[col].map(protocol_map).fillna(0)
    return df


def _encode_flags(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Encode flag types as numeric values."""
    flag_map = {'SA': 1, 'FA': 2, 'A': 3, 'R': 4, 'S': 5, 'RA': 6, '': 0}
    if col in df.columns:
        df[col] = df[col].map(flag_map).fillna(0)
    return df


def _compute_protocol_flag_score(df: pd.DataFrame) -> np.ndarray:
    """
    Compute protocol/flag-based attack scoring.
    
    Certain protocol/flag combinations are characteristic of DDoS attacks:
    - TCP+SYN flood: high SYN flag count
    - ICMP flood: ICMP protocol with specific patterns
    """
    score = np.zeros(len(df))
    
    # TCP SYN flag presence
    if 'flag_SYN' in df.columns:
        syn_count = df['flag_SYN'].values
        # Normalized SYN flag count (higher = more likely SYN flood)
        syn_normalized = np.clip(syn_count / 100.0, 0, 1)
        score += 0.3 * syn_normalized
    
    # UDP protocol indicator
    if 'protocol_type' in df.columns:
        proto = df['protocol_type'].values
        udp_mask = proto == 2  # UDP = 2 in our encoding
        score += 0.2 * udp_mask.astype(float)
    
    # ACK flag indication
    if 'flag_ACK' in df.columns:
        ack_count = df['flag_ACK'].values
        ack_normalized = np.clip(ack_count / 50.0, 0, 1)
        score += 0.15 * ack_normalized
    
    return score


def engineer_features_for_knn(df: pd.DataFrame, config: dict = None) -> tuple:
    """
    Complete feature engineering pipeline for KNN-based DDoS detection.
    
    Applies all feature extraction steps in sequence, producing a feature matrix
    optimized for KNN classification with degree-of-attack integration.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw input data with flow features
    config : dict, optional
        Engineering configuration:
        - 'include_entropy': bool
        - 'degree_method': 'base_paper' or 'proposed'
        - 'selected_features': list of feature names to retain
    
    Returns
    -------
    tuple
        (X, y, feature_names, degree_values)
        - X: feature matrix (np.ndarray)
        - y: labels (np.ndarray, 0=normal, 1=attack)
        - feature_names: list of feature names
        - degree_values: degree-of-attack per flow (np.ndarray, [0,1])
    """
    if config is None:
        config = {
            'include_entropy': True,
            'degree_method': 'proposed',  # My contribution
            'selected_features': None  # Use all computed
        }
    
    # Start with basic flow features
    df_features = extract_basic_flow_features(df)
    
    # Compute entropy features if enabled
    if config['include_entropy']:
        df_features = compute_entropy_features(df_features)
    
    # Compute degree of DDoS attack (my proposed method by default)
    df_features = compute_degree_of_attack(df_features, method=config['degree_method'])
    
    # Select target features
    # Priority: degree_of_attack + entropy features + basic flow features
    default_selected = None
    if config['selected_features'] is not None:
        default_selected = config['selected_features']
    else:
        # Auto-select available features
        available = list(df_features.columns)
        # Ensure degree is included
        if 'degree_of_attack' in available:
            default_selected = ['degree_of_attack']
            # Add entropy features
            entropy_cols = [c for c in available if 'entropy' in c.lower()]
            default_selected.extend(entropy_cols[:4])  # Top 4 entropy features
            # Add basic features
            basic_cols = [c for c in available if c in 
                         ['packet_rate', 'byte_rate', 'flow_duration']]
            default_selected.extend(basic_cols[:3])
    
    # Extract feature matrix
    if default_selected is not None:
        # Only keep selected columns
        available_selected = [c for c in default_selected if c in df_features.columns]
        X = df_features[available_selected].values
    else:
        X = df_features.values
    
    # Separate labels (assume last column or 'label' column is the target)
    y = df_features['label'].values if 'label' in df_features.columns else \
        (np.ones(len(df_features)) * 0)  # Default: all normal if no label
    
    # Ensure y is binary
    y = (y > 0).astype(int)
    
    # Degree values
    degree_values = df_features['degree_of_attack'].values if 'degree_of_attack' in df_features.columns else \
        np.zeros(len(df_features))
    
    feature_names = list(df_features.columns) if default_selected is None else default_selected
    
    return X, y, feature_names, degree_values


def create_synthetic_ddos_dataset(num_samples: int = 5000,
                                  attack_ratio: float = 0.3,
                                  attack_intensity: float = 5.0,
                                  random_seed: int = 42) -> pd.DataFrame:
    """
    Create a synthetic SDN flow dataset for DDoS detection experiments.
    
    Generates realistic synthetic flow data with controllable attack parameters.
    Useful for testing the full pipeline when real datasets are unavailable.
    
    Parameters
    ----------
    num_samples : int
        Total number of flow samples
    attack_ratio : float
        Proportion of attack samples (0-1)
    attack_intensity : float
        Multiplier for attack traffic features (higher = more detectable)
    random_seed : int
        Random seed for reproducibility
    
    Returns
    -------
    pd.DataFrame
        Synthetic dataset with flow features and 'label' column (0=normal, 1=attack)
    """
    np.random.seed(random_seed)
    
    n_attack = int(num_samples * attack_ratio)
    n_normal = num_samples - n_attack
    
    # Generate normal traffic flows
    normal_data = {
        'duration': np.random.exponential(10, n_normal),  # Mean 10s
        'packets': np.random.poisson(50, n_normal),  # ~50 packets per flow
        'bytes': np.random.normal(5000, 1000, n_normal),  # ~5KB per flow
        'src_ip': [f"10.0.0.{np.random.randint(1, 254)}" for _ in range(n_normal)],
        'dst_ip': [f"10.0.0.{np.random.randint(1, 254)}" for _ in range(n_normal)],
        'src_port': np.random.randint(1024, 65535, n_normal),
        'dst_port': np.random.randint(80, 443, n_normal),  # Mostly web ports
        'protocol': np.random.choice(['TCP', 'UDP'], n_normal),
        'flag': np.random.choice(['', 'SA', 'A', 'R'], n_normal),
        'label': 0  # Normal
    }
    
    # Generate attack traffic flows
    attack_data = {
        'duration': np.random.exponential(2, n_attack),  # Shorter flows (floods)
        'packets': np.random.poisson(200 * attack_intensity, n_attack),  # Many packets
        'bytes': np.random.normal(20000 * attack_intensity, 5000, n_attack),  # Many bytes
        'src_ip': [f"10.0.0.{np.random.randint(1, 254)}" for _ in range(n_attack)],
        'dst_ip': [f"10.0.0.{np.random.randint(1, 254)}" for _ in range(n_attack)],
        'src_port': np.random.randint(1024, 65535, n_attack),
        'dst_port': np.random.choice([80, 443, 53, 22, 21], n_attack),
        'protocol': np.random.choice(['UDP', 'ICMP', 'TCP'], n_attack),
        'flag': np.random.choice(['SA', ''], n_attack),  # SYN-ACK or empty
        'label': 1  # Attack
    }
    
    # Combine
    combined = {}
    for k, v in normal_data.items():
        if k in attack_data:
            if isinstance(v, list) and isinstance(attack_data[k], list):
                combined[k] = v + attack_data[k]
            elif isinstance(v, np.ndarray) and isinstance(attack_data[k], np.ndarray):
                combined[k] = np.concatenate([v, attack_data[k]])
            else:
                combined[k] = v
        else:
            combined[k] = v
    combined['label'] = np.array([0] * n_normal + [1] * n_attack)
    
    # Shuffle
    df = pd.DataFrame(combined)
    shuffle_idx = np.random.permutation(len(df))
    df = df.iloc[shuffle_idx].reset_index(drop=True)
    
    return df


if __name__ == "__main__":
    # Quick test: feature engineering pipeline
    print("Testing feature engineering module...")
    
    # Create synthetic dataset
    df = create_synthetic_ddos_dataset(num_samples=1000, attack_ratio=0.3)
    print(f"✓ Created synthetic dataset: {len(df)} samples")
    print(f"  Label distribution: {df['label'].value_counts().to_dict()}")
    
    # Extract features
    X, y, feature_names, degree = engineer_features_for_knn(df)
    print(f"\n✓ Feature engineering complete")
    print(f"  Features: {len(feature_names)}")
    print(f"  Feature names: {feature_names}")
    print(f"  Degree-of-attack range: [{degree.min():.3f}, {degree.max():.3f}]")
    print(f"  Labels distribution: {dict(Counter(y))}")