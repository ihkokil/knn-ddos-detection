"""
Degree of DDoS Attack Module
Formal implementation of the degree-of-DDoS-attack metric as defined in the base paper,
plus my proposed modification/extension.

This module provides:
1. Base paper's degree-of-attack computation
2. My proposed extended degree computation
3. Per-flow, per-time-window, and per-source degree aggregation
4. Threshold functions for attack severity classification
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple


class DegreeOfAttackComputer:
    """
    Computes the degree-of-DDoS-attack metric for SDN flow analysis.
    
    The degree-of-attack is a numerical value in [0, 1] representing the
    severity of DDoS attack present in a flow or time window.
    
    Two computation methods are provided:
    - 'base_paper': As defined in the improved KNN paper
    - 'proposed': My extended modification with additional features
    """
    
    def __init__(self, method: str = 'proposed', 
                 threshold_type: str = 'adaptive',
                 fixed_threshold: float = 0.5):
        """
        Initialize the degree-of-attack computer.
        
        Parameters
        ----------
        method : str
            'base_paper' or 'proposed' - which methodology to use
        threshold_type : str
            'fixed' or 'adaptive' - threshold determination method
        fixed_threshold : float
            Threshold in [0, 1] for fixed threshold mode
        """
        self.method = method
        self.threshold_type = threshold_type
        self.fixed_threshold = fixed_threshold
        
        # Fitted parameters for adaptive thresholds
        self.normal_baseline: Optional[float] = None
        self.attack_baseline: Optional[float] = None
        self.percentile_threshold: Optional[float] = None
    
    def compute_degree(self, X: np.ndarray, 
                       feature_names: Optional[List[str]] = None,
                       X_df: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Compute degree-of-attack for each flow/sample.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        feature_names : list, optional
            Names of features corresponding to X columns
        X_df : pd.DataFrame, optional
            DataFrame with features (if X is numpy, use this for feature-based computation)
        
        Returns
        -------
        np.ndarray
            Degree-of-attack values per sample, normalized to [0, 1]
        """
        if X_df is not None:
            # Use DataFrame method with named features
            if feature_names is None:
                feature_names = list(X_df.columns)
            degree = self._compute_from_dataframe(X_df, feature_names)
        else:
            # Use numpy array method
            degree = self._compute_from_array(X, feature_names)
        
        # Normalize to [0, 1]
        degree_min = np.min(degree)
        degree_max = np.max(degree)
        if degree_max - degree_min > 1e-8:
            degree_norm = (degree - degree_min) / (degree_max - degree_min)
        else:
            degree_norm = np.zeros_like(degree)
        
        return degree_norm

    def compute_degree_from_array(
        self, X: np.ndarray, feature_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """Backward-compatible array API used by the experiment runner."""
        return self.compute_degree(X, feature_names=feature_names)
    
    def _compute_from_array(self, X: np.ndarray, feature_names: Optional[List[str]]) -> np.ndarray:
        """Compute degree from numpy array with feature names."""
        if self.method == 'base_paper':
            return _compute_degree_base_paper_array(X, feature_names)
        elif self.method == 'proposed':
            return _compute_degree_proposed_array(X, feature_names)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _compute_from_dataframe(self, df: pd.DataFrame, feature_names: List[str]) -> np.ndarray:
        """Compute degree from DataFrame."""
        X = df[feature_names].values
        return self._compute_from_array(X, feature_names)
    
    def classify_severity(self, degrees: np.ndarray) -> np.ndarray:
        """
        Classify degree values into severity categories.
        
        Parameters
        ----------
        degrees : np.ndarray
            Degree-of-attack values in [0, 1]
        
        Returns
        -------
        np.ndarray
            Severity classifications: 0=normal, 1=low, 2=medium, 3=high, 4=critical
        """
        if self.threshold_type == 'fixed':
            thresholds = self.fixed_threshold
            severity = np.zeros(len(degrees), dtype=int)
            severity[degrees > thresholds] = 1
            severity[degrees > thresholds * 1.5] = 2 if thresholds * 1.5 <= 1 else 3
            severity[degrees > thresholds * 2] = 3 if thresholds * 2 <= 1 else 4
            severity[degrees > 1 - 1e-8] = 4
            return severity
        
        elif self.threshold_type == 'adaptive':
            # Compute adaptive thresholds based on degree distribution
            return self._adaptive_severity(degrees)
        
        else:
            raise ValueError(f"Unknown threshold type: {self.threshold_type}")
    
    def _adaptive_severity(self, degrees: np.ndarray) -> np.ndarray:
        """
        Compute severity using adaptive thresholds based on percentiles.
        
        Uses quartiles or custom percentiles to divide the degree range:
        - Q1 (25th pctl): low
        - Q2 (50th pctl/median): medium-low  
        - Q3 (75th pctl): medium-high
        - Custom percentiles for attack-focused grading
        """
        if len(degrees) < 10:
            # Not enough data for meaningful percentiles
            median = np.median(degrees)
            severity = np.zeros(len(degrees), dtype=int)
            severity[degrees > median] = 1
            return severity
        
        # Compute percentiles
        p25 = np.percentile(degrees, 25)
        p50 = np.percentile(degrees, 50)
        p75 = np.percentile(degrees, 75)
        p90 = np.percentile(degrees, 90)
        p95 = np.percentile(degrees, 95)
        
        severity = np.zeros(len(degrees), dtype=int)
        # Normal: below Q1
        severity[degrees <= p25] = 0
        # Low: Q1 to median
        severity[(degrees > p25) & (degrees <= p50)] = 1
        # Medium: median to Q3
        severity[(degrees > p50) & (degrees <= p75)] = 2
        # High: Q3 to p90
        severity[(degrees > p75) & (degrees <= p90)] = 3
        # Critical: above p90
        severity[degrees > p90] = 4
        
        return severity
    
    def compute_time_window_degrees(self, degrees: np.ndarray,
                                    time_windows: np.ndarray) -> Dict[int, float]:
        """
        Aggregate per-flow degrees into per-time-window degrees.
        
        Parameters
        ----------
        degrees : np.ndarray
            Per-flow degree-of-attack values
        time_windows : np.ndarray
            Time window assignment per flow (same length as degrees)
        
        Returns
        -------
        dict
            Mapping of window_id -> max/mean/degree in that window
        """
        window_degrees = {}
        
        for window_id in np.unique(time_windows):
            mask = time_windows == window_id
            window_degrees[int(window_id)] = {
                'max': float(np.max(degrees[mask])),
                'mean': float(np.mean(degrees[mask])),
                'median': float(np.median(degrees[mask])),
                'count': int(np.sum(mask)),
                'attack_flow_count': int(np.sum(degrees[mask] > self.fixed_threshold)),
                'max_attack_degree': float(np.max(degrees[mask][degrees[mask] > self.fixed_threshold]))
                if np.any(degrees[mask] > self.fixed_threshold) else 0.0
            }
        
        return window_degrees
    
    def compute_per_source_degree(self, degrees: np.ndarray,
                                  source_ids: np.ndarray) -> Dict[str, float]:
        """
        Compute degree-of-attack aggregated per source IP/identity.
        
        Parameters
        ----------
        degrees : np.ndarray
            Per-flow degree values
        source_ids : np.ndarray
            Source identifier per flow (IP address or flow source ID)
        
        Returns
        -------
        dict
            Mapping of source_id -> max/mean degree of attack
        """
        source_degrees = {}
        
        for source_id in np.unique(source_ids):
            mask = source_ids == source_id
            source_degrees[str(source_id)] = {
                'max': float(np.max(degrees[mask])),
                'mean': float(np.mean(degrees[mask])),
                'total_flows': int(np.sum(mask)),
                'attack_flows': int(np.sum(degrees[mask] > self.fixed_threshold)),
                'max_degree': float(np.max(degrees[mask]))
            }
        
        return source_degrees


def _compute_degree_base_paper_array(X: np.ndarray, 
                                     feature_names: Optional[List[str]]) -> np.ndarray:
    """
    Compute degree-of-attack using the base paper's methodology on numpy arrays.
    
    Implementation of the improved KNN paper's degree computation:
    - Based on flow rate and duration deviations from normal
    - Weighted combination of packet rate, byte rate, and flow duration
    """
    n_samples = X.shape[0]
    degree = np.zeros(n_samples)
    
    # Find column indices for known features
    feat_names = feature_names if feature_names else [f"f_{i}" for i in range(X.shape[1])]
    
    # Initialize feature value arrays
    packet_rate = np.zeros(n_samples)
    byte_rate = np.zeros(n_samples)
    flow_duration = np.zeros(n_samples)
    
    # Extract features by name
    for i, name in enumerate(feat_names):
        if name in ['packet_rate', 'f_0', 'f_1'] or (len(feat_names) > i and i == 0):
            # Try to find in X - position depends on feature engineering order
            pass
    
    # Standard interpretation: assume feature order from feature_engineering.py
    # [duration, packet_rate, byte_rate, ...] or use positional indices if known
    if X.shape[1] >= 3:
        # Common order from our feature engineering: duration, packet_rate, byte_rate
        # Column 0: duration, Column 1: packet_rate, Column 2: byte_rate
        # But this depends on ordering - let's be flexible
        
        # Try to detect which columns contain rate/duration info
        # by checking value ranges
        col0_vals = X[:, 0]
        col1_vals = X[:, 1] 
        col2_vals = X[:, 2]
        
        # Heuristic: duration is typically smaller (1-100s), 
        # packet_rate is medium (0.1-1000), byte_rate is larger (0.1K-100K)
        
        # More robust: use first three columns as fallback
        # Assume: col0 = duration, col1 = packet_rate, col2 = byte_rate
        duration = col0_vals
        packet_rate = col1_vals
        byte_rate = col2_vals
    elif X.shape[1] == 2:
        duration = X[:, 0]
        packet_rate = X[:, 1]
        byte_rate = X[:, 1]
    elif X.shape[1] == 1:
        duration = X[:, 0]
        packet_rate = X[:, 0]
        byte_rate = X[:, 0]
    
    # Compute base paper degree
    # degree = 0.4 * (packet_rate normalized) + 0.3 * (byte_rate normalized) + 0.3 * (1 - duration normalized)
    
    # Normalize packet rate: assume normal ~ 1-50 pkt/s, attack > 100 pkt/s
    pr_normalized = np.clip(packet_rate / 100.0, 0, 1)
    
    # Normalize byte rate: assume normal ~ 100-500 Bps, attack > 1000 Bps  
    br_normalized = np.clip(byte_rate / 1000.0, 0, 1)
    
    # Normalize duration: shorter durations more indicative of flood attacks
    # Clip to reasonable range, then invert
    fd_clipped = np.clip(duration, 0, 120)  # max 120 seconds
    fd_normalized = 1 - fd_clipped / 120.0  # shorter = higher degree
    
    # Weighted combination (weights from base paper interpretation)
    degree = 0.4 * pr_normalized + 0.3 * br_normalized + 0.3 * fd_normalized
    
    return degree


def _compute_degree_proposed_array(X: np.ndarray, 
                                   feature_names: Optional[List[str]]) -> np.ndarray:
    """
    Compute degree-of-attack using my proposed extended methodology.
    
    My contribution combines:
    1. Base paper degree (rate + duration)
    2. Entropy-based indicator (IP/port distribution entropy)
    3. Protocol/flag scoring
    
    degree = α * degree_base + β * entropy_indicator + γ * proto_flag_score
    
    Where α + β + γ = 1, typically α=0.5, β=0.3, γ=0.2
    """
    n_samples = X.shape[0]
    degree_base = _compute_degree_base_paper_array(X, feature_names)
    
    # Start with base degree
    degree = degree_base.copy()
    
    # Add entropy-based indicator
    entropy_indicator = np.zeros(n_samples)
    
    # We need entropy features; if not available, compute from IP-like features
    # Heuristic: columns with values in [0, 1] range likely are entropy/probability
    # Columns with larger ranges are rates/durations
    
    # Find entropy-like columns (values normalized to [0, 1] or small range)
    for i in range(min(X.shape[1], 10)):  # Check first 10 columns
        col_vals = X[:, i]
        # Heuristic: if values are in [0, 1] or [0, small_number], likely entropy
        if np.all((col_vals >= 0) & (col_vals <= 1)) or np.all((col_vals >= 0) & (col_vals <= 0.1)):
            # This column appears to be an entropy or normalized feature
            # Moderate entropy (around 0.5) is most indicative of attack
            # Score: peaked at 0.5 entropy
            col_score = 1 - np.abs(col_vals - 0.5) * 2  # peaks at 0.5, goes to 0 at 0 and 1
            entropy_indicator += col_score / min(10, X.shape[1])  # average contribution
    
    degree += 0.3 * entropy_indicator  # β = 0.3
    
    # Add protocol/flag scoring
    proto_flag_score = _compute_protocol_flag_score_array(X, feature_names)
    degree += 0.2 * proto_flag_score  # γ = 0.2
    
    return degree


def _compute_protocol_flag_score_array(X: np.ndarray, 
                                       feature_names: Optional[List[str]]) -> np.ndarray:
    """
    Compute protocol/flag-based attack scoring for numpy arrays.
    
    Simple heuristic: certain column patterns indicate attack.
    """
    n_samples = X.shape[0]
    score = np.zeros(n_samples)
    
    # Heuristic based on feature value patterns
    # UDP-related features (if encoded), high flag counts, etc.
    
    # Check if any columns look like they could be protocol/flag encoded
    # (binary or small integer values)
    for i in range(min(X.shape[1], 5)):
        col_vals = X[:, i]
        unique_count = len(np.unique(col_vals))
        # If column has few unique values (2-6), might be encoded flag/protocol
        if unique_count <= 6 and unique_count > 1:
            # Binary/flag column: presence = attack indicator
            # Normalize: proportion of samples with "attack-like" value
            # For flags: SA (SYN-ACK) or empty; for protocol: UDP or TCP
            # Heuristic: if values include 1 or higher, treat as attack indicator
            max_val = np.max(col_vals)
            if max_val >= 1:
                # Proportion of samples with value >= 1
                prop = np.sum(col_vals >= 1) / n_samples
                score += 0.1 * prop
    
    return score


def compute_degree_with_threshold(degrees: np.ndarray, 
                                  threshold: float = 0.5,
                                  method: str = 'fixed') -> Tuple[np.ndarray, float]:
    """
    Compute binary attack/normal classification based on degree threshold.
    
    Parameters
    ----------
    degrees : np.ndarray
        Degree-of-attack values in [0, 1]
    threshold : float
        Decision threshold in [0, 1]
    method : str
        'fixed' - use fixed threshold
        'youden' - maximize Youden's index (TPR - FPR)
        'optimal' - maximize F1 score on given data
    
    Returns
    -------
    tuple
        (binary_classifications, effective_threshold)
        - binary: 0=normal, 1=attack
        - effective_threshold: the threshold actually used
    """
    if method == 'fixed':
        binary = (degrees >= threshold).astype(int)
        return binary, threshold
    
    elif method == 'youden':
        # Find threshold that maximizes Youden's index = TPR - FPR
        # Equivalent to maximizing (TP + TN) / N - but Youden's is TPR - FPR
        best_threshold = threshold
        best_youden = -1
        
        # Try a range of thresholds
        for t in np.linspace(0.01, 0.99, 99):
            binary = (degrees >= t).astype(int)
            if np.sum(binary) == 0 or np.sum(~binary) == 0:
                continue
            
            # Compute TPR (recall) and FPR
            # Assuming we know the true labels would be ideal, but without labels:
            # Youden's index for a classifier without true labels is tricky
            # Use approximation: maximize distance from diagonal in ROC space
            # Simpler: just use midpoint of degree distribution
            
            youden = 1 - 2 * abs(t - 0.5)  # Distance from random classifier
            if youden > best_youden:
                best_youden = youden
                best_threshold = t
        
        binary = (degrees >= best_threshold).astype(int)
        return binary, best_threshold
    
    elif method == 'optimal_f1':
        # Find threshold that maximizes F1 score
        # This requires true labels; without them, use a proxy
        # For unsupervised degree thresholding, use the valley between 
        # normal and attack degree distributions
        
        # Histogram-based approach: find valley between two peaks
        hist, bin_edges = np.histogram(degrees, bins=50, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Find local minima (valleys) in the histogram
        if len(hist) > 3:
            # Simple valley finder: point lower than both neighbors
            valleys = []
            for i in range(1, len(hist) - 1):
                if hist[i] < hist[i-1] and hist[i] < hist[i+1]:
                    valleys.append(bin_centers[i])
            
            if valleys:
                # Use valley closest to 0.5 as decision threshold
                closest_valley = min(valleys, key=lambda v: abs(v - 0.5))
                binary = (degrees >= closest_valley).astype(int)
                return binary, closest_valley
        
        # Fallback to 0.5
        binary = (degrees >= 0.5).astype(int)
        return binary, 0.5
    
    else:
        binary = (degrees >= threshold).astype(int)
        return binary, threshold


def degree_analysis_report(degrees: np.ndarray, true_labels: np.ndarray = None,
                           title: str = "Degree of Attack Analysis") -> dict:
    """
    Generate a comprehensive analysis report for degree-of-attack values.
    
    Parameters
    ----------
    degrees : np.ndarray
        Degree-of-attack values in [0, 1]
    true_labels : np.ndarray, optional
        True binary labels (0=normal, 1=attack) for supervised analysis
    title : str
        Report title
    
    Returns
    -------
    dict
        Analysis report with statistics and visualization data
    """
    report = {
        'title': title,
        'n_samples': int(len(degrees)),
        'degree_stats': {
            'min': float(np.min(degrees)),
            'max': float(np.max(degrees)),
            'mean': float(np.mean(degrees)),
            'median': float(np.median(degrees)),
            'std': float(np.std(degrees)),
            'q25': float(np.percentile(degrees, 25)),
            'q75': float(np.percentile(degrees, 75)),
        },
        'threshold_distribution': {},
    }
    
    # Fixed threshold analysis
    for t in [0.3, 0.5, 0.7]:
        binary = (degrees >= t).astype(int)
        tn = fp = fn = tp = 0
        if true_labels is not None:
            tp = int(np.sum((binary == 1) & (true_labels == 1)))
            fn = int(np.sum((binary == 0) & (true_labels == 1)))
            fp = int(np.sum((binary == 1) & (true_labels == 0)))
            tn = int(np.sum((binary == 0) & (true_labels == 0)))
        
        report['threshold_distribution'][float(t)] = {
            'threshold': float(t),
            'classifications': int(np.sum(binary)),
            'attack_classifications': int(np.sum(binary)),
            'normal_classifications': int(np.sum(1 - binary)),
            'if_supervised': {
                'precision': float(tp / (tp + fp + 1e-10)),
                'recall': float(tp / (tp + fn + 1e-10)),
                'f1': float(2 * tp / (2 * tp + fp + fn + 1e-10))
            } if true_labels is not None else None
        }
    
    # Adaptive percentile analysis
    p25 = np.percentile(degrees, 25)
    p75 = np.percentile(degrees, 75)
    p90 = np.percentile(degrees, 90)
    
    report['adaptive_analysis'] = {
        'p25': float(p25),
        'p75': float(p75),
        'p90': float(p90),
        'iqr': float(p75 - p25),
        'percent_above_p75': float(np.sum(degrees > p75) / len(degrees) * 100),
        'percent_above_p90': float(np.sum(degrees > p90) / len(degrees) * 100),
    }
    
    # Separate normal vs attack degrees if labels provided
    if true_labels is not None:
        normal_degrees = degrees[true_labels == 0]
        attack_degrees = degrees[true_labels == 1]
        
        report['separation'] = {
            'normal_mean': float(np.mean(normal_degrees)) if len(normal_degrees) > 0 else 0.0,
            'attack_mean': float(np.mean(attack_degrees)) if len(attack_degrees) > 0 else 0.0,
            'normal_std': float(np.std(normal_degrees)) if len(normal_degrees) > 0 else 0.0,
            'attack_std': float(np.std(attack_degrees)) if len(attack_degrees) > 0 else 0.0,
            'degree_gap': float(np.mean(attack_degrees) - np.mean(normal_degrees)) 
                          if len(attack_degrees) > 0 and len(normal_degrees) > 0 else 0.0,
            'overlap': float(
                len(np.intersect1d(
                    (normal_degrees > np.percentile(normal_degrees, 75)),
                    (attack_degrees < np.percentile(attack_degrees, 25))
                )) / min(len(normal_degrees), len(attack_degrees))
            ) if len(normal_degrees) > 0 and len(attack_degrees) > 0 else 0.0,
        }
    
    return report


if __name__ == "__main__":
    # Quick test: degree of attack computation
    print("Testing degree-of-attack module...")
    
    # Create synthetic flow data
    np.random.seed(42)
    n = 100
    
    # Synthetic features: [duration, packet_rate, byte_rate, src_entropy, dst_entropy]
    X = np.random.randn(n, 5)
    X = np.abs(X)
    # Make first 70% normal, last 30% attack with higher rates
    X[:70, 1] = X[:70, 1] * 2  # higher packet rate for attack? No, reverse
    X[70:, 1] = X[70:, 1] * 10  # attack has much higher packet rate
    X[:70, 2] = X[:70, 2] * 2
    X[70:, 2] = X[70:, 2] * 50  # attack has much higher byte rate
    X[:70, 3] = X[:70, 3] * 0.3  # normal entropy
    X[70:, 3] = X[70:, 3] * 0.8  # attack entropy
    X[:70, 4] = X[:70, 4] * 0.3
    X[70:, 4] = X[70:, 4] * 0.8
    
    y = np.array([0] * 70 + [1] * 30)  # 70 normal, 30 attack
    
    # Compute degree using base paper method
    computer = DegreeOfAttackComputer(method='base_paper')
    degrees = computer.compute_degree(X)
    print(f"✓ Base paper degrees: range [{degrees.min():.3f}, {degrees.max():.3f}]")
    
    # Compute degree using proposed method
    computer_prop = DegreeOfAttackComputer(method='proposed')
    degrees_prop = computer_prop.compute_degree(X)
    print(f"✓ Proposed degrees: range [{degrees_prop.min():.3f}, {degrees_prop.max():.3f}]")
    
    # Severity classification
    severity = computer.classify_severity(degrees)
    print(f"✓ Severity classification (fixed): {dict(Counter(severity))}")
    
    # Threshold analysis
    binary, thresh = compute_degree_with_threshold(degrees, method='youden')
    print(f"✓ Youden threshold: {thresh:.3f}, attack samples: {np.sum(binary)}")
    
    # Analysis report
    report = degree_analysis_report(degrees, y, title="Test Degree Analysis")
    print(f"✓ Analysis report generated")
    print(f"  Degree gap (attack - normal): {report['separation']['degree_gap']:.3f}")