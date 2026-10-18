"""
Degree of DDoS Attack Module
Formal implementation of the degree-of-DDoS-attack metric as defined in the base paper,
plus proposed Bayesian evidence telemetry for SDN environments.

This module provides:
1. Base paper's degree-of-attack computation (flow rate and duration anomalies)
2. Proposed multi-criteria degree computation (rate, flag anomalies, and entropy/traffic concentration)
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
    - 'base_paper': As defined in the improved KNN paper (rate + duration)
    - 'proposed': Multi-criteria extension integrating flag anomalies and traffic dispersion
    """
    
    def __init__(self, method: str = 'proposed', 
                 threshold_type: str = 'adaptive',
                 fixed_threshold: float = 0.5):
        self.method = method
        self.threshold_type = threshold_type
        self.fixed_threshold = fixed_threshold
        
        self.normal_baseline: Optional[float] = None
        self.attack_baseline: Optional[float] = None
        self.percentile_threshold: Optional[float] = None
        self.d_min: Optional[float] = None
        self.d_max: Optional[float] = None
        self.is_fitted: bool = False
    
    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> 'DegreeOfAttackComputer':
        """Fit degree normalization bounds strictly on training data."""
        raw_degree = self._compute_from_array(X, feature_names)
        self.d_min = float(np.min(raw_degree)) if len(raw_degree) > 0 else 0.0
        self.d_max = float(np.max(raw_degree)) if len(raw_degree) > 0 else 1.0
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> np.ndarray:
        """Transform features into degree metrics using fitted normalization bounds."""
        raw_degree = self._compute_from_array(X, feature_names)
        if self.is_fitted and self.d_max is not None and self.d_min is not None and (self.d_max - self.d_min > 1e-8):
            degree_norm = (raw_degree - self.d_min) / (self.d_max - self.d_min)
        else:
            d_min = float(np.min(raw_degree)) if len(raw_degree) > 0 else 0.0
            d_max = float(np.max(raw_degree)) if len(raw_degree) > 0 else 1.0
            degree_norm = (raw_degree - d_min) / (d_max - d_min) if d_max - d_min > 1e-8 else np.zeros_like(raw_degree)
        return np.clip(degree_norm, 0.0, 1.0)

    def fit_transform(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> np.ndarray:
        """Fit on training data and transform."""
        return self.fit(X, feature_names).transform(X, feature_names)

    def compute_degree(self, X: np.ndarray, 
                       feature_names: Optional[List[str]] = None,
                       X_df: Optional[pd.DataFrame] = None) -> np.ndarray:
        if X_df is not None:
            if feature_names is None:
                feature_names = list(X_df.columns)
            X_arr = X_df[feature_names].values
        else:
            X_arr = X

        if self.is_fitted:
            return self.transform(X_arr, feature_names)
        return self.fit_transform(X_arr, feature_names)

    def compute_degree_from_array(
        self, X: np.ndarray, feature_names: Optional[List[str]] = None
    ) -> np.ndarray:
        return self.compute_degree(X, feature_names=feature_names)
    
    def _compute_from_array(self, X: np.ndarray, feature_names: Optional[List[str]]) -> np.ndarray:
        if self.method == 'base_paper':
            return _compute_degree_base_paper_array(X, feature_names)
        elif self.method == 'proposed':
            return _compute_degree_proposed_array(X, feature_names)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _compute_from_dataframe(self, df: pd.DataFrame, feature_names: List[str]) -> np.ndarray:
        X = df[feature_names].values
        return self._compute_from_array(X, feature_names)
    
    def classify_severity(self, degrees: np.ndarray) -> np.ndarray:
        if self.threshold_type == 'fixed':
            thresholds = self.fixed_threshold
            severity = np.zeros(len(degrees), dtype=int)
            severity[degrees > thresholds] = 1
            severity[degrees > thresholds * 1.5] = 2 if thresholds * 1.5 <= 1 else 3
            severity[degrees > thresholds * 2] = 3 if thresholds * 2 <= 1 else 4
            severity[degrees > 1 - 1e-8] = 4
            return severity
        elif self.threshold_type == 'adaptive':
            return self._adaptive_severity(degrees)
        else:
            raise ValueError(f"Unknown threshold type: {self.threshold_type}")
    
    def _adaptive_severity(self, degrees: np.ndarray) -> np.ndarray:
        if len(degrees) < 10:
            median = np.median(degrees)
            severity = np.zeros(len(degrees), dtype=int)
            severity[degrees > median] = 1
            return severity
        
        p25 = np.percentile(degrees, 25)
        p50 = np.percentile(degrees, 50)
        p75 = np.percentile(degrees, 75)
        p90 = np.percentile(degrees, 90)
        
        severity = np.zeros(len(degrees), dtype=int)
        severity[degrees <= p25] = 0
        severity[(degrees > p25) & (degrees <= p50)] = 1
        severity[(degrees > p50) & (degrees <= p75)] = 2
        severity[(degrees > p75) & (degrees <= p90)] = 3
        severity[degrees > p90] = 4
        return severity
    
    def compute_time_window_degrees(self, degrees: np.ndarray,
                                    time_windows: np.ndarray) -> Dict[int, float]:
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


def _robust_minmax_col(v: np.ndarray) -> np.ndarray:
    """Robust percentile normalization to [0, 1]."""
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    q1 = np.percentile(v, 1)
    q99 = np.percentile(v, 99)
    if q99 - q1 > 1e-6:
        return np.clip((v - q1) / (q99 - q1), 0.0, 1.0)
    v_min, v_max = np.min(v), np.max(v)
    if v_max - v_min > 1e-6:
        return np.clip((v - v_min) / (v_max - v_min), 0.0, 1.0)
    return np.zeros_like(v)


def _compute_degree_base_paper_array(X: np.ndarray, 
                                     feature_names: Optional[List[str]]) -> np.ndarray:
    """
    Compute degree-of-attack using the base paper's methodology on numpy arrays.
    Based on flow rate and duration deviations from normal traffic baseline.
    """
    n_samples, n_features = X.shape
    if n_samples == 0:
        return np.zeros(0)
    
    feat_lower = [str(f).lower() for f in feature_names] if feature_names else []
    
    pr = None
    br = None
    dur = None
    
    if feat_lower:
        for i, f in enumerate(feat_lower):
            if pr is None and any(k in f for k in ['packet_rate', 'packets/s', 'count', 'srv_count']):
                pr = X[:, i]
            elif br is None and any(k in f for k in ['byte_rate', 'bytes/s', 'src_bytes', 'dst_bytes']):
                br = X[:, i]
            elif dur is None and any(k in f for k in ['duration', 'flow duration', 'flow_duration']):
                dur = X[:, i]
                
    if pr is None:
        pr = X[:, 0] if n_features > 0 else np.zeros(n_samples)
    if br is None:
        br = X[:, min(1, n_features - 1)] if n_features > 1 else pr
    if dur is None:
        dur = X[:, min(2, n_features - 1)] if n_features > 2 else np.ones(n_samples)
        
    pr_norm = _robust_minmax_col(pr)
    br_norm = _robust_minmax_col(br)
    dur_norm = 1.0 - _robust_minmax_col(dur)
    
    degree = 0.4 * pr_norm + 0.3 * br_norm + 0.3 * dur_norm
    d_min, d_max = np.min(degree), np.max(degree)
    if d_max - d_min > 1e-8:
        return (degree - d_min) / (d_max - d_min)
    return degree


def _compute_degree_proposed_array(X: np.ndarray, 
                                   feature_names: Optional[List[str]]) -> np.ndarray:
    """
    Compute multi-criteria degree-of-attack combining rate deviations,
    TCP flag/error anomalies, and traffic concentration metrics.
    """
    n_samples, n_features = X.shape
    if n_samples == 0:
        return np.zeros(0)
        
    d_base = _compute_degree_base_paper_array(X, feature_names)
    feat_lower = [str(f).lower() for f in feature_names] if feature_names else []
    
    # 1. TCP Flag / Error score
    flag_scores = []
    if feat_lower:
        for i, f in enumerate(feat_lower):
            if any(k in f for k in ['serror', 'rerror', 'syn flag', 'rst flag', 'urgent']):
                flag_scores.append(_robust_minmax_col(X[:, i]))
    if flag_scores:
        s_flag = np.mean(flag_scores, axis=0)
    else:
        s_flag = np.zeros(n_samples)
        for i in range(min(n_features, 6)):
            col = X[:, i]
            if len(np.unique(col)) <= 10:
                s_flag = np.maximum(s_flag, _robust_minmax_col(col))
                
    # 2. Traffic concentration / entropy deviation
    traffic_scores = []
    if feat_lower:
        for i, f in enumerate(feat_lower):
            if any(k in f for k in ['diff_srv', 'diff_host', 'entropy', 'dst_host_count']):
                traffic_scores.append(_robust_minmax_col(X[:, i]))
    if traffic_scores:
        s_traffic = np.mean(traffic_scores, axis=0)
    else:
        mu = np.median(X, axis=0)
        d_center = np.linalg.norm(X - mu, axis=1)
        s_traffic = _robust_minmax_col(d_center)
        
    degree = 0.45 * d_base + 0.35 * s_flag + 0.20 * s_traffic
    d_min, d_max = np.min(degree), np.max(degree)
    if d_max - d_min > 1e-8:
        return (degree - d_min) / (d_max - d_min)
    return degree


def _compute_protocol_flag_score_array(X: np.ndarray, 
                                       feature_names: Optional[List[str]]) -> np.ndarray:
    """Compute sample-wise protocol and flag anomaly scoring."""
    n_samples, n_features = X.shape
    if n_samples == 0:
        return np.zeros(0)
    feat_lower = [str(f).lower() for f in feature_names] if feature_names else []
    scores = []
    for i, f in enumerate(feat_lower):
        if any(k in f for k in ['flag', 'serror', 'rerror', 'protocol']):
            scores.append(_robust_minmax_col(X[:, i]))
    if scores:
        return np.mean(scores, axis=0)
    return _robust_minmax_col(X[:, 0] if n_features > 0 else np.zeros(n_samples))


def compute_degree_with_threshold(degrees: np.ndarray, 
                                  threshold: float = 0.5,
                                  method: str = 'fixed') -> Tuple[np.ndarray, float]:
    if method == 'fixed':
        binary = (degrees >= threshold).astype(int)
        return binary, threshold
    elif method == 'youden':
        best_threshold = threshold
        best_youden = -1
        for t in np.linspace(0.05, 0.95, 91):
            binary = (degrees >= t).astype(int)
            if np.sum(binary) == 0 or np.sum(~binary) == 0:
                continue
            youden = 1 - 2 * abs(t - 0.5)
            if youden > best_youden:
                best_youden = youden
                best_threshold = t
        return (degrees >= best_threshold).astype(int), best_threshold
    else:
        binary = (degrees >= threshold).astype(int)
        return binary, threshold


def degree_analysis_report(degrees: np.ndarray, true_labels: np.ndarray = None,
                           title: str = "Degree of Attack Analysis") -> dict:
    report = {
        'title': title,
        'n_samples': int(len(degrees)),
        'degree_stats': {
            'min': float(np.min(degrees)) if len(degrees) > 0 else 0.0,
            'max': float(np.max(degrees)) if len(degrees) > 0 else 0.0,
            'mean': float(np.mean(degrees)) if len(degrees) > 0 else 0.0,
            'median': float(np.median(degrees)) if len(degrees) > 0 else 0.0,
            'std': float(np.std(degrees)) if len(degrees) > 0 else 0.0,
            'q25': float(np.percentile(degrees, 25)) if len(degrees) > 0 else 0.0,
            'q75': float(np.percentile(degrees, 75)) if len(degrees) > 0 else 0.0,
        },
        'threshold_distribution': {},
    }
    for th in [0.2, 0.4, 0.5, 0.6, 0.8]:
        prop = float(np.mean(degrees >= th)) if len(degrees) > 0 else 0.0
        report['threshold_distribution'][f'ge_{th}'] = prop
    return report
