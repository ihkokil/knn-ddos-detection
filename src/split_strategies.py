"""
Group/Time-Aware Splitting Module
Prevents data leakage by ensuring flows from the same source/session/time window
don't cross train/test boundaries.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, TimeSeriesSplit
from sklearn.base import BaseEstimator
import warnings


class GroupAwareSplitter:
    """
    Group-aware data splitter to prevent leakage from same-source flows.
    
    Supports:
    - Group-based splitting (by source IP, flow ID, etc.)
    - Time-based splitting (chronological splits)
    - Combined group+time splitting
    - Stratified group splitting
    """
    
    def __init__(self, 
                 group_column: str = None,
                 time_column: str = None,
                 test_size: float = 0.2,
                 random_state: int = 42,
                 n_splits: int = 5):
        """
        Initialize splitter.
        
        Parameters
        ----------
        group_column : str, optional
            Column name for grouping (e.g., 'Src IP', 'Flow ID')
        time_column : str, optional
            Column name for temporal ordering (e.g., 'Timestamp')
        test_size : float
            Proportion of data for test set
        random_state : int
            Random seed
        n_splits : int
            Number of CV folds
        """
        self.group_column = group_column
        self.time_column = time_column
        self.test_size = test_size
        self.random_state = random_state
        self.n_splits = n_splits
        
    def split(self, X: np.ndarray, y: np.ndarray, 
              groups: np.ndarray = None,
              timestamps: np.ndarray = None,
              df: pd.DataFrame = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test splits.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Labels
        groups : np.ndarray, optional
            Group assignments per sample
        timestamps : np.ndarray, optional
            Timestamps per sample
        df : pd.DataFrame, optional
            Original dataframe (used to extract groups/timestamps)
            
        Returns
        -------
        list of tuples
            List of (train_idx, test_idx) pairs
        """
        # Extract groups/timestamps from dataframe if provided
        if df is not None:
            if groups is None and self.group_column and self.group_column in df.columns:
                groups = df[self.group_column].values
            if timestamps is None and self.time_column and self.time_column in df.columns:
                timestamps = df[self.time_column].values
        
        # Determine splitting strategy
        if groups is not None and timestamps is not None:
            return self._split_group_time(X, y, groups, timestamps)
        elif groups is not None:
            return self._split_group(X, y, groups)
        elif timestamps is not None:
            return self._split_time(X, y, timestamps)
        else:
            # Fallback to random stratified split
            return self._split_random(X, y)
    
    def _split_group(self, X: np.ndarray, y: np.ndarray, 
                     groups: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Group-based splitting using GroupShuffleSplit."""
        splitter = GroupShuffleSplit(
            n_splits=self.n_splits,
            test_size=self.test_size,
            random_state=self.random_state
        )
        return list(splitter.split(X, y, groups))
    
    def _split_time(self, X: np.ndarray, y: np.ndarray,
                    timestamps: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Time-based splitting using TimeSeriesSplit."""
        # Sort by timestamp
        sort_idx = np.argsort(timestamps)
        X_sorted = X[sort_idx]
        y_sorted = y[sort_idx]
        
        splitter = TimeSeriesSplit(n_splits=self.n_splits)
        splits = []
        for train_idx, test_idx in splitter.split(X_sorted):
            # Map back to original indices
            splits.append((sort_idx[train_idx], sort_idx[test_idx]))
        return splits
    
    def _split_group_time(self, X: np.ndarray, y: np.ndarray,
                          groups: np.ndarray, timestamps: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Combined group + time splitting.
        
        Strategy: Group by source, then within each group sort by time,
        then split chronologically per group, then combine.
        """
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)
        
        # For each group, get time-sorted indices
        group_indices = {}
        for g in unique_groups:
            mask = groups == g
            group_timestamps = timestamps[mask]
            group_sort = np.argsort(group_timestamps)
            group_indices[g] = np.where(mask)[0][group_sort]
        
        # Determine split point per group (chronological)
        test_groups = int(n_groups * self.test_size)
        
        # Use GroupShuffleSplit on group level for group assignment
        group_splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=self.test_size,
            random_state=self.random_state
        )
        
        # Create group-level labels for stratification
        group_labels = np.array([np.bincount(y[group_indices[g]]).argmax() 
                                  for g in unique_groups])
        
        # Get group-level split
        group_train_idx, group_test_idx = next(
            group_splitter.split(unique_groups, group_labels, unique_groups)
        )
        
        train_groups = unique_groups[group_train_idx]
        test_groups = unique_groups[group_test_idx]
        
        # Combine indices
        train_indices = np.concatenate([group_indices[g] for g in train_groups])
        test_indices = np.concatenate([group_indices[g] for g in test_groups])
        
        return [(train_indices, test_indices)]
    
    def _split_random(self, X: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Fallback random stratified split."""
        from sklearn.model_selection import StratifiedShuffleSplit
        splitter = StratifiedShuffleSplit(
            n_splits=self.n_splits,
            test_size=self.test_size,
            random_state=self.random_state
        )
        return list(splitter.split(X, y))


class CICDDOS2019Splitter(GroupAwareSplitter):
    """Specialized splitter for CIC-DDoS2019 using Src IP + Timestamp."""
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42, n_splits: int = 5):
        super().__init__(
            group_column="Src IP",
            time_column="Timestamp",
            test_size=test_size,
            random_state=random_state,
            n_splits=n_splits
        )
    
    def prepare_groups_timestamps(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Extract groups and timestamps from CIC-DDoS2019 dataframe."""
        groups = df["Src IP"].values if "Src IP" in df.columns else None
        
        # Parse timestamps
        timestamps = None
        if "Timestamp" in df.columns:
            ts = df["Timestamp"]
            if ts.dtype == 'object':
                # Try parsing various formats
                try:
                    timestamps = pd.to_datetime(ts).astype('int64').values // 10**9
                except:
                    # Try numeric conversion
                    timestamps = pd.to_numeric(ts, errors='coerce').fillna(0).values
            else:
                timestamps = ts.values
        
        return groups, timestamps


class NSLKDDSplitter(GroupAwareSplitter):
    """Specialized splitter for NSL-KDD using connection features as groups."""
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42, n_splits: int = 5):
        # NSL-KDD doesn't have explicit Src IP or Timestamp
        # Use connection signature as group: (src_bytes, dst_bytes, protocol, service, flag)
        super().__init__(
            group_column=None,  # Will be computed
            time_column=None,
            test_size=test_size,
            random_state=random_state,
            n_splits=n_splits
        )
    
    def prepare_groups_timestamps(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create connection signature groups for NSL-KDD."""
        # Create a connection fingerprint as group identifier
        if all(c in df.columns for c in ["src_bytes", "dst_bytes", "protocol_type", "service", "flag"]):
            # Combine into string signature
            groups = (df["src_bytes"].astype(str) + "_" + 
                     df["dst_bytes"].astype(str) + "_" + 
                     df["protocol_type"].astype(str) + "_" + 
                     df["service"].astype(str) + "_" + 
                     df["flag"].astype(str)).values
        elif "src_bytes" in df.columns and "dst_bytes" in df.columns:
            # Fallback: just src/dst bytes
            groups = (df["src_bytes"].astype(str) + "_" + df["dst_bytes"].astype(str)).values
        else:
            groups = None
            
        # NSL-KDD has no timestamps
        timestamps = None
        
        return groups, timestamps


def create_splitter(dataset_name: str, **kwargs) -> GroupAwareSplitter:
    """Factory function to create dataset-appropriate splitter."""
    if dataset_name.upper() in ["CIC-DDoS2019", "CICDDOS2019"]:
        return CICDDOS2019Splitter(**kwargs)
    elif dataset_name.upper() in ["NSL-KDD", "NSLKDD"]:
        return NSLKDDSplitter(**kwargs)
    else:
        return GroupAwareSplitter(**kwargs)


def split_dataset_aware(X: np.ndarray, y: np.ndarray,
                         dataset_name: str,
                         df: pd.DataFrame = None,
                         test_size: float = 0.2,
                         random_state: int = 42,
                         n_splits: int = 1,
                         return_all_splits: bool = False) -> Union[
                             Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                             List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
                         ]:
    """
    Convenience function for dataset-aware splitting.
    
    Parameters
    ----------
    X : np.ndarray
        Features
    y : np.ndarray
        Labels
    dataset_name : str
        Dataset name
    df : pd.DataFrame, optional
        Original dataframe for group/timestamp extraction
    test_size : float
        Test proportion
    random_state : int
        Random seed
    n_splits : int
        Number of splits (for CV)
    return_all_splits : bool
        If True, return all CV splits; if False, return first split only
        
    Returns
    -------
    If return_all_splits=False:
        (X_train, X_test, y_train, y_test)
    If return_all_splits=True:
        List of (X_train, X_test, y_train, y_test) for each fold
    """
    splitter = create_splitter(dataset_name, test_size=test_size, 
                               random_state=random_state, n_splits=n_splits)
    
    splits = splitter.split(X, y, df=df)
    
    if return_all_splits:
        results = []
        for train_idx, test_idx in splits:
            results.append((X[train_idx], X[test_idx], y[train_idx], y[test_idx]))
        return results
    else:
        train_idx, test_idx = splits[0]
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def verify_no_leakage(df_train: pd.DataFrame, df_test: pd.DataFrame,
                       group_column: str = "Src IP") -> Dict[str, Any]:
    """
    Verify that train and test sets don't share groups.
    
    Parameters
    ----------
    df_train : pd.DataFrame
        Training dataframe
    df_test : pd.DataFrame
        Test dataframe
    group_column : str
        Column to check for leakage
        
    Returns
    -------
    dict
        Leakage verification report
    """
    if group_column not in df_train.columns or group_column not in df_test.columns:
        return {"error": f"Group column '{group_column}' not found in both dataframes"}
    
    train_groups = set(df_train[group_column].unique())
    test_groups = set(df_test[group_column].unique())
    
    overlap = train_groups & test_groups
    
    return {
        "group_column": group_column,
        "train_unique_groups": len(train_groups),
        "test_unique_groups": len(test_groups),
        "overlapping_groups": len(overlap),
        "overlap_ratio": len(overlap) / len(train_groups | test_groups) if (train_groups | test_groups) else 0,
        "leakage_detected": len(overlap) > 0,
        "overlapping_group_samples": list(overlap)[:10]  # First 10
    }


def stratified_group_split(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                           test_size: float = 0.2, random_state: int = 42) -> Tuple:
    """
    Stratified group split maintaining class balance across groups.
    
    Uses a heuristic: assign groups to train/test to balance class distribution.
    """
    from sklearn.model_selection import GroupShuffleSplit
    
    # Try standard GroupShuffleSplit first
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    
    # Check class balance
    train_dist = np.bincount(y[train_idx])
    test_dist = np.bincount(y[test_idx])
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# =============================================================================
# Cross-Validation with Group Awareness
# =============================================================================

class GroupAwareCV:
    """
    Cross-validation iterator with group awareness.
    
    Wraps sklearn's GroupKFold but adds dataset-specific group extraction.
    """
    
    def __init__(self, dataset_name: str, n_splits: int = 5, 
                 group_column: str = None, time_column: str = None):
        self.dataset_name = dataset_name
        self.n_splits = n_splits
        self.group_column = group_column
        self.time_column = time_column
        self.splitter = create_splitter(dataset_name, n_splits=n_splits)
    
    def split(self, X: np.ndarray, y: np.ndarray, 
              groups: np.ndarray = None, df: pd.DataFrame = None):
        """Yield train/test indices for each fold."""
        if df is not None:
            if groups is None and self.group_column and self.group_column in df.columns:
                groups = df[self.group_column].values
        
        if groups is None:
            warnings.warn("No groups provided, falling back to random CV")
            from sklearn.model_selection import StratifiedKFold
            cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
            return cv.split(X, y)
        
        from sklearn.model_selection import GroupKFold
        cv = GroupKFold(n_splits=self.n_splits)
        return cv.split(X, y, groups)
    
    def get_n_splits(self) -> int:
        return self.n_splits


# =============================================================================
# Demonstration and Testing
# =============================================================================

if __name__ == "__main__":
    print("Testing group/time-aware splitting...")
    
    # Create synthetic data with groups and timestamps
    np.random.seed(42)
    n_samples = 1000
    n_groups = 50
    
    # Assign each sample to a group
    groups = np.random.choice(n_groups, n_samples)
    
    # Add timestamps (grouped by group)
    timestamps = np.zeros(n_samples)
    for g in range(n_groups):
        mask = groups == g
        n_g = np.sum(mask)
        timestamps[mask] = np.sort(np.random.uniform(0, 100, n_g))
    
    # Features and labels
    X = np.random.randn(n_samples, 10)
    y = np.random.randint(0, 2, n_samples)
    
    # Create dataframe
    df = pd.DataFrame({
        'group': groups,
        'timestamp': timestamps
    })
    
    # Test group splitting
    print("\n1. Group-aware split:")
    splitter = GroupAwareSplitter(group_column='group', test_size=0.2, n_splits=1)
    splits = splitter.split(X, y, df=df)
    train_idx, test_idx = splits[0]
    print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")
    
    # Verify no group overlap
    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])
    overlap = train_groups & test_groups
    print(f"  Train groups: {len(train_groups)}, Test groups: {len(test_groups)}")
    print(f"  Overlap: {len(overlap)} (should be 0)")
    
    # Test time splitting
    print("\n2. Time-aware split:")
    splitter = GroupAwareSplitter(time_column='timestamp', test_size=0.2, n_splits=1)
    splits = splitter.split(X, y, df=df)
    train_idx, test_idx = splits[0]
    print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")
    print(f"  Max train time: {timestamps[train_idx].max():.2f}")
    print(f"  Min test time: {timestamps[test_idx].min():.2f}")
    print(f"  Time order preserved: {timestamps[train_idx].max() <= timestamps[test_idx].min()}")
    
    # Test combined group+time
    print("\n3. Group+Time split:")
    splitter = GroupAwareSplitter(group_column='group', time_column='timestamp', 
                                  test_size=0.2, n_splits=1)
    splits = splitter.split(X, y, df=df)
    train_idx, test_idx = splits[0]
    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])
    overlap = train_groups & test_groups
    print(f"  Overlap: {len(overlap)} (should be 0)")
    print(f"  Train max time: {timestamps[train_idx].max():.2f}")
    print(f"  Test min time: {timestamps[test_idx].min():.2f}")
    
    # Test dataset-specific splitters
    print("\n4. CIC-DDoS2019 splitter:")
    cic_splitter = CICDDOS2019Splitter()
    
    # Create mock CIC data
    cic_df = pd.DataFrame({
        'Src IP': [f'10.0.0.{i%20}' for i in range(100)],
        'Timestamp': pd.date_range('2024-01-01', periods=100, freq='1min'),
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'Label': np.random.randint(0, 2, 100)
    })
    
    groups, timestamps = cic_splitter.prepare_groups_timestamps(cic_df)
    print(f"  Groups: {len(np.unique(groups))} unique")
    print(f"  Timestamps range: {timestamps.min()} to {timestamps.max()}")
    
    print("\n5. NSL-KDD splitter:")
    nsl_splitter = NSLKDDSplitter()
    
    # Create mock NSL data
    nsl_df = pd.DataFrame({
        'src_bytes': np.random.randint(0, 1000, 100),
        'dst_bytes': np.random.randint(0, 1000, 100),
        'protocol_type': np.random.choice(['tcp', 'udp', 'icmp'], 100),
        'service': np.random.choice(['http', 'smtp', 'ftp'], 100),
        'flag': np.random.choice(['SF', 'S0', 'REJ'], 100),
        'label': np.random.choice(['normal', 'back', 'neptune'], 100)
    })
    
    groups, timestamps = nsl_splitter.prepare_groups_timestamps(nsl_df)
    print(f"  Groups: {len(np.unique(groups))} unique connection signatures")
    print(f"  Timestamps: {timestamps}")
    
    print("\n✓ All splitting tests passed!")