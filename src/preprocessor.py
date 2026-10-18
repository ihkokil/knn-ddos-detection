"""
Preprocessor Module
Handles normalization, encoding, missing value handling, and train/test split
for DDoS detection flow features.

Based on: Base paper improved KNN methodology and research extension requirements.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek


class FeaturePreprocessor:
    """
    Comprehensive preprocessor for SDN flow-based DDoS detection features.
    
    Handles:
    - Missing value imputation
    - Categorical feature encoding (protocol_type, service, flag)
    - Numerical feature normalization/standardization
    - Train/test split with optional class balancing
    - Feature selection/retention
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize preprocessor with configuration.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary with keys:
            - 'scaling': 'minmax' or 'standard' or None
            - 'encode_categorical': bool
            - 'handle_missing': str or None (e.g., 'mean', 'median', 'drop')
            - 'balance_classes': bool (use SMOTE or similar)
            - 'test_size': float in (0, 1)
            - 'random_seed': int
        """
        default_config = {
            'scaling': 'minmax',
            'encode_categorical': True,
            'handle_missing': 'mean',
            'balance_classes': False,
            'test_size': 0.2,
            'random_seed': 42
        }
        self.config = {**default_config, **(config or {})}
        self.scaler = None
        self.label_encoders = {}
        self.imputation_values = {}
        self.feature_indices = None
        self.feature_names_out = None
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray = None,
                      feature_names: list = None, df: pd.DataFrame = None) -> tuple:
        """
        Fit preprocessor on training data and transform.
        
        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Feature matrix
        y : np.ndarray, optional
            Labels
        feature_names : list, optional
            Names of features
        df : pd.DataFrame, optional
            DataFrame if X is derived from DataFrame
        
        Returns
        -------
        tuple
            (X_processed, y_processed, feature_names_out)
        """
        # Convert to DataFrame if numpy array with feature names provided
        if isinstance(X, np.ndarray) and feature_names is not None:
            X = pd.DataFrame(X, columns=feature_names)
        elif isinstance(X, np.ndarray) and df is not None:
            # Use column names from DataFrame if available
            pass
        
        # Split before fitting imputers, encoders, or scalers to prevent leakage.
        X_frame = X.copy() if isinstance(X, pd.DataFrame) else X
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_frame, y, test_size=self.config['test_size'],
            random_state=self.config['random_seed'],
            stratify=y if y is not None else None
        )

        X_processed = self._handle_missing(X_train_raw)
        
        # Encode categorical features
        if self.config['encode_categorical'] and isinstance(X_processed, pd.DataFrame):
            X_processed = self._encode_categorical(X_processed)
        
        # Convert back to numpy if DataFrame
        if isinstance(X_processed, pd.DataFrame):
            X_array = X_processed.values
        else:
            X_array = np.array(X_processed, dtype=float)
        
        # Handle any remaining NaN/Inf
        X_array = np.nan_to_num(X_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Feature selection: keep only relevant DDoS features
        X_array = self._select_ddos_features(X_array, feature_names)
        
        # Scale numerical features
        if self.config['scaling'] == 'minmax':
            self.scaler = MinMaxScaler()
            X_scaled = self.scaler.fit_transform(X_array)
        elif self.config['scaling'] == 'standard':
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_array)
        else:
            X_scaled = X_array
        
        # Store output feature names
        if isinstance(X_processed, pd.DataFrame):
            self.feature_names_out = list(X_processed.columns)
        else:
            self.feature_names_out = [f"feature_{i}" for i in range(X_scaled.shape[1])]
        
        X_test_processed = self._handle_missing(X_test_raw, fit=False)
        if self.config['encode_categorical'] and isinstance(X_test_processed, pd.DataFrame):
            X_test_processed = self._encode_categorical(X_test_processed, fit=False)
        X_test_array = np.asarray(X_test_processed, dtype=float)
        X_test_array = np.nan_to_num(X_test_array, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_array = self._select_ddos_features(X_test_array, feature_names, fit=False)
        X_test_scaled = self.scaler.transform(X_test_array) if self.scaler else X_test_array
        
        # Class balancing (SMOTE if enabled)
        if self.config['balance_classes'] and len(np.unique(y_train)) > 1:
            try:
                smote = SMOTE(random_state=self.config['random_seed'], k_neighbors=5)
                X_train, y_train = smote.fit_resample(X_scaled, y_train)
                print(f"  Applied SMOTE: {len(y_train)} training samples after balancing")
            except Exception as e:
                print(f"  SMOTE application warning: {e}")
        
        return X_scaled, X_test_scaled, y_train, y_test
    
    def transform(self, X: np.ndarray, feature_names: list = None) -> np.ndarray:
        """
        Transform data using fitted preprocessor parameters.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix to transform
        feature_names : list, optional
            Feature names corresponding to X columns
        
        Returns
        -------
        np.ndarray
            Transformed feature matrix
        """
        # Handle missing values
        X_processed = self._handle_missing(X, fit=False)
        
        # Encode categorical (using fitted encoders)
        if isinstance(X_processed, pd.DataFrame) and self.label_encoders:
            X_processed = self._encode_categorical(X_processed, fit=False)
        
        X_array = np.array(X_processed, dtype=float) if isinstance(X_processed, pd.DataFrame) else np.array(X, dtype=float)
        X_array = np.nan_to_num(X_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Select same features
        X_array = self._select_ddos_features(X_array, feature_names, fit=False)
        
        # Scale using fitted scaler
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X_array)
        else:
            X_scaled = X_array
        
        return X_scaled
    
    def _handle_missing(self, X, fit=True):
        """Handle missing values in the dataset."""
        if isinstance(X, pd.DataFrame):
            X = X.copy()
            if fit:
                for col in X.columns:
                    if X[col].isnull().any():
                        if self.config['handle_missing'] == 'mean':
                            value = X[col].mean()
                        elif self.config['handle_missing'] == 'median':
                            value = X[col].median()
                        elif self.config['handle_missing'] == 'drop':
                            continue
                        else:
                            value = 0.0
                        self.imputation_values[col] = (
                            0.0 if pd.isna(value) else value
                        )
                        X[col] = X[col].fillna(self.imputation_values[col])
            else:
                for col in X.columns:
                    if X[col].isnull().any():
                        X[col] = X[col].fillna(
                            self.imputation_values.get(col, 0.0)
                        )
            return X
        else:
            # Numpy array: replace NaN with 0
            X = np.array(X, dtype=float)
            X = np.nan_to_num(X, nan=0.0)
            return X
    
    def _encode_categorical(self, df: pd.DataFrame, fit=True) -> pd.DataFrame:
        """Encode categorical features using LabelEncoder."""
        # Common categorical columns in network flow data
        categorical_cols = []
        
        # Identify categorical columns by name or type
        for col in df.columns:
            if df[col].dtype == object or col in ['protocol_type', 'service', 'flag']:
                categorical_cols.append(col)
        
        for col in categorical_cols:
            if fit:
                le = LabelEncoder()
                if fit:
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
            else:
                le = self.label_encoders.get(col)
                if le is None:
                    raise ValueError(f"Categorical encoder for '{col}' is not fitted")
                classes = set(le.classes_)
                df[col] = df[col].astype(str).map(
                    lambda value: value if value in classes else le.classes_[0]
                )
                df[col] = le.transform(df[col])
        
        return df
    
    def _select_ddos_features(self, X: np.ndarray, feature_names: list = None, 
                              fit=True) -> np.ndarray:
        """
        Select and order features relevant to DDoS detection.
        
        Based on the base paper's feature set and my proposed extensions.
        """
        # Default feature order based on typical DDoS detection features:
        # [duration, packet_rate, byte_rate, flow_count, inter_arrival_stats,
        #  entropy_features, protocol_encoded, flag_counts]
        # This can be customized based on feature_importance or selection methods
        
        default_ddos_features = [
            'duration', 'packet_rate', 'byte_rate', 
            'flow_duration_stats', 'inter_arrival_mean',
            'inter_arrival_std', 'src_ip_entropy', 'dst_ip_entropy',
            'port_distribution_entropy', 'protocol_type',
            'flag_SYN', 'flag_ACK', 'flag_FIN', 'flag_RST',
            'degree_of_attack'  # My proposed contribution
        ]
        
        if feature_names is not None and fit:
            # Map available features to default order
            available_set = set(feature_names)
            selected = []
            for df_name in default_ddos_features:
                # Try exact match or substring match
                match = [fn for fn in feature_names if df_name in fn]
                if match:
                    selected.append(match[0])
                elif df_name in available_set:
                    selected.append(df_name)
            
            # Add any remaining features not in the default set
            for fn in feature_names:
                if fn not in selected:
                    selected.append(fn)
            
            # Reorder X to match selected feature order
            feature_set = set(selected)
            col_indices = [feature_names.index(f) for f in selected if f in feature_names]
            if col_indices:
                X = X[:, col_indices]
            self.feature_indices = col_indices
            self.feature_names_out = selected
        elif not fit and self.feature_indices is not None:
            X = X[:, self.feature_indices]
        
        return X
    
    def get_feature_names(self) -> list:
        """Return the output feature names after preprocessing."""
        return self.feature_names_out or []


def quick_preprocess(X: np.ndarray, y: np.ndarray, config: dict = None) -> tuple:
    """
    Quick preprocessing pipeline for rapid experimentation.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Labels
    config : dict, optional
        Preprocessor config
    
    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test) with preprocessing applied
    """
    if config is None:
        config = {
            'scaling': 'minmax',
            'encode_categorical': False,  # X is already numeric
            'handle_missing': 'mean',
            'balance_classes': True,
            'test_size': 0.2,
            'random_seed': 42
        }
    
    preprocessor = FeaturePreprocessor(config)
    X_train, X_test, y_train, y_test = preprocessor.fit_transform(
        X, y, feature_names=[f"f_{i}" for i in range(X.shape[1])]
    )
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Quick test: create synthetic data and preprocess
    print("Testing preprocessor module...")
    np.random.seed(42)
    X = np.random.randn(1000, 15)
    X = np.abs(X)
    y = np.array([0] * 800 + [1] * 200)  # 80% normal, 20% attack
    
    X_train, X_test, y_train, y_test = quick_preprocess(X, y)
    print(f"✓ Preprocessing complete")
    print(f"  Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"  Train class distribution: {dict(Counter(y_train))}")
    print(f"  Test class distribution: {dict(Counter(y_test))}")