"""
Dataset-Specific Feature Schema Module
Explicit feature schemas for CIC-DDoS2019 and NSL-KDD to replace fallback-heavy
feature engineering with deterministic, documented feature extraction.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


# =============================================================================
# CIC-DDoS2019 Feature Schema
# =============================================================================

CICDDOS2019_RAW_FEATURES = [
    # Flow identifiers (non-features, used for grouping)
    "Flow ID",
    "Src IP",
    "Src Port", 
    "Dst IP",
    "Dst Port",
    "Protocol",
    "Timestamp",
    
    # Flow statistics
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    
    # Fwd packet length stats
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    
    # Bwd packet length stats
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    
    # Flow rates
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    
    # Fwd IAT stats
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    
    # Bwd IAT stats
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    
    # Flag counts
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    
    # Packet length stats
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    
    # TCP flag counts
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
    
    # Window sizes
    "Down/Up Ratio",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Fwd Header Length.1",  # Duplicate column name in some versions
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
    
    # Subflow stats
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    
    # Init window bytes
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    
    # Active/Idle stats
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
    
    # Label
    "Label"
]

# Feature groups for CIC-DDoS2019
CICDDOS2019_FEATURE_GROUPS = {
    "flow_id": ["Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol", "Timestamp"],
    "basic_flow": ["Flow Duration", "Total Fwd Packets", "Total Backward Packets"],
    "packet_length_fwd": ["Fwd Packet Length Max", "Fwd Packet Length Min", 
                           "Fwd Packet Length Mean", "Fwd Packet Length Std"],
    "packet_length_bwd": ["Bwd Packet Length Max", "Bwd Packet Length Min",
                           "Bwd Packet Length Mean", "Bwd Packet Length Std"],
    "flow_rates": ["Flow Bytes/s", "Flow Packets/s"],
    "flow_iat": ["Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min"],
    "fwd_iat": ["Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min"],
    "bwd_iat": ["Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min"],
    "flags": ["Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
              "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
              "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count"],
    "header_lengths": ["Fwd Header Length", "Bwd Header Length"],
    "packet_rates": ["Fwd Packets/s", "Bwd Packets/s"],
    "packet_length_overall": ["Min Packet Length", "Max Packet Length", 
                               "Packet Length Mean", "Packet Length Std", "Packet Length Variance"],
    "window_size": ["Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size",
                    "Avg Bwd Segment Size"],
    "bulk_stats": ["Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
                   "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate"],
    "subflow": ["Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes"],
    "init_window": ["Init_Win_bytes_forward", "Init_Win_bytes_backward"],
    "active_idle": ["act_data_pkt_fwd", "min_seg_size_forward",
                    "Active Mean", "Active Std", "Active Max", "Active Min",
                    "Idle Mean", "Idle Std", "Idle Max", "Idle Min"],
    "label": ["Label"]
}

# CIC-DDoS2019 Attack Type Mapping
CICDDOS2019_ATTACK_TYPES = {
    "Benign": 0,
    "UDP": 1,
    "TCP-SYN": 2,  # Syn flood
    "TCP-ACK": 3,  # Ack flood
    "ICMP": 4,
    "HTTP": 5,     # HTTP flood / Slowloris
    "Slowloris": 6
}

# Label patterns in CIC-DDoS2019
CICDDOS2019_LABEL_PATTERNS = {
    "Benign": ["benign", "benign"],
    "UDP": ["udp", "udp-flood", "udpflood"],
    "TCP-SYN": ["syn", "tcp-syn", "synflood", "syn-flood"],
    "TCP-ACK": ["ack", "tcp-ack", "ackflood", "ack-flood"],
    "ICMP": ["icmp", "icmp-flood", "icmpflood"],
    "HTTP": ["http", "http-flood", "httpflood"],
    "Slowloris": ["slowloris", "slowloris-attack"]
}


# =============================================================================
# NSL-KDD Feature Schema
# =============================================================================

NSL_KDD_RAW_FEATURES = [
    # Basic features (9)
    "duration",
    "protocol_type",      # tcp, udp, icmp
    "service",            # http, smtp, finger, etc.
    "flag",               # SF, S0, REJ, RSTR, etc.
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    
    # Content features (13)
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creates",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    
    # Traffic features (9)
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    
    # Host-based features (10)
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    
    # Labels
    "label",              # normal or attack type
    "difficulty"          # difficulty level (ignored for classification)
]

# NSL-KDD Feature Groups
NSL_KDD_FEATURE_GROUPS = {
    "basic": ["duration", "protocol_type", "service", "flag", "src_bytes", 
              "dst_bytes", "land", "wrong_fragment", "urgent"],
    "content": ["hot", "num_failed_logins", "logged_in", "num_compromised",
                "root_shell", "su_attempted", "num_root", "num_file_creates",
                "num_shells", "num_access_files", "num_outbound_cmds",
                "is_host_login", "is_guest_login"],
    "traffic": ["count", "srv_count", "serror_rate", "srv_serror_rate",
                "rerror_rate", "srv_rerror_rate", "same_srv_rate",
                "diff_srv_rate", "srv_diff_host_rate"],
    "host": ["dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
             "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
             "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
             "dst_host_srv_serror_rate", "dst_host_rerror_rate",
             "dst_host_srv_rerror_rate"],
    "labels": ["label", "difficulty"]
}

# NSL-KDD Attack Categories (4 main categories + normal)
NSL_KDD_ATTACK_CATEGORIES = {
    "normal": 0,
    "DoS": 1,       # Denial of Service: back, land, neptune, pod, smurf, teardrop, apache2, udpstorm, processtable, mailbomb
    "Probe": 2,     # Surveillance/Probing: ipsweep, nmap, portsweep, satan, mscan, saint
    "R2L": 3,       # Remote to Local: ftp_write, guess_passwd, imap, multihop, phf, spy, warezclient, warezmaster, xlock, xsnoop, snmpguess, snmpgetattack, httptunnel, sendmail, named
    "U2R": 4        # User to Root: buffer_overflow, loadmodule, perl, rootkit, ps, sqlattack, xterm, worm
}

# Detailed attack type mapping for NSL-KDD
NSL_KDD_ATTACK_TYPES = {
    # Normal
    "normal": 0,
    # DoS attacks
    "back": 1, "land": 1, "neptune": 1, "pod": 1, "smurf": 1, "teardrop": 1,
    "apache2": 1, "udpstorm": 1, "processtable": 1, "mailbomb": 1,
    # Probe attacks
    "ipsweep": 2, "nmap": 2, "portsweep": 2, "satan": 2, "mscan": 2, "saint": 2,
    # R2L attacks
    "ftp_write": 3, "guess_passwd": 3, "imap": 3, "multihop": 3, "phf": 3,
    "spy": 3, "warezclient": 3, "warezmaster": 3, "xlock": 3, "xsnoop": 3,
    "snmpguess": 3, "snmpgetattack": 3, "httptunnel": 3, "sendmail": 3, "named": 3,
    # U2R attacks
    "buffer_overflow": 4, "loadmodule": 4, "perl": 4, "rootkit": 4,
    "ps": 4, "sqlattack": 4, "xterm": 4, "worm": 4
}

# Categorical feature encodings for NSL-KDD
NSL_KDD_CATEGORICAL_ENCODINGS = {
    "protocol_type": {"tcp": 1, "udp": 2, "icmp": 3},
    "service": {
        "http": 1, "smtp": 2, "finger": 3, "domain_u": 4, "auth": 5, "telnet": 6,
        "ftp": 7, "eco_i": 8, "ntp_u": 9, "ecr_i": 10, "other": 11, "private": 12,
        "pop_3": 13, "ftp_data": 14, "rje": 15, "time": 16, "mtp": 17, "link": 18,
        "remote_job": 19, "gopher": 20, "ssh": 21, "name": 22, "whois": 23,
        "domain": 24, "login": 25, "imap4": 26, "daytime": 27, "ctf": 28,
        "nntp": 29, "shell": 30, "IRC": 31, "nnsp": 32, "http_443": 33,
        "exec": 34, "printer": 35, "efs": 36, "courier": 37, "uucp": 38,
        "klogin": 39, "kshell": 40, "echo": 41, "discard": 42, "systat": 43,
        "supdup": 44, "iso_tsap": 45, "hostnames": 46, "csnet_ns": 47,
        "uucp_path": 48, "netbios_ns": 49, "netbios_dgm": 50, "netbios_ssn": 51,
        "sql_net": 52, "vmnet": 53, "bgp": 54, "Z39_50": 55, "ldap": 56
    },
    "flag": {"SF": 1, "S0": 2, "REJ": 3, "RSTR": 4, "RSTO": 5, "SH": 6, 
             "S1": 7, "S2": 8, "S3": 9, "OTH": 10}
}


# =============================================================================
# Schema Classes
# =============================================================================

class DatasetFeatureSchema:
    """Base class for dataset-specific feature schemas."""
    
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.raw_features = []
        self.feature_groups = {}
        self.categorical_features = []
        self.numerical_features = []
        self.label_column = ""
        self.attack_type_mapping = {}
        self.label_patterns = {}
    
    def get_feature_columns(self, include_id: bool = False) -> List[str]:
        """Get list of feature columns for model input."""
        raise NotImplementedError
    
    def get_categorical_columns(self) -> List[str]:
        """Get list of categorical columns requiring encoding."""
        return self.categorical_features
    
    def get_numerical_columns(self) -> List[str]:
        """Get list of numerical columns."""
        return self.numerical_features
    
    def map_attack_type(self, label: str) -> int:
        """Map raw label to attack type index."""
        raise NotImplementedError
    
    def get_attack_type_names(self) -> List[str]:
        """Get list of attack type names."""
        return list(self.attack_type_mapping.keys())


class CICDDOS2019Schema(DatasetFeatureSchema):
    """Explicit feature schema for CIC-DDoS2019 dataset."""
    
    def __init__(self):
        super().__init__("CIC-DDoS2019")
        self.raw_features = CICDDOS2019_RAW_FEATURES
        self.feature_groups = CICDDOS2019_FEATURE_GROUPS
        self.label_column = "Label"
        self.attack_type_mapping = CICDDOS2019_ATTACK_TYPES
        self.label_patterns = CICDDOS2019_LABEL_PATTERNS
        
        # Categorical features that need encoding
        self.categorical_features = ["Protocol"]
        
        # Numerical features (all flow statistics)
        id_cols = set(self.feature_groups["flow_id"])
        label_cols = set(self.feature_groups["label"])
        self.numerical_features = [f for f in self.raw_features 
                                    if f not in id_cols and f not in label_cols]
    
    def get_feature_columns(self, include_id: bool = False) -> List[str]:
        """Get feature columns for model input."""
        if include_id:
            return self.numerical_features + self.categorical_features
        return self.numerical_features
    
    def get_group_columns(self, group: str) -> List[str]:
        """Get columns for a specific feature group."""
        return self.feature_groups.get(group, [])
    
    def map_attack_type(self, label: str) -> int:
        """Map CIC-DDoS2019 label to attack type index."""
        label_lower = str(label).lower().strip()
        
        # Check each attack type pattern
        for attack_type, patterns in self.label_patterns.items():
            for pattern in patterns:
                if pattern.lower() in label_lower:
                    return self.attack_type_mapping.get(attack_type, 0)
        
        # Default to benign
        return 0
    
    def get_binary_label(self, label: str) -> int:
        """Map to binary: 0=benign, 1=attack."""
        return 0 if self.map_attack_type(label) == 0 else 1
    
    def get_flow_id_columns(self) -> List[str]:
        """Get flow identifier columns for grouping/splitting."""
        return self.feature_groups["flow_id"]
    
    def get_timestamp_column(self) -> str:
        """Get timestamp column for time-aware splitting."""
        return "Timestamp"


class NSLKDDSchema(DatasetFeatureSchema):
    """Explicit feature schema for NSL-KDD dataset."""
    
    def __init__(self):
        super().__init__("NSL-KDD")
        self.raw_features = NSL_KDD_RAW_FEATURES
        self.feature_groups = NSL_KDD_FEATURE_GROUPS
        self.label_column = "label"
        self.attack_type_mapping = NSL_KDD_ATTACK_TYPES
        self.attack_categories = NSL_KDD_ATTACK_CATEGORIES
        self.categorical_encodings = NSL_KDD_CATEGORICAL_ENCODINGS
        
        # Categorical features
        self.categorical_features = ["protocol_type", "service", "flag"]
        
        # Numerical features
        label_cols = set(self.feature_groups["labels"])
        cat_cols = set(self.categorical_features)
        self.numerical_features = [f for f in self.raw_features 
                                    if f not in label_cols and f not in cat_cols]
    
    def get_feature_columns(self, include_id: bool = False) -> List[str]:
        """Get feature columns for model input (includes encoded categoricals)."""
        return self.numerical_features + self.categorical_features
    
    def get_group_columns(self, group: str) -> List[str]:
        """Get columns for a specific feature group."""
        return self.feature_groups.get(group, [])
    
    def map_attack_type(self, label: str) -> int:
        """Map NSL-KDD label to detailed attack type index."""
        label_lower = str(label).lower().strip().replace(".", "")
        return self.attack_type_mapping.get(label_lower, 0)
    
    def map_attack_category(self, label: str) -> int:
        """Map NSL-KDD label to attack category index (0=normal, 1=DoS, 2=Probe, 3=R2L, 4=U2R)."""
        detailed_idx = self.map_attack_type(label)
        
        # Reverse lookup: detailed index -> category index
        # attack_categories = {"normal": 0, "DoS": 1, "Probe": 2, "R2L": 3, "U2R": 4}
        category_map = {v: k for k, v in self.attack_categories.items()}
        category_name = category_map.get(detailed_idx, "normal")
        return self.attack_categories.get(category_name, 0)
    
    def get_binary_label(self, label: str) -> int:
        """Map to binary: 0=normal, 1=attack."""
        return 0 if self.map_attack_type(label) == 0 else 1
    
    def get_categorical_encodings(self) -> Dict[str, Dict[str, int]]:
        """Get predefined categorical encodings."""
        return self.categorical_encodings
    
    def get_connection_key_columns(self) -> List[str]:
        """Get columns that define a unique connection for grouping."""
        return ["src_bytes", "dst_bytes", "protocol_type", "service", "flag"]


# =============================================================================
# InSDN Feature Schema
# =============================================================================

INSDN_RAW_FEATURES = [
    'Protocol', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts',
    'Fwd Pkt Len Max', 'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std',
    'Bwd Pkt Len Max', 'Bwd Pkt Len Min', 'Bwd Pkt Len Mean', 'Bwd Pkt Len Std',
    'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Tot', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
    'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
    'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
    'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s', 'Bwd Pkts/s',
    'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std', 'Pkt Len Var',
    'FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt',
    'CWE Flag Count', 'ECE Flag Cnt', 'Down/Up Ratio', 'Pkt Size Avg',
    'Fwd Seg Size Avg', 'Bwd Seg Size Avg', 'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg',
    'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg',
    'Subflow Fwd Pkts', 'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts',
    'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Act Data Pkts', 'Fwd Seg Size Min',
    'Active Mean', 'Active Std', 'Active Max', 'Active Min',
    'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min', 'Label'
]


class InSDNSchema(DatasetFeatureSchema):
    """Schema for InSDN dataset (dedicated SDN intrusion dataset)."""
    
    def __init__(self):
        super().__init__("InSDN")
        self.raw_features = INSDN_RAW_FEATURES
        self.label_column = "Label"
        self.categorical_features = []
        self.numerical_features = [f for f in INSDN_RAW_FEATURES if f != "Label"]
        self.attack_type_mapping = {"Normal": 0, "DDoS": 1}
        self.feature_groups = {
            "flow_stats": ['Flow Duration', 'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min'],
            "packet_stats": ['Tot Fwd Pkts', 'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std', 'Pkt Len Var', 'Pkt Size Avg'],
            "tcp_flags": ['FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt', 'ECE Flag Cnt', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags'],
            "window_segment": ['Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Seg Size Avg', 'Bwd Seg Size Avg', 'Fwd Seg Size Min'],
            "active_idle": ['Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min']
        }
        
    def get_feature_columns(self, include_id: bool = False) -> List[str]:
        return [f for f in self.raw_features if f != self.label_column]
        
    def map_attack_type(self, label: Any) -> int:
        if isinstance(label, (int, np.integer)):
            return int(label)
        label_str = str(label).strip()
        return 0 if label_str.lower() == 'normal' else 1
        
    def get_binary_label(self, label: Any) -> int:
        return self.map_attack_type(label)


# =============================================================================
# Schema Factory
# =============================================================================

def get_dataset_schema(dataset_name: str) -> DatasetFeatureSchema:
    """Factory function to get dataset schema."""
    if dataset_name.upper() in ["CIC-DDoS2019", "CICDDOS2019", "CIC"]:
        return CICDDOS2019Schema()
    elif dataset_name.upper() in ["NSL-KDD", "NSLKDD", "NSL"]:
        return NSLKDDSchema()
    elif dataset_name.upper() in ["INSDN", "IN-SDN"]:
        return InSDNSchema()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: CIC-DDoS2019, NSL-KDD, InSDN")


# =============================================================================
# Feature Engineering with Explicit Schema
# =============================================================================

def engineer_features_cicddos2019(df: pd.DataFrame, 
                                   schema: CICDDOS2019Schema = None,
                                   include_entropy: bool = True,
                                   include_degree: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """
    Engineer features for CIC-DDoS2019 using explicit schema.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw CIC-DDoS2019 dataframe
    schema : CICDDOS2019Schema
        Feature schema (created if not provided)
    include_entropy : bool
        Whether to compute entropy features
    include_degree : bool
        Whether to compute degree-of-attack
        
    Returns
    -------
    tuple
        (X, y, feature_names, degree_values)
    """
    if schema is None:
        schema = CICDDOS2019Schema()
    
    df = df.copy()
    
    # Ensure Label column exists
    if "Label" not in df.columns:
        raise ValueError("CIC-DDoS2019 data must have 'Label' column")
    
    # Map labels
    y_binary = df["Label"].apply(schema.get_binary_label).values
    y_attack_type = df["Label"].apply(schema.map_attack_type).values
    
    # Get feature columns (exclude ID and label)
    feature_cols = schema.get_feature_columns(include_id=False)
    
    # Verify all features exist
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"Warning: Missing features: {missing}")
        feature_cols = [c for c in feature_cols if c in df.columns]
    
    X = df[feature_cols].copy()
    
    # Handle categorical features
    if "Protocol" in X.columns:
        # Protocol is typically numeric (6=TCP, 17=UDP, 1=ICMP) but ensure numeric
        X["Protocol"] = pd.to_numeric(X["Protocol"], errors="coerce").fillna(0)
    
    # Convert all to numeric
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    
    feature_names = list(X.columns)
    X_array = X.values.astype(float)
    
    # Compute degree of attack if requested
    degree_values = np.zeros(len(df))
    if include_degree:
        from src.degree_of_attack import DegreeOfAttackComputer
        deg_computer = DegreeOfAttackComputer(method="proposed")
        degree_values = deg_computer.compute_degree(X_array, feature_names=feature_names)
    
    return X_array, y_binary, feature_names, degree_values, y_attack_type


def engineer_features_nsl_kdd(df: pd.DataFrame,
                               schema: NSLKDDSchema = None,
                               include_entropy: bool = True,
                               include_degree: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """
    Engineer features for NSL-KDD using explicit schema.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw NSL-KDD dataframe
    schema : NSLKDDSchema
        Feature schema (created if not provided)
    include_entropy : bool
        Whether to compute entropy features (placeholder - not applicable to NSL-KDD)
    include_degree : bool
        Whether to compute degree-of-attack
        
    Returns
    -------
    tuple
        (X, y, feature_names, degree_values, y_attack_category)
    """
    if schema is None:
        schema = NSLKDDSchema()
    
    df = df.copy()
    
    # Ensure label column exists
    if "label" not in df.columns:
        raise ValueError("NSL-KDD data must have 'label' column")
    
    # Map labels
    y_binary = df["label"].apply(schema.get_binary_label).values
    y_attack_category = df["label"].apply(schema.map_attack_category).values
    y_attack_type = df["label"].apply(schema.map_attack_type).values
    
    # Get feature columns
    feature_cols = schema.get_feature_columns(include_id=False)
    
    # Verify all features exist
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"Warning: Missing features: {missing}")
        feature_cols = [c for c in feature_cols if c in df.columns]
    
    X = df[feature_cols].copy()
    
    # Encode categorical features using predefined mappings
    encodings = schema.get_categorical_encodings()
    for col, mapping in encodings.items():
        if col in X.columns:
            X[col] = X[col].map(mapping).fillna(0)
    
    # Convert all to numeric
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    
    feature_names = list(X.columns)
    X_array = X.values.astype(float)
    
    # Compute degree of attack if requested
    degree_values = np.zeros(len(df))
    if include_degree:
        from src.degree_of_attack import DegreeOfAttackComputer
        deg_computer = DegreeOfAttackComputer(method="base_paper")  # NSL-KDD uses base paper method
        degree_values = deg_computer.compute_degree(X_array, feature_names=feature_names)
    
    return X_array, y_binary, feature_names, degree_values, y_attack_category, y_attack_type


def engineer_features_insdn(df: pd.DataFrame,
                            schema: InSDNSchema = None,
                            include_entropy: bool = True,
                            include_degree: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, np.ndarray]:
    """
    Engineer features for InSDN using explicit schema.
    """
    if schema is None:
        schema = InSDNSchema()
        
    df = df.copy()
    lbl_col = "Label" if "Label" in df.columns else ("label" if "label" in df.columns else None)
    if lbl_col is None:
        raise ValueError("InSDN data must have 'Label' column")
        
    y_binary = df[lbl_col].apply(schema.get_binary_label).values
    y_attack_type = y_binary.copy()
    
    feature_cols = [c for c in df.columns if c not in ["Label", "label", "Target", "target"]]
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).replace([np.inf, -np.inf], 0.0)
    feature_names = list(X.columns)
    X_array = X.values.astype(float)
    
    degree_values = np.zeros(len(df))
    if include_degree:
        from src.degree_of_attack import DegreeOfAttackComputer
        deg_computer = DegreeOfAttackComputer(method="proposed")
        degree_values = deg_computer.compute_degree(X_array, feature_names=feature_names)
        
    return X_array, y_binary, feature_names, degree_values, y_attack_type


def engineer_features_universal(df: pd.DataFrame, 
                                 dataset_name: str,
                                 **kwargs) -> Dict[str, Any]:
    """
    Universal feature engineering entry point using explicit schemas.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe
    dataset_name : str
        Dataset name ("CIC-DDoS2019", "NSL-KDD", or "InSDN")
        
    Returns
    -------
    dict
        Dictionary with X, y, feature_names, degrees, and attack-type labels
    """
    schema = get_dataset_schema(dataset_name)
    
    if isinstance(schema, CICDDOS2019Schema):
        X, y_bin, feat_names, degrees, y_attack_type = engineer_features_cicddos2019(df, schema, **kwargs)
        return {
            "X": X,
            "y_binary": y_bin,
            "y_attack_type": y_attack_type,
            "feature_names": feat_names,
            "degree_values": degrees,
            "dataset": dataset_name,
            "schema": schema
        }
    elif isinstance(schema, NSLKDDSchema):
        X, y_bin, feat_names, degrees, y_attack_cat, y_attack_type = engineer_features_nsl_kdd(df, schema, **kwargs)
        return {
            "X": X,
            "y_binary": y_bin,
            "y_attack_category": y_attack_cat,
            "y_attack_type": y_attack_type,
            "feature_names": feat_names,
            "degree_values": degrees,
            "dataset": dataset_name,
            "schema": schema
        }
    elif isinstance(schema, InSDNSchema):
        X, y_bin, feat_names, degrees, y_attack_type = engineer_features_insdn(df, schema, **kwargs)
        return {
            "X": X,
            "y_binary": y_bin,
            "y_attack_type": y_attack_type,
            "feature_names": feat_names,
            "degree_values": degrees,
            "dataset": dataset_name,
            "schema": schema
        }
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


# =============================================================================
# Validation
# =============================================================================

def validate_schema_compliance(df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
    """
    Validate that a dataframe complies with the expected schema.
    
    Returns
    -------
    dict
        Validation report with missing/extra columns, type issues, etc.
    """
    schema = get_dataset_schema(dataset_name)
    
    expected = set(schema.raw_features)
    actual = set(df.columns)
    
    missing = expected - actual
    extra = actual - expected
    
    # Check label column
    label_present = schema.label_column in df.columns
    
    # Check categorical columns
    cat_present = [c for c in schema.categorical_features if c in df.columns]
    cat_missing = [c for c in schema.categorical_features if c not in df.columns]
    
    # Check numerical columns
    num_present = [c for c in schema.numerical_features if c in df.columns]
    num_missing = [c for c in schema.numerical_features if c not in df.columns]
    
    report = {
        "dataset": dataset_name,
        "total_expected": len(expected),
        "total_actual": len(actual),
        "missing_columns": list(missing),
        "extra_columns": list(extra),
        "label_column_present": label_present,
        "categorical_present": cat_present,
        "categorical_missing": cat_missing,
        "numerical_present": len(num_present),
        "numerical_missing": num_missing,
        "compliance_score": 1.0 - (len(missing) / len(expected)) if expected else 0.0
    }
    
    return report


if __name__ == "__main__":
    print("Testing dataset feature schemas...")
    
    # Test CIC-DDoS2019 schema
    print("\n1. CIC-DDoS2019 Schema:")
    cic_schema = CICDDOS2019Schema()
    print(f"  Dataset: {cic_schema.dataset_name}")
    print(f"  Raw features: {len(cic_schema.raw_features)}")
    print(f"  Feature groups: {list(cic_schema.feature_groups.keys())}")
    print(f"  Numerical features: {len(cic_schema.numerical_features)}")
    print(f"  Categorical features: {cic_schema.categorical_features}")
    print(f"  Attack types: {cic_schema.attack_type_mapping}")
    
    # Test NSL-KDD schema
    print("\n2. NSL-KDD Schema:")
    nsl_schema = NSLKDDSchema()
    print(f"  Dataset: {nsl_schema.dataset_name}")
    print(f"  Raw features: {len(nsl_schema.raw_features)}")
    print(f"  Feature groups: {list(nsl_schema.feature_groups.keys())}")
    print(f"  Numerical features: {len(nsl_schema.numerical_features)}")
    print(f"  Categorical features: {nsl_schema.categorical_features}")
    print(f"  Attack categories: {nsl_schema.attack_categories}")
    
    # Test factory
    print("\n3. Schema Factory:")
    for name in ["CIC-DDoS2019", "NSL-KDD"]:
        schema = get_dataset_schema(name)
        print(f"  {name}: {schema.dataset_name} - {len(schema.get_feature_columns())} features")
    
    print("\n✓ Schema validation complete!")