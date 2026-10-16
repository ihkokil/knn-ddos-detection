"""
Traffic Generation Module
Generates synthetic normal and DDoS attack traffic for SDN environment simulation.

Supports:
- Normal traffic generation with realistic flow characteristics
- DDoS attack traffic: UDP flood, SYN flood, ICMP flood, HTTP flood
- Configurable attack intensity and duration
- Labeled output DataFrames for training and evaluation
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Any


class TrafficGenerator:
    """
    Generates synthetic SDN traffic data for DDoS detection experiments.
    
    Can generate:
    1. Pure normal traffic
    2. Pure attack traffic (various types)
    3. Mixed traffic (normal + attacks at configurable intensities)
    4. Time-evolving traffic with changing attack patterns
    
    All data is labeled (0 = normal, 1 = attack) and includes flow-level features.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the traffic generator.
        
        Parameters
        ----------
        config : dict
            Generation configuration:
            - 'normal_mean_rate': mean packet rate for normal traffic
            - 'normal_std_rate': std dev for packet rate
            - 'attack_types': list of attack types to generate
            - 'attack_intensities': multiples of normal rate
            - 'attack_duration': seconds of attack per scenario
            - 'random_seed': for reproducibility
        """
        if config is None:
            config = {
                'normal_mean_rate': 10.0,
                'normal_std_rate': 2.0,
                'attack_intensities': [1, 5, 10, 20, 50],
                'attack_types': ['udp_flood', 'syn_flood', 'icmp_flood'],
                'attack_duration': 30.0,
                'random_seed': 42
            }
        self.config = config
        np.random.seed(config.get('random_seed', 42)) if config.get('random_seed') else None
    
    def generate_normal_traffic(self, 
                                num_flows: int = 1000,
                                seed: Optional[int] = None) -> pd.DataFrame:
        """
        Generate synthetic normal (non-attack) SDN flow traffic.
        
        Normal traffic characteristics:
        - Lower packet/byte rates
        - Longer flow durations (typical interactive/bulk transfers)
        - Diverse protocols but low intensity
        - Entropy features near uniform distribution
        
        Parameters
        ----------
        num_flows : int
            Number of normal flow samples to generate
        seed : int, optional
            Random seed for reproducibility
        
        Returns
        -------
        pd.DataFrame
            DataFrame with normal flow data and 'label'=0 column
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Flow characteristics for normal traffic
        n = num_flows
        
        data = pd.DataFrame({
            'flow_id': range(n),
            'duration': np.random.exponential(10, n),  # Mean 10 seconds
            'packets': np.random.poisson(50, n),  # ~50 packets per flow
            'bytes': np.random.normal(5000, 1000, n),  # ~5KB per flow
            'src_ip': [f"10.0.0.{np.random.randint(1, 254)}" for _ in range(n)],
            'dst_ip': [f"10.0.0.{np.random.randint(1, 254)}" for _ in range(n)],
            'src_port': np.random.randint(1024, 65535, n),
            'dst_port': np.random.randint(80, 443, n),  # Mostly web/ app ports
            'protocol': np.random.choice(['TCP', 'UDP'], n),
            'flag': np.random.choice(['', 'SA', 'A', 'R'], n),
            'label': 0  # Normal
        })
        
        # Add entropy features (simulated)
        data = self._add_entropy_features(data)
        
        # Shuffle
        data = data.sample(frac=1, random_state=seed).reset_index(drop=True)
        
        return data
    
    def generate_attack_traffic(self,
                               num_flows: int = 500,
                               attack_type: str = 'udp_flood',
                               intensity: float = 10.0,
                               seed: Optional[int] = None) -> pd.DataFrame:
        """
        Generate synthetic DDoS attack traffic.
        
        Attack traffic characteristics by type:
        - UDP flood: High packet/byte rates, UDP protocol, short durations
        - SYN flood: TCP protocol, SYN flags, medium rates
        - ICMP flood: ICMP protocol, echo requests, high rates
        - HTTP flood: TCP, port 80/443, application-layer patterns
        
        Parameters
        ----------
        num_flows : int
            Number of attack flow samples
        attack_type : str
            Type of DDoS attack: 'udp_flood', 'syn_flood', 'icmp_flood', 'http_flood'
        intensity : float
            Multiplier for attack severity (1.0 = baseline, higher = more intense)
        seed : int, optional
            Random seed
        
        Returns
        -------
        pd.DataFrame
            DataFrame with attack flow data and 'label'=1 column
        """
        if seed is not None:
            np.random.seed(seed)
        
        n = num_flows
        
        # Base attack characteristics (before intensity multiplication)
        if attack_type == 'udp_flood':
            # UDP flood: high rates, UDP protocol
            data = pd.DataFrame({
                'flow_id': range(n),
                'duration': np.random.exponential(2, n),  # Short flows
                'packets': np.random.poisson(200 * intensity, n),  # Many packets
                'bytes': np.random.normal(20000 * intensity, 5000, n),  # Many bytes
                'src_ip': [f"10.0.0.{np.random.randint(1, 254)}" for _ in range(n)],
                'dst_ip': [f"10.0.0.{np.random.randint(1, 254)}" for _ in range(n)],
                'src_port': np.random.randint(1024, 65535, n),
                'dst_port': np.random.choice([53, 67, 123, 161], n),  # Common UDP services
                'protocol': 'UDP',
                'flag': np.random.choice(['', 'SA'], n),
                'label': 1  # Attack
            })
        
        elif attack_type == 'syn_flood':
            # SYN flood: TCP with SYN flags
            data = pd.DataFrame({
                'flow_id': range(n),
                'duration': np.random.exponential(5, n),
                'packets': np.random.poisson(100 * intensity, n),
                'bytes': np.random.normal(10000 * intensity, 3000, n),
                'src_ip': [f"10.0.0.{np.random.randint(1, 254)}" for _ in range(n)],
                'dst_ip': [f"10.0.0.{np.random.randint(1, 254)}" for _ in range(n)],
                'src_port': np.random.randint(1024, 65535, n),
                'dst_port': np.random.choice([80, 443, 22], n),  # Web/SSH ports
                'protocol': 'TCP',
                'flag': np.random.choice(['SA', ''], n),  # SYN-ACK or empty
                'label': 1
            })
        
        elif attack_type == 'icmp_flood':
            # ICMP flood: ICMP protocol
            data = pd.DataFrame({
                'flow_id': range(n),
                'duration': np.random.exponential(1, n),  # Very short
                'packets': np.random.poisson(500 * intensity, n),  # Very many packets
                'bytes': np.random.normal(5000 * intensity, 2000, n),
                'src_ip': [f"10.0.0.{np.random.randint(1, 254)}" for _ in range(n)],
                'dst_ip': [f"10.0.0.{np.random.randint(1, 254)}" for _ in range(n)],
                'src_port': np.random.randint(1024, 65535, n),
                'dst_port': np.random.randint(1, 65535, n),  # Any port
                'protocol': 'ICMP',
                'flag': np.random.choice(['', 'EchoRequest', 'EchoReply'], n),
                'label': 1
            })
        
        elif attack_type == 'http_flood':
            # HTTP flood: application layer, TCP, port 80/443
            data = pd.DataFrame({
                'flow_id': range(n),
                'duration': np.random.exponential(10, n),
                'packets': np.random.poisson(100 * intensity, n),
                'bytes': np.random.normal(10000 * intensity, 4000, n),
                'src_ip': [f"10.0.0.{np.random.randint(1, 254)}" for _ in range(n)],
                'dst_ip': [f"10.0.0.{np.random.randint(1, 254)}" for _ in range(n)],
                'src_port': np.random.randint(1024, 65535, n),
                'dst_port': np.random.choice([80, 443], n),
                'protocol': 'TCP',
                'flag': np.random.choice(['', 'A', 'R'], n),
                'label': 1
            })
        
        else:
            raise ValueError(f"Unknown attack type: {attack_type}. "
                           f"Available: udp_flood, syn_flood, icmp_flood, http_flood")
        
        # Add entropy features
        data = self._add_entropy_features(data)
        
        # Shuffle
        data = data.sample(frac=1, random_state=seed).reset_index(drop=True)
        
        return data
    
    def generate_mixed_traffic(self,
                              normal_flows: int = 1000,
                              attack_configs: List[Dict] = None,
                              seed: Optional[int] = None) -> pd.DataFrame:
        """
        Generate mixed traffic combining normal and multiple attack types.
        
        Parameters
        ----------
        normal_flows : int
            Number of normal flow samples
        attack_configs : list of dict, optional
            List of attack configurations, each with keys:
            - 'num_flows': int
            - 'attack_type': str
            - 'intensity': float
        seed : int, optional
            Random seed
        
        Returns
        -------
        pd.DataFrame
            Labeled mixed traffic DataFrame
        """
        if seed is not None:
            np.random.seed(seed)
        
        if attack_configs is None:
            attack_configs = [
                {'num_flows': 200, 'attack_type': 'udp_flood', 'intensity': 10.0},
                {'num_flows': 200, 'attack_type': 'syn_flood', 'intensity': 5.0},
                {'num_flows': 100, 'attack_type': 'icmp_flood', 'intensity': 20.0}
            ]
        
        # Generate normal traffic
        normal_data = self.generate_normal_traffic(normal_flows, seed)
        
        # Generate and concatenate attack traffic
        attack_lists = []
        for config in attack_configs:
            attack_data = self.generate_attack_traffic(
                num_flows=config['num_flows'],
                attack_type=config['attack_type'],
                intensity=config['intensity'],
                seed=seed
            )
            attack_lists.append(attack_data)
        
        # Combine all
        if attack_lists:
            attack_combined = pd.concat(attack_lists, ignore_index=True)
        else:
            attack_combined = pd.DataFrame()
        
        # Combine normal + attack
        if not attack_combined.empty:
            mixed_data = pd.concat([normal_data, attack_combined], ignore_index=True)
        else:
            mixed_data = normal_data
        
        # Shuffle thoroughly
        mixed_data = mixed_data.sample(frac=1, random_state=seed).reset_index(drop=True)
        
        return mixed_data
    
    def _add_entropy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add entropy-based features to traffic data.
        
        Entropy measures the unpredictability of source IPs, destination IPs,
        and port distributions. Useful for distinguishing attack patterns.
        """
        df = df.copy()
        n = len(df)
        
        # Source IP entropy (simulated: based on diversity of src IPs)
        # In practice: compute from actual IP distribution
        unique_src_ips = df['src_ip'].nunique()
        df['src_ip_entropy'] = -np.log2(1 / unique_src_ips) / np.log2(n) if unique_src_ips > 1 else 0
        df['src_ip_entropy'] = np.clip(df['src_ip_entropy'], 0, 1)
        
        # Destination IP entropy
        unique_dst_ips = df['dst_ip'].nunique()
        df['dst_ip_entropy'] = -np.log2(1 / unique_dst_ips) / np.log2(n) if unique_dst_ips > 1 else 0
        df['dst_ip_entropy'] = np.clip(df['dst_ip_entropy'], 0, 1)
        
        # Port distribution entropy
        unique_src_ports = df['src_port'].nunique()
        df['src_port_entropy'] = -np.log2(1 / unique_src_ports) / np.log2(n) if unique_src_ports > 1 else 0
        df['src_port_entropy'] = np.clip(df['src_port_entropy'], 0, 1)
        
        unique_dst_ports = df['dst_port'].nunique()
        df['dst_port_entropy'] = -np.log2(1 / unique_dst_ports) / np.log2(n) if unique_dst_ports > 1 else 0
        df['dst_port_entropy'] = np.clip(df['dst_port_entropy'], 0, 1)
        
        return df
    
    def generate_time_evolving_traffic(self,
                                      duration_minutes: float = 60.0,
                                      detection_interval: float = 1.0,
                                      attack_start_minute: float = 15.0,
                                      attack_intensity: float = 10.0,
                                      seed: Optional[int] = None) -> pd.DataFrame:
        """
        Generate time-evolving traffic with attack starting at a specific time.
        
        Useful for simulating real-time detection scenarios where attack
        begins after a period of normal operation.
        
        Parameters
        ----------
        duration_minutes : float
            Total simulation duration in minutes
        detection_interval : float
            Interval between detection runs (not used in generation, for context)
        attack_start_minute : float
            When attack starts (as fraction of total duration)
        attack_intensity : float
            Attack intensity multiplier
        seed : int, optional
            Random seed
        
        Returns
        -------
        pd.DataFrame
            Time-evolving labeled traffic data
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Total flows based on rate (flows per minute)
        flows_per_minute_normal = 50
        flows_per_minute_attack = 200  # Much higher during attack
        
        total_minutes = int(duration_minutes)
        n_flows_total = 0
        
        # Build traffic per minute
        all_minutes = []
        
        for minute in range(total_minutes):
            # Determine if attack is active this minute
            is_attack = minute >= int(attack_start_minute)
            
            # Number of flows this minute
            if is_attack:
                n_flows_this_minute = int(flows_per_minute_attack * attack_intensity / 10.0)
                attack_type = np.random.choice(['udp_flood', 'syn_flood', 'icmp_flood'])
                intensity = attack_intensity
            else:
                n_flows_this_minute = flows_per_minute_normal
                attack_type = None
                intensity = 1.0
            
            # Generate flows for this minute
            if is_attack:
                attack_data = self.generate_attack_traffic(
                    num_flows=n_flows_this_minute,
                    attack_type=attack_type,
                    intensity=intensity,
                    seed=seed + minute  # Different seed per minute
                )
                # Add time stamp info
                attack_data['minute'] = minute
                attack_data['is_attack_period'] = True
            else:
                normal_data = self.generate_normal_traffic(n_flows_this_minute, seed + minute)
                normal_data['minute'] = minute
                normal_data['is_attack_period'] = False
            
            all_minutes.append(attack_data if is_attack else normal_data)
        
        # Combine all minutes
        if all_minutes:
            full_data = pd.concat(all_minutes, ignore_index=True)
        else:
            full_data = self.generate_normal_traffic(1000, seed)
        
        # Shuffle
        full_data = full_data.sample(frac=1, random_state=seed).reset_index(drop=True)
        
        return full_data


# Convenience functions

def create_traffic_generator(config: Dict[str, Any] = None) -> TrafficGenerator:
    """
    Create a TrafficGenerator instance from configuration.
    
    Parameters
    ----------
    config : dict, optional
        Generator configuration
    
    Returns
    -------
    TrafficGenerator
        Configured traffic generator
    """
    return TrafficGenerator(config=config)


def quick_generate_dataset(normal_flows: int = 2000,
                           attack_flows_per_type: Dict[str, int] = None,
                           seed: int = 42) -> pd.DataFrame:
    """
    Quick function to generate a labeled dataset for DDoS detection.
    
    Parameters
    ----------
    normal_flows : int
        Number of normal flow samples
    attack_flows_per_type : dict, optional
        Attack type -> number of flows mapping
    seed : int
        Random seed
    
    Returns
    -------
    pd.DataFrame
        Labeled dataset ready for feature engineering
    """
    if attack_flows_per_type is None:
        attack_flows_per_type = {
            'udp_flood': 500,
            'syn_flood': 500,
            'icmp_flood': 300
        }
    
    gen = TrafficGenerator({'random_seed': seed})
    
    # Generate normal
    normal = gen.generate_normal_traffic(normal_flows, seed)
    
    # Generate attacks
    attack_list = []
    for atype, n_attack in attack_flows_per_type.items():
        attack = gen.generate_attack_traffic(
            num_flows=n_attack,
            attack_type=atype,
            intensity=10.0,
            seed=seed
        )
        attack_list.append(attack)
    
    # Combine
    if attack_list:
        attack_combined = pd.concat(attack_list, ignore_index=True)
        dataset = pd.concat([normal, attack_combined], ignore_index=True)
    else:
        dataset = normal
    
    # Shuffle
    dataset = dataset.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    print(f"Generated dataset: {len(dataset)} samples")
    print(f"  Normal: {np.sum(dataset['label'] == 0)} ({100 * np.sum(dataset['label'] == 0) / len(dataset):.1f}%)")
    print(f"  Attack: {np.sum(dataset['label'] == 1)} ({100 * np.sum(dataset['label'] == 1) / len(dataset):.1f}%)")
    
    return dataset


if __name__ == "__main__":
    # Quick test: traffic generation
    print("Testing traffic generation module...")
    
    gen = TrafficGenerator({'random_seed': 42, 'attack_types': ['udp_flood', 'syn_flood', 'icmp_flood']})
    
    # Generate normal traffic
    normal_data = gen.generate_normal_traffic(num_flows=500, seed=42)
    print(f"\n✓ Normal traffic: {len(normal_data)} flows")
    print(f"  Label distribution: {normal_data['label'].value_counts().to_dict()}")
    print(f"  Features: {list(normal_data.columns)}")
    
    # Generate attack traffic
    udp_data = gen.generate_attack_traffic(num_flows=200, attack_type='udp_flood', intensity=10.0, seed=42)
    print(f"\n✓ UDP flood attack: {len(udp_data)} flows")
    print(f"  Duration mean: {udp_data['duration'].mean():.1f}s")
    print(f"  Packet rate mean: {udp_data['packet_rate'].mean():.1f} pps")
    print(f"  Label distribution: {udp_data['label'].value_counts().to_dict()}")
    
    # Generate mixed traffic
    mixed_data = gen.generate_mixed_traffic(
        normal_flows=800,
        attack_configs=[
            {'num_flows': 300, 'attack_type': 'udp_flood', 'intensity': 10.0},
            {'num_flows': 200, 'attack_type': 'syn_flood', 'intensity': 5.0}
        ],
        seed=42
    )
    print(f"\n✓ Mixed traffic: {len(mixed_data)} flows")
    print(f"  Attack percentage: {100 * np.sum(mixed_data['label'] == 1) / len(mixed_data):.1f}%")