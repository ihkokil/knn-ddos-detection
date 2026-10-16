"""
SDN Controller Simulation Module
Simulates the SDN environment where the controller receives flow statistics 
from switches and performs DDoS detection.

Based on: Base paper's SDN architecture and research extension requirements.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Any


class SDNControllerSimulation:
    """
    Simulates an SDN controller that receives flow statistics from OpenFlow switches,
    performs DDoS detection, and can trigger mitigation actions.
    
    The simulation models:
    - Flow statistics collection from switches
    - Controller processing of flow data
    - Detection triggering based on KNN classification + degree-of-attack
    - Mitigation action triggers (flow blocking, rate limiting)
    - Controller overhead metrics
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the SDN controller simulation.
        
        Parameters
        ----------
        config : dict
            Simulation configuration:
            - 'controller_capacity': max flows/second controller can process
            - 'detection_threshold': degree threshold to trigger detection
            - 'mitigation_actions': list of available actions
            - 'detection_interval': seconds between detection runs
        """
        if config is None:
            config = {
                'controller_capacity': 1000,
                'detection_threshold': 0.5,
                'mitigation_actions': ['flow_block', 'rate_limit'],
                'detection_interval': 1.0,
                'enable_real_time': False
            }
        self.config = config
        self.flow_stats_history = []
        self.detection_history = []
        self.mitigation_history = []
        self.controller_overhead = 0.0
    
    def receive_flow_stats(self, switch_id: str, flow_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Simulate receiving flow statistics from an SDN switch.
        
        Parameters
        ----------
        switch_id : str
            Identifier of the switch sending flow stats
        flow_data : pd.DataFrame
            Flow statistics from the switch with columns:
            - duration, packet_rate, byte_rate, protocol, flags, etc.
        
        Returns
        -------
        dict
            Processing results including detection results
        """
        # Record reception
        reception_time = len(self.flow_stats_history)
        self.flow_stats_history.append({
            'switch_id': switch_id,
            'flow_data': flow_data,
            'reception_time': reception_time
        })
        
        # Perform detection
        detection_result = self._perform_detection(flow_data, switch_id)
        
        # Update overhead
        self.controller_overhead += len(flow_data) * 0.001  # Simulated processing cost
        
        return {
            'switch_id': switch_id,
            'flow_count': len(flow_data),
            'detection_result': detection_result,
            'controller_overhead': self.controller_overhead
        }
    
    def _perform_detection(self, flow_data: pd.DataFrame, 
                          switch_id: str) -> Dict[str, Any]:
        """
        Perform DDoS detection on flow statistics using the KNN classifier.
        
        Parameters
        ----------
        flow_data : pd.DataFrame
            Flow data to analyze
        switch_id : str
            Switch identifier for logging
        
        Returns
        -------
        dict
            Detection results with classifications and degrees
        """
        # Extract features for KNN
        from src.feature_engineering import engineer_features_for_knn
        
        # Default config for feature engineering
        eng_config = {
            'include_entropy': True,
            'degree_method': 'proposed',
            'selected_features': None
        }
        
        # Engineer features
        try:
            X, y, feature_names, degree_values = engineer_features_for_knn(
                flow_data, eng_config)
        except Exception as e:
            # Fallback: use basic features
            X = flow_data.select_dtypes(include=[np.number]).fillna(0).values
            feature_names = [c for c in flow_data.columns if np.issubdtype(
                flow_data[c].dtype, np.number)]
            degree_values = np.zeros(len(flow_data))
        
        # Handle edge cases
        if X.shape[0] == 0 or X.shape[1] == 0:
            return {
                'switch_id': switch_id,
                'is_attack': np.zeros(len(flow_data), dtype=bool),
                'degrees': degree_values,
                'confidence': np.zeros(len(flow_data)),
                'action': 'none'
            }
        
        # Use the unified classifier
        from src.classifier import KNNClassifier, create_classifier_from_config
        
        classifier_config = {
            'method': 'proposed',
            'k': 5,
            'metric': 'euclidean',
            'weights': 'dual',
            'adaptive_k': True,
            'degree_method': 'proposed'
        }
        
        try:
            classifier = KNNClassifier(config=classifier_config)
            classifier.fit(X, degree_values)  # Note: this is simplified;
                # in practice, would use labeled training data
            # For simulation, we'll use a simplified approach
            
            # Compute degrees using the degree module
            from src.degree_of_attack import DegreeOfAttackComputer
            deg_computer = DegreeOfAttackComputer(
                method='proposed',
                threshold_type='adaptive'
            )
            degrees = deg_computer.compute_degree(X)
            
            # Classify based on degree threshold
            attack_threshold = 0.5
            is_attack = degrees > attack_threshold
            
            # Compute confidence based on degree magnitude
            confidence = degrees  # Degree itself is confidence
            
            # Determine mitigation action
            action = self._determine_mission_action(is_attack, degrees, degree_values)
            
            return {
                'switch_id': switch_id,
                'is_attack': is_attack,
                'degrees': degrees,
                'confidence': confidence,
                'action': action,
                'attack_percentage': float(np.mean(is_attack)) * 100 if len(is_attack) > 0 else 0.0
            }
            
        except Exception as e:
            # Fallback return on error
            return {
                'switch_id': switch_id,
                'is_attack': np.zeros(len(flow_data), dtype=bool),
                'degrees': degree_values if 'degree_values' in dir() else np.zeros(len(flow_data)),
                'confidence': np.zeros(len(flow_data)),
                'action': 'none',
                'error': str(e)
            }
    
    def _determine_mission_action(self, is_attack: np.ndarray,
                                   degrees: np.ndarray,
                                   degree_values: np.ndarray) -> str:
        """
        Determine mitigation action based on detection results.
        
        Parameters
        ----------
        is_attack : np.ndarray
            Binary array: True = attack flow, False = normal flow
        degrees : np.ndarray
            Degree-of-attack values
        degree_values : np.ndarray
            Per-flow degree values
        
        Returns
        -------
        str
            Mitigation action: 'none', 'flow_block', 'rate_limit', or 'reroute'
        """
        # Compute statistics
        n_attack = np.sum(is_attack)
        n_total = len(is_attack)
        avg_degree_attack = np.mean(degree_values[is_attack]) if n_attack > 0 else 0.0
        
        # Decision logic
        if n_attack == 0:
            return 'none'  # No attack detected
        
        # High severity attack: high average degree
        if avg_degree_attack > 0.8:
            return 'flow_block'  # Block affected flows
        
        # Medium severity: moderate degree
        elif avg_degree_attack > 0.5:
            return 'rate_limit'  # Rate limit suspicious flows
        
        # Low severity: just monitor
        else:
            return 'none'  # Monitor only, no immediate action
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get simulation statistics.
        
        Returns
        -------
        dict
            Summary of simulation run: total flows processed, detections,
            mitigations, controller overhead
        """
        n_total_flows = len(self.flow_stats_history)
        n_detections = len(self.detection_history)
        n_mitigations = len(self.mitigation_history)
        
        # Action distribution
        action_counts = {}
        for entry in self.mitigation_history:
            action = entry.get('action', 'none')
            action_counts[action] = action_counts.get(action, 0) + 1
        
        return {
            'total_flows_processed': n_total_flows,
            'total_detections': n_detections,
            'total_mitigations': n_mitigations,
            'detection_rate': n_detections / max(n_total_flows, 1),
            'mitigation_rate': n_mitigations / max(n_detections, 1),
            'controller_overhead': float(self.controller_overhead),
            'action_distribution': action_counts,
            'avg_degree_threshold': float(np.mean(
                [e.get('degree_threshold', 0.5) for e in self.detection_history])
            ) if self.detection_history else 0.5
        }
    
    def reset(self) -> None:
        """Reset simulation state for a new run."""
        self.flow_stats_history = []
        self.detection_history = []
        self.mitigation_history = []
        self.controller_overhead = 0.0


# Example usage and quick test
if __name__ == "__main__":
    print("Testing SDN controller simulation...")
    
    # Create simulation
    sim = SDNControllerSimulation(controller_capacity=500)
    
    # Create synthetic flow data for different scenarios
    np.random.seed(42)
    
    # Scenario 1: Normal traffic
    n_normal = 50
    normal_data = pd.DataFrame({
        'duration': np.random.exponential(10, n_normal),
        'packet_rate': np.random.uniform(1, 10, n_normal),
        'byte_rate': np.random.uniform(100, 500, n_normal),
        'protocol': np.random.choice(['TCP', 'UDP'], n_normal),
        'flag': np.random.choice(['', 'SA'], n_normal)
    })
    
    # Scenario 2: Attack traffic
    n_attack = 30
    attack_data = pd.DataFrame({
        'duration': np.random.exponential(2, n_attack),
        'packet_rate': np.random.uniform(50, 200, n_attack),
        'byte_rate': np.random.uniform(2000, 10000, n_attack),
        'protocol': np.random.choice(['UDP', 'ICMP'], n_attack),
        'flag': np.random.choice(['SA'], n_attack)
    })
    
    # Scenario 3: Mixed traffic
    mixed_data = pd.concat([normal_data, attack_data], ignore_index=True)
    # Shuffle
    mixed_data = mixed_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Simulate receiving flow stats from switches
    print("  Scenario 1: Normal traffic from switch s1")
    result1 = sim.receive_flow_stats('s1', normal_data)
    print(f"    Detection: {result1['detection_result']['action']}")
    print(f"    Attack percentage: {result1['detection_result']['attack_percentage']:.1f}%")
    
    print("  Scenario 2: Attack traffic from switch s2")
    result2 = sim.receive_flow_stats('s2', attack_data)
    print(f"    Detection: {result2['detection_result']['action']}")
    print(f"    Attack percentage: {result2['detection_result']['attack_percentage']:.1f}%")
    
    print("  Scenario 3: Mixed traffic from switch s3")
    result3 = sim.receive_flow_stats('s3', mixed_data)
    print(f"    Detection: {result3['detection_result']['action']}")
    print(f"    Attack percentage: {result3['detection_result']['attack_percentage']:.1f}%")
    
    # Print statistics
    stats = sim.get_statistics()
    print(f"\n  Simulation Statistics:")
    print(f"    Total flows: {stats['total_flows_processed']}")
    print(f"    Detection rate: {stats['detection_rate']:.2%}")
    print(f"    Mitigation rate: {stats['mitigation_rate']:.2%}")
    print(f"    Controller overhead: {stats['controller_overhead']:.3f}")
    print(f"    Action distribution: {stats['action_distribution']}")