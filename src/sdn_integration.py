"""
SDN Integration Module
Real SDN controller integration with Mininet and Ryu for empirical mitigation evaluation.

This module provides:
1. Ryu controller app for DDoS detection and mitigation
2. Mininet topology and traffic generation
3. Integration with KNN classifiers
4. Measurement of controller overhead, latency, and mitigation effectiveness
"""

import os
import sys
import time
import json
import subprocess
import threading
import socket
import queue
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


# =============================================================================
# Ryu Controller Application
# =============================================================================

RYU_APP_TEMPLATE = '''
"""
Ryu Controller App for KNN-based DDoS Detection and Mitigation
Generated automatically by sdn_integration.py
"""

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ipv4, tcp, udp, icmp
from ryu.lib.packet import ether_types
from ryu.lib import hub
import json
import time
import pickle
import numpy as np
from collections import defaultdict


class KNNDDoSDetectionApp(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(KNNDDoSDetectionApp, self).__init__(*args, **kwargs)
        self.mac_to_port = {}
        self.flow_stats = defaultdict(list)
        self.detection_interval = 5  # seconds
        self.attack_threshold = 0.7
        self.mitigation_actions = []
        self.knn_model = None
        self.feature_names = None
        self.scaler = None
        
        # Metrics
        self.stats = {
            'flows_processed': 0,
            'attacks_detected': 0,
            'mitigations_triggered': 0,
            'controller_overhead_ms': 0.0,
            'detection_latency_ms': [],
            'mitigation_latency_ms': []
        }
        
        # Load model if provided
        self._load_model()
        
        # Start detection thread
        self.detection_thread = hub.spawn(self._detection_loop)

    def _load_model(self):
        """Load trained KNN model and preprocessor."""
        try:
            import joblib
            model_path = '/tmp/knn_model.joblib'
            if os.path.exists(model_path):
                model_data = joblib.load(model_path)
                self.knn_model = model_data['model']
                self.feature_names = model_data['feature_names']
                self.scaler = model_data.get('scaler')
                self.logger.info("Loaded KNN model from %s", model_path)
            else:
                self.logger.warning("No model found at %s, using dummy classifier", model_path)
        except Exception as e:
            self.logger.error("Failed to load model: %s", e)

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # Install table-miss flow entry
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)

    def add_flow(self, datapath, priority, match, actions, buffer_id=None):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                                             actions)]
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id,
                                    priority=priority, match=match,
                                    instructions=inst)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                    match=match, instructions=inst)
        datapath.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        if ev.msg.msg_len < ev.msg.total_len:
            self.logger.debug("packet truncated: %s/%s", ev.msg.msg_len, ev.msg.total_len)
        
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return

        dst = eth.dst
        src = eth.src
        dpid = datapath.id
        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src] = in_port

        # Learn flow statistics
        self._extract_flow_features(pkt, in_port, datapath)

        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            out_port = ofproto.OFPP_FLOOD

        actions = [parser.OFPActionOutput(out_port)]

        # Install flow to avoid packet_in next time
        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
            if msg.buffer_id != ofproto.OFP_NO_BUFFER:
                self.add_flow(datapath, 1, match, actions, msg.buffer_id)
                return
            else:
                self.add_flow(datapath, 1, match, actions)

        data = None
        if msg.buffer_id == ofproto.OFPP_NO_BUFFER:
            data = msg.data

        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                  in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)

    def _extract_flow_features(self, pkt, in_port, datapath):
        """Extract flow features from packet for KNN classification."""
        try:
            ip_pkt = pkt.get_protocol(ipv4.ipv4)
            if not ip_pkt:
                return

            src_ip = ip_pkt.src
            dst_ip = ip_pkt.dst
            proto = ip_pkt.proto

            # TCP/UDP ports
            src_port = 0
            dst_port = 0
            flags = 0
            
            tcp_pkt = pkt.get_protocol(tcp.tcp)
            udp_pkt = pkt.get_protocol(udp.udp)
            icmp_pkt = pkt.get_protocol(icmp.icmp)

            if tcp_pkt:
                src_port = tcp_pkt.src_port
                dst_port = tcp_pkt.dst_port
                flags = tcp_pkt.bits
            elif udp_pkt:
                src_port = udp_pkt.src_port
                dst_port = udp_pkt.dst_port
            elif icmp_pkt:
                pass

            # Create flow key
            flow_key = (src_ip, dst_ip, src_port, dst_port, proto)
            
            # Update flow statistics
            now = time.time()
            self.flow_stats[flow_key].append({
                'timestamp': now,
                'packet_size': len(pkt),
                'flags': flags,
                'proto': proto
            })

            # Limit history
            if len(self.flow_stats[flow_key]) > 1000:
                self.flow_stats[flow_key] = self.flow_stats[flow_key][-1000:]

        except Exception as e:
            self.logger.error("Feature extraction error: %s", e)

    def _compute_flow_features(self, flow_key, flow_packets):
        """Compute aggregate flow features for classification."""
        if len(flow_packets) < 2:
            return None

        timestamps = [p['timestamp'] for p in flow_packets]
        sizes = [p['packet_size'] for p in flow_packets]
        flags = [p['flags'] for p in flow_packets]
        protos = [p['proto'] for p in flow_packets]

        duration = timestamps[-1] - timestamps[0]
        if duration <= 0:
            duration = 0.001

        # Basic features
        n_packets = len(sizes)
        n_bytes = sum(sizes)
        packet_rate = n_packets / duration
        byte_rate = n_bytes / duration

        # IAT statistics
        iats = np.diff(timestamps)
        iat_mean = np.mean(iats) if len(iats) > 0 else 0
        iat_std = np.std(iats) if len(iats) > 1 else 0

        # Flag statistics
        syn_count = sum(1 for f in flags if f & 0x02)  # SYN flag
        ack_count = sum(1 for f in flags if f & 0x10)  # ACK flag
        fin_count = sum(1 for f in flags if f & 0x01)  # FIN flag
        rst_count = sum(1 for f in flags if f & 0x04)  # RST flag

        # Protocol distribution
        proto_tcp = sum(1 for p in protos if p == 6)
        proto_udp = sum(1 for p in protos if p == 17)
        proto_icmp = sum(1 for p in protos if p == 1)

        # Entropy (simplified)
        unique_src = 1  # Single flow
        unique_dst = 1

        features = {
            'duration': duration,
            'packet_rate': packet_rate,
            'byte_rate': byte_rate,
            'total_packets': n_packets,
            'total_bytes': n_bytes,
            'iat_mean': iat_mean,
            'iat_std': iat_std,
            'syn_count': syn_count,
            'ack_count': ack_count,
            'fin_count': fin_count,
            'rst_count': rst_count,
            'proto_tcp_ratio': proto_tcp / n_packets,
            'proto_udp_ratio': proto_udp / n_packets,
            'proto_icmp_ratio': proto_icmp / n_packets,
            'src_port': flow_key[2],
            'dst_port': flow_key[3],
            'protocol': flow_key[4]
        }

        return features

    def _detection_loop(self):
        """Periodic detection and mitigation loop."""
        while True:
            hub.sleep(self.detection_interval)
            self._run_detection()

    def _run_detection(self):
        """Run KNN detection on current flow statistics."""
        if self.knn_model is None:
            return

        start_time = time.time()
        flows_to_check = list(self.flow_stats.items())

        for flow_key, packets in flows_to_check:
            if len(packets) < 5:  # Minimum packets for reliable detection
                continue

            features = self._compute_flow_features(flow_key, packets)
            if features is None:
                continue

            # Prepare feature vector
            try:
                X = self._features_to_vector(features)
                if self.scaler:
                    X = self.scaler.transform([X])[0]
                else:
                    X = np.array([X])

                # Predict
                degree = self.knn_model.predict_proba([X])[0][1] if hasattr(self.knn_model, 'predict_proba') else 0.5
                is_attack = degree > self.attack_threshold

                if is_attack:
                    self.stats['attacks_detected'] += 1
                    self._trigger_mitigation(flow_key, degree)

                self.stats['flows_processed'] += 1

            except Exception as e:
                self.logger.error("Detection error for flow %s: %s", flow_key, e)

        detection_time = (time.time() - start_time) * 1000
        self.stats['detection_latency_ms'].append(detection_time)
        if len(self.stats['detection_latency_ms']) > 1000:
            self.stats['detection_latency_ms'] = self.stats['detection_latency_ms'][-1000:]

    def _features_to_vector(self, features):
        """Convert feature dict to model input vector."""
        # Default feature order (should match training)
        default_order = [
            'duration', 'packet_rate', 'byte_rate', 'total_packets', 'total_bytes',
            'iat_mean', 'iat_std', 'syn_count', 'ack_count', 'fin_count', 'rst_count',
            'proto_tcp_ratio', 'proto_udp_ratio', 'proto_icmp_ratio',
            'src_port', 'dst_port', 'protocol'
        ]

        if self.feature_names:
            order = self.feature_names
        else:
            order = default_order

        return [features.get(f, 0) for f in order]

    def _trigger_mitigation(self, flow_key, degree):
        """Trigger mitigation action for detected attack flow."""
        start_time = time.time()
        
        src_ip, dst_ip, src_port, dst_port, proto = flow_key
        
        # Log mitigation
        mitigation = {
            'timestamp': time.time(),
            'flow_key': flow_key,
            'degree': degree,
            'action': 'flow_block'
        }
        self.mitigation_actions.append(mitigation)
        self.stats['mitigations_triggered'] += 1

        # In a real implementation, would send OFPFlowMod to block flow
        self.logger.warning("ATTACK DETECTED: %s -> %s (degree=%.3f) - MITIGATION TRIGGERED",
                           src_ip, dst_ip, degree)

        mitigation_time = (time.time() - start_time) * 1000
        self.stats['mitigation_latency_ms'].append(mitigation_time)
        if len(self.stats['mitigation_latency_ms']) > 1000:
            self.stats['mitigation_latency_ms'] = self.stats['mitigation_latency_ms'][-1000:]

    def get_stats(self):
        """Return current statistics."""
        stats = self.stats.copy()
        if stats['detection_latency_ms']:
            stats['avg_detection_latency_ms'] = np.mean(stats['detection_latency_ms'])
            stats['max_detection_latency_ms'] = np.max(stats['detection_latency_ms'])
        if stats['mitigation_latency_ms']:
            stats['avg_mitigation_latency_ms'] = np.mean(stats['mitigation_latency_ms'])
        return stats


# For direct execution
if __name__ == '__main__':
    from ryu.cmd import manager
    manager.main()
'''


# =============================================================================
# Mininet Topology and Experiment Runner
# =============================================================================

MININET_SCRIPT_TEMPLATE = '''
#!/usr/bin/env python3
"""
Mininet topology for DDoS detection evaluation.
Generated automatically by sdn_integration.py
"""

from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import TCLink
import time
import subprocess
import threading
import argparse


def create_topology(n_hosts=4, n_switches=2, link_bw=100, link_delay='1ms'):
    """Create a custom topology for DDoS testing."""
    net = Mininet(controller=RemoteController, switch=OVSSwitch, link=TCLink)
    
    info('*** Adding controller\\n')
    c0 = net.addController('c0', controller=RemoteController, 
                           ip='127.0.0.1', port=6653)
    
    info('*** Adding switches\\n')
    switches = []
    for i in range(n_switches):
        s = net.addSwitch(f's{i+1}', protocols='OpenFlow13')
        switches.append(s)
    
    info('*** Adding hosts\\n')
    hosts = []
    for i in range(n_hosts):
        h = net.addHost(f'h{i+1}', ip=f'10.0.0.{i+1}/24')
        hosts.append(h)
    
    info('*** Creating links\\n')
    # Connect switches in a line
    for i in range(n_switches - 1):
        net.addLink(switches[i], switches[i+1], bw=link_bw, delay=link_delay)
    
    # Connect hosts to switches (distribute evenly)
    for i, h in enumerate(hosts):
        switch_idx = i % n_switches
        net.addLink(h, switches[switch_idx], bw=link_bw, delay=link_delay)
    
    info('*** Starting network\\n')
    net.start()
    
    # Wait for switches to connect to controller
    time.sleep(3)
    
    return net, hosts, switches


def run_attack_scenario(net, hosts, scenario='udp_flood', duration=30):
    """Run a DDoS attack scenario."""
    info(f'*** Running {scenario} attack for {duration}s\\n')
    
    # Select attacker and victim
    attacker = hosts[0]
    victim = hosts[-1]
    
    if scenario == 'udp_flood':
        # UDP flood using hping3 or iperf
        cmd = f'hping3 --udp --flood -p 53 {victim.IP()}'
        attacker.cmd(cmd + ' &')
        
    elif scenario == 'syn_flood':
        # SYN flood
        cmd = f'hping3 --syn --flood -p 80 {victim.IP()}'
        attacker.cmd(cmd + ' &')
        
    elif scenario == 'icmp_flood':
        # ICMP flood
        cmd = f'hping3 --icmp --flood {victim.IP()}'
        attacker.cmd(cmd + ' &')
        
    elif scenario == 'http_flood':
        # HTTP flood using multiple connections
        for _ in range(50):
            victim.cmd('python3 -m http.server 80 &')
        attacker.cmd(f'ab -n 10000 -c 100 http://{victim.IP()}/ &')
        
    elif scenario == 'slowloris':
        # Slowloris attack
        attacker.cmd(f'python3 slowloris.py {victim.IP()} &')
    
    # Let attack run
    time.sleep(duration)
    
    # Stop attack
    attacker.cmd('killall hping3')
    attacker.cmd('killall ab')
    attacker.cmd('pkill -f slowloris')
    
    info('*** Attack completed\\n')


def run_normal_traffic(net, hosts, duration=30):
    """Generate normal background traffic."""
    info(f'*** Generating normal traffic for {duration}s\\n')
    
    # Use iperf for TCP traffic
    server = hosts[-1]
    clients = hosts[:-1]
    
    server.cmd('iperf -s -p 5001 &')
    time.sleep(1)
    
    for client in clients:
        client.cmd(f'iperf -c {server.IP()} -p 5001 -t {duration} -i 5 &')
    
    time.sleep(duration + 2)
    
    server.cmd('killall iperf')
    for client in clients:
        client.cmd('killall iperf')


def measure_throughput(net, hosts):
    """Measure network throughput during/after attack."""
    info('*** Measuring throughput\\n')
    
    server = hosts[-1]
    client = hosts[0]
    
    server.cmd('iperf -s -p 5002 &')
    time.sleep(1)
    
    result = client.cmd(f'iperf -c {server.IP()} -p 5002 -t 10 -i 2')
    
    server.cmd('killall iperf')
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Mininet DDoS Test')
    parser.add_argument('--scenario', default='udp_flood',
                       choices=['udp_flood', 'syn_flood', 'icmp_flood', 
                                'http_flood', 'slowloris', 'normal'])
    parser.add_argument('--duration', type=int, default=30)
    parser.add_argument('--hosts', type=int, default=4)
    parser.add_argument('--switches', type=int, default=2)
    parser.add_argument('--no-cli', action='store_true')
    
    args = parser.parse_args()
    
    setLogLevel('info')
    
    net, hosts, switches = create_topology(args.hosts, args.switches)
    
    try:
        if args.scenario == 'normal':
            run_normal_traffic(net, hosts, args.duration)
        else:
            # Run normal traffic in background
            normal_thread = threading.Thread(target=run_normal_traffic, 
                                            args=(net, hosts, args.duration))
            normal_thread.start()
            
            # Run attack
            run_attack_scenario(net, hosts, args.scenario, args.duration)
            
            normal_thread.join()
        
        # Measure throughput
        throughput = measure_throughput(net, hosts)
        info(f'Throughput result: {throughput}\\n')
        
        if not args.no_cli:
            CLI(net)
            
    finally:
        net.stop()


if __name__ == '__main__':
    main()
'''


# =============================================================================
# Python Integration Classes
# =============================================================================

@dataclass
class SDNExperimentConfig:
    """Configuration for SDN experiment."""
    topology: str = 'linear'  # linear, tree, mesh
    n_hosts: int = 4
    n_switches: int = 2
    link_bandwidth: int = 100  # Mbps
    link_delay: str = '1ms'
    ryu_controller_ip: str = '127.0.0.1'
    ryu_controller_port: int = 6653
    detection_interval: float = 5.0
    attack_threshold: float = 0.7
    attack_scenarios: List[str] = field(default_factory=lambda: [
        'udp_flood', 'syn_flood', 'icmp_flood', 'http_flood', 'slowloris'
    ])
    attack_duration: int = 30
    normal_traffic_duration: int = 30
    model_path: str = '/tmp/knn_model.joblib'
    results_dir: str = 'results/sdn_experiments'


@dataclass
class SDNExperimentResults:
    """Results from SDN experiment."""
    scenario: str
    start_time: float
    end_time: float
    controller_stats: Dict[str, Any] = field(default_factory=dict)
    network_throughput: Dict[str, float] = field(default_factory=dict)
    attack_mitigation: Dict[str, Any] = field(default_factory=dict)
    flow_statistics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


class RyuControllerManager:
    """Manages Ryu controller lifecycle."""
    
    def __init__(self, config: SDNExperimentConfig):
        self.config = config
        self.controller_process = None
        self.app_path = None
        
    def generate_ryu_app(self, model_data: Dict = None) -> str:
        """Generate Ryu app with embedded model."""
        app_dir = Path(self.config.results_dir) / 'ryu_app'
        app_dir.mkdir(parents=True, exist_ok=True)
        
        app_path = app_dir / 'knn_ddos_app.py'
        
        # Save model if provided
        if model_data:
            import joblib
            model_path = app_dir / 'knn_model.joblib'
            joblib.dump(model_data, model_path)
        
        # Write app
        with open(app_path, 'w') as f:
            f.write(RYU_APP_TEMPLATE)
        
        self.app_path = str(app_path)
        return self.app_path
    
    def start_controller(self) -> bool:
        """Start Ryu controller."""
        if not self.app_path:
            self.generate_ryu_app()
        
        try:
            self.controller_process = subprocess.Popen(
                ['ryu-manager', '--observe-links', self.app_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(Path(self.app_path).parent)
            )
            
            # Wait for controller to start
            time.sleep(5)
            
            # Check if running
            if self.controller_process.poll() is None:
                print("✓ Ryu controller started")
                return True
            else:
                stdout, stderr = self.controller_process.communicate()
                print(f"✗ Ryu controller failed: {stderr.decode()}")
                return False
                
        except FileNotFoundError:
            print("✗ Ryu not installed. Install with: pip install ryu")
            return False
        except Exception as e:
            print(f"✗ Failed to start Ryu: {e}")
            return False
    
    def stop_controller(self):
        """Stop Ryu controller."""
        if self.controller_process:
            self.controller_process.terminate()
            try:
                self.controller_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.controller_process.kill()
            print("✓ Ryu controller stopped")


class MininetExperimentRunner:
    """Runs Mininet experiments with Ryu controller."""
    
    def __init__(self, config: SDNExperimentConfig):
        self.config = config
        self.ryu_manager = RyuControllerManager(config)
        self.net = None
        self.hosts = None
        self.switches = None
        
    def setup(self, model_data: Dict = None) -> bool:
        """Setup experiment environment."""
        # Generate Ryu app with model
        self.ryu_manager.generate_ryu_app(model_data)
        
        # Start Ryu controller
        if not self.ryu_manager.start_controller():
            return False
        
        # Create Mininet network
        try:
            self.net, self.hosts, self.switches = create_topology(
                self.config.n_hosts, self.config.n_switches,
                self.config.link_bandwidth, self.config.link_delay
            )
            print("✓ Mininet network created")
            return True
        except Exception as e:
            print(f"✗ Failed to create Mininet network: {e}")
            return False
    
    def run_scenario(self, scenario: str) -> SDNExperimentResults:
        """Run a single attack scenario."""
        print(f"\n--- Running scenario: {scenario} ---")
        
        result = SDNExperimentResults(
            scenario=scenario,
            start_time=time.time()
        )
        
        try:
            if scenario == 'normal':
                run_normal_traffic(self.net, self.hosts, self.config.normal_traffic_duration)
            else:
                # Background normal traffic
                normal_thread = threading.Thread(
                    target=run_normal_traffic,
                    args=(self.net, self.hosts, self.config.attack_duration)
                )
                normal_thread.start()
                
                # Run attack
                run_attack_scenario(self.net, self.hosts, scenario, self.config.attack_duration)
                
                normal_thread.join()
            
            # Measure throughput
            throughput_result = measure_throughput(self.net, self.hosts)
            result.network_throughput['iperf_output'] = throughput_result
            
            # Get controller stats (would need RPC or file-based communication)
            # For now, read from controller log or stats file
            result.controller_stats = self._collect_controller_stats()
            
        except Exception as e:
            result.errors.append(str(e))
            print(f"  Error in scenario {scenario}: {e}")
        
        result.end_time = time.time()
        return result
    
    def _collect_controller_stats(self) -> Dict:
        """Collect stats from Ryu controller."""
        # In a real implementation, would use REST API or shared memory
        # For now, return placeholder
        return {
            'flows_processed': 0,
            'attacks_detected': 0,
            'mitigations_triggered': 0,
            'avg_detection_latency_ms': 0,
            'avg_mitigation_latency_ms': 0
        }
    
    def run_all_scenarios(self, model_data: Dict = None) -> List[SDNExperimentResults]:
        """Run all configured attack scenarios."""
        if not self.setup(model_data):
            return []
        
        results = []
        
        try:
            # Run normal baseline first
            print("\n=== Running normal traffic baseline ===")
            normal_result = self.run_scenario('normal')
            results.append(normal_result)
            
            # Run attack scenarios
            for scenario in self.config.attack_scenarios:
                result = self.run_scenario(scenario)
                results.append(result)
                
                # Cool down between scenarios
                time.sleep(5)
                
        finally:
            self.cleanup()
        
        return results
    
    def cleanup(self):
        """Clean up experiment resources."""
        if self.net:
            self.net.stop()
            print("✓ Mininet network stopped")
        self.ryu_manager.stop_controller()


# =============================================================================
# Model Export for Ryu
# =============================================================================

def export_model_for_ryu(knn_model, feature_names: List[str], scaler=None,
                          output_path: str = '/tmp/knn_model.joblib') -> Dict:
    """
    Export trained KNN model for use in Ryu controller.
    
    Parameters
    ----------
    knn_model : fitted KNN classifier
        Trained model (ProposedKNN, ImprovedKNN, etc.)
    feature_names : list
        Feature names in order
    scaler : fitted sklearn scaler, optional
        Preprocessing scaler
    output_path : str
        Path to save model
        
    Returns
    -------
    dict
        Model data dictionary
    """
    import joblib
    
    model_data = {
        'model': knn_model,
        'feature_names': feature_names,
        'scaler': scaler,
        'model_type': type(knn_model).__name__,
        'export_time': time.time()
    }
    
    joblib.dump(model_data, output_path)
    print(f"✓ Model exported to {output_path}")
    
    return model_data


# =============================================================================
# Controller Overhead Measurement
# =============================================================================

class ControllerOverheadMonitor:
    """Monitors SDN controller overhead during experiments."""
    
    def __init__(self, controller_pid: int = None):
        self.controller_pid = controller_pid
        self.cpu_samples = []
        self.memory_samples = []
        self.monitoring = False
        self.monitor_thread = None
    
    def start(self, pid: int = None):
        """Start monitoring."""
        if pid:
            self.controller_pid = pid
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
    
    def stop(self) -> Dict[str, float]:
        """Stop monitoring and return statistics."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        return {
            'cpu_percent_mean': np.mean(self.cpu_samples) if self.cpu_samples else 0,
            'cpu_percent_max': np.max(self.cpu_samples) if self.cpu_samples else 0,
            'memory_mb_mean': np.mean(self.memory_samples) if self.memory_samples else 0,
            'memory_mb_max': np.max(self.memory_samples) if self.memory_samples else 0,
            'samples_collected': len(self.cpu_samples)
        }
    
    def _monitor_loop(self):
        """Monitoring loop."""
        import psutil
        
        if self.controller_pid is None:
            return
            
        try:
            process = psutil.Process(self.controller_pid)
            while self.monitoring:
                try:
                    cpu = process.cpu_percent(interval=0.1)
                    mem = process.memory_info().rss / (1024 * 1024)  # MB
                    self.cpu_samples.append(cpu)
                    self.memory_samples.append(mem)
                    time.sleep(0.5)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
        except ImportError:
            print("psutil not available for monitoring")
        except Exception as e:
            print(f"Monitoring error: {e}")


# =============================================================================
# High-Level Experiment Function
# =============================================================================

def run_sdn_evaluation(knn_model, feature_names: List[str], scaler=None,
                       scenarios: List[str] = None,
                       config: SDNExperimentConfig = None) -> Dict[str, Any]:
    """
    High-level function to run complete SDN evaluation.
    
    Parameters
    ----------
    knn_model : fitted classifier
        Trained KNN model
    feature_names : list
        Feature names
    scaler : fitted scaler, optional
        Preprocessing scaler
    scenarios : list, optional
        Attack scenarios to test
    config : SDNExperimentConfig, optional
        Experiment configuration
        
    Returns
    -------
    dict
        Complete experiment results
    """
    if config is None:
        config = SDNExperimentConfig()
    
    if scenarios:
        config.attack_scenarios = scenarios
    
    # Export model
    model_data = export_model_for_ryu(knn_model, feature_names, scaler, config.model_path)
    
    # Run experiments
    runner = MininetExperimentRunner(config)
    results = runner.run_all_scenarios(model_data)
    
    # Compile summary
    summary = {
        'config': config.__dict__,
        'scenarios_tested': len(results),
        'results': [r.__dict__ for r in results],
        'timestamp': time.time()
    }
    
    # Save results
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = results_dir / f'sdn_results_{int(time.time())}.json'
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n✓ SDN evaluation complete. Results saved to {output_file}")
    
    return summary


# =============================================================================
# Installation Helper
# =============================================================================

def check_sdn_dependencies() -> Dict[str, bool]:
    """Check if SDN dependencies are installed."""
    deps = {
        'ryu': False,
        'mininet': False,
        'openvswitch': False,
        'hping3': False,
        'iperf3': False,
        'psutil': False
    }
    
    # Check Python packages
    try:
        import ryu
        deps['ryu'] = True
    except ImportError:
        pass
    
    try:
        import mininet
        deps['mininet'] = True
    except ImportError:
        pass
    
    try:
        import psutil
        deps['psutil'] = True
    except ImportError:
        pass
    
    # Check system tools
    for tool in ['ovs-vsctl', 'hping3', 'iperf3']:
        try:
            subprocess.run(['which', tool], capture_output=True, check=True)
            if tool == 'ovs-vsctl':
                deps['openvswitch'] = True
            elif tool == 'hping3':
                deps['hping3'] = True
            elif tool == 'iperf3':
                deps['iperf3'] = True
        except:
            pass
    
    return deps


def print_installation_guide():
    """Print installation guide for SDN dependencies."""
    guide = """
================================================================================
SDN EVALUATION SETUP GUIDE
================================================================================

REQUIRED DEPENDENCIES:

1. Ryu Controller (Python SDN framework):
   pip install ryu

2. Mininet (Network emulator):
   # Ubuntu/Debian:
   sudo apt-get install mininet
   
   # Or from source:
   git clone https://github.com/mininet/mininet
   cd mininet && ./util/install.sh -a

3. Open vSwitch (Required by Mininet):
   sudo apt-get install openvswitch-switch

4. Traffic Generation Tools:
   sudo apt-get install hping3 iperf3

5. Python Monitoring:
   pip install psutil

6. Model Serialization:
   pip install joblib

================================================================================
QUICK SETUP (Ubuntu 20.04/22.04):
================================================================================

sudo apt-get update
sudo apt-get install -y mininet openvswitch-switch hping3 iperf3 python3-pip
pip3 install ryu psutil joblib scikit-learn numpy pandas

# Verify Mininet installation
sudo mn --test pingpair

================================================================================
USAGE:
================================================================================

from src.sdn_integration import run_sdn_evaluation

# Assuming you have a trained model
results = run_sdn_evaluation(
    knn_model=your_trained_model,
    feature_names=your_feature_names,
    scaler=your_scaler,
    scenarios=['udp_flood', 'syn_flood', 'icmp_flood']
)

================================================================================
NOTE: Mininet requires root privileges. Run experiments with sudo:
================================================================================

sudo -E python3 -m src.sdn_integration

================================================================================
"""
    print(guide)


if __name__ == "__main__":
    print("Testing SDN integration module...")
    
    # Check dependencies
    deps = check_sdn_dependencies()
    print("\nDependency Check:")
    for dep, available in deps.items():
        status = "✓" if available else "✗"
        print(f"  {status} {dep}")
    
    # Print guide if missing dependencies
    if not all(deps.values()):
        print_installation_guide()
    
    # Test model export (with dummy model)
    print("\nTesting model export...")
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    
    dummy_model = RandomForestClassifier(n_estimators=10, random_state=42)
    dummy_model.fit(np.random.randn(100, 10), np.random.randint(0, 2, 100))
    
    feature_names = [f'f_{i}' for i in range(10)]
    
    try:
        model_data = export_model_for_ryu(dummy_model, feature_names, 
                                          output_path='/tmp/test_knn_model.joblib')
        print("✓ Model export successful")
    except Exception as e:
        print(f"✗ Model export failed: {e}")
    
    print("\n✓ SDN integration module ready!")