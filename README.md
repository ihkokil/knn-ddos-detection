# KNN-based DDoS Detection & Mitigation in Software-Defined Networks

A comprehensive, reproducible research framework for DDoS attack detection and mitigation in Software-Defined Networks (SDN) using an improved K-Nearest Neighbors (KNN) model with degree-of-attack weighting and novel architectural enhancements.

## Overview

This project implements, evaluates, and extends:
- **Baseline KNN**: Standard K-Nearest Neighbors classifier
- **Improved KNN**: Base paper method incorporating degree-of-attack weighting and distance adjustment
- **Proposed KNN**: Novel extensions including:
  1. **Bounded Distance Compression**: `d' = d × (1 - degree/max_degree)`
  2. **Adaptive K Selection**: Per-sample dynamic K selection based on local attack severity
  3. **Dual Voting Scheme**: Combined `weight = proximity × degree` weighting

---

## Key Framework Features

- ✅ **Benchmark Datasets with Provenance**: Full support for CIC-DDoS2019 and NSL-KDD with verifiable checksums
- ✅ **Group & Time-Aware Splits**: Prevents data leakage from identical flow sources across train/test splits
- ✅ **Multi-Trial Statistical Rigor**: 30 independent runs with distinct seeds for robust confidence intervals
- ✅ **Competitive Machine Learning Baselines**: Random Forest, Support Vector Machines (SVM), XGBoost, LightGBM, Multi-Layer Perceptron (MLP), and Logistic Regression
- ✅ **Attack-Specific Granular Metrics**: Breakdown across UDP, TCP-SYN, TCP-ACK, ICMP, HTTP, and Slowloris floods
- ✅ **Calibration & Performance Profiling**: Expected Calibration Error (ECE), Maximum Calibration Error (MCE), inference throughput, latency, and memory footprint
- ✅ **Modular Component Ablation**: Independent validation of all 4 proposed algorithmic modules
- ✅ **Non-Parametric Statistical Testing**: Wilcoxon signed-rank, Friedman omnibus tests, Cohen's d / Cliff's δ effect sizes, and Holm-Bonferroni correction
- ✅ **Emulated SDN Environment**: Mininet topology integration with Ryu SDN controller for real-time traffic monitoring and automated flow-table mitigation

---

## Installation

```bash
# Core dependencies
pip install -r requirements.txt

# For optional SDN Mininet emulation (Linux / root):
sudo apt-get install mininet openvswitch-switch hping3 iperf3
pip install ryu psutil
```

---

## Quick Start

```bash
# 1. Quick pipeline verification on synthetic traffic
python main.py --phase full_eval --quick-test

# 2. Comprehensive evaluation on NSL-KDD (10 trials)
python main.py --phase full_eval --datasets NSL-KDD --num-trials 10

# 3. Comprehensive evaluation on both datasets (30 trials)
python main.py --phase full_eval --datasets both --num-trials 30 --download-data

# 4. Full evaluation with baselines and statistical testing
python main.py --phase full_eval --datasets both --num-trials 30 \
    --no-baselines=False --no-stats=False
```

---

## Command Line Options

### Phase Selection
- `--phase data` - Load and preprocess dataset only
- `--phase train` - Data loading + model training
- `--phase evaluate` - Data loading + training + evaluation
- `--phase experiment` - Run experiment pipeline (grid search, multi-trial comparison)
- `--phase visualize` - Generate figures from saved experiment outputs
- `--phase full_eval` - **Comprehensive empirical evaluation suite** (recommended)
- `--phase full` - Complete legacy training and testing pipeline

### Comprehensive Evaluation Options
| Option | Description | Default |
|--------|-------------|---------|
| `--datasets` | Datasets to evaluate (`NSL-KDD`, `CIC-DDoS2019`, `both`) | `NSL-KDD` |
| `--num-trials` | Number of independent trials per dataset | `30` |
| `--no-baselines` | Disable baseline comparisons (RF, SVM, XGBoost, etc.) | `False` |
| `--no-ablations` | Disable ablation study suite | `False` |
| `--no-stats` | Disable statistical tests and effect sizes | `False` |
| `--sdn` | Enable Mininet + Ryu controller SDN evaluation | `False` |
| `--download-data` | Automatically download datasets if missing | `False` |
| `--output` | Output directory for metrics and figures | `results/evaluation` |

---

## Project Structure

```
knn-ddos-detection/
├── main.py                    # Main pipeline CLI entry-point
├── requirements.txt           # Python dependencies
├── config/
│   └── config.yaml           # Pipeline configuration
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Dataset ingestion (NSL-KDD, CIC-DDoS2019, synthetic)
│   ├── dataset_downloader.py # Dataset download & checksum validation
│   ├── feature_schema.py     # Explicit dataset schemas
│   ├── split_strategies.py   # Group/time-aware data splitting
│   ├── feature_engineering.py # Flow feature extraction
│   ├── preprocessor.py       # Data scaling and normalization
│   ├── degree_of_attack.py   # Attack degree metrics calculation
│   ├── knn_baseline.py       # Standard KNN classifier
│   ├── knn_improved.py       # Base paper Improved KNN
│   ├── knn_proposed.py       # Proposed KNN with bounded compression & dual voting
│   ├── classifier.py         # Unified model interface
│   ├── evaluator.py          # Classification and calibration metrics
│   ├── baselines.py          # Random Forest, SVM, XGBoost, LightGBM, MLP
│   ├── ablation.py           # Component ablation framework
│   ├── statistical_tests.py  # Statistical significance and effect size tests
│   ├── sdn_integration.py    # Mininet + Ryu SDN testbed integration
│   ├── full_evaluation.py    # Comprehensive evaluation orchestrator
│   ├── experiment_runner.py  # Multi-run experiment manager
│   ├── visualizer.py         # Matplotlib / Seaborn plotting utilities
│   ├── traffic_generator.py  # Synthetic benign & attack traffic generation
│   ├── sdn_simulator.py      # Discrete-event SDN controller simulation
│   └── utils.py              # Logging, timer, and I/O helpers
├── data/
│   └── raw/                  # Raw dataset archives
└── results/                  # Generated CSV metrics and figures
```

---

## Datasets

### NSL-KDD
- **Source**: [UNB NSL-KDD Dataset](https://www.unb.ca/cic/datasets/nsl.html)
- **Files**: `KDDTrain+.txt`, `KDDTest+.txt`
- **Auto-downloaded**: Yes (automatically fetched via `--download-data`)
- **Features**: 41 network flow features + label + difficulty score
- **Attack Categories**: DoS, Probe, R2L, U2R

### CIC-DDoS2019
- **Source**: [UNB CIC-DDoS2019 Dataset](https://www.unb.ca/cic/datasets/ddos-2019.html)
- **Files**: `Wednesday_workingHours_pcap_ISCX.csv`, `Thursday_workingHours_pcap_ISCX.csv`
- **Attack Types**: UDP flood, TCP-SYN, TCP-ACK, ICMP flood, HTTP flood, Slowloris

---

## Output Structure

Results are stored under `results/evaluation/` (or your `--output` path):
- `evaluation_complete.json` - Comprehensive aggregate metrics and statistical comparisons
- `nsl_kdd_results.json` / `cic_ddos2019_results.json` - Per-dataset performance breakdowns
- `plots/` - Publication-ready confusion matrices, ROC/PR curves, and calibration plots
- `metrics/` - Per-trial CSV exports for external analysis

---

## Ablation Study Matrix

The ablation suite evaluates the marginal contribution of each enhancement:

| Variant | Degree Weighting | Distance Compression | Adaptive K | Dual Voting |
|---------|:----------------:|:--------------------:|:----------:|:-----------:|
| `baseline_knn` | ✗ | ✗ | ✗ | ✗ |
| `base_paper` | ✓ | ✗ | ✗ | ✗ |
| `distance_compression_only` | ✓ | ✓ | ✗ | ✗ |
| `adaptive_k_only` | ✓ | ✗ | ✓ | ✗ |
| `dual_voting_only` | ✓ | ✓ | ✗ | ✓ |
| `distance_adaptive` | ✓ | ✓ | ✓ | ✗ |
| `distance_dual` | ✓ | ✓ | ✗ | ✓ |
| `full_proposed` | ✓ | ✓ | ✓ | ✓ |

---

## Author & Maintainer

- **Md. Iqbal Haider Khan**
  - GitHub: [@ihkokil](https://github.com/ihkokil)
  - Email: [ihkokil@gmail.com](mailto:ihkokil@gmail.com)

---

## Citation

If you use this codebase in your research, please cite:

```bibtex
@article{khan2026ddos,
  title={Adaptive DDoS Detection and Mitigation Framework Using Machine Learning in Software-Defined Networks},
  author={Khan, Md. Iqbal Haider},
  year={2026}
}

@inproceedings{base_paper,
  title={DDoS Attack Detection Method Based on Improved KNN With the Degree of DDoS Attack in Software-Defined Networks},
  author={Sharafaldin, I. and others},
  booktitle={International Conference on Cyber Security and Forensics},
  year={2019}
}
```

---

## License

Academic research use only. Datasets are subject to their respective licenses from the Canadian Institute for Cybersecurity (CIC).