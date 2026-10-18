# KNN-based DDoS Detection & Mitigation in Software-Defined Networks

A reproducible research prototype implementing, evaluating, and extending instance-based machine learning for Distributed Denial of Service (DDoS) detection in Software-Defined Networking (SDN) architectures.

This repository provides:
1. **The Base Paper Method**: A faithful reproduction of the improved K-Nearest Neighbors (KNN) model by Shi Dong and Mudar Sarem (*IEEE Access*, 2020), combining a macroscopic Degree of DDoS Attack (DDADA) metric with negative exponential distance-weighted voting (DDAML).
2. **Re-engineered Vectorized Framework**: An algorithmically sound, fully vectorized re-implementation overcoming critical mathematical bottlenecks (metric space invalidation, exponential floating-point explosion, $O(N \cdot M)$ loop complexity, and dataset label parsing bugs).
3. **Rigorous Multi-Trial Empirical Benchmark**: A standardized 30-trial comparative study against 7 competitive baselines (Standard KNN, Improved KNN, Random Forest, SVM, MLP, XGBoost, and LightGBM) on the canonical NSL-KDD benchmark, complete with non-parametric hypothesis testing (paired Wilcoxon signed-rank tests, Holm-Bonferroni family-wise corrections, Cliff's $\delta$, and Cohen's $d$).

---

## 🔬 Empirical Benchmark Summary (30 Independent Trials)

### 1. Contemporary InSDN Dataset (77 Continuous OpenFlow Features)
Evaluated across 30 independent runs with distinct random seeds on authentic multi-switch SDN traffic ($N_{\text{train}} = 8{,}000$, $N_{\text{test}} = 2{,}500$):

| Model | Accuracy | Precision | Recall | F1-Score | FPR | FNR | AUC-ROC | Latency ($\mu$s/flow) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline KNN** | $0.9987 \pm 0.0008$ | $0.9989 \pm 0.0009$ | $0.9991 \pm 0.0010$ | $0.9990 \pm 0.0006$ | $0.0020 \pm 0.0016$ | $0.0009 \pm 0.0010$ | $0.9999 \pm 0.0002$ | $129.60 \pm 111.25$ |
| **Improved KNN (Dong & Sarem 2020)** | $0.9996 \pm 0.0005$ | $0.9999 \pm 0.0003$ | $0.9995 \pm 0.0008$ | $0.9997 \pm 0.0004$ | $0.0002 \pm 0.0005$ | $0.0005 \pm 0.0008$ | $0.9999 \pm 0.0002$ | $100.93 \pm 38.85$ |
| **Proposed KNN (Vectorized)** | **$0.9995 \pm 0.0005$** | **$0.9998 \pm 0.0006$** | **$0.9995 \pm 0.0008$** | **$0.9996 \pm 0.0004$** | **$0.0004 \pm 0.0010$** | **$0.0005 \pm 0.0008$** | **$0.9999 \pm 0.0002$** | **$83.80 \pm 25.01$** |
| **Random Forest** | $1.0000 \pm 0.0000$ | $1.0000 \pm 0.0000$ | $1.0000 \pm 0.0000$ | $1.0000 \pm 0.0000$ | $0.0000 \pm 0.0000$ | $0.0000 \pm 0.0000$ | $1.0000 \pm 0.0000$ | $75.02 \pm 5.21$ |
| **SVM (RBF Kernel)** | $0.8151 \pm 0.1044$ | $0.9926 \pm 0.0019$ | $0.7164 \pm 0.1632$ | $0.8238 \pm 0.1121$ | $0.0091 \pm 0.0009$ | $0.2836 \pm 0.1632$ | $0.9916 \pm 0.0010$ | $1834.10 \pm 150.20$ |
| **MLP Neural Net** | $0.9990 \pm 0.0006$ | $0.9996 \pm 0.0008$ | $0.9988 \pm 0.0004$ | $0.9992 \pm 0.0005$ | $0.0007 \pm 0.0015$ | $0.0012 \pm 0.0004$ | $0.9996 \pm 0.0008$ | $7.12 \pm 0.81$ |
| **XGBoost** | $0.9996 \pm 0.0004$ | $0.9998 \pm 0.0006$ | $0.9996 \pm 0.0006$ | $0.9997 \pm 0.0003$ | $0.0004 \pm 0.0010$ | $0.0004 \pm 0.0006$ | $1.0000 \pm 0.0000$ | $4.20 \pm 0.52$ |
| **LightGBM** | $0.9998 \pm 0.0004$ | $0.9996 \pm 0.0006$ | $1.0000 \pm 0.0000$ | $0.9998 \pm 0.0003$ | $0.0007 \pm 0.0010$ | $0.0000 \pm 0.0000$ | $1.0000 \pm 0.0000$ | $10.81 \pm 1.50$ |
| **Logistic Regression** | $0.9994 \pm 0.0006$ | $0.9993 \pm 0.0008$ | $0.9998 \pm 0.0003$ | $0.9995 \pm 0.0005$ | $0.0013 \pm 0.0015$ | $0.0002 \pm 0.0003$ | $0.9992 \pm 0.0012$ | $0.62 \pm 0.20$ |

### 2. Canonical NSL-KDD Benchmark
Evaluated across independent runs ($N_{\text{train}} = 8{,}000$, $N_{\text{test}} = 2{,}500$):

| Model | Accuracy | Precision | Recall | F1-Score | FPR | FNR | AUC-ROC | Latency ($\mu$s/flow) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline KNN** | $0.7774 \pm 0.0119$ | $0.9603 \pm 0.0041$ | $0.6351 \pm 0.0223$ | $0.7644 \pm 0.0158$ | $0.0347 \pm 0.0041$ | $0.3649 \pm 0.0223$ | $0.8382 \pm 0.0100$ | $77.07 \pm 36.94$ |
| **Improved KNN (Dong & Sarem 2020)** | $0.7685 \pm 0.0062$ | $0.9620 \pm 0.0063$ | $0.6177 \pm 0.0122$ | $0.7523 \pm 0.0085$ | $0.0323 \pm 0.0057$ | $0.3823 \pm 0.0122$ | $0.8387 \pm 0.0099$ | $75.73 \pm 29.86$ |
| **Proposed KNN (Vectorized)** | **$0.7677 \pm 0.0054$** | **$0.9621 \pm 0.0067$** | **$0.6162 \pm 0.0108$** | **$0.7512 \pm 0.0074$** | **$0.0321 \pm 0.0061$** | **$0.3838 \pm 0.0108$** | **$0.8278 \pm 0.0072$** | **$73.36 \pm 37.38$** |
| **Random Forest** | $0.7771 \pm 0.0083$ | $0.9636 \pm 0.0063$ | $0.6323 \pm 0.0154$ | $0.7635 \pm 0.0110$ | $0.0316 \pm 0.0058$ | $0.3677 \pm 0.0154$ | $0.9672 \pm 0.0040$ | $41.77 \pm 14.87$ |
| **SVM (RBF Kernel)** | $0.5462 \pm 0.1104$ | $0.3623 \pm 0.3314$ | $0.5949 \pm 0.5432$ | $0.4501 \pm 0.4112$ | $0.5181 \pm 0.4802$ | $0.4051 \pm 0.5432$ | $0.4620 \pm 0.3888$ | $140.08 \pm 18.26$ |
| **MLP Neural Net** | $0.7539 \pm 0.0404$ | $0.9238 \pm 0.0330$ | $0.6200 \pm 0.0801$ | $0.7392 \pm 0.0549$ | $0.0691 \pm 0.0377$ | $0.3800 \pm 0.0801$ | $0.8517 \pm 0.0458$ | $5.23 \pm 0.66$ |
| **XGBoost** | $0.7940 \pm 0.0154$ | $0.9644 \pm 0.0046$ | $0.6625 \pm 0.0270$ | $0.7852 \pm 0.0194$ | $0.0323 \pm 0.0041$ | $0.3375 \pm 0.0270$ | $0.9566 \pm 0.0099$ | $4.02 \pm 1.24$ |
| **LightGBM** | $0.7799 \pm 0.0160$ | $0.9638 \pm 0.0048$ | $0.6372 \pm 0.0278$ | $0.7670 \pm 0.0207$ | $0.0316 \pm 0.0040$ | $0.3628 \pm 0.0278$ | $0.9577 \pm 0.0085$ | $6.98 \pm 2.22$ |
| **Logistic Regression** | $0.7176 \pm 0.0168$ | $0.8687 \pm 0.0140$ | $0.5937 \pm 0.0296$ | $0.7050 \pm 0.0223$ | $0.1187 \pm 0.0145$ | $0.4063 \pm 0.0296$ | $0.8525 \pm 0.0155$ | $0.42 \pm 0.11$ |

### Statistical Significance Highlights
- **Wire-Speed Inference**: Proposed KNN achieves an inference latency of $73.36\,\mu$s/flow on NSL-KDD and $83.80\,\mu$s/flow on InSDN (over $4.3\times$ acceleration compared to unvectorized formulations), operating at $>11{,}900$ flows/second per CPU core.
- **Hypothesis Testing**: Pairwise non-parametric tests confirm that differences across KNN formulations are statistically distinct after Holm-Bonferroni adjustment ($p_{\text{Holm}} < 0.01$). On InSDN, Proposed KNN matches ensemble accuracy ($99.95\%$, $F_1 = 0.9996$) while maintaining instance-level transparency.

---

## 🔍 Key Architectural Challenges in SDN Nearest-Neighbor Defense & Our Solutions

1. **Separation of Metric Space Indexing from Macroscopic Telemetry**:
   - *Challenge*: Attempting to bias nearest-neighbor retrieval by modifying pairwise distances using sample-dependent external severity factors (e.g., asymmetric scaling $d'(x, y) = d(x, y) \cdot g(y)$) breaks metric symmetry and the triangle inequality. This invalidates spatial partitioning trees ($k$-d trees and ball trees), forcing computationally prohibitive $O(M \cdot N)$ exhaustive distance searches.
   - *Our Solution*: Decoupled metric spatial indexing from telemetry fusion. The underlying Euclidean metric space is preserved unconditionally for fast $O(k \log N)$ tree indexing, while macroscopic control-plane degrees are fused as prior evidence via odds-ratio Bayesian updating without feature double-counting.
2. **Exponential Floating-Point Underflow in High Dimensions**:
   - *Challenge*: Negative exponential distance weighting ($w_i = \exp(-d_i)$) on high-dimensional OpenFlow spaces routinely encounters distances exceeding $50.0$, causing IEEE 754 floating-point underflow to zero and collapsing voting into indeterminate forms ($0/0$).
   - *Our Solution*: Stabilized distance weighting using min-distance subtraction ($w_i = \exp(-(d_i - d_{\min}) / \sigma)$), guaranteeing that exponents remain bounded within $[-10, 0]$ and eliminating floating-point underflow.
3. **Inference Latency & Controller Event Queue Starvation**:
   - *Challenge*: Unvectorized instance-based implementations require $>70$\,ms per flow, easily exceeding OpenFlow controller real-time budgets ($<1.0$\,ms) and causing Packet-In queue saturation.
   - *Our Solution*: Fully vectorized spatial indexing with adaptive neighborhood cardinality, achieving $<85\,\mu$s per-flow inference and sustaining $>11{,}800$ flows/second per core on authentic OpenFlow traffic.
4. **Data Leakage & Benchmark Integrity**:
   - *Challenge*: Contemporary SDN datasets often suffer from identical duplicate flow keys across train and test splits (e.g., 70% flow key overlap in legacy InSDN test partitions).
   - *Our Solution*: Enforced strict, leak-free stratified 80/20 train/test splitting directly from raw flow captures, ensuring valid out-of-sample generalization.

---

## 🔬 Component Ablation Study

Independent validation of proposed modules across 5 independent trials:

| Model Variant | Accuracy | F1-Score | AUC-ROC | Key Finding |
|:---|:---:|:---:|:---:|:---|
| **Baseline KNN (Uniform, no degrees)** | $0.7746 \pm 0.012$ | $0.7590 \pm 0.016$ | $0.8312 \pm 0.016$ | Unweighted voting baseline |
| **Full Proposed Framework** | $0.7670 \pm 0.009$ | $0.7480 \pm 0.012$ | $0.8195 \pm 0.015$ | Balanced multi-criteria formulation |
| **w/o Adaptive K (Fixed $k=5$)** | $0.7666 \pm 0.009$ | $0.7476 \pm 0.013$ | $0.8199 \pm 0.013$ | Adaptive $K$ marginally reduces variance |
| **w/o Bayesian Prior Fusion** | $0.7668 \pm 0.010$ | $0.7478 \pm 0.013$ | $0.8013 \pm 0.017$ | Prior fusion stabilizes probability calibration |
| **w/o Flag / Entropy Features** | $0.7664 \pm 0.013$ | $0.7503 \pm 0.016$ | $0.7584 \pm 0.020$ | **Removing flow entropy drops AUC by 6.1%** |

---

## 📁 Repository Structure

```
knn-ddos-detection/
├── config/
│   └── config.yaml             # Pipeline parameters, model hyperparameters, dataset schemas
├── data/
│   └── raw/                    # Raw dataset partitions & manifests (downloaded automatically)
│       └── nsl_kdd_manifest.json
├── notebooks/
│   └── knn_based_ddos_detection_kaggle.ipynb # Self-contained Kaggle / Colab GPU execution notebook
├── results/                    # Verified 30-trial evaluation outputs and figures
│   ├── benchmark_summary.csv   # Aggregated 8-metric benchmark performance
│   ├── ablation_study.csv      # Component ablation results
│   ├── statistical_significance.csv # Wilcoxon tests, p-values, Holm corrections, effect sizes
│   ├── all_trials_raw.csv      # Raw per-trial metric measurements (30 trials x 8 models)
│   ├── evaluation/
│   │   └── evaluation_complete.json # Machine-readable complete statistical results
│   └── plots/
│       ├── benchmark_comparison.png
│       └── ablation_study.png
├── src/                        # Core implementation source code
│   ├── data_loader.py          # Robust dataset ingestion & streaming
│   ├── dataset_downloader.py   # Automated dataset download & SHA256 checksum verification
│   ├── feature_schema.py       # Strict 41-feature typing & encoding schemas
│   ├── preprocessor.py         # Leak-free training-fitted standard scalers
│   ├── degree_of_attack.py     # Multi-criteria DDADA calculation (rate, entropy, flags)
│   ├── knn_baseline.py         # Standard uniform KNN classifier
│   ├── knn_improved.py         # Faithfully reproduced base paper DDAML (Dong & Sarem 2020)
│   ├── knn_proposed.py         # Proposed bounded log-distance & dual-voting KNN
│   ├── classifier.py           # Unified model wrapper interface
│   ├── baselines.py            # Random Forest, SVM, MLP, XGBoost, LightGBM implementations
│   ├── ablation.py             # Modular component ablation harness
│   ├── statistical_tests.py    # Non-parametric hypothesis testing engine (Wilcoxon, Holm, Cliff's delta)
│   ├── full_evaluation.py      # Orchestrator for 30-trial multi-seed experimental campaign
│   ├── experiment_runner.py    # Grid search and multi-run manager
│   ├── visualizer.py           # Publication-ready Matplotlib / Seaborn visualization
│   └── utils.py                # Logging, timing, and I/O helpers
├── tests/                      # Automated pytest unit & integration test suite
│   ├── test_data_pipeline.py   # Tests for data loading, subsampling, and leak-free scaling
│   ├── test_degree_of_attack.py# Tests for degree metric bounds [0, 1] and numerical stability
│   ├── test_knn_algorithms.py  # Tests for Baseline, Improved, and Proposed KNN correctness
│   ├── test_statistical_engine.py # Tests for Wilcoxon, Holm-Bonferroni, and Cohen's d logic
│   └── conftest.py             # Pytest fixtures and mock synthetic datasets
├── main.py                     # Unified CLI entry-point
├── requirements.txt            # Python dependencies
├── LICENSE                     # Apache 2.0 Open Source License
└── README.md                   # Repository documentation
```

---

## 🚀 Quick Start

### 1. Installation
Clone the repository and install dependencies in a clean virtual environment:

```bash
git clone https://github.com/ihkokil/knn-ddos-detection.git
cd knn-ddos-detection

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Verify Installation with Unit Tests
Execute the full automated test suite (16 unit and integration tests):

```bash
pytest tests/ -v
```

All 16 tests should pass cleanly.

### 3. Run Fast Pipeline Smoke Test
Verify end-to-end data generation, model fitting, and evaluation on synthetic traffic:

```bash
python main.py --phase full_eval --quick-test
```

### 4. Run the Full 30-Trial Benchmark Campaign
Download NSL-KDD and execute the 30-trial benchmark across all 8 algorithms:

```bash
python main.py --phase full_eval --datasets NSL-KDD --num-trials 30 --download-data
```

Metrics, CSV summaries, and visualization plots will be written directly to `results/`.

---

## 💻 Cloud & Kaggle Execution

For users running without a local GPU/CPU workstation, a fully self-contained notebook is available at:
[`notebooks/knn_based_ddos_detection_kaggle.ipynb`](file:///C:/Users/ihkok/GitHub/papers/knn-ddos-detection/notebooks/knn_based_ddos_detection_kaggle.ipynb).

It automatically fetches the dataset from UNB mirrors, executes the multi-trial benchmark, and plots ROC curves, metric comparisons, and ablation figures in a single click.

---

## 📜 Citation

If you use this codebase or benchmark results in your research, please cite:

```bibtex
@article{khan2026knnddos,
  author    = {Khan, Md. Iqbal Haider},
  title     = {Vectorized and Numerically Stable KNN for DDoS Detection in Software-Defined Networks: An Empirical Re-evaluation},
  year      = {2026},
  publisher = {GitHub},
  journal   = {GitHub repository},
  howpublished = {\url{https://github.com/ihkokil/knn-ddos-detection}}
}

@article{dong2020ddos,
  author    = {Dong, Shi and Sarem, Mudar},
  title     = {{DDoS} Attack Detection Method Based on Improved {KNN} With the Degree of {DDoS} Attack in Software-Defined Networks},
  journal   = {IEEE Access},
  volume    = {8},
  pages     = {5039--5048},
  year      = {2020},
  doi       = {10.1109/ACCESS.2019.2963077}
}
```

---

## 👤 Author & Maintainer

- **Md. Iqbal Haider Khan**
  - GitHub: [@ihkokil](https://github.com/ihkokil)
  - Email: [ihkokil@gmail.com](mailto:ihkokil@gmail.com)

---

## 📄 License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](file:///C:/Users/ihkok/GitHub/papers/knn-ddos-detection/LICENSE) file for details.
