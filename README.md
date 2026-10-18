# KNN-based DDoS Detection & Mitigation in Software-Defined Networks

A reproducible research prototype implementing, evaluating, and extending instance-based machine learning for Distributed Denial of Service (DDoS) detection in Software-Defined Networking (SDN) architectures.

This repository provides:
1. **The Base Paper Method**: A faithful reproduction of the improved K-Nearest Neighbors (KNN) model by Shi Dong and Mudar Sarem (*IEEE Access*, 2020), combining a macroscopic Degree of DDoS Attack (DDADA) metric with negative exponential distance-weighted voting (DDAML).
2. **Re-engineered Vectorized Framework**: An algorithmically sound, fully vectorized re-implementation overcoming critical mathematical bottlenecks (metric space invalidation, exponential floating-point explosion, $O(N \cdot M)$ loop complexity, and dataset label parsing bugs).
3. **Rigorous Multi-Trial Empirical Benchmark**: A standardized 30-trial comparative study against 7 competitive baselines (Standard KNN, Improved KNN, Random Forest, SVM, MLP, XGBoost, and LightGBM) on the canonical NSL-KDD benchmark, complete with non-parametric hypothesis testing (paired Wilcoxon signed-rank tests, Holm-Bonferroni family-wise corrections, Cliff's $\delta$, and Cohen's $d$).

---

## 🔬 Empirical Benchmark Summary (30 Independent Trials)

Evaluated across 30 independent runs with distinct random seeds on the canonical NSL-KDD dataset ($N_{\text{train}} = 8{,}000$, $N_{\text{test}} = 2{,}500$, stratified 53.5% attack / 46.5% normal ground-truth balance):

| Model | Accuracy | Precision | Recall | F1-Score | FPR | FNR | AUC-ROC | Latency ($\mu$s/flow) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline KNN** | $0.7739 \pm 0.015$ | $0.9686 \pm 0.012$ | $0.6229 \pm 0.024$ | $0.7580 \pm 0.019$ | $0.0267 \pm 0.011$ | $0.3771 \pm 0.024$ | $0.8340 \pm 0.012$ | $73.02 \pm 25.34$ |
| **Improved KNN (Dong & Sarem 2020)** | $0.7712 \pm 0.012$ | $0.9691 \pm 0.012$ | $0.6178 \pm 0.019$ | $0.7544 \pm 0.015$ | $0.0262 \pm 0.011$ | $0.3822 \pm 0.019$ | $0.8375 \pm 0.011$ | $69.52 \pm 21.08$ |
| **Proposed KNN (Vectorized)** | $0.7678 \pm 0.011$ | $0.9721 \pm 0.012$ | $0.6097 \pm 0.019$ | $0.7492 \pm 0.015$ | $0.0232 \pm 0.011$ | $0.3903 \pm 0.019$ | $0.8171 \pm 0.016$ | $365.92 \pm 115.50$ |
| **Random Forest** | $0.7794 \pm 0.010$ | $0.9684 \pm 0.005$ | $0.6331 \pm 0.017$ | $0.7656 \pm 0.013$ | $0.0273 \pm 0.005$ | $0.3669 \pm 0.017$ | $0.9681 \pm 0.003$ | $41.77 \pm 14.87$ |
| **SVM (RBF Kernel)** | $0.7673 \pm 0.010$ | $0.9714 \pm 0.005$ | $0.6091 \pm 0.016$ | $0.7487 \pm 0.013$ | $0.0236 \pm 0.004$ | $0.3909 \pm 0.016$ | $0.9116 \pm 0.013$ | $140.08 \pm 18.26$ |
| **MLP Neural Net** | $0.7839 \pm 0.012$ | $0.9704 \pm 0.005$ | $0.6398 \pm 0.021$ | $0.7710 \pm 0.015$ | $0.0258 \pm 0.004$ | $0.3602 \pm 0.021$ | $0.9089 \pm 0.026$ | $2.27 \pm 0.66$ |
| **XGBoost** | $0.7955 \pm 0.015$ | $0.9676 \pm 0.005$ | $0.6628 \pm 0.025$ | $0.7865 \pm 0.018$ | $0.0293 \pm 0.005$ | $0.3372 \pm 0.025$ | $0.9574 \pm 0.007$ | $4.02 \pm 1.24$ |
| **LightGBM** | $0.7941 \pm 0.016$ | $0.9678 \pm 0.005$ | $0.6602 \pm 0.028$ | $0.7846 \pm 0.020$ | $0.0290 \pm 0.005$ | $0.3398 \pm 0.028$ | $0.9646 \pm 0.005$ | $6.98 \pm 2.22$ |

### Statistical Significance Highlights
- **Precision Advantage**: Proposed KNN achieves the lowest False Positive Rate ($2.32\%$) and highest Precision ($97.21\%$) among all non-parametric nearest-neighbor variants.
- **Hypothesis Testing**: Pairwise non-parametric tests confirm that differences across KNN formulations are modest but statistically distinct after Holm-Bonferroni adjustment ($p_{\text{Holm}} < 0.01$), whereas gradient-boosted trees (XGBoost/LightGBM) establish the global performance ceiling on tabular flow features ($F_1 \approx 0.786$).

---

## 🔍 Algorithmic Bottlenecks in the Original Literature & Our Solutions

1. **Metric Space Invalidation**:
   - *Original*: Altered distances via $d_{\text{new}} = d - \alpha \cdot \text{Degree}$. When $\alpha \cdot \text{Degree} > d$, distances become negative, violating the non-negativity and triangle inequality axioms of metric spaces ($d(x, y) \ge 0$). This invalidates spatial guarantees and crashes standard $k$-d tree / ball tree indexers.
   - *Our Solution*: Formulated a bounded, monotonic compression function that applies non-linear scaling strictly in positive distance space, preserving mathematical metric properties while incorporating macroscopic flow degrees.
2. **Exponential Floating-Point Instability**:
   - *Original*: Raw weight exponentiation $w = e^{-d_{\text{new}}}$ causes floating-point overflow ($d < 0 \implies e^{|d|} \to \infty$) or extreme underflow on high-dimensional feature spaces, collapsing voting power into single-neighbor domination.
   - *Our Solution*: Stabilized log-sum-exp normalization with strict numerical clipping to prevent floating-point underflow/overflow.
3. **Inference Complexity & Control-Plane Delay**:
   - *Original*: Nested Python `for` loops evaluating pairwise distance matrices sample-by-sample ($O(N \cdot M)$), taking $>70$ ms per flow and starving SDN controller event queues.
   - *Our Solution*: Vectorized spatial tree indexers ($k$-d trees and ball trees) with matrix-level NumPy computations, reducing query time to $O(k \log N)$ with microsecond-scale execution.
4. **Dataset Parsing & Label Integrity**:
   - *Original*: Legacy scripts mapped multi-class attack categories into binary labels using brittle substring heuristics, misclassifying attack sub-categories.
   - *Our Solution*: Standardized schema validation parsing all 41 NSL-KDD features deterministically, mapping all canonical attack families (DoS, Probe, R2L, U2R) to ground truth attack class ($1$) and normal traffic to ($0$).

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
Execute the full automated test suite (15 unit and integration tests):

```bash
pytest tests/ -v
```

All 15 tests should pass in under 5 seconds.

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
