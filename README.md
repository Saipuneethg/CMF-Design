# Spectral Metameric Fiber Analyzer (CMF-Design)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Academic%20Use-green.svg)](#license)
[![Optimization](https://img.shields.io/badge/Optimization-HiGHS%20%7C%20L--BFGS--B-orange.svg)](#mathematical-framework)
[![Documentation](https://img.shields.io/badge/Docs-Research%20Paper%20Reference-purple.svg)](documentation/Complete_Project_Technical_Documentation_Research_Paper.pdf)

> **Mathematical modeling, convex polytope analysis, and global optimization of Camera Spectral Sensitivity Functions (CSSFs) and Color Matching Functions (CMFs) for minimizing spectral metamerism.**

---

## 📌 Overview

In standard trichromatic digital imaging, optical sensors record only three scalar color coordinates ($R, G, B$) from continuous spectral radiance distributions. This dimension reduction from continuous functional Hilbert spaces to $\mathbb{R}^3$ creates an ill-posed inverse problem known as **spectral metamerism**: infinitely many distinct spectral reflectance curves produce identical RGB measurements under identical illumination.

The **Spectral Metameric Fiber Analyzer (SMFA)** provides a rigorous mathematical, algorithmic, and computational framework to bound, sample, quantify, and minimize this fundamental ambiguity. By representing the set of all metamers as a bounded 28-dimensional convex polytope in $\mathbb{R}^{31}$ (the *metameric fiber*), the system enables:

1. **Exact Boundary Bounding:** Dual-simplex linear programming via the modern HiGHS solver to extract theoretical minimum and maximum reflectance envelopes.
2. **Uniform Polytope Sampling:** Hit-and-Run Markov Chain Monte Carlo (MCMC) sampling constrained strictly to the null space $\ker(\mathbf{C})$.
3. **Spectral Ambiguity Quantification:** Rigorous evaluation of mean pairwise Euclidean spectral distance ($\Delta S_{\text{metameric}}$) across color gamuts.
4. **Hardware Sensitivity Optimization:** Constrained non-linear optimization (L-BFGS-B/SLSQP) of parameterized camera sensitivity curves, reducing metameric ambiguity by up to **42.8%** compared to commercial cameras and standard CIE observer functions.
5. **Solution Space Topology:** Demonstration that optimal camera sensitivity curves inhabit continuous connected low-ambiguity manifolds, providing tolerance for physical optical filter manufacturing.

---

## 📐 Mathematical Formulation

### Forward Optical Measurement Model
For continuous spectral reflectance $s(\lambda)$, illuminant SPD $I(\lambda)$, and camera spectral sensitivity $C_k(\lambda)$ ($k \in \{R, G, B\}$):

$$m_k = \int_{\lambda_{\min}}^{\lambda_{\max}} C_k(\lambda) \cdot I(\lambda) \cdot s(\lambda) \, d\lambda$$

Discretizing into $N = 31$ uniform wavelength intervals (400–700 nm, $\Delta\lambda = 10\text{ nm}$):

$$\mathbf{m} = \mathbf{W}\mathbf{s} = \mathbf{C} \, \text{diag}(\mathbf{I}) \, \mathbf{s} = [m_R, m_G, m_B]^T \in \mathbb{R}^3$$

where $\mathbf{s} \in [0, 1]^{31}$ and $\mathbf{C} \in \mathbb{R}^{3 \times 31}$.

### Metameric Fiber Convex Polytope
For an observed color $\mathbf{m} \in \mathbb{R}^3$, the metameric fiber $\mathcal{F}(\mathbf{m})$ is defined as:

$$\mathcal{F}(\mathbf{m}) = \left\{ \mathbf{s} \in \mathbb{R}^{31} : \mathbf{W}\mathbf{s} = \mathbf{m}, \quad \mathbf{0} \le \mathbf{s} \le \mathbf{1} \right\}$$

$\mathcal{F}(\mathbf{m})$ is a closed, bounded, convex polytope in $\mathbb{R}^{31}$ with intrinsic dimension $N - 3 = 28$.

### Null-Space Parameterization
Using an orthonormal basis $\mathbf{N} \in \mathbb{R}^{31 \times 28}$ for $\ker(\mathbf{W})$ and minimum-norm reconstruction $\mathbf{s}_0 = \mathbf{W}^+ \mathbf{m}$:

$$\mathbf{s}(\mathbf{z}) = \mathbf{s}_0 + \mathbf{N}\mathbf{z}, \quad \text{subject to } \mathbf{0} \le \mathbf{s}_0 + \mathbf{N}\mathbf{z} \le \mathbf{1}$$

### Spectral Ambiguity Metric
For $K$ uniform samples drawn from $\mathcal{F}(\mathbf{m})$:

$$\Delta S_{\text{metameric}} = \frac{1}{\binom{K}{2}} \sum_{1 \le i < j \le K} \|\mathbf{s}_i - \mathbf{s}_j\|_2$$

---

## 🏗️ System Architecture

```mermaid
graph TD
    A[Input Hyperspectral Cubes: 1,269 Munsell Patches] --> B[Spatial Pooling & Energy Conservation [0,1]]
    B --> C[Statistical Prior: PCA Eigendecomposition >99.2% Var]
    C --> D[Camera Model: W = C · diag(I_D65)]
    D --> E[HiGHS LP Solver: 62 Extreme Ray Bounding Envelopes]
    D --> F[Hit-and-Run MCMC: Uniform Sampling in 28D Null Space]
    F --> G[Ambiguity Quantification: Mean Pairwise L2 Distance ΔS]
    G --> H[Camera Optimizer: L-BFGS-B with Smoothness & Overlap Penalties]
    H --> I[Validation Suites: Illuminant D50/A, Sensor Noise SNRs, Curve Interpolation]
    I --> J[Outputs: Technical PDF Reports, Benchmark CSVs, REST API Dashboard]
```

---

## 📂 Repository Structure

```text
├── src/                                  # Core computational engine modules
│   ├── camera_model.py                   # Camera spectral sensitivity model & CSSF ingestion
│   ├── metamer_solver.py                 # HiGHS dual-simplex LP extreme ray solver
│   ├── hit_and_run.py                    # Hit-and-Run MCMC uniform polytope sampler
│   ├── ambiguity_metrics.py              # Pairwise Euclidean ambiguity & fiber diameter metrics
│   ├── camera_optimizer.py               # L-BFGS-B / SLSQP curve optimization engine
│   ├── fiber_geometry.py                 # Convex hull volume and PCA-projected polytope analysis
│   ├── gamut_checker.py                  # Linear programming camera gamut validation
│   ├── batch_analyzer.py                 # Multi-color batch evaluation pipeline
│   ├── chromaticity_analyzer.py          # CIE xy chromaticity and CIEDE2000 calculations
│   └── app.py                            # Interactive Flask REST API & Web Dashboard
│
├── scripts/                              # Execution scripts for experiments and benchmarks
│   ├── run_experiment_4_initialization_independence.py  # 13-run multi-initialization evaluation
│   ├── run_experiment_5_curve_interpolation.py           # Geodesic curve interpolation pipeline
│   ├── run_illuminant_noise_robustness.py                # Experiment 2 & 3 robustness sweeps
│   ├── run_scientific_validation_and_improvements.py    # Statistical significance verification
│   └── evaluate_all_cameras.py                           # Benchmark against 30+ commercial cameras
│
├── documentation/                        # Comprehensive technical documentation & monograph
│   └── Complete_Project_Technical_Documentation_Research_Paper.pdf
│
├── camera comparisions/                  # Comparative plots, rankings, and audit reports
│   └── plots/                            # High-resolution benchmark figures
│
├── Advanced_Global_Optimized_CMF/        # Authoritative global optimized CMF parameters & analysis
│   └── Chromaticity_Ambiguity_Analysis/  # Ambiguity distributions across chromaticity gamuts
│
├── Dataset/                              # Colorimetric reference datasets & CIE CMF tables
├── Spectral/                             # Munsell hyperspectral reflectance patches (128x128x31)
└── tests/                                # Automated unit test suites (LP bounds, MCMC, API)
```

---

## 📊 Key Results & Quantitative Benchmarks

| Camera Model / Sensor | Sensor Type | Mean Ambiguity ($\Delta S$) | Peak Wavelengths $[B, G, R]$ | Improvement vs. Baseline |
| :--- | :--- | :--- | :--- | :--- |
| **Canon 1D Mark III** | Commercial DSLR | $0.1684 \pm 0.038$ | $[450, 530, 600]\text{ nm}$ | Baseline ($0.0\%$) |
| **Nikon D200** | Commercial DSLR | $0.1592 \pm 0.035$ | $[460, 540, 610]\text{ nm}$ | $+5.5\%$ |
| **Sony ICX285** | Industrial CCD | $0.1541 \pm 0.033$ | $[460, 530, 620]\text{ nm}$ | $+8.5\%$ |
| **CIE 1931 $2^\circ$ CMF** | Standard Human Observer | $0.1482 \pm 0.031$ | $[445, 555, 600]\text{ nm}$ | $+12.0\%$ |
| **CIE 1964 $10^\circ$ CMF** | Standard Human Observer | $0.1415 \pm 0.029$ | $[440, 545, 595]\text{ nm}$ | $+16.0\%$ |
| **SMFA Global Optimized Model** | Optimized CSSF | **$0.0821 \pm 0.014$** | $[470, 545, 635]\text{ nm}$ | **$+51.2\%$ Improvement** |

### Robustness Highlights
* **Illuminant Invariance (Exp 2):** Under CIE Illuminant A (warm tungsten), commercial cameras suffer $45\%-60\%$ ambiguity surges, while the optimized sensor exhibits only an **$8.4\%$** variation.
* **Sensor Noise Invariance (Exp 3):** Condition number $\kappa(\mathbf{C}) = 4.12$ prevents noise amplification under Poisson shot and Gaussian read noise across 20–60 dB SNR.
* **Continuous Valley Topology (Exp 5):** Pairwise geodesic interpolation across 13 independent optimization runs confirms that **$92.3\%$** of interpolated curves remain within $5\%$ of optimal ambiguity, proving a connected near-optimal valley.

---

## 🚀 Quick Start

### 1. Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/Saipuneethg/CMF-Design.git
cd CMF-Design
pip install -r requirements.txt
```

### 2. Run Ambiguity Analysis on a Target Color
```python
from src.camera_model import CameraModel
from src.metamer_solver import MetamerSolver
from src.hit_and_run import HitAndRunSampler
from src.ambiguity_metrics import AmbiguityMetrics

# Initialize camera model
camera = CameraModel('PointGrey')

# Compute extreme ray bounds using HiGHS dual-simplex LP
solver = MetamerSolver()
bounds = solver.solve_bounds(camera.sensitivity_matrix, rgb_target=[0.45, 0.52, 0.38])

# Sample interior metameric fiber via Hit-and-Run MCMC
sampler = HitAndRunSampler()
samples = sampler.sample(camera.sensitivity_matrix, rgb_target=[0.45, 0.52, 0.38], num_samples=500)

# Compute pairwise spectral ambiguity metric
ambiguity = AmbiguityMetrics.compute(samples)
print(f"Mean Pairwise Ambiguity (ΔS): {ambiguity.mean_distance:.4f}")
```

### 3. Launch Interactive Web Dashboard
```bash
python src/app.py
```
Navigate to `http://localhost:5000` to interactively select RGB colors, visualize 3D PCA metameric fibers, and evaluate camera sensitivity profiles.

---

## 📄 Documentation

For full mathematical proofs, algorithmic pseudocode, asymptotic complexity analysis, and experimental methodology, refer to the technical monograph:
* [`documentation/Complete_Project_Technical_Documentation_Research_Paper.pdf`](documentation/Complete_Project_Technical_Documentation_Research_Paper.pdf)

---

## 📜 Citation

If you use this project or framework in your research, please cite:

```bibtex
@article{smfa2026cmf,
  title={Spectral Metameric Fiber Analyzer: Convex Polytope Geometry and Sensitivity Optimization for Minimizing Metameric Ambiguity},
  author={Saipuneeth G},
  year={2026},
  journal={Technical Reference & Monograph},
  url={https://github.com/Saipuneethg/CMF-Design}
}
```

---

## 📄 License
This project is released for academic, research, and educational purposes.
