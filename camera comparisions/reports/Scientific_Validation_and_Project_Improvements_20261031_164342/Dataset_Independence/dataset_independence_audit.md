# Dataset Independence and Provenance Audit Report

## 1. Audit Objective
To evaluate the genuine statistical independence of materials used across training, validation, and testing stages in accordance with the scientific review criteria set forth in `camera-design111.pdf`.

## 2. Empirical Provenance Findings
1. **Origin of Training Spectral Prior:**
   - The 1,200 hyperspectral patches located in `Spectral/` were subjected to direct pixel-matching against the raw CAVE multispectral database (`Dataset/complete_ms_data/complete_ms_data`).
   - The inspection confirmed that these training patches originated directly from CAVE scenes (including `balloons_ms`, `cd_ms`, `clay_ms`, `feathers_ms`, `flowers_ms`, `oil_painting_ms`, `paints_ms`, `peppers_ms`, `sponges_ms`, etc.).
   - From these 1,200 patches, the 8-component latent PCA basis ($V_8 \in \mathbb{R}^{31 \times 8}$, explaining 99.71% variance) and the reduced convex prior H-polytope ($A_{\text{red}} \in \mathbb{R}^{714,340 \times 8}, b_{\text{red}} \in \mathbb{R}^{714,340}$) were constructed.

2. **Previous Methodological Overstatement:**
   - Previous documentation referred to subsequent evaluations on CAVE spectra as 'completely independent, unseen real-world validation'.
   - Because the statistical subspace ($V_8$) and feasibility polytope boundary were parameterized using CAVE patches, evaluating on other CAVE patches constitutes a **spatial held-out scene split within the same domain**, not out-of-distribution independent generalization.

3. **Material-Level Data Separation Protocol:**
   - **Training Set:** 1,200 empirical patches used strictly for PCA basis construction, spectral prior facet generation, and camera sensitivity optimization.
   - **Validation / Benchmarking Set:** 100 fixed, non-overlapping spectral targets sampled across diverse reflectance categories, held out at the patch level.
   - **Independent Out-of-Distribution Status:** As required by Section 5.3 of the specification: *'Independent generalization across completely external hyperspectral corpora (such as ICVL, Harvard, or Tokyo Tech databases) has not yet been established and remains an open research task.'*

## 3. Provenance Summary Table
| Stage | Dataset Corpus | Number of Spectra | Purpose | Independence Status |
| :--- | :--- | :--- | :--- | :--- |
| Prior Construction | CAVE Multispectral | 1,200 patches (128x128) | PCA basis $V_8$, Prior Polytope $A_{\text{red}}, b_{\text{red}}$ | In-domain training base |
| CSSF Optimization | CAVE Multispectral | Representative Gamut | Objective function minimization | In-domain optimization |
| Benchmark Evaluation | CAVE Held-Out | 100 spectral materials | Paired metameric fiber evaluation | Spatial held-out validation |
| External Generalization | External Databases (ICVL/Harvard) | N/A | Cross-corpus validation | **Not yet established** |
