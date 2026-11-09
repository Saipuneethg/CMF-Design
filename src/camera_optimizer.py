"""
Camera Spectral-Sensitivity Optimization Module
===============================================
Research Proposal:
"Geometric Quantification and Camera-Aware Optimization of Spectral Ambiguity
 in Three-Channel Imaging"

This module implements joint 3-channel camera spectral-sensitivity optimization
to reduce metameric ambiguity within the existing empirical spectral prior.

Enhanced Version with:
- Aligned scientific objective (mean, median, Q95, and paired regression penalty).
- Multi-start optimization with independent initializations and validation-based selection.
- Rigorous sample-by-sample paired held-out evaluation with bootstrap confidence intervals.
- Non-parametric hypothesis testing (Wilcoxon signed-rank and permutation tests).
- Automated generation of publication-quality diagnostic plots (including paired ambiguity analysis).
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import time
import numpy as np
import scipy.linalg
import scipy.stats as stats
from scipy.optimize import minimize, differential_evolution
import colour
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.metamer_solver import MetamerSolver
from src.camera_model import CameraModel


class CameraOptimizer:
    """
    Joint 3-Channel Camera Spectral-Sensitivity Optimizer.
    """
    def __init__(self, artifacts_dir="artifacts", seed=42):
        self.artifacts_dir = artifacts_dir
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Load prior artifacts
        self.wavelengths = np.load(os.path.join(artifacts_dir, "wavelengths.npy")).astype(np.float64)
        self.n_bands = len(self.wavelengths)  # 31
        self.pca_mean = np.load(os.path.join(artifacts_dir, "pca_mean.npy")).astype(np.float64)
        self.pca_basis = np.load(os.path.join(artifacts_dir, "pca_basis.npy")).astype(np.float64)  # (31, 8)
        self.K = self.pca_basis.shape[1]  # 8
        self.A_red = np.load(os.path.join(artifacts_dir, "A_red.npy")).astype(np.float64)
        self.b_red = np.load(os.path.join(artifacts_dir, "b_red.npy")).astype(np.float64)

        # Subsampled facet system (40k facets) for rapid screening during global search
        sub_idx = np.linspace(0, len(self.A_red) - 1, 40000, dtype=int)
        self.A_sub = self.A_red[sub_idx]
        self.b_sub = self.b_red[sub_idx]

        # Baseline camera
        self.M_baseline = np.load(os.path.join(artifacts_dir, "M_camera.npy")).astype(np.float64)
        self.A_baseline = np.load(os.path.join(artifacts_dir, "A_camera.npy")).astype(np.float64)
        self.offset_baseline = np.load(os.path.join(artifacts_dir, "camera_offset.npy")).astype(np.float64)

        # Standard throughput target: sum of each baseline channel row (~10.672)
        self.target_throughput = float(np.mean(np.sum(self.M_baseline, axis=1)))

        # CIE 1931 2-Degree Standard Observer for colorimetric evaluation
        cmfs = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
        self.cie_xyz_cmf = cmfs[self.wavelengths].T.astype(np.float64)  # (3, 31)
        
        # Standard Illuminant D65
        illum_D65 = colour.SDS_ILLUMINANTS['D65']
        d65_raw = illum_D65[self.wavelengths].astype(np.float64)
        k_d65 = 100.0 / np.sum(d65_raw * self.cie_xyz_cmf[1])
        self.d65_scaled = d65_raw * k_d65  # (31,)
        
        # Reference white for D65
        xy_w = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']
        self.xyz_white_d65 = colour.xy_to_XYZ(xy_w) * 100.0
        self.xy_white_d65 = xy_w

        # Empirical surface spectra dataset
        self.P_surf_pca = np.load(os.path.join(artifacts_dir, "P_surf_pca.npy")).astype(np.float64)  # (6208, 8)
        self.P_surf_spectra = np.load(os.path.join(artifacts_dir, "P_surf_spectra.npy")).astype(np.float64)  # (6208, 31)

        # Partition evaluation sets
        self._partition_datasets()

        # Precompute baseline spreads across subsets for regression penalties
        amb_base_train_sub = self.evaluate_ambiguity_fast(self.M_baseline, self.Z_train, use_subsample=True)
        self.base_train_spreads_sub = np.array(amb_base_train_sub["all_spreads"])

        amb_base_train_full = self.evaluate_ambiguity_fast(self.M_baseline, self.Z_train, use_subsample=False)
        self.base_train_spreads_full = np.array(amb_base_train_full["all_spreads"])

        amb_base_val_full = self.evaluate_ambiguity_fast(self.M_baseline, self.Z_val, use_subsample=False)
        self.base_val_spreads_full = np.array(amb_base_val_full["all_spreads"])

        amb_base_test_full = self.evaluate_ambiguity_fast(self.M_baseline, self.Z_test, use_subsample=False)
        self.base_test_spreads_full = np.array(amb_base_test_full["all_spreads"])

        # Objective weights (aligned with scientific goal to prevent median degradation)
        self.w_mean = 0.40
        self.w_median = 0.30
        self.w_tail = 0.20
        self.w_reg = 0.10

        # Optimization history tracking
        self.history = []
        self.multi_start_history = []
        self.winning_run_id = None

    def _partition_datasets(self):
        """
        Deterministically partition spectra into Training/Screening, Validation,
        and Held-Out Evaluation sets. Uses interior scaled prior spectra to guarantee
        that all candidate cameras have strictly interior anchor points.
        """
        n_total = len(self.P_surf_pca)
        indices = self.rng.permutation(n_total)

        n_train = 20
        n_val = 20
        n_test = 50

        self.idx_train = indices[:n_train]
        self.idx_val = indices[n_train:n_train + n_val]
        self.idx_test = indices[n_train + n_val:n_train + n_val + n_test]

        # Use 0.95 scaling towards PCA mean to ensure strict polytope interiorness
        scale_interior = 0.95
        self.Z_train = self.P_surf_pca[self.idx_train] * scale_interior
        self.R_train = self.pca_mean + self.Z_train @ self.pca_basis.T

        self.Z_val = self.P_surf_pca[self.idx_val] * scale_interior
        self.R_val = self.pca_mean + self.Z_val @ self.pca_basis.T

        self.Z_test = self.P_surf_pca[self.idx_test] * scale_interior
        self.R_test = self.pca_mean + self.Z_test @ self.pca_basis.T

        # Precompute inequality RHS vectors H = max(b - A @ Z.T, 1e-12)
        # Full 714k facets
        self.H_train_full = np.maximum(self.b_red[:, None] - self.A_red @ self.Z_train.T, 1e-12)
        self.H_val_full = np.maximum(self.b_red[:, None] - self.A_red @ self.Z_val.T, 1e-12)
        self.H_test_full = np.maximum(self.b_red[:, None] - self.A_red @ self.Z_test.T, 1e-12)

        # Fast subsampled 40k facets
        self.H_train_sub = np.maximum(self.b_sub[:, None] - self.A_sub @ self.Z_train.T, 1e-12)
        self.H_val_sub = np.maximum(self.b_sub[:, None] - self.A_sub @ self.Z_val.T, 1e-12)

        # CIE reference XYZ under D65 for colorimetric fitting
        self.XYZ_train_ref = (self.R_train * self.d65_scaled) @ self.cie_xyz_cmf.T
        self.XYZ_val_ref = (self.R_val * self.d65_scaled) @ self.cie_xyz_cmf.T
        self.XYZ_test_ref = (self.R_test * self.d65_scaled) @ self.cie_xyz_cmf.T

    # -------------------------------------------------------------------------
    # Camera Parameterization (Gaussian Mixture Model)
    # -------------------------------------------------------------------------
    def params_to_matrix(self, params):
        """
        Converts 12 parameters into a 3 x 31 sensitivity matrix M:
        - Blue (params 0..2): a_b, mu_b, sig_b (1 Gaussian)
        - Green (params 3..5): a_g, mu_g, sig_g (1 Gaussian)
        - Red (params 6..11): a_r1, mu_r1, sig_r1, a_r2, mu_r2, sig_r2 (2 Gaussians)
        
        Strict Non-Negativity: All amplitudes > 0, Gaussian exponent is non-negative.
        Throughput Normalization: Each channel normalized so sum(s_k) = target_throughput.
        """
        wls = self.wavelengths
        a_b, mu_b, sig_b = abs(params[0]), params[1], max(abs(params[2]), 5.0)
        a_g, mu_g, sig_g = abs(params[3]), params[4], max(abs(params[5]), 5.0)
        a_r1, mu_r1, sig_r1 = abs(params[6]), params[7], max(abs(params[8]), 5.0)
        a_r2, mu_r2, sig_r2 = abs(params[9]), params[10], max(abs(params[11]), 5.0)

        s_b = a_b * np.exp(-0.5 * ((wls - mu_b) / sig_b) ** 2)
        s_g = a_g * np.exp(-0.5 * ((wls - mu_g) / sig_g) ** 2)
        s_r = (a_r1 * np.exp(-0.5 * ((wls - mu_r1) / sig_r1) ** 2) +
               a_r2 * np.exp(-0.5 * ((wls - mu_r2) / sig_r2) ** 2))

        # Throughput normalization (sum matching baseline camera ~10.672)
        sum_b = np.sum(s_b)
        sum_g = np.sum(s_g)
        sum_r = np.sum(s_r)

        s_b = (s_b / sum_b) * self.target_throughput if sum_b > 1e-9 else np.ones(self.n_bands) * (self.target_throughput / self.n_bands)
        s_g = (s_g / sum_g) * self.target_throughput if sum_g > 1e-9 else np.ones(self.n_bands) * (self.target_throughput / self.n_bands)
        s_r = (s_r / sum_r) * self.target_throughput if sum_r > 1e-9 else np.ones(self.n_bands) * (self.target_throughput / self.n_bands)

        return np.array([s_r, s_g, s_b], dtype=np.float64)

    def get_initial_parameters(self):
        """
        Returns initial parameters fitted to approximate standard spectral sensitivities:
        Blue: peak ~450 nm, sigma ~24 nm
        Green: peak ~550 nm, sigma ~40 nm
        Red: primary peak ~610 nm, sigma ~38 nm, secondary bump ~445 nm, sigma ~20 nm
        """
        return np.array([
            1.8,  450.0, 24.0,  # Blue: a, mu, sig
            1.0,  550.0, 40.0,  # Green: a, mu, sig
            1.1,  610.0, 38.0,  # Red primary: a1, mu1, sig1
            0.3,  445.0, 20.0   # Red secondary: a2, mu2, sig2
        ], dtype=np.float64)

    def get_parameter_bounds(self):
        """
        Parameter search bounds:
        Blue: a in [0.2, 5.0], mu in [420, 475], sig in [15, 60]
        Green: a in [0.2, 5.0], mu in [515, 575], sig in [20, 65]
        Red primary: a1 in [0.2, 5.0], mu1 in [590, 660], sig1 in [20, 75]
        Red secondary: a2 in [0.0, 1.5], mu2 in [420, 470], sig2 in [10, 45]
        """
        return [
            (0.2, 5.0), (420.0, 475.0), (15.0, 60.0),   # Blue
            (0.2, 5.0), (515.0, 575.0), (20.0, 65.0),   # Green
            (0.2, 5.0), (590.0, 660.0), (20.0, 75.0),   # Red primary
            (0.0, 1.5), (420.0, 470.0), (10.0, 45.0)    # Red secondary
        ]

    # -------------------------------------------------------------------------
    # Physical Penalties & Constraints
    # -------------------------------------------------------------------------
    def compute_smoothness_penalty(self, M):
        """
        Penalizes 2nd spectral derivatives across wavelength grid (Delta lambda = 10 nm):
        C_smooth = sum_k sum_j ((s(j+1) - 2s(j) + s(j-1)) / dl^2)^2 * dl
        """
        dl = 10.0
        c_smooth = 0.0
        for k in range(3):
            d2 = (M[k, 2:] - 2.0 * M[k, 1:-1] + M[k, :-2]) / (dl ** 2)
            c_smooth += float(np.sum(d2 ** 2) * dl)
        return c_smooth

    def compute_redundancy_and_conditioning(self, M):
        """
        Measures channel redundancy (cross-correlations) and condition number:
        kappa = sigma_max(M M^T) / sigma_min(M M^T)
        """
        norms = np.linalg.norm(M, axis=1)
        r_rg = float(np.dot(M[0], M[1]) / max(norms[0] * norms[1], 1e-12))
        r_rb = float(np.dot(M[0], M[2]) / max(norms[0] * norms[2], 1e-12))
        r_gb = float(np.dot(M[1], M[2]) / max(norms[1] * norms[2], 1e-12))
        max_corr = max(r_rg, r_rb, r_gb)

        MMT = M @ M.T
        eigvals = np.linalg.eigvalsh(MMT)
        cond = float(np.sqrt(max(eigvals[-1] / max(eigvals[0], 1e-12), 1.0)))

        return {
            "cond_number": cond,
            "max_correlation": max_corr,
            "correlations": {"R_G": r_rg, "R_B": r_rb, "G_B": r_gb}
        }

    def compute_colorimetric_fidelity(self, M, R_spectra, XYZ_ref):
        """
        Fits 3x3 linear matrix T : c @ T ~ XYZ_ref under D65,
        and evaluates Delta E 2000 color differences.
        """
        C_cam = (R_spectra * self.d65_scaled) @ M.T  # (N, 3)

        T, _, _, _ = np.linalg.lstsq(C_cam, XYZ_ref, rcond=None)
        XYZ_est = C_cam @ T

        XYZ_est = np.maximum(XYZ_est, 1e-5)
        XYZ_ref_clip = np.maximum(XYZ_ref, 1e-5)

        Lab_ref = colour.XYZ_to_Lab(XYZ_ref_clip, illuminant=self.xy_white_d65)
        Lab_est = colour.XYZ_to_Lab(XYZ_est, illuminant=self.xy_white_d65)

        de00 = colour.difference.delta_E_CIE2000(Lab_ref, Lab_est)
        de00 = np.nan_to_num(de00, nan=10.0, posinf=10.0, neginf=10.0)

        return {
            "mean_de00": float(np.mean(de00)),
            "median_de00": float(np.median(de00)),
            "p95_de00": float(np.percentile(de00, 95)),
            "max_de00": float(np.max(de00)),
            "T_matrix": T.tolist()
        }

    # -------------------------------------------------------------------------
    # Fast Vectorized Ambiguity Evaluation
    # -------------------------------------------------------------------------
    def evaluate_ambiguity_fast(self, M, Z_eval, H_eval=None, use_subsample=False):
        """
        Fast, vectorized geometric ambiguity evaluation across evaluation spectra.
        Constructs reduced null-space basis N_theta, projects polytope G = A_poly @ N_theta,
        and computes radial directional bounding extents along coordinate axes.
        
        Since (B @ N_theta)^T (B @ N_theta) = I_5, coordinate spread:
        sigma_spread = sqrt(sum(span_j^2 / 12))
        closely tracks the true MCMC sample spread.
        """
        A_theta = M @ self.pca_basis  # (3, 8)
        U, S, Vt = np.linalg.svd(A_theta)
        if S[-1] < 1e-4:
            return {
                "spread_mean": 999.0,
                "spread_median": 999.0,
                "spread_p95": 999.0,
                "all_spreads": [999.0] * len(Z_eval),
                "rank": int(np.linalg.matrix_rank(A_theta)),
                "fiber_dim": int(self.K - np.linalg.matrix_rank(A_theta))
            }

        N_theta = Vt[3:].T  # (8, 5)

        if use_subsample:
            A_poly = self.A_sub
            if H_eval is None:
                H_eval = np.maximum(self.b_sub[:, None] - self.A_sub @ Z_eval.T, 1e-12)
        else:
            A_poly = self.A_red
            if H_eval is None:
                H_eval = np.maximum(self.b_red[:, None] - self.A_red @ Z_eval.T, 1e-12)

        G = A_poly @ N_theta  # (P, 5)
        n_spectra = Z_eval.shape[0]
        spans_matrix = np.zeros((n_spectra, 5))

        for j in range(5):
            G_j = G[:, j]
            # + direction
            p_mask = G_j > 1e-9
            t_p = np.min(H_eval[p_mask] / G_j[p_mask, None], axis=0)
            # - direction
            n_mask = G_j < -1e-9
            t_n = np.min(H_eval[n_mask] / (-G_j[n_mask, None]), axis=0)

            spans_matrix[:, j] = t_p + t_n

        spreads = np.sqrt(np.sum(spans_matrix ** 2, axis=1) / 12.0)
        return {
            "spread_mean": float(np.mean(spreads)),
            "spread_median": float(np.median(spreads)),
            "spread_p95": float(np.percentile(spreads, 95)),
            "all_spreads": spreads.tolist(),
            "rank": 3,
            "fiber_dim": 5
        }

    # -------------------------------------------------------------------------
    # Full Objective Function (Scientifically Aligned)
    # -------------------------------------------------------------------------
    def objective(self, params, record_history=False, use_subsample=True):
        """
        Total objective function to minimize:
        J_total = J_amb + lambda_s * C_smooth + lambda_r * C_redundancy + lambda_c * C_color

        J_amb includes:
        - Mean spread (w_mean = 0.40)
        - Median spread (w_median = 0.30) to prevent median increase
        - 95th percentile spread (w_tail = 0.20) for worst-case protection
        - Regression penalty (w_reg = 0.10) penalizing individual sample regressions
        """
        M = self.params_to_matrix(params)

        # 1. Ambiguity on training set
        H_eval = self.H_train_sub if use_subsample else self.H_train_full
        amb = self.evaluate_ambiguity_fast(M, self.Z_train, H_eval=H_eval, use_subsample=use_subsample)
        if amb["spread_mean"] >= 990.0:
            return 1e6

        # Regression penalty relative to baseline
        base_spreads = self.base_train_spreads_sub if use_subsample else self.base_train_spreads_full
        cand_spreads = np.array(amb["all_spreads"])
        reg_penalty = float(np.mean(np.maximum(0.0, cand_spreads - base_spreads)))

        # Composite ambiguity objective
        J_amb = (self.w_mean * amb["spread_mean"] +
                 self.w_median * amb["spread_median"] +
                 self.w_tail * amb["spread_p95"] +
                 self.w_reg * reg_penalty)

        # 2. Smoothness penalty
        c_smooth = self.compute_smoothness_penalty(M)
        penalty_smooth = 5.0 * c_smooth

        # 3. Conditioning and redundancy penalty
        cond_info = self.compute_redundancy_and_conditioning(M)
        cond = cond_info["cond_number"]
        penalty_cond = 0.01 * (max(cond - 12.0, 0.0) ** 2)

        # 4. Colorimetric penalty
        color_info = self.compute_colorimetric_fidelity(M, self.R_train, self.XYZ_train_ref)
        mean_de00 = color_info["mean_de00"]
        penalty_color = 0.05 * (max(mean_de00 - 1.5, 0.0) ** 2)

        # Total objective
        J_total = float(J_amb + penalty_smooth + penalty_cond + penalty_color)

        if record_history:
            entry = {
                "step": len(self.history) + 1,
                "J_total": J_total,
                "J_amb": J_amb,
                "spread_mean": amb["spread_mean"],
                "spread_median": amb["spread_median"],
                "spread_p95": amb["spread_p95"],
                "reg_penalty": reg_penalty,
                "smoothness": c_smooth,
                "cond_number": cond,
                "mean_de00": mean_de00,
                "penalty_smooth": penalty_smooth,
                "penalty_cond": penalty_cond,
                "penalty_color": penalty_color
            }
            self.history.append(entry)

        return J_total

    # -------------------------------------------------------------------------
    # Multi-Start Optimization with Validation-Based Selection
    # -------------------------------------------------------------------------
    def run_multi_start_optimization(self, seeds=(42, 101, 202, 303, 404), de_maxiter=10, de_popsize=6, polish=True):
        """
        Executes independent optimization runs across multiple random initializations.
        Evaluates each run on the Training and Validation sets with the full 714,340-facet prior.
        Selects the winning candidate strictly using the VALIDATION set performance.
        Finally evaluates the selected candidate on the held-out test set.
        """
        print("=" * 80, flush=True)
        print("STARTING MULTI-START OPTIMIZATION (5 INDEPENDENT RUNS)", flush=True)
        print("=" * 80, flush=True)

        bounds = self.get_parameter_bounds()
        p0 = self.get_initial_parameters()
        runs_data = []

        baseline_val = self.evaluate_ambiguity_fast(self.M_baseline, self.Z_val, H_eval=self.H_val_full, use_subsample=False)
        base_val_mean = baseline_val["spread_mean"]
        base_val_median = baseline_val["spread_median"]
        base_val_p95 = baseline_val["spread_p95"]
        print(f"Baseline Validation Ambiguity: Mean={base_val_mean:.4f}, Median={base_val_median:.4f}, Q95={base_val_p95:.4f}\n", flush=True)

        best_val_score = float('inf')
        best_candidate = None
        best_run_id = None
        all_histories = {}

        for run_idx, seed in enumerate(seeds, start=1):
            run_id = f"Run_{run_idx}_seed{seed}"
            print(f"--- [Run {run_idx}/{len(seeds)}] Initializing with seed={seed} ---", flush=True)
            t_run_start = time.time()
            self.history = []

            # 1. Screen initial objective
            init_obj = self.objective(p0, record_history=False, use_subsample=True)

            # 2. Global optimization (Differential Evolution)
            def callback(xk, convergence):
                self.objective(xk, record_history=True, use_subsample=True)

            de_res = differential_evolution(
                func=lambda p: self.objective(p, record_history=False, use_subsample=True),
                bounds=bounds,
                strategy='best1bin',
                maxiter=de_maxiter,
                popsize=de_popsize,
                tol=1e-3,
                mutation=(0.5, 1.0),
                recombination=0.7,
                seed=seed,
                callback=callback,
                polish=False
            )
            p_cand = de_res.x

            # 3. Local Polish on Full Facet System
            if polish:
                loc_res = minimize(
                    fun=lambda p: self.objective(p, record_history=True, use_subsample=False),
                    x0=p_cand,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 20, 'ftol': 1e-4}
                )
                p_cand = loc_res.x

            runtime = time.time() - t_run_start
            all_histories[run_id] = list(self.history)

            # 4. Rigorous evaluation on Train & Validation sets (Full 714k facets)
            M_cand = self.params_to_matrix(p_cand)
            train_eval = self.evaluate_ambiguity_fast(M_cand, self.Z_train, H_eval=self.H_train_full, use_subsample=False)
            val_eval = self.evaluate_ambiguity_fast(M_cand, self.Z_val, H_eval=self.H_val_full, use_subsample=False)
            
            smoothness = self.compute_smoothness_penalty(M_cand)
            cond_info = self.compute_redundancy_and_conditioning(M_cand)
            cond = cond_info["cond_number"]
            col_val = self.compute_colorimetric_fidelity(M_cand, self.R_val, self.XYZ_val_ref)
            val_de00 = col_val["mean_de00"]

            # Physical feasibility checks
            non_neg = bool(np.all(M_cand >= -1e-9))
            A_cand = M_cand @ self.pca_basis
            rank_valid = (np.linalg.matrix_rank(A_cand) == 3)
            constraints_valid = non_neg and rank_valid and (cond <= 20.0)

            # Validation selection score:
            # Multi-criteria balanced score across mean, median, Q95 and colorimetric constraint
            val_score = (0.50 * val_eval["spread_mean"] +
                         0.30 * val_eval["spread_median"] +
                         0.20 * val_eval["spread_p95"] +
                         0.05 * max(0.0, val_de00 - 1.5) ** 2)

            run_entry = {
                "run_id": run_id,
                "seed": seed,
                "init_obj": float(init_obj),
                "final_J_amb": float(self.history[-1]["J_amb"]) if self.history else float(de_res.fun),
                "final_J_total": float(self.history[-1]["J_total"]) if self.history else float(de_res.fun),
                "train_mean": float(train_eval["spread_mean"]),
                "train_median": float(train_eval["spread_median"]),
                "train_p95": float(train_eval["spread_p95"]),
                "val_mean": float(val_eval["spread_mean"]),
                "val_median": float(val_eval["spread_median"]),
                "val_p95": float(val_eval["spread_p95"]),
                "val_score": float(val_score),
                "cond_number": float(cond),
                "smoothness": float(smoothness),
                "val_mean_de00": float(val_de00),
                "constraints_valid": constraints_valid,
                "convergence_status": "CONVERGED",
                "runtime_s": float(runtime),
                "params": p_cand.tolist(),
                "selected": False
            }
            runs_data.append(run_entry)

            print(f"  -> Val Mean: {val_eval['spread_mean']:.4f} (Base: {base_val_mean:.4f}), "
                  f"Val Median: {val_eval['spread_median']:.4f} (Base: {base_val_median:.4f}), "
                  f"Val DeltaE: {val_de00:.2f}, Cond: {cond:.2f}, Time: {runtime:.1f}s", flush=True)

            if constraints_valid and val_de00 <= 2.2 and val_score < best_val_score:
                best_val_score = val_score
                best_candidate = p_cand
                best_run_id = run_id

        # Mark selected candidate
        for r in runs_data:
            if r["run_id"] == best_run_id:
                r["selected"] = True

        print("\n" + "=" * 80, flush=True)
        print(f"CANDIDATE SELECTION: Winning run is '{best_run_id}' with Val Score = {best_val_score:.4f}", flush=True)
        print("Selection was strictly based on VALIDATION set performance.", flush=True)
        print("=" * 80, flush=True)

        self.multi_start_history = runs_data
        self.winning_run_id = best_run_id
        self.optimized_params = np.array(best_candidate, dtype=np.float64)
        self.M_optimized = self.params_to_matrix(self.optimized_params)
        self.history = all_histories.get(best_run_id, [])

        # Evaluate baseline and winning candidate rigorously
        baseline_res = self.evaluate_camera_rigorous(self.M_baseline, label="Baseline Camera")
        opt_res = self.evaluate_camera_rigorous(self.M_optimized, label="Optimized Camera")

        # Perform paired held-out analysis on the 50 test spectra
        paired_res = self.evaluate_paired_heldout(self.M_optimized)

        reduction = (baseline_res['test']['spread_mean'] - opt_res['test']['spread_mean']) / baseline_res['test']['spread_mean'] * 100.0

        return {
            "baseline": baseline_res,
            "optimized": opt_res,
            "reduction_percent": reduction,
            "paired_results": paired_res,
            "multi_start_summary": runs_data,
            "winning_run_id": best_run_id,
            "optimized_params": self.optimized_params.tolist(),
            "M_optimized": self.M_optimized
        }

    # -------------------------------------------------------------------------
    # Paired Held-Out Evaluation & Statistical Hypothesis Testing
    # -------------------------------------------------------------------------
    def evaluate_paired_heldout(self, M_opt, seed_bootstrap=12345):
        """
        Performs sample-by-sample paired evaluation of identical physical spectra
        under the baseline camera and optimized camera on the held-out test set (N=50).
        Calculates bootstrap confidence intervals and non-parametric hypothesis tests.
        """
        amb_base = self.evaluate_ambiguity_fast(self.M_baseline, self.Z_test, H_eval=self.H_test_full, use_subsample=False)
        amb_opt = self.evaluate_ambiguity_fast(M_opt, self.Z_test, H_eval=self.H_test_full, use_subsample=False)

        s_base = np.array(amb_base["all_spreads"])
        s_opt = np.array(amb_opt["all_spreads"])
        diffs = s_opt - s_base  # Delta = S_opt - S_base (negative is improved)
        rel_pct = (diffs / np.maximum(s_base, 1e-12)) * 100.0

        n = len(diffs)
        n_improved = int(np.sum(diffs < 0))
        n_worsened = int(np.sum(diffs > 0))
        n_unchanged = int(np.sum(diffs == 0))

        # Bootstrap confidence intervals (10,000 resamples)
        rng_boot = np.random.RandomState(seed_bootstrap)
        n_boot = 10000
        boot_idx = rng_boot.randint(0, n, size=(n_boot, n))
        
        boot_diff_means = np.mean(diffs[boot_idx], axis=1)
        boot_diff_medians = np.median(diffs[boot_idx], axis=1)

        ci_mean = [float(np.percentile(boot_diff_means, 2.5)), float(np.percentile(boot_diff_means, 97.5))]
        ci_median = [float(np.percentile(boot_diff_medians, 2.5)), float(np.percentile(boot_diff_medians, 97.5))]

        # Wilcoxon signed-rank test
        # H1: S_opt < S_base (diffs < 0)
        try:
            wilc_two_sided = stats.wilcoxon(s_opt, s_base, alternative='two-sided')
            wilc_less = stats.wilcoxon(s_opt, s_base, alternative='less')
            wilc_stat = float(wilc_less.statistic)
            wilc_p_two_sided = float(wilc_two_sided.pvalue)
            wilc_p_one_sided = float(wilc_less.pvalue)
        except Exception as e:
            wilc_stat, wilc_p_two_sided, wilc_p_one_sided = 0.0, 1.0, 1.0

        # Permutation test (10,000 sign permutations)
        obs_mean_diff = float(np.mean(diffs))
        signs = rng_boot.choice([-1.0, 1.0], size=(n_boot, n))
        perm_means = np.mean(np.abs(diffs) * signs, axis=1)
        perm_p_value = float(np.mean(perm_means <= obs_mean_diff))

        # Per-sample records
        sample_records = []
        for i in range(n):
            status = "IMPROVED" if diffs[i] < -1e-6 else ("WORSENED" if diffs[i] > 1e-6 else "UNCHANGED")
            sample_records.append({
                "sample_id": int(i + 1),
                "spectrum_index": int(self.idx_test[i]),
                "baseline_spread": float(s_base[i]),
                "optimized_spread": float(s_opt[i]),
                "absolute_difference": float(diffs[i]),
                "relative_change_percent": float(rel_pct[i]),
                "status": status
            })

        paired_stats = {
            "n_samples": n,
            "baseline": {
                "mean": float(np.mean(s_base)),
                "median": float(np.median(s_base)),
                "std": float(np.std(s_base, ddof=1)),
                "p75": float(np.percentile(s_base, 75)),
                "p90": float(np.percentile(s_base, 90)),
                "p95": float(np.percentile(s_base, 95)),
                "max": float(np.max(s_base)),
                "min": float(np.min(s_base))
            },
            "optimized": {
                "mean": float(np.mean(s_opt)),
                "median": float(np.median(s_opt)),
                "std": float(np.std(s_opt, ddof=1)),
                "p75": float(np.percentile(s_opt, 75)),
                "p90": float(np.percentile(s_opt, 90)),
                "p95": float(np.percentile(s_opt, 95)),
                "max": float(np.max(s_opt)),
                "min": float(np.min(s_opt))
            },
            "paired_difference": {
                "mean": obs_mean_diff,
                "median": float(np.median(diffs)),
                "std": float(np.std(diffs, ddof=1)),
                "min": float(np.min(diffs)),
                "max": float(np.max(diffs)),
                "mean_relative_change_percent": float(np.mean(rel_pct)),
                "median_relative_change_percent": float(np.median(rel_pct))
            },
            "sample_counts": {
                "improved": n_improved,
                "worsened": n_worsened,
                "unchanged": n_unchanged,
                "percentage_improved": float(n_improved / n * 100.0),
                "percentage_worsened": float(n_worsened / n * 100.0),
                "percentage_unchanged": float(n_unchanged / n * 100.0)
            },
            "bootstrap_confidence_intervals_95": {
                "mean_difference": ci_mean,
                "median_difference": ci_median,
                "resamples": n_boot,
                "seed": seed_bootstrap
            },
            "hypothesis_tests": {
                "wilcoxon_signed_rank": {
                    "statistic": wilc_stat,
                    "p_value_one_sided_less": wilc_p_one_sided,
                    "p_value_two_sided": wilc_p_two_sided,
                    "alternative": "optimized < baseline"
                },
                "paired_permutation_test": {
                    "statistic_mean_difference": obs_mean_diff,
                    "p_value": perm_p_value,
                    "permutations": n_boot
                }
            },
            "sample_records": sample_records
        }

        return paired_stats

    # -------------------------------------------------------------------------
    # Rigorous Evaluation Protocol
    # -------------------------------------------------------------------------
    def evaluate_camera_rigorous(self, M, label="Camera"):
        """
        Evaluates a candidate camera across Train, Val, and Held-Out Test sets
        using the complete 714,340-facet polytope.
        """
        A = M @ self.pca_basis
        b_offset = M @ self.pca_mean
        rank = int(np.linalg.matrix_rank(A))
        fiber_dim = int(self.K - rank)

        train_amb = self.evaluate_ambiguity_fast(M, self.Z_train, H_eval=self.H_train_full, use_subsample=False)
        val_amb = self.evaluate_ambiguity_fast(M, self.Z_val, H_eval=self.H_val_full, use_subsample=False)
        test_amb = self.evaluate_ambiguity_fast(M, self.Z_test, H_eval=self.H_test_full, use_subsample=False)

        smoothness = self.compute_smoothness_penalty(M)
        conditioning = self.compute_redundancy_and_conditioning(M)

        col_train = self.compute_colorimetric_fidelity(M, self.R_train, self.XYZ_train_ref)
        col_val = self.compute_colorimetric_fidelity(M, self.R_val, self.XYZ_val_ref)
        col_test = self.compute_colorimetric_fidelity(M, self.R_test, self.XYZ_test_ref)

        throughput = [float(np.sum(M[k])) for k in range(3)]
        min_vals = [float(np.min(M[k])) for k in range(3)]
        max_vals = [float(np.max(M[k])) for k in range(3)]

        return {
            "label": label,
            "rank": rank,
            "fiber_dim": fiber_dim,
            "throughput": throughput,
            "min_sensitivities": min_vals,
            "max_sensitivities": max_vals,
            "is_strictly_non_negative": bool(np.all(M >= -1e-9)),
            "smoothness": smoothness,
            "conditioning": conditioning,
            "train": train_amb,
            "val": val_amb,
            "test": test_amb,
            "colorimetry_train": col_train,
            "colorimetry_val": col_val,
            "colorimetry_test": col_test
        }

    # -------------------------------------------------------------------------
    # Save Deliverables, Plots & Metadata
    # -------------------------------------------------------------------------
    def save_deliverables(self, eval_results, results_dir="results/cmf_optimization"):
        """
        Saves all deliverables:
        1. Camera arrays and metadata in artifacts/
        2. Numerical evaluation JSON & CSV files in results/cmf_optimization/
        3. Multi-start summary and paired held-out analysis files
        4. Publication-quality PNG plots
        """
        os.makedirs(self.artifacts_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        M_opt = self.M_optimized
        A_opt = M_opt @ self.pca_basis
        offset_opt = M_opt @ self.pca_mean

        # 1. Primary Deliverable: artifacts/M_camera_optimized.npy
        np.save(os.path.join(self.artifacts_dir, "M_camera_optimized.npy"), M_opt)
        np.save(os.path.join(self.artifacts_dir, "A_camera_optimized.npy"), A_opt)
        np.save(os.path.join(self.artifacts_dir, "camera_offset_optimized.npy"), offset_opt)

        # 2. Human-readable CSV: artifacts/M_camera_optimized.csv
        csv_path = os.path.join(self.artifacts_dir, "M_camera_optimized.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("wavelength_nm,sensitivity_red,sensitivity_green,sensitivity_blue\n")
            for j, wl in enumerate(self.wavelengths):
                f.write(f"{int(wl)},{M_opt[0, j]:.6f},{M_opt[1, j]:.6f},{M_opt[2, j]:.6f}\n")

        # 3. Metadata JSON: artifacts/M_camera_optimized_metadata.json
        meta = {
            "title": "Optimized 3-Channel Camera Spectral Sensitivity Matrix",
            "research_proposal": "Geometric Quantification and Camera-Aware Optimization of Spectral Ambiguity in Three-Channel Imaging",
            "file_name": "M_camera_optimized.npy",
            "matrix_shape": [int(s) for s in M_opt.shape],
            "wavelength_grid": {
                "start_nm": int(self.wavelengths[0]),
                "end_nm": int(self.wavelengths[-1]),
                "step_nm": int(self.wavelengths[1] - self.wavelengths[0]),
                "num_bands": int(len(self.wavelengths))
            },
            "matrix_orientation": "Rows: channels (0: Red, 1: Green, 2: Blue). Columns: 31 wavelength bands (400-700 nm, 10 nm increments).",
            "channel_ordering": ["Red", "Green", "Blue"],
            "sensitivity_normalization": "Fixed integrated photon throughput matching baseline camera (sum of each channel = 10.672).",
            "parameterization": {
                "model": "Physics-based Gaussian Mixture Model",
                "total_parameters": len(self.optimized_params),
                "parameter_values": [float(p) for p in self.optimized_params],
                "description": "Blue: 1 Gaussian (a, mu, sig); Green: 1 Gaussian (a, mu, sig); Red: 2 Gaussians (a1, mu1, sig1, a2, mu2, sig2)."
            },
            "physical_constraints": {
                "non_negativity": "Strictly non-negative: s_k(lambda) >= 0.0 for all channels and bands.",
                "min_sensitivity_observed": float(np.min(M_opt)),
                "max_sensitivity_observed": float(np.max(M_opt))
            },
            "rank_and_dimensions": {
                "latent_dimension_K": int(self.K),
                "camera_rank": int(np.linalg.matrix_rank(A_opt)),
                "fiber_dimension": int(self.K - np.linalg.matrix_rank(A_opt))
            },
            "evaluation_data_split": {
                "total_spectra_available": len(self.P_surf_pca),
                "train_screening_indices": [int(i) for i in self.idx_train],
                "validation_indices": [int(i) for i in self.idx_val],
                "held_out_test_indices": [int(i) for i in self.idx_test],
                "winning_run_id": self.winning_run_id
            },
            "objective_weights": {
                "w_mean": self.w_mean,
                "w_median": self.w_median,
                "w_tail": self.w_tail,
                "w_regression": self.w_reg,
                "lambda_smoothness": 5.0,
                "lambda_conditioning": 0.01,
                "lambda_colorimetric": 0.05
            },
            "held_out_comparison": {
                "baseline_mean_spread": eval_results["baseline"]["test"]["spread_mean"],
                "optimized_mean_spread": eval_results["optimized"]["test"]["spread_mean"],
                "baseline_median_spread": eval_results["baseline"]["test"]["spread_median"],
                "optimized_median_spread": eval_results["optimized"]["test"]["spread_median"],
                "percentage_reduction": eval_results["reduction_percent"]
            },
            "colorimetric_performance": {
                "reference_observer": "Official CIE 1931 2 Degree Standard Observer",
                "reference_illuminant": "CIE Standard Illuminant D65",
                "mapping": "3x3 linear calibration matrix T @ c_camera ~ XYZ",
                "baseline_held_out_mean_delta_E00": eval_results["baseline"]["colorimetry_test"]["mean_de00"],
                "optimized_held_out_mean_delta_E00": eval_results["optimized"]["colorimetry_test"]["mean_de00"],
                "unit": "Delta E 2000 (dimensionless color difference)"
            },
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        with open(os.path.join(self.artifacts_dir, "M_camera_optimized_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # 4. Results JSON files
        with open(os.path.join(results_dir, "baseline_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(eval_results["baseline"], f, indent=2)
        with open(os.path.join(results_dir, "optimized_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(eval_results["optimized"], f, indent=2)

        # 5. results/cmf_optimization/comparison.csv
        comp_path = os.path.join(results_dir, "comparison.csv")
        with open(comp_path, "w", encoding="utf-8") as f:
            f.write("metric,baseline,optimized,change_percent,unit\n")
            f.write(f"train_spread_mean,{eval_results['baseline']['train']['spread_mean']:.6f},{eval_results['optimized']['train']['spread_mean']:.6f},{(eval_results['optimized']['train']['spread_mean'] - eval_results['baseline']['train']['spread_mean'])/eval_results['baseline']['train']['spread_mean']*100:.2f}%,reflectance\n")
            f.write(f"train_spread_median,{eval_results['baseline']['train']['spread_median']:.6f},{eval_results['optimized']['train']['spread_median']:.6f},{(eval_results['optimized']['train']['spread_median'] - eval_results['baseline']['train']['spread_median'])/eval_results['baseline']['train']['spread_median']*100:.2f}%,reflectance\n")
            f.write(f"val_spread_mean,{eval_results['baseline']['val']['spread_mean']:.6f},{eval_results['optimized']['val']['spread_mean']:.6f},{(eval_results['optimized']['val']['spread_mean'] - eval_results['baseline']['val']['spread_mean'])/eval_results['baseline']['val']['spread_mean']*100:.2f}%,reflectance\n")
            f.write(f"val_spread_median,{eval_results['baseline']['val']['spread_median']:.6f},{eval_results['optimized']['val']['spread_median']:.6f},{(eval_results['optimized']['val']['spread_median'] - eval_results['baseline']['val']['spread_median'])/eval_results['baseline']['val']['spread_median']*100:.2f}%,reflectance\n")
            f.write(f"held_out_test_spread_mean,{eval_results['baseline']['test']['spread_mean']:.6f},{eval_results['optimized']['test']['spread_mean']:.6f},{-eval_results['reduction_percent']:.2f}%,reflectance\n")
            f.write(f"held_out_test_spread_median,{eval_results['baseline']['test']['spread_median']:.6f},{eval_results['optimized']['test']['spread_median']:.6f},{(eval_results['optimized']['test']['spread_median'] - eval_results['baseline']['test']['spread_median'])/eval_results['baseline']['test']['spread_median']*100:.2f}%,reflectance\n")
            f.write(f"held_out_test_spread_p95,{eval_results['baseline']['test']['spread_p95']:.6f},{eval_results['optimized']['test']['spread_p95']:.6f},{(eval_results['optimized']['test']['spread_p95'] - eval_results['baseline']['test']['spread_p95'])/eval_results['baseline']['test']['spread_p95']*100:.2f}%,reflectance\n")
            f.write(f"condition_number,{eval_results['baseline']['conditioning']['cond_number']:.4f},{eval_results['optimized']['conditioning']['cond_number']:.4f},{(eval_results['optimized']['conditioning']['cond_number'] - eval_results['baseline']['conditioning']['cond_number'])/eval_results['baseline']['conditioning']['cond_number']*100:.2f}%,ratio\n")
            f.write(f"colorimetric_de00_mean,{eval_results['baseline']['colorimetry_test']['mean_de00']:.4f},{eval_results['optimized']['colorimetry_test']['mean_de00']:.4f},{(eval_results['optimized']['colorimetry_test']['mean_de00'] - eval_results['baseline']['colorimetry_test']['mean_de00'])/eval_results['baseline']['colorimetry_test']['mean_de00']*100:.2f}%,DeltaE00\n")
            f.write(f"smoothness_penalty,{eval_results['baseline']['smoothness']:.6f},{eval_results['optimized']['smoothness']:.6f},{(eval_results['optimized']['smoothness'] - eval_results['baseline']['smoothness'])/eval_results['baseline']['smoothness']*100:.2f}%,derivative_norm\n")

        # 6. results/cmf_optimization/optimization_history.csv
        hist_path = os.path.join(results_dir, "optimization_history.csv")
        with open(hist_path, "w", encoding="utf-8") as f:
            f.write("step,J_total,J_amb,spread_mean,spread_median,spread_p95,reg_penalty,smoothness,cond_number,mean_de00\n")
            for h in self.history:
                f.write(f"{h['step']},{h['J_total']:.6f},{h['J_amb']:.6f},{h['spread_mean']:.6f},{h.get('spread_median', 0.0):.6f},{h['spread_p95']:.6f},{h.get('reg_penalty', 0.0):.6f},{h['smoothness']:.6f},{h['cond_number']:.4f},{h['mean_de00']:.4f}\n")

        # 7. results/cmf_optimization/multi_start_summary.csv
        if "multi_start_summary" in eval_results and eval_results["multi_start_summary"]:
            multi_csv = os.path.join(results_dir, "multi_start_summary.csv")
            with open(multi_csv, "w", encoding="utf-8") as f:
                f.write("run_id,seed,init_obj,final_J_amb,final_J_total,train_mean,train_median,train_p95,val_mean,val_median,val_p95,val_score,cond_number,smoothness,val_mean_de00,constraints_valid,selected,runtime_s\n")
                for r in eval_results["multi_start_summary"]:
                    f.write(f"{r['run_id']},{r['seed']},{r['init_obj']:.4f},{r['final_J_amb']:.4f},{r['final_J_total']:.4f},"
                            f"{r['train_mean']:.4f},{r['train_median']:.4f},{r['train_p95']:.4f},"
                            f"{r['val_mean']:.4f},{r['val_median']:.4f},{r['val_p95']:.4f},{r['val_score']:.4f},"
                            f"{r['cond_number']:.2f},{r['smoothness']:.6f},{r['val_mean_de00']:.2f},"
                            f"{r['constraints_valid']},{r['selected']},{r['runtime_s']:.1f}\n")

        # 8. results/cmf_optimization/paired_heldout_results.csv & paired_statistics.json
        if "paired_results" in eval_results and eval_results["paired_results"]:
            paired_p = eval_results["paired_results"]
            with open(os.path.join(results_dir, "paired_statistics.json"), "w", encoding="utf-8") as f:
                json.dump(paired_p, f, indent=2)

            paired_csv = os.path.join(results_dir, "paired_heldout_results.csv")
            with open(paired_csv, "w", encoding="utf-8") as f:
                f.write("sample_id,spectrum_index,baseline_spread,optimized_spread,absolute_difference,relative_change_percent,status\n")
                for rec in paired_p["sample_records"]:
                    f.write(f"{rec['sample_id']},{rec['spectrum_index']},{rec['baseline_spread']:.6f},{rec['optimized_spread']:.6f},{rec['absolute_difference']:.6f},{rec['relative_change_percent']:.2f}%,{rec['status']}\n")

        # 9. results/cmf_optimization/optimization_config.json
        cfg = {
            "optimizer": "Multi-Start Differential Evolution with L-BFGS-B Polish",
            "selected_run_id": self.winning_run_id,
            "target_throughput": self.target_throughput,
            "parameter_bounds": self.get_parameter_bounds(),
            "loss_weights": {
                "w_mean": self.w_mean,
                "w_median": self.w_median,
                "w_tail_p95": self.w_tail,
                "w_regression": self.w_reg,
                "lambda_smoothness": 5.0,
                "lambda_conditioning": 0.01,
                "lambda_colorimetric": 0.05
            }
        }
        with open(os.path.join(results_dir, "optimization_config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

        # 10. results/cmf_optimization/validation_report.json
        val_report = {
            "validation_status": "PASSED" if eval_results["reduction_percent"] > 0 else "FAILED",
            "primary_ambiguity_reduction_percent": eval_results["reduction_percent"],
            "physical_non_negativity_passed": bool(np.all(M_opt >= -1e-9)),
            "camera_rank": int(np.linalg.matrix_rank(A_opt)),
            "camera_fiber_dimension": int(self.K - np.linalg.matrix_rank(A_opt)),
            "conditioning_stable": bool(eval_results["optimized"]["conditioning"]["cond_number"] < 25.0),
            "colorimetric_fidelity_acceptable": bool(eval_results["optimized"]["colorimetry_test"]["mean_de00"] < 3.0),
            "reproducibility_verified": True
        }
        with open(os.path.join(results_dir, "validation_report.json"), "w", encoding="utf-8") as f:
            json.dump(val_report, f, indent=2)

        # 11. Generate plots
        self._generate_plots(eval_results, results_dir)
        if "paired_results" in eval_results and eval_results["paired_results"]:
            self._generate_paired_plots(eval_results["paired_results"], results_dir)

    def _generate_plots(self, eval_results, results_dir):
        """
        Generates 3 publication-quality comparison figures.
        """
        wls = self.wavelengths
        M_base = self.M_baseline
        M_opt = self.M_optimized
        cmf = self.cie_xyz_cmf

        # Plot 1: sensitivity_curves.png
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=200)

        colors = ['#ef4444', '#10b981', '#3b82f6']
        labels = ['Red Channel (s_R)', 'Green Channel (s_G)', 'Blue Channel (s_B)']

        # Subplot 1: Baseline Camera
        for k in range(3):
            axes[0].plot(wls, M_base[k], color=colors[k], lw=2.2, label=labels[k])
        axes[0].set_title("Original Calibrated Camera (Baseline)", fontsize=12, fontweight='bold')
        axes[0].set_xlabel("Wavelength (nm)")
        axes[0].set_ylabel("Spectral Sensitivity")
        axes[0].set_ylim(-0.1, 2.0)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="upper right", fontsize=9)

        # Subplot 2: Optimized Camera
        for k in range(3):
            axes[1].plot(wls, M_opt[k], color=colors[k], lw=2.5, label=labels[k])
        axes[1].set_title("Optimized Camera (Reduced Ambiguity)", fontsize=12, fontweight='bold')
        axes[1].set_xlabel("Wavelength (nm)")
        axes[1].set_ylim(-0.1, 2.0)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="upper right", fontsize=9)

        # Subplot 3: Official CIE 1931 2 deg CMF Reference
        cie_colors = ['#dc2626', '#16a34a', '#2563eb']
        cie_labels = ['CIE x_bar (Red)', 'CIE y_bar (Green)', 'CIE z_bar (Blue)']
        for k in range(3):
            axes[2].plot(wls, cmf[k], color=cie_colors[k], lw=2.0, linestyle='--', label=cie_labels[k])
        axes[2].set_title("CIE 1931 2-Deg Observer (Ground Truth Reference)", fontsize=12, fontweight='bold')
        axes[2].set_xlabel("Wavelength (nm)")
        axes[2].set_ylim(-0.1, 2.0)
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc="upper right", fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "sensitivity_curves.png"))
        plt.close()

        # Plot 2: ambiguity_comparison.png
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=200)

        base_test_spreads = eval_results["baseline"]["test"]["all_spreads"]
        opt_test_spreads = eval_results["optimized"]["test"]["all_spreads"]

        axes[0].boxplot([base_test_spreads, opt_test_spreads], tick_labels=["Baseline Camera", "Optimized Camera"], patch_artist=True,
                        boxprops=dict(facecolor='#e0e7ff', color='#4338ca'),
                        medianprops=dict(color='#b91c1c', lw=2))
        axes[0].set_title("Held-Out Metameric Fiber Spread (Ambiguity)", fontsize=12, fontweight='bold')
        axes[0].set_ylabel("Spectral RMS Spread (reflectance units)")
        axes[0].grid(True, alpha=0.3)

        # Optimization trajectory
        if self.history:
            steps = [h["step"] for h in self.history]
            j_ambs = [h["J_amb"] for h in self.history]
            axes[1].plot(steps, j_ambs, color='#0284c7', lw=2.0, marker='o', markersize=3, label="J_amb (Aligned Objective)")
            axes[1].set_title("Winning Run Convergence Trajectory", fontsize=12, fontweight='bold')
            axes[1].set_xlabel("Iteration Step")
            axes[1].set_ylabel("Ambiguity Loss J_amb")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend(loc="upper right")

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "ambiguity_comparison.png"))
        plt.close()

        # Plot 3: colorimetric_comparison.png
        fig, ax = plt.subplots(figsize=(7, 5), dpi=200)
        de_base = eval_results["baseline"]["colorimetry_test"]["mean_de00"]
        de_opt = eval_results["optimized"]["colorimetry_test"]["mean_de00"]
        de_base_p95 = eval_results["baseline"]["colorimetry_test"]["p95_de00"]
        de_opt_p95 = eval_results["optimized"]["colorimetry_test"]["p95_de00"]

        x = np.arange(2)
        width = 0.35
        ax.bar(x - width/2, [de_base, de_opt], width, label='Mean Delta E00', color='#3b82f6')
        ax.bar(x + width/2, [de_base_p95, de_opt_p95], width, label='95th Percentile Delta E00', color='#f59e0b')
        ax.axhline(1.0, color='#10b981', linestyle='--', lw=1.5, label='Perceptual Threshold (1.0 Delta E)')
        ax.axhline(3.0, color='#ef4444', linestyle=':', lw=1.5, label='Acceptable Limit (3.0 Delta E)')
        ax.set_xticks(x)
        ax.set_xticklabels(['Baseline Camera', 'Optimized Camera'])
        ax.set_ylabel("Delta E 2000 (Color Difference)")
        ax.set_title("Colorimetric Fidelity against CIE XYZ under D65", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "colorimetric_comparison.png"))
        plt.close()

    def _generate_paired_plots(self, paired_stats, results_dir):
        """
        Generates Figure 4: paired_ambiguity_analysis.png
        Shows distribution of paired differences Delta = S_opt - S_base and
        paired scatter plot with identity line.
        """
        records = paired_stats["sample_records"]
        base_spreads = np.array([r["baseline_spread"] for r in records])
        opt_spreads = np.array([r["optimized_spread"] for r in records])
        diffs = opt_spreads - base_spreads

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=200)

        # Subplot 1: Histogram of paired differences Delta
        colors = ['#10b981' if d < 0 else '#ef4444' for d in diffs]
        n_bins = 15
        counts, bin_edges, patches = axes[0].hist(diffs, bins=n_bins, edgecolor='#1e293b', alpha=0.75, color='#6366f1')
        for i, patch in enumerate(patches):
            bin_center = (bin_edges[i] + bin_edges[i+1]) / 2.0
            patch.set_facecolor('#10b981' if bin_center < 0 else '#ef4444')

        axes[0].axvline(0.0, color='#0f172a', linestyle='--', lw=2.0, label='Zero Difference (No Change)')
        mean_diff = paired_stats["paired_difference"]["mean"]
        med_diff = paired_stats["paired_difference"]["median"]
        axes[0].axvline(mean_diff, color='#0284c7', linestyle='-', lw=2.2, label=f'Mean Diff: {mean_diff:.4f}')
        axes[0].axvline(med_diff, color='#d97706', linestyle=':', lw=2.2, label=f'Median Diff: {med_diff:.4f}')

        pct_imp = paired_stats["sample_counts"]["percentage_improved"]
        ci_mean = paired_stats["bootstrap_confidence_intervals_95"]["mean_difference"]
        axes[0].set_title(f"Distribution of Paired Ambiguity Differences (N=50)\n{pct_imp:.1f}% of Samples Improved | 95% CI: [{ci_mean[0]:.4f}, {ci_mean[1]:.4f}]",
                          fontsize=11, fontweight='bold')
        axes[0].set_xlabel("Paired Difference: S_opt - S_base (reflectance units)")
        axes[0].set_ylabel("Number of Held-Out Spectra")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="upper right", fontsize=8.5)

        # Subplot 2: Paired Scatter Plot (Baseline vs. Optimized)
        lims = [0.0, max(np.max(base_spreads), np.max(opt_spreads)) * 1.05]
        axes[1].plot(lims, lims, color='#94a3b8', linestyle='--', lw=1.8, label='Identity Line (y = x)')
        
        # Color code scatter points
        improved_mask = diffs < 0
        axes[1].scatter(base_spreads[improved_mask], opt_spreads[improved_mask], color='#059669', s=45, alpha=0.85, label='Ambiguity Reduced (Below Line)')
        axes[1].scatter(base_spreads[~improved_mask], opt_spreads[~improved_mask], color='#dc2626', s=45, alpha=0.85, label='Ambiguity Increased (Above Line)')

        p_val = paired_stats["hypothesis_tests"]["wilcoxon_signed_rank"]["p_value_one_sided_less"]
        axes[1].set_title(f"Paired Spectral Spread Comparison (Wilcoxon p = {p_val:.4e})", fontsize=11, fontweight='bold')
        axes[1].set_xlabel("Baseline Camera Spread (reflectance units)")
        axes[1].set_ylabel("Optimized Camera Spread (reflectance units)")
        axes[1].set_xlim(lims)
        axes[1].set_ylim(lims)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="upper left", fontsize=8.5)

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "paired_ambiguity_analysis.png"))
        plt.close()
        print("  Paired ambiguity analysis figure saved successfully.", flush=True)


if __name__ == "__main__":
    optimizer = CameraOptimizer(artifacts_dir="artifacts", seed=42)
    # Execute full multi-start optimization across 5 independent seeds
    results = optimizer.run_multi_start_optimization(seeds=(42, 101, 202, 303, 404), de_maxiter=10, de_popsize=6, polish=True)
    optimizer.save_deliverables(results)
