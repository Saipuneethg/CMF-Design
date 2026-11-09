import numpy as np


class AmbiguityMetrics:
    """
    Computes rigorous quantitative geometric and spectral ambiguity measures
    for the feasible metameric fiber:
    
    1. Mathematical Fiber Dimension: d_f = K - rank(A)
    2. Spectral Spread Score:
       - S_spread = sqrt(1/N * sum ||R_i - R_bar||_2^2) = sqrt((N-1)/N * Tr(Sigma_R))
       - sigma_spread = sqrt(Tr(Sigma_alpha) / d_f)
    3. Pairwise Spectral Distance Distribution:
       - D_ij = ||R_i - R_j||_2
       - Mean, median, 95th percentile, and max observed pairwise distance
    4. Full Spectral Covariance and Principal Ambiguity Directions:
       - Sigma_R = 1/(N-1) * sum (R_i - R_bar)(R_i - R_bar)^T in R^{31 x 31}
       - Eigenvalues, eigenvectors (ambiguity modes), explained variance %, cumulative %
       - Sample projections onto leading directions
    5. Determinant-Based Volume and Generalized Variance Scale:
       - Generalized variance: det(Sigma_alpha)
       - Ellipsoidal volume scale: A_det = (det(Sigma_alpha))^(1 / (2 * d_f))
       - Sample bounding box volume: prod(max(alpha) - min(alpha))
    6. Wavelength-wise Statistics and Uncertainty:
       - Mean, std, min, max, span per band
       - Integrated uncertainty Delta R_int = trapz(sigma_R, lambda)
    7. Optional Composite Ambiguity Index:
       - Documented combination of normalized spread, mean span, and volume scale
    8. MCMC Effective Sample Size (ESS) and Autocorrelation Diagnostics
    """

    @staticmethod
    def compute_fiber_dimension(A_camera, K=8):
        """
        Computes the mathematical affine fiber dimension: d_f = K - rank(A).
        """
        rank = int(np.linalg.matrix_rank(A_camera))
        d_f = int(K - rank)
        return {
            "K": int(K),
            "camera_rank": rank,
            "fiber_dimension": d_f,
            "interpretation": (
                f"Affine equality fiber has dimension {d_f} in {K}D PCA space. "
                "Feasible intersection with polytope is a compact convex subset."
            )
        }

    @staticmethod
    def validate_synthetic_geometry(dim=5, num_samples=5000, seed=42):
        """
        Validates sample spread, bounding box, and volume estimation on known canonical
        synthetic polytopes where exact values are known analytically:
        
        Synthetic Benchmark: d-dimensional hypercube [-1, 1]^d
        - True side length: 2.0
        - True bounding volume: 2^d
        - True coordinate variance: Var(X_i) = 1/3 ~ 0.3333
        - True trace spread: sqrt(1/3) ~ 0.57735
        - True volume scale: (det(Cov))^(1/(2d)) = (1/3)^(1/2) ~ 0.57735
        """
        rng = np.random.RandomState(seed)
        samples = rng.uniform(-1.0, 1.0, size=(num_samples, dim))
        
        sample_mean = np.mean(samples, axis=0)
        cov = np.cov(samples, rowvar=False)
        var_coords = np.diag(cov)
        
        det_cov = float(np.linalg.det(cov))
        est_vol_scale = float(det_cov ** (1.0 / (2.0 * dim))) if det_cov > 0 else 0.0
        true_vol_scale = 1.0 / np.sqrt(3.0)
        
        sample_min = np.min(samples, axis=0)
        sample_max = np.max(samples, axis=0)
        
        box_vol = float(np.prod(sample_max - sample_min))
        true_box_vol = 2.0 ** dim
        
        return {
            "dimension": dim,
            "num_samples": num_samples,
            "true_mean": [0.0] * dim,
            "sample_mean_norm": float(np.linalg.norm(sample_mean)),
            "true_variance_per_coord": 1.0 / 3.0,
            "mean_sample_variance": float(np.mean(var_coords)),
            "variance_relative_error": float(abs(np.mean(var_coords) - (1.0 / 3.0)) / (1.0 / 3.0)),
            "true_volume_scale": float(true_vol_scale),
            "estimated_volume_scale": est_vol_scale,
            "volume_scale_relative_error": float(abs(est_vol_scale - true_vol_scale) / true_vol_scale),
            "true_box_volume": true_box_vol,
            "sample_box_volume": box_vol,
            "box_volume_relative_error": float(abs(box_vol - true_box_vol) / true_box_vol)
        }

    @staticmethod
    def compute_pairwise_distances(spectra_samples, max_pairs=50000, seed=42):
        """
        Computes the Euclidean pairwise spectral distance distribution:
            D_ij = ||R_i - R_j||_2
        
        Uses an exact all-pairs calculation if N <= 1000 (up to 499,500 pairs),
        or a reproducible random sample of max_pairs distinct pairs if N > 1000.
        
        Parameters:
        - spectra_samples: Array (N, 31) of reconstructed reflectance spectra.
        - max_pairs: Maximum number of pairs to evaluate if subsampling.
        - seed: Integer random seed for reproducible pair sampling.
        
        Returns:
        - Dict with mean, median, p95, max observed distance, pair count, and units.
        """
        N = len(spectra_samples)
        if N < 2:
            return {
                "mean": 0.0,
                "median": 0.0,
                "p95": 0.0,
                "max_observed": 0.0,
                "pairs_evaluated": 0,
                "method": "insufficient_samples",
                "unit": "reflectance-vector distance (dimensionless)",
                "kind": "sample_estimate"
            }

        rng = np.random.RandomState(seed)
        total_possible = N * (N - 1) // 2

        if total_possible <= max_pairs:
            # All pairs via upper triangle indices
            i_idx, j_idx = np.triu_indices(N, k=1)
            diffs = spectra_samples[i_idx] - spectra_samples[j_idx]
            dists = np.linalg.norm(diffs, axis=1)
            method = f"all_pairs ({len(dists)} evaluated)"
        else:
            # Reproducible random pair sampling
            i_idx = rng.randint(0, N, size=max_pairs)
            j_idx = rng.randint(0, N, size=max_pairs)
            # Filter self-pairs
            valid = i_idx != j_idx
            i_idx = i_idx[valid]
            j_idx = j_idx[valid]
            diffs = spectra_samples[i_idx] - spectra_samples[j_idx]
            dists = np.linalg.norm(diffs, axis=1)
            method = f"random_sampling ({len(dists)} of {total_possible} pairs)"

        mean_dist = float(np.mean(dists))
        median_dist = float(np.median(dists))
        p95_dist = float(np.percentile(dists, 95))
        max_dist = float(np.max(dists))

        return {
            "mean": {
                "value": mean_dist,
                "unit": "reflectance-vector distance (dimensionless)",
                "kind": "sample_estimate",
                "label": "Mean Pairwise Distance"
            },
            "median": {
                "value": median_dist,
                "unit": "reflectance-vector distance (dimensionless)",
                "kind": "sample_estimate",
                "label": "Median Pairwise Distance"
            },
            "p95": {
                "value": p95_dist,
                "unit": "reflectance-vector distance (dimensionless)",
                "kind": "sample_estimate",
                "label": "95th Percentile Distance"
            },
            "max_observed": {
                "value": max_dist,
                "unit": "reflectance-vector distance (dimensionless)",
                "kind": "sample_estimate",
                "label": "Maximum Observed Pairwise Distance",
                "note": "Sample extrema; lower bound for continuous fiber diameter."
            },
            "pairs_evaluated": len(dists),
            "total_possible_pairs": total_possible,
            "sampling_method": method
        }

    @staticmethod
    def compute_mcmc_diagnostics(alpha_samples, max_lag=30):
        """
        Computes Effective Sample Size (ESS) and autocorrelation diagnostics
        per reduced coordinate using Geyer's initial positive sequence estimator.
        """
        N, d = alpha_samples.shape
        if N < 10:
            return {
                "ess_per_coord": [float(N)] * d,
                "min_ess": float(N),
                "mean_ess": float(N),
                "lag1_autocorr": [0.0] * d
            }

        ess_list = []
        lag1_list = []

        for j in range(d):
            x = alpha_samples[:, j] - np.mean(alpha_samples[:, j])
            var_x = np.var(x)
            if var_x < 1e-14:
                ess_list.append(float(N))
                lag1_list.append(0.0)
                continue

            # Autocorrelation function
            n_lags = min(max_lag, N // 2)
            acf = np.zeros(n_lags)
            acf[0] = 1.0
            for k in range(1, n_lags):
                cov_k = np.mean(x[:-k] * x[k:])
                acf[k] = cov_k / var_x

            lag1_list.append(float(acf[1]) if n_lags > 1 else 0.0)

            # Geyer's initial positive sequence estimator
            gamma_sum = 0.0
            for k in range(1, (n_lags - 1) // 2 * 2, 2):
                pair_sum = acf[k] + acf[k + 1]
                if pair_sum <= 0:
                    break
                gamma_sum += pair_sum

            act = max(1.0 + 2.0 * gamma_sum, 1.0)
            ess = float(min(N / act, float(N)))
            ess_list.append(ess)

        return {
            "ess_per_coord": [float(v) for v in ess_list],
            "min_ess": float(np.min(ess_list)),
            "mean_ess": float(np.mean(ess_list)),
            "lag1_autocorr": [float(v) for v in lag1_list],
            "mean_lag1_autocorr": float(np.mean(lag1_list))
        }

    @staticmethod
    def compute_composite_ambiguity_index(spread_score, mean_span, volume_scale):
        """
        Calculates an optional composite ambiguity index combining normalized:
        1. Spectral spread score (normalized to typical prior spread scale 0.20)
        2. Mean wavelength span (normalized to typical gamut span scale 0.30)
        3. Determinant volume scale (normalized to typical ellipsoid scale 0.10)
        
        Formula:
            I_composite = 1/3 * (spread / 0.20 + mean_span / 0.30 + vol_scale / 0.10)
        
        Clearly labeled as a constructed index with explicit normalization.
        """
        norm_spread = spread_score / 0.20 if spread_score is not None else 0.0
        norm_span = mean_span / 0.30 if mean_span is not None else 0.0
        norm_vol = volume_scale / 0.10 if volume_scale is not None else 0.0

        composite_val = float((norm_spread + norm_span + norm_vol) / 3.0)

        return {
            "value": composite_val,
            "unit": "constructed index (dimensionless)",
            "kind": "constructed_index",
            "label": "Composite Ambiguity Index",
            "formula": "1/3 * (spread/0.20 + mean_span/0.30 + vol_scale/0.10)",
            "components": {
                "normalized_spread": float(norm_spread),
                "normalized_mean_span": float(norm_span),
                "normalized_volume_scale": float(norm_vol)
            },
            "weights": [1/3, 1/3, 1/3],
            "description": (
                "Constructed unitless composite index benchmarking overall spectral freedom "
                "relative to nominal prior dispersion scales (1.0 = nominal prior ambiguity)."
            )
        }

    @staticmethod
    def compute_sample_statistics(alpha_samples, z_samples, spectra_samples, wavelengths=None):
        """
        Computes comprehensive sample-based statistics, covariance decompositions,
        pairwise distances, principal ambiguity directions, and uncertainty metrics.
        """
        N = len(alpha_samples)
        if N < 2:
            raise ValueError(f"Need at least 2 samples to compute covariance, got {N}")

        d_f = alpha_samples.shape[1]
        K = z_samples.shape[1]
        
        if wavelengths is None:
            wavelengths = np.linspace(400, 700, spectra_samples.shape[1])
        wavelengths = np.asarray(wavelengths, dtype=float)

        # ---------------------------------------------------------------------
        # 1. Alpha-Space Statistics (Reduced Coordinates in R^{d_f})
        # ---------------------------------------------------------------------
        alpha_centroid = np.mean(alpha_samples, axis=0)
        cov_alpha = np.cov(alpha_samples, rowvar=False)
        if cov_alpha.ndim == 0:
            cov_alpha = np.array([[float(cov_alpha)]])

        eigvals_alpha, eigvecs_alpha = np.linalg.eigh(cov_alpha)
        idx_a = np.argsort(eigvals_alpha)[::-1]
        eigvals_alpha = eigvals_alpha[idx_a]
        eigvecs_alpha = eigvecs_alpha[:, idx_a]

        var_tot_alpha = float(np.sum(eigvals_alpha))
        exp_var_ratio_alpha = (eigvals_alpha / max(var_tot_alpha, 1e-12)).tolist()

        # Coordinate-wise spread score: sigma_spread = sqrt(Tr(Sigma_alpha) / d_f)
        spread_alpha = float(np.sqrt(max(var_tot_alpha / d_f, 0.0)))

        # Determinant / generalized variance volume scale: A_det = det(Sigma_alpha)^(1 / (2 * d_f))
        pos_eigvals = eigvals_alpha[eigvals_alpha > 1e-12]
        if len(pos_eigvals) == d_f:
            det_alpha = float(np.prod(pos_eigvals))
            volume_scale_alpha = float(det_alpha ** (1.0 / (2.0 * d_f)))
        else:
            det_alpha = 0.0
            volume_scale_alpha = 0.0

        # Alpha bounding box extents
        alpha_min = np.min(alpha_samples, axis=0)
        alpha_max = np.max(alpha_samples, axis=0)
        alpha_widths = alpha_max - alpha_min
        alpha_box_vol = float(np.prod(alpha_widths))

        # ---------------------------------------------------------------------
        # 2. Spectral Space Statistics (31 Wavelengths in R^{31})
        # ---------------------------------------------------------------------
        spec_mean = np.mean(spectra_samples, axis=0)
        spec_std = np.std(spectra_samples, axis=0, ddof=1)
        spec_min = np.min(spectra_samples, axis=0)
        spec_max = np.max(spectra_samples, axis=0)
        spec_span = spec_max - spec_min

        # Spectral covariance matrix Sigma_R in R^{31 x 31}
        cov_spectra = np.cov(spectra_samples, rowvar=False)
        eigvals_spec, eigvecs_spec = np.linalg.eigh(cov_spectra)
        idx_s = np.argsort(eigvals_spec)[::-1]
        eigvals_spec = eigvals_spec[idx_s]
        eigvecs_spec = eigvecs_spec[:, idx_s]

        # Leading d_f eigenvalues/eigenvectors represent exact non-zero modes
        var_tot_spec = float(np.sum(eigvals_spec))
        exp_var_spec = (eigvals_spec[:d_f] / max(var_tot_spec, 1e-12)).tolist()
        cum_var_spec = np.cumsum(eigvals_spec[:d_f] / max(var_tot_spec, 1e-12)).tolist()

        # Spectral RMS spread around sample mean:
        # S_spread = sqrt(1/N * sum ||R_i - R_bar||_2^2) = sqrt((N-1)/N * Tr(Sigma_R))
        diffs_mean = spectra_samples - spec_mean
        rms_spread_spec = float(np.sqrt(np.mean(np.sum(diffs_mean ** 2, axis=1))))

        # Integrated uncertainty across 400-700 nm
        trap_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
        integrated_std = float(trap_fn(spec_std, wavelengths))
        integrated_sample_span = float(trap_fn(spec_span, wavelengths))
        mean_spectral_span = float(np.mean(spec_span))

        # Sample projections onto leading 3 principal spectral directions
        leading_modes_31 = eigvecs_spec[:, :min(d_f, 5)] # (31, 5)
        projections_leading = (diffs_mean @ leading_modes_31[:, :3]).tolist()

        # ---------------------------------------------------------------------
        # 3. Pairwise Spectral Distances
        # ---------------------------------------------------------------------
        pairwise_stats = AmbiguityMetrics.compute_pairwise_distances(spectra_samples)

        # ---------------------------------------------------------------------
        # 4. MCMC Chain Diagnostics
        # ---------------------------------------------------------------------
        mcmc_diag = AmbiguityMetrics.compute_mcmc_diagnostics(alpha_samples)

        # ---------------------------------------------------------------------
        # 5. Composite Ambiguity Index
        # ---------------------------------------------------------------------
        composite_idx = AmbiguityMetrics.compute_composite_ambiguity_index(
            spread_score=rms_spread_spec,
            mean_span=mean_spectral_span,
            volume_scale=volume_scale_alpha
        )

        # ---------------------------------------------------------------------
        # 6. PCA Space Statistics (Latent Z in R^K)
        # ---------------------------------------------------------------------
        z_centroid = np.mean(z_samples, axis=0)
        cov_z = np.cov(z_samples, rowvar=False)
        eigvals_z, eigvecs_z = np.linalg.eigh(cov_z)
        idx_z = np.argsort(eigvals_z)[::-1]
        eigvals_z = eigvals_z[idx_z]
        eigvecs_z = eigvecs_z[:, idx_z]
        var_tot_z = float(np.sum(eigvals_z))
        exp_var_ratio_z = (eigvals_z / max(var_tot_z, 1e-12)).tolist()

        # ---------------------------------------------------------------------
        # 7. Selected Metamer Spectra (5 representative samples)
        # ---------------------------------------------------------------------
        # Sample: first point, min energy, max energy, furthest point from mean
        energies = np.mean(spectra_samples, axis=1)
        dists_from_mean = np.linalg.norm(diffs_mean, axis=1)

        sel_indices = [
            0,
            int(np.argmin(energies)),
            int(np.argmax(energies)),
            int(np.argmax(dists_from_mean)),
            N // 2
        ]
        # Deduplicate while preserving order
        unique_sel = []
        for idx in sel_indices:
            if idx not in unique_sel:
                unique_sel.append(idx)
        selected_spectra = spectra_samples[unique_sel].tolist()

        return {
            "sample_count": N,
            "fiber_dimension": d_f,
            "ambient_dimension": 31,
            "spread_score": {
                "spectral_rms": {
                    "value": rms_spread_spec,
                    "unit": "reflectance (dimensionless)",
                    "kind": "sample_estimate",
                    "formula": "S_spread = sqrt(1/N * sum ||R_i - R_bar||_2^2)",
                    "label": "Spectral RMS Spread Score"
                },
                "alpha_trace": {
                    "value": spread_alpha,
                    "unit": "reflectance (dimensionless)",
                    "kind": "sample_estimate",
                    "formula": "sigma_spread = sqrt(Tr(Sigma_alpha) / d_f)",
                    "label": "Coordinate Spread Score (alpha-space)"
                }
            },
            "pairwise_distances": pairwise_stats,
            "volume_scores": {
                "ellipsoid_scale_a_det": {
                    "value": volume_scale_alpha,
                    "unit": "reflectance (dimensionless)",
                    "kind": "sample_estimate",
                    "formula": "A_det = det(Sigma_alpha)^(1 / (2 * d_f))",
                    "label": "Ellipsoidal Volume Scale (A_det)"
                },
                "generalized_variance_det": {
                    "value": det_alpha,
                    "unit": f"reflectance^{2 * d_f}",
                    "kind": "sample_estimate",
                    "formula": "det(Sigma_alpha)",
                    "label": "Generalized Variance (Determinant)"
                },
                "bounding_box_volume": {
                    "value": alpha_box_vol,
                    "unit": f"reflectance^{d_f}",
                    "kind": "sample_estimate",
                    "formula": "prod(max(alpha) - min(alpha))",
                    "label": "Sample Bounding Box Volume"
                }
            },
            "composite_index": composite_idx,
            "spectral_covariance": {
                "leading_eigenvalues": [float(v) for v in eigvals_spec[:d_f]],
                "explained_variance_ratio": exp_var_spec,
                "cumulative_explained_variance": cum_var_spec,
                "leading_modes_31": leading_modes_31.tolist(),
                "projections_3d": projections_leading
            },
            "alpha_space": {
                "centroid": alpha_centroid.tolist(),
                "covariance_matrix": cov_alpha.tolist(),
                "eigenvalues": [float(v) for v in eigvals_alpha],
                "explained_variance_ratio": exp_var_ratio_alpha,
                "widths": [float(w) for w in alpha_widths],
                "box_volume": alpha_box_vol
            },
            "pca_space": {
                "coordinate_system": "PCA coordinates z in R^K",
                "centroid": z_centroid.tolist(),
                "eigenvalues": [float(v) for v in eigvals_z],
                "explained_variance_ratio": exp_var_ratio_z,
                "leading_principal_directions": eigvecs_z[:, :3].tolist()
            },
            "wavelength_space": {
                "wavelengths": wavelengths.tolist(),
                "mean": spec_mean.tolist(),
                "std": spec_std.tolist(),
                "min": spec_min.tolist(),
                "max": spec_max.tolist(),
                "span": spec_span.tolist(),
                "mean_span": mean_spectral_span,
                "integrated_std_uncertainty": integrated_std,
                "integrated_sample_span": integrated_sample_span,
                "selected_spectra": selected_spectra
            },
            "mcmc_diagnostics": mcmc_diag
        }
