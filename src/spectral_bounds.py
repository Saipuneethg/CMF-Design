from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.optimize import linprog


class SpectralBoundsCalculator:
    """
    Computes exact wavelength-wise reflectance bounds for the feasible metameric
    fiber using linear optimization (LP) over the reduced-space polytope:
    
        min / max  R_j(alpha) = mu_j + B_j @ z0 + (B_j @ N) @ alpha
        subject to G @ alpha <= h
        
    Provides ground-truth geometric bounds on spectral ambiguity for comparison
    with empirical Hit-and-Run sample distributions and percentile ribbons.
    All outputs strictly carry units and explicit solver status.
    """
    def __init__(self, pca_basis, pca_mean, N, G, wavelengths=None):
        self.B = np.asarray(pca_basis, dtype=np.float64)
        self.mu = np.asarray(pca_mean, dtype=np.float64)
        self.N = np.asarray(N, dtype=np.float64)
        self.G = np.asarray(G, dtype=np.float64)
        self.num_bands = self.B.shape[0]
        self.dim = self.N.shape[1]
        
        if wavelengths is None:
            self.wavelengths = np.linspace(400, 700, self.num_bands)
        else:
            self.wavelengths = np.asarray(wavelengths, dtype=float)

    def compute_wavelength_bound(self, wave_idx, z0, h, method='highs'):
        """
        Solves LP for minimum and maximum reflectance at a specific wavelength index.
        
        Returns:
        - dict with exact min, max, solver status, feasibility, and units.
        """
        if wave_idx < 0 or wave_idx >= self.num_bands:
            raise IndexError(f"Wavelength index {wave_idx} out of range [0, {self.num_bands-1}]")

        # Objective coefficients in alpha-space:
        # R_j(alpha) = mu_j + B_j @ z0 + (B_j @ N) @ alpha
        c_obj = self.B[wave_idx] @ self.N
        base_val = float(self.mu[wave_idx] + self.B[wave_idx] @ z0)

        bounds = [(None, None) for _ in range(self.dim)]

        # 1. Minimize: c_obj @ alpha
        res_min = linprog(c_obj, A_ub=self.G, b_ub=h, bounds=bounds, method=method)
        # 2. Maximize: -c_obj @ alpha
        res_max = linprog(-c_obj, A_ub=self.G, b_ub=h, bounds=bounds, method=method)

        success = bool(res_min.success and res_max.success)
        min_val = float(base_val + res_min.fun) if res_min.success else None
        max_val = float(base_val - res_max.fun) if res_max.success else None

        return {
            "wavelength_idx": int(wave_idx),
            "wavelength_nm": float(self.wavelengths[wave_idx]),
            "exact_min": min_val,
            "exact_max": max_val,
            "exact_range": float(max_val - min_val) if success else None,
            "min_solver_status": res_min.message,
            "max_solver_status": res_max.message,
            "success": success,
            "unit": "reflectance (0-1)",
            "wavelength_unit": "nm",
            "method": "exact optimization (HiGHS LP)"
        }

    def compute_all_bounds(self, z0, h, wave_indices=None, method='highs', max_workers=1):
        """
        Computes exact bounds across a list of wavelength indices (or all 31 bands).
        Uses sequential LP solving by default to ensure thread safety with the underlying C++ HiGHS solver.
        """
        if wave_indices is None:
            wave_indices = list(range(self.num_bands))

        if len(wave_indices) == 1 or max_workers <= 1:
            results = [self.compute_wavelength_bound(w_idx, z0, h, method=method) for w_idx in wave_indices]
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(self.compute_wavelength_bound, w_idx, z0, h, method) for w_idx in wave_indices]
                results = [f.result() for f in futures]
                
        # Sort by wavelength index
        results.sort(key=lambda x: x["wavelength_idx"])
        return results

    @staticmethod
    def compare_with_samples(exact_bounds_list, reconstructed_spectra, percentiles=(1, 5, 25, 50, 75, 95, 99)):
        """
        Compares exact LP bounds against empirical sample distributions.
        
        Parameters:
        - exact_bounds_list: List of dicts returned by compute_all_bounds.
        - reconstructed_spectra: Array of shape (N_samples, 31).
        - percentiles: Tuple of percentiles to compute.
        
        Returns:
        - comparisons: List of dicts comparing exact vs sampled quantities per wavelength.
        """
        comparisons = []
        spectra = np.asarray(reconstructed_spectra, dtype=np.float64)

        for b in exact_bounds_list:
            w_idx = b["wavelength_idx"]
            band_samples = spectra[:, w_idx]
            
            sample_min = float(np.min(band_samples))
            sample_max = float(np.max(band_samples))
            sample_mean = float(np.mean(band_samples))
            sample_std = float(np.std(band_samples))
            pct_vals = {f"p{p}": float(np.percentile(band_samples, p)) for p in percentiles}

            comp = {
                "wavelength_idx": w_idx,
                "wavelength_nm": b["wavelength_nm"],
                "exact_min": b["exact_min"],
                "exact_max": b["exact_max"],
                "exact_range": b.get("exact_range"),
                "sample_min": sample_min,
                "sample_max": sample_max,
                "sample_range": float(sample_max - sample_min),
                "sample_mean": sample_mean,
                "sample_std": sample_std,
                "min_undercoverage": float(sample_min - b["exact_min"]) if b["exact_min"] is not None else None,
                "max_undercoverage": float(b["exact_max"] - sample_max) if b["exact_max"] is not None else None,
                "percentiles": pct_vals,
                "method_exact": b.get("method", "exact optimization (HiGHS LP)"),
                "method_sampled": "empirical sample extrema",
                "unit": "reflectance (0-1)",
                "wavelength_unit": "nm"
            }
            comparisons.append(comp)

        return comparisons

    @staticmethod
    def compute_complete_envelope(reconstructed_spectra, wavelengths=None, exact_bounds_list=None):
        """
        Computes the complete 31-band envelope including sample statistics for all bands
        and matching exact LP bounds where computed.
        """
        spectra = np.asarray(reconstructed_spectra, dtype=np.float64)
        N, num_bands = spectra.shape
        if wavelengths is None:
            wavelengths = np.linspace(400, 700, num_bands)
        wavelengths = np.asarray(wavelengths, dtype=float)

        exact_map = {}
        if exact_bounds_list:
            for b in exact_bounds_list:
                exact_map[b["wavelength_idx"]] = b

        envelope = []
        for j in range(num_bands):
            band_samples = spectra[:, j]
            s_min = float(np.min(band_samples))
            s_max = float(np.max(band_samples))
            s_mean = float(np.mean(band_samples))
            s_std = float(np.std(band_samples))
            p5 = float(np.percentile(band_samples, 5))
            p25 = float(np.percentile(band_samples, 25))
            p50 = float(np.percentile(band_samples, 50))
            p75 = float(np.percentile(band_samples, 75))
            p95 = float(np.percentile(band_samples, 95))

            ex = exact_map.get(j)
            item = {
                "wavelength_idx": j,
                "wavelength_nm": float(wavelengths[j]),
                "sample_min": s_min,
                "sample_max": s_max,
                "sample_span": float(s_max - s_min),
                "sample_mean": s_mean,
                "sample_std": s_std,
                "p5": p5,
                "p25": p25,
                "p50": p50,
                "p75": p75,
                "p95": p95,
                "exact_min": ex["exact_min"] if ex else None,
                "exact_max": ex["exact_max"] if ex else None,
                "exact_span": ex["exact_range"] if ex else None,
                "exact_status": ex["min_solver_status"] if ex else "Not computed for this band",
                "unit": "reflectance (0-1)",
                "wavelength_unit": "nm"
            }
            envelope.append(item)

        return envelope
