import os
import numpy as np
from scipy.linalg import null_space
from src.gamut_checker import GamutChecker
from src.hit_and_run import HitAndRunSampler
from src.spectral_bounds import SpectralBoundsCalculator
from src.ambiguity_metrics import AmbiguityMetrics
from src.spectral_reconstruction import SpectralReconstructor
from src.camera_model import CameraModel
from src.fiber_dimension import FiberDimensionAnalyzer


class MetamerSolver:
    """
    Core engine for the Reduced-Dimensional Spectral Metameric Fiber Analyzer.
    
    Mathematical Formulation:
    - Spectral representation: R = mu + B @ z (31 bands, K=8 latent PCA)
    - Measurement equation: c = M @ R = M @ mu + (M @ B) @ z
    - Camera equality system: A @ z = b, where A = M @ B, b = c - M @ mu
    - Spectral prior polytope: C @ z <= d (C = A_red, d = b_red)
    - Fiber dimension: d_f = K - rank(A) = 8 - 3 = 5
    - Null-space basis N: A @ N = 0
    - Exact reduced fiber: Phi(c) = { mu + B @ (z0 + N @ alpha) : G @ alpha <= h }
      where G = C @ N, h = d - C @ z0
    """
    def __init__(self, artifacts_dir, camera_model=None):
        self.artifacts_dir = artifacts_dir
        self.A_red = np.load(os.path.join(artifacts_dir, "A_red.npy")).astype(np.float64)
        self.b_red = np.load(os.path.join(artifacts_dir, "b_red.npy")).astype(np.float64)
        self.pca_mean = np.load(os.path.join(artifacts_dir, "pca_mean.npy")).astype(np.float64)
        self.pca_basis = np.load(os.path.join(artifacts_dir, "pca_basis.npy")).astype(np.float64)
        
        # Load or use camera model
        if camera_model is None:
            self.camera_model = CameraModel.from_artifacts(artifacts_dir)
            self.A_camera = np.load(os.path.join(artifacts_dir, "A_camera.npy")).astype(np.float64)
            self.camera_offset = np.load(os.path.join(artifacts_dir, "camera_offset.npy")).astype(np.float64)
        else:
            self.camera_model = camera_model
            self.A_camera, self.camera_offset = self.camera_model.project_pca(self.pca_basis, self.pca_mean)

        self.K = self.A_camera.shape[1]
        
        # Dynamic rank and null space computation
        self.constraint_rank = int(np.linalg.matrix_rank(self.A_camera))
        self.fiber_dimension = int(self.K - self.constraint_rank)
        self.N = null_space(self.A_camera) # shape (K, d_f)
        
        # Precompute and cache reduced constraint matrix G = C @ N
        # This is invariant to the target RGB observation
        self.G = self.A_red @ self.N # shape (714340, d_f)

        self.gamut_checker = GamutChecker(
            self.A_red, self.b_red, self.A_camera, self.camera_offset
        )
        self.reconstructor = SpectralReconstructor(
            pca_basis=self.pca_basis, pca_mean=self.pca_mean
        )
        self.bounds_calc = SpectralBoundsCalculator(
            self.pca_basis, self.pca_mean, self.N, self.G
        )

    def construct_fiber(self, rgb):
        """
        Phase 1: Exact Mathematical Fiber Construction.
        Verifies gamut feasibility and returns exact mathematical representation:
        anchor z0, null-space N, reduced matrix G, RHS h.
        """
        feasible, z0 = self.gamut_checker.check_feasibility(rgb)
        if not feasible or z0 is None:
            return {"feasible": False}

        # Exact reduced inequality RHS: h = d - C @ z0
        h = self.b_red - self.A_red @ z0

        return {
            "feasible": True,
            "z0": z0,
            "N": self.N,
            "G": self.G,
            "h": h,
            "constraint_rank": self.constraint_rank,
            "fiber_dimension": self.fiber_dimension,
            "K": self.K
        }

    def solve(
        self,
        rgb,
        num_points=1000,
        n_samples=None,
        burn_in=50,
        thinning=2,
        seed=42,
        compute_bounds=False,
        bound_wavelengths=None,
        progress_callback=None
    ):
        """
        Complete pipeline separating exact fiber construction, user-configured Hit-and-Run sampling,
        statistical characterization, and spectral bounds.
        """
        if n_samples is not None:
            num_points = n_samples

        num_points = int(num_points)
        burn_in = int(burn_in)
        thinning = int(thinning)

        # 1. Exact fiber construction (independent of N)
        fiber = self.construct_fiber(rgb)
        if not fiber["feasible"]:
            return {"feasible": False}

        z0 = fiber["z0"]
        h = fiber["h"]
        G = fiber["G"]
        N = fiber["N"]

        # 2. Sample the feasible region using user-configured Hit-and-Run
        sampler = HitAndRunSampler(G, h)
        alpha_points, diag = sampler.sample_chain(
            num_samples=num_points,
            alpha_init=np.zeros(self.fiber_dimension),
            burn_in=burn_in,
            thinning=thinning,
            seed=seed,
            progress_callback=progress_callback
        )

        # 3. Map reduced alpha samples back to PCA coordinates: z = z0 + alpha @ N.T
        if len(alpha_points) > 0:
            z_points = z0 + alpha_points @ N.T
        else:
            z_points = np.array([z0])

        # 4. Reconstruct spectra
        spectra = self.reconstructor.reconstruct(z_points)

        # 5. Calculate statistics and quantitative ambiguity metrics (estimated from N samples)
        stats = AmbiguityMetrics.compute_sample_statistics(
            alpha_samples=alpha_points if len(alpha_points) > 0 else np.zeros((1, self.fiber_dimension)),
            z_samples=z_points,
            spectra_samples=spectra
        )

        z_centroid = np.array(stats["pca_space"]["centroid"])
        principal_directions = np.array(stats["pca_space"]["leading_principal_directions"])

        # 6. Dimensional and rank analysis
        dimension_analysis = FiberDimensionAnalyzer.analyze_dimensions(
            self.A_camera, K=self.K
        )

        # 7. Exact spectral bounds via linear programming and complete 31-band envelope
        exact_bounds = None
        bound_comparisons = None
        if compute_bounds:
            if bound_wavelengths is None:
                # Key wavelength sample indices: [0, 5, 10, 15, 20, 25, 30] -> 400 to 700 nm in 50nm steps
                bound_wavelengths = [0, 5, 10, 15, 20, 25, 30]
            exact_bounds = self.bounds_calc.compute_all_bounds(z0, h, wave_indices=bound_wavelengths)
            bound_comparisons = self.bounds_calc.compare_with_samples(exact_bounds, spectra)

        # Complete 31-band envelope (sample distributions + exact bounds where computed)
        spectral_envelope = self.bounds_calc.compute_complete_envelope(
            spectra, exact_bounds_list=exact_bounds
        )

        return {
            "feasible": True,
            "z0": z0,
            "N": N,
            "G": G,
            "h": h,
            "C_alpha": G,
            "d_alpha": h,
            "alpha_points": alpha_points,
            "z_points": z_points,
            "z_centroid": z_centroid,
            "principal_directions": principal_directions,
            "fiber_dimension": self.fiber_dimension,
            "constraint_rank": self.constraint_rank,
            "dimension_analysis": dimension_analysis,
            "ambiguity_metrics": stats,
            "sampling_diagnostics": diag,
            "exact_bounds": exact_bounds,
            "bound_comparisons": bound_comparisons,
            "spectral_envelope": spectral_envelope,
            "n_samples": num_points,
            "burn_in": burn_in,
            "thinning": thinning
        }
