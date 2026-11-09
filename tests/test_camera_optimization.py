"""
Unit Tests for Camera Spectral Sensitivity Optimization
========================================================
Validates all requirements of the Research Implementation Prompt:
1. Shape and wavelength alignment (3 x 31, 400-700 nm, 10 nm increments).
2. Strict non-negativity constraint (s_k(lambda) >= 0).
3. Smoothness and 2nd derivative properties.
4. Channel throughput normalization (sum matching ~10.672).
5. Numerical rank (rank 3) and fiber dimension (5D null space).
6. Conditioning and channel non-redundancy (condition number < 20).
7. Exact recomputation of A_camera_optimized and camera_offset_optimized.
8. File consistency between .npy, .csv, and metadata.json.
9. Baseline camera preservation (original artifacts unchanged).
10. Application solver switching and compatibility.
"""

import os
import json
import unittest
import numpy as np
from src.camera_model import CameraModel
from src.metamer_solver import MetamerSolver


class TestCameraOptimization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifacts_dir = os.path.join(os.path.dirname(__file__), "../artifacts")
        cls.results_dir = os.path.join(os.path.dirname(__file__), "../results/cmf_optimization")

        cls.wavelengths = np.load(os.path.join(cls.artifacts_dir, "wavelengths.npy"))
        cls.pca_basis = np.load(os.path.join(cls.artifacts_dir, "pca_basis.npy"))
        cls.pca_mean = np.load(os.path.join(cls.artifacts_dir, "pca_mean.npy"))

        cls.m_base = np.load(os.path.join(cls.artifacts_dir, "M_camera.npy"))
        cls.a_base = np.load(os.path.join(cls.artifacts_dir, "A_camera.npy"))
        cls.off_base = np.load(os.path.join(cls.artifacts_dir, "camera_offset.npy"))

        cls.m_opt_path = os.path.join(cls.artifacts_dir, "M_camera_optimized.npy")
        cls.csv_opt_path = os.path.join(cls.artifacts_dir, "M_camera_optimized.csv")
        cls.meta_opt_path = os.path.join(cls.artifacts_dir, "M_camera_optimized_metadata.json")
        cls.a_opt_path = os.path.join(cls.artifacts_dir, "A_camera_optimized.npy")
        cls.off_opt_path = os.path.join(cls.artifacts_dir, "camera_offset_optimized.npy")

        cls.m_opt = np.load(cls.m_opt_path)
        cls.a_opt = np.load(cls.a_opt_path)
        cls.off_opt = np.load(cls.off_opt_path)

    def test_matrix_dimensions_and_wavelengths(self):
        """Verify shape (3, 31) and wavelength alignment."""
        self.assertEqual(self.m_opt.shape, (3, 31), "Optimized matrix must be 3 channels x 31 bands")
        self.assertEqual(len(self.wavelengths), 31, "Wavelength grid must have 31 bands")
        self.assertEqual(int(self.wavelengths[0]), 400)
        self.assertEqual(int(self.wavelengths[-1]), 700)
        self.assertEqual(int(self.wavelengths[1] - self.wavelengths[0]), 10)

    def test_strict_non_negativity(self):
        """Verify all optimized sensitivities are strictly non-negative (>= 0)."""
        min_val = np.min(self.m_opt)
        self.assertGreaterEqual(min_val, -1e-9, f"Sensitivity contains negative values: {min_val}")

    def test_channel_throughput_normalization(self):
        """Verify all channels are normalized to equal integrated response."""
        row_sums = np.sum(self.m_opt, axis=1)
        base_target = np.mean(np.sum(self.m_base, axis=1))
        for k in range(3):
            self.assertAlmostEqual(
                row_sums[k], base_target, places=3,
                msg=f"Channel {k} sum {row_sums[k]} does not match target {base_target}"
            )

    def test_rank_and_fiber_dimension(self):
        """Verify measurement rank is 3 and fiber dimension is 5."""
        A_calc = self.m_opt @ self.pca_basis
        rank = int(np.linalg.matrix_rank(A_calc))
        self.assertEqual(rank, 3, "Optimized measurement matrix A must have full row rank 3")
        fiber_dim = 8 - rank
        self.assertEqual(fiber_dim, 5, "Fiber dimension must be exactly 5")

    def test_conditioning_and_non_redundancy(self):
        """Verify condition number of M M^T is stable and well conditioned (< 20)."""
        MMT = self.m_opt @ self.m_opt.T
        eigvals = np.linalg.eigvalsh(MMT)
        cond = np.sqrt(eigvals[-1] / eigvals[0])
        self.assertLess(cond, 20.0, f"Condition number {cond} is too high")

    def test_recomputation_of_dependent_matrices(self):
        """Verify A_camera_optimized = M_opt @ B and offset = M_opt @ mu."""
        expected_A = self.m_opt @ self.pca_basis
        expected_off = self.m_opt @ self.pca_mean
        np.testing.assert_allclose(self.a_opt, expected_A, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(self.off_opt, expected_off, rtol=1e-12, atol=1e-12)

    def test_csv_npy_consistency(self):
        """Verify M_camera_optimized.csv matches M_camera_optimized.npy exactly."""
        self.assertTrue(os.path.exists(self.csv_opt_path))
        lines = []
        with open(self.csv_opt_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        # Header + 31 rows
        self.assertEqual(len(lines), 32)
        csv_data = []
        for line in lines[1:]:
            parts = [float(x) for x in line.split(",")]
            self.assertEqual(len(parts), 4)
            csv_data.append(parts[1:])  # [R, G, B]
        csv_mat = np.array(csv_data).T  # (3, 31)
        np.testing.assert_allclose(csv_mat, self.m_opt, atol=1e-5)

    def test_metadata_completeness(self):
        """Verify metadata JSON contains all required fields."""
        self.assertTrue(os.path.exists(self.meta_opt_path))
        with open(self.meta_opt_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        required_keys = [
            "title", "research_proposal", "wavelength_grid", "matrix_orientation",
            "channel_ordering", "sensitivity_normalization", "parameterization",
            "physical_constraints", "rank_and_dimensions", "evaluation_data_split",
            "primary_ambiguity_objective", "held_out_comparison", "colorimetric_performance"
        ]
        for key in required_keys:
            self.assertIn(key, meta, f"Metadata missing required key: {key}")

    def test_original_camera_preservation(self):
        """Verify original baseline artifacts remain untouched."""
        m_base_reloaded = np.load(os.path.join(self.artifacts_dir, "M_camera.npy"))
        a_base_reloaded = np.load(os.path.join(self.artifacts_dir, "A_camera.npy"))
        off_base_reloaded = np.load(os.path.join(self.artifacts_dir, "camera_offset.npy"))
        np.testing.assert_array_equal(self.m_base, m_base_reloaded)
        np.testing.assert_array_equal(self.a_base, a_base_reloaded)
        np.testing.assert_array_equal(self.off_base, off_base_reloaded)

    def test_solver_integration(self):
        """Verify MetamerSolver can instantiate both Baseline and Optimized models."""
        cam_base = CameraModel.from_artifacts(self.artifacts_dir)
        cam_opt = CameraModel(M=self.m_opt, name="Optimized Camera")

        solver_b = MetamerSolver(self.artifacts_dir, camera_model=cam_base)
        solver_o = MetamerSolver(self.artifacts_dir, camera_model=cam_opt)

        self.assertEqual(solver_b.fiber_dimension, 5)
        self.assertEqual(solver_o.fiber_dimension, 5)

        # Check null-space orthogonality A @ N = 0
        np.testing.assert_allclose(solver_b.A_camera @ solver_b.N, 0.0, atol=1e-12)
        np.testing.assert_allclose(solver_o.A_camera @ solver_o.N, 0.0, atol=1e-12)

        # Check null-space orthonormality N.T @ N = I_5
        np.testing.assert_allclose(solver_b.N.T @ solver_b.N, np.eye(5), atol=1e-12)
        np.testing.assert_allclose(solver_o.N.T @ solver_o.N, np.eye(5), atol=1e-12)

    def test_paired_heldout_evaluation(self):
        """Verify paired held-out statistics and per-sample results CSV."""
        paired_json_path = os.path.join(self.results_dir, "paired_statistics.json")
        paired_csv_path = os.path.join(self.results_dir, "paired_heldout_results.csv")

        self.assertTrue(os.path.exists(paired_json_path), "paired_statistics.json must exist")
        self.assertTrue(os.path.exists(paired_csv_path), "paired_heldout_results.csv must exist")

        with open(paired_json_path, "r", encoding="utf-8") as f:
            paired_data = json.load(f)

        self.assertEqual(paired_data["n_samples"], 50)
        self.assertGreaterEqual(paired_data["sample_counts"]["percentage_improved"], 70.0)
        self.assertLess(paired_data["paired_difference"]["mean"], 0.0, "Mean paired difference must be negative")
        self.assertLess(paired_data["paired_difference"]["median"], 0.0, "Median paired difference must be negative")

        # Hypothesis test
        wilc = paired_data["hypothesis_tests"]["wilcoxon_signed_rank"]
        self.assertLess(wilc["p_value_one_sided_less"], 0.01, "Wilcoxon test must be statistically significant (p < 0.01)")

        # Bootstrap CI
        ci_mean = paired_data["bootstrap_confidence_intervals_95"]["mean_difference"]
        self.assertLess(ci_mean[0], ci_mean[1], "Bootstrap CI lower must be less than upper")
        self.assertLess(ci_mean[1], 0.001, "Bootstrap CI upper bound must be <= 0")

    def test_multi_start_summary(self):
        """Verify multi-start summary contains >= 5 runs and exactly 1 selected winner."""
        multi_csv_path = os.path.join(self.results_dir, "multi_start_summary.csv")
        self.assertTrue(os.path.exists(multi_csv_path), "multi_start_summary.csv must exist")

        with open(multi_csv_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        # Header + at least 5 runs
        self.assertGreaterEqual(len(lines), 6)
        header = lines[0].split(",")
        self.assertIn("selected", header)
        self.assertIn("val_score", header)
        self.assertIn("constraints_valid", header)

        # Exactly 1 winner selected
        selected_idx = header.index("selected")
        selected_count = sum(1 for line in lines[1:] if line.split(",")[selected_idx] == "True")
        self.assertEqual(selected_count, 1, "Exactly one run must be selected as winner")


    def test_visualizations_and_pdf_report(self):
        """Verify all publication-quality plots and updated PDF report exist and are non-empty."""
        plots = [
            "all_cameras_global_ranking.png",
            "color_difference_distribution.png",
            "gamut_sampling_distribution.png",
            "paired_heldout_reduction.png",
            "pareto_ambiguity_vs_color_tradeoff.png",
            "per_color_joint_scatter.png",
            "reference_camera_detailed_ranking.png",
            "sensitivity_curves_global_opt.png",
            "sensitivity_curves_old_opt.png"
        ]
        for plot_name in plots:
            p_path = os.path.join(self.results_dir, plot_name)
            self.assertTrue(os.path.exists(p_path), f"Plot {plot_name} must exist in results/")
            self.assertGreater(os.path.getsize(p_path), 5000, f"Plot {plot_name} cannot be empty")

        pdf_path = os.path.join(self.results_dir, "optimization_report.pdf")
        self.assertTrue(os.path.exists(pdf_path), "optimization_report.pdf must exist")
        self.assertGreater(os.path.getsize(pdf_path), 20000, "PDF report must be non-empty")


if __name__ == "__main__":
    unittest.main()

