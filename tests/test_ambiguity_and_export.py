import os
import unittest
import numpy as np
from src.fiber_dimension import FiberDimensionAnalyzer
from src.ambiguity_metrics import AmbiguityMetrics
from src.export_utils import export_as_json, export_as_csv, export_as_markdown
from src.spectral_bounds import SpectralBoundsCalculator
from src.metamer_solver import MetamerSolver


class TestAmbiguityAndExport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifacts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts"))
        cls.solver = MetamerSolver(cls.artifacts_dir)

    def test_fiber_dimension_analysis(self):
        """Verifies FiberDimensionAnalyzer exact mathematical outputs and units."""
        A = self.solver.A_camera
        res = FiberDimensionAnalyzer.analyze_dimensions(A, K=8, inradius=0.0635)
        
        self.assertEqual(res["pca_component_count"]["value"], 8)
        self.assertEqual(res["pca_component_count"]["unit"], "components")
        self.assertEqual(res["constraint_rank"]["value"], 3)
        self.assertEqual(res["constraint_rank"]["unit"], "rank")
        self.assertEqual(res["affine_dimension"]["value"], 5)
        self.assertEqual(res["affine_dimension"]["unit"], "dimensions")
        self.assertEqual(res["feasible_dimension"]["value"], 5)
        self.assertEqual(res["feasible_dimension"]["kind"], "exact")

    def test_pairwise_distances(self):
        """Verifies pairwise spectral distance calculation and units."""
        rng = np.random.RandomState(42)
        # 100 spectra with 31 bands
        spectra = rng.uniform(0.1, 0.9, size=(100, 31))
        pw = AmbiguityMetrics.compute_pairwise_distances(spectra, max_pairs=5000)
        
        self.assertIn("mean", pw)
        self.assertGreater(pw["mean"]["value"], 0.0)
        self.assertEqual(pw["mean"]["unit"], "reflectance-vector distance (dimensionless)")
        self.assertGreater(pw["max_observed"]["value"], pw["mean"]["value"])
        self.assertGreaterEqual(pw["p95"]["value"], pw["median"]["value"])
        self.assertEqual(pw["pairs_evaluated"], 100 * 99 // 2)

    def test_composite_ambiguity_index(self):
        """Verifies composite ambiguity index calculation and documented weights."""
        comp = AmbiguityMetrics.compute_composite_ambiguity_index(
            spread_score=0.20,
            mean_span=0.30,
            volume_scale=0.10
        )
        # Each normalized component is 1.0, so composite value should be exactly 1.0
        self.assertAlmostEqual(comp["value"], 1.0, places=5)
        self.assertEqual(comp["unit"], "constructed index (dimensionless)")
        self.assertEqual(len(comp["weights"]), 3)

    def test_mcmc_diagnostics(self):
        """Verifies MCMC ESS and lag-1 autocorrelation diagnostics."""
        rng = np.random.RandomState(42)
        samples = rng.randn(200, 5)
        diag = AmbiguityMetrics.compute_mcmc_diagnostics(samples)
        
        self.assertEqual(len(diag["ess_per_coord"]), 5)
        self.assertGreater(diag["min_ess"], 10.0)
        self.assertLessEqual(diag["mean_ess"], 200.0)

    def test_export_utilities(self):
        """Verifies JSON, CSV, and Markdown export formats."""
        sample_data = {
            "feasible": True,
            "rgb": [0.5, 0.5, 0.5],
            "n_samples": 500,
            "burn_in": 50,
            "thinning": 2,
            "dimension_analysis": FiberDimensionAnalyzer.analyze_dimensions(self.solver.A_camera, K=8),
            "ambiguity_metrics": {
                "spread_score": {
                    "spectral_rms": {"value": 0.045, "unit": "reflectance", "label": "RMS Spread", "formula": "..."}
                },
                "pairwise_distances": {
                    "mean": {"value": 0.062, "unit": "reflectance-vector distance"},
                    "pairs_evaluated": 1000
                },
                "composite_index": {"value": 0.85, "unit": "constructed index", "label": "Composite Index"}
            },
            "validation": {
                "mean_error": 1.2e-6,
                "max_error": 2.5e-6,
                "rms_error": 1.5e-6,
                "constraint_residual": 3.0e-7,
                "max_violation": -1.2e-4,
                "percent_valid": 100.0,
                "valid_points": 500,
                "total_points": 500,
                "overall_status": "PASS"
            },
            "spectral_envelope": [
                {
                    "wavelength_nm": 550.0,
                    "sample_min": 0.026,
                    "sample_max": 0.066,
                    "exact_min": 0.025,
                    "exact_max": 0.067,
                    "sample_span": 0.040
                }
            ]
        }

        # JSON Export
        json_out = export_as_json(sample_data)
        self.assertIn('"feasible": true', json_out)
        self.assertIn('550.0', json_out)

        # CSV Export
        csv_out = export_as_csv(sample_data)
        self.assertIn("Category,Metric,Value,Unit", csv_out)
        self.assertIn("Dimensions", csv_out)
        self.assertIn("Spectral Bounds", csv_out)

        # Markdown Export
        md_out = export_as_markdown(sample_data)
        self.assertIn("# Spectral Metameric Fiber Analysis Report", md_out)
        self.assertIn("| Metric | Value | Unit |", md_out)
        self.assertIn("550.0 nm", md_out)

    def test_complete_envelope_calculation(self):
        """Verifies SpectralBoundsCalculator.compute_complete_envelope across 31 bands."""
        rng = np.random.RandomState(42)
        fake_spectra = rng.uniform(0.1, 0.8, size=(50, 31))
        calc = self.solver.bounds_calc
        env = calc.compute_complete_envelope(fake_spectra)
        
        self.assertEqual(len(env), 31)
        self.assertEqual(env[0]["wavelength_nm"], 400.0)
        self.assertEqual(env[-1]["wavelength_nm"], 700.0)
        for item in env:
            self.assertGreaterEqual(item["sample_max"], item["sample_min"])
            self.assertEqual(item["unit"], "reflectance (0-1)")


if __name__ == '__main__':
    unittest.main()
