import unittest
import numpy as np
import os
from src.metamer_solver import MetamerSolver
from src.spectral_bounds import SpectralBoundsCalculator


class TestSpectralBounds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifacts_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts")
        cls.solver = MetamerSolver(cls.artifacts_dir)
        cls.rgb = [0.5, 0.5, 0.5]
        cls.fiber = cls.solver.construct_fiber(cls.rgb)

    def test_lp_bounds_consistency(self):
        """Verifies that R_min <= R_max at tested wavelengths."""
        calc = self.solver.bounds_calc
        z0 = self.fiber["z0"]
        h = self.fiber["h"]
        
        # Test 3 representative wavelengths: 400 nm (idx 0), 550 nm (idx 15), 700 nm (idx 30)
        test_indices = [0, 15, 30]
        results = calc.compute_all_bounds(z0, h, wave_indices=test_indices)

        for res in results:
            self.assertTrue(res["success"], f"LP solver failed at wavelength {res['wavelength_nm']}")
            self.assertIsNotNone(res["exact_min"])
            self.assertIsNotNone(res["exact_max"])
            self.assertLessEqual(
                res["exact_min"],
                res["exact_max"] + 1e-7,
                f"Min bound exceeds max bound at {res['wavelength_nm']} nm"
            )

    def test_samples_enclosed_in_exact_bounds(self):
        """Verifies that empirical sample extrema do not violate exact LP bounds."""
        res_solve = self.solver.solve(self.rgb, num_points=100, seed=42)
        spectra = self.solver.reconstructor.reconstruct(res_solve["z_points"])
        
        test_indices = [0, 15, 30]
        bounds = self.solver.bounds_calc.compute_all_bounds(
            self.fiber["z0"], self.fiber["h"], wave_indices=test_indices
        )
        comparisons = SpectralBoundsCalculator.compare_with_samples(bounds, spectra)

        for comp in comparisons:
            exact_min = comp["exact_min"]
            exact_max = comp["exact_max"]
            sample_min = comp["sample_min"]
            sample_max = comp["sample_max"]

            # Sample min must be >= exact min (within small solver tolerance 1e-5)
            self.assertGreaterEqual(
                sample_min,
                exact_min - 1e-5,
                f"Sample min {sample_min} violates exact min {exact_min} at {comp['wavelength_nm']} nm"
            )
            # Sample max must be <= exact max (within small solver tolerance 1e-5)
            self.assertLessEqual(
                sample_max,
                exact_max + 1e-5,
                f"Sample max {sample_max} violates exact max {exact_max} at {comp['wavelength_nm']} nm"
            )


if __name__ == '__main__':
    unittest.main()
